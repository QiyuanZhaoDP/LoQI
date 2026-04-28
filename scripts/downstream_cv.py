"""5-fold cross-validation for a downstream property-prediction dataset on
top of a frozen LoQI backbone.

Pipeline (per dataset):
    1. Load prepared .pt (from scripts/prepare_downstream_dataset.py).
    2. Extract H once for every molecule (frozen backbone, cached to disk).
    3. Split molecule indices into K folds (deterministic by --seed).
    4. For each fold:
         - Train a small head (MLP) on the other K-1 folds.
         - Evaluate on the held-out fold (MAE, RMSE, R^2).
    5. Write per-fold metrics + aggregated mean ± std to JSON.

Only the heads are trained; the LoQI backbone stays frozen across folds.
Caching means the expensive H-extraction happens once even if you sweep
head hyperparameters.

Usage:
    python scripts/downstream_cv.py \\
        --ckpt data/loqi.ckpt \\
        --config scripts/conf/loqi/loqi.yaml \\
        --dataset-pt data/downstream/delaney.pt \\
        --out-dir /tmp/downstream/delaney \\
        --n-folds 5 \\
        --epochs 50 --batch-size 128 --lr 3e-4 \\
        --head-hidden 256 --n-mp-layers 2 \\
        --device cuda
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from rdkit.Chem.rdchem import Mol
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_max, scatter_mean, scatter_softmax, scatter_sum
from tqdm import tqdm

from megalodon.data.batch_preprocessor import BatchPreProcessor
from megalodon.models.module import Graph3DInterpolantModel
from megalodon.models.thermo_heads import AtomMolMP


# ---------------------------------------------------------------------------
# Single-target head — attention-pooled message passing only.
# Mirrors megalodon.models.thermo_heads.AtomMolMP but with n_targets=1.
# ---------------------------------------------------------------------------

class SingleTargetHead(nn.Module):
    """Wraps AtomMolMP (the same architecture as the thermo head) with
    n_targets=1, so we can warm-start from the ckpt's trained
    `dynamics.thermo_heads.mp.*` weights when --init-head-from-thermo is set.

    Module-name parity with `megalodon.models.thermo_heads.AtomMolMP` is
    important: `SingleTargetHead.mp.<...>` ↔ `dynamics.thermo_heads.mp.<...>`
    in the ckpt state_dict. Only `mp.final[3]` (the last Linear) has a
    different output dim (1 vs 5) and is always randomly initialized.
    """

    def __init__(self, dim=256, hidden=128, n_mp_layers=2, n_heads=4):
        super().__init__()
        self.mp = AtomMolMP(
            dim=dim, n_layers=n_mp_layers, n_heads=n_heads,
            hidden=hidden, n_targets=1,
        )

    def forward(self, H, batch_idx):
        return self.mp(H, batch_idx).squeeze(-1)


def load_thermo_head_into(head: "SingleTargetHead", ckpt_path: str) -> int:
    """Load the trained thermo head's weights into a SingleTargetHead's
    inner AtomMolMP. Returns the number of tensors copied.

    Tries multiple candidate prefixes because Lightning's checkpoint path
    depends on whether the model uses an EMA wrapper:
        dynamics.ema_model.thermo_heads.mp.<...>      (EMA-wrapped)
        dynamics.online_model.thermo_heads.mp.<...>   (EMA-wrapped, raw)
        dynamics.thermo_heads.mp.<...>                (no EMA)
    Prefers EMA when present (better val numbers), falls back to online,
    then plain.

    Skips final.3 (last Linear) because output dim differs (5 → 1).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    candidates = [
        "dynamics.ema_model.thermo_heads.mp.",
        "dynamics.online_model.thermo_heads.mp.",
        "dynamics.thermo_heads.mp.",
    ]
    chosen_prefix = None
    for p in candidates:
        if any(k.startswith(p) for k in sd.keys()):
            chosen_prefix = p
            break
    if chosen_prefix is None:
        # Last-resort scan: any key containing the substring
        for k in sd.keys():
            if "thermo_heads.mp." in k:
                chosen_prefix = k.split("thermo_heads.mp.", 1)[0] + "thermo_heads.mp."
                break
    if chosen_prefix is None:
        print(f"  [warm-init] no thermo_heads.mp.* keys in {ckpt_path} — "
              f"falling back to random init.")
        return 0

    src = {}
    for k, v in sd.items():
        if not k.startswith(chosen_prefix):
            continue
        local = k[len(chosen_prefix):]
        # Drop the final-layer Linear that maps to 5 thermo targets — shape
        # mismatch with our 1-target head; leave it random-init.
        if local in ("final.3.weight", "final.3.bias"):
            continue
        src[local] = v
    print(f"  [warm-init] using prefix {chosen_prefix!r}; matched {len(src)} tensors")
    missing, unexpected = head.mp.load_state_dict(src, strict=False)
    n_copied = len(src)
    if unexpected:
        print(f"  [warm-init] unexpected keys (skipped): {unexpected[:5]}"
              + (f" + {len(unexpected)-5} more" if len(unexpected) > 5 else ""))
    if missing:
        # `final.3.*` will always be missing (we dropped them on purpose).
        residual_missing = [k for k in missing if not k.startswith("final.3")]
        if residual_missing:
            print(f"  [warm-init] still-missing keys (will random-init): "
                  f"{residual_missing[:5]}"
                  + (f" + {len(residual_missing)-5} more" if len(residual_missing) > 5 else ""))
    return n_copied


# ---------------------------------------------------------------------------
# Dataset helpers / H extraction
# ---------------------------------------------------------------------------

class _TempDataset(InMemoryDataset):
    def __init__(self, data, slices):
        super().__init__(".")
        self.data, self.slices = data, slices
        self._indices = None


def load_prepared_pt(pt_path):
    with torch.serialization.safe_globals(
        [DataEdgeAttr, DataTensorAttr, GlobalStorage, Mol]
    ):
        data, slices = torch.load(pt_path)
    return _TempDataset(data, slices)


def load_backbone(ckpt, cfg_path, device):
    cfg = OmegaConf.load(cfg_path)
    pre = BatchPreProcessor(cfg.data.aug_rotations, cfg.data.scale_coords)
    # loss_fn=None + strict=False: avoid loading any saved aux loss into the
    # frozen backbone here (we don't need it for downstream feature
    # extraction). Also sidesteps the "old non-nn.Module loss_fn pickle"
    # backward-compat pitfall when reading pre-torchmetrics-rewrite ckpts.
    model = Graph3DInterpolantModel.load_from_checkpoint(
        ckpt,
        loss_fn=None,
        loss_params=cfg.loss,
        interpolant_params=cfg.interpolant,
        sampling_params=cfg.sample,
        batch_preprocessor=pre,
        map_location=device,
        strict=False,
    )
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model, cfg


@torch.no_grad()
def extract_H(model, cfg, ds, indices, batch_size, device, cache_path):
    """Run the frozen backbone; cache H + per-mol offsets + targets to disk."""
    if cache_path.exists():
        print(f"  H cache already exists: {cache_path}")
        d = torch.load(cache_path)
        return d["H"], d["offsets"], d["targets"], d["has_target"]

    t_type = str(cfg.interpolant.time_type)
    t_max = cfg.interpolant.timesteps - 1 if t_type == "discrete" else 1.0
    subset = [ds[i] for i in indices]
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    H_chunks = []
    offsets = [0]
    targets = []
    has_target = []
    for batch in tqdm(loader, desc="extract-H"):
        batch = batch.to(device)
        bs = int(batch.batch.max().item()) + 1
        tgt = batch.target.view(-1).float().cpu()
        has = batch.has_target.view(-1).bool().cpu()

        batch = model.batch_preprocessor(batch)
        if t_type == "discrete":
            t = torch.full((bs,), t_max, dtype=torch.long, device=device)
        else:
            t = torch.full((bs,), t_max, dtype=torch.float32, device=device)
        out, _, _ = model(batch, t)
        H = out["H"].to(torch.bfloat16).cpu()
        counts = torch.bincount(batch.batch.cpu(), minlength=bs).tolist()
        for c in counts:
            offsets.append(offsets[-1] + c)
        H_chunks.append(H)
        targets.append(tgt)
        has_target.append(has)

    H = torch.cat(H_chunks, dim=0)
    offsets = torch.tensor(offsets, dtype=torch.long)
    targets = torch.cat(targets, dim=0)
    has_target = torch.cat(has_target, dim=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"H": H, "offsets": offsets,
                "targets": targets, "has_target": has_target}, cache_path)
    print(f"  saved H cache -> {cache_path}  H.shape={tuple(H.shape)}")
    return H, offsets, targets, has_target


# ---------------------------------------------------------------------------
# Fold training
# ---------------------------------------------------------------------------

def batch_iter(H, offsets, targets, indices, batch_size, device, shuffle,
               group_ids=None, group_batched=False):
    """Yields (H, batch_idx, targets, group_ids_or_None).

    When `group_batched=True`, sorts indices by group_id then shuffles group
    order, so each batch is dominated by full K-conformer groups (gives
    invariance loss strong within-batch signal). Without it, random shuffle
    of Data indices means few groups have ≥2 samples per batch and the
    invariance signal is too noisy.
    """
    indices = np.asarray(indices)
    if group_batched:
        if group_ids is None:
            raise ValueError("group_batched requires group_ids")
        gids_local = group_ids[indices]
        # Stable sort: members of the same group end up contiguous.
        sort_order = np.argsort(gids_local, kind="stable")
        indices = indices[sort_order]
        gids_local = gids_local[sort_order]
        # Group boundaries.
        unique_g, first_idx, counts = np.unique(
            gids_local, return_index=True, return_counts=True,
        )
        if shuffle:
            perm = np.random.permutation(len(unique_g))
            indices = np.concatenate([
                indices[first_idx[p]:first_idx[p] + counts[p]] for p in perm
            ])
    else:
        if shuffle:
            indices = indices.copy()
            np.random.shuffle(indices)

    for s in range(0, len(indices), batch_size):
        mids = indices[s:s + batch_size]
        Hs, bs_idx = [], []
        for bi, mi in enumerate(mids):
            a, b = int(offsets[mi]), int(offsets[mi + 1])
            Hs.append(H[a:b])
            bs_idx.append(torch.full((b - a,), bi, dtype=torch.long))
        gids_b = (torch.tensor(group_ids[mids], dtype=torch.long, device=device)
                  if group_ids is not None else None)
        yield (
            torch.cat(Hs).to(device=device, dtype=torch.float32),
            torch.cat(bs_idx).to(device),
            targets[mids].to(device),
            gids_b,
        )


def _within_group_var_loss(pred, group_ids):
    """Within-group variance of predictions, averaged over groups with ≥2
    samples. Drives the head toward conformer-invariant predictions when
    added to the loss with a positive weight λ.
    """
    _, inv = torch.unique(group_ids, return_inverse=True)
    n_groups = int(inv.max().item()) + 1 if inv.numel() > 0 else 0
    if n_groups == 0:
        return pred.new_zeros(())
    mean = scatter_mean(pred, inv, dim=0, dim_size=n_groups)
    sq_dev = (pred - mean[inv]) ** 2
    var = scatter_mean(sq_dev, inv, dim=0, dim_size=n_groups)
    counts = torch.bincount(inv, minlength=n_groups)
    valid = counts >= 2
    if not bool(valid.any()):
        return pred.new_zeros(())
    return var[valid].mean()


# ---------------------------------------------------------------------------
# LoRA: low-rank adapters on backbone Linear layers. When --lora-r > 0, the
# downstream FT path keeps the backbone in memory and lets gradients flow
# through it (only into LoRA A/B params + head; the base Linear weights stay
# frozen). LoRA params are reset between folds so each fold is independent.
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """y = base(x) + (alpha / r) * (B @ A @ x), with base frozen.

    LoRA paper init: A ~ kaiming_uniform, B = 0  →  initial output equals
    base(x), so injecting LoRA at any point preserves the model's behavior
    until the adapters get gradient.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float | None = None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = int(r)
        self.scaling = (float(alpha) if alpha is not None else float(r)) / float(r)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x):
        # base(x) + scaling * (x @ A.T) @ B.T
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling


def inject_lora(module: nn.Module, target_names: set, r: int, alpha=None) -> int:
    """Recursively wrap target Linears with LoRALinear. Returns count wrapped.

    A child whose attribute name is in `target_names` is treated as a target.
    Two cases:
      * direct Linear  → wrap it
      * Sequential / ModuleList → wrap every direct Linear child of the
        container (catches the two Linears inside swiglu_ffn, since they're
        anonymous indices 0 and 2 of an `ffn` Sequential and otherwise
        un-targetable by attribute name).

    Anything else that doesn't match: recurse into it."""
    n = 0
    for name, child in list(module.named_children()):
        if name in target_names:
            if isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
                n += 1
            elif isinstance(child, (nn.Sequential, nn.ModuleList)):
                for sub_name, sub_child in list(child.named_children()):
                    if isinstance(sub_child, nn.Linear):
                        setattr(child, sub_name,
                                LoRALinear(sub_child, r=r, alpha=alpha))
                        n += 1
            # don't recurse into matched names — already handled.
        else:
            n += inject_lora(child, target_names, r, alpha)
    return n


def reset_lora_params(module: nn.Module) -> None:
    """Re-init all LoRALinear A/B params to their starting values. Call
    between folds so a shared backbone object can serve fresh fold trainings."""
    for m in module.modules():
        if isinstance(m, LoRALinear):
            nn.init.kaiming_uniform_(m.lora_A, a=5 ** 0.5)
            nn.init.zeros_(m.lora_B)


def lora_parameters(module: nn.Module):
    """Iterator over LoRA-only trainable parameters in a module tree."""
    for n, p in module.named_parameters():
        if n.split(".")[-1] in ("lora_A", "lora_B"):
            yield p


def train_one_fold(H, offsets, targets, has_target, train_idx, val_idx,
                    args, device, ensemble_groups=None,
                    wandb_run=None, fold_i=0):
    """Train head on train_idx Data points, evaluate on val_idx.

    If `ensemble_groups` is provided (numpy array length n, one int per Data
    pointing to a group id), val predictions are aggregated by group (mean)
    before computing metrics. This is what makes K-conformer ensembling
    work: K Data of the same input share a group_id, so we get one
    prediction per input molecule even with K-augmented training data.
    """
    # z-score normalize on train only
    tr_has = has_target[train_idx].bool().numpy()
    tr_targets = targets[train_idx][tr_has].float().numpy()
    mean, std = float(tr_targets.mean()), float(tr_targets.std() or 1.0)
    tgt_norm = ((targets - mean) / std).float()

    model = SingleTargetHead(
        dim=H.shape[-1],
        hidden=args.head_hidden,
        n_mp_layers=args.n_mp_layers,
        n_heads=args.mp_n_heads,
    ).to(device)
    if getattr(args, "init_head_from_thermo", False):
        n_copied = load_thermo_head_into(model, args.ckpt)
        model = model.to(device)
        # Print only on first fold to keep logs clean.
        if not getattr(args, "_thermo_warm_announced", False):
            print(f"  [warm-init] copied {n_copied} tensors from thermo head; "
                  "final Linear (5→1) random-init.")
            args._thermo_warm_announced = True
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    total_steps = max(1, (len(train_idx) // args.batch_size) * args.epochs)
    if args.lr_schedule == "constant":
        # Identity factor — keeps lr fixed; still steps so LightningModule-
        # like logging stays consistent.
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda _: 1.0)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_steps, eta_min=args.lr_min
        )

    # Pre-compute val_idx tensor once for efficient per-epoch eval.
    val_idx_list = val_idx.tolist() if hasattr(val_idx, "tolist") else list(val_idx)
    val_idx_arr_local = np.asarray(val_idx)
    val_has_local = has_target[val_idx_arr_local].bool().numpy()
    y_true_local = targets[val_idx_arr_local].float().numpy()
    val_mask_local = val_has_local & ~np.isnan(y_true_local)
    val_groups_local = (ensemble_groups[val_idx_arr_local]
                         if ensemble_groups is not None else None)

    inv_lambda = float(getattr(args, "invariance_lambda", 0.0) or 0.0)
    use_inv = inv_lambda > 0.0 and ensemble_groups is not None
    if use_inv and not getattr(args, "_inv_announced", False):
        print(f"  [inv-loss] λ={inv_lambda} on within-group prediction "
              f"variance (group-batched train shuffle).")
        args._inv_announced = True

    # Best-val tracking + optional early stopping. We always track val MAE
    # per epoch (cheap on cached H) and at the end restore the best model
    # for the comprehensive eval, so cv_report.json reflects val_min not
    # val_last regardless of any late-epoch overfit.
    patience_n = int(getattr(args, "early_stopping_patience", 0) or 0)
    best_val_mae = float("inf")
    best_state = None
    best_epoch = -1
    patience_counter = 0
    n_epochs_run = 0

    for ep in range(args.epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        epoch_abs_err_sum = 0.0   # accumulate Σ|pred - target| in z-space
        epoch_inv_sum = 0.0
        epoch_inv_count = 0
        for H_b, b_b, t_b, g_b in batch_iter(
                H, offsets, tgt_norm, train_idx.tolist(),
                args.batch_size, device, shuffle=True,
                group_ids=ensemble_groups if use_inv else None,
                group_batched=use_inv):
            pred = model(H_b, b_b)             # [B] scalar
            valid = ~torch.isnan(t_b)
            if not valid.any():
                continue
            residual = pred[valid] - t_b[valid]
            loss_mse = (residual ** 2).mean()
            if use_inv and g_b is not None:
                loss_inv = _within_group_var_loss(pred[valid], g_b[valid])
                loss = loss_mse + inv_lambda * loss_inv
                epoch_inv_sum += float(loss_inv.detach().item())
                epoch_inv_count += 1
            else:
                loss = loss_mse
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            n_valid = int(valid.sum().item())
            epoch_loss_sum += float(loss_mse.detach().item()) * n_valid
            epoch_abs_err_sum += float(residual.detach().abs().sum().item())
            epoch_loss_count += n_valid

        # Per-epoch val MAE in physical units. Always computed (cheap on
        # cached H) so we can drive best-val tracking + early stopping
        # regardless of whether wandb is enabled.
        model.eval()
        with torch.no_grad():
            vp = []
            for H_b, b_b, _, _ in batch_iter(H, offsets, tgt_norm,
                                            val_idx_list,
                                            args.batch_size, device,
                                            shuffle=False):
                vp.append(model(H_b, b_b).cpu().numpy())
        vp_phys = np.concatenate(vp) * std + mean
        if val_groups_local is None:
            err = vp_phys[val_mask_local] - y_true_local[val_mask_local]
            val_mae_ep = float(np.mean(np.abs(err)))
        else:
            vp_m = vp_phys[val_mask_local]
            yt_m = y_true_local[val_mask_local]
            grps = val_groups_local[val_mask_local]
            _, inv = np.unique(grps, return_inverse=True)
            cnt = np.bincount(inv).clip(min=1)
            pm = np.bincount(inv, weights=vp_m) / cnt
            ym = np.bincount(inv, weights=yt_m) / cnt
            val_mae_ep = float(np.mean(np.abs(pm - ym)))

        # Best-val tracking. Snapshot model state on improvement so the
        # final eval uses val_min, not val_last.
        n_epochs_run = ep + 1
        if val_mae_ep < best_val_mae:
            best_val_mae = val_mae_ep
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if wandb_run is not None:
            train_loss_zspace = (epoch_loss_sum / max(epoch_loss_count, 1))
            train_mae_phys = (
                epoch_abs_err_sum / max(epoch_loss_count, 1) * std
            )
            train_rmse_phys = float(np.sqrt(train_loss_zspace)) * std
            current_lr = float(opt.param_groups[0]["lr"])
            log_dict = {
                f"fold_{fold_i}/train_loss":      train_loss_zspace,
                f"fold_{fold_i}/train_mae_phys":  train_mae_phys,
                f"fold_{fold_i}/train_rmse_phys": train_rmse_phys,
                f"fold_{fold_i}/lr":              current_lr,
                f"fold_{fold_i}/epoch":           ep,
                f"fold_{fold_i}/val_mae":         val_mae_ep,
                f"fold_{fold_i}/best_val_mae":    best_val_mae,
            }
            if use_inv and epoch_inv_count > 0:
                log_dict[f"fold_{fold_i}/train_inv_var"] = (
                    epoch_inv_sum / epoch_inv_count
                )
            wandb_run.log(log_dict)

        # Early stop after the improvement check / wandb log.
        if patience_n > 0 and patience_counter >= patience_n:
            print(f"  [early-stop] no improvement for {patience_n} epochs "
                  f"(best val_mae={best_val_mae:.4f} @ ep {best_epoch+1}); "
                  f"stopped at ep {ep+1}/{args.epochs}")
            break

    # Restore best-val state for the comprehensive eval below.
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Evaluate (preds in val_idx ORDER, then de-normalize to physical units).
    model.eval()
    preds = []
    with torch.no_grad():
        for H_b, b_b, _, _ in batch_iter(H, offsets, tgt_norm,
                                       val_idx.tolist(),
                                       args.batch_size, device, shuffle=False):
            pred = model(H_b, b_b)
            preds.append(pred.cpu().numpy())
    preds = np.concatenate(preds) * std + mean
    val_idx_arr = np.asarray(val_idx)
    val_has = has_target[val_idx_arr].bool().numpy()
    y_true = targets[val_idx_arr].float().numpy()
    mask = val_has & ~np.isnan(y_true)
    n_train_total = int(tr_has.sum())

    if ensemble_groups is None:
        # Standard path: one prediction per Data.
        return {
            "n_train": n_train_total, "n_val": int(mask.sum()),
            "target_mean": mean, "target_std": std,
            "mae":  float(mean_absolute_error(y_true[mask], preds[mask])),
            "rmse": float(np.sqrt(mean_squared_error(y_true[mask], preds[mask]))),
            "r2":   float(r2_score(y_true[mask], preds[mask])),
            "best_epoch":   best_epoch + 1,
            "n_epochs_run": n_epochs_run,
        }

    # Ensemble path: aggregate K conformer preds per group_id, then metrics.
    val_groups = ensemble_groups[val_idx_arr]
    val_groups_masked = val_groups[mask]
    preds_masked = preds[mask]
    y_true_masked = y_true[mask]

    # group_id -> mean / std of preds, first y_true (all should be equal)
    unique_groups, inv = np.unique(val_groups_masked, return_inverse=True)
    counts = np.bincount(inv, minlength=len(unique_groups)).clip(min=1)
    sums = np.bincount(inv, weights=preds_masked, minlength=len(unique_groups))
    sums_sq = np.bincount(inv, weights=preds_masked ** 2, minlength=len(unique_groups))
    pred_per_group = sums / counts
    # Per-group prediction std across the K conformers — diagnoses how
    # conformer-stable the head is. Low value = robust prediction.
    pred_var_per_group = np.maximum(sums_sq / counts - pred_per_group ** 2, 0.0)
    pred_std_per_group = np.sqrt(pred_var_per_group)
    y_sums = np.bincount(inv, weights=y_true_masked, minlength=len(unique_groups))
    y_per_group = y_sums / counts

    # Per-conformer metrics (UniMol-style: MAE on raw K predictions, no
    # ensemble averaging). Computed exactly here, complementing the
    # ensemble metrics above. y_true is broadcast to per-Data so the
    # comparison sees all K (X_ij, y_i) pairs.
    pc_mae  = float(mean_absolute_error(y_true_masked, preds_masked))
    pc_rmse = float(np.sqrt(mean_squared_error(y_true_masked, preds_masked)))
    pc_r2   = float(r2_score(y_true_masked, preds_masked))

    return {
        "n_train":       n_train_total,
        "n_val":         int(mask.sum()),
        "n_val_groups":  int(len(unique_groups)),
        "ensemble_K":    int(np.median(counts)),
        "target_mean":   mean, "target_std": std,
        "mae":  float(mean_absolute_error(y_per_group, pred_per_group)),
        "rmse": float(np.sqrt(mean_squared_error(y_per_group, pred_per_group))),
        "r2":   float(r2_score(y_per_group, pred_per_group)),
        # Per-conformer (single-conformer) eval — directly comparable to UniMol:
        "mae_per_conformer":  pc_mae,
        "rmse_per_conformer": pc_rmse,
        "r2_per_conformer":   pc_r2,
        # Conformer-spread diagnostics (per-input-id pred-stddev, in target units):
        "ensemble_pred_std_mean":   float(np.mean(pred_std_per_group)),
        "ensemble_pred_std_median": float(np.median(pred_std_per_group)),
        "ensemble_pred_std_p95":    float(np.percentile(pred_std_per_group, 95)),
        # Express conformer noise as fraction of target spread for portability:
        "ensemble_pred_std_over_target_std": float(np.mean(pred_std_per_group) / max(std, 1e-12)),
        "best_epoch":   best_epoch + 1,
        "n_epochs_run": n_epochs_run,
    }


def train_one_fold_lora(ds, model, train_idx, val_idx, args, device,
                        ensemble_groups=None, wandb_run=None, fold_i=0,
                        t_type=None, t_max=None):
    """LoRA-adapted FT path: each batch runs full backbone + head forward,
    gradients flow into LoRA A/B + head only. Mirrors train_one_fold's
    semantics for early stopping / best-val tracking / ensemble metrics
    so cv_report.json shape stays identical."""

    reset_lora_params(model)

    # --- Train-side z-score normalization (cheap iter over Data) --------
    tr_targets = []
    for i in train_idx:
        d = ds[int(i)]
        if bool(d.has_target.item()):
            tr_targets.append(float(d.target.item()))
    mean = float(np.mean(tr_targets)) if tr_targets else 0.0
    std = float(np.std(tr_targets)) if len(tr_targets) > 1 else 1.0
    if std == 0.0:
        std = 1.0

    # --- DataLoaders -----------------------------------------------------
    train_subset = [ds[int(i)] for i in train_idx]
    val_subset = [ds[int(i)] for i in val_idx]
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

    # Discover H_dim from a probe forward (cheap; one batch on val_loader).
    model.eval()
    with torch.no_grad():
        probe = next(iter(val_loader)).to(device)
        bs = int(probe.batch.max().item()) + 1
        probe_pp = model.batch_preprocessor(probe)
        t_p = (torch.full((bs,), int(t_max), dtype=torch.long, device=device)
               if t_type == "discrete"
               else torch.full((bs,), float(t_max), dtype=torch.float32, device=device))
        out, _, _ = model(probe_pp, t_p)
        H_dim = int(out["H"].shape[-1])
    head = SingleTargetHead(
        dim=H_dim, hidden=args.head_hidden,
        n_mp_layers=args.n_mp_layers, n_heads=args.mp_n_heads,
    ).to(device)
    if getattr(args, "init_head_from_thermo", False):
        n_copied = load_thermo_head_into(head, args.ckpt)
        head = head.to(device)
        if not getattr(args, "_thermo_warm_announced", False):
            print(f"  [warm-init] copied {n_copied} tensors from thermo head; "
                  "final Linear (5→1) random-init.")
            args._thermo_warm_announced = True

    # --- Optimizer over LoRA A/B + head params ---------------------------
    lora_p = list(lora_parameters(model))
    head_p = list(head.parameters())
    if not getattr(args, "_lora_announced", False):
        n_lora = sum(p.numel() for p in lora_p)
        n_head = sum(p.numel() for p in head_p)
        n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"  [LoRA] r={args.lora_r}  trainable: lora={n_lora:,}  "
              f"head={n_head:,}  frozen_backbone={n_frozen:,}")
        args._lora_announced = True
    opt = torch.optim.AdamW(lora_p + head_p, lr=args.lr,
                             weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    if args.lr_schedule == "constant":
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda _: 1.0)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_steps, eta_min=args.lr_min,
        )

    # --- Pre-computed val info -------------------------------------------
    val_idx_arr = np.asarray(val_idx)
    val_has = np.array([bool(ds[int(i)].has_target.item()) for i in val_idx_arr],
                       dtype=bool)
    y_true_val = np.array([float(ds[int(i)].target.item()) for i in val_idx_arr],
                          dtype=np.float32)
    val_mask = val_has & ~np.isnan(y_true_val)
    val_groups = (ensemble_groups[val_idx_arr] if ensemble_groups is not None
                  else None)
    n_train_total = int(sum(1 for i in train_idx
                            if bool(ds[int(i)].has_target.item())))

    # --- Best-val tracking + early stopping ------------------------------
    patience_n = int(getattr(args, "early_stopping_patience", 0) or 0)
    best_val_mae = float("inf")
    best_state = None
    best_epoch = -1
    patience_counter = 0
    n_epochs_run = 0

    def _snapshot():
        return {
            "head": {k: v.detach().cpu().clone()
                     for k, v in head.state_dict().items()},
            "lora": {n: p.detach().cpu().clone()
                     for n, p in model.named_parameters()
                     if n.split(".")[-1] in ("lora_A", "lora_B")},
        }

    def _restore(s):
        head.load_state_dict({k: v.to(device) for k, v in s["head"].items()})
        live = dict(model.named_parameters())
        for n, p in s["lora"].items():
            live[n].data.copy_(p.to(device))

    def _val_pass():
        model.eval(); head.eval()
        chunks = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                bs_ = int(batch.batch.max().item()) + 1
                bp = model.batch_preprocessor(batch)
                t_v = (torch.full((bs_,), int(t_max), dtype=torch.long, device=device)
                       if t_type == "discrete"
                       else torch.full((bs_,), float(t_max), dtype=torch.float32, device=device))
                out_v, _, _ = model(bp, t_v)
                chunks.append(head(out_v["H"], bp.batch).cpu().numpy())
        return np.concatenate(chunks) * std + mean

    def _val_mae_aggr(vp_phys):
        if val_groups is None:
            return float(np.mean(np.abs(vp_phys[val_mask] - y_true_val[val_mask])))
        vp_m = vp_phys[val_mask]; yt_m = y_true_val[val_mask]
        grps = val_groups[val_mask]
        _, inv = np.unique(grps, return_inverse=True)
        cnt = np.bincount(inv).clip(min=1)
        pm = np.bincount(inv, weights=vp_m) / cnt
        ym = np.bincount(inv, weights=yt_m) / cnt
        return float(np.mean(np.abs(pm - ym)))

    # --- Train loop ------------------------------------------------------
    for ep in range(args.epochs):
        model.train(); head.train()
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        epoch_abs_err_sum = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            bs_ = int(batch.batch.max().item()) + 1
            tgt = batch.target.view(-1).float()
            has = batch.has_target.view(-1).bool()
            tgt_norm = (tgt - mean) / std
            valid = has & ~torch.isnan(tgt_norm)
            if not valid.any():
                continue
            bp = model.batch_preprocessor(batch)
            t_b = (torch.full((bs_,), int(t_max), dtype=torch.long, device=device)
                   if t_type == "discrete"
                   else torch.full((bs_,), float(t_max), dtype=torch.float32, device=device))
            out, _, _ = model(bp, t_b)
            pred = head(out["H"], bp.batch)
            residual = pred[valid] - tgt_norm[valid]
            loss = (residual ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_p + head_p, 1.0)
            opt.step()
            sched.step()
            n_valid = int(valid.sum().item())
            epoch_loss_sum += float(loss.detach().item()) * n_valid
            epoch_abs_err_sum += float(residual.detach().abs().sum().item())
            epoch_loss_count += n_valid

        # Per-epoch val + best/early-stop
        vp_phys = _val_pass()
        val_mae_ep = _val_mae_aggr(vp_phys)
        n_epochs_run = ep + 1
        if val_mae_ep < best_val_mae:
            best_val_mae = val_mae_ep
            best_epoch = ep
            best_state = _snapshot()
            patience_counter = 0
        else:
            patience_counter += 1

        if wandb_run is not None:
            train_loss_zspace = epoch_loss_sum / max(epoch_loss_count, 1)
            train_mae_phys = epoch_abs_err_sum / max(epoch_loss_count, 1) * std
            train_rmse_phys = float(np.sqrt(train_loss_zspace)) * std
            wandb_run.log({
                f"fold_{fold_i}/train_loss":      train_loss_zspace,
                f"fold_{fold_i}/train_mae_phys":  train_mae_phys,
                f"fold_{fold_i}/train_rmse_phys": train_rmse_phys,
                f"fold_{fold_i}/lr":              float(opt.param_groups[0]["lr"]),
                f"fold_{fold_i}/epoch":           ep,
                f"fold_{fold_i}/val_mae":         val_mae_ep,
                f"fold_{fold_i}/best_val_mae":    best_val_mae,
            })

        if patience_n > 0 and patience_counter >= patience_n:
            print(f"  [early-stop] no improvement for {patience_n} epochs "
                  f"(best val_mae={best_val_mae:.4f} @ ep {best_epoch+1}); "
                  f"stopped at ep {ep+1}/{args.epochs}")
            break

    # Restore best LoRA + head, then comprehensive eval.
    if best_state is not None:
        _restore(best_state)
    preds = _val_pass()

    if ensemble_groups is None:
        return {
            "n_train": n_train_total, "n_val": int(val_mask.sum()),
            "target_mean": mean, "target_std": std,
            "mae":  float(mean_absolute_error(y_true_val[val_mask], preds[val_mask])),
            "rmse": float(np.sqrt(mean_squared_error(y_true_val[val_mask], preds[val_mask]))),
            "r2":   float(r2_score(y_true_val[val_mask], preds[val_mask])),
            "best_epoch":   best_epoch + 1,
            "n_epochs_run": n_epochs_run,
        }

    val_groups_masked = val_groups[val_mask]
    preds_masked = preds[val_mask]
    y_true_masked = y_true_val[val_mask]
    unique_groups, inv = np.unique(val_groups_masked, return_inverse=True)
    counts = np.bincount(inv, minlength=len(unique_groups)).clip(min=1)
    sums = np.bincount(inv, weights=preds_masked, minlength=len(unique_groups))
    sums_sq = np.bincount(inv, weights=preds_masked ** 2, minlength=len(unique_groups))
    pred_per_group = sums / counts
    pred_var_per_group = np.maximum(sums_sq / counts - pred_per_group ** 2, 0.0)
    pred_std_per_group = np.sqrt(pred_var_per_group)
    y_sums = np.bincount(inv, weights=y_true_masked, minlength=len(unique_groups))
    y_per_group = y_sums / counts

    # Per-conformer eval (UniMol-comparable; see train_one_fold).
    pc_mae  = float(mean_absolute_error(y_true_masked, preds_masked))
    pc_rmse = float(np.sqrt(mean_squared_error(y_true_masked, preds_masked)))
    pc_r2   = float(r2_score(y_true_masked, preds_masked))

    return {
        "n_train":       n_train_total,
        "n_val":         int(val_mask.sum()),
        "n_val_groups":  int(len(unique_groups)),
        "ensemble_K":    int(np.median(counts)),
        "target_mean":   mean, "target_std": std,
        "mae":  float(mean_absolute_error(y_per_group, pred_per_group)),
        "rmse": float(np.sqrt(mean_squared_error(y_per_group, pred_per_group))),
        "r2":   float(r2_score(y_per_group, pred_per_group)),
        "mae_per_conformer":  pc_mae,
        "rmse_per_conformer": pc_rmse,
        "r2_per_conformer":   pc_r2,
        "ensemble_pred_std_mean":   float(np.mean(pred_std_per_group)),
        "ensemble_pred_std_median": float(np.median(pred_std_per_group)),
        "ensemble_pred_std_p95":    float(np.percentile(pred_std_per_group, 95)),
        "ensemble_pred_std_over_target_std": float(np.mean(pred_std_per_group) / max(std, 1e-12)),
        "best_epoch":   best_epoch + 1,
        "n_epochs_run": n_epochs_run,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--dataset-pt", required=True,
                   help=".pt file from scripts/prepare_downstream_dataset.py")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--extract-batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-min", type=float, default=0.0)
    p.add_argument("--lr-schedule", choices=["cosine", "constant"], default="cosine",
                   help="LR schedule. 'constant' keeps lr fixed at --lr the "
                        "whole run (use to test whether cosine decay is "
                        "choking late-epoch learning).")
    p.add_argument("--lora-r", type=int, default=0,
                   help="LoRA rank for backbone-adapter FT. 0 = disabled "
                        "(use cached-H head-only path). When > 0 we keep the "
                        "backbone in memory, wrap target Linears with rank-r "
                        "adapters, and let gradients flow into LoRA A/B + "
                        "head. Base backbone weights stay frozen.")
    p.add_argument("--lora-alpha", type=float, default=None,
                   help="LoRA scaling: forward = base + (alpha/r) * delta. "
                        "Default = r (so alpha/r = 1).")
    p.add_argument("--lora-target", type=str,
                   default="qkv_proj,out_projection",
                   help="Comma-separated attribute names of nn.Linear modules "
                        "to wrap. Defaults to attention QKV + out projection. "
                        "Add ffn_norm/etc to expand coverage.")
    p.add_argument("--early-stopping-patience", type=int, default=0,
                   help="Stop training a fold if val MAE has not improved "
                        "for this many epochs. 0 = disabled. The reported "
                        "metrics always come from the val_min epoch "
                        "regardless of this flag (best model is restored "
                        "before final eval).")
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--n-mp-layers", type=int, default=2)
    p.add_argument("--mp-n-heads", type=int, default=4)
    p.add_argument("--head-hidden", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ensemble-by", default=None,
                   help="Data attribute that groups multiple Data into one "
                        "input molecule (e.g. 'input_id' for K-conformer "
                        "ensemble). When set: 5-fold splits by group, train "
                        "uses all K Data per group as augmentation, val "
                        "averages preds across the K conformers per group "
                        "before computing metrics.")
    p.add_argument("--invariance-lambda", type=float, default=0.0,
                   help="Weight on within-group prediction-variance loss "
                        "(forces the head to predict the same value across "
                        "the K conformers of one input). z-score units; "
                        "useful range ~1-10. Requires --ensemble-by. Enables "
                        "group-batched train shuffling so each batch is "
                        "dominated by full K-conformer groups.")
    p.add_argument("--max-k-per-input", type=int, default=None,
                   help="Cap on conformers per input on the TRAIN side only "
                        "(val keeps all K for apples-to-apples comparison). "
                        "Use this to ablate the K-conformer augmentation: "
                        "--max-k-per-input 1 trains as if you'd prepared a "
                        "K=1 dataset, without re-running prepare_downstream_K_pt. "
                        "Only valid with --ensemble-by. Picks the first K_cap "
                        "Data per group (deterministic).")
    p.add_argument("--init-head-from-thermo", action="store_true",
                   help="Warm-start the downstream head's AtomMolMP weights "
                        "from the ckpt's trained thermo head. Auto-aligns "
                        "n_mp_layers/mp_n_heads/head_hidden to "
                        "cfg.dynamics.model_args.thermo_head_args so the "
                        "state_dict loads cleanly. The final Linear "
                        "(output 5→1) is always random-init.")
    # ---- wandb logging (opt-in) ----
    p.add_argument("--wandb", action="store_true",
                   help="Log per-fold metrics + cross-fold summary to wandb.")
    p.add_argument("--wandb-project", default="downstream_cv")
    p.add_argument("--wandb-group", default=None,
                   help="wandb group (e.g. 'warm' / 'vanilla') for run grouping.")
    p.add_argument("--wandb-name", default=None,
                   help="wandb run name (default: <dataset basename>_<group>).")
    args = p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    if args.max_k_per_input is not None and args.ensemble_by is None:
        raise SystemExit("--max-k-per-input requires --ensemble-by "
                         "(it caps per-group conformer count).")
    if args.invariance_lambda > 0 and args.ensemble_by is None:
        raise SystemExit("--invariance-lambda requires --ensemble-by "
                         "(it needs group_ids to compute within-group var).")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ---- wandb init (opt-in) -----------------------------------------
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            run_name = args.wandb_name or (
                f"{Path(args.dataset_pt).stem}"
                + (f"_{args.wandb_group}" if args.wandb_group else "")
            )
            wandb_run = wandb.init(
                project=args.wandb_project,
                group=args.wandb_group,
                name=run_name,
                config=vars(args),
                reinit=True,
            )
        except Exception as e:
            print(f"WARNING: wandb init failed ({e}); continuing without wandb.")
            wandb_run = None

    print(f"Loading dataset {args.dataset_pt}")
    ds = load_prepared_pt(args.dataset_pt)
    n = len(ds)
    print(f"  {n:,} molecules")

    # Extract H for every molecule (one-pass cache)
    print(f"Loading backbone {args.ckpt}")
    model, cfg = load_backbone(args.ckpt, args.config, device)

    # If warm-starting the head from the ckpt's thermo head, the head dims
    # MUST match the trained thermo_head_args (otherwise state_dict won't
    # load). Override the user's CLI/default head dims accordingly.
    if args.init_head_from_thermo:
        th = OmegaConf.select(cfg, "dynamics.model_args.thermo_head_args",
                              default=None)
        if th is None:
            raise SystemExit(
                "--init-head-from-thermo requires the ckpt's config to define "
                "dynamics.model_args.thermo_head_args (so we know what dims "
                "to instantiate)."
            )
        n_mp_layers_cfg = int(OmegaConf.select(th, "n_mp_layers", default=2))
        mp_n_heads_cfg  = int(OmegaConf.select(th, "mp_n_heads",  default=4))
        hidden_cfg      = int(OmegaConf.select(th, "hidden",      default=128))
        if (args.n_mp_layers, args.mp_n_heads, args.head_hidden) != \
           (n_mp_layers_cfg, mp_n_heads_cfg, hidden_cfg):
            print(f"  [warm-init] aligning head dims to thermo_head_args: "
                  f"n_mp_layers={n_mp_layers_cfg}, mp_n_heads={mp_n_heads_cfg}, "
                  f"head_hidden={hidden_cfg} "
                  f"(was {args.n_mp_layers}/{args.mp_n_heads}/{args.head_hidden})")
            args.n_mp_layers = n_mp_layers_cfg
            args.mp_n_heads  = mp_n_heads_cfg
            args.head_hidden = hidden_cfg

    use_lora = args.lora_r > 0
    if use_lora:
        # LoRA path: keep backbone in memory, inject adapters once. We
        # still need targets/has_target for the fold split — derive them
        # by iterating the dataset (no backbone forward).
        target_names = {s.strip() for s in args.lora_target.split(",")
                         if s.strip()}
        n_wrapped = inject_lora(model, target_names,
                                r=args.lora_r, alpha=args.lora_alpha)
        if n_wrapped == 0:
            raise SystemExit(
                f"[LoRA] no Linears matched names {target_names}. Check "
                f"--lora-target against your backbone module names."
            )
        n_lora_total = sum(p.numel() for p in lora_parameters(model))
        print(f"[LoRA] wrapped {n_wrapped} Linears, "
              f"alpha={args.lora_alpha if args.lora_alpha is not None else args.lora_r}, "
              f"trainable LoRA params: {n_lora_total:,}")
        # Keep base backbone frozen (LoRALinear.__init__ already did this
        # for wrapped Linears; freeze the rest too).
        for n_p, p in model.named_parameters():
            if n_p.split(".")[-1] not in ("lora_A", "lora_B"):
                p.requires_grad = False

        # Targets / has_target without H caching.
        targets = torch.tensor(
            [float(ds[i].target.item()) for i in range(n)], dtype=torch.float)
        has_target = torch.tensor(
            [bool(ds[i].has_target.item()) for i in range(n)], dtype=torch.bool)
        H = offsets = None  # not used in LoRA path
    else:
        cache_path = out_dir / "H_cache.pt"
        H, offsets, targets, has_target = extract_H(
            model, cfg, ds, list(range(n)),
            args.extract_batch_size, device, cache_path,
        )
        del model  # free GPU memory

    # K-fold split on indices where has_target==True.
    # In ensemble mode we split by GROUP (e.g. input_id), so all K conformers
    # of one input molecule stay in the same fold — no leakage.
    all_idx = np.arange(n)
    labeled_mask = has_target.bool().numpy()
    labeled = all_idx[labeled_mask]
    print(f"Labeled Data points: {len(labeled):,} / {n:,}")

    ensemble_groups = None  # numpy[int] of length n; only set in ensemble mode
    if args.ensemble_by is not None:
        # Pull the group key from each Data; tolerate missing keys.
        keys = []
        for i in range(n):
            d = ds[i]
            keys.append(getattr(d, args.ensemble_by, str(i)))
        unique_keys = list(dict.fromkeys(keys))   # preserves first-seen order
        key_to_idx = {k: idx for idx, k in enumerate(unique_keys)}
        ensemble_groups = np.array([key_to_idx[k] for k in keys], dtype=np.int64)
        # Per-group "labeled" = ANY Data in the group has has_target.
        groups_labeled = np.zeros(len(unique_keys), dtype=bool)
        for di, gi in enumerate(ensemble_groups):
            if labeled_mask[di]:
                groups_labeled[gi] = True
        labeled_groups = np.where(groups_labeled)[0]
        print(f"Ensemble mode: grouping by '{args.ensemble_by}' — "
              f"{len(unique_keys):,} unique groups, "
              f"{len(labeled_groups):,} labeled.")
    else:
        labeled_groups = None

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_reports = []
    t0 = time.time()

    if args.ensemble_by is not None:
        split_iter = kf.split(labeled_groups)
    else:
        split_iter = kf.split(labeled)

    for fold_i, (tr, vl) in enumerate(split_iter):
        if args.ensemble_by is not None:
            # Expand group indices → all Data indices in those groups.
            train_groups = labeled_groups[tr]
            val_groups   = labeled_groups[vl]
            train_set = set(train_groups.tolist())
            val_set   = set(val_groups.tolist())
            train_idx = np.array(
                [i for i in labeled if int(ensemble_groups[i]) in train_set],
                dtype=np.int64,
            )
            val_idx = np.array(
                [i for i in labeled if int(ensemble_groups[i]) in val_set],
                dtype=np.int64,
            )

            # K-cap on training side only (val keeps all K conformers).
            # First K_cap Data per group — deterministic; conformer order
            # in the .pt is set by prepare_downstream_K_pt.py.
            if args.max_k_per_input is not None:
                n_before = len(train_idx)
                seen: dict[int, int] = {}
                kept = []
                for i in train_idx:
                    g = int(ensemble_groups[i])
                    if seen.get(g, 0) < args.max_k_per_input:
                        kept.append(i)
                        seen[g] = seen.get(g, 0) + 1
                train_idx = np.array(kept, dtype=np.int64)
                print(f"  [K-cap] train Data: {n_before} -> {len(train_idx)}  "
                      f"(max_k_per_input={args.max_k_per_input})")

            print(f"\n=== Fold {fold_i+1}/{args.n_folds} | "
                  f"train={len(train_idx)} ({len(train_groups)} grp)  "
                  f"val={len(val_idx)} ({len(val_groups)} grp) ===")
        else:
            train_idx = labeled[tr]
            val_idx   = labeled[vl]
            print(f"\n=== Fold {fold_i+1}/{args.n_folds} | "
                  f"train={len(train_idx)}  val={len(val_idx)} ===")
        if use_lora:
            t_type = str(cfg.interpolant.time_type)
            t_max = (cfg.interpolant.timesteps - 1
                      if t_type == "discrete" else 1.0)
            rep = train_one_fold_lora(
                ds, model, train_idx, val_idx, args, device,
                ensemble_groups=ensemble_groups,
                wandb_run=wandb_run, fold_i=fold_i,
                t_type=t_type, t_max=t_max,
            )
        else:
            rep = train_one_fold(H, offsets, targets, has_target,
                                  train_idx, val_idx, args, device,
                                  ensemble_groups=ensemble_groups,
                                  wandb_run=wandb_run, fold_i=fold_i)
        rep["fold"] = fold_i
        fold_reports.append(rep)
        # Compact per-fold summary; include conformer-spread when in ensemble mode.
        ens_str = ""
        if "ensemble_pred_std_mean" in rep:
            ens_str = (f"  pred_σ_mean={rep['ensemble_pred_std_mean']:.4f} "
                       f"(={rep['ensemble_pred_std_over_target_std']*100:.1f}% of target σ)")
        ep_str = (f"  best_ep={rep['best_epoch']}/{rep['n_epochs_run']}"
                  if "best_epoch" in rep else "")
        print(f"  MAE={rep['mae']:.4f}  RMSE={rep['rmse']:.4f}  "
              f"R2={rep['r2']:.3f}{ep_str}{ens_str}")
        if wandb_run is not None:
            wandb_run.log({f"fold/{fold_i}/{k}": v for k, v in rep.items()
                            if isinstance(v, (int, float))})

    # Aggregate
    summary = {
        "mae_mean":  float(np.mean([r["mae"]  for r in fold_reports])),
        "mae_std":   float(np.std ([r["mae"]  for r in fold_reports])),
        "rmse_mean": float(np.mean([r["rmse"] for r in fold_reports])),
        "r2_mean":   float(np.mean([r["r2"]   for r in fold_reports])),
        "folds":     fold_reports,
        "args":      vars(args),
        "n_molecules": n,
        "n_labeled":   int(has_target.sum().item()),
        "wall_seconds": round(time.time()-t0, 1),
    }
    if "ensemble_pred_std_mean" in fold_reports[0]:
        summary["ensemble_pred_std_mean_avg"] = float(np.mean(
            [r["ensemble_pred_std_mean"] for r in fold_reports]))
        summary["ensemble_pred_std_over_target_std_avg"] = float(np.mean(
            [r["ensemble_pred_std_over_target_std"] for r in fold_reports]))
    if "mae_per_conformer" in fold_reports[0]:
        summary["mae_per_conformer_mean"]  = float(np.mean(
            [r["mae_per_conformer"]  for r in fold_reports]))
        summary["mae_per_conformer_std"]   = float(np.std(
            [r["mae_per_conformer"]  for r in fold_reports]))
        summary["rmse_per_conformer_mean"] = float(np.mean(
            [r["rmse_per_conformer"] for r in fold_reports]))
        summary["r2_per_conformer_mean"]   = float(np.mean(
            [r["r2_per_conformer"]   for r in fold_reports]))
    report_path = out_dir / "cv_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print final table
    print("\n" + "=" * 70)
    print(f"{args.n_folds}-fold CV summary  |  {args.dataset_pt}")
    print("-" * 70)
    print(f"  MAE  = {summary['mae_mean']:.4f} ± {summary['mae_std']:.4f}")
    print(f"  RMSE = {summary['rmse_mean']:.4f}")
    print(f"  R²   = {summary['r2_mean']:.3f}")
    if "mae_per_conformer_mean" in summary:
        print(f"  ---- per-conformer (UniMol-style, no ensembling) ----")
        print(f"  MAE  = {summary['mae_per_conformer_mean']:.4f} "
              f"± {summary['mae_per_conformer_std']:.4f}")
        print(f"  RMSE = {summary['rmse_per_conformer_mean']:.4f}")
        print(f"  R²   = {summary['r2_per_conformer_mean']:.3f}")
    if "ensemble_pred_std_mean_avg" in summary:
        print(f"  pred σ across K conformers (avg over folds): "
              f"{summary['ensemble_pred_std_mean_avg']:.4f}  "
              f"({100*summary['ensemble_pred_std_over_target_std_avg']:.1f}% of target σ)")
    print("=" * 70)
    print(f"Report -> {report_path}")

    # ---- wandb final summary ------------------------------------------
    if wandb_run is not None:
        flat_summary = {k: v for k, v in summary.items()
                         if isinstance(v, (int, float))}
        wandb_run.summary.update(flat_summary)
        wandb_run.finish()


if __name__ == "__main__":
    main()
