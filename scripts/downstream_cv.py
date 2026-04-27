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
from omegaconf import OmegaConf
from rdkit.Chem.rdchem import Mol
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean, scatter_softmax, scatter_sum
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

def batch_iter(H, offsets, targets, indices, batch_size, device, shuffle):
    order = np.array(indices)
    if shuffle:
        np.random.shuffle(order)
    for s in range(0, len(order), batch_size):
        mids = order[s:s + batch_size]
        Hs, bs_idx = [], []
        for bi, mi in enumerate(mids):
            a, b = int(offsets[mi]), int(offsets[mi + 1])
            Hs.append(H[a:b])
            bs_idx.append(torch.full((b - a,), bi, dtype=torch.long))
        yield (
            torch.cat(Hs).to(device=device, dtype=torch.float32),
            torch.cat(bs_idx).to(device),
            targets[mids].to(device),
        )


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

    for ep in range(args.epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        for H_b, b_b, t_b in batch_iter(H, offsets, tgt_norm,
                                          train_idx.tolist(),
                                          args.batch_size, device, shuffle=True):
            pred = model(H_b, b_b)             # [B] scalar
            valid = ~torch.isnan(t_b)
            if not valid.any():
                continue
            loss = ((pred[valid] - t_b[valid]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            epoch_loss_sum += float(loss.detach().item()) * int(valid.sum().item())
            epoch_loss_count += int(valid.sum().item())

        # Per-epoch logging — drives wandb training curves. Cheap because
        # head is small (~M params); val pass on cached H is sub-second.
        if wandb_run is not None:
            train_loss_zspace = (epoch_loss_sum / max(epoch_loss_count, 1))
            current_lr = float(opt.param_groups[0]["lr"])
            log_dict = {
                f"fold_{fold_i}/train_loss":  train_loss_zspace,   # z-score MSE
                f"fold_{fold_i}/lr":          current_lr,
                f"fold_{fold_i}/epoch":       ep,
            }
            # Quick val MAE in physical units (every epoch).
            model.eval()
            with torch.no_grad():
                vp = []
                for H_b, b_b, _ in batch_iter(H, offsets, tgt_norm,
                                                val_idx_list,
                                                args.batch_size, device,
                                                shuffle=False):
                    vp.append(model(H_b, b_b).cpu().numpy())
            vp_phys = np.concatenate(vp) * std + mean
            if val_groups_local is None:
                err = vp_phys[val_mask_local] - y_true_local[val_mask_local]
                log_dict[f"fold_{fold_i}/val_mae"] = float(np.mean(np.abs(err)))
            else:
                # Aggregate by group_id before MAE — same as final eval.
                vp_m = vp_phys[val_mask_local]
                yt_m = y_true_local[val_mask_local]
                grps = val_groups_local[val_mask_local]
                _, inv = np.unique(grps, return_inverse=True)
                cnt = np.bincount(inv).clip(min=1)
                pm = np.bincount(inv, weights=vp_m) / cnt
                ym = np.bincount(inv, weights=yt_m) / cnt
                log_dict[f"fold_{fold_i}/val_mae"] = float(np.mean(np.abs(pm - ym)))
            wandb_run.log(log_dict)

    # Evaluate (preds in val_idx ORDER, then de-normalize to physical units).
    model.eval()
    preds = []
    with torch.no_grad():
        for H_b, b_b, _ in batch_iter(H, offsets, tgt_norm,
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

    return {
        "n_train":       n_train_total,
        "n_val":         int(mask.sum()),
        "n_val_groups":  int(len(unique_groups)),
        "ensemble_K":    int(np.median(counts)),
        "target_mean":   mean, "target_std": std,
        "mae":  float(mean_absolute_error(y_per_group, pred_per_group)),
        "rmse": float(np.sqrt(mean_squared_error(y_per_group, pred_per_group))),
        "r2":   float(r2_score(y_per_group, pred_per_group)),
        # Conformer-spread diagnostics (per-input-id pred-stddev, in target units):
        "ensemble_pred_std_mean":   float(np.mean(pred_std_per_group)),
        "ensemble_pred_std_median": float(np.median(pred_std_per_group)),
        "ensemble_pred_std_p95":    float(np.percentile(pred_std_per_group, 95)),
        # Express conformer noise as fraction of target spread for portability:
        "ensemble_pred_std_over_target_std": float(np.mean(pred_std_per_group) / max(std, 1e-12)),
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
        print(f"  MAE={rep['mae']:.4f}  RMSE={rep['rmse']:.4f}  "
              f"R2={rep['r2']:.3f}{ens_str}")
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
