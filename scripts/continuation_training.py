"""Continuation training: unfreeze the last N layers of the LoQI backbone
and train end-to-end with the thermo heads.

Differences vs scripts/finetune_thermo_head.py:
  * NO cached H — backbone is (partially) trained, so gradients must flow
    through each forward. We iterate the labeled dataset directly with a
    PyG DataLoader, applying the BatchPreProcessor on the fly.
  * Selective unfreeze: only the last N DiTeBlock layers AND the matching
    last N XEGNN layers in the backbone. Everything else frozen.
  * Split learning rates:
        heads              → --lr              (e.g. 3e-4)
        unfrozen backbone  → --backbone-lr     (e.g. 1e-5)
  * wandb logging (optional via --wandb).

The heads are the SAME objects used by finetune_thermo_head.py, imported
from megalodon.models.thermo_heads, so you can start from that script's
state and keep iterating on the same head definitions.

Usage:
  python scripts/continuation_training.py \\
      --ckpt data/loqi.ckpt --config scripts/conf/loqi/loqi.yaml \\
      --train-pt data/chembl3d_stereo/processed/train_h_thermo.pt \\
      --test-pt  data/chembl3d_stereo/processed/test_h_thermo.pt \\
      --head-init /tmp/ft_cache_500k/heads_best.pt  (optional warm start) \\
      --out-dir /tmp/continuation_run \\
      --unfreeze-layers 2 \\
      --max-train 200000 --max-test 20000 \\
      --epochs 10 --batch-size 32 \\
      --lr 3e-4 --backbone-lr 1e-5 \\
      --device cuda --wandb
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from rdkit.Chem.rdchem import Mol
from sklearn.metrics import mean_absolute_error, r2_score
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from megalodon.data.batch_preprocessor import BatchPreProcessor
from megalodon.models.module import Graph3DInterpolantModel
from megalodon.models.thermo_heads import (
    EXTENSIVE_IDX,
    TARGET_FIELDS,
    TARGET_UNITS,
    ThermoHeadModel,
    masked_mse,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class _TempDataset(InMemoryDataset):
    def __init__(self, data, slices):
        super().__init__(".")
        self.data, self.slices = data, slices
        self._indices = None


def load_labeled_subset(pt_path, max_n, seed):
    with torch.serialization.safe_globals(
        [DataEdgeAttr, DataTensorAttr, GlobalStorage, Mol]
    ):
        data, slices = torch.load(pt_path)
    ds = _TempDataset(data, slices)
    flag = ds.data.thermo_has_label.view(-1)
    idx = [i for i in range(len(ds)) if bool(flag[i].item())]
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    if max_n is not None:
        idx = idx[:max_n]
    return [ds[i] for i in idx]


# ---------------------------------------------------------------------------
# Backbone load + selective unfreeze
# ---------------------------------------------------------------------------

def load_backbone(ckpt, cfg_path, device):
    cfg = OmegaConf.load(cfg_path)
    pre = BatchPreProcessor(cfg.data.aug_rotations, cfg.data.scale_coords)
    model = Graph3DInterpolantModel.load_from_checkpoint(
        ckpt,
        loss_params=cfg.loss,
        interpolant_params=cfg.interpolant,
        sampling_params=cfg.sample,
        batch_preprocessor=pre,
        map_location=device,
    )
    model.to(device)
    return model, cfg


def unfreeze_last_n_layers(backbone, n):
    """Freeze the whole backbone, then unfreeze the last n DiTeBlock AND the
    last n XEGNN layers (each round uses both in lockstep).
    Returns the list of trainable backbone parameters (for a separate LR group).
    """
    for p in backbone.parameters():
        p.requires_grad = False

    dit = backbone.dynamics.dit_layers
    egnn = backbone.dynamics.egnn_layers
    assert len(dit) == len(egnn), \
        f"dit/egnn layer count mismatch ({len(dit)} vs {len(egnn)})"
    total = len(dit)
    n = max(0, min(n, total))

    trainable = []
    if n > 0:
        for blk in list(dit[-n:]) + list(egnn[-n:]):
            for p in blk.parameters():
                p.requires_grad = True
                trainable.append(p)
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in backbone.parameters())
    print(f"Unfroze last {n}/{total} DiTeBlock + XEGNN pairs  "
          f"({n_train:,} / {n_total:,} backbone params trainable, "
          f"{100*n_train/max(n_total,1):.1f}%)")
    return trainable


# ---------------------------------------------------------------------------
# Forward through backbone → H → heads
# ---------------------------------------------------------------------------

def forward_heads(backbone, heads, batch, t_max, t_type):
    bs = int(batch.batch.max().item()) + 1
    if backbone.batch_preprocessor is not None:
        batch = backbone.batch_preprocessor(batch)
    if t_type == "discrete":
        time_tensor = torch.full((bs,), t_max, dtype=torch.long, device=batch.batch.device)
    else:
        time_tensor = torch.full((bs,), t_max, dtype=torch.float32, device=batch.batch.device)
    out, batch, _ = backbone(batch, time_tensor)
    return heads(out["H"], batch.batch)


@torch.no_grad()
def eval_loop(backbone, heads, loader, device, t_max, t_type, target_mean, target_std):
    backbone.eval(); heads.eval()
    preds_ext = []
    preds_mp  = []
    tgts_raw  = []
    for batch in loader:
        batch = batch.to(device)
        tgt = torch.stack([batch[f].view(-1).float() for f in TARGET_FIELDS], dim=1).cpu()
        preds = forward_heads(backbone, heads, batch, t_max, t_type)
        preds_ext.append(preds["ext"].cpu())
        preds_mp.append(preds["mp"].cpu())
        tgts_raw.append(tgt)
    preds_ext = torch.cat(preds_ext).numpy() * target_std[EXTENSIVE_IDX] + target_mean[EXTENSIVE_IDX]
    preds_mp  = torch.cat(preds_mp ).numpy() * target_std + target_mean
    tgts_raw  = torch.cat(tgts_raw ).numpy()

    rows = []
    for i, name in enumerate(TARGET_FIELDS):
        mask = ~np.isnan(tgts_raw[:, i])
        if mask.sum() < 20:
            rows.append({"target": name, "note": "too few"})
            continue
        y_true = tgts_raw[mask, i]
        row = {"target": name, "unit": TARGET_UNITS[name], "n_test": int(mask.sum()),
               "mae_mp": float(mean_absolute_error(y_true, preds_mp[mask, i])),
               "r2_mp":  float(r2_score(y_true, preds_mp[mask, i]))}
        if i in EXTENSIVE_IDX:
            j = EXTENSIVE_IDX.index(i)
            row["mae_ext"] = float(mean_absolute_error(y_true, preds_ext[mask, j]))
            row["r2_ext"]  = float(r2_score(y_true, preds_ext[mask, j]))
        rows.append(row)
    return rows


def print_report(rows):
    print("\n" + "=" * 92)
    print(f"{'target':<14s} {'unit':<11s} {'MAE_ext':>10s} {'R2_ext':>8s} "
          f"{'MAE_mp':>10s} {'R2_mp':>8s} {'n_test':>8s}")
    print("-" * 92)
    for r in rows:
        if "note" in r:
            print(f"{r['target']:<14s} {r['note']}")
            continue
        mae_ext = f"{r['mae_ext']:>10.3f}" if "mae_ext" in r else f"{'-':>10s}"
        r2_ext  = f"{r['r2_ext']:>8.3f}"  if "r2_ext"  in r else f"{'-':>8s}"
        print(f"{r['target']:<14s} {r['unit']:<11s} "
              f"{mae_ext} {r2_ext} "
              f"{r['mae_mp']:>10.3f} {r['r2_mp']:>8.3f} "
              f"{r['n_test']:>8d}")
    print("=" * 92)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_target_stats(dataset):
    """Per-target mean / std (masking NaN)."""
    tgt = torch.stack([torch.stack([d[f].view(-1) for f in TARGET_FIELDS], dim=0)
                       for d in dataset], dim=0).squeeze(-1)  # [N, 5]
    tgt = tgt.float()
    means, stds = [], []
    for i in range(len(TARGET_FIELDS)):
        v = tgt[:, i]
        v = v[~torch.isnan(v)]
        means.append(float(v.mean()))
        stds.append(float(v.std().clamp(min=1e-6)))
    return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--train-pt", required=True)
    p.add_argument("--test-pt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--head-init", default=None,
                   help="Optional .pt of heads state_dict to warm-start from.")
    p.add_argument("--unfreeze-layers", type=int, default=2)
    p.add_argument("--max-train", type=int, default=200000)
    p.add_argument("--max-test",  type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4, help="Heads LR.")
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--n-mp-layers", type=int, default=2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="thermogen")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-group", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print(f"Loading labeled train from {args.train_pt}")
    train_list = load_labeled_subset(args.train_pt, args.max_train, args.seed)
    print(f"  {len(train_list):,} train molecules")
    print(f"Loading labeled test from {args.test_pt}")
    test_list  = load_labeled_subset(args.test_pt,  args.max_test,  args.seed)
    print(f"  {len(test_list):,} test molecules")

    target_mean, target_std = compute_target_stats(train_list)
    print(f"target means: {dict(zip(TARGET_FIELDS, target_mean))}")
    print(f"target stds:  {dict(zip(TARGET_FIELDS, target_std))}")

    # Attach normalized targets on every Data object
    t_mean_t = torch.tensor(target_mean, dtype=torch.float32)
    t_std_t  = torch.tensor(target_std,  dtype=torch.float32)
    for d in train_list + test_list:
        for i, f in enumerate(TARGET_FIELDS):
            d[f + "_norm"] = (d[f].view(-1) - t_mean_t[i]) / t_std_t[i]

    train_loader = DataLoader(train_list, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_list,  batch_size=args.batch_size, shuffle=False)

    # --- Load backbone + heads ---
    device = torch.device(args.device)
    backbone, cfg = load_backbone(args.ckpt, args.config, device)
    bb_trainable = unfreeze_last_n_layers(backbone, args.unfreeze_layers)

    heads = ThermoHeadModel(
        dim=cfg.dynamics.model_args.invariant_node_feat_dim,
        n_mp_layers=args.n_mp_layers,
    ).to(device)
    if args.head_init:
        print(f"Warm-starting heads from {args.head_init}")
        heads.load_state_dict(torch.load(args.head_init, map_location=device))

    # --- Optimizer with two LR groups ---
    param_groups = [{"params": heads.parameters(), "lr": args.lr, "name": "heads"}]
    if bb_trainable:
        param_groups.append({"params": bb_trainable, "lr": args.backbone_lr,
                              "name": "backbone_tail"})
    opt = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # --- wandb ---
    wb = None
    if args.wandb:
        import wandb as _wandb
        wb = _wandb
        wb.init(project=args.wandb_project,
                 name=args.wandb_name or f"cont_u{args.unfreeze_layers}_n{args.max_train}_s{args.seed}",
                 group=args.wandb_group, config=vars(args))

    t_type = str(cfg.interpolant.time_type)
    t_max = cfg.interpolant.timesteps - 1 if t_type == "discrete" else 1.0

    # --- Train ---
    global_step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        backbone.train(); heads.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"ep {epoch+1}/{args.epochs}"):
            batch = batch.to(device)
            tgt_norm = torch.stack(
                [batch[f + "_norm"].view(-1) for f in TARGET_FIELDS], dim=1
            )  # [B, 5]
            preds = forward_heads(backbone, heads, batch, t_max, t_type)
            loss_ext = masked_mse(preds["ext"], tgt_norm[:, EXTENSIVE_IDX])
            loss_mp  = masked_mse(preds["mp"],  tgt_norm)
            loss = loss_ext + loss_mp
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(heads.parameters()) + bb_trainable, args.grad_clip
            )
            opt.step()
            losses.append(loss.item())
            if wb is not None:
                lrs = {g["name"]: g["lr"] for g in opt.param_groups}
                wb.log({"train/loss": float(loss.item()),
                         "train/loss_ext": float(loss_ext.item()),
                         "train/loss_mp":  float(loss_mp.item()),
                         **{f"lr/{k}": v for k, v in lrs.items()}},
                       step=global_step)
            global_step += 1
        sched.step()

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            rows = eval_loop(backbone, heads, test_loader, device,
                             t_max, t_type, target_mean, target_std)
            avg_mae_mp_norm = float(np.mean(
                [r["mae_mp"] / target_std[TARGET_FIELDS.index(r["target"])]
                 for r in rows if "mae_mp" in r]
            ))
            print(f"[ep {epoch+1:>3d}]  train_loss={np.mean(losses):.4f}  "
                  f"test_mae(std-norm avg)={avg_mae_mp_norm:.4f}")
            if wb is not None:
                log = {"epoch": epoch + 1,
                       "train/loss_epoch": float(np.mean(losses)),
                       "test/mae_avg_norm_mp": avg_mae_mp_norm}
                for r in rows:
                    for k in ("mae_ext", "r2_ext", "mae_mp", "r2_mp"):
                        if k in r:
                            log[f"test/{k}_{r['target']}"] = r[k]
                wb.log(log, step=global_step)

    print(f"\nTotal wall time: {time.time()-t0:.1f}s")

    # --- Final report + save ---
    rows = eval_loop(backbone, heads, test_loader, device,
                     t_max, t_type, target_mean, target_std)
    print_report(rows)

    torch.save(heads.state_dict(), out_dir / "heads_final.pt")
    # Save a slim backbone checkpoint that includes only unfrozen weights deltas
    # (full state is fine too for small configs).
    torch.save({k: v.cpu() for k, v in backbone.state_dict().items()},
               out_dir / "backbone_final.pt")

    with open(out_dir / "report.json", "w") as f:
        json.dump({"args": vars(args), "rows": rows,
                   "target_mean": target_mean.tolist(),
                   "target_std":  target_std.tolist()}, f, indent=2)
    print(f"Saved checkpoints + report to {out_dir}")
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
