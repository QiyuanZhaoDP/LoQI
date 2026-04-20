"""Phase 0/1 hybrid: train thermo prediction heads on top of a frozen LoQI
backbone.

Two heads are trained jointly on the SAME per-atom feature map H from the
frozen checkpoint:

  ExtensiveSumHead  — MLP(H_atom) + scatter_sum over atoms.
                      Physically-principled for ADDITIVE properties
                      (Hf_0, Hf_298, Gf_298, Cv).

  AtomMolMP         — bidirectional message passing between atoms and a
                      per-molecule virtual node, with attention-based
                      atom→mol pooling. Learns arbitrary (non-additive)
                      aggregation — intended for intensive / shape-dependent
                      properties (S0 is the main one; also a general fallback).

Targets: 5 TCIT columns produced by scripts/label_thermo.py.

Workflow:
  Step A. Cache per-molecule H tensors by running the frozen backbone once
          per split (chunked, written with torch.save).
  Step B. Train both heads jointly on cached H with z-scored targets and
          NaN-masked loss (semi-supervised: missing labels don't contribute).
  Step C. De-normalize predictions, report MAE + R^2 per head per target,
          and compare to the Ridge baseline.

Usage:
  python scripts/finetune_thermo_head.py \\
      --ckpt data/loqi.ckpt --config scripts/conf/loqi/loqi.yaml \\
      --train-pt data/chembl3d_stereo/processed/train_h_thermo.pt \\
      --test-pt  data/chembl3d_stereo/processed/test_h_thermo.pt \\
      --cache-dir /tmp/ft_cache \\
      --max-train 50000 --max-test 20000 \\
      --epochs 50 --lr 3e-4 --batch-size 256 --device cuda

Re-running with the same --cache-dir reuses cached H; only heads retrain.
"""
import argparse
import glob
import json
import math
import sys
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
    apply_thermo_config_yaml,
    masked_mse,
)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class _TempDataset(InMemoryDataset):
    def __init__(self, data, slices):
        super().__init__(".")
        self.data, self.slices = data, slices
        self._indices = None


def load_labeled_indices(pt_path, max_n=None, seed=0):
    with torch.serialization.safe_globals(
        [DataEdgeAttr, DataTensorAttr, GlobalStorage, Mol]
    ):
        data, slices = torch.load(pt_path)
    ds = _TempDataset(data, slices)
    idx = []
    flag = ds.data.thermo_has_label.view(-1)
    for i in range(len(ds)):
        if bool(flag[i].item()):
            idx.append(i)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    if max_n is not None:
        idx = idx[:max_n]
    return ds, idx


# ---------------------------------------------------------------------------
# Step A — cache per-molecule H (frozen backbone forward)
# ---------------------------------------------------------------------------

def load_model(ckpt, cfg_path, device):
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
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model, cfg


_CACHE_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


@torch.no_grad()
def extract_and_cache_H(model, cfg, ds, indices, batch_size, device,
                        cache_path, desc, cache_dtype="bf16"):
    """Run the frozen backbone once per molecule; save H concatenated plus
    a per-molecule offset array + targets. H is stored in `cache_dtype`
    (bf16 by default to halve disk/RAM footprint at full-dataset scale).
    """
    t_type = str(cfg.interpolant.time_type)
    t_max = cfg.interpolant.timesteps - 1 if t_type == "discrete" else 1.0

    subset = [ds[i] for i in indices]
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    H_chunks = []
    offsets = [0]
    targets = []
    for batch in tqdm(loader, desc=desc):
        batch = batch.to(device)
        bs = int(batch.batch.max().item()) + 1
        tgt = torch.stack(
            [batch[f].view(-1).float() for f in TARGET_FIELDS], dim=1
        )  # [bs, 5]

        if model.batch_preprocessor is not None:
            batch = model.batch_preprocessor(batch)

        if t_type == "discrete":
            time_tensor = torch.full((bs,), t_max, dtype=torch.long, device=device)
        else:
            time_tensor = torch.full((bs,), t_max, dtype=torch.float32, device=device)

        out, _, _ = model(batch, time_tensor)
        H = out["H"].cpu()

        # split H by molecule into per-mol pieces to build offsets
        counts = torch.bincount(batch.batch.cpu(), minlength=bs).tolist()
        for c in counts:
            offsets.append(offsets[-1] + c)
        H_chunks.append(H)
        targets.append(tgt.cpu())

    H_all = torch.cat(H_chunks, dim=0).contiguous()
    target_dtype = _CACHE_DTYPES[cache_dtype]
    if target_dtype != torch.float32:
        H_all = H_all.to(target_dtype)
    offsets = torch.tensor(offsets, dtype=torch.long)
    targets = torch.cat(targets, dim=0).contiguous()

    torch.save(
        {"H": H_all, "offsets": offsets, "targets": targets,
         "cache_dtype": cache_dtype},
        cache_path,
    )
    bytes_per = {"fp32": 4, "bf16": 2, "fp16": 2}[cache_dtype]
    mb = H_all.numel() * bytes_per / (1024 ** 2)
    print(f"  saved H cache -> {cache_path}  "
          f"H.shape={tuple(H_all.shape)} dtype={cache_dtype}  "
          f"H size={mb:.1f} MB")


def load_H_cache(path):
    d = torch.load(path)
    return d["H"], d["offsets"], d["targets"]


def merge_shard_caches(shard_paths, out_path):
    """Concatenate per-shard H caches into the merged cache file expected by
    the training loop. Order across shards is preserved; offsets are
    adjusted so each molecule's atom-slice still lands on the right rows.
    """
    H_parts, target_parts = [], []
    merged_offsets = [0]
    saved_dtype = None
    for path in shard_paths:
        d = torch.load(path)
        H, offsets, tgt = d["H"], d["offsets"], d["targets"]
        saved_dtype = d.get("cache_dtype", "fp32")
        H_parts.append(H)
        target_parts.append(tgt)
        base = merged_offsets[-1]
        # offsets[0] is 0, subsequent entries are cumulative within shard
        for o in offsets[1:].tolist():
            merged_offsets.append(int(o) + base)
    torch.save(
        {"H": torch.cat(H_parts, dim=0),
         "offsets": torch.tensor(merged_offsets, dtype=torch.long),
         "targets": torch.cat(target_parts, dim=0),
         "cache_dtype": saved_dtype},
        out_path,
    )
    return len(H_parts)


# ---------------------------------------------------------------------------
# Step B — heads
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def batch_iter(H, offsets, targets, indices, batch_size, device, shuffle=True):
    """Yield (H_batch, batch_index, targets_batch) from a cached index subset.
    Casts H to fp32 on transfer (cache may be bf16/fp16)."""
    order = np.array(indices)
    if shuffle:
        np.random.shuffle(order)
    for start in range(0, len(order), batch_size):
        mol_ids = order[start:start + batch_size]
        H_list = []
        batch_list = []
        for bi, mi in enumerate(mol_ids):
            s, e = int(offsets[mi]), int(offsets[mi + 1])
            H_list.append(H[s:e])
            batch_list.append(torch.full((e - s,), bi, dtype=torch.long))
        H_b = torch.cat(H_list, dim=0).to(device=device, dtype=torch.float32)
        b_b = torch.cat(batch_list, dim=0).to(device)
        t_b = targets[mol_ids].to(device)
        yield H_b, b_b, t_b


def evaluate(model, H_te, off_te, tgt_te_norm, indices, batch_size, device,
             target_mean, target_std):
    model.eval()
    preds_ext = np.full((len(indices), len(EXTENSIVE_IDX)), np.nan)
    preds_mp  = np.full((len(indices), len(TARGET_FIELDS)), np.nan)
    tgts_raw  = tgt_te_norm[indices].cpu().numpy() * target_std + target_mean
    pos = 0
    with torch.no_grad():
        for H_b, b_b, t_b in batch_iter(
            H_te, off_te, tgt_te_norm, indices, batch_size, device, shuffle=False
        ):
            out = model(H_b, b_b)
            preds_ext[pos:pos + out["ext"].shape[0]] = (
                out["ext"].cpu().numpy() * target_std[EXTENSIVE_IDX]
                + target_mean[EXTENSIVE_IDX]
            )
            preds_mp[pos:pos + out["mp"].shape[0]] = (
                out["mp"].cpu().numpy() * target_std + target_mean
            )
            pos += out["ext"].shape[0]

    rows = []
    for i, name in enumerate(TARGET_FIELDS):
        mask = ~np.isnan(tgts_raw[:, i])
        if mask.sum() < 20:
            rows.append({"target": name, "note": "too few"})
            continue
        y_true = tgts_raw[mask, i]
        row = {
            "target": name, "unit": TARGET_UNITS[name],
            "n_test": int(mask.sum()),
        }
        # MP head always predicts all 5
        yp_mp = preds_mp[mask, i]
        row["mae_mp"] = float(mean_absolute_error(y_true, yp_mp))
        row["r2_mp"]  = float(r2_score(y_true, yp_mp))
        # Ext head only for additive properties
        if i in EXTENSIVE_IDX:
            j = EXTENSIVE_IDX.index(i)
            yp_ext = preds_ext[mask, j]
            row["mae_ext"] = float(mean_absolute_error(y_true, yp_ext))
            row["r2_ext"]  = float(r2_score(y_true, yp_ext))
        rows.append(row)
    return rows


def print_report(rows):
    print("\n" + "=" * 92)
    print(f"{'target':<14s} {'unit':<11s} "
          f"{'MAE_ext':>10s} {'R2_ext':>8s} "
          f"{'MAE_mp':>10s} {'R2_mp':>8s} "
          f"{'n_test':>8s}")
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True,
                   help="LoQI backbone config YAML (scripts/conf/loqi/loqi.yaml).")
    p.add_argument("--thermo-config", default=None,
                   help="Thermo head + training YAML "
                        "(scripts/conf/thermo/finetune.yaml). "
                        "YAML values override argparse defaults; CLI flags "
                        "still override the YAML.")
    p.add_argument("--train-pt", required=True)
    p.add_argument("--val-pt", required=True,
                   help="Used for per-epoch evaluation during training.")
    p.add_argument("--test-pt", required=True,
                   help="Used ONLY for final evaluation after training.")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--max-train", type=int, default=None,
                   help="Cap on labeled train molecules (default: use all).")
    p.add_argument("--max-val",   type=int, default=None,
                   help="Cap on labeled val molecules (default: use all).")
    p.add_argument("--max-test",  type=int, default=None,
                   help="Cap on labeled test molecules (default: use all).")
    p.add_argument("--extract-batch-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--n-mp-layers", type=int, default=2,
                   help="Number of atom<->mol MP rounds in AtomMolMP.")
    p.add_argument("--mp-n-heads", type=int, default=4,
                   help="Attention heads in AtomMolMP. Must divide 256.")
    p.add_argument("--head-hidden", type=int, default=128,
                   help="Hidden dim inside both heads. Scale up to 256/512 "
                        "for more capacity.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--cache-dtype", choices=list(_CACHE_DTYPES.keys()), default="bf16",
                   help="H cache storage dtype. bf16 halves footprint vs fp32 with "
                        "effectively no quality loss for head training.")
    p.add_argument("--shard-id", type=int, default=None,
                   help="For multi-GPU extraction: 0..(n_shards-1). When set, this "
                        "process extracts only its shard and exits (no training). "
                        "Training run (shard-id=None) auto-merges shard files.")
    p.add_argument("--n-shards", type=int, default=1)
    # wandb
    p.add_argument("--wandb", action="store_true", help="Log to Weights & Biases.")
    p.add_argument("--wandb-project", default="thermogen")
    p.add_argument("--wandb-name", default=None,
                   help="Run name. Default: ft_thermo_n<max_train>_s<seed>.")
    p.add_argument("--wandb-group", default=None)

    # Two-pass parsing: read --thermo-config first, then apply its values as
    # new argparse defaults so subsequent CLI flags still win.
    known, _ = p.parse_known_args()
    if known.thermo_config:
        applied = apply_thermo_config_yaml(p, known.thermo_config)
        print(f"Loaded thermo config {known.thermo_config}: {applied}")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_tr = cache_dir / f"train_H_n{args.max_train}_s{args.seed}.pt"
    cache_va = cache_dir / f"val_H_n{args.max_val}_s{args.seed}.pt"
    cache_te = cache_dir / f"test_H_n{args.max_test}_s{args.seed}.pt"

    # --- Shard extraction mode: extract my slice, exit ---
    if args.shard_id is not None:
        assert 0 <= args.shard_id < args.n_shards, "shard-id out of range"
        shard_tr = cache_tr.with_name(
            cache_tr.stem + f".shard{args.shard_id}_of_{args.n_shards}.pt"
        )
        print(f"[shard {args.shard_id}/{args.n_shards}] loading frozen backbone")
        model_back, cfg = load_model(args.ckpt, args.config, args.device)
        if not shard_tr.exists() or args.no_cache:
            ds_tr, idx_tr = load_labeled_indices(args.train_pt, args.max_train, args.seed)
            my_idx = idx_tr[args.shard_id::args.n_shards]
            print(f"[shard {args.shard_id}] extracting {len(my_idx):,} molecules")
            extract_and_cache_H(
                model_back, cfg, ds_tr, my_idx,
                args.extract_batch_size, args.device, shard_tr,
                desc=f"train-H[{args.shard_id}]",
                cache_dtype=args.cache_dtype,
            )
        # Shard 0 also extracts val + test (small, single-GPU is fine)
        if args.shard_id == 0:
            if not cache_va.exists() or args.no_cache:
                ds_va, idx_va = load_labeled_indices(args.val_pt, args.max_val, args.seed)
                extract_and_cache_H(
                    model_back, cfg, ds_va, idx_va,
                    args.extract_batch_size, args.device, cache_va,
                    desc="val-H", cache_dtype=args.cache_dtype,
                )
            if not cache_te.exists() or args.no_cache:
                ds_te, idx_te = load_labeled_indices(args.test_pt, args.max_test, args.seed)
                extract_and_cache_H(
                    model_back, cfg, ds_te, idx_te,
                    args.extract_batch_size, args.device, cache_te,
                    desc="test-H", cache_dtype=args.cache_dtype,
                )
        print(f"[shard {args.shard_id}] done.")
        sys.exit(0)

    # --- Merge shard files into main cache if present ---
    if not cache_tr.exists() or args.no_cache:
        shard_pattern = str(cache_tr.with_name(cache_tr.stem + ".shard*_of_*.pt"))
        shard_paths = sorted(glob.glob(shard_pattern))
        if shard_paths:
            print(f"Merging {len(shard_paths)} shard cache(s) -> {cache_tr}")
            merge_shard_caches(shard_paths, cache_tr)

    # --- Step A: extract H if still needed (non-shard single-GPU path) ---
    need_extract = args.no_cache or not all(
        c.exists() for c in (cache_tr, cache_va, cache_te)
    )
    if need_extract:
        print("Loading frozen backbone")
        model_back, cfg = load_model(args.ckpt, args.config, args.device)
        for (pt, cache_path, mx, desc) in [
            (args.train_pt, cache_tr, args.max_train, "train-H"),
            (args.val_pt,   cache_va, args.max_val,   "val-H"),
            (args.test_pt,  cache_te, args.max_test,  "test-H"),
        ]:
            if not cache_path.exists() or args.no_cache:
                ds, idx = load_labeled_indices(pt, mx, args.seed)
                extract_and_cache_H(model_back, cfg, ds, idx,
                                    args.extract_batch_size, args.device, cache_path,
                                    desc=desc, cache_dtype=args.cache_dtype)
        del model_back
        torch.cuda.empty_cache() if args.device == "cuda" else None

    # --- Load cached H ---
    H_tr, off_tr, tgt_tr = load_H_cache(cache_tr)
    H_va, off_va, tgt_va = load_H_cache(cache_va)
    H_te, off_te, tgt_te = load_H_cache(cache_te)
    n_train = tgt_tr.shape[0]
    n_val   = tgt_va.shape[0]
    n_test  = tgt_te.shape[0]
    print(f"\nLoaded cache — train: {n_train:,} mols / {H_tr.shape[0]:,} atoms,  "
          f"val: {n_val:,} mols,  test: {n_test:,} mols")

    # --- z-score normalize targets (from train stats, masking NaN) ---
    mask_tr = ~torch.isnan(tgt_tr)
    mean_list, std_list = [], []
    for i in range(len(TARGET_FIELDS)):
        vals = tgt_tr[:, i][mask_tr[:, i]]
        mean_list.append(float(vals.mean()))
        std_list.append(float(vals.std().clamp(min=1e-6)))
    target_mean = np.array(mean_list, dtype=np.float32)
    target_std  = np.array(std_list,  dtype=np.float32)
    print(f"target means: {dict(zip(TARGET_FIELDS, target_mean))}")
    print(f"target stds:  {dict(zip(TARGET_FIELDS, target_std))}")

    tgt_tr_norm = (tgt_tr - torch.tensor(target_mean)) / torch.tensor(target_std)
    tgt_va_norm = (tgt_va - torch.tensor(target_mean)) / torch.tensor(target_std)
    tgt_te_norm = (tgt_te - torch.tensor(target_mean)) / torch.tensor(target_std)

    # --- wandb init (optional) ---
    wb = None
    if args.wandb:
        import wandb as _wandb
        wb = _wandb
        wb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"ft_thermo_n{args.max_train}_s{args.seed}",
            group=args.wandb_group,
            config=vars(args),
        )

    # --- Step B: train heads ---
    device = torch.device(args.device)
    model = ThermoHeadModel(
        dim=H_tr.shape[-1],
        n_mp_layers=args.n_mp_layers,
        n_mp_heads=args.mp_n_heads,
        hidden=args.head_hidden,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    idx_train = np.arange(n_train)
    idx_val   = np.arange(n_val)
    idx_test  = np.arange(n_test)

    best_val_mae = float("inf")
    t0 = time.time()
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for H_b, b_b, t_b in batch_iter(
            H_tr, off_tr, tgt_tr_norm, idx_train, args.batch_size, device, shuffle=True
        ):
            out = model(H_b, b_b)
            loss_ext = masked_mse(out["ext"], t_b[:, EXTENSIVE_IDX])
            loss_mp  = masked_mse(out["mp"],  t_b)
            loss = loss_ext + loss_mp
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            if wb is not None:
                wb.log({"train/loss": float(loss.item()),
                         "train/loss_ext": float(loss_ext.item()),
                         "train/loss_mp":  float(loss_mp.item()),
                         "train/lr": sched.get_last_lr()[0]}, step=global_step)
            global_step += 1
        sched.step()

        if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == args.epochs - 1:
            rows = evaluate(model, H_va, off_va, tgt_va_norm, idx_val,
                            args.batch_size, device,
                            target_mean, target_std)
            avg_mae_mp = np.mean([r["mae_mp"] / target_std[TARGET_FIELDS.index(r["target"])]
                                   for r in rows if "mae_mp" in r])
            print(f"[ep {epoch+1:>3d}]  train_loss={np.mean(losses):.4f}  "
                  f"val_mae(std-norm avg)={avg_mae_mp:.4f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}")
            if avg_mae_mp < best_val_mae:
                best_val_mae = avg_mae_mp
            if wb is not None:
                eval_log = {"epoch": epoch + 1,
                            "train/loss_epoch": float(np.mean(losses)),
                            "val/mae_avg_norm_mp": float(avg_mae_mp)}
                for r in rows:
                    if "mae_mp" in r:
                        eval_log[f"val/mae_mp_{r['target']}"] = r["mae_mp"]
                        eval_log[f"val/r2_mp_{r['target']}"]  = r["r2_mp"]
                    if "mae_ext" in r:
                        eval_log[f"val/mae_ext_{r['target']}"] = r["mae_ext"]
                        eval_log[f"val/r2_ext_{r['target']}"]  = r["r2_ext"]
                wb.log(eval_log, step=global_step)

    print(f"\nTotal training time: {time.time()-t0:.1f}s")

    # --- Step C: final report on HELD-OUT TEST set ---
    print("\n=== Final evaluation on held-out test set ===")
    rows = evaluate(model, H_te, off_te, tgt_te_norm, idx_test,
                    args.batch_size, device, target_mean, target_std)
    print_report(rows)

    out_path = cache_dir / "finetune_report.json"
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "rows": rows,
                   "target_mean": target_mean.tolist(),
                   "target_std":  target_std.tolist()}, f, indent=2)
    # Save heads so continuation_training.py can --head-init from here.
    heads_path = cache_dir / "heads_final.pt"
    torch.save(model.state_dict(), heads_path)
    print(f"Heads saved -> {heads_path}")
    if wb is not None:
        final_log = {}
        for r in rows:
            for k in ("mae_ext", "r2_ext", "mae_mp", "r2_mp"):
                if k in r:
                    final_log[f"final_test/{k}_{r['target']}"] = r[k]
        wb.log(final_log, step=global_step)
        wb.finish()
    print(f"\nReport saved -> {out_path}")


if __name__ == "__main__":
    main()
