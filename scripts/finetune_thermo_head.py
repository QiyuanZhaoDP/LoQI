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
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from rdkit.Chem.rdchem import Mol
from sklearn.metrics import mean_absolute_error, r2_score
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean, scatter_softmax, scatter_sum
from tqdm import tqdm

from megalodon.data.batch_preprocessor import BatchPreProcessor
from megalodon.models.module import Graph3DInterpolantModel

TARGET_FIELDS = ["enthalpy_298", "gibbs_298", "cv_gas", "entropy_gas", "enthalpy_0"]
# Treat as additive (sum over atoms is physical):
EXTENSIVE_IDX = [0, 1, 2, 4]   # enthalpy_298, gibbs_298, cv_gas, enthalpy_0
TARGET_UNITS = {
    "enthalpy_298": "kJ/mol", "gibbs_298": "kJ/mol",
    "cv_gas": "J/(mol*K)",    "entropy_gas": "J/(mol*K)",
    "enthalpy_0": "kJ/mol",
}


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


# ---------------------------------------------------------------------------
# Step B — heads
# ---------------------------------------------------------------------------

class ExtensiveSumHead(nn.Module):
    """per-atom MLP → scatter_sum → extensive property."""
    def __init__(self, dim=256, hidden=128, n_targets=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_targets),
        )

    def forward(self, H, batch):
        per_atom = self.mlp(H)
        return scatter_sum(per_atom, batch, dim=0)  # [N_mols, n_targets]


class AtomMolMP(nn.Module):
    """Bidirectional message passing between atoms and a per-molecule virtual
    node with attention-based atom → mol pooling.

    Each layer:
        q = W_q * mol_H,  k = W_k * H,  v = W_v * H
        alpha_i = softmax_per_mol(q_{b(i)} . k_i / sqrt(d))
        mol_H  ← mol_H  + MLP1( sum_i alpha_i * v_i )
        H      ← H      + MLP2( [H | mol_H[b(i)]] )
    """
    def __init__(self, dim=256, n_layers=2, hidden=128, n_targets=5):
        super().__init__()
        self.n_layers = n_layers
        self.dim = dim
        self.q_proj = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_layers))
        self.k_proj = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_layers))
        self.v_proj = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_layers))
        self.mol_update = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim),
                          nn.SiLU(), nn.Linear(dim, dim))
            for _ in range(n_layers)
        )
        self.atom_update = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim),
                          nn.SiLU(), nn.Linear(dim, dim))
            for _ in range(n_layers)
        )
        self.final = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_targets),
        )

    def forward(self, H, batch):
        mol_H = scatter_mean(H, batch, dim=0)  # [N_mols, dim]
        scale = 1.0 / math.sqrt(self.dim)
        for l in range(self.n_layers):
            q = self.q_proj[l](mol_H)              # [N_mols, dim]
            k = self.k_proj[l](H)                  # [N_atoms, dim]
            v = self.v_proj[l](H)                  # [N_atoms, dim]
            q_at = q[batch]                         # [N_atoms, dim]
            scores = (q_at * k).sum(-1) * scale     # [N_atoms]
            alpha = scatter_softmax(scores, batch, dim=0)  # [N_atoms]
            weighted = alpha.unsqueeze(-1) * v             # [N_atoms, dim]
            agg = scatter_sum(weighted, batch, dim=0)      # [N_mols, dim]
            mol_H = mol_H + self.mol_update[l](agg)

            mol_at = mol_H[batch]
            H = H + self.atom_update[l](torch.cat([H, mol_at], dim=-1))
        return self.final(mol_H)                   # [N_mols, n_targets]


class ThermoHeadModel(nn.Module):
    def __init__(self, dim=256, n_mp_layers=2):
        super().__init__()
        self.ext = ExtensiveSumHead(dim=dim, n_targets=len(EXTENSIVE_IDX))
        self.mp  = AtomMolMP(dim=dim, n_layers=n_mp_layers,
                              n_targets=len(TARGET_FIELDS))

    def forward(self, H, batch):
        return {"ext": self.ext(H, batch), "mp": self.mp(H, batch)}


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


def masked_mse(pred, target):
    """pred, target: [B, K], target may have NaN; returns scalar mean over valid."""
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    diff = (pred - torch.nan_to_num(target)) * mask
    return (diff ** 2).sum() / mask.sum()


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
    p.add_argument("--config", required=True)
    p.add_argument("--train-pt", required=True)
    p.add_argument("--test-pt", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--max-train", type=int, default=50000)
    p.add_argument("--max-test",  type=int, default=20000)
    p.add_argument("--extract-batch-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--n-mp-layers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--cache-dtype", choices=list(_CACHE_DTYPES.keys()), default="bf16",
                   help="H cache storage dtype. bf16 halves footprint vs fp32 with "
                        "effectively no quality loss for head training.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_tr = cache_dir / f"train_H_n{args.max_train}_s{args.seed}.pt"
    cache_te = cache_dir / f"test_H_n{args.max_test}_s{args.seed}.pt"

    # --- Step A: extract H if needed ---
    need_extract = args.no_cache or not cache_tr.exists() or not cache_te.exists()
    if need_extract:
        print("Loading frozen backbone")
        model_back, cfg = load_model(args.ckpt, args.config, args.device)
        if not cache_tr.exists() or args.no_cache:
            ds_tr, idx_tr = load_labeled_indices(args.train_pt, args.max_train, args.seed)
            extract_and_cache_H(model_back, cfg, ds_tr, idx_tr,
                                args.extract_batch_size, args.device, cache_tr,
                                desc="train-H", cache_dtype=args.cache_dtype)
        if not cache_te.exists() or args.no_cache:
            ds_te, idx_te = load_labeled_indices(args.test_pt, args.max_test, args.seed)
            extract_and_cache_H(model_back, cfg, ds_te, idx_te,
                                args.extract_batch_size, args.device, cache_te,
                                desc="test-H", cache_dtype=args.cache_dtype)
        del model_back
        torch.cuda.empty_cache() if args.device == "cuda" else None

    # --- Load cached H ---
    H_tr, off_tr, tgt_tr = load_H_cache(cache_tr)
    H_te, off_te, tgt_te = load_H_cache(cache_te)
    n_train = tgt_tr.shape[0]
    n_test  = tgt_te.shape[0]
    print(f"\nLoaded cache — train: {n_train:,} mols / {H_tr.shape[0]:,} atoms,  "
          f"test: {n_test:,} mols / {H_te.shape[0]:,} atoms")

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
    tgt_te_norm = (tgt_te - torch.tensor(target_mean)) / torch.tensor(target_std)

    # --- Step B: train heads ---
    device = torch.device(args.device)
    model = ThermoHeadModel(dim=H_tr.shape[-1], n_mp_layers=args.n_mp_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    idx_train = np.arange(n_train)
    idx_test  = np.arange(n_test)

    best_test_mae = float("inf")
    t0 = time.time()
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
        sched.step()

        if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == args.epochs - 1:
            rows = evaluate(model, H_te, off_te, tgt_te_norm, idx_test,
                            args.batch_size, device,
                            target_mean, target_std)
            avg_mae_mp = np.mean([r["mae_mp"] / target_std[TARGET_FIELDS.index(r["target"])]
                                   for r in rows if "mae_mp" in r])
            print(f"[ep {epoch+1:>3d}]  train_loss={np.mean(losses):.4f}  "
                  f"test_mae(std-norm avg)={avg_mae_mp:.4f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}")
            if avg_mae_mp < best_test_mae:
                best_test_mae = avg_mae_mp

    print(f"\nTotal training time: {time.time()-t0:.1f}s")

    # --- Step C: final report ---
    rows = evaluate(model, H_te, off_te, tgt_te_norm, idx_test,
                    args.batch_size, device, target_mean, target_std)
    print_report(rows)

    out_path = cache_dir / "finetune_report.json"
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "rows": rows,
                   "target_mean": target_mean.tolist(),
                   "target_std":  target_std.tolist()}, f, indent=2)
    print(f"\nReport saved -> {out_path}")


if __name__ == "__main__":
    main()
