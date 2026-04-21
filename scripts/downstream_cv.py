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


# ---------------------------------------------------------------------------
# Single-target head — sum over atoms (extensive-friendly) PLUS attention-
# pooled MP for intensive-like properties. Ensemble both; pick per dataset
# at eval time.
# ---------------------------------------------------------------------------

class SingleTargetHead(nn.Module):
    """Per-atom MLP + scatter_sum AND attention-pooled MP, concat, MLP head.
    Reusing the pattern from megalodon.models.thermo_heads but for one scalar."""

    def __init__(self, dim=256, hidden=128, n_mp_layers=2, n_heads=4):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.head_dim = dim // n_heads
        self.n_heads = n_heads
        self.n_mp_layers = n_mp_layers

        # Extensive branch
        self.ext_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

        # MP branch
        self.q = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_mp_layers))
        self.k = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_mp_layers))
        self.v = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_mp_layers))
        self.o = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_mp_layers))
        self.mol_upd = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim),
                          nn.SiLU(), nn.Linear(dim, dim))
            for _ in range(n_mp_layers)
        )
        self.atm_upd = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim),
                          nn.SiLU(), nn.Linear(dim, dim))
            for _ in range(n_mp_layers)
        )
        self.mp_final = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, H, batch_idx):
        # Ext
        per_atom = self.ext_mlp(H)
        ext_pred = scatter_sum(per_atom, batch_idx, dim=0).squeeze(-1)

        # MP
        N_mols = int(batch_idx.max().item()) + 1
        mol_H = scatter_mean(H, batch_idx, dim=0)
        scale = self.head_dim ** -0.5
        for l in range(self.n_mp_layers):
            q = self.q[l](mol_H).view(N_mols,      self.n_heads, self.head_dim)
            k = self.k[l](H).view(H.size(0),       self.n_heads, self.head_dim)
            v = self.v[l](H).view(H.size(0),       self.n_heads, self.head_dim)
            q_at = q[batch_idx]
            scores = (q_at * k).sum(-1) * scale
            alpha = scatter_softmax(scores, batch_idx, dim=0)
            agg = scatter_sum(alpha.unsqueeze(-1) * v, batch_idx, dim=0)
            agg = agg.reshape(N_mols, self.dim)
            agg = self.o[l](agg)
            mol_H = mol_H + self.mol_upd[l](agg)
            H = H + self.atm_upd[l](torch.cat([H, mol_H[batch_idx]], dim=-1))
        mp_pred = self.mp_final(mol_H).squeeze(-1)
        return {"ext": ext_pred, "mp": mp_pred}


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
                    args, device):
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
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    total_steps = max(1, (len(train_idx) // args.batch_size) * args.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps, eta_min=args.lr_min
    )

    for ep in range(args.epochs):
        model.train()
        for H_b, b_b, t_b in batch_iter(H, offsets, tgt_norm,
                                          train_idx.tolist(),
                                          args.batch_size, device, shuffle=True):
            has = has_target[train_idx[
                np.searchsorted(train_idx,
                                 np.arange(len(train_idx)))  # no-op slice
            ]]
            pred = model(H_b, b_b)
            valid = ~torch.isnan(t_b)
            if not valid.any():
                continue
            # Average of ext + mp as the loss target (symmetric)
            pe = pred["ext"][valid]
            pm = pred["mp"][valid]
            tv = t_b[valid]
            loss = 0.5 * ((pe - tv) ** 2).mean() + 0.5 * ((pm - tv) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

    # Evaluate
    model.eval()
    preds_ext, preds_mp, tgts_raw = [], [], []
    with torch.no_grad():
        for H_b, b_b, _ in batch_iter(H, offsets, tgt_norm,
                                       val_idx.tolist(),
                                       args.batch_size, device, shuffle=False):
            pred = model(H_b, b_b)
            preds_ext.append(pred["ext"].cpu().numpy())
            preds_mp.append(pred["mp"].cpu().numpy())
    preds_ext = np.concatenate(preds_ext) * std + mean
    preds_mp  = np.concatenate(preds_mp ) * std + mean
    val_has = has_target[val_idx].bool().numpy()
    y_true = targets[val_idx].float().numpy()
    mask = val_has & ~np.isnan(y_true)

    def _metrics(y_pred):
        return {
            "mae":  float(mean_absolute_error(y_true[mask], y_pred[mask])),
            "rmse": float(mean_squared_error(y_true[mask], y_pred[mask],
                                              squared=False)),
            "r2":   float(r2_score(y_true[mask], y_pred[mask])),
        }

    return {"n_train": int(tr_has.sum()), "n_val": int(mask.sum()),
            "target_mean": mean, "target_std": std,
            "ext": _metrics(preds_ext),
            "mp":  _metrics(preds_mp)}


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
    args = p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading dataset {args.dataset_pt}")
    ds = load_prepared_pt(args.dataset_pt)
    n = len(ds)
    print(f"  {n:,} molecules")

    # Extract H for every molecule (one-pass cache)
    print(f"Loading backbone {args.ckpt}")
    model, cfg = load_backbone(args.ckpt, args.config, device)
    cache_path = out_dir / "H_cache.pt"
    H, offsets, targets, has_target = extract_H(
        model, cfg, ds, list(range(n)),
        args.extract_batch_size, device, cache_path,
    )
    del model  # free GPU memory

    # K-fold split on indices where has_target==True
    all_idx = np.arange(n)
    labeled = all_idx[has_target.bool().numpy()]
    print(f"Labeled molecules: {len(labeled):,} / {n:,}")

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_reports = []
    t0 = time.time()
    for fold_i, (tr, vl) in enumerate(kf.split(labeled)):
        train_idx = labeled[tr]
        val_idx   = labeled[vl]
        print(f"\n=== Fold {fold_i+1}/{args.n_folds} | "
              f"train={len(train_idx)}  val={len(val_idx)} ===")
        rep = train_one_fold(H, offsets, targets, has_target,
                              train_idx, val_idx, args, device)
        rep["fold"] = fold_i
        fold_reports.append(rep)
        print(f"  ext  MAE={rep['ext']['mae']:.4f}  R2={rep['ext']['r2']:.3f}")
        print(f"  mp   MAE={rep['mp']['mae']:.4f}  R2={rep['mp']['r2']:.3f}")

    # Aggregate
    def _agg(key):
        vals = [r[key]["mae"] for r in fold_reports]
        return {"mae_mean": float(np.mean(vals)),
                 "mae_std":  float(np.std(vals)),
                 "r2_mean":  float(np.mean([r[key]["r2"] for r in fold_reports])),
                 "rmse_mean": float(np.mean([r[key]["rmse"] for r in fold_reports]))}
    summary = {"ext": _agg("ext"), "mp": _agg("mp"),
                "folds": fold_reports,
                "args": vars(args),
                "n_molecules": n,
                "n_labeled": int(has_target.sum().item()),
                "wall_seconds": round(time.time()-t0, 1)}
    report_path = out_dir / "cv_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print final table
    print("\n" + "=" * 70)
    print(f"{args.n_folds}-fold CV summary  |  {args.dataset_pt}")
    print("-" * 70)
    for head in ("ext", "mp"):
        a = summary[head]
        print(f"  {head:<4s}  MAE = {a['mae_mean']:.4f} ± {a['mae_std']:.4f}"
              f"  R² = {a['r2_mean']:.3f}  RMSE = {a['rmse_mean']:.4f}")
    print("=" * 70)
    print(f"Report -> {report_path}")


if __name__ == "__main__":
    main()
