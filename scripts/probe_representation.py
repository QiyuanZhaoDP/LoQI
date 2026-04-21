"""Phase 0: Probe frozen LoQI representations for thermo prediction.

Extract atom-level invariant features H from a frozen Graph3DInterpolantModel
checkpoint at the cleanest time step, pool to molecule-level via
scatter_mean, cache [N_mols, 256] embeddings per split, then train Ridge
regression per thermodynamic target and report MAE + R^2 on the test split.

Targets (from scripts/label_thermo.py output):
    enthalpy_298  Hf_298   kJ/mol
    gibbs_298     Gf_298   kJ/mol
    cv_gas        Cv       J/(mol*K)
    entropy_gas   S0       J/(mol*K)
    enthalpy_0    Hf_0     kJ/mol

Baselines reported alongside Ridge-on-H:
    mean     — always predict training mean (trivial lower bound)
    n_atoms  — Ridge on [n_atoms, 1] feature (captures size scaling)

Usage:
    python scripts/probe_representation.py \\
        --ckpt       data/loqi.ckpt \\
        --config     scripts/conf/loqi/loqi.yaml \\
        --train-pt   data/chembl3d_stereo/processed/train_h_thermo.pt \\
        --test-pt    data/chembl3d_stereo/processed/test_h_thermo.pt \\
        --cache-dir  /tmp/probe_cache \\
        --max-train  50000 --max-test 20000 \\
        --batch-size 64 --device cuda

Re-running with the same --cache-dir reuses cached embeddings and only
re-fits the regressors (fast iteration on probe analysis).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from rdkit.Chem.rdchem import Mol
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean
from tqdm import tqdm

from megalodon.data.batch_preprocessor import BatchPreProcessor
from megalodon.models.module import Graph3DInterpolantModel

TARGET_FIELDS = ["enthalpy_298", "gibbs_298", "cv_gas", "entropy_gas", "enthalpy_0"]
TARGET_UNITS = {
    "enthalpy_298": "kJ/mol",
    "gibbs_298":    "kJ/mol",
    "cv_gas":       "J/(mol*K)",
    "entropy_gas":  "J/(mol*K)",
    "enthalpy_0":   "kJ/mol",
}


class _TempDataset(InMemoryDataset):
    def __init__(self, data, slices):
        super().__init__(".")
        self.data, self.slices = data, slices
        self._indices = None


def load_labeled_dataset(pt_path, labeled_only=True, max_n=None, seed=0):
    with torch.serialization.safe_globals(
        [DataEdgeAttr, DataTensorAttr, GlobalStorage, Mol]
    ):
        data, slices = torch.load(pt_path)
    ds = _TempDataset(data, slices)
    indices = list(range(len(ds)))
    if labeled_only:
        keep = []
        for i in indices:
            d = ds[i]
            flag = getattr(d, "thermo_has_label", None)
            if flag is not None and bool(flag.item() if hasattr(flag, "item") else flag):
                keep.append(i)
        indices = keep
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    if max_n is not None:
        indices = indices[:max_n]
    return ds, indices


def load_model(ckpt_path, config_path, device):
    cfg = OmegaConf.load(config_path)
    preprocessor = BatchPreProcessor(cfg.data.aug_rotations, cfg.data.scale_coords)
    model = Graph3DInterpolantModel.load_from_checkpoint(
        ckpt_path,
        loss_params=cfg.loss,
        interpolant_params=cfg.interpolant,
        sampling_params=cfg.sample,
        batch_preprocessor=preprocessor,
        map_location=device,
    )
    model.eval()
    model.to(device)
    return model, cfg


@torch.no_grad()
def extract_embeddings(model, cfg, dataset, indices, batch_size, device, desc=""):
    """Run frozen model on subset (via index list); return (emb, targets, n_atoms).

    emb:     [N_mols, 256]     scatter_mean(H, batch) at t=max
    targets: [N_mols, 5]       in order of TARGET_FIELDS
    n_atoms: [N_mols]          atoms per molecule (for size baseline)
    """
    # Discrete t_max = timesteps - 1 (cleanest end of denoising).
    t_type = str(cfg.interpolant.time_type)
    t_max = cfg.interpolant.timesteps - 1 if t_type == "discrete" else 1.0

    subset = [dataset[i] for i in indices]
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    embs, targets, atom_counts = [], [], []
    for batch in tqdm(loader, desc=desc or "extract"):
        batch = batch.to(device)

        # Capture targets + atom counts BEFORE preprocessing (the preprocessor
        # renames batch.x -> batch.h which can confuse downstream field access).
        bs = int(batch.batch.max().item()) + 1
        tgt_vals = torch.stack(
            [batch[f].view(-1).float() for f in TARGET_FIELDS], dim=1
        )  # [bs, 5]
        n_atoms_batch = torch.bincount(batch.batch, minlength=bs).cpu()

        # Training-path preprocessing (rotate/scale coords, x<-pos, h<-x, FC graph).
        if model.batch_preprocessor is not None:
            batch = model.batch_preprocessor(batch)

        if t_type == "discrete":
            time_tensor = torch.full((bs,), t_max, dtype=torch.long, device=device)
        else:
            time_tensor = torch.full((bs,), t_max, dtype=torch.float32, device=device)

        out, _, _ = model(batch, time_tensor)
        H = out["H"]                                      # [N_atoms, 256]
        mol_repr = scatter_mean(H, batch.batch, dim=0)    # [bs, 256]

        embs.append(mol_repr.cpu())
        targets.append(tgt_vals.cpu())
        atom_counts.append(n_atoms_batch)

    return (
        torch.cat(embs).numpy(),
        torch.cat(targets).numpy(),
        torch.cat(atom_counts).numpy(),
    )


def fit_and_report(X_train, y_train, n_train, X_test, y_test, n_test):
    """Fit three predictors per target; return dict of MAE+R^2 tables."""
    rows = []
    for ti, name in enumerate(TARGET_FIELDS):
        mask_tr = ~np.isnan(y_train[:, ti])
        mask_te = ~np.isnan(y_test[:, ti])
        y_tr = y_train[mask_tr, ti]
        y_te = y_test[mask_te, ti]
        if len(y_tr) < 50 or len(y_te) < 50:
            rows.append({"target": name, "note": "too few labels"})
            continue

        mean_pred = np.full_like(y_te, fill_value=float(y_tr.mean()))
        mae_mean = mean_absolute_error(y_te, mean_pred)

        reg_n = Ridge(alpha=1.0).fit(n_train[mask_tr].reshape(-1, 1), y_tr)
        y_pred_n = reg_n.predict(n_test[mask_te].reshape(-1, 1))
        mae_n = mean_absolute_error(y_te, y_pred_n)
        r2_n = r2_score(y_te, y_pred_n)

        reg_h = Ridge(alpha=1.0).fit(X_train[mask_tr], y_tr)
        y_pred_h = reg_h.predict(X_test[mask_te])
        mae_h = mean_absolute_error(y_te, y_pred_h)
        r2_h = r2_score(y_te, y_pred_h)

        rows.append({
            "target": name,
            "unit": TARGET_UNITS[name],
            "n_train": int(mask_tr.sum()),
            "n_test": int(mask_te.sum()),
            "mae_mean_baseline": float(mae_mean),
            "mae_ridge_natoms":  float(mae_n),
            "r2_ridge_natoms":   float(r2_n),
            "mae_ridge_H":       float(mae_h),
            "r2_ridge_H":        float(r2_h),
            "mae_improvement_%": float(100 * (mae_n - mae_h) / max(mae_n, 1e-9)),
        })
    return rows


def print_report(rows):
    print("\n" + "=" * 88)
    print(f"{'target':<14s} {'unit':<11s} {'MAE_mean':>10s} "
          f"{'MAE_nA':>10s} {'MAE_H':>10s} {'R2_nA':>8s} {'R2_H':>8s} "
          f"{'H_vs_nA':>9s}")
    print("-" * 88)
    for r in rows:
        if "note" in r:
            print(f"{r['target']:<14s} {r['note']}")
            continue
        print(
            f"{r['target']:<14s} {r['unit']:<11s} "
            f"{r['mae_mean_baseline']:>10.3f} "
            f"{r['mae_ridge_natoms']:>10.3f} "
            f"{r['mae_ridge_H']:>10.3f} "
            f"{r['r2_ridge_natoms']:>8.3f} "
            f"{r['r2_ridge_H']:>8.3f} "
            f"{r['mae_improvement_%']:>8.1f}%"
        )
    print("=" * 88)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--train-pt", required=True)
    p.add_argument("--test-pt", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--max-train", type=int, default=50000)
    p.add_argument("--max-test", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-cache", action="store_true",
                   help="Recompute embeddings even if cache exists.")
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_tr = cache_dir / f"train_emb_n{args.max_train}_s{args.seed}.pt"
    cache_te = cache_dir / f"test_emb_n{args.max_test}_s{args.seed}.pt"

    need_model = args.no_cache or not cache_tr.exists() or not cache_te.exists()
    model = cfg = None
    if need_model:
        print(f"Loading model from {args.ckpt}")
        t0 = time.time()
        model, cfg = load_model(args.ckpt, args.config, args.device)
        print(f"  ({time.time()-t0:.1f}s)")

    def run_split(name, pt_path, max_n, cache_path):
        if cache_path.exists() and not args.no_cache:
            print(f"[{name}] cache hit: {cache_path}")
            blob = torch.load(cache_path)
            return blob["emb"], blob["targets"], blob["n_atoms"]
        print(f"[{name}] loading dataset {pt_path}")
        ds, idx = load_labeled_dataset(pt_path, labeled_only=True,
                                       max_n=max_n, seed=args.seed)
        print(f"[{name}] labeled={len(idx):,} (using {min(len(idx), max_n):,})")
        emb, tgt, nA = extract_embeddings(
            model, cfg, ds, idx, args.batch_size, args.device, desc=name
        )
        torch.save({"emb": emb, "targets": tgt, "n_atoms": nA,
                    "indices": idx}, cache_path)
        print(f"[{name}] cached -> {cache_path}")
        return emb, tgt, nA

    X_tr, y_tr, nA_tr = run_split("train", args.train_pt, args.max_train, cache_tr)
    X_te, y_te, nA_te = run_split("test",  args.test_pt,  args.max_test,  cache_te)

    print(f"\nTrain embeddings: {X_tr.shape}  Test embeddings: {X_te.shape}")
    rows = fit_and_report(X_tr, y_tr, nA_tr, X_te, y_te, nA_te)
    print_report(rows)

    report_path = cache_dir / "probe_report.json"
    with open(report_path, "w") as f:
        json.dump({"args": vars(args), "rows": rows}, f, indent=2)
    print(f"\nReport saved -> {report_path}")


if __name__ == "__main__":
    main()
