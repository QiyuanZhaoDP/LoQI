"""Estimate per-conformer MAE/RMSE from cv_report.json files.

Our downstream_cv.py reports the ENSEMBLE metric: predictions are averaged
across K conformers per molecule before computing MAE/RMSE/R². For an
apples-to-apples comparison with UniMol (which uses single-conformer
prediction on val), we want the per-conformer error.

Algebraic identity (per molecule i, K conformers, target y_i):
    Σ_j (X_ij - y_i)² = Σ_j (X_ij - mean_j X_ij)² + K·(mean_j X_ij - y_i)²
    ⇒ mean_ij[(X_ij - y_i)²] = mean_i[var_j(X_ij)] + ensemble_RMSE²

So:
    per-conformer RMSE² ≈ ensemble_RMSE² + (ensemble_pred_std_mean)²
    per-conformer MAE   ≈ ensemble_MAE × (per-conf RMSE / ensemble RMSE)

The MAE step is an approximation (no closed form without the full
distribution), but it's accurate to ~1-2% when residuals are roughly
Gaussian, which they are once the head is reasonably converged.

Usage:
    python scripts/estimate_per_conformer_mae.py \\
        outputs/downstream_cv_K8/*_cold_large/cv_report.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np


def estimate(rep_path: str) -> dict:
    with open(rep_path) as f:
        d = json.load(f)
    folds = d.get("folds", [])
    if not folds:
        return {"name": os.path.basename(os.path.dirname(rep_path)),
                "error": "no folds"}
    rows = []
    for f in folds:
        e_mae = f.get("mae")
        e_rmse = f.get("rmse")
        sigma = f.get("ensemble_pred_std_mean", 0.0) or 0.0
        if e_mae is None or e_rmse is None:
            continue
        pc_rmse = float(np.sqrt(e_rmse ** 2 + sigma ** 2))
        ratio = pc_rmse / max(e_rmse, 1e-12)
        pc_mae = e_mae * ratio
        rows.append({
            "fold": f.get("fold"),
            "ens_mae": e_mae, "ens_rmse": e_rmse,
            "pred_sigma": sigma,
            "pc_mae": pc_mae, "pc_rmse": pc_rmse,
        })
    if not rows:
        return {"name": os.path.basename(os.path.dirname(rep_path)),
                "error": "no usable folds"}

    return {
        "name": os.path.basename(os.path.dirname(rep_path)),
        "n_folds": len(rows),
        "ens_mae_mean":  float(np.mean([r["ens_mae"]  for r in rows])),
        "ens_mae_std":   float(np.std ([r["ens_mae"]  for r in rows])),
        "ens_rmse_mean": float(np.mean([r["ens_rmse"] for r in rows])),
        "pred_sigma_mean": float(np.mean([r["pred_sigma"] for r in rows])),
        "pc_mae_mean":   float(np.mean([r["pc_mae"]   for r in rows])),
        "pc_mae_std":    float(np.std ([r["pc_mae"]   for r in rows])),
        "pc_rmse_mean":  float(np.mean([r["pc_rmse"]  for r in rows])),
        "delta_mae":     float(np.mean([r["pc_mae"] - r["ens_mae"]
                                         for r in rows])),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+",
                   help="cv_report.json paths or globs")
    args = p.parse_args()

    files = []
    for x in args.paths:
        files.extend(glob.glob(x) if any(c in x for c in "*?[") else [x])
    files = sorted(set(files))
    if not files:
        print("No matching files.", file=sys.stderr)
        sys.exit(1)

    print(f"{'run':<32s}  {'ens_MAE':>10s}  {'pred_σ':>8s}   "
          f"{'PC_MAE':>10s}   {'Δ':>6s}   {'PC_RMSE':>10s}")
    print("-" * 92)
    for f in files:
        r = estimate(f)
        if "error" in r:
            print(f"{r['name']:<32s}  ({r['error']})")
            continue
        print(f"{r['name']:<32s}  "
              f"{r['ens_mae_mean']:>7.3f}±{r['ens_mae_std']:<2.2f}  "
              f"{r['pred_sigma_mean']:>8.3f}   "
              f"{r['pc_mae_mean']:>7.3f}±{r['pc_mae_std']:<2.2f}  "
              f"{r['delta_mae']:>+6.2f}   "
              f"{r['pc_rmse_mean']:>10.3f}")


if __name__ == "__main__":
    main()
