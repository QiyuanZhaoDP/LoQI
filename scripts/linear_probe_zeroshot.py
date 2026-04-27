"""Affine linear probe on top of the frozen thermo head's predictions.

Reads the JSON from predict_thermo_zeroshot.py (per-mol target + K-averaged
pred_mean), fits y = a*z + b via OLS in n-fold CV, and reports MAE/RMSE/R².

Diagnostic value:
  * If linear-probe MAE  ≈  full-FT MAE   → FT is mostly calibration. The
    foundation model's thermo head representation is already good; the gap
    to ground truth is a per-dataset affine shift.
  * If linear-probe MAE  ≈  zero-shot MAE → FT is genuinely reweighting
    features. K-conformer info matters; head architecture / training matter.

Usage:
  python scripts/linear_probe_zeroshot.py \\
      --zeroshot-json outputs/zeroshot/gas_Hf_H298.json \\
      --output        outputs/zeroshot/gas_Hf_H298.linprobe.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zeroshot-json", required=True,
                   help="Output of predict_thermo_zeroshot.py")
    p.add_argument("--output", required=True)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42,
                   help="Match downstream_cv.py's KFold seed for apples-to-apples")
    args = p.parse_args()

    with open(args.zeroshot_json) as f:
        d = json.load(f)
    rows = d["per_mol"]
    z = np.array([r["pred_mean"] for r in rows]).reshape(-1, 1)
    y = np.array([r["target"] for r in rows])
    print(f"loaded {len(rows):,} mols from {args.zeroshot_json}")
    print(f"  zero-shot MAE = {d['mae']:.2f}  R² = {d['r2']:.3f}")

    # ---- n-fold CV with affine fit ---------------------------------------
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_metrics = []
    for fold, (tr, va) in enumerate(kf.split(z)):
        reg = LinearRegression().fit(z[tr], y[tr])
        a = float(reg.coef_[0]); b = float(reg.intercept_)
        y_hat = reg.predict(z[va])
        mae = float(mean_absolute_error(y[va], y_hat))
        rmse = float(np.sqrt(((y_hat - y[va]) ** 2).mean()))
        r2 = float(r2_score(y[va], y_hat))
        fold_metrics.append({"fold": fold, "a": a, "b": b,
                             "mae": mae, "rmse": rmse, "r2": r2})
        print(f"  fold {fold}: a={a:.3f}  b={b:.3f}  "
              f"MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.3f}")

    mae_mean = float(np.mean([m["mae"] for m in fold_metrics]))
    mae_std  = float(np.std ([m["mae"] for m in fold_metrics]))
    rmse_mean = float(np.mean([m["rmse"] for m in fold_metrics]))
    r2_mean  = float(np.mean([m["r2"]  for m in fold_metrics]))

    # In-sample global fit for inspection
    reg_all = LinearRegression().fit(z, y)
    a_all = float(reg_all.coef_[0]); b_all = float(reg_all.intercept_)
    mae_all = float(mean_absolute_error(y, reg_all.predict(z)))

    print()
    print("=" * 60)
    print(f"  LINEAR PROBE  ({args.n_folds}-fold CV, seed={args.seed})")
    print("=" * 60)
    print(f"  MAE  = {mae_mean:.2f} ± {mae_std:.2f}")
    print(f"  RMSE = {rmse_mean:.2f}")
    print(f"  R²   = {r2_mean:.3f}")
    print(f"  global affine: y = {a_all:.3f}·z + {b_all:.3f}  "
          f"(in-sample MAE={mae_all:.2f})")
    print("=" * 60)
    print(f"  vs zero-shot:  MAE {d['mae']:.2f} → {mae_mean:.2f}  "
          f"(Δ = {d['mae'] - mae_mean:+.2f})")
    print("=" * 60)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "args": vars(args),
            "n_mols": len(rows),
            "zeroshot_mae": d["mae"],
            "zeroshot_rmse": d["rmse"],
            "zeroshot_r2": d["r2"],
            "linear_probe_mae_mean": mae_mean,
            "linear_probe_mae_std": mae_std,
            "linear_probe_rmse_mean": rmse_mean,
            "linear_probe_r2_mean": r2_mean,
            "global_affine": {"a": a_all, "b": b_all, "in_sample_mae": mae_all},
            "folds": fold_metrics,
        }, f, indent=2)
    print(f"\nReport -> {out}")


if __name__ == "__main__":
    main()
