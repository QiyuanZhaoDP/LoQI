"""Morgan FP + Random Forest 5-fold CV baseline for downstream datasets.

A vanilla 2D-only baseline. For each cleaned downstream CSV (under
`downstream_ft/clean/`):
  1. parse SMILES, drop unparseable / NaN-target rows
  2. compute Morgan fingerprints (small grid over radius)
  3. run 5-fold CV (KFold seed=42, same as downstream_cv.py) for each
     hyperparameter combination
  4. pick the combo with the lowest mean val MAE
  5. report that combo's per-fold + aggregated metrics

Output files match the cv_report.json shape produced by downstream_cv.py
where it makes sense (mae_mean, mae_std, rmse_mean, r2_mean, folds[]),
so the baseline numbers slot directly into the same comparison tables.

Reference baseline only — RF on 2D fingerprints can't see 3D conformers,
so the gap to the LoQI/ThermoGen FT results is the value-add of 3D
representation.

Default grid: radius ∈ {2,3} × n_estimators ∈ {200,500} = 4 combos.
9 datasets × 4 combos × 5 folds × ~1s/RF ≈ a couple of minutes total.

Usage:
    python scripts/morgan_rf_baseline.py
    python scripts/morgan_rf_baseline.py --datasets gas_Hf,Cp
    python scripts/morgan_rf_baseline.py --radii 2 --n-estimators 200,500,1000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")

# (smiles_col, target_col) per dataset. smiles col is case-sensitive.
DATASET_COLS: dict[str, tuple[str, str]] = {
    "Cp":         ("SMILES", "TARGET"),
    "V_cp":       ("SMILES", "TARGET"),
    "de":         ("SMILES", "TARGET"),
    "gas_Hf":     ("smiles", "mean"),
    "k":          ("SMILES", "TARGET"),
    "liquid_Hf":  ("smiles", "mean"),
    "delaney_s":  ("SMILES", "TARGET"),
    "freesolv_s": ("SMILES", "TARGET"),
    "lipo_s":     ("SMILES", "TARGET"),
}


def parse_and_fingerprint(df: pd.DataFrame, smi_col: str, tgt_col: str,
                          radii: list[int], n_bits: int):
    """Return (X_by_radius: dict[r, np.ndarray], y: np.ndarray, n_dropped: int)."""
    fps_by_radius: dict[int, list[np.ndarray]] = {r: [] for r in radii}
    ys: list[float] = []
    n_drop_smi = n_drop_tgt = 0

    for smi, y_raw in zip(df[smi_col].astype(str), df[tgt_col]):
        if pd.isna(y_raw):
            n_drop_tgt += 1
            continue
        mol = Chem.MolFromSmiles(str(smi).strip())
        if mol is None:
            n_drop_smi += 1
            continue
        for r in radii:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, r, nBits=n_bits)
            arr = np.zeros((n_bits,), dtype=np.uint8)
            ConvertToNumpyArray(fp, arr)
            fps_by_radius[r].append(arr)
        ys.append(float(y_raw))

    X_by_radius = {r: np.stack(v) for r, v in fps_by_radius.items()}
    y = np.asarray(ys, dtype=np.float64)
    return X_by_radius, y, n_drop_smi + n_drop_tgt


def cv_fit(X: np.ndarray, y: np.ndarray, n_folds: int, seed: int,
           **rf_kwargs) -> dict:
    """Run KFold CV with the given RF hyperparameters."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for fold_i, (tr, va) in enumerate(kf.split(X)):
        rf = RandomForestRegressor(n_jobs=-1, random_state=seed, **rf_kwargs)
        rf.fit(X[tr], y[tr])
        pred = rf.predict(X[va])
        folds.append({
            "fold":    fold_i,
            "n_train": int(len(tr)),
            "n_val":   int(len(va)),
            "mae":  float(mean_absolute_error(y[va], pred)),
            "rmse": float(np.sqrt(mean_squared_error(y[va], pred))),
            "r2":   float(r2_score(y[va], pred)),
        })
    return {
        "folds":    folds,
        "mae_mean":  float(np.mean([f["mae"]  for f in folds])),
        "mae_std":   float(np.std ([f["mae"]  for f in folds])),
        "rmse_mean": float(np.mean([f["rmse"] for f in folds])),
        "r2_mean":   float(np.mean([f["r2"]   for f in folds])),
    }


def run_one_dataset(name: str, root: Path, out_dir: Path,
                    radii: list[int], n_estimators_grid: list[int],
                    n_bits: int, n_folds: int, seed: int) -> dict:
    smi_col, tgt_col = DATASET_COLS[name]
    csv = root / f"{name}.csv"
    df = pd.read_csv(csv)
    n_raw = len(df)

    t0 = time.time()
    X_by_radius, y, n_dropped = parse_and_fingerprint(
        df, smi_col, tgt_col, radii, n_bits,
    )
    n = len(y)
    target_std = float(np.std(y)) if n > 1 else 1.0
    print(f"  [{name}] {n_raw:,} raw → {n:,} fingerprinted "
          f"({n_dropped} dropped). target σ = {target_std:.4f}. "
          f"FP gen {time.time()-t0:.1f}s.")

    # Hyperparameter grid: radius × n_estimators
    all_configs = []
    for radius in radii:
        for n_est in n_estimators_grid:
            t1 = time.time()
            metrics = cv_fit(
                X_by_radius[radius], y, n_folds=n_folds, seed=seed,
                n_estimators=n_est, max_features="sqrt", min_samples_leaf=1,
                max_depth=None,
            )
            metrics["config"] = {
                "radius": radius, "n_estimators": n_est,
                "n_bits": n_bits, "max_features": "sqrt",
            }
            metrics["wall_seconds"] = round(time.time() - t1, 1)
            print(f"    r={radius}  n_est={n_est:>4d}  "
                  f"MAE={metrics['mae_mean']:.4f}±{metrics['mae_std']:.4f}  "
                  f"R²={metrics['r2_mean']:.3f}  "
                  f"({metrics['wall_seconds']:.1f}s)")
            all_configs.append(metrics)

    # Pick best by mean MAE
    best = min(all_configs, key=lambda c: c["mae_mean"])

    summary = {
        "name":           name,
        "n_raw":          n_raw,
        "n_used":         n,
        "n_dropped":      n_dropped,
        "target_std":     target_std,
        "n_folds":        n_folds,
        "seed":           seed,
        "best_config":    best["config"],
        "best_metrics":   {k: best[k] for k in
                            ["folds", "mae_mean", "mae_std",
                             "rmse_mean", "r2_mean"]},
        "all_configs":    all_configs,
        "input_csv":      str(csv),
    }

    out_path = out_dir / f"{name}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    return summary


def write_summary_md(results: list[dict], path: Path) -> None:
    lines = []
    lines.append("# Morgan FP + Random Forest 5-fold CV baseline")
    lines.append("")
    lines.append("Reference 2D-only baseline. RF on Morgan FPs (no 3D info)")
    lines.append("with small radius × n_estimators grid; reports the best")
    lines.append("config per dataset (selected on mean val MAE).")
    lines.append("")
    lines.append(f"5-fold CV, seed=42, sklearn RandomForestRegressor "
                 "(max_features='sqrt', min_samples_leaf=1).")
    lines.append("")
    lines.append("| dataset | n | target σ | best config | MAE | RMSE | R² |")
    lines.append("|---|---:|---:|---|---:|---:|---:|")
    for r in results:
        cfg = r["best_config"]
        m = r["best_metrics"]
        lines.append(
            f"| {r['name']} | {r['n_used']:,} | {r['target_std']:.3f} | "
            f"r={cfg['radius']}, n={cfg['n_estimators']} | "
            f"**{m['mae_mean']:.3f}**±{m['mae_std']:.3f} | "
            f"{m['rmse_mean']:.3f} | "
            f"{m['r2_mean']:.3f} |"
        )
    lines.append("")
    lines.append(
        "MAE / RMSE are in target physical units. R² ≥ 0.7 is decent for "
        "a 2D-only baseline; gas_Hf and similar quantum-mechanically-"
        "demanding properties typically need 3D info to push R² higher."
    )
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="downstream_ft/clean",
                   help="Where to read cleaned CSVs from.")
    p.add_argument("--out-dir", default="outputs/baselines/morgan_rf",
                   help="Where to write per-dataset JSONs + SUMMARY.md.")
    p.add_argument("--datasets", default=None,
                   help="Comma-separated subset of dataset names. Default: all 9.")
    p.add_argument("--radii", default="2,3",
                   help="Comma-separated Morgan radii to try.")
    p.add_argument("--n-estimators", default="200,500",
                   help="Comma-separated n_estimators values to try.")
    p.add_argument("--n-bits", type=int, default=2048,
                   help="Morgan FP bit length.")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42,
                   help="KFold seed (matches downstream_cv.py).")
    args = p.parse_args()

    root = Path(args.input_dir)
    if not root.exists():
        raise SystemExit(f"input dir not found: {root}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.datasets:
        names = [s.strip() for s in args.datasets.split(",") if s.strip()]
    else:
        names = list(DATASET_COLS.keys())
    radii = [int(x) for x in args.radii.split(",")]
    n_estimators_grid = [int(x) for x in args.n_estimators.split(",")]

    print("=" * 70)
    print(f"Morgan FP + RF baseline   |   input: {root}   →   out: {out_dir}")
    print(f"  radii: {radii}   n_estimators: {n_estimators_grid}   "
          f"n_bits: {args.n_bits}")
    print(f"  datasets: {names}")
    print("=" * 70)

    results = []
    for name in names:
        print(f"\n--- {name} ---")
        try:
            r = run_one_dataset(name, root, out_dir, radii, n_estimators_grid,
                                args.n_bits, args.n_folds, args.seed)
            results.append(r)
        except Exception as e:
            print(f"  [{name}] FAILED: {e}")

    if results:
        md_path = out_dir / "SUMMARY.md"
        write_summary_md(results, md_path)

        print("\n" + "=" * 70)
        print(f"{'dataset':<14s}  {'n':>6s}  {'best_cfg':<14s}  "
              f"{'MAE':>10s}  {'R²':>6s}")
        print("-" * 70)
        for r in results:
            cfg = r["best_config"]
            m = r["best_metrics"]
            cfg_str = f"r={cfg['radius']}, n={cfg['n_estimators']}"
            print(f"{r['name']:<14s}  {r['n_used']:>6,}  {cfg_str:<14s}  "
                  f"{m['mae_mean']:>7.3f}±{m['mae_std']:<2.2f}  "
                  f"{m['r2_mean']:>6.3f}")
        print("=" * 70)
        print(f"Per-dataset JSONs: {out_dir}/<name>.json")
        print(f"Markdown summary:  {md_path}")


if __name__ == "__main__":
    main()
