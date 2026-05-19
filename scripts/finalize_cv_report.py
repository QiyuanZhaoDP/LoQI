#!/usr/bin/env python3
"""Aggregate per-fold caches (fold_cache/fold_*.json) into a cv_report.json.

Used after running CV in pool mode where each fold was trained as an
independent job via `downstream_cv.py --only-fold N`. Reconstructs the
cv_report.json that downstream_cv.py would have written had it run all
5 folds sequentially.

Idempotent: safe to run multiple times; recomputes from current fold_cache
contents.

Usage:
    # finalize one dataset's CV output
    python scripts/finalize_cv_report.py outputs/cv_XXX/<ds>_<cfg>/

    # finalize all dataset outputs under a runs root
    python scripts/finalize_cv_report.py outputs/cv_XXX/*/

    # require all N folds present, fail otherwise (default: warn-and-skip)
    python scripts/finalize_cv_report.py --strict --n-folds 5 outputs/cv_XXX/*/
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def finalize(out_dir: Path, n_folds_expected: int = 5,
             strict: bool = False) -> dict | None:
    """Read fold_cache/fold_*.json, return aggregated summary dict and
    write cv_report.json. Returns None if folds incomplete and not strict.
    """
    fold_cache_dir = out_dir / "fold_cache"
    if not fold_cache_dir.is_dir():
        print(f"  [{out_dir.name}] no fold_cache/ — skip")
        return None

    fold_paths = sorted(fold_cache_dir.glob("fold_*.json"))
    folds: list[dict] = []
    for fp in fold_paths:
        with open(fp) as fh:
            try:
                folds.append(json.load(fh))
            except json.JSONDecodeError as e:
                print(f"  [{out_dir.name}] FAIL to parse {fp.name}: {e}")
                if strict:
                    sys.exit(2)
                return None

    if len(folds) < n_folds_expected:
        msg = (f"  [{out_dir.name}] only {len(folds)}/{n_folds_expected} folds "
               f"present (fold_paths: {[p.name for p in fold_paths]})")
        if strict:
            print(msg + " — STRICT mode, abort")
            sys.exit(2)
        print(msg + " — skip (not all folds done yet)")
        return None

    # Sanity check: fold indices should be 0..n_folds_expected-1
    fold_indices = sorted([f.get("fold", i) for i, f in enumerate(folds)])
    if fold_indices != list(range(n_folds_expected)):
        print(f"  [{out_dir.name}] WARN: fold indices = {fold_indices}, "
              f"expected {list(range(n_folds_expected))}")

    summary = {
        "mae_mean":  float(np.mean([r["mae"]  for r in folds])),
        "mae_std":   float(np.std ([r["mae"]  for r in folds])),
        "rmse_mean": float(np.mean([r["rmse"] for r in folds])),
        "r2_mean":   float(np.mean([r["r2"]   for r in folds])),
        "folds":     folds,
        "n_folds":   n_folds_expected,
    }
    # Optional fields (only present if all folds report them)
    optional_means = [
        ("ensemble_pred_std_mean",        "ensemble_pred_std_mean_avg"),
        ("ensemble_pred_std_over_target_std", "ensemble_pred_std_over_target_std_avg"),
        ("mae_per_conformer",             "mae_per_conformer_mean"),
        ("rmse_per_conformer",            "rmse_per_conformer_mean"),
        ("r2_per_conformer",              "r2_per_conformer_mean"),
        ("mae_last_stable",               "mae_last_stable_mean"),
        ("rmse_last_stable",              "rmse_last_stable_mean"),
        ("r2_last_stable",                "r2_last_stable_mean"),
    ]
    optional_stds = [
        ("mae_per_conformer", "mae_per_conformer_std"),
        ("mae_last_stable",   "mae_last_stable_std"),
    ]
    for src_key, out_key in optional_means:
        if all(src_key in r for r in folds):
            summary[out_key] = float(np.mean([r[src_key] for r in folds]))
    for src_key, out_key in optional_stds:
        if all(src_key in r for r in folds):
            summary[out_key] = float(np.std([r[src_key] for r in folds]))

    # Inherit per-fold "args" (split_dir, lr, ckpt, etc.) from fold 0 if present
    if folds and "args" in folds[0]:
        summary["args"] = folds[0]["args"]
    if folds and "n_molecules" in folds[0]:
        summary["n_molecules"] = folds[0]["n_molecules"]
    if folds and "n_labeled" in folds[0]:
        summary["n_labeled"] = folds[0]["n_labeled"]

    out_path = out_dir / "cv_report.json"
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"  [{out_dir.name}] wrote cv_report.json  "
          f"MAE_mean={summary['mae_mean']:.4f}  R²_mean={summary['r2_mean']:.4f}  "
          f"n_folds={n_folds_expected}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_dirs", nargs="+",
                   help="One or more per-dataset CV out_dirs containing fold_cache/")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--strict", action="store_true",
                   help="Abort if any out_dir lacks all n_folds folds.")
    args = p.parse_args()

    done = 0
    skipped = 0
    for d in args.out_dirs:
        d_path = Path(d)
        if not d_path.is_dir():
            print(f"  [{d}] not a directory — skip")
            skipped += 1
            continue
        res = finalize(d_path, n_folds_expected=args.n_folds, strict=args.strict)
        if res is None:
            skipped += 1
        else:
            done += 1
    print(f"\nFinalized {done}, skipped {skipped} of {len(args.out_dirs)} dirs.")


if __name__ == "__main__":
    main()
