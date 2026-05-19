#!/usr/bin/env python3
"""Aggregate CV results across datasets — report BOTH averaged and pooled metrics.

For each per-dataset CV output directory, this script computes two parallel
metric tracks:

  Averaged (current default in cv_report.json)
    avg_mae   = mean across K folds of per-fold MAE
    avg_rmse  = mean across K folds of per-fold RMSE
    avg_r2    = mean across K folds of per-fold R² (each fold uses its
                own per-fold target mean as R² reference)
    avg_X_std = cross-fold std of metric X

  Pooled (the recommended primary metric)
    pooled_mae = mean_i(|y_i - ŷ_i|)  over ALL N test points concatenated
    pooled_rmse = sqrt(mean_i((y_i-ŷ_i)²)) over the same
    pooled_r2  = 1 - SSE_total / TSS_total, where TSS uses the GLOBAL mean
                 of y across all K test folds (= one R² reference for the
                 entire dataset, not per-fold).

Pooled vs Averaged can differ because:
  (1) Fold size imbalance — pooled is size-weighted, averaged is fold-weighted.
  (2) Per-fold target distributions differ — averaged R² uses K different
      means/stds; pooled R² uses one global mean/std.

When per-sample preds (preds_fold{1..5}.csv from DUMP_PREDS=1) are
available we compute true pooled metrics. When they're not, we fall
back to size-weighted MAE/RMSE (a safe approximation of pooled MAE/RMSE)
and leave pooled R² blank (no good approximation exists from summary
stats alone).

Usage:
    python scripts/aggregate_benchmark.py \\
        --runs-root outputs/cv_0518_cold_balanced \\
        --out /tmp/cv_0518_summary.csv

    # Aggregate multiple runs side-by-side into a wide table
    python scripts/aggregate_benchmark.py \\
        --runs-root outputs/cv_0515_cold outputs/cv_0516_cold outputs/cv_0518_cold_balanced \\
        --out /tmp/multi_run_summary.csv
"""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def _safe_mean_std(xs):
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def _read_preds_fold(fold_csv: Path):
    """Read preds_fold{N}.csv -> (y_true_list, y_pred_best_list, y_pred_ls_list)."""
    y_true, y_best, y_ls = [], [], []
    with open(fold_csv) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                # has_target=1 means this test point has a real label
                if int(float(row.get("has_target", "1"))) != 1:
                    continue
                y_true.append(float(row["y_true"]))
                y_best.append(float(row["y_pred_best"]))
                y_ls.append(float(row["y_pred_last_stable"]))
            except (ValueError, KeyError):
                continue
    return y_true, y_best, y_ls


def _pooled_metrics(y_true, y_pred):
    """Compute pooled MAE, RMSE, R² over a single concatenated array."""
    if not y_true:
        return {"mae": None, "rmse": None, "r2": None, "n": 0}
    n = len(y_true)
    err = [yp - yt for yp, yt in zip(y_pred, y_true)]
    mae = sum(abs(e) for e in err) / n
    sse = sum(e * e for e in err)
    rmse = math.sqrt(sse / n)
    mean_y = sum(y_true) / n
    tss = sum((y - mean_y) ** 2 for y in y_true)
    r2 = 1.0 - sse / tss if tss > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "n": n}


def _aggregate_run_for_dataset(ds_dir: Path):
    """Read cv_report.json + (optional) preds_fold*.csv from one dataset dir.
    Returns dict with averaged + pooled metrics (and 'src' flag indicating
    whether pooled was computed from raw preds or via size-weighted fallback)."""
    cv_report = ds_dir / "cv_report.json"
    if not cv_report.exists():
        return None
    with open(cv_report) as fh:
        rep = json.load(fh)

    # ---- 1. Averaged track from cv_report.json folds ----
    folds = rep.get("folds", [])
    fold_mae   = [f["mae"]  for f in folds if "mae"  in f]
    fold_rmse  = [f["rmse"] for f in folds if "rmse" in f]
    fold_r2    = [f["r2"]   for f in folds if "r2"   in f]
    fold_n     = [f.get("n_val", f.get("n_test", 0)) for f in folds]

    # last-stable variants if present
    fold_mae_ls  = [f["mae_last_stable"]  for f in folds if "mae_last_stable"  in f]
    fold_rmse_ls = [f["rmse_last_stable"] for f in folds if "rmse_last_stable" in f]
    fold_r2_ls   = [f["r2_last_stable"]   for f in folds if "r2_last_stable"   in f]

    avg_mae, avg_mae_std = _safe_mean_std(fold_mae)
    avg_rmse, _          = _safe_mean_std(fold_rmse)
    avg_r2, avg_r2_std   = _safe_mean_std(fold_r2)
    avg_mae_ls, _        = _safe_mean_std(fold_mae_ls)
    avg_rmse_ls, _       = _safe_mean_std(fold_rmse_ls)
    avg_r2_ls, _         = _safe_mean_std(fold_r2_ls)

    # ---- 2. Pooled track ----
    n_folds = len(folds) or rep.get("n_folds", 0)
    pred_files = sorted([ds_dir / f"preds_fold{k}.csv"
                         for k in range(1, n_folds + 1) if (ds_dir / f"preds_fold{k}.csv").exists()])

    pooled_best = pooled_ls = None
    pooled_src = "weighted-fallback"

    if pred_files:
        all_yt, all_yp_best, all_yp_ls = [], [], []
        for fp in pred_files:
            yt, yb, yls = _read_preds_fold(fp)
            all_yt += yt; all_yp_best += yb; all_yp_ls += yls
        pooled_best = _pooled_metrics(all_yt, all_yp_best)
        pooled_ls   = _pooled_metrics(all_yt, all_yp_ls)
        pooled_src  = "preds-concat"

    # Fallback: size-weighted MAE / RMSE (no pooled R² approximation)
    if pooled_best is None and fold_mae and fold_n:
        total_n = sum(fold_n)
        if total_n > 0:
            pooled_best = {
                "mae":  sum(n * m for n, m in zip(fold_n, fold_mae))  / total_n,
                "rmse": math.sqrt(sum(n * (r ** 2) for n, r in zip(fold_n, fold_rmse)) / total_n),
                "r2":   None,
                "n":    total_n,
            }
            if fold_mae_ls and fold_rmse_ls:
                pooled_ls = {
                    "mae":  sum(n * m for n, m in zip(fold_n, fold_mae_ls))  / total_n,
                    "rmse": math.sqrt(sum(n * (r ** 2) for n, r in zip(fold_n, fold_rmse_ls)) / total_n),
                    "r2":   None,
                    "n":    total_n,
                }

    return {
        "n_folds": n_folds,
        "n_total": sum(fold_n) if fold_n else None,
        "fold_sizes": fold_n,
        # averaged
        "avg_mae":  avg_mae, "avg_mae_std": avg_mae_std,
        "avg_rmse": avg_rmse,
        "avg_r2":   avg_r2,  "avg_r2_std": avg_r2_std,
        "avg_mae_ls":  avg_mae_ls,
        "avg_rmse_ls": avg_rmse_ls,
        "avg_r2_ls":   avg_r2_ls,
        # pooled
        "pooled_mae":   pooled_best.get("mae")  if pooled_best else None,
        "pooled_rmse":  pooled_best.get("rmse") if pooled_best else None,
        "pooled_r2":    pooled_best.get("r2")   if pooled_best else None,
        "pooled_mae_ls":  pooled_ls.get("mae")  if pooled_ls else None,
        "pooled_rmse_ls": pooled_ls.get("rmse") if pooled_ls else None,
        "pooled_r2_ls":   pooled_ls.get("r2")   if pooled_ls else None,
        "pooled_src": pooled_src,
        "pooled_n":   pooled_best.get("n") if pooled_best else 0,
    }


def _fmt(x, w=8, prec=4):
    if x is None:
        return "    n/a"
    if isinstance(x, str):
        return f"{x:>{w}}"
    if abs(x) >= 1000:
        return f"{x:>{w}.1f}"
    if abs(x) < 0.001 and x != 0:
        return f"{x:>{w}.3e}"
    return f"{x:>{w}.{prec}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", type=str, nargs="+", required=True,
                   help="One or more CV run roots (e.g. outputs/cv_0518_cold_balanced). "
                        "Each contains <dataset>_<cfg>/cv_report.json subdirs.")
    p.add_argument("--out", type=str, default=None,
                   help="Optional output CSV. If omitted, prints to stdout.")
    args = p.parse_args()

    rows = []
    for runs_root_str in args.runs_root:
        runs_root = Path(runs_root_str)
        if not runs_root.exists():
            print(f"  WARN: {runs_root} does not exist, skipping")
            continue
        run_name = runs_root.name
        for ds_dir in sorted(runs_root.iterdir()):
            if not ds_dir.is_dir():
                continue
            cv_report = ds_dir / "cv_report.json"
            if not cv_report.exists():
                continue
            stats = _aggregate_run_for_dataset(ds_dir)
            if stats is None:
                continue
            stats["run_name"] = run_name
            stats["dataset"]  = ds_dir.name
            rows.append(stats)

    if not rows:
        print("No cv_report.json found under any --runs-root.")
        return

    # ---- Print summary table ----
    print()
    print("=" * 175)
    print(f"{'Run':<28} {'Dataset':<30} {'n':>4} {'n_tot':>6} "
          f"| {'avg_MAE':>9} {'±std':>7} {'pool_MAE':>9} "
          f"| {'avg_RMSE':>9} {'pool_RMSE':>9} "
          f"| {'avg_R²':>7} {'±std':>7} {'pool_R²':>8} "
          f"| {'src':>13}")
    print("-" * 175)
    for r in rows:
        print(f"{r['run_name']:<28} {r['dataset']:<30} {r['n_folds']:>4} {r['n_total'] or 0:>6} "
              f"| {_fmt(r['avg_mae'], 9)} {_fmt(r['avg_mae_std'], 7, 3)} {_fmt(r['pooled_mae'], 9)} "
              f"| {_fmt(r['avg_rmse'], 9)} {_fmt(r['pooled_rmse'], 9)} "
              f"| {_fmt(r['avg_r2'], 7, 4)} {_fmt(r['avg_r2_std'], 7, 4)} {_fmt(r['pooled_r2'], 8, 4)} "
              f"| {r['pooled_src']:>13}")

    # ---- Write CSV ----
    if args.out:
        cols = [
            "run_name", "dataset", "n_folds", "n_total", "fold_sizes",
            "avg_mae", "avg_mae_std", "avg_rmse", "avg_r2", "avg_r2_std",
            "avg_mae_ls", "avg_rmse_ls", "avg_r2_ls",
            "pooled_mae", "pooled_rmse", "pooled_r2",
            "pooled_mae_ls", "pooled_rmse_ls", "pooled_r2_ls",
            "pooled_src", "pooled_n",
        ]
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for r in rows:
                w.writerow([r.get(c) for c in cols])
        print(f"\nWrote CSV -> {out_path}  ({len(rows)} rows)")

    # ---- Summary diagnostics: pooled vs averaged disagreement ----
    print()
    print("=" * 90)
    print("Pooled vs Averaged disagreement scan (MAE)")
    print("=" * 90)
    flagged = 0
    for r in rows:
        if r["avg_mae"] is None or r["pooled_mae"] is None:
            continue
        delta = abs(r["pooled_mae"] - r["avg_mae"])
        rel = delta / max(abs(r["avg_mae"]), 1e-9) * 100
        # Flag if >5% relative diff or if pooled_src is fallback
        if rel > 5.0 or r["pooled_src"] == "weighted-fallback":
            flag = "  FALLBACK" if r["pooled_src"] == "weighted-fallback" else f"  Δ={rel:.1f}%"
            print(f"  {r['run_name']:<28} {r['dataset']:<28} "
                  f"avg={r['avg_mae']:.4f}  pooled={r['pooled_mae']:.4f}{flag}")
            flagged += 1
    if not flagged:
        print("  (none — averaged and pooled agree within 5% for all rows)")
    else:
        print(f"\n  {flagged} row(s) flagged. Reasons:")
        print("    * Δ>5%  : fold-size imbalance OR per-fold target distributions differ")
        print("    * FALLBACK: preds_fold*.csv missing → set DUMP_PREDS=1 to get true pooled")


if __name__ == "__main__":
    main()
