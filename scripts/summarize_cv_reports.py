"""Print the FINAL SUMMARY table for any directory of cv_report.json files,
without re-running any training/extract/sampling stages.

Use this when you've completed a CV sweep and want to regenerate the
summary block — e.g., after updating the printer layout (this script
is the canonical implementation; run_cv.sh's inline summary mirrors it).

Usage:
    python scripts/summarize_cv_reports.py outputs/cv_0515_warm
    python scripts/summarize_cv_reports.py outputs/cv_0515_warm outputs/cv_0515_cold
    python scripts/summarize_cv_reports.py outputs/cv_0515_*       # shell glob ok

Output: one row per cv_report.json under each given root, with
        best-val (MAE, RMSE, R²) and last-stable (MAE, RMSE, R²)
        side by side, plus mean best-epoch across folds.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path


def collect(root: str):
    rows = []
    for rep in sorted(glob.glob(os.path.join(root, "*", "cv_report.json"))):
        suffix = Path(rep).parent.name
        try:
            d = json.load(open(rep))
        except Exception:
            rows.append((suffix, math.nan, math.nan, math.nan,
                          math.nan, math.nan, math.nan, 0))
            continue
        ep = (sum(f.get("best_epoch", 0) for f in d.get("folds", []))
              / max(len(d.get("folds", [])), 1))
        rows.append((suffix,
                     d.get("mae_mean", math.nan),
                     d.get("rmse_mean", math.nan),
                     d.get("r2_mean", math.nan),
                     d.get("mae_last_stable_mean", math.nan),
                     d.get("rmse_last_stable_mean", math.nan),
                     d.get("r2_last_stable_mean", math.nan),
                     ep))
    return rows


def print_table(rows, header: str = ""):
    if not rows:
        print("No cv_report.json found.")
        return
    if header:
        print(f"\n{header}")
        print("=" * len(header))
    print(f"\n{'suffix':<36s}  {'bv_MAE':>10s} {'bv_RMSE':>10s} {'bv_R²':>7s}   "
          f"{'ls_MAE':>10s} {'ls_RMSE':>10s} {'ls_R²':>7s}  {'ep':>4s}")
    print("-" * 112)
    for suffix, mae_bv, rmse_bv, r2_bv, mae_ls, rmse_ls, r2_ls, ep in rows:
        print(f"{suffix:<36s}  {mae_bv:>10.4f} {rmse_bv:>10.4f} {r2_bv:>7.3f}   "
              f"{mae_ls:>10.4f} {rmse_ls:>10.4f} {r2_ls:>7.3f}  {ep:>4.0f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="+",
                   help="One or more output-root dirs containing per-task "
                        "subdirs with cv_report.json (e.g. outputs/cv_0515_warm).")
    p.add_argument("--csv", default=None,
                   help="Also write the combined table as a CSV to this path.")
    args = p.parse_args()

    all_rows = []
    for root in args.roots:
        if not Path(root).is_dir():
            print(f"WARN: {root} is not a directory", file=sys.stderr)
            continue
        rows = collect(root)
        print_table(rows, header=root)
        all_rows.extend([(root, *r) for r in rows])

    if args.csv and all_rows:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["out_root", "suffix",
                        "bv_MAE", "bv_RMSE", "bv_R2",
                        "ls_MAE", "ls_RMSE", "ls_R2",
                        "best_epoch_mean"])
            for r in all_rows:
                w.writerow(r)
        print(f"\nWrote {len(all_rows)} rows → {args.csv}")


if __name__ == "__main__":
    main()
