"""Screen bad thermo labels out of property_table.parquet.

Two filters, both conservative:

  L1. Physical impossibilities   — entropy_gas < 0  (third law) or
                                   cv_gas < 0  (impossible).
      TCIT occasionally fails on ring + tautomer systems and emits
      negative S° / Cv. Always drop.

  L2. 6·MAD statistical outliers — |x - median| > 6 · MAD · 1.4826
      per thermo field. Using MAD instead of σ because the σ itself is
      already inflated by the outliers we're trying to remove.

Affected rows keep their geometry + RDKit descriptors (still usable by
the denoising objective and the RDKit head). Only the 5 thermo fields
are NaN'd out and `has_thermo_label` is flipped to False.

Run once on a built parquet; build_property_table.py doesn't need to
re-run.

Usage:
  # inspect what would happen (no writes):
  python data_processing/screen_thermo_outliers.py \\
      --parquet data/property_table.parquet --dry-run

  # actually filter in place:
  python data_processing/screen_thermo_outliers.py \\
      --parquet data/property_table.parquet
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


THERMO = ["enthalpy_0", "enthalpy_298", "gibbs_298", "cv_gas", "entropy_gas"]


def _mad_bounds(x, k=6.0):
    """Median ± k · MAD · 1.4826 (σ-equivalent for Gaussian data)."""
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    sigma_equiv = 1.4826 * mad
    return median - k * sigma_equiv, median + k * sigma_equiv, median, sigma_equiv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--output", default=None,
                   help="Output parquet (default: overwrite input).")
    p.add_argument("--mad-k", type=float, default=6.0,
                   help="Threshold in MAD units (default 6).")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute & report without writing the output.")
    p.add_argument("--no-backup", action="store_true",
                   help="Skip the .bak copy when overwriting in-place.")
    args = p.parse_args()

    in_path = Path(args.parquet)
    out_path = Path(args.output) if args.output else in_path
    print(f"Reading {in_path}")
    df = pd.read_parquet(in_path)
    n = len(df)
    print(f"  {n:,} rows  |  has_thermo_label: {df.has_thermo_label.sum():,}")

    labeled = df.has_thermo_label.values.astype(bool)
    bad = np.zeros(n, dtype=bool)

    # ---- L1: physical impossibilities ----------------------------------
    phys_S  = labeled & (df.entropy_gas.values < 0)
    phys_Cv = labeled & (df.cv_gas.values      < 0)
    print(f"\n[L1] physical impossibilities")
    print(f"     entropy_gas < 0 : {int(phys_S.sum()):,}")
    print(f"     cv_gas      < 0 : {int(phys_Cv.sum()):,}")
    bad |= phys_S | phys_Cv

    # ---- L2: 6·MAD per thermo field ------------------------------------
    # Compute MAD on the labeled subset AFTER removing L1 violations so
    # the physically-impossible tail doesn't distort the scale estimate.
    clean = labeled & ~bad
    print(f"\n[L2] {args.mad_k}·MAD outliers  (per field, on {int(clean.sum()):,} L1-clean rows)")
    for f in THERMO:
        vals = df.loc[clean, f].dropna().values
        lo, hi, med, sigma = _mad_bounds(vals, k=args.mad_k)
        col = df[f].values
        mask = labeled & ~bad & ((col < lo) | (col > hi))
        print(f"     {f:<14s} bounds=[{lo:>10.2f}, {hi:>10.2f}]  "
              f"median={med:>8.2f}  σ≈MAD={sigma:>7.2f}   outliers={int(mask.sum()):>5,}")
        bad |= mask

    # ---- Aggregate report ----------------------------------------------
    n_drop = int(bad.sum())
    print(f"\n=========== SUMMARY ===========")
    print(f"  labeled before         : {int(labeled.sum()):,}")
    print(f"  labels removed         : {n_drop:,}   ({100*n_drop/max(labeled.sum(),1):.3f}%)")
    print(f"  labeled after          : {int(labeled.sum() - n_drop):,}")
    print(f"  rows in table          : {n:,} (unchanged — geometry + RDKit kept)")

    if args.dry_run:
        print("\n[dry-run] no file written")
        return

    # ---- Apply filter & write ------------------------------------------
    df2 = df.copy()
    df2.loc[bad, THERMO] = np.nan
    df2.loc[bad, "has_thermo_label"] = False

    # Safety: backup before in-place overwrite.
    if out_path == in_path and not args.no_backup:
        bak = in_path.with_suffix(in_path.suffix + ".bak")
        shutil.copy2(in_path, bak)
        print(f"  backup saved to        : {bak}")

    df2.to_parquet(out_path, index=False)
    print(f"  written to             : {out_path}  "
          f"({out_path.stat().st_size / 1024**2:.1f} MB)")

    # ---- Post-filter stats to paste into YAML --------------------------
    print("\n=========== NEW THERMO STATS (post-filter) ===========")
    labeled_new = df2[df2.has_thermo_label]
    print(f"  n = {len(labeled_new):,}")
    means, stds = [], []
    print(f"\n{'field':<14s} {'mean':>10s} {'std':>10s}  {'min':>10s} {'max':>10s}")
    for f in THERMO:
        x = labeled_new[f].dropna().values
        m, s = float(x.mean()), float(x.std())
        means.append(m); stds.append(s)
        print(f"{f:<14s} {m:>10.4f} {s:>10.4f}  {x.min():>10.2f} {x.max():>10.2f}")

    print(f"\n# paste into scripts/conf/loqi/*.yaml (thermo field order = [H298, G298, Cv, S°, H0]):")
    order = ["enthalpy_298", "gibbs_298", "cv_gas", "entropy_gas", "enthalpy_0"]
    ord_idx = [THERMO.index(f) for f in order]
    om = [round(means[i], 2) for i in ord_idx]
    os_ = [round(stds[i],  2) for i in ord_idx]
    print(f"  target_mean: {om}")
    print(f"  target_std:  {os_}")


if __name__ == "__main__":
    main()
