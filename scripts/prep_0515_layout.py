"""One-shot setup: makes the 0515_final layout look like 0511_cc_audit
so run_cv.sh's existing INPUT_DIR + SPLIT_DIR_ROOT machinery works
without any pipeline changes.

Source layout:
  downstream_ft/0515_final/
    per_property/<prop>.csv         # inchikey,smiles,value,tier,scaffold
    csv_data/<prop>/Split/random_cv5/cv{1-5}_{train,valid,test}.csv  # SMILES,TARGET

Target layout (what run_cv.sh expects):
  downstream_ft/0515_final/
    Clean/<prop>.csv                # SMILES,TARGET (renamed from per_property)
    Split/<prop>/random_cv5/        # symlink → csv_data/<prop>/Split/random_cv5

Idempotent: re-running won't clobber Clean csvs unless their input changed.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="downstream_ft/0515_final",
                   help="0515_final directory containing per_property/ and csv_data/")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would happen without writing anything.")
    args = p.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"ERROR: {root} does not exist", file=sys.stderr); sys.exit(2)
    per_prop = root / "per_property"
    csv_data = root / "csv_data"
    if not per_prop.is_dir() or not csv_data.is_dir():
        print(f"ERROR: missing per_property/ or csv_data/ under {root}",
              file=sys.stderr); sys.exit(2)

    clean_dir = root / "Clean"
    split_dir = root / "Split"
    if not args.dry_run:
        clean_dir.mkdir(exist_ok=True)
        split_dir.mkdir(exist_ok=True)

    props = sorted([p.name for p in csv_data.iterdir() if p.is_dir()])
    print(f"Found {len(props)} properties under csv_data/")

    n_clean = n_split = n_skip_clean = n_skip_split = n_missing = 0
    for prop in props:
        src_csv   = per_prop / f"{prop}.csv"
        clean_csv = clean_dir / f"{prop}.csv"
        dst_split_parent = split_dir / prop

        # 1. per_property/<prop>.csv → Clean/<prop>.csv (rename smiles→SMILES, value→TARGET).
        if not src_csv.exists():
            print(f"  [missing] {src_csv}"); n_missing += 1; continue
        if clean_csv.exists() and clean_csv.stat().st_mtime >= src_csv.stat().st_mtime:
            n_skip_clean += 1
        else:
            if args.dry_run:
                print(f"  [would write] {clean_csv}")
            else:
                df = pd.read_csv(src_csv)
                cols = {c.lower(): c for c in df.columns}
                smi_col = cols.get("smiles") or cols.get("smile") or next(
                    c for c in df.columns if "smile" in c.lower())
                tgt_col = cols.get("value") or cols.get("target")
                if tgt_col is None:
                    # take last numeric column
                    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                    tgt_col = nums[0] if nums else df.columns[-1]
                out = pd.DataFrame({"SMILES": df[smi_col], "TARGET": df[tgt_col]}).dropna()
                out.to_csv(clean_csv, index=False)
            n_clean += 1

        # 2. Symlink BOTH random_cv5 and scaffold_cv5 (when they exist) so
        #    downstream wrappers can choose via SPLIT_KIND env var.
        for split_kind in ("random_cv5", "scaffold_cv5"):
            src_split_kind = csv_data / prop / "Split" / split_kind
            dst_split_kind = dst_split_parent / split_kind
            if not src_split_kind.is_dir():
                continue   # not all properties may have both — skip silently
            if args.dry_run:
                print(f"  [would link] {dst_split_kind} -> {src_split_kind}")
                n_split += 1
                continue
            dst_split_parent.mkdir(exist_ok=True)
            # Relative symlink so the repo works on any host.
            rel_target = os.path.relpath(src_split_kind, dst_split_kind.parent)
            if dst_split_kind.is_symlink() or dst_split_kind.exists():
                try:
                    current_target = (os.readlink(dst_split_kind)
                                      if dst_split_kind.is_symlink() else None)
                    if current_target == rel_target:
                        n_skip_split += 1
                        continue
                    if dst_split_kind.is_symlink() or dst_split_kind.is_file():
                        dst_split_kind.unlink()
                    else:
                        import shutil; shutil.rmtree(dst_split_kind)
                except FileNotFoundError:
                    pass
            dst_split_kind.symlink_to(rel_target, target_is_directory=True)
            n_split += 1

    print()
    print(f"  Clean csvs   : wrote {n_clean}, skipped {n_skip_clean} (up-to-date)")
    print(f"  Split links  : created/updated {n_split}, "
          f"skipped {n_skip_split} (already correct)")
    if n_missing:
        print(f"  Missing inputs: {n_missing}")
    print()
    print(f"Now use INPUT_DIR={clean_dir} and SPLIT_DIR_ROOT={split_dir} in your wrapper.")


if __name__ == "__main__":
    main()
