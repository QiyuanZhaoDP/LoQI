"""Assemble downstream_data/cv_0519/ from downstream_ft/0515_final/.

Produces a self-contained snapshot of the 42-property dataset at its
post-2026-05-19 state (PCCP-trusted dielectric + visc_liq ≤50 cap) with
just random_cv5 splits — no symlinks, no extra split kinds, no
per-property scratch dirs.  Layout:

    downstream_data/cv_0519/
        Clean/<prop>.csv                              (SMILES, TARGET)
        per_property/<prop>.csv                       (full schema)
        Split/<prop>/random_cv5/cv{1-5}_{train,valid,test}.csv
        README.md
        master.csv
        splits_summary.csv

Idempotent: re-runs replace the directory content cleanly.

Usage:
    python scripts/build_cv_0519.py
    python scripts/build_cv_0519.py --src downstream_ft/0515_final \
                                    --dst downstream_data/cv_0519
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default='downstream_ft/0515_final',
                    help='Source mirror dir (default: downstream_ft/0515_final).')
    ap.add_argument('--dst', default='downstream_data/cv_0519',
                    help='Output dir (default: downstream_data/cv_0519).')
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not src.is_dir():
        raise SystemExit(f'ERROR: source {src} not found')

    # Properties = whatever's in per_property/
    props = sorted(p.stem for p in (src / 'per_property').glob('*.csv'))
    print(f'  source: {src}')
    print(f'  dest:   {dst}')
    print(f'  {len(props)} properties found')

    dst.mkdir(parents=True, exist_ok=True)
    (dst / 'Clean').mkdir(exist_ok=True)
    (dst / 'per_property').mkdir(exist_ok=True)
    (dst / 'Split').mkdir(exist_ok=True)

    n_rows_total = 0
    summary_rows = []
    for prop in props:
        # 1) Clean csv
        clean_src = src / 'Clean' / f'{prop}.csv'
        if clean_src.exists():
            shutil.copyfile(clean_src, dst / 'Clean' / f'{prop}.csv')
        # 2) per_property csv
        pp_src = src / 'per_property' / f'{prop}.csv'
        if pp_src.exists():
            shutil.copyfile(pp_src, dst / 'per_property' / f'{prop}.csv')
            n_rows_total += sum(1 for _ in pp_src.open()) - 1
        # 3) random_cv5 fold csvs — resolve symlink and copy as real files
        split_src = src / 'Split' / prop / 'random_cv5'
        if split_src.is_symlink():
            split_src = split_src.resolve()
        if not split_src.is_dir():
            print(f'  WARN: {prop} has no random_cv5 — skipping splits')
            continue
        split_dst = dst / 'Split' / prop / 'random_cv5'
        split_dst.mkdir(parents=True, exist_ok=True)
        fold_sizes = []
        for fp in sorted(split_src.glob('cv*_*.csv')):
            shutil.copyfile(fp, split_dst / fp.name)
            if fp.name.endswith('_test.csv'):
                fold_sizes.append(sum(1 for _ in fp.open()) - 1)
        summary_rows.append({
            'property': prop,
            'n_molecules': sum(1 for _ in (dst / 'Clean' / f'{prop}.csv').open()) - 1,
            'random_fold_sizes': '|'.join(str(n) for n in fold_sizes),
        })

    # 4) splits_summary.csv (random-only subset)
    with (dst / 'splits_summary.csv').open('w', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=['property', 'n_molecules', 'random_fold_sizes'])
        w.writeheader(); w.writerows(summary_rows)

    # 5) master.csv (copy as-is)
    if (src / 'master.csv').exists():
        shutil.copyfile(src / 'master.csv', dst / 'master.csv')

    # 6) README
    n_mol_unique = sum(1 for _ in (dst / 'master.csv').open()) - 1 \
                   if (dst / 'master.csv').exists() else 0
    readme = f"""# ThermoGen CV 0519 — 42 properties, random_cv5

Snapshot of `downstream_ft/0515_final/` taken on 2026-05-19, packaged as
a self-contained, no-symlink directory for portable CV runs.

## What's here

    Clean/<prop>.csv                          ⟵  SMILES, TARGET (training-ready)
    per_property/<prop>.csv                   ⟵  inchikey, smiles, value, tier,
                                                  scaffold, sources
    Split/<prop>/random_cv5/cv{{1-5}}_*.csv     ⟵  random 5-fold CV partitions
                                                  (train/valid/test per fold)
    master.csv                                ⟵  wide pivot, all 42 properties
    splits_summary.csv                        ⟵  n_molecules + fold sizes
    README.md                                 ⟵  this file

## Provenance

Base: `downstream_ft/0515_final/` after the 2026-05-19 data cleanup pass
(commit `7c13f1a` on main):

  * dielectric_298K: +73 PCCP-trusted secondary_single rows (N-oxides,
    branched sulfones, butylene/pentylene carbonates etc.) — total 1,435
  * visc_liq_298K_cP: rows with value > 50 mPa·s dropped — total 1,188
  * visc_liq_298K_cP_manual: removed (auto pipeline is the canonical source)
  * 42 properties × {n_mol_unique:,} unique molecules × {n_rows_total:,} cells

## How to use with run_cv.sh

    INPUT_DIR=downstream_data/cv_0519/Clean \\
    SPLIT_DIR_ROOT=downstream_data/cv_0519/Split \\
    bash scripts/run_cv_0519_baseline_cold.sh

See `scripts/run_cv_0519_baseline_cold.sh` for the full reference wrapper.
"""
    (dst / 'README.md').write_text(readme)

    print(f'\n  Wrote {dst}:')
    print(f'    Clean/        {len(props)} files')
    print(f'    per_property/ {len(props)} files')
    print(f'    Split/        {len(props)} props × random_cv5 (15 csv each = {15*len(props)} files)')
    print(f'    master.csv, splits_summary.csv, README.md')
    print(f'\n  Total cells across all properties: {n_rows_total:,}')


if __name__ == '__main__':
    main()
