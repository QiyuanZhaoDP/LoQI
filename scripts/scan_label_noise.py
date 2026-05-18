"""Rank molecules by out-of-fold prediction error to surface likely
label noise in a CV dataset.

How it works
------------
When a CV run is invoked with `DUMP_PREDS=1`, each fold writes a
`preds_fold{k}.csv` next to its `cv_report.json`:

    out_dir/
        cv_report.json
        preds_fold1.csv   columns: smiles,y_true,y_pred_best,y_pred_last_stable,group_id,has_target
        preds_fold2.csv
        ...
        preds_fold5.csv

In a clean 5-fold CV, every labeled molecule lands in the test fold of
exactly one fold.  This script:

  1. Loads all preds_fold*.csv under <task_dir>.
  2. Aggregates by canonical SMILES (mean if a SMILES shows up more
     than once — which happens in ensemble mode).
  3. Computes residual = y_pred - y_true and |residual| / σ(y_true).
  4. Sorts by |residual| descending and dumps the top-N as candidates
     for manual data-quality review.

The σ used for normalisation is the population std of *y_true* across
all rows in the dataset — so |residual|/σ ≥ 1.0 means the row is at
least one full data-std away from where the model thinks it should be.
Two or three σ is a strong "this label is probably wrong" signal.

Usage
-----
    # Single task
    python scripts/scan_label_noise.py outputs/cv_0518_revisit_cold/cold_combined_K8/visc_liq_298K_cP

    # Scan every task under an OUT_ROOT, write per-task CSVs into a
    # sibling _noise_scan/ directory
    python scripts/scan_label_noise.py outputs/cv_0518_revisit_cold/cold_combined_K8 \
        --batch --out-root outputs/cv_0518_revisit_cold/_noise_scan

    # Custom thresholds (default: top 30 OR rows with |Δ|/σ ≥ 1.5)
    python scripts/scan_label_noise.py <task_dir> --top-n 50 --sigma-cutoff 2.0
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import sys
from pathlib import Path


def _load_fold_preds(task_dir: Path):
    """Return list of dicts: {smiles, y_true, y_pred_best, y_pred_ls, fold, group_id}."""
    rows = []
    for fp in sorted(task_dir.glob('preds_fold*.csv')):
        fold = fp.stem.replace('preds_fold', '')
        with fp.open() as fh:
            for r in csv.DictReader(fh):
                try:
                    if int(r['has_target']) == 0:
                        continue
                except (KeyError, ValueError):
                    pass
                try:
                    y = float(r['y_true']); yp = float(r['y_pred_best'])
                    yls = float(r.get('y_pred_last_stable', yp))
                except (KeyError, ValueError):
                    continue
                rows.append({
                    'smiles': r['smiles'],
                    'y_true': y,
                    'y_pred_best': yp,
                    'y_pred_ls':   yls,
                    'group_id':    int(r.get('group_id', -1) or -1),
                    'fold':        int(fold),
                })
    return rows


def _aggregate_by_smiles(rows):
    """Average duplicate (smiles) rows — happens in ensemble mode."""
    by_smi = {}
    for r in rows:
        s = r['smiles']
        if s not in by_smi:
            by_smi[s] = {'smiles': s, 'y_true': r['y_true'],
                          'y_pred_best_sum': 0.0, 'y_pred_ls_sum': 0.0,
                          'n': 0, 'fold': r['fold']}
        by_smi[s]['y_pred_best_sum'] += r['y_pred_best']
        by_smi[s]['y_pred_ls_sum']   += r['y_pred_ls']
        by_smi[s]['n']               += 1
    out = []
    for s, d in by_smi.items():
        yp  = d['y_pred_best_sum'] / d['n']
        yls = d['y_pred_ls_sum']   / d['n']
        out.append({
            'smiles': s, 'y_true': d['y_true'], 'fold': d['fold'],
            'n_conformers': d['n'],
            'y_pred_best': yp, 'y_pred_ls': yls,
            'residual_best': yp - d['y_true'],
            'residual_ls':   yls - d['y_true'],
        })
    return out


def _std(values):
    n = len(values)
    if n <= 1:
        return 0.0
    m = sum(values) / n
    return math.sqrt(sum((v - m) ** 2 for v in values) / n)


def scan_one(task_dir: Path, top_n: int, sigma_cutoff: float, out_csv: Path | None):
    rows = _load_fold_preds(task_dir)
    if not rows:
        print(f'  [{task_dir.name}] no preds_fold*.csv found — skipping')
        return None

    agg = _aggregate_by_smiles(rows)
    sigma = _std([r['y_true'] for r in agg])
    if sigma <= 0:
        sigma = 1.0

    for r in agg:
        r['abs_residual_best']  = abs(r['residual_best'])
        r['abs_residual_ls']    = abs(r['residual_ls'])
        r['rel_residual_best']  = r['abs_residual_best'] / sigma
        r['rel_residual_ls']    = r['abs_residual_ls']   / sigma

    # Rank by best-val abs residual (descending).  In addition to top-N,
    # always include rows whose |Δ|/σ ≥ sigma_cutoff.
    agg.sort(key=lambda r: r['abs_residual_best'], reverse=True)
    keep = []
    for r in agg:
        if r['rel_residual_best'] >= sigma_cutoff or len(keep) < top_n:
            keep.append(r)
        else:
            break

    fieldnames = ['rank', 'smiles', 'y_true',
                  'y_pred_best', 'residual_best', 'rel_residual_best',
                  'y_pred_ls', 'residual_ls', 'rel_residual_ls',
                  'n_conformers', 'fold']
    if out_csv is None:
        out_csv = task_dir / 'noise_candidates.csv'
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for i, r in enumerate(keep, 1):
            w.writerow({
                'rank': i,
                'smiles': r['smiles'],
                'y_true': round(r['y_true'], 6),
                'y_pred_best': round(r['y_pred_best'], 6),
                'residual_best': round(r['residual_best'], 6),
                'rel_residual_best': round(r['rel_residual_best'], 3),
                'y_pred_ls': round(r['y_pred_ls'], 6),
                'residual_ls': round(r['residual_ls'], 6),
                'rel_residual_ls': round(r['rel_residual_ls'], 3),
                'n_conformers': r['n_conformers'],
                'fold': r['fold'],
            })

    n_over = sum(1 for r in agg if r['rel_residual_best'] >= sigma_cutoff)
    print(f'  [{task_dir.name}] n={len(agg)}  σ_y={sigma:.4g}  '
          f'|Δ|/σ ≥ {sigma_cutoff}: {n_over}  → {out_csv}')
    if keep[:5]:
        print('    top 5:')
        for i, r in enumerate(keep[:5], 1):
            print(f"      {i}. {r['smiles'][:60]:<60s}  "
                  f"y={r['y_true']:.4g}  ŷ={r['y_pred_best']:.4g}  "
                  f"|Δ|={r['abs_residual_best']:.4g}  |Δ|/σ={r['rel_residual_best']:.2f}")
    return out_csv


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('path', help='Task dir containing preds_fold*.csv OR '
                                 'a parent containing several task subdirs (use --batch).')
    p.add_argument('--batch', action='store_true',
                   help='Treat <path> as a parent dir; scan every immediate '
                        'subdir that contains preds_fold*.csv.')
    p.add_argument('--out-root', default=None,
                   help='When --batch: write per-task noise_candidates.csv '
                        'into this directory (named <task>.csv).  Default: '
                        'write noise_candidates.csv inside each task dir.')
    p.add_argument('--top-n', type=int, default=30,
                   help='Always include at least this many highest-residual rows (default 30).')
    p.add_argument('--sigma-cutoff', type=float, default=1.5,
                   help='Also include any row with |residual|/σ_y ≥ this value (default 1.5).')
    args = p.parse_args()

    root = Path(args.path).resolve()
    if not root.is_dir():
        print(f'ERROR: {root} is not a directory', file=sys.stderr); sys.exit(2)

    if args.batch:
        tasks = [d for d in sorted(root.iterdir())
                 if d.is_dir() and any(d.glob('preds_fold*.csv'))]
        if not tasks:
            print(f'No task dirs with preds_fold*.csv found under {root}', file=sys.stderr)
            sys.exit(1)
        print(f'Batch scan: {len(tasks)} task dirs under {root}\n')
        out_root = Path(args.out_root).resolve() if args.out_root else None
        if out_root:
            out_root.mkdir(parents=True, exist_ok=True)
        for td in tasks:
            out_csv = (out_root / f'{td.name}.csv') if out_root else None
            scan_one(td, args.top_n, args.sigma_cutoff, out_csv)
    else:
        scan_one(root, args.top_n, args.sigma_cutoff, None)


if __name__ == '__main__':
    main()
