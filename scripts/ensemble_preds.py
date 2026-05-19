"""Post-hoc ensemble of two CV runs that emitted preds_fold*.csv.

Reads `preds_fold{k}.csv` from both task dirs, joins on (smiles, fold),
averages the y_pred_best columns, and recomputes MAE / RMSE / R² as if a
single ensemble model produced the averaged predictions.

Use case: the 0518 head_pool ablation showed attention wins MAE on most
targets while atomwise wins RMSE/R² on size-extensive thermo properties.
The two heads have complementary error modes, so simply averaging the
predictions of (run_A, run_B) often improves both metrics.

Usage
-----
    # Single property
    python scripts/ensemble_preds.py \
        outputs/cv_0519_baseline_cold/cold_combined_K8/BP_K \
        outputs/cv_0519_atomwise_cold/cold_combined_K8/BP_K

    # Batch over every common task under two OUT_ROOTs
    python scripts/ensemble_preds.py \
        outputs/cv_0519_baseline_cold/cold_combined_K8 \
        outputs/cv_0519_atomwise_cold/cold_combined_K8 \
        --batch --out /tmp/cv_0519_ensemble.csv

Output
------
For each task: prints baseline-A / baseline-B / ensemble metrics side by
side; in --batch mode also writes a CSV with one row per task.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def _load_fold_dict(task_dir: Path):
    """Return {(smiles, fold): {'y': y, 'p_b': p_best, 'p_ls': p_ls}}."""
    out = {}
    for fp in sorted(task_dir.glob('preds_fold*.csv')):
        fold = int(fp.stem.replace('preds_fold', ''))
        for r in csv.DictReader(fp.open()):
            try:
                if int(r['has_target']) == 0:
                    continue
            except (KeyError, ValueError):
                pass
            try:
                y = float(r['y_true'])
                pb = float(r['y_pred_best'])
                pls = float(r.get('y_pred_last_stable', pb))
            except (KeyError, ValueError):
                continue
            out[(r['smiles'], fold)] = {'y': y, 'p_b': pb, 'p_ls': pls}
    return out


def _aggregate_by_smiles(samples):
    """samples: {(smi, fold): row}.  Returns {smi: row} averaging duplicates."""
    by_smi = {}
    for (smi, _fold), row in samples.items():
        if smi not in by_smi:
            by_smi[smi] = {'y': row['y'], 'p_b_sum': 0.0, 'p_ls_sum': 0.0, 'n': 0}
        by_smi[smi]['p_b_sum'] += row['p_b']
        by_smi[smi]['p_ls_sum'] += row['p_ls']
        by_smi[smi]['n'] += 1
    return {smi: {'y': d['y'],
                  'p_b': d['p_b_sum'] / d['n'],
                  'p_ls': d['p_ls_sum'] / d['n']}
            for smi, d in by_smi.items()}


def _metrics(rows, key):
    """rows: list of {'y':..., key:...} → (mae, rmse, r2)."""
    ys = [r['y'] for r in rows]
    ps = [r[key] for r in rows]
    n = len(ys)
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    res = [(p - y) for p, y in zip(ps, ys)]
    mae = sum(abs(r) for r in res) / n
    rmse = math.sqrt(sum(r * r for r in res) / n)
    mean_y = sum(ys) / n
    sst = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1 - sum(r * r for r in res) / sst if sst > 0 else float('nan')
    return mae, rmse, r2


def ensemble_one(task_A: Path, task_B: Path):
    """Return dict with metrics for A, B, ensemble.  None on failure."""
    A_raw = _load_fold_dict(task_A); B_raw = _load_fold_dict(task_B)
    if not A_raw or not B_raw:
        return None
    # Restrict to (smiles, fold) pairs present in both runs.
    keys_common = set(A_raw) & set(B_raw)
    if not keys_common:
        return None
    ens_raw = {k: {'y': A_raw[k]['y'],
                   'p_b': 0.5 * (A_raw[k]['p_b'] + B_raw[k]['p_b']),
                   'p_ls': 0.5 * (A_raw[k]['p_ls'] + B_raw[k]['p_ls'])}
               for k in keys_common}
    # Aggregate by SMILES (ensemble mode K conformers → per-mol mean).
    A_agg = list(_aggregate_by_smiles({k: A_raw[k] for k in keys_common}).values())
    B_agg = list(_aggregate_by_smiles({k: B_raw[k] for k in keys_common}).values())
    E_agg = list(_aggregate_by_smiles(ens_raw).values())
    out = {'n_mols': len(E_agg)}
    for tag, rows in (('A', A_agg), ('B', B_agg), ('E', E_agg)):
        m_b, r_b, r2_b = _metrics(rows, 'p_b')
        m_l, r_l, r2_l = _metrics(rows, 'p_ls')
        out[f'{tag}_MAE_bv']  = m_b; out[f'{tag}_RMSE_bv']  = r_b; out[f'{tag}_R2_bv']  = r2_b
        out[f'{tag}_MAE_ls']  = m_l; out[f'{tag}_RMSE_ls']  = r_l; out[f'{tag}_R2_ls']  = r2_l
    return out


def _print_one(name, m):
    print(f'\n  {name}    n_mols={m["n_mols"]}')
    print(f'  {"":<24s} {"MAE_bv":>9s} {"RMSE_bv":>9s} {"R²_bv":>8s}   '
          f'{"MAE_ls":>9s} {"RMSE_ls":>9s} {"R²_ls":>8s}')
    for tag, lbl in (('A','head A   '), ('B','head B   '), ('E','ensemble ★')):
        print(f'  {lbl:<24s} '
              f'{m[f"{tag}_MAE_bv"]:>9.4f} {m[f"{tag}_RMSE_bv"]:>9.4f} {m[f"{tag}_R2_bv"]:>+8.3f}   '
              f'{m[f"{tag}_MAE_ls"]:>9.4f} {m[f"{tag}_RMSE_ls"]:>9.4f} {m[f"{tag}_R2_ls"]:>+8.3f}')
    # Was ensemble worth it?
    delta_mae_bv = m['A_MAE_bv'] - m['E_MAE_bv']
    delta_r2_bv  = m['E_R2_bv']  - max(m['A_R2_bv'], m['B_R2_bv'])
    flag_mae = '✓' if m['E_MAE_bv'] <= min(m['A_MAE_bv'], m['B_MAE_bv']) else '✗'
    flag_r2  = '✓' if m['E_R2_bv']  >= max(m['A_R2_bv'],  m['B_R2_bv'])  else '✗'
    print(f'  {"ensemble vs best":<24s}   '
          f'ΔMAE_bv={m["E_MAE_bv"]-min(m["A_MAE_bv"],m["B_MAE_bv"]):+.4f} ({flag_mae})   '
          f'ΔR²_bv={m["E_R2_bv"]-max(m["A_R2_bv"],m["B_R2_bv"]):+.4f} ({flag_r2})')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('task_a', help='Run A task dir (or parent with --batch)')
    ap.add_argument('task_b', help='Run B task dir (or parent with --batch)')
    ap.add_argument('--batch', action='store_true',
                   help='Treat task_a / task_b as parent dirs; ensemble every '
                        'common subtask.')
    ap.add_argument('--out', default=None,
                   help='(batch) write a CSV with one row per task.')
    args = ap.parse_args()

    a = Path(args.task_a).resolve()
    b = Path(args.task_b).resolve()

    if not args.batch:
        m = ensemble_one(a, b)
        if m is None:
            print(f'ERROR: no usable preds_fold*.csv pair found', file=sys.stderr)
            sys.exit(2)
        _print_one(f'{a.name}', m)
        return

    # Batch
    a_tasks = {p.name: p for p in sorted(a.iterdir())
               if p.is_dir() and any(p.glob('preds_fold*.csv'))}
    b_tasks = {p.name: p for p in sorted(b.iterdir())
               if p.is_dir() and any(p.glob('preds_fold*.csv'))}
    common = sorted(set(a_tasks) & set(b_tasks))
    if not common:
        print(f'ERROR: no common task names with preds under {a} & {b}', file=sys.stderr)
        sys.exit(2)
    print(f'Ensembling {len(common)} tasks under {a.name}  vs  {b.name}')

    all_rows = []
    n_e_wins_mae_bv = n_e_wins_r2_bv = 0
    for name in common:
        m = ensemble_one(a_tasks[name], b_tasks[name])
        if m is None: continue
        _print_one(name, m)
        m_out = {'task': name, **m}
        all_rows.append(m_out)
        if m['E_MAE_bv'] <= min(m['A_MAE_bv'], m['B_MAE_bv']): n_e_wins_mae_bv += 1
        if m['E_R2_bv']  >= max(m['A_R2_bv'],  m['B_R2_bv']):  n_e_wins_r2_bv  += 1

    print(f'\n=== Ensemble win count over {len(all_rows)} tasks ===')
    print(f'  ensemble MAE_bv ≤ both singles: {n_e_wins_mae_bv}/{len(all_rows)}')
    print(f'  ensemble R²_bv  ≥ both singles: {n_e_wins_r2_bv}/{len(all_rows)}')

    if args.out:
        with open(args.out, 'w', newline='') as fp:
            w = csv.DictWriter(fp, fieldnames=list(all_rows[0].keys()))
            w.writeheader(); w.writerows(all_rows)
        print(f'\nWrote {len(all_rows)} rows → {args.out}')


if __name__ == '__main__':
    main()
