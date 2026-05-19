# ThermoGen CV 0519 — 42 properties × 3 split kinds

Self-contained snapshot of `downstream_ft/0515_final/` taken on 2026-05-19,
with no symlinks — meant as the canonical, portable dataset for all
downstream CV / benchmarking work.

## What's here

    Clean/<prop>.csv                                ⟵  SMILES, TARGET (training-ready)
    per_property/<prop>.csv                         ⟵  inchikey, smiles, value, tier,
                                                          scaffold, sources
    Split/<prop>/random_cv5/cv{1-5}_*.csv           ⟵  IID baseline (i.i.d. random 5-fold)
    Split/<prop>/scaffold_diverse_cv5/cv{1-5}_*.csv ⟵  OOD via FP-distance maximization
    Split/<prop>/scaffold_hybrid_cv5/cv{1-5}_*.csv  ⟵  OOD via Lloyd→rebal→swap (harder)
    master.csv                                      ⟵  wide pivot, all 42 properties
    splits_summary.csv                              ⟵  n_molecules + per-split fold sizes
    README.md                                       ⟵  this file

## Split kinds (3)

| Kind                   | Method                                   | Median OOD distance |
|------------------------|------------------------------------------|--------------------:|
| `random_cv5`           | i.i.d. 5-fold by molecule                | ~0.45 (IID baseline)|
| `scaffold_diverse_cv5` | ECFP-distance OOD-maximization           | ~0.60               |
| `scaffold_hybrid_cv5`  | Lloyd→rebal→swap (most aggressive OOD)   | ~0.63               |

Pilot 8-property comparison (`outputs/cv_0519_*_subset/`) showed
scaffold_hybrid_cv5 is systematically harder than scaffold_diverse_cv5
by ~5-15% R²; the latter is recommended as the default OOD benchmark
unless you specifically want to stress-test generalization.

## Provenance

Base: `downstream_ft/0515_final/` after the 2026-05-19 data cleanup pass
(commit `7c13f1a` on main):

  * dielectric_298K: +73 PCCP-trusted secondary_single rows (N-oxides,
    branched sulfones, butylene/pentylene carbonates etc.) — total 1,435
  * visc_liq_298K_cP: rows with value > 50 mPa·s dropped — total 1,188
  * visc_liq_298K_cP_manual: removed (auto pipeline is the canonical source)
  * 42 properties × 20,841 unique molecules × 62,925 cells

## How to use with run_cv.sh

    INPUT_DIR=downstream_data/cv_0519/Clean \
    SPLIT_DIR_ROOT=downstream_data/cv_0519/Split \
    SPLIT_KIND=random_cv5 \   # or scaffold_diverse_cv5 / scaffold_hybrid_cv5
    bash scripts/run_cv_0519_baseline_cold.sh

See `scripts/run_cv_0519_*.sh` for reference wrappers.
