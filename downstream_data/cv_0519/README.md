# ThermoGen CV 0519 — 42 properties, random_cv5

Snapshot of `downstream_ft/0515_final/` taken on 2026-05-19, packaged as
a self-contained, no-symlink directory for portable CV runs.

## What's here

    Clean/<prop>.csv                          ⟵  SMILES, TARGET (training-ready)
    per_property/<prop>.csv                   ⟵  inchikey, smiles, value, tier,
                                                  scaffold, sources
    Split/<prop>/random_cv5/cv{1-5}_*.csv     ⟵  random 5-fold CV partitions
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
  * 42 properties × 20,841 unique molecules × 62,925 cells

## How to use with run_cv.sh

    INPUT_DIR=downstream_data/cv_0519/Clean \
    SPLIT_DIR_ROOT=downstream_data/cv_0519/Split \
    bash scripts/run_cv_0519_baseline_cold.sh

See `scripts/run_cv_0519_baseline_cold.sh` for the full reference wrapper.
