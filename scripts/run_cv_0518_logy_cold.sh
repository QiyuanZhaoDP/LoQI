#!/usr/bin/env bash
# 0518 log-target variant — same 5 datasets, same cold_combined cold-init,
# but trains the head on log10(y + 1e-3) instead of raw y.  Predictions are
# inverse-transformed (10**ŷ) before MAE/RMSE/R²/dump_preds, so the
# reported metrics and noise_candidates.csv are still in raw units and
# directly comparable to the baseline run_cv_0518_revisit_cold.sh.
#
# Why this should help (per /Users/zhao922/Desktop/preds_fold{1,2}.csv
# diagnosis): for viscosity ≈ log-normal target with mean 4 mPa·s, median
# 0.92, and a thin H-bond tail up to 188 mPa·s, raw-MSE training over-
# weights ~16 polyol/polyamine outliers (3% of the dataset, 93% of total
# SSE).  Log-target training rebalances loss across the dynamic range.
#
# Cache discipline:
#   Re-uses the same RUN_TAG=0518rev as run_cv_0518_revisit_cold.sh →
#   Stage A/B/B.5 caches are SHARED across variants (only Stage C differs).
#   If you've already run the baseline once, this script will skip the
#   heavy stages and run training only.
#
# Usage — sequential (after baseline; uses all 8 GPUs):
#   nohup bash scripts/run_cv_0518_logy_cold.sh \
#       > /tmp/cv_0518_logy_cold.log 2>&1 & disown
#
# Usage — parallel with the filter variant on a single 8-GPU box:
#   CUDA_DEVICES=0,1,2,3 N_GPUS=4 nohup bash scripts/run_cv_0518_logy_cold.sh \
#       > /tmp/cv_0518_logy_cold.log 2>&1 & disown
#   CUDA_DEVICES=4,5,6,7 N_GPUS=4 nohup bash scripts/run_cv_0518_filt50_cold.sh \
#       > /tmp/cv_0518_filt50_cold.log 2>&1 & disown

set -uo pipefail
cd "$(dirname "$0")/.."

python scripts/prep_0515_layout.py --root downstream_ft/0515_final || {
    echo "ERROR: prep_0515_layout.py failed; aborting" >&2; exit 1; }

export N_GPUS="${N_GPUS:-8}"
export CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
export TASKS_PER_GPU="${TASKS_PER_GPU:-4}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BATCH="${BATCH:-32}"
export EXTRACT_BATCH="${EXTRACT_BATCH:-16}"
export SAMPLE_BATCH="${SAMPLE_BATCH:-32}"

# Per-sample dump so we can re-run scan_label_noise.py on the log variant.
export DUMP_PREDS="${DUMP_PREDS:-1}"

# ── Variant-specific knob ─────────────────────────────────────────────────
# downstream_cv.py: when LOG_TARGET=1, head trains on log10(y + LOG_TARGET_EPS).
# Predictions are 10**(.) before metrics → cv_report.json and preds CSVs are
# already in raw units and directly comparable to the baseline.
export LOG_TARGET=1
export LOG_TARGET_EPS="${LOG_TARGET_EPS:-1e-3}"
# ──────────────────────────────────────────────────────────────────────────

export INPUT_DIR=downstream_ft/0515_final/Clean
export SPLIT_DIR_ROOT=downstream_ft/0515_final/Split

export DATASETS_FILTER="ST_298K_mNm,Hvap_at_TB_kJmol,dielectric_298K,visc_liq_298K_cP"

# Same RUN_TAG → share Stage A/B/B.5 caches with the baseline.  Distinct
# OUT_ROOT + WANDB project so the log variant doesn't collide with the
# baseline metrics on disk or in wandb.
export RUN_TAG="${RUN_TAG:-0518rev}"
export OUT_ROOT="${OUT_ROOT:-outputs/cv_0518_logy_cold}"
export LOG_DIR="${LOG_DIR:-/tmp/cv_0518_logy_cold}"
export WANDB="${WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-downstream_cv_0518_logy}"
export SWANLAB_SYNC="${SWANLAB_SYNC:-1}"

CKPT_DEFS=(
    "cold_combined|data/ft_ckpts/thermo_flow_cold_combined.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold_combined.yaml|0"
)

SAMPLING_MODES=(
    "standard|K8|8|10"
)

source scripts/run_cv.sh
