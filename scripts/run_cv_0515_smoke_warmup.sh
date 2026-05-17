#!/usr/bin/env bash
# Smoke test for the warmup + tighter-grad-clip fix.
#
# Goal: verify that LR warmup (10% of steps) + grad-clip=0.1 eliminates
# the step-30-style training-MAE spike user observed on k_liq_298K and
# other properties.
#
# Workload: 6 small datasets × 2 inits (warm + cold) = 12 CV jobs.
# Single ckpt (cold_combined), K8 only. All on one 8-GPU machine with
# 2 tasks/GPU (so 16 slots, ~1 round). Each fold is small (train ≤ 1000)
# → total wall ~30-60 min.
#
# Datasets chosen for diversity + small size (faster smoke):
#   k_liq_298K (716)             — user's reference plot
#   freesolv_dG_kcalmol (462)    — solvation, very small
#   ESOL_logS (804)              — solvation, slightly larger
#   Cp_gas_298K (621)            — thermochemistry
#   Hfus_at_TF_kJmol (472)       — fusion enthalpy
#   visc_gas_298K_uPas (451)     — transport / kinetic
#
# After it finishes, replot train_mae_phys_step from any of the per-fold
# logs and verify the spike is gone.
#
# Outputs land in outputs/cv_0515_smoke/<ds>_cold_combined_W_K8 (warm)
# and outputs/cv_0515_smoke/<ds>_cold_combined_C_K8 (cold) — distinct
# suffixes so warm vs cold don't collide on the same dataset.
#
# Usage:
#   nohup bash scripts/run_cv_0515_smoke_warmup.sh \
#       > /tmp/cv_0515_smoke_warmup.log 2>&1 & disown

set -uo pipefail
cd "$(dirname "$0")/.."

# ── 0. layout prep (no-op if already done) ─────────────────────────────────
python scripts/prep_0515_layout.py --root downstream_ft/0515_final >/dev/null || {
    echo "ERROR: prep_0515_layout.py failed; aborting" >&2; exit 1; }

# ── 1. settings ────────────────────────────────────────────────────────────
export N_GPUS=8
export CUDA_DEVICES=0,1,2,3,4,5,6,7
export TASKS_PER_GPU=2          # 12 jobs / 16 slots — one near-round

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BATCH=32
export EXTRACT_BATCH=16
export SAMPLE_BATCH=32

# ★★★ New defaults being smoke-tested — match production wrappers post-update ★★★
# (these match the new run_cv.sh defaults so smoke == prod after this commit)
export WARMUP_FRACTION=0.2      # 20% of total steps as linear warmup
export GRAD_CLIP=0.1            # tight clip, paired with warmup
# Also the new epoch / patience defaults from run_cv.sh:
#   EPOCHS_LARGE=150 (was 200)  EPOCHS_SMALL=100 (was 150)  EARLY_STOP_PATIENCE=50 (was 100)
# These all 6 datasets are "small" (n_train < 1000) → EPOCHS_SMALL=100 applies.

export INPUT_DIR=downstream_ft/0515_final/Clean
export SPLIT_DIR_ROOT=downstream_ft/0515_final/Split

# 6 small / fast datasets
export DATASETS_FILTER="k_liq_298K,freesolv_dG_kcalmol,ESOL_logS,Cp_gas_298K,Hfus_at_TF_kJmol,visc_gas_298K_uPas"

export RUN_TAG=0515
export OUT_ROOT=outputs/cv_0515_smoke
export LOG_DIR=/tmp/cv_0515_smoke
export WANDB=1
export WANDB_PROJECT=downstream_cv_0515_smoke
export SWANLAB_SYNC=1

# Both warm AND cold init for each dataset — direct A/B with the fix applied.
# Different labels so OUT_ROOT/<ds>_<label>_K8 paths don't collide.
CKPT_DEFS=(
    "cold_combined_W|data/ft_ckpts/thermo_flow_cold_combined.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold_combined.yaml|1"
    "cold_combined_C|data/ft_ckpts/thermo_flow_cold_combined.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold_combined.yaml|0"
)

# K8 only.
SAMPLING_MODES=(
    "standard|K8|8|10"
)

source scripts/run_cv.sh
