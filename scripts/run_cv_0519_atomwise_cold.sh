#!/usr/bin/env bash
# 0519 atomwise variant — same datasets, same config as the baseline, but
# uses --head-type atomwise (size-extensive per-atom MLP + scatter_sum).
#
# Together with scripts/run_cv_0519_baseline_cold.sh this is the A1
# experiment: train both heads on the same folds with DUMP_PREDS=1, then
# scripts/ensemble_preds.py averages the per-sample predictions to compute
# ensemble MAE / RMSE / R².  If the post-hoc average beats either single
# head, A3 (joint HybridHead) is worth running.
#
# Cache discipline: reuses RUN_TAG=0519 caches (Stage A/B/B.5) from the
# baseline.  Only Stage C trains the new head.
#
# Workload: 42 ds × 1 ckpt × 1 K = 42 jobs.  Stage A/B/B.5 skipped.
# On 8 GPUs × 4 slots → ~1.5 rounds Stage C; wall ~30-60 min.
#
# Usage:
#   nohup bash scripts/run_cv_0519_atomwise_cold.sh \
#       > /tmp/cv_0519_atomwise_cold.log 2>&1 & disown

set -uo pipefail
cd "$(dirname "$0")/.."

python scripts/build_cv_0519.py >/dev/null || {
    echo "ERROR: build_cv_0519.py failed" >&2; exit 1; }

export N_GPUS="${N_GPUS:-8}"
export CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
export TASKS_PER_GPU="${TASKS_PER_GPU:-4}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BATCH="${BATCH:-32}"
export EXTRACT_BATCH="${EXTRACT_BATCH:-16}"
export SAMPLE_BATCH="${SAMPLE_BATCH:-32}"

# DUMP_PREDS=1 is required for scripts/ensemble_preds.py to do the
# post-hoc average across the two head variants.
export DUMP_PREDS="${DUMP_PREDS:-1}"

# Variant: atomwise head — per-atom MLP + scatter_sum (size-extensive).
export HEAD_TYPE="atomwise"

export INPUT_DIR=downstream_data/cv_0519/Clean
export SPLIT_DIR_ROOT=downstream_data/cv_0519/Split
export SPLIT_KIND="${SPLIT_KIND:-random_cv5}"

export DATASETS_FILTER="log_solubility_water_molL,BP_K,Lipophilicity_logD,Hf_gas_kJmol,Pvap_log10mmHg,ST_298K_mNm,fusion_T_K,Hf_liq_kJmol,dielectric_298K,Hvap_at_TB_kJmol,PPBR_pct,H_combus_kJmol,Tc_K,Pc_bar,Vc_cm3mol,Sf_gas_JmolK,Gf_gas_kJmol,ESOL_logS,visc_liq_298K_cP,omega,UEL_volpct,Cp_liq_298K,LEL_volpct,flash_point_K,density_liq_298K_gcm3,expand_coeff_liq_K-1,gyration_radius_A,k_liq_298K,S_gas_JmolK,RI_298K,CEP_PCE,Cp_gas_298K,log_solubility_water_ppm,Q_10ppmv_mgg,dipole_moment_D,log_Koc,Hfus_at_TF_kJmol,freesolv_dG_kcalmol,visc_gas_298K_uPas,log_Henry_atmmolfrac,autoignition_K,k_gas_298K"

# Shares pkl/pt caches with the baseline (same RUN_TAG); different OUT_ROOT.
export RUN_TAG="${RUN_TAG:-0519}"
export OUT_ROOT="${OUT_ROOT:-outputs/cv_0519_atomwise_cold}"
export LOG_DIR="${LOG_DIR:-/tmp/cv_0519_atomwise_cold}"
export WANDB="${WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-downstream_cv_0519}"
export WANDB_GROUP="${WANDB_GROUP:-atomwise}"
export SWANLAB_SYNC="${SWANLAB_SYNC:-1}"

CKPT_DEFS=(
    "cold_combined|data/ft_ckpts/thermo_flow_cold_combined.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold_combined.yaml|0"
)
SAMPLING_MODES=(
    "standard|K8|8|10"
)

source scripts/run_cv.sh
