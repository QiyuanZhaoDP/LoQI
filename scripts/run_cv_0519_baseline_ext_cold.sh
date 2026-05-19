#!/usr/bin/env bash
# 0519 attention-head baseline RESTRICTED to the 12 size-extensive properties.
#
# Mirror of run_cv_0519_baseline_cold.sh but with the same DATASETS_FILTER
# as the atomwise + hybrid wrappers, so the three heads (attention /
# atomwise / hybrid) can be compared head-to-head on the EXACT SAME splits.
# Sequence A1 / A3 against this baseline; the full 42-property attention
# run still lives in run_cv_0519_baseline_cold.sh.

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
export DUMP_PREDS="${DUMP_PREDS:-1}"

export HEAD_TYPE="attention"

export INPUT_DIR=downstream_data/cv_0519/Clean
export SPLIT_DIR_ROOT=downstream_data/cv_0519/Split
export SPLIT_KIND="${SPLIT_KIND:-random_cv5}"

# 12-property extensive subset (same as atomwise + hybrid wrappers).
export DATASETS_FILTER="Hf_gas_kJmol,Hf_liq_kJmol,Hvap_at_TB_kJmol,H_combus_kJmol,Vc_cm3mol,Sf_gas_JmolK,Gf_gas_kJmol,Cp_liq_298K,gyration_radius_A,S_gas_JmolK,Cp_gas_298K,Hfus_at_TF_kJmol"

export RUN_TAG="${RUN_TAG:-0519}"
export OUT_ROOT="${OUT_ROOT:-outputs/cv_0519_baseline_ext_cold}"
export LOG_DIR="${LOG_DIR:-/tmp/cv_0519_baseline_ext_cold}"
export WANDB="${WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-downstream_cv_0519}"
export WANDB_GROUP="${WANDB_GROUP:-attention_ext}"
export SWANLAB_SYNC="${SWANLAB_SYNC:-1}"

CKPT_DEFS=(
    "cold_combined|data/ft_ckpts/thermo_flow_cold_combined.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold_combined.yaml|0"
)
SAMPLING_MODES=(
    "standard|K8|8|10"
)

source scripts/run_cv.sh
