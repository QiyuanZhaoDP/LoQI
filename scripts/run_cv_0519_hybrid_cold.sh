#!/usr/bin/env bash
# 0519 hybrid head variant — joint architecture for the A3 experiment.
#
#     ŷ = attention(H)  +  α · atomwise(H)
#
# where α is a learned scalar initialized at 0.1 (overridable via
# HYBRID_ALPHA_INIT env var).  Single backbone H, single forward pass,
# one ckpt.
#
# Scope: ONLY the 12 size-extensive properties (same set as
# run_cv_0519_atomwise_cold.sh).  The additive `α·atomwise(H)` term
# introduces a per-atom-summed component that physically only makes sense
# for extensive quantities (Hf, H_combus, Cp, S, Vc, ...).  Adding it to
# intensive properties like BP / ε / viscosity / density would let α drift
# to compensate for size dependence the target doesn't actually have —
# at best wasted capacity, at worst injects noise.
#
# Pair-comparison setup with cv_0519_baseline_cold + cv_0519_atomwise_cold
# on the SAME 12 datasets; same RUN_TAG=0519 caches; distinct OUT_ROOT
# + wandb group so the four (baseline / atomwise / hybrid / [HP cells])
# can be summarized side-by-side.
#
# Usage:
#   nohup bash scripts/run_cv_0519_hybrid_cold.sh \
#       > /tmp/cv_0519_hybrid_cold.log 2>&1 & disown
#
# To start α larger if you suspect the additive correction is more
# dominant than 10% of the attention output:
#   HYBRID_ALPHA_INIT=0.5 nohup bash scripts/run_cv_0519_hybrid_cold.sh ...

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

# Variant: hybrid head (attention + α·atomwise joint).
export HEAD_TYPE="hybrid"
export HYBRID_ALPHA_INIT="${HYBRID_ALPHA_INIT:-0.1}"

export INPUT_DIR=downstream_data/cv_0519/Clean
export SPLIT_DIR_ROOT=downstream_data/cv_0519/Split
export SPLIT_KIND="${SPLIT_KIND:-random_cv5}"

# Extensive subset (12).  Same set as run_cv_0519_atomwise_cold.sh.
export DATASETS_FILTER="Hf_gas_kJmol,Hf_liq_kJmol,Hvap_at_TB_kJmol,H_combus_kJmol,Vc_cm3mol,Sf_gas_JmolK,Gf_gas_kJmol,Cp_liq_298K,gyration_radius_A,S_gas_JmolK,Cp_gas_298K,Hfus_at_TF_kJmol"

export RUN_TAG="${RUN_TAG:-0519}"
export OUT_ROOT="${OUT_ROOT:-outputs/cv_0519_hybrid_cold}"
export LOG_DIR="${LOG_DIR:-/tmp/cv_0519_hybrid_cold}"
export WANDB="${WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-downstream_cv_0519}"
export WANDB_GROUP="${WANDB_GROUP:-hybrid_a${HYBRID_ALPHA_INIT}}"
export SWANLAB_SYNC="${SWANLAB_SYNC:-1}"

CKPT_DEFS=(
    "cold_combined|data/ft_ckpts/thermo_flow_cold_combined.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold_combined.yaml|0"
)
SAMPLING_MODES=(
    "standard|K8|8|10"
)

source scripts/run_cv.sh
