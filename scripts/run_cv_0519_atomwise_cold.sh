#!/usr/bin/env bash
# 0519 atomwise variant — runs --head-type atomwise on ONLY the
# size-extensive subset of properties (12 of 42).
#
# Why only extensive: atomwise = per-atom MLP + scatter_sum, so the output
# scales linearly with N_atoms.  This is the right physical inductive bias
# for formation energies, heat capacities, entropies, critical volume etc.
# (which are themselves extensive), but actively WRONG for intensive
# properties like BP_K / ε / viscosity / density / partition coefficients
# where running atomwise would introduce a spurious size dependence.
#
# Intensive properties (30 of 42) keep the attention head — see
# scripts/run_cv_0519_baseline_cold.sh.
#
# Pair-comparison setup with cv_0519_baseline_cold on the SAME 12
# extensive datasets so scripts/ensemble_preds.py can do a head-to-head
# ensemble (A1 experiment).  RUN_TAG=0519 shares Stage A/B/B.5 caches.
#
# Workload: 12 ds × 1 ckpt × 1 K = 12 jobs.  Stage A/B/B.5 skipped.
# On 8 GPUs × 4 slots → 1 round Stage C; wall ~15-30 min.
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

# Extensive subset (12).  Sorted descending by row count.
# Includes: formation H/G/S (gas+liq), Cp, Hvap, Hfus, H_combus, Vc, R_g.
# Excludes intensive properties (BP, Tc, ε, η, ρ, ω, RI, partition, ...).
export DATASETS_FILTER="Hf_gas_kJmol,Hf_liq_kJmol,Hvap_at_TB_kJmol,H_combus_kJmol,Vc_cm3mol,Sf_gas_JmolK,Gf_gas_kJmol,Cp_liq_298K,gyration_radius_A,S_gas_JmolK,Cp_gas_298K,Hfus_at_TF_kJmol"

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
