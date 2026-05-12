#!/usr/bin/env bash
# 0511 downstream CV — 4-GPU machine: 21 smaller datasets.
#
# Paired with run_cv_0511_8gpu.sh. See that file's header for method matrix
# and rationale for skipped variants.
#
# Datasets handled here (sorted descending by size):
#   Hf_L (4790), TPT (4461), Lipophilicity (4199), MP (3008), Cp (1958),
#   pKa (1492), de (1486), PPBR (1412), k (1329), Hf_C (1279), Vcp (1244),
#   ESOL (1115), Solubility_ethanol (925), CEP (882), AOH (692),
#   freesolv (641), BCF (595), Density (565), Clearance (492),
#   HalfLife (490), ST (304)
#
# Total: 21 datasets × 6 configs = 126 CV jobs.
#
# Usage:
#   nohup bash scripts/run_cv_0511_4gpu.sh > /tmp/cv_0511_4gpu.log 2>&1 & disown

set -uo pipefail
cd "$(dirname "$0")/.."

export N_GPUS=4
export CUDA_DEVICES=0,1,2,3
export TASKS_PER_GPU=1

# Smaller datasets here (≤ ~4800 mols), default batches should fit fine; just
# enable expandable_segments as a safety net for the largest few (Hf_L, TPT).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export INPUT_DIR=downstream_ft/0511_cc_audit/Clean
export SPLIT_DIR_ROOT=downstream_ft/0511_cc_audit/Split
export DATASETS_FILTER="Hf_L,TPT,Lipophilicity,MP,Cp,pKa,de,PPBR,k,Hf_C,Vcp,ESOL,Solubility_ethanol,CEP,AOH,freesolv,BCF,Density,Clearance,HalfLife,ST"

export RUN_TAG=0511
export OUT_ROOT=outputs/cv_0511
export LOG_DIR=/tmp/cv_0511
export WANDB=1
export WANDB_PROJECT=downstream_cv_0511

export CKPT_DEFS=(
    "loqi_flow|data/ft_ckpts/loqi_flow.ckpt|scripts/conf/loqi/loqi_flow.yaml|0"
    "cold_early|data/ft_ckpts/thermo_flow_cold_early.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold.yaml|0"
    "cold_last|data/ft_ckpts/thermo_flow_cold_last.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold.yaml|0"
)

export SAMPLING_MODES=(
    "standard|K8|8|10"
    "multistep|K12ms|12|4:10:7 8 9"
)

exec bash scripts/run_cv.sh
