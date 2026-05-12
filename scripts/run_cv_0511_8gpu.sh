#!/usr/bin/env bash
# 0511 downstream CV — 8-GPU machine: 7 largest datasets.
#
# Workload split (paired with run_cv_0511_4gpu.sh):
#   8-GPU: 7 datasets, ~70% of total rows (Solubility_water, Tc, Pc, RI,
#          AcuteToxicity, BP, Hf_G)  — 7 × 6 configs = 42 CV jobs
#   4-GPU: 21 datasets,  ~30% of total rows                — 126 CV jobs
# Designed so both machines wall-clock ≈ even.
#
# Method matrix (3 ckpts × 2 K-modes = 6 configs):
#   loqi_flow       (no thermo pretrain — baseline)
#   cold_early      (cold backbone, early ckpt, cold head)
#   cold_last       (cold backbone, last  ckpt, cold head)
#   × K8 (standard) and K12ms (multistep 4 traj × 3 snapshots)
#
# Skipped on purpose (already known to be worse from prior 223-file analysis):
#   warm_last / *_w / cold_late / K5 / K9ms
#
# Usage:
#   nohup bash scripts/run_cv_0511_8gpu.sh > /tmp/cv_0511_8gpu.log 2>&1 & disown

set -uo pipefail
cd "$(dirname "$0")/.."

export N_GPUS=8
export CUDA_DEVICES=0,1,2,3,4,5,6,7
export TASKS_PER_GPU=1

# Big-dataset memory tuning — these 7 datasets have 6k–11k molecules; with
# K12ms × 4 trajectories × ~12 conformers the .pt blows up to 130k+ rows,
# and the default BATCH=64 → 19 GiB OOM on 80 GB cards. Half the batch and
# enable expandable_segments to allow torch to reclaim fragmented memory.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BATCH=32           # CV training
export EXTRACT_BATCH=16   # H-cache extraction (heaviest VRAM stage)
export SAMPLE_BATCH=32    # conformer sampling

export INPUT_DIR=downstream_ft/0511_cc_audit/Clean
export SPLIT_DIR_ROOT=downstream_ft/0511_cc_audit/Split
export DATASETS_FILTER="Solubility_water,Tc,Pc,RI,AcuteToxicity,BP,Hf_G"

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
