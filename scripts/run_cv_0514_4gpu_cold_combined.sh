#!/usr/bin/env bash
# 0514 downstream CV — 4-GPU machine, cold_combined warm-init, Stage C only.
#
# Assumes Stage A/B (conformer sampling) and Stage B.5 (H extraction) have
# already been completed on another machine (typically the 8-GPU box running
# run_cv_0514_8gpu_cold_combined.sh). This wrapper skips both via:
#   SKIP_SMI=1, SKIP_SAMPLE=1, SKIP_EXTRACT=1
#
# What it expects on disk (RUN_TAG=0511 names):
#   data/0511_pkl_cold_combined_{k8,k12ms}/<dataset>.pkl
#   data/0511_pt_cold_combined_{k8,k12ms}/<dataset>_K{8,12}.pt
#   data/0511_pt_cold_combined_{k8,k12ms}/<dataset>_H.pt
#
# If a dataset's pkl is missing, run_cv.sh:343 silently drops it from the
# Stage C queue with no warning — verify before launching:
#
#   ls data/0511_pkl_cold_combined_k8/ | sort > /tmp/k8_pkls
#   ls data/0511_pkl_cold_combined_k12ms/ | sort > /tmp/k12_pkls
#   wc -l /tmp/k*_pkls    # expect 21 each (matches DATASETS_FILTER count)
#
# Datasets: 21 smaller ones (same set as run_cv_0511_4gpu.sh).
# Workload: 21 ds × 1 ckpt × 2 sampling = 42 CV jobs.
# With 4 GPUs × TASKS_PER_GPU=4 = 16 slots → ~3 rounds.
# Stage C only: roughly 1-2 h total wall time on H-cached training.
#
# Usage:
#   nohup bash scripts/run_cv_0514_4gpu_cold_combined.sh \
#       > /tmp/cv_0514_4gpu_cold_combined.log 2>&1 & disown

set -uo pipefail
cd "$(dirname "$0")/.."

export N_GPUS=4
export CUDA_DEVICES=0,1,2,3
# 4 tasks/GPU at bs=32 — same proven config as the original 4-GPU wrapper.
# H caches are small for these 21 datasets (~100-500 MB each), so 4 ×
# ~300 MB shared H + ~1 GB per-task state ≈ 5-6 GB per GPU.
export TASKS_PER_GPU=4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# BATCH defaults from run_cv.sh (32). No need to override here.

# Skip the heavy stages — assumes caches already exist from the 8-GPU run.
export SKIP_SMI=1
export SKIP_SAMPLE=1
export SKIP_EXTRACT=1

export INPUT_DIR=downstream_ft/0511_cc_audit/Clean
export SPLIT_DIR_ROOT=downstream_ft/0511_cc_audit/Split

# 21 smaller datasets (matches run_cv_0511_4gpu.sh's set). Sorted descending
# by training-set size for LPT scheduling.
export DATASETS_FILTER="Hf_L,TPT,Lipophilicity,MP,Cp,pKa,de,PPBR,k,Hf_C,Vcp,ESOL,Solubility_ethanol,CEP,AOH,freesolv,BCF,Density,Clearance,HalfLife,ST"

# RUN_TAG=0511 — share the on-disk data caches with the 8-GPU run; the
# cv_0514 output namespace stays for run results / wandb so this and the
# 8-GPU wrapper share one combined project.
export RUN_TAG=0511
export OUT_ROOT=outputs/cv_0514
export LOG_DIR=/tmp/cv_0514
export WANDB=1
export WANDB_PROJECT=downstream_cv_0514
export SWANLAB_SYNC=1

# Single ckpt: cold_combined with warm-init head from the trained 14-target
# combined head (load_thermo_head_into picks
# dynamics.ema_model.module.combined_heads.mp.* after bb75d1d).
CKPT_DEFS=(
    "cold_combined|data/ft_ckpts/thermo_flow_cold_combined.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold_combined.yaml|1"
)

# K=8 + K=12 multi-step. Drop one if the corresponding pkls aren't there.
SAMPLING_MODES=(
    "standard|K8|8|10"
    "multistep|K12ms|12|4:10:7 8 9"
)

source scripts/run_cv.sh
