#!/usr/bin/env bash
# 0511 downstream CV — 8-GPU machine: 4 mid-large datasets.
#
# Excluded (n > 10,000 molecules — extract takes >7 h/dataset at bs=16,
# and Tc/Pc contain long-chain hydrocarbons that OOM at bs=16 around 46%
# of extract due to torch.cat in fn_model.py:498):
#   Solubility_water (11,301), Tc (10,909), Pc (10,732)
# If you need those, run them as a one-off with EXTRACT_BATCH=4-8.
#
# Workload split:
#   8-GPU: 4 datasets (RI, AcuteToxicity, BP, Hf_G)    — 4 × 6 = 24 jobs
#   4-GPU: 21 datasets                                — 21 × 6 = 126 jobs
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

# Memory tuning — these 4 datasets have 6k–8.3k molecules; with K12ms ×
# 4 trajectories × ~12 conformers the .pt is 70k–100k rows. Default
# BATCH=64 OOMs on 80 GB cards (fragmented caching allocator), bs=16
# was confirmed stable for RI/AcuteToxicity/BP/Hf_G. expandable_segments
# helps torch reclaim fragments between batches.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BATCH=128          # CV training — bigger here because these 4 sets
                          # are the larger ones (5k-8k mols) and benefit from
                          # the kernel-launch amortization. 1 task per GPU.
export EXTRACT_BATCH=16   # H-cache extraction (heaviest VRAM stage)
export SAMPLE_BATCH=32    # conformer sampling

export INPUT_DIR=downstream_ft/0511_cc_audit/Clean
export SPLIT_DIR_ROOT=downstream_ft/0511_cc_audit/Split
export DATASETS_FILTER="RI,AcuteToxicity,BP,Hf_G"

# RUN_TAG stays 0511 — names the on-disk PKL/PT/H caches under
# data/0511_pt_*; changing it would invalidate Stage B.5's cached H
# (heaviest step). Today's outputs use 0513 for clean separation.
export RUN_TAG=0511
export OUT_ROOT=outputs/cv_0513
export LOG_DIR=/tmp/cv_0513
export WANDB=1
export WANDB_PROJECT=downstream_cv_0513
export SWANLAB_SYNC=1    # mirror every wandb run to swanlab (requires `pip install swanlab && swanlab login`)

# Bash arrays don't survive `exec bash` / subprocess — must `source`.
CKPT_DEFS=(
    "loqi_flow|data/ft_ckpts/loqi_flow.ckpt|scripts/conf/loqi/loqi_flow.yaml|0"
    "cold_early|data/ft_ckpts/thermo_flow_cold_early.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold.yaml|0"
    "cold_last|data/ft_ckpts/thermo_flow_cold_last.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold.yaml|0"
)

SAMPLING_MODES=(
    "standard|K8|8|10"
    "multistep|K12ms|12|4:10:7 8 9"
)

source scripts/run_cv.sh
