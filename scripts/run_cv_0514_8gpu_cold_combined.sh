#!/usr/bin/env bash
# 0514 downstream CV — 8-GPU machine, just the cold_combined ckpt.
#
# Tests the unified 14-target (5 thermo + 9 rdkit) combined-head
# pretrained backbone across ALL 25 downstream datasets at once,
# replacing the prior 3-ckpt sweep (loqi_flow + cold_early + cold_last).
# Workload: 25 ds × 1 ckpt × 2 sampling modes = 50 CV jobs.
#
# Datasets (union of the previous 4-GPU + 8-GPU wrappers, 25 total):
#   Hf_L (4790), TPT (4461), Lipophilicity (4199), MP (3008), Cp (1958),
#   pKa (1492), de (1486), PPBR (1412), k (1329), Hf_C (1279), Vcp (1244),
#   ESOL (1115), Solubility_ethanol (925), CEP (882), AOH (692),
#   freesolv (641), BCF (595), Density (565), Clearance (492),
#   HalfLife (490), ST (304),
#   RI (812), AcuteToxicity (7328), BP (4664), Hf_G (5978)
#
# Skipped on purpose (extract OOM/timeout on prior runs):
#   Solubility_water (11,301), Tc (10,909), Pc (10,732)
#   — run as one-offs with EXTRACT_BATCH=4-8 if needed.
#
# Warm-init: init_thermo=1 → load_thermo_head_into picks up the
# combined_heads.mp.* prefix automatically (the first 5 outputs are
# the same thermo targets so weights transfer; final 14→1 Linear is
# random-init).
#
# Usage:
#   nohup bash scripts/run_cv_0514_8gpu_cold_combined.sh \
#       > /tmp/cv_0514_8gpu_cold_combined.log 2>&1 & disown

set -uo pipefail
cd "$(dirname "$0")/.."

export N_GPUS=8
export CUDA_DEVICES=0,1,2,3,4,5,6,7
# 4 tasks/GPU — verified by test_4x_one_gpu.sh on cold_early that the
# small-head workload at bs=32 keeps per-task throughput unchanged under
# 4-way concurrency. Memory per task at the heaviest dataset (Hf_G K=12
# H = ~2 GB) × 4 tasks = ~8-12 GB per GPU, well under 80 GB.
export TASKS_PER_GPU=4

# Stage B.5 extract uses the BACKBONE, which is heavy at K=12 ms with
# long-chain alkanes. bs=16 confirmed stable for all 25 datasets here.
# expandable_segments helps torch reclaim fragments between batches.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BATCH=32           # CV training — bs=32 + 4 tasks/GPU is the proven combo
export EXTRACT_BATCH=16   # H-cache extraction (heaviest VRAM stage)
export SAMPLE_BATCH=32    # conformer sampling

export INPUT_DIR=downstream_ft/0511_cc_audit/Clean
export SPLIT_DIR_ROOT=downstream_ft/0511_cc_audit/Split

# 25-dataset union — explicit so we don't accidentally pick up the
# 3 OOM-prone large sets if they exist in INPUT_DIR.
export DATASETS_FILTER="Hf_L,TPT,Lipophilicity,MP,Cp,pKa,de,PPBR,k,Hf_C,Vcp,ESOL,Solubility_ethanol,CEP,AOH,freesolv,BCF,Density,Clearance,HalfLife,ST,RI,AcuteToxicity,BP,Hf_G"

# RUN_TAG stays 0511 only for naming PKL/PT/H caches (Stage B+B.5 outputs).
# Today's CV outputs land under cv_0514 to keep them visually distinct
# from yesterday's cv_0513 (3-ckpt sweep). The cold_combined caches under
# data/0511_pkl_cold_combined_* / data/0511_pt_cold_combined_* will be
# fresh — different backbone weights than cold_early/cold_last so the
# H caches CANNOT be reused.
export RUN_TAG=0511
export OUT_ROOT=outputs/cv_0514
export LOG_DIR=/tmp/cv_0514
export WANDB=1
export WANDB_PROJECT=downstream_cv_0514
export SWANLAB_SYNC=1    # mirror wandb → swanlab (requires `pip install swanlab && swanlab login`)

# Bash arrays don't survive `exec bash` / subprocess — must `source`.
# Just cold_combined — init_thermo=1 warm-inits the downstream
# SingleTargetHead's AtomMolMP from dynamics.ema_model.combined_heads.mp.*
# (load_thermo_head_into already has the prefix candidates).
CKPT_DEFS=(
    "cold_combined|data/ft_ckpts/thermo_flow_cold_combined.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold_combined.yaml|1"
)

# Both K=8 (standard) and K=12 multi-step (4 traj × 3 snapshots).
SAMPLING_MODES=(
    "standard|K8|8|10"
    "multistep|K12ms|12|4:10:7 8 9"
)

source scripts/run_cv.sh
