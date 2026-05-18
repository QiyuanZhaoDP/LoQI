#!/usr/bin/env bash
# 0518 quick re-CV — 4 datasets touched by today's upstream cleanups.
#
# Same cold_combined cold-init (init_thermo=0), K8-only config as
# scripts/run_cv_0515_cold_8gpu.sh, restricted to:
#
#   visc_liq_298K_cP    — 2099 → 1211 rows  (200.0 visc_LB fix + intra-tier
#                          drop + NO_STAR_FILTER removal)
#   ST_298K_mNm         — 2304 → 2264 rows  (intra-tier drop)
#   dielectric_298K     — 1386 → 1362 rows  (intra-tier drop)
#   Hvap_at_TB_kJmol    — 1430 → 1414 rows  (intra-tier drop)
#
# Cache discipline:
#   Uses a FRESH RUN_TAG (default 0518rev) so the new csvs trigger full
#   Stage A/B/B.5/C from scratch. This avoids racing with any 0515 caches
#   that may still hold the pre-fix data.  Wandb project is also separate
#   so you can A/B against the 0515 baseline without contamination.
#
# Workload: 4 ds × 1 ckpt × 1 K-mode = 4 CV jobs.
#   On 4 GPUs × TASKS_PER_GPU=4 → 1 round of Stage C; Stage A/B/B.5 are
#   dataset-parallel.  Wall: ~30-60 min for the four small ds.
#
# Usage (8-GPU box):
#   nohup bash scripts/run_cv_0518_revisit_cold.sh \
#       > /tmp/cv_0518_revisit_cold.log 2>&1 & disown
#
# Usage (smaller box, override gpu count + slots):
#   N_GPUS=4 CUDA_DEVICES=0,1,2,3 TASKS_PER_GPU=2 \
#       bash scripts/run_cv_0518_revisit_cold.sh

set -uo pipefail
cd "$(dirname "$0")/.."

# ── 0. one-shot layout prep (idempotent) ───────────────────────────────────
echo "================================================================"
echo " 0518 revisit layout prep — Clean/ + Split/ from updated csv_data"
echo "================================================================"
python scripts/prep_0515_layout.py --root downstream_ft/0515_final || {
    echo "ERROR: prep_0515_layout.py failed; aborting" >&2; exit 1; }
echo

# ── 1. run_cv.sh exports ───────────────────────────────────────────────────
export N_GPUS="${N_GPUS:-8}"
export CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
export TASKS_PER_GPU="${TASKS_PER_GPU:-4}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BATCH="${BATCH:-32}"
export EXTRACT_BATCH="${EXTRACT_BATCH:-16}"
export SAMPLE_BATCH="${SAMPLE_BATCH:-32}"

export INPUT_DIR=downstream_ft/0515_final/Clean
export SPLIT_DIR_ROOT=downstream_ft/0515_final/Split

# Only the four datasets whose data changed today.  Sorted descending by
# row count so the LPT scheduler picks the heaviest first.
export DATASETS_FILTER="ST_298K_mNm,Hvap_at_TB_kJmol,dielectric_298K,visc_liq_298K_cP"

# Fresh RUN_TAG → fresh pkl/pt caches under data/0518rev_pkl_cold_combined_k8/
# + data/0518rev_pt_cold_combined_k8/.  Override to RUN_TAG=0515 if you
# want to reuse 0515's caches AFTER manually deleting the stale four:
#   rm -f data/0515_pkl_cold_combined_k8/{ST_298K_mNm,Hvap_at_TB_kJmol,dielectric_298K,visc_liq_298K_cP}.pkl
#   rm -f data/0515_pt_cold_combined_k8/{ST_298K_mNm,Hvap_at_TB_kJmol,dielectric_298K,visc_liq_298K_cP}*.pt
export RUN_TAG="${RUN_TAG:-0518rev}"
export OUT_ROOT="${OUT_ROOT:-outputs/cv_0518_revisit_cold}"
export LOG_DIR="${LOG_DIR:-/tmp/cv_0518_revisit_cold}"
export WANDB="${WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-downstream_cv_0518_revisit}"
export SWANLAB_SYNC="${SWANLAB_SYNC:-1}"

# Single ckpt, COLD-init (init_thermo=0) — same as 0515_cold.
CKPT_DEFS=(
    "cold_combined|data/ft_ckpts/thermo_flow_cold_combined.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold_combined.yaml|0"
)

# K8 only (matches the 0515 ablation winner).
SAMPLING_MODES=(
    "standard|K8|8|10"
)

source scripts/run_cv.sh
