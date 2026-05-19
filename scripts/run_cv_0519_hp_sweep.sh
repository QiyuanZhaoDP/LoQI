#!/usr/bin/env bash
# 0519 hyperparameter sweep — warmup_fraction × LR grid on the cv_0519
# baseline (cold_combined K=8 attention head, 42 datasets, random_cv5).
#
# Goal: identify whether the current run_cv.sh defaults
#   (LR=1e-4, WARMUP_FRACTION=0.20)
# are actually optimal for the cleaned 42-property dataset, or whether a
# different (warmup, lr) cell would systematically improve R² / MAE.
#
# Grid (4 cells by default; edit GRID below to widen).  Each cell runs
# its OWN full 42-dataset CV, all sharing RUN_TAG=0519 caches, distinct
# OUT_ROOT / wandb group:
#
#     cell                                       wandb_group
#     ----                                       -----------
#     warmup=0.05  LR=1e-4   (sharp warmup, default LR)   hp_w05_lr1e4
#     warmup=0.20  LR=3e-4   (long warmup, hi LR)         hp_w20_lr3e4
#     warmup=0.05  LR=3e-4   (sharp warmup, hi LR)        hp_w05_lr3e4
#     warmup=0.10  LR=1e-4   (medium warmup)              hp_w10_lr1e4
#
# Baseline (warmup=0.20, LR=1e-4) is NOT re-run — use the existing
# outputs/cv_0519_baseline_cold/ for comparison.
#
# Workload per cell: 42 ds × 1 ckpt × 1 K = 42 jobs ≈ 30-60 min.
# Total sweep wall: 4 cells × 1 round ≈ 2-4 h sequential, or run cells
# in parallel by passing a subset of CUDA_DEVICES.
#
# Usage:
#   nohup bash scripts/run_cv_0519_hp_sweep.sh \
#       > /tmp/cv_0519_hp_sweep.log 2>&1 & disown
#
# To run just one cell (parallelize across machines):
#   HP_CELL_INDEX=2 bash scripts/run_cv_0519_hp_sweep.sh

set -uo pipefail
cd "$(dirname "$0")/.."

python scripts/build_cv_0519.py >/dev/null || {
    echo "ERROR: build_cv_0519.py failed" >&2; exit 1; }

# ── Grid definition.  Each row: "warmup_fraction|LR|wandb_group_suffix" ────
GRID=(
    "0.05|1e-4|hp_w05_lr1e4"
    "0.20|3e-4|hp_w20_lr3e4"
    "0.05|3e-4|hp_w05_lr3e4"
    "0.10|1e-4|hp_w10_lr1e4"
)

# Filter to one cell (1-indexed) for parallelization across machines.
if [[ -n "${HP_CELL_INDEX:-}" ]]; then
    GRID=("${GRID[${HP_CELL_INDEX}-1]}")
    echo "Running single cell: ${GRID[0]}"
fi

# ── Fixed exports (shared across cells) ────────────────────────────────────
export N_GPUS="${N_GPUS:-8}"
export CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
export TASKS_PER_GPU="${TASKS_PER_GPU:-4}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BATCH="${BATCH:-32}"
export EXTRACT_BATCH="${EXTRACT_BATCH:-16}"
export SAMPLE_BATCH="${SAMPLE_BATCH:-32}"
export DUMP_PREDS="${DUMP_PREDS:-1}"
export HEAD_TYPE="${HEAD_TYPE:-attention}"

export INPUT_DIR=downstream_data/cv_0519/Clean
export SPLIT_DIR_ROOT=downstream_data/cv_0519/Split
export SPLIT_KIND="${SPLIT_KIND:-random_cv5}"
export RUN_TAG="${RUN_TAG:-0519}"

export DATASETS_FILTER="log_solubility_water_molL,BP_K,Lipophilicity_logD,Hf_gas_kJmol,Pvap_log10mmHg,ST_298K_mNm,fusion_T_K,Hf_liq_kJmol,dielectric_298K,Hvap_at_TB_kJmol,PPBR_pct,H_combus_kJmol,Tc_K,Pc_bar,Vc_cm3mol,Sf_gas_JmolK,Gf_gas_kJmol,ESOL_logS,visc_liq_298K_cP,omega,UEL_volpct,Cp_liq_298K,LEL_volpct,flash_point_K,density_liq_298K_gcm3,expand_coeff_liq_K-1,gyration_radius_A,k_liq_298K,S_gas_JmolK,RI_298K,CEP_PCE,Cp_gas_298K,log_solubility_water_ppm,Q_10ppmv_mgg,dipole_moment_D,log_Koc,Hfus_at_TF_kJmol,freesolv_dG_kcalmol,visc_gas_298K_uPas,log_Henry_atmmolfrac,autoignition_K,k_gas_298K"

export WANDB="${WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-downstream_cv_0519_hp}"
export SWANLAB_SYNC="${SWANLAB_SYNC:-1}"

CKPT_DEFS=(
    "cold_combined|data/ft_ckpts/thermo_flow_cold_combined.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold_combined.yaml|0"
)
SAMPLING_MODES=(
    "standard|K8|8|10"
)

# ── Loop over the grid ────────────────────────────────────────────────────
for cell in "${GRID[@]}"; do
    IFS='|' read -r _wu _lr _tag <<< "$cell"
    echo
    echo "================================================================"
    echo " HP cell: WARMUP=${_wu}  LR=${_lr}    start=$(date +'%F %T')"
    echo "================================================================"
    export WARMUP_FRACTION="${_wu}"
    export LR="${_lr}"
    export OUT_ROOT="outputs/cv_0519_${_tag}"
    export LOG_DIR="/tmp/cv_0519_${_tag}"
    export WANDB_GROUP="${_tag}"
    mkdir -p "$OUT_ROOT" "$LOG_DIR"
    (
        source scripts/run_cv.sh
    )
    echo "[$(date +'%F %T')] cell ${_tag} done."
done

echo
echo "================================================================"
echo " HP sweep complete @ $(date +'%F %T')"
echo " Compare via: python scripts/summarize_cv_reports.py \\"
echo "     outputs/cv_0519_baseline_cold  \\         (default HP)"
for cell in "${GRID[@]}"; do
    IFS='|' read -r _ _ _tag <<< "$cell"
    echo "     outputs/cv_0519_${_tag}  \\"
done
echo "================================================================"
