#!/usr/bin/env bash
# 0519 — warm-init baseline using the cold_combined ckpt.
#
# Identical to run_cv_0519_baseline_cold.sh except the CKPT_DEFS line
# flips init_thermo from 0 → 1, so load_thermo_head_into() picks the
# trained combined_heads.mp.* weights and warm-inits the downstream
# AtomMolMP head with them.  Dim auto-alignment (commit a86b9bc)
# silently bumps N_MP_LAYERS to match the ckpt's 6-layer combined head
# at runtime, so no manual head-config overrides are needed.
#
# Companion to run_cv_0519_baseline_cold.sh (cold init, same ckpt,
# same head, same 42 task) — gives the head-to-head warm-vs-cold
# comparison directly.
#
# Workload: 42 ds × 5 folds × cold_combined K8 = 210 jobs.
# Reuses RUN_TAG=0519 caches (H tensor / pkl / pt) from the baseline
# — only Stage C training differs.  Wall ≈ 1-2 h on 8 GPUs.
#
# Usage (8-GPU):
#   nohup bash scripts/run_cv_0519_warm_cold.sh \
#       > /tmp/cv_0519_warm.log 2>&1 & disown
#
# Usage (3 GPUs, leave GPU 0 free):
#   N_GPUS=3 CUDA_DEVICES=1,2,3 \
#       nohup bash scripts/run_cv_0519_warm_cold.sh \
#           > /tmp/cv_0519_warm.log 2>&1 & disown

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
export HEAD_TYPE="${HEAD_TYPE:-attention}"

export INPUT_DIR=downstream_data/cv_0519/Clean
export SPLIT_DIR_ROOT=downstream_data/cv_0519/Split
export SPLIT_KIND="${SPLIT_KIND:-random_cv5}"

# All 42 properties (same set + ordering as baseline_cold).
export DATASETS_FILTER="log_solubility_water_molL,BP_K,Lipophilicity_logD,Hf_gas_kJmol,Pvap_log10mmHg,ST_298K_mNm,fusion_T_K,Hf_liq_kJmol,dielectric_298K,Hvap_at_TB_kJmol,PPBR_pct,H_combus_kJmol,Tc_K,Pc_bar,Vc_cm3mol,Sf_gas_JmolK,Gf_gas_kJmol,ESOL_logS,visc_liq_298K_cP,omega,UEL_volpct,Cp_liq_298K,LEL_volpct,flash_point_K,density_liq_298K_gcm3,expand_coeff_liq_K-1,gyration_radius_A,k_liq_298K,S_gas_JmolK,RI_298K,CEP_PCE,Cp_gas_298K,log_solubility_water_ppm,Q_10ppmv_mgg,dipole_moment_D,log_Koc,Hfus_at_TF_kJmol,freesolv_dG_kcalmol,visc_gas_298K_uPas,log_Henry_atmmolfrac,autoignition_K,k_gas_298K"

# Shared RUN_TAG → reuses pkl/pt/H caches from the baseline run; only
# Stage C training differs.  Distinct OUT_ROOT + wandb group.
export RUN_TAG="${RUN_TAG:-0519}"
export OUT_ROOT="${OUT_ROOT:-outputs/cv_0519_warm_cold}"
export LOG_DIR="${LOG_DIR:-/tmp/cv_0519_warm_cold}"
export WANDB="${WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-downstream_cv_0519}"
export WANDB_GROUP="${WANDB_GROUP:-warm}"
export SWANLAB_SYNC="${SWANLAB_SYNC:-1}"

# Single ckpt, WARM-init (init_thermo=1 — the 4th `|`-separated field).
# Everything else is identical to baseline_cold.sh.  load_thermo_head_into
# picks combined_heads.mp.* via its candidate-prefix order; the auto-dim
# alignment in commit a86b9bc handles the 6-vs-4 layer mismatch silently.
CKPT_DEFS=(
    "cold_combined|data/ft_ckpts/thermo_flow_cold_combined.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold_combined.yaml|1"
)
SAMPLING_MODES=(
    "standard|K8|8|10"
)

source scripts/run_cv.sh
