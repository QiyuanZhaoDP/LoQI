#!/usr/bin/env bash
# 0519 baseline CV — full 42-property sweep on the cleaned dataset.
#
# Dataset:  downstream_data/cv_0519/  (built by scripts/build_cv_0519.py)
#   42 properties × 20,841 unique molecules × 62,925 cells
#   random_cv5 splits only
#   Post-2026-05-19 data cleanup:
#     - dielectric_298K +73 PCCP-trusted secondary_single rows
#     - visc_liq_298K_cP capped at 50 mPa·s (23 high-tail rows removed)
#     - visc_liq_298K_cP_manual dropped (auto pipeline is canonical)
#
# Backbone: cold_combined  (data/ft_ckpts/thermo_flow_cold_combined.ckpt)
# Init:     cold (init_thermo=0; downstream head random-init)
# Pooling:  attention  (default SingleTargetHead; AtomMolMP attention-mean)
# Sampling: K=8 standard
#
# Cache discipline:
#   RUN_TAG=0519 → fresh pkl/pt caches under data/0519_pkl_cold_combined_k8/
#   + data/0519_pt_cold_combined_k8/ because the underlying csvs changed
#   today (dielectric / visc_liq).  Reuse caches across both warm and cold
#   variants if you launch more 0519 wrappers later.
#
# Workload: 42 ds × 1 ckpt × 1 K-mode = 42 jobs.
#   On 8 GPUs × TASKS_PER_GPU=4 = 32 slots → ~1.5 rounds Stage C.
#   Plus Stage A/B (sampling) + Stage B.5 (H extraction).
#   Wall: ~2-4 h end-to-end on a fresh box.
#
# Usage (8-GPU box):
#   nohup bash scripts/run_cv_0519_baseline_cold.sh \
#       > /tmp/cv_0519_baseline_cold.log 2>&1 & disown
#
# Usage (4-GPU box):
#   N_GPUS=4 CUDA_DEVICES=0,1,2,3 TASKS_PER_GPU=4 \
#       bash scripts/run_cv_0519_baseline_cold.sh

set -uo pipefail
cd "$(dirname "$0")/.."

# ── 0. one-shot dataset materialization (idempotent) ───────────────────────
echo "================================================================"
echo " 0519 dataset assembly — downstream_data/cv_0519/ from 0515_final"
echo "================================================================"
python scripts/build_cv_0519.py \
    --src downstream_ft/0515_final \
    --dst downstream_data/cv_0519 || {
    echo "ERROR: build_cv_0519.py failed; aborting" >&2; exit 1; }
echo

# ── 1. run_cv.sh exports ───────────────────────────────────────────────────
export N_GPUS="${N_GPUS:-8}"
export CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
export TASKS_PER_GPU="${TASKS_PER_GPU:-4}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BATCH="${BATCH:-32}"
export EXTRACT_BATCH="${EXTRACT_BATCH:-16}"
export SAMPLE_BATCH="${SAMPLE_BATCH:-32}"

# Per-sample predictions for downstream scan_label_noise.py if desired.
export DUMP_PREDS="${DUMP_PREDS:-1}"

export INPUT_DIR=downstream_data/cv_0519/Clean
export SPLIT_DIR_ROOT=downstream_data/cv_0519/Split
export SPLIT_KIND="${SPLIT_KIND:-random_cv5}"

# All 42 properties.  Sorted descending by row count (rough LPT scheduling).
export DATASETS_FILTER="log_solubility_water_molL,BP_K,Lipophilicity_logD,Hf_gas_kJmol,Pvap_log10mmHg,ST_298K_mNm,fusion_T_K,Hf_liq_kJmol,dielectric_298K,Hvap_at_TB_kJmol,PPBR_pct,H_combus_kJmol,Tc_K,Pc_bar,Vc_cm3mol,Sf_gas_JmolK,Gf_gas_kJmol,ESOL_logS,visc_liq_298K_cP,omega,UEL_volpct,Cp_liq_298K,LEL_volpct,flash_point_K,density_liq_298K_gcm3,expand_coeff_liq_K-1,gyration_radius_A,k_liq_298K,S_gas_JmolK,RI_298K,CEP_PCE,Cp_gas_298K,log_solubility_water_ppm,Q_10ppmv_mgg,dipole_moment_D,log_Koc,Hfus_at_TF_kJmol,freesolv_dG_kcalmol,visc_gas_298K_uPas,log_Henry_atmmolfrac,autoignition_K,k_gas_298K"

# Fresh RUN_TAG → fresh caches because data changed today.  Override to
# RUN_TAG=0515 only if you want to reuse old caches AFTER manually
# invalidating dielectric + visc_liq pkl/pt files.
export RUN_TAG="${RUN_TAG:-0519}"
export OUT_ROOT="${OUT_ROOT:-outputs/cv_0519_baseline_cold}"
export LOG_DIR="${LOG_DIR:-/tmp/cv_0519_baseline_cold}"
export WANDB="${WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-downstream_cv_0519}"
export SWANLAB_SYNC="${SWANLAB_SYNC:-1}"

# Single ckpt, COLD-init (init_thermo=0).
CKPT_DEFS=(
    "cold_combined|data/ft_ckpts/thermo_flow_cold_combined.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold_combined.yaml|0"
)

# K8 only (winning variant from prior 0515 ablation).
SAMPLING_MODES=(
    "standard|K8|8|10"
)

source scripts/run_cv.sh
