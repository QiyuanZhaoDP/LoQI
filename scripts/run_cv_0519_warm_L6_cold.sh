#!/usr/bin/env bash
# 0519 F1 verification — warm-init with N_MP_LAYERS=6 (matches combined head).
#
# Hypothesis: warm-init has been underperforming cold-init because the
# downstream head default (N_MP_LAYERS=4) doesn't match the combined head
# trained with n_mp_layers=6.  `load_state_dict(strict=False)` silently
# loaded only the first 4 of 6 trained layers + a final block trained for
# layer-6 inputs sitting on layer-4 outputs → frankenstein init worse
# than random.
#
# This wrapper fixes that mismatch: N_MP_LAYERS=6 (matches combined head)
# + INIT_FROM_THERMO=1 (warm-init).  If warm-init suddenly beats or ties
# the L=4 cold baseline (outputs/cv_0519_baseline_cold) then the mismatch
# was the dominant cause — patch the wrapper defaults globally.  If it
# still loses, multi-task-to-single-task negative transfer dominates and
# warm-init is just not worth it.
#
# Schedule: targets GPUs 1,2,3 (leaving GPU 0 free for other work).
# N_GPUS=3 × TASKS_PER_GPU=4 = 12 slots.  Full 42-ds sweep ≈ 1-2 h.
#
# Cache discipline: RUN_TAG=0519 → shares Stage A/B/B.5 caches with
# baseline + atomwise + hybrid wrappers.  Only Stage C trains the new
# (warm + L=6) head.
#
# Usage:
#   nohup bash scripts/run_cv_0519_warm_L6_cold.sh \
#       > /tmp/cv_0519_warm_L6.log 2>&1 & disown

set -uo pipefail
cd "$(dirname "$0")/.."

python scripts/build_cv_0519.py >/dev/null || {
    echo "ERROR: build_cv_0519.py failed" >&2; exit 1; }

# ── GPU pool: 1,2,3 ───────────────────────────────────────────────────────
export N_GPUS="${N_GPUS:-3}"
export CUDA_DEVICES="${CUDA_DEVICES:-1,2,3}"
export TASKS_PER_GPU="${TASKS_PER_GPU:-4}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BATCH="${BATCH:-32}"
export EXTRACT_BATCH="${EXTRACT_BATCH:-16}"
export SAMPLE_BATCH="${SAMPLE_BATCH:-32}"
export DUMP_PREDS="${DUMP_PREDS:-1}"

# ── Variant: warm-init + L=6 (matches combined head exactly) ─────────────
export HEAD_TYPE="attention"
export N_MP_LAYERS=6          # ← critical: matches combined_head_args.n_mp_layers
export HEAD_HIDDEN=256        # matches combined_head_args.hidden
export MP_N_HEADS=4           # matches combined_head_args.mp_n_heads

export INPUT_DIR=downstream_data/cv_0519/Clean
export SPLIT_DIR_ROOT=downstream_data/cv_0519/Split
export SPLIT_KIND="${SPLIT_KIND:-random_cv5}"

export DATASETS_FILTER="log_solubility_water_molL,BP_K,Lipophilicity_logD,Hf_gas_kJmol,Pvap_log10mmHg,ST_298K_mNm,fusion_T_K,Hf_liq_kJmol,dielectric_298K,Hvap_at_TB_kJmol,PPBR_pct,H_combus_kJmol,Tc_K,Pc_bar,Vc_cm3mol,Sf_gas_JmolK,Gf_gas_kJmol,ESOL_logS,visc_liq_298K_cP,omega,UEL_volpct,Cp_liq_298K,LEL_volpct,flash_point_K,density_liq_298K_gcm3,expand_coeff_liq_K-1,gyration_radius_A,k_liq_298K,S_gas_JmolK,RI_298K,CEP_PCE,Cp_gas_298K,log_solubility_water_ppm,Q_10ppmv_mgg,dipole_moment_D,log_Koc,Hfus_at_TF_kJmol,freesolv_dG_kcalmol,visc_gas_298K_uPas,log_Henry_atmmolfrac,autoignition_K,k_gas_298K"

export RUN_TAG="${RUN_TAG:-0519}"
export OUT_ROOT="${OUT_ROOT:-outputs/cv_0519_warm_L6_cold}"
export LOG_DIR="${LOG_DIR:-/tmp/cv_0519_warm_L6_cold}"
export WANDB="${WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-downstream_cv_0519}"
export WANDB_GROUP="${WANDB_GROUP:-warm_L6}"
export SWANLAB_SYNC="${SWANLAB_SYNC:-1}"

# ── Single ckpt, WARM-init (init_thermo=1 — the 4th field) ──────────────
CKPT_DEFS=(
    "cold_combined|data/ft_ckpts/thermo_flow_cold_combined.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold_combined.yaml|1"
)
SAMPLING_MODES=(
    "standard|K8|8|10"
)

source scripts/run_cv.sh
