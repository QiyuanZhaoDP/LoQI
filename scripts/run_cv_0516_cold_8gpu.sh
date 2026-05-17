#!/usr/bin/env bash
# 0516 downstream CV — 8-GPU, COLD-init (init_thermo=0), K8, SCAFFOLD split.
#
# Companion to run_cv_0515_cold_8gpu.sh (random_cv5) — same backbone, same
# random head init, just scaffold_cv5 partitions instead of random_cv5.
#
# Workload: 43 ds × 1 ckpt × 1 K-mode = 43 CV jobs. Stage A + B + B.5
# caches are SHARED with 0515 runs (same RUN_TAG=0515, same ckpt). Only
# Stage C reruns.
#
# Usage:
#   nohup bash scripts/run_cv_0516_cold_8gpu.sh \
#       > /tmp/cv_0516_cold_8gpu.log 2>&1 & disown

set -uo pipefail
cd "$(dirname "$0")/.."

# ── 0. one-shot layout prep (idempotent) ───────────────────────────────────
echo "================================================================"
echo " 0516 layout prep — building Clean/ + Split/ (random + scaffold)"
echo "================================================================"
python scripts/prep_0515_layout.py --root downstream_ft/0515_final || {
    echo "ERROR: prep_0515_layout.py failed; aborting" >&2; exit 1; }
echo

# ── 1. run_cv.sh exports ───────────────────────────────────────────────────
export N_GPUS=8
export CUDA_DEVICES=0,1,2,3,4,5,6,7
export TASKS_PER_GPU=4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BATCH=32
export EXTRACT_BATCH=16
export SAMPLE_BATCH=32

# Inherits run_cv.sh defaults (warmup 0.2, clip 0.1, lr 1e-4, epochs 150/100, patience 50).

export INPUT_DIR=downstream_ft/0515_final/Clean
export SPLIT_DIR_ROOT=downstream_ft/0515_final/Split
export SPLIT_KIND=scaffold_cv5             # ★ scaffold split

export DATASETS_FILTER="log_solubility_water_molL,BP_K,Lipophilicity_logD,Hf_gas_kJmol,Pvap_log10mmHg,ST_298K_mNm,visc_liq_298K_cP,fusion_T_K,Hf_liq_kJmol,Hvap_at_TB_kJmol,dielectric_298K,PPBR_pct,H_combus_kJmol,Tc_K,omega,Pc_bar,Vc_cm3mol,Sf_gas_JmolK,Gf_gas_kJmol,ESOL_logS,UEL_volpct,Cp_liq_298K,LEL_volpct,flash_point_K,density_liq_298K_gcm3,expand_coeff_liq_K-1,kinematic_viscosity_298K_cSt,gyration_radius_A,k_liq_298K,S_gas_JmolK,RI_298K,CEP_PCE,Cp_gas_298K,log_solubility_water_ppm,Q_10ppmv_mgg,dipole_moment_D,log_Koc,Hfus_at_TF_kJmol,freesolv_dG_kcalmol,visc_gas_298K_uPas,log_Henry_atmmolfrac,autoignition_K,k_gas_298K"

export RUN_TAG=0515
export OUT_ROOT=outputs/cv_0516_cold        # ← scaffold + cold
export LOG_DIR=/tmp/cv_0516_cold
export WANDB=1
export WANDB_PROJECT=downstream_cv_0516_cold
export SWANLAB_SYNC=1

CKPT_DEFS=(
    "cold_combined|data/ft_ckpts/thermo_flow_cold_combined.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold_combined.yaml|0"
)

SAMPLING_MODES=(
    "standard|K8|8|10"
)

source scripts/run_cv.sh
