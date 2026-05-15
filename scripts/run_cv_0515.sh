#!/usr/bin/env bash
# 0515 downstream CV — full 43-property sweep on the new 0515_final dataset.
#
# Source: downstream_ft/0515_final/  (strict 12-element DB, 21,277 mols × 43 props)
#
# Layout note — the new dataset uses csv_data/<prop>/Split/random_cv5/...
# which differs from 0511_cc_audit's Split/<dataset>/random_cv5/... layout.
# A one-shot prep step lays out an audit-style Clean/ + Split/ tree on top
# of the existing files (CSVs are written, splits are symlinked). After
# the prep, run_cv.sh's existing INPUT_DIR / SPLIT_DIR_ROOT machinery
# works unchanged.
#
# Workload: 43 datasets × 1 ckpt (cold_combined) × 2 sampling = 86 CV jobs.
# With 8 GPUs × TASKS_PER_GPU=4 = 32 slots → ~3 rounds Stage C.
#
# Usage:
#   nohup bash scripts/run_cv_0515.sh > /tmp/cv_0515.log 2>&1 & disown

set -uo pipefail
cd "$(dirname "$0")/.."

# ── 0. one-shot layout prep (idempotent) ───────────────────────────────────
echo "================================================================"
echo " 0515 layout prep — building Clean/ + Split/ from csv_data + per_property"
echo "================================================================"
python scripts/prep_0515_layout.py --root downstream_ft/0515_final || {
    echo "ERROR: prep_0515_layout.py failed; aborting" >&2
    exit 1
}
echo

# ── 1. run_cv.sh exports ───────────────────────────────────────────────────
export N_GPUS=8
export CUDA_DEVICES=0,1,2,3,4,5,6,7
# 4 tasks/GPU — verified throughput-neutral by test_4x_one_gpu.sh. Each task
# pins H to GPU (~280 MB-2 GB depending on dataset); 4 × largest ≈ 8 GB / GPU,
# well under 80 GB.
export TASKS_PER_GPU=4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BATCH=32
export EXTRACT_BATCH=16   # cold_combined backbone is dim=384 — keep small for OOM safety
export SAMPLE_BATCH=32

export INPUT_DIR=downstream_ft/0515_final/Clean
export SPLIT_DIR_ROOT=downstream_ft/0515_final/Split

# All 43 properties — explicit so we get a deterministic queue and can
# trivially trim if any cause OOM. Sorted descending by training-set size
# (LPT scheduling — biggest first so they don't trail the tail).
export DATASETS_FILTER="log_solubility_water_molL,BP_K,Lipophilicity_logD,Hf_gas_kJmol,Pvap_log10mmHg,ST_298K_mNm,visc_liq_298K_cP,fusion_T_K,Hf_liq_kJmol,Hvap_at_TB_kJmol,dielectric_298K,PPBR_pct,H_combus_kJmol,Tc_K,omega,Pc_bar,Vc_cm3mol,Sf_gas_JmolK,Gf_gas_kJmol,ESOL_logS,UEL_volpct,Cp_liq_298K,LEL_volpct,flash_point_K,density_liq_298K_gcm3,expand_coeff_liq_K-1,kinematic_viscosity_298K_cSt,gyration_radius_A,k_liq_298K,S_gas_JmolK,RI_298K,CEP_PCE,Cp_gas_298K,log_solubility_water_ppm,Q_10ppmv_mgg,dipole_moment_D,log_Koc,Hfus_at_TF_kJmol,freesolv_dG_kcalmol,visc_gas_298K_uPas,log_Henry_atmmolfrac,autoignition_K,k_gas_298K"

# RUN_TAG=0515 — different from 0511 so we get fresh data caches under
# data/0515_pkl_cold_combined_* / data/0515_pt_cold_combined_*. (Reusing
# 0511's caches isn't possible anyway since the dataset names differ —
# BP_K vs BP, Cp_gas_298K vs Cp, etc.)
export RUN_TAG=0515
export OUT_ROOT=outputs/cv_0515
export LOG_DIR=/tmp/cv_0515
export WANDB=1
export WANDB_PROJECT=downstream_cv_0515
export SWANLAB_SYNC=1

# Single ckpt — the cold_combined unified 14-target head, warm-inited
# into the downstream SingleTargetHead.
CKPT_DEFS=(
    "cold_combined|data/ft_ckpts/thermo_flow_cold_combined.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold_combined.yaml|1"
)

# K=8 + K=12 multi-step.
SAMPLING_MODES=(
    "standard|K8|8|10"
    "multistep|K12ms|12|4:10:7 8 9"
)

source scripts/run_cv.sh
