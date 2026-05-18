#!/usr/bin/env bash
# 0515 downstream CV — 8-GPU, WARM-init (init_thermo=1), K8 only.
#
# Companion: scripts/run_cv_0515_cold_8gpu.sh (run on the other 8-GPU box).
# Workload split: each machine runs the same 43 datasets with the same
# cold_combined backbone, but the downstream head is warm-inited from the
# combined ckpt here vs random-inited there. Outputs land in distinct
# OUT_ROOT directories so the two runs never collide.
#
# K-mode: K8 only — confirmed by the cold-vs-warm + K8-vs-K12ms ablation
# that K8 wins ~60 % of pairs at half the wall time.
#
# Requires the warm-init fixes already on main:
#   a86b9bc  accept combined_head_args in --init-head-from-thermo dim check
#   bb75d1d  add newer-Lightning EMA prefix variants to warm-init candidates
# `git log --oneline | head` should show both commits.
#
# Workload: 43 ds × 1 ckpt × 1 K-mode = 43 CV jobs.
# With 8 GPUs × TASKS_PER_GPU=4 = 32 slots → ~1.5 rounds Stage C.
# Plus Stage A/B (conformer sampling) and Stage B.5 (H extraction) for K8.
# Total: ~2-4 h end-to-end on a fresh box.
#
# Usage:
#   nohup bash scripts/run_cv_0515_warm_8gpu.sh \
#       > /tmp/cv_0515_warm_8gpu.log 2>&1 & disown

set -uo pipefail
cd "$(dirname "$0")/.."

# ── 0. one-shot layout prep (idempotent) ───────────────────────────────────
echo "================================================================"
echo " 0515 layout prep — building Clean/ + Split/ from csv_data + per_property"
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
export EXTRACT_BATCH=16   # cold_combined backbone dim=384 — keep small for OOM safety
export SAMPLE_BATCH=32

export INPUT_DIR=downstream_ft/0515_final/Clean
export SPLIT_DIR_ROOT=downstream_ft/0515_final/Split

# All 43 properties, sorted descending by training-set size (LPT scheduling).
export DATASETS_FILTER="log_solubility_water_molL,BP_K,Lipophilicity_logD,Hf_gas_kJmol,Pvap_log10mmHg,ST_298K_mNm,visc_liq_298K_cP,fusion_T_K,Hf_liq_kJmol,Hvap_at_TB_kJmol,dielectric_298K,PPBR_pct,H_combus_kJmol,Tc_K,omega,Pc_bar,Vc_cm3mol,Sf_gas_JmolK,Gf_gas_kJmol,ESOL_logS,UEL_volpct,Cp_liq_298K,LEL_volpct,flash_point_K,density_liq_298K_gcm3,expand_coeff_liq_K-1,gyration_radius_A,k_liq_298K,S_gas_JmolK,RI_298K,CEP_PCE,Cp_gas_298K,log_solubility_water_ppm,Q_10ppmv_mgg,dipole_moment_D,log_Koc,Hfus_at_TF_kJmol,freesolv_dG_kcalmol,visc_gas_298K_uPas,log_Henry_atmmolfrac,autoignition_K,k_gas_298K"

# RUN_TAG=0515 names the on-disk caches: data/0515_pkl_cold_combined_k8/,
# data/0515_pt_cold_combined_k8/. The companion cold-init wrapper uses the
# SAME RUN_TAG so if they share a filesystem they share Stage B + B.5
# caches — the H tensor extracted from the backbone is identical regardless
# of head init.
export RUN_TAG=0515
export OUT_ROOT=outputs/cv_0515_warm        # ← warm-specific
export LOG_DIR=/tmp/cv_0515_warm
export WANDB=1
export WANDB_PROJECT=downstream_cv_0515_warm
export SWANLAB_SYNC=1

# Single ckpt, WARM-init (init_thermo=1). Downstream SingleTargetHead's
# AtomMolMP is warm-inited from dynamics.ema_model.module.combined_heads.mp.*
# (matched by load_thermo_head_into after bb75d1d). The final 14→1 Linear
# stays random-init because output dim differs from the trained 14.
CKPT_DEFS=(
    "cold_combined|data/ft_ckpts/thermo_flow_cold_combined.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold_combined.yaml|1"
)

# K8 ONLY — confirmed superior to K12ms in the ablation (mean +6 % K12 penalty,
# K8 wins 20/33 pairs holding init fixed).
SAMPLING_MODES=(
    "standard|K8|8|10"
)

source scripts/run_cv.sh
