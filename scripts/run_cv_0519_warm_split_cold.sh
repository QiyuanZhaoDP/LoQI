#!/usr/bin/env bash
# 0519 — "correct" warm-init: use the SPLIT-HEAD cold ckpt
# (loqi_thermo_flow_cold) instead of the COMBINED-HEAD cold_combined ckpt.
#
# Why this is the right ckpt for warm-init:
#
#   loqi_thermo_flow_cold.yaml:
#     thermo_head_args:  {n_mp_layers: 4, mp_n_heads: 4, hidden: 256}
#     rdkit_head_args:   {n_mp_layers: 2, mp_n_heads: 4, hidden: 128}
#   → ckpt has separate dynamics.thermo_heads.mp.* (only 5 thermo targets)
#     and dynamics.rdkit_heads.mp.* (9 RDKit targets, lives in a separate
#     head — does NOT pollute the thermo features).
#
#   loqi_thermo_flow_cold_combined.yaml (what previous wrappers used):
#     combined_head_args: {n_mp_layers: 6, mp_n_heads: 4, hidden: 256}
#   → single 14-target head; finetuning inherits a "5-thermo + 9-RDKit
#     compromise" representation where 9/14 of capacity went to trivial
#     RDKit descriptors (logp, tpsa, ...) that don't help thermo tasks.
#
# load_thermo_head_into() prefers `thermo_heads.mp.*` candidates over
# `combined_heads.mp.*` candidates, so swapping the ckpt is the only
# change — no code edits needed.
#
# Head dim match:
#   split-head thermo_head_args = 4/4/256  ←→  default N_MP_LAYERS=4,
#   HEAD_HIDDEN=256, MP_N_HEADS=4.  NO auto-alignment needed (the
#   combined wrapper required 6/4/256, which auto-aligned at runtime).
#
# Scope: full 42 properties to be a clean apples-to-apples vs
# outputs/cv_0519_baseline_cold (cold-init, same head dims).
#
# Cache discipline: RUN_TAG=0519_warmsplit → SEPARATE pkl/pt dir from
# 0519 baseline because the backbone in `cold` ckpt may differ from
# `cold_combined` (different combined-vs-split heads sit on the same
# 12-layer backbone, but the backbone weights are otherwise the same;
# if you confirm the backbone is byte-identical, you can override to
# RUN_TAG=0519 to share caches).
#
# Usage (8-GPU box):
#   nohup bash scripts/run_cv_0519_warm_split_cold.sh \
#       > /tmp/cv_0519_warm_split.log 2>&1 & disown
#
# Usage (3-GPU subset, as in the warm_L6 wrapper):
#   N_GPUS=3 CUDA_DEVICES=1,2,3 \
#       bash scripts/run_cv_0519_warm_split_cold.sh

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

# ── Variant: attention head, dims match split-head's thermo_head_args ────
export HEAD_TYPE="attention"
export N_MP_LAYERS=4        # matches thermo_head_args.n_mp_layers in cold yaml
export HEAD_HIDDEN=256      # matches thermo_head_args.hidden
export MP_N_HEADS=4         # matches thermo_head_args.mp_n_heads

export INPUT_DIR=downstream_data/cv_0519/Clean
export SPLIT_DIR_ROOT=downstream_data/cv_0519/Split
export SPLIT_KIND="${SPLIT_KIND:-random_cv5}"

export DATASETS_FILTER="log_solubility_water_molL,BP_K,Lipophilicity_logD,Hf_gas_kJmol,Pvap_log10mmHg,ST_298K_mNm,fusion_T_K,Hf_liq_kJmol,dielectric_298K,Hvap_at_TB_kJmol,PPBR_pct,H_combus_kJmol,Tc_K,Pc_bar,Vc_cm3mol,Sf_gas_JmolK,Gf_gas_kJmol,ESOL_logS,visc_liq_298K_cP,omega,UEL_volpct,Cp_liq_298K,LEL_volpct,flash_point_K,density_liq_298K_gcm3,expand_coeff_liq_K-1,gyration_radius_A,k_liq_298K,S_gas_JmolK,RI_298K,CEP_PCE,Cp_gas_298K,log_solubility_water_ppm,Q_10ppmv_mgg,dipole_moment_D,log_Koc,Hfus_at_TF_kJmol,freesolv_dG_kcalmol,visc_gas_298K_uPas,log_Henry_atmmolfrac,autoignition_K,k_gas_298K"

# RUN_TAG is separate from 0519 to keep this experiment's H caches
# decoupled in case backbone differs.  Override to "0519" if you want
# to share H caches with the baseline (only safe if you confirm the
# two ckpts share the same backbone weights).
export RUN_TAG="${RUN_TAG:-0519_warmsplit}"
export OUT_ROOT="${OUT_ROOT:-outputs/cv_0519_warm_split_cold}"
export LOG_DIR="${LOG_DIR:-/tmp/cv_0519_warm_split_cold}"
export WANDB="${WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-downstream_cv_0519}"
export WANDB_GROUP="${WANDB_GROUP:-warm_split_4L}"
export SWANLAB_SYNC="${SWANLAB_SYNC:-1}"

# Split-head cold ckpt: warm-init (init_thermo=1) reads thermo_heads.mp.*
# (the trained 5-target standalone head) — no RDKit dilution.
#
# Adjust the ckpt PATH below to match where your split-head cold ckpt is
# saved.  By default it's expected at data/ft_ckpts/thermo_flow_cold.ckpt;
# if you trained from loqi_thermo_flow_cold.yaml it might live at
# outputs/loqi_thermo_flow_cold_v2/checkpoints/last.ckpt — point CKPT
# accordingly via env or by editing the line below.
CKPT_DEFS=(
    "cold_split|data/ft_ckpts/thermo_flow_cold.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold.yaml|1"
)
SAMPLING_MODES=(
    "standard|K8|8|10"
)

source scripts/run_cv.sh
