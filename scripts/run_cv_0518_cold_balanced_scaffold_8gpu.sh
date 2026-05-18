#!/usr/bin/env bash
# 0518 downstream CV — COLD-init (init_thermo=0), K8, BALANCED SCAFFOLD split.
#
# Companion to run_cv_0516_cold_8gpu.sh (old scaffold_cv5) — same backbone,
# same head init, same K=8, but uses the new scaffold_balanced_cv5 splits
# (LPT bin-packing on Bemis-Murcko clusters; ~1.0-1.2x max/min test ratio
# vs 5-17x for the old scaffold_cv5).
#
# H CACHE REUSE:
#   RUN_TAG=0515 — same tag as run_cv_0515_cold_8gpu.sh, so per-ds H caches
#   at data/0515_pt_cold_combined_K8/<DS>_H.pt are REUSED. New split partitions
#   the same molecules, so cached H is mathematically identical. Stage A/B
#   (build .pt) and Stage C (extract H) are skipped if caches exist.
#
#   To force re-extract:  rm -r data/0515_pt_cold_combined_K8/  (or use new RUN_TAG)
#
# Workload: 43 ds × 1 ckpt × 1 K-mode = 43 CV jobs, head-only training.
# Should be ~30 min/ds × 43 ds / (8 gpu × 4 tasks) ≈ 1 hour wall-time.
#
# Usage:
#   nohup bash scripts/run_cv_0518_cold_balanced_scaffold_8gpu.sh \
#       > /tmp/cv_0518_cold_balanced.log 2>&1 & disown

set -uo pipefail
cd "$(dirname "$0")/.."

# ── 0. one-shot layout prep (idempotent) ───────────────────────────────────
echo "================================================================"
echo " 0518 layout prep — building Clean/ + Split/ (assumes scaffold_balanced_cv5 already generated)"
echo "================================================================"
python scripts/prep_0515_layout.py --root downstream_ft/0515_final || {
    echo "ERROR: prep_0515_layout.py failed; aborting" >&2; exit 1; }

# Verify balanced split files actually exist
_missing_balanced=0
for ds_dir in downstream_ft/0515_final/Split/*/; do
    if [[ ! -d "${ds_dir}scaffold_balanced_cv5" ]]; then
        echo "  MISSING: ${ds_dir}scaffold_balanced_cv5/"
        _missing_balanced=$((_missing_balanced+1))
    fi
done
if (( _missing_balanced > 0 )); then
    echo "ERROR: $_missing_balanced datasets are missing scaffold_balanced_cv5/."
    echo "Run: python /tmp/balanced_scaffold_split.py" >&2
    exit 1
fi
echo "  scaffold_balanced_cv5/ present for all datasets ✓"
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
export SPLIT_KIND=scaffold_balanced_cv5    # ★ NEW: balanced scaffold split

# Same 43 datasets as run_cv_0516 (LPT-ordered by training-set size).
export DATASETS_FILTER="log_solubility_water_molL,BP_K,Lipophilicity_logD,Hf_gas_kJmol,Pvap_log10mmHg,ST_298K_mNm,visc_liq_298K_cP,fusion_T_K,Hf_liq_kJmol,Hvap_at_TB_kJmol,dielectric_298K,PPBR_pct,H_combus_kJmol,Tc_K,omega,Pc_bar,Vc_cm3mol,Sf_gas_JmolK,Gf_gas_kJmol,ESOL_logS,UEL_volpct,Cp_liq_298K,LEL_volpct,flash_point_K,density_liq_298K_gcm3,expand_coeff_liq_K-1,gyration_radius_A,k_liq_298K,S_gas_JmolK,RI_298K,CEP_PCE,Cp_gas_298K,log_solubility_water_ppm,Q_10ppmv_mgg,dipole_moment_D,log_Koc,Hfus_at_TF_kJmol,freesolv_dG_kcalmol,visc_gas_298K_uPas,log_Henry_atmmolfrac,autoignition_K,k_gas_298K"

export RUN_TAG=0515                         # ★ REUSE: same H caches as 0515 / 0516
export OUT_ROOT=outputs/cv_0518_cold_balanced
export LOG_DIR=/tmp/cv_0518_cold_balanced
export WANDB=1
export WANDB_PROJECT=downstream_cv_0518_cold_balanced
export SWANLAB_SYNC=1

CKPT_DEFS=(
    "cold_combined|data/ft_ckpts/thermo_flow_cold_combined.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold_combined.yaml|0"
)

SAMPLING_MODES=(
    "standard|K8|8|10"
)

source scripts/run_cv.sh
