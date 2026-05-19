#!/usr/bin/env bash
# 0519 SUBSET CV on the new scaffold_hybrid_cv5 (Lloyd→rebal→swap) splits.
#
# Purpose: pilot test the hybrid OOD split on a representative subset of
# 8 datasets — one from each physics group — to see whether the +5% OOD
# distance (median 0.60→0.63) actually changes the W/L picture vs
# scaffold_balanced_cv5 / scaffold_diverse_cv5.
#
# Subset (1 per group, 8 ds total):
#   PT  BP_K                  (largest, distinctive scaffolds)
#   TH  Hf_gas_kJmol          (thermo; TG lost to suiren here)
#   HC  Cp_gas_298K           (heat capacity)
#   TR  density_liq_298K_gcm3 (transport)
#   SL  ESOL_logS             (solubility)
#   EL  dielectric_298K       (TG's known weak spot)
#   SF  flash_point_K         (safety)
#   BX  PPBR_pct              (bio/pharma)
#
# Workload: 8 ds × 5 folds × cold_combined K8 = 40 CV jobs.
# H cache REUSED from RUN_TAG=0515 (same molecules, only fold partitioning
# changes). With 8 GPUs × 4 tasks each = 32 slots, head training only:
# total wall ~15-30 min.
#
# Usage:
#   nohup bash scripts/run_cv_0519_hybrid_subset_8gpu.sh \
#       > /tmp/cv_0519_hybrid_subset.log 2>&1 & disown

set -uo pipefail
cd "$(dirname "$0")/.."

# ── 0. layout prep + verify hybrid split exists ────────────────────────────
python scripts/prep_0515_layout.py --root downstream_ft/0515_final >/dev/null || {
    echo "ERROR: prep_0515_layout.py failed" >&2; exit 1; }

_missing=0
for ds in BP_K Hf_gas_kJmol Cp_gas_298K density_liq_298K_gcm3 \
          ESOL_logS dielectric_298K flash_point_K PPBR_pct; do
    if [[ ! -d "downstream_ft/0515_final/Split/$ds/scaffold_hybrid_cv5" ]]; then
        echo "  MISSING: $ds/scaffold_hybrid_cv5/"; _missing=$((_missing+1))
    fi
done
if (( _missing > 0 )); then
    echo "ERROR: $_missing subset ds missing scaffold_hybrid_cv5/."
    echo "Run: python scripts/maxmin_scaffold_split.py --sim 0.30 --balance 1.25 \\"
    echo "         --algorithm hybrid --out-subdir scaffold_hybrid_cv5" >&2
    exit 1
fi
echo "  scaffold_hybrid_cv5/ present for all 8 subset ds ✓"

# ── 1. run_cv.sh exports ───────────────────────────────────────────────────
export N_GPUS=8
export CUDA_DEVICES=0,1,2,3,4,5,6,7
export TASKS_PER_GPU=4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BATCH=32
export EXTRACT_BATCH=16
export SAMPLE_BATCH=32

# Inherit run_cv.sh defaults (warmup 0.2, clip 0.1, lr 1e-4, epochs 150/100,
# patience 50). DUMP_PREDS=1 ensures per-sample preds for pooled-metric
# computation by scripts/aggregate_benchmark.py downstream.
export DUMP_PREDS=1

export INPUT_DIR=downstream_ft/0515_final/Clean
export SPLIT_DIR_ROOT=downstream_ft/0515_final/Split
export SPLIT_KIND=scaffold_hybrid_cv5    # ★ Lloyd→rebal→swap split

# 8 subset ds — one per physics group
export DATASETS_FILTER="BP_K,Hf_gas_kJmol,Cp_gas_298K,density_liq_298K_gcm3,ESOL_logS,dielectric_298K,flash_point_K,PPBR_pct"

export RUN_TAG=0515                       # reuse cached H
export OUT_ROOT=outputs/cv_0519_hybrid_subset
export LOG_DIR=/tmp/cv_0519_hybrid_subset
export WANDB=1
export WANDB_PROJECT=downstream_cv_0519_hybrid_subset
export SWANLAB_SYNC=1

CKPT_DEFS=(
    "cold_combined|data/ft_ckpts/thermo_flow_cold_combined.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold_combined.yaml|0"
)
SAMPLING_MODES=(
    "standard|K8|8|10"
)

source scripts/run_cv.sh

# ── 2. Post-run aggregation hint ───────────────────────────────────────────
echo
echo "================================================================"
echo " CV done. Aggregate results with:"
echo "   python scripts/aggregate_benchmark.py \\"
echo "       --runs-root outputs/cv_0519_hybrid_subset \\"
echo "       --out /tmp/cv_0519_hybrid_subset_summary.csv"
echo
echo " To compare against balanced/diverse (if those output dirs exist):"
echo "   python scripts/aggregate_benchmark.py \\"
echo "       --runs-root outputs/cv_0518_cold_balanced \\"
echo "                   outputs/cv_0518_cold_diverse \\"
echo "                   outputs/cv_0519_hybrid_subset \\"
echo "       --out /tmp/cv_split_compare.csv"
echo "================================================================"
