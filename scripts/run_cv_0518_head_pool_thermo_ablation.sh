#!/usr/bin/env bash
# 0518 head-pool ablation on size-extensive thermo properties.
#
# Two head types are tested on the same backbone (cold_combined K=8) and
# the same balanced scaffold split (scaffold_balanced_cv5):
#
#   attention : SingleTargetHead = AtomMolMP attention-weighted MEAN pool
#               (size-invariant; default; warm-init from thermo ckpt)
#   atomwise  : AtomwiseHead     = deeper residual per-atom MLP + scatter_SUM
#               (size-extensive; physically motivated for extensive thermo
#               targets like H_combus, Hf, Vc; NequIP/PaiNN style)
#
# The earlier SumPoolHead variant was removed on 2026-05-19 — it was
# dominated by AtomwiseHead on every metric where the two differed, and
# never beat `attention` on MAE.  See the original ablation log for
# context (3 head types × 10 ds, kept here as historical baseline).
#
# Each head trains on the SAME cached H features (Stage A/B/B.5 skipped if
# data/0515_pt_cold_combined_K8/<ds>_H.pt exists), so the only difference
# between the two runs is the head architecture + pooling scheme.
#
# Datasets: 10 size-extensive thermo / structural ds (TH + HC + Vc).
# Total work: 2 head_types × 10 ds × 5 folds = 100 jobs.
# On 8 GPUs × 4 tasks = 32 slots, each head-type sweep ≈ 30-45 min.
# Total wall ≈ 60-90 min sequential.
#
# Output: outputs/cv_0518_head_pool/<head_type>/<ds>_<cfg>/cv_report.json
#
# Usage:
#   nohup bash scripts/run_cv_0518_head_pool_thermo_ablation.sh \
#       > /tmp/cv_0518_head_pool.log 2>&1 & disown
#
# To run a single head_type (e.g. to parallelize across machines):
#   HEAD_TYPE=atomwise bash scripts/run_cv_0518_head_pool_thermo_ablation.sh

set -uo pipefail
cd "$(dirname "$0")/.."

# ── 0. layout prep (idempotent) ────────────────────────────────────────────
python scripts/prep_0515_layout.py --root downstream_ft/0515_final >/dev/null || {
    echo "ERROR: prep_0515_layout.py failed" >&2; exit 1; }

# Verify balanced split exists
for ds_dir in downstream_ft/0515_final/Split/*/; do
    if [[ ! -d "${ds_dir}scaffold_balanced_cv5" ]]; then
        echo "ERROR: missing ${ds_dir}scaffold_balanced_cv5/" >&2
        echo "Run: python scripts/balanced_scaffold_split.py" >&2; exit 1
    fi
done

# ── 1. fixed exports (shared across all head types) ────────────────────────
export N_GPUS=8
export CUDA_DEVICES=0,1,2,3,4,5,6,7
export TASKS_PER_GPU=4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BATCH=32
export EXTRACT_BATCH=16
export SAMPLE_BATCH=32

export INPUT_DIR=downstream_ft/0515_final/Clean
export SPLIT_DIR_ROOT=downstream_ft/0515_final/Split
export SPLIT_KIND=scaffold_balanced_cv5

# 10 size-extensive ds:
#   TH group (6): H_combus, Sf_gas, Gf_gas, S_gas, Hf_liq, Hf_gas
#   HC group (3): Cp_gas, gyration_radius, Cp_liq
#   PT group (1): Vc_cm3mol (extensive volume)
export DATASETS_FILTER="H_combus_kJmol,Sf_gas_JmolK,Gf_gas_kJmol,S_gas_JmolK,Hf_liq_kJmol,Hf_gas_kJmol,Cp_gas_298K,gyration_radius_A,Cp_liq_298K,Vc_cm3mol"

export RUN_TAG=0515                       # reuse cached H
export WANDB=1
export WANDB_PROJECT=downstream_cv_0518_head_pool
export SWANLAB_SYNC=1

CKPT_DEFS=(
    "cold_combined|data/ft_ckpts/thermo_flow_cold_combined.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold_combined.yaml|0"
)
SAMPLING_MODES=(
    "standard|K8|8|10"
)

# ── 2. loop over head types ────────────────────────────────────────────────
# If HEAD_TYPE is set in the environment, only run that one variant (useful
# for splitting across machines). Otherwise run all three sequentially.
if [[ -n "${HEAD_TYPE:-}" ]]; then
    HEAD_TYPES=("$HEAD_TYPE")
else
    HEAD_TYPES=(attention atomwise)
fi

for ht in "${HEAD_TYPES[@]}"; do
    echo
    echo "================================================================"
    echo " head_type = ${ht}    start = $(date +'%F %T')"
    echo "================================================================"
    export HEAD_TYPE="$ht"
    export OUT_ROOT="outputs/cv_0518_head_pool/${ht}"
    export LOG_DIR="/tmp/cv_0518_head_pool/${ht}"
    mkdir -p "$OUT_ROOT" "$LOG_DIR"
    # WANDB group per head_type for easy filtering
    export WANDB_GROUP="head_${ht}"
    (
        source scripts/run_cv.sh
    )
    echo "[$(date +'%F %T')] head_type=${ht} sweep done."
done

echo
echo "================================================================"
echo " ALL head_pool ablation sweeps complete @ $(date +'%F %T')"
echo " Results: outputs/cv_0518_head_pool/<head_type>/<ds>_<cfg>/cv_report.json"
echo "================================================================"
