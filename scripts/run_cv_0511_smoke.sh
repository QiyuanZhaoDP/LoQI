#!/usr/bin/env bash
# 0511 downstream CV — SMOKE TEST.
#
# End-to-end verification of all 3 stages (sample → extract-H → CV) on
# the smallest dataset (ST, 304 mols) with one ckpt and one K-mode.
# Reduced epochs so the whole pipeline finishes in ~5 minutes.
#
# Success criterion:
#   outputs/cv_0511_smoke/ST_cold_early_c_K8/cv_report.json exists and
#   contains finite mae_mean, rmse_mean, r2_mean.
#
# Outputs are namespaced to `cv_0511_smoke` so they don't collide with the
# real run's `cv_0511`. Use ONE GPU (CUDA_DEVICES=0) by default.
#
# Usage:
#   bash scripts/run_cv_0511_smoke.sh
#   # or pin a different GPU
#   CUDA_DEVICES=2 bash scripts/run_cv_0511_smoke.sh

set -uo pipefail
cd "$(dirname "$0")/.."

export N_GPUS=1
export CUDA_DEVICES=${CUDA_DEVICES:-0}
export TASKS_PER_GPU=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BATCH=32
export EXTRACT_BATCH=16
export SAMPLE_BATCH=32

export INPUT_DIR=downstream_ft/0511_cc_audit/Clean
export SPLIT_DIR_ROOT=downstream_ft/0511_cc_audit/Split
export DATASETS_FILTER="ST"          # 304 mols — smallest in audit output

# Smoke epochs: short enough to finish quickly, long enough to verify the
# loss decreases (cold-start head needs at least a few epochs to move).
export EPOCHS=10
export AUTO_EPOCHS=0                  # don't escalate to 200 for n<2000
export EPOCHS_LARGE=10
export EPOCHS_SMALL=10
export EARLY_STOP_PATIENCE=0          # don't stop early in smoke

# Force-rerun even if a stale cv_report.json from a previous smoke test
# is hanging around.
export FORCE_CV=1

export RUN_TAG=0511_smoke
export OUT_ROOT=outputs/cv_0511_smoke
export LOG_DIR=/tmp/cv_0511_smoke
export WANDB=0                        # no wandb noise during smoke

# Just one ckpt + K-mode. cold_early uses the smaller thermo_flow_cold ckpt
# (faster to load than loqi_flow). K8 standard sampling is faster than K12ms
# multistep (no snapshot logic), good enough to verify the pipeline.
# Bash arrays don't survive subprocess boundaries — `source` keeps them
# in the current shell so run_cv.sh sees these overrides instead of its
# own defaults.
CKPT_DEFS=(
    "cold_early|data/ft_ckpts/thermo_flow_cold_early.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold.yaml|0"
)
SAMPLING_MODES=(
    "standard|K8|8|10"
)

echo "============================================================"
echo "SMOKE TEST — 0511 downstream CV pipeline"
echo "  dataset:    ST (304 mols)"
echo "  ckpt:       cold_early"
echo "  K-mode:     K8 standard"
echo "  GPU:        $CUDA_DEVICES"
echo "  epochs:     $EPOCHS (5 folds = ~2-3 min total CV time)"
echo "  out:        $OUT_ROOT/"
echo "  expected:   $OUT_ROOT/ST_cold_early_c_K8/cv_report.json"
echo "============================================================"

source scripts/run_cv.sh

# ---- Verify ----
echo ""
echo "============================================================"
echo "SMOKE TEST RESULTS"
echo "============================================================"
_report="$OUT_ROOT/ST_cold_early_c_K8/cv_report.json"
if [[ -f "$_report" ]]; then
    echo "✅ cv_report.json produced at $_report"
    python3 - <<PY
import json, math
r = json.load(open("$_report"))
print(f"  mae_mean    = {r.get('mae_mean'):.4f} ± {r.get('mae_std'):.4f}")
print(f"  rmse_mean   = {r.get('rmse_mean'):.4f}")
print(f"  r2_mean     = {r.get('r2_mean'):.4f}")
print(f"  wall_seconds = {r.get('wall_seconds'):.1f}")
print(f"  n_molecules  = {r.get('n_molecules')}")
print(f"  n_folds      = {len(r.get('folds', []))}")
for k in ('mae_mean','rmse_mean','r2_mean'):
    v = r.get(k)
    assert v is not None and math.isfinite(v), f"  ❌ {k} is not finite"
print("✅ All metrics finite — pipeline is healthy.")
PY
else
    echo "❌ cv_report.json NOT produced — check logs:"
    echo "   - per-stage logs: $LOG_DIR/"
    echo "   - python output:  $OUT_ROOT/ST_cold_early_c_K8/cv.log"
    echo "   - prep step:      $OUT_ROOT/ST_cold_early_c_K8/prep.log"
    exit 1
fi
