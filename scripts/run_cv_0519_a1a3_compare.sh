#!/usr/bin/env bash
# 0519 — fast A1 vs A3 vs baseline head comparison.
#
# Runs three head variants on the SAME 12 size-extensive properties and
# the SAME random_cv5 splits, then auto-generates the comparison tables:
#
#   attention   →  outputs/cv_0519_baseline_ext_cold/
#   atomwise    →  outputs/cv_0519_atomwise_cold/      (A1 partner)
#   hybrid      →  outputs/cv_0519_hybrid_cold/        (A3)
#
# Schedule on a single 8-GPU box:
#
#   ──── Phase 1 ─────────────────────────────────────────────
#   attention baseline (all 8 GPUs) — runs Stage A/B/B.5 + C
#   end-to-end on the 12 extensive datasets.  Wall ≈ 25-40 min.
#
#   ──── Phase 2 ─────────────────────────────────────────────
#   atomwise (GPUs 0-3, N_GPUS=4)   ┐  Stage C only — caches
#   hybrid   (GPUs 4-7, N_GPUS=4)   ┘  populated by Phase 1.
#   Both launched in parallel.  Wall ≈ 10-20 min.
#
#   ──── Phase 3 ─────────────────────────────────────────────
#   ensemble_preds.py + summarize_cv_reports.py — auto-write
#   /tmp/cv_0519_a1a3_summary.csv and /tmp/cv_0519_ensemble.csv.
#
# Total wall ≈ 40-60 min on an 8-GPU box.
#
# Usage:
#   nohup bash scripts/run_cv_0519_a1a3_compare.sh \
#       > /tmp/cv_0519_a1a3_compare.log 2>&1 & disown

set -uo pipefail
cd "$(dirname "$0")/.."
date_str() { date +'%F %T'; }

OUT_ATT="outputs/cv_0519_baseline_ext_cold"
OUT_ATM="outputs/cv_0519_atomwise_cold"
OUT_HYB="outputs/cv_0519_hybrid_cold"

echo
echo "================================================================"
echo " [$(date_str)] Phase 1 — attention baseline (full Stage A/B/B.5/C)"
echo "================================================================"
bash scripts/run_cv_0519_baseline_ext_cold.sh || {
    echo "ERROR: Phase 1 (attention baseline) failed; aborting" >&2; exit 1; }

echo
echo "================================================================"
echo " [$(date_str)] Phase 2 — atomwise + hybrid (parallel, Stage C only)"
echo "================================================================"
echo "  atomwise on GPUs 0-3 (N_GPUS=4) → ${OUT_ATM}"
echo "  hybrid   on GPUs 4-7 (N_GPUS=4) → ${OUT_HYB}"

SKIP_SMI=1 SKIP_SAMPLE=1 SKIP_EXTRACT=1 \
CUDA_DEVICES=0,1,2,3 N_GPUS=4 TASKS_PER_GPU=4 \
    bash scripts/run_cv_0519_atomwise_cold.sh \
        > /tmp/cv_0519_phase2_atomwise.log 2>&1 &
PID_ATOM=$!

SKIP_SMI=1 SKIP_SAMPLE=1 SKIP_EXTRACT=1 \
CUDA_DEVICES=4,5,6,7 N_GPUS=4 TASKS_PER_GPU=4 \
    bash scripts/run_cv_0519_hybrid_cold.sh \
        > /tmp/cv_0519_phase2_hybrid.log 2>&1 &
PID_HYB=$!

echo "  atomwise pid=$PID_ATOM  hybrid pid=$PID_HYB  waiting..."
wait $PID_ATOM ; RC_ATOM=$?
wait $PID_HYB  ; RC_HYB=$?
echo "  atomwise rc=$RC_ATOM  hybrid rc=$RC_HYB"
if (( RC_ATOM != 0 )); then
    echo "WARN: atomwise exited with $RC_ATOM — see /tmp/cv_0519_phase2_atomwise.log" >&2
fi
if (( RC_HYB != 0 )); then
    echo "WARN: hybrid exited with $RC_HYB — see /tmp/cv_0519_phase2_hybrid.log" >&2
fi

echo
echo "================================================================"
echo " [$(date_str)] Phase 3 — analysis"
echo "================================================================"

echo "--- three-way summary (attention / atomwise / hybrid) ---"
python scripts/summarize_cv_reports.py \
    "$OUT_ATT" "$OUT_ATM" "$OUT_HYB" \
    --csv /tmp/cv_0519_a1a3_summary.csv

echo
echo "--- A1 post-hoc ensemble (attention + atomwise) ---"
python scripts/ensemble_preds.py \
    "$OUT_ATT/cold_combined_K8" \
    "$OUT_ATM/cold_combined_K8" \
    --batch --out /tmp/cv_0519_ensemble.csv

echo
echo "================================================================"
echo " [$(date_str)] done."
echo " Tables: /tmp/cv_0519_a1a3_summary.csv  /tmp/cv_0519_ensemble.csv"
echo " Per-run reports under:  $OUT_ATT  $OUT_ATM  $OUT_HYB"
echo "================================================================"
