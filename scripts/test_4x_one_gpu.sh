#!/usr/bin/env bash
# 4× identical Hf_G CV on one GPU — multiprocessing stress test.
#
# Why this exists: --concurrent-folds (threading) failed because GIL
# serialized the 5 fold threads (commit dd63586 reverts it). This test
# uses 4 SEPARATE PROCESSES on the same GPU — no GIL fight, each
# process owns its own CUDA context, kernels from different processes
# interleave on the GPU's command queue.
#
# What it measures:
#   * Per-task wall time vs single-task baseline (~28 s/epoch at bs=32)
#   * Total wall time for 4 tasks vs sequential 4× single-task
#   * Whether GPU util actually rises above the ~10 % we saw with 1 task
#
# Interpretation:
#   * If 4-task wall ≈ 1-task wall × 4 → no parallelism, GPU command
#     processor is saturated even at single-task launch rates
#   * If 4-task wall ≈ 1-task wall × 1-2 → good parallelism, justifies
#     TASKS_PER_GPU=4 in production wrappers
#   * Anywhere in between → partial win
#
# Usage:
#   bash scripts/test_4x_one_gpu.sh
#   GPU=3 N_TASKS=2 bash scripts/test_4x_one_gpu.sh   # 2 tasks on GPU 3
#
# Run nvidia-smi in another terminal during the test:
#   watch -n 1 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -i $GPU'

set -uo pipefail
cd "$(dirname "$0")/.."

GPU=${GPU:-0}
N_TASKS=${N_TASKS:-4}
EPOCHS=${EPOCHS:-5}
BATCH=${BATCH:-32}
OUT_BASE=${OUT_BASE:-/tmp/hfg_4x_test}

CKPT=${CKPT:-data/ft_ckpts/loqi_flow.ckpt}
CONFIG=${CONFIG:-scripts/conf/loqi/loqi_flow.yaml}
DATA_PT=${DATA_PT:-data/0511_pt_loqi_flow_k8/Hf_G_K8.pt}
H_CACHE=${H_CACHE:-data/0511_pt_loqi_flow_k8/Hf_G_H.pt}
SPLIT_DIR=${SPLIT_DIR:-downstream_ft/0511_cc_audit/Split/Hf_G/random_cv5}

# Pre-flight: every task uses the same input files, so check once.
for f in "$CKPT" "$CONFIG" "$DATA_PT" "$H_CACHE"; do
    [[ -e "$f" ]] || { echo "ERROR: missing file: $f" >&2; exit 1; }
done
[[ -d "$SPLIT_DIR" ]] || { echo "ERROR: missing split dir: $SPLIT_DIR" >&2; exit 1; }

# Wipe previous results so fold_cache doesn't make tasks finish early.
rm -rf "$OUT_BASE"
mkdir -p "$OUT_BASE"

echo "================================================================"
echo " 4× same-task GPU stress test"
echo "================================================================"
echo "  GPU            : $GPU"
echo "  tasks          : $N_TASKS"
echo "  epochs / fold  : $EPOCHS"
echo "  folds per task : 1 (so each task runs exactly $EPOCHS train epochs)"
echo "  batch size     : $BATCH"
echo "  out            : $OUT_BASE/task_{0..$((N_TASKS-1))}"
echo "================================================================"

START=$(date +%s)
pids=()

for i in $(seq 0 $((N_TASKS-1))); do
    out_dir="$OUT_BASE/task_$i"
    log_file="$OUT_BASE/task_$i.log"
    mkdir -p "$out_dir"

    CUDA_VISIBLE_DEVICES=$GPU \
    python -u scripts/downstream_cv.py \
        --ckpt   "$CKPT" \
        --config "$CONFIG" \
        --dataset-pt   "$DATA_PT" \
        --h-cache-path "$H_CACHE" \
        --split-dir    "$SPLIT_DIR" \
        --ensemble-by input_id \
        --out-dir "$out_dir" \
        --n-folds 1 \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH" \
        --debug-timing \
        --device cuda \
        > "$log_file" 2>&1 &
    pids+=($!)
    echo "  [launch] task $i  PID ${pids[-1]}  → $log_file"
done

echo
echo "[$(date +%T)] waiting for $N_TASKS tasks…"
fails=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        ((fails++))
    fi
done

END=$(date +%s)
ELAPSED=$((END - START))

echo
echo "================================================================"
echo " results"
echo "================================================================"
echo "  total wall time : $ELAPSED s   ($N_TASKS tasks, $fails failed)"
echo

# Per-task per-epoch averages (skip epoch 1 — includes CUDA warmup)
echo "  per-task avg s/epoch (epochs 2–$EPOCHS):"
for i in $(seq 0 $((N_TASKS-1))); do
    log="$OUT_BASE/task_$i.log"
    if [[ -f "$log" ]]; then
        avg=$(grep "timing.*total=" "$log" \
                | tail -n +2 \
                | awk -F'total=' '{print $2}' \
                | awk -F's' '{sum+=$1; n++} END {if(n>0) printf "%.2f", sum/n; else print "n/a"}')
        # Also extract the final val MAE for correctness sanity
        mae=$(grep -oE "MAE=[0-9.]+" "$log" | head -1 | tr -d 'MAE=' )
        echo "    task $i:  $avg s/epoch     fold-1 MAE=${mae:-?}"
    else
        echo "    task $i:  no log"
    fi
done

# Compare to expected baseline. Single-task at bs=32 took ~28 s/epoch
# in the May-13 smoke. Sequential 4× would be ~4× single-task wall:
#   1-task EPOCHS × ~28 = base_one
#   4-task sequential ≈ 4 × base_one
# Concurrent achieves wall ≈ base_one × slowdown_factor.
SINGLE_REF_S=$((28 * EPOCHS))   # 28 s/epoch baseline × epochs
SEQ_REF_S=$((SINGLE_REF_S * N_TASKS))
echo
echo "  baseline 1-task  ≈ $SINGLE_REF_S s  (28 s/epoch × $EPOCHS ep)"
echo "  sequential ${N_TASKS}x ≈ $SEQ_REF_S s  (no concurrency)"
if (( ELAPSED > 0 )); then
    # speedup vs sequential — anything > 1 means we got *some* parallelism
    speedup_x100=$(( SEQ_REF_S * 100 / ELAPSED ))
    echo "  effective speedup vs sequential: $(( speedup_x100 / 100 )).$(( speedup_x100 % 100 ))×"
fi
echo "================================================================"
