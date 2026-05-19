#!/usr/bin/env bash
# CV TASK POOL: each (dataset, fold) is a fully-independent task.
#
# Stage A/B/C (pkl / pt / H extract) are NOT done here — must be already
# cached. This script ONLY runs Stage D (head training on cached H) with
# fold-level granularity:
#
#   tasks = product(datasets, [0..N_FOLDS-1])
#
# All tasks go into a single global queue. N_GPUS × TASKS_PER_GPU slots
# pull tasks from the queue. Each slot invokes downstream_cv.py with
# --only-fold N which writes fold_cache/fold_N.json and exits without
# trying to aggregate.
#
# After all tasks complete, scripts/finalize_cv_report.py is run for
# each dataset's out_dir to combine fold_cache/fold_*.json into
# cv_report.json (the same file downstream_cv.py would have written).
#
# Compared to the current dataset-level pool (run_cv.sh), this gives
# strictly better load balancing: a 5572-mol dataset (e.g. BP_K) takes
# 5x longer than a 600-mol one for full CV, leaving GPU slots idle while
# big jobs finish. Fold-level pooling lets the big dataset's 5 folds
# spread across slots and finish in 1/5 the wall time.
#
# Required env vars (or hard-coded defaults below):
#   N_GPUS                e.g. 8
#   CUDA_DEVICES          e.g. 0,1,2,3,4,5,6,7
#   TASKS_PER_GPU         e.g. 4
#   CKPT                  backbone checkpoint
#   CONFIG                yaml config
#   PT_DIR                directory containing <ds>.pt and <ds>_H.pt (H cache)
#   OUT_ROOT              outputs/<run_name>/
#   OUT_SUFFIX            suffix for per-ds dirs, e.g. "cold_combined_K8"
#   SPLIT_DIR_ROOT        downstream_ft/0515_final/Split/
#   SPLIT_KIND            scaffold_hybrid_cv5 | scaffold_diverse_cv5 | random_cv5
#   DATASETS_FILTER       comma-separated dataset names
#   K                     conformer count, e.g. 8
#   EPOCHS, LR, WARMUP_FRACTION, GRAD_CLIP, BATCH  (head training hyperparams)
#   AUTO_EPOCHS=1 + EPOCHS_LARGE + EPOCHS_SMALL    (adaptive epoch budget)
#   EARLY_STOP_PATIENCE
#   INIT_FROM_THERMO
#   HEAD_HIDDEN, N_MP_LAYERS, MP_N_HEADS
#   DUMP_PREDS=1                                   (write per-sample preds for pooled metric)
#   LOG_DIR                /tmp/<run_name>
#
# Usage:
#   N_GPUS=8 CUDA_DEVICES=0,1,2,3,4,5,6,7 TASKS_PER_GPU=4 \
#   CKPT=data/ft_ckpts/thermo_flow_cold_combined.ckpt \
#   CONFIG=scripts/conf/loqi/loqi_thermo_flow_cold_combined.yaml \
#   PT_DIR=data/0515_pt_cold_combined_K8 \
#   OUT_ROOT=outputs/cv_0519_pool \
#   OUT_SUFFIX=cold_combined_K8 \
#   SPLIT_DIR_ROOT=downstream_ft/0515_final/Split \
#   SPLIT_KIND=scaffold_hybrid_cv5 \
#   DATASETS_FILTER=BP_K,Hf_gas_kJmol,... \
#   K=8 EPOCHS=150 LR=1e-4 WARMUP_FRACTION=0.2 GRAD_CLIP=0.1 BATCH=32 \
#   AUTO_EPOCHS=1 EPOCHS_LARGE=150 EPOCHS_SMALL=100 EARLY_STOP_PATIENCE=50 \
#   INIT_FROM_THERMO=0 HEAD_HIDDEN=256 N_MP_LAYERS=6 MP_N_HEADS=4 \
#   DUMP_PREDS=1 LOG_DIR=/tmp/cv_0519_pool \
#       bash scripts/run_cv_pool_8gpu.sh

set -uo pipefail
cd "$(dirname "$0")/.."

# ── Required vars ─────────────────────────────────────────────────────────
: "${CKPT:?CKPT required}"
: "${CONFIG:?CONFIG required}"
: "${PT_DIR:?PT_DIR required (must contain <ds>.pt and <ds>_H.pt H cache)}"
: "${OUT_ROOT:?OUT_ROOT required}"
: "${OUT_SUFFIX:=cold_combined_K8}"
: "${SPLIT_DIR_ROOT:?SPLIT_DIR_ROOT required}"
: "${SPLIT_KIND:=scaffold_hybrid_cv5}"
: "${DATASETS_FILTER:?DATASETS_FILTER required (comma-separated ds names)}"

: "${N_GPUS:=8}"
: "${CUDA_DEVICES:=0,1,2,3,4,5,6,7}"
: "${TASKS_PER_GPU:=4}"
: "${N_FOLDS:=5}"

: "${K:=8}"
: "${EPOCHS:=150}"
: "${LR:=1e-4}"
: "${WARMUP_FRACTION:=0.2}"
: "${GRAD_CLIP:=0.1}"
: "${BATCH:=32}"
: "${AUTO_EPOCHS:=1}"
: "${EPOCHS_LARGE:=150}"
: "${EPOCHS_SMALL:=100}"
: "${EARLY_STOP_PATIENCE:=50}"
: "${INIT_FROM_THERMO:=0}"
: "${HEAD_HIDDEN:=256}"
: "${N_MP_LAYERS:=6}"
: "${MP_N_HEADS:=4}"
: "${DUMP_PREDS:=1}"

: "${LOG_DIR:=/tmp/cv_pool_$$}"
mkdir -p "$LOG_DIR" "$OUT_ROOT"

# ── Pre-flight ────────────────────────────────────────────────────────────
[[ -f "$CKPT"   ]] || { echo "ERROR: ckpt not found: $CKPT" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }
[[ -d "$PT_DIR" ]] || { echo "ERROR: PT_DIR not found: $PT_DIR" >&2; exit 1; }
[[ -d "$SPLIT_DIR_ROOT" ]] || { echo "ERROR: SPLIT_DIR_ROOT not found" >&2; exit 1; }

# Parse GPUs into array
IFS=',' read -ra GPU_LIST <<< "$CUDA_DEVICES"
N_SLOTS=$(( ${#GPU_LIST[@]} * TASKS_PER_GPU ))

echo "================================================================"
echo " CV pool — fold-level task scheduling"
echo "   GPUs                : ${GPU_LIST[*]}  (× ${TASKS_PER_GPU} tasks each = ${N_SLOTS} slots)"
echo "   N_FOLDS             : $N_FOLDS"
echo "   SPLIT_KIND          : $SPLIT_KIND"
echo "   PT_DIR              : $PT_DIR"
echo "   OUT_ROOT            : $OUT_ROOT"
echo "   OUT_SUFFIX          : $OUT_SUFFIX"
echo "   DUMP_PREDS          : $DUMP_PREDS"
echo "================================================================"

# ── Build task list: (ds, fold) ───────────────────────────────────────────
# Sort datasets by .pt file size descending (longest-processing-time first).
IFS=',' read -ra DATASETS <<< "$DATASETS_FILTER"
echo "Discovering datasets and ordering by .pt size (LPT)..."
declare -a DS_BY_SIZE=()
for ds in "${DATASETS[@]}"; do
    pt_path="$PT_DIR/${ds}.pt"
    h_path="$PT_DIR/${ds}_H.pt"
    if [[ ! -f "$pt_path" ]]; then
        echo "  SKIP $ds  — missing $pt_path"; continue
    fi
    if [[ ! -f "$h_path" ]]; then
        echo "  SKIP $ds  — missing $h_path  (run Stage C first)"; continue
    fi
    size=$(stat -f%z "$pt_path" 2>/dev/null || stat -c%s "$pt_path" 2>/dev/null || echo 0)
    DS_BY_SIZE+=("${size}|${ds}")
done
# Sort numerically descending; if tied, by name
IFS=$'\n' DS_SORTED=($(printf "%s\n" "${DS_BY_SIZE[@]}" | sort -t'|' -k1,1nr -k2,2))
unset IFS

# Build tasks: for each ds (largest first), enumerate folds 0..N_FOLDS-1
TASKS=()
for entry in "${DS_SORTED[@]}"; do
    ds="${entry#*|}"
    for fold in $(seq 0 $((N_FOLDS - 1))); do
        # Skip if fold_cache already exists (resume support)
        out_dir="$OUT_ROOT/${ds}_${OUT_SUFFIX}"
        fc="$out_dir/fold_cache/fold_${fold}.json"
        if [[ -f "$fc" ]]; then
            echo "  [skip] $ds fold $fold — cache exists"
            continue
        fi
        TASKS+=("${ds}|${fold}")
    done
done
N_TASKS=${#TASKS[@]}
echo "Total tasks: $N_TASKS"

if (( N_TASKS == 0 )); then
    echo "Nothing to do."
else
    # ── Slot scheduler ─────────────────────────────────────────────────────
    declare -A PID_GPU
    declare -A PID_TASK

    launch_task() {
        local task="$1" gpu="$2"
        local ds="${task%|*}" fold="${task#*|}"
        local out_dir="$OUT_ROOT/${ds}_${OUT_SUFFIX}"
        local pt_path="$PT_DIR/${ds}.pt"
        local h_cache="$PT_DIR/${ds}_H.pt"
        local split_dir="$SPLIT_DIR_ROOT/${ds}/$SPLIT_KIND"
        local log_file="$LOG_DIR/${ds}_fold${fold}.log"

        if [[ ! -d "$split_dir" ]]; then
            echo "  [WARN] missing split dir: $split_dir  — skipping fold"
            return
        fi

        mkdir -p "$out_dir"

        local warm_args
        if [[ "$INIT_FROM_THERMO" == "1" ]]; then
            warm_args="--init-head-from-thermo"
        else
            warm_args="--head-hidden $HEAD_HIDDEN --n-mp-layers $N_MP_LAYERS --mp-n-heads $MP_N_HEADS"
        fi
        local epoch_args
        if [[ "$AUTO_EPOCHS" == "1" ]]; then
            epoch_args="--auto-epochs --epochs-large $EPOCHS_LARGE --epochs-small $EPOCHS_SMALL"
        else
            epoch_args="--epochs $EPOCHS"
        fi
        local stop_args=""
        (( EARLY_STOP_PATIENCE > 0 )) && stop_args="--early-stopping-patience $EARLY_STOP_PATIENCE"

        CUDA_VISIBLE_DEVICES=$gpu DUMP_PREDS=$DUMP_PREDS \
            python -u scripts/downstream_cv.py \
                --ckpt "$CKPT" --config "$CONFIG" \
                --dataset-pt "$pt_path" \
                --ensemble-by input_id \
                --out-dir "$out_dir" \
                --n-folds "$N_FOLDS" \
                --only-fold "$fold" \
                --split-dir "$split_dir" \
                --h-cache-path "$h_cache" \
                --lr "$LR" --warmup-fraction "$WARMUP_FRACTION" \
                --grad-clip "$GRAD_CLIP" \
                --batch-size "$BATCH" --device cuda \
                $warm_args $epoch_args $stop_args \
                >> "$log_file" 2>&1 &
        local pid=$!
        PID_GPU[$pid]=$gpu
        PID_TASK[$pid]="${ds}/fold${fold}"
    }

    # Initialize: fill all slots with first N_SLOTS tasks
    idx=0
    fail_count=0; done_count=0
    t0=$(date +%s)

    # Round-robin GPU assignment across slots
    while (( idx < N_TASKS && ${#PID_GPU[@]} < N_SLOTS )); do
        gpu=${GPU_LIST[$(( idx % ${#GPU_LIST[@]} ))]}
        task="${TASKS[$idx]}"
        launch_task "$task" "$gpu"
        echo "[$(date +%T)] [$((idx+1))/$N_TASKS] launch ${task} → GPU $gpu (pid=$!)"
        idx=$((idx + 1))
    done

    # Wait for any to finish; launch next task on freed slot
    while (( ${#PID_GPU[@]} > 0 )); do
        fpid=0
        wait -n -p fpid 2>/dev/null
        wst=$?
        fpid=${fpid:-0}
        if (( wst == 127 )) || [[ -z "${PID_GPU[$fpid]:-}" ]]; then
            # No more children, or unknown pid
            if (( ${#PID_GPU[@]} == 0 )); then break; fi
            # Mark all remaining as failed if wait returned 127
            if (( wst == 127 )); then
                for op in "${!PID_GPU[@]}"; do
                    echo "[$(date +%T)] FAIL ${PID_TASK[$op]} (pid=$op orphaned)"
                    fail_count=$((fail_count + 1)); done_count=$((done_count + 1))
                done
                break
            fi
            continue
        fi
        gpu="${PID_GPU[$fpid]}"; task_tag="${PID_TASK[$fpid]}"
        unset 'PID_GPU[$fpid]' 'PID_TASK[$fpid]'
        done_count=$((done_count + 1))
        if (( wst != 0 )); then
            fail_count=$((fail_count + 1))
            echo "[$(date +%T)] FAIL [$done_count/$N_TASKS] $task_tag (exit $wst) — check $LOG_DIR/${task_tag//\//_}.log"
        else
            echo "[$(date +%T)] DONE [$done_count/$N_TASKS] $task_tag"
        fi
        # Launch next task if any
        if (( idx < N_TASKS )); then
            new_gpu=${GPU_LIST[$(( idx % ${#GPU_LIST[@]} ))]}
            new_task="${TASKS[$idx]}"
            launch_task "$new_task" "$new_gpu"
            echo "[$(date +%T)] [$((idx+1))/$N_TASKS] launch ${new_task} → GPU $new_gpu (pid=$!)"
            idx=$((idx + 1))
        fi
    done

    t1=$(date +%s)
    echo
    echo "================================================================"
    echo " Pool exhausted: $((done_count - fail_count))/$N_TASKS succeeded, $fail_count failed.  Wall: $((t1 - t0))s"
    echo "================================================================"
fi

# ── Finalize: aggregate fold_cache → cv_report.json per dataset ──────────
echo
echo "Finalizing cv_report.json per dataset..."
finalize_args=()
for entry in "${DS_SORTED[@]}"; do
    ds="${entry#*|}"
    out_dir="$OUT_ROOT/${ds}_${OUT_SUFFIX}"
    [[ -d "$out_dir/fold_cache" ]] && finalize_args+=("$out_dir")
done
if (( ${#finalize_args[@]} > 0 )); then
    python scripts/finalize_cv_report.py --n-folds "$N_FOLDS" "${finalize_args[@]}"
fi
echo "Done. cv_report.json files: $OUT_ROOT/<ds>_${OUT_SUFFIX}/cv_report.json"
