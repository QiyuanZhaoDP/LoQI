#!/usr/bin/env bash
# ThermoGen Phase 0 / continuation pipeline runner.
#
# Usage:
#   bash scripts/run_thermo.sh extract     # ${N_GPUS}-way parallel H extraction
#   bash scripts/run_thermo.sh train       # single-GPU head training (auto-merges shards)
#   bash scripts/run_thermo.sh continue    # continuation training (unfreeze last N layers)
#   bash scripts/run_thermo.sh seeds       # ensemble seeds 1,2,3 on GPUs 1,2,3 (optional)
#   bash scripts/run_thermo.sh all         # extract -> train -> continue
#
# Edit the CONFIG section below before running.

set -euo pipefail
cd "$(dirname "$0")/.."  # project root

# ============ CONFIG ============
CKPT=data/loqi.ckpt
CONFIG=scripts/conf/loqi/loqi.yaml
TRAIN_PT=data/chembl3d_stereo/processed/train_h_thermo.pt
TEST_PT=data/chembl3d_stereo/processed/test_h_thermo.pt
CACHE=/tmp/ft_cache_500k
CONT_OUT=/tmp/continuation_u2

MAX_TRAIN=500000
MAX_TEST=20000
N_GPUS=4
SEED=0

# Head-training (frozen backbone)
EXTRACT_BS=128
TRAIN_BS=512
TRAIN_EPOCHS=30
TRAIN_LR=3e-4

# Continuation (unfreeze last N backbone layer pairs)
CONT_MAX_TRAIN=200000
CONT_BS=32
CONT_EPOCHS=10
CONT_UNFREEZE=2
CONT_HEAD_LR=3e-4
CONT_BB_LR=1e-5

# wandb
WANDB_PROJECT=thermogen
# ================================

mkdir -p "$CACHE" "$CONT_OUT"

stage_extract() {
    echo "==> [$(date +%T)] Launching ${N_GPUS}-GPU H extraction"
    local pids=()
    for ((i=0; i<N_GPUS; i++)); do
        CUDA_VISIBLE_DEVICES=$i python scripts/finetune_thermo_head.py \
            --ckpt "$CKPT" --config "$CONFIG" \
            --train-pt "$TRAIN_PT" --test-pt "$TEST_PT" \
            --cache-dir "$CACHE" \
            --max-train "$MAX_TRAIN" --max-test "$MAX_TEST" \
            --cache-dtype bf16 --extract-batch-size "$EXTRACT_BS" \
            --shard-id "$i" --n-shards "$N_GPUS" \
            --seed "$SEED" --device cuda \
            > "$CACHE/extract_shard$i.log" 2>&1 &
        pids+=("$!")
        echo "   shard $i -> pid ${pids[$i]}, GPU $i, log $CACHE/extract_shard$i.log"
    done
    echo "==> Waiting for all shards..."
    wait "${pids[@]}"
    echo "==> Shard cache files:"
    ls -lh "$CACHE"/*shard*_of_*.pt 2>/dev/null || echo "   (none)"
    echo "==> [$(date +%T)] extract DONE"
}

stage_train() {
    echo "==> [$(date +%T)] Single-GPU head training (auto-merges shards if present)"
    CUDA_VISIBLE_DEVICES=0 python scripts/finetune_thermo_head.py \
        --ckpt "$CKPT" --config "$CONFIG" \
        --train-pt "$TRAIN_PT" --test-pt "$TEST_PT" \
        --cache-dir "$CACHE" \
        --max-train "$MAX_TRAIN" --max-test "$MAX_TEST" \
        --cache-dtype bf16 \
        --epochs "$TRAIN_EPOCHS" --batch-size "$TRAIN_BS" --lr "$TRAIN_LR" \
        --seed "$SEED" \
        --wandb --wandb-project "$WANDB_PROJECT" \
        --wandb-name "ft_n${MAX_TRAIN}_s${SEED}" \
        --device cuda 2>&1 | tee "$CACHE/train.log"
    echo "==> [$(date +%T)] train DONE"
}

stage_continue() {
    echo "==> [$(date +%T)] Continuation training (unfreeze last ${CONT_UNFREEZE} backbone layer pairs)"
    local head_init="$CACHE/heads_final.pt"
    if [[ ! -f "$head_init" ]]; then
        echo "ERROR: $head_init not found — run 'train' stage first"
        exit 1
    fi
    CUDA_VISIBLE_DEVICES=0 python scripts/continuation_training.py \
        --ckpt "$CKPT" --config "$CONFIG" \
        --train-pt "$TRAIN_PT" --test-pt "$TEST_PT" \
        --head-init "$head_init" \
        --out-dir "$CONT_OUT" \
        --unfreeze-layers "$CONT_UNFREEZE" \
        --max-train "$CONT_MAX_TRAIN" --max-test "$MAX_TEST" \
        --epochs "$CONT_EPOCHS" --batch-size "$CONT_BS" \
        --lr "$CONT_HEAD_LR" --backbone-lr "$CONT_BB_LR" \
        --seed "$SEED" \
        --wandb --wandb-project "$WANDB_PROJECT" \
        --wandb-name "cont_u${CONT_UNFREEZE}_n${CONT_MAX_TRAIN}_s${SEED}" \
        --device cuda 2>&1 | tee "$CONT_OUT/train.log"
    echo "==> [$(date +%T)] continue DONE"
}

stage_seeds() {
    echo "==> [$(date +%T)] Launching ensemble seeds 1,2,3 on GPUs 1,2,3"
    local pids=()
    for s in 1 2 3; do
        CUDA_VISIBLE_DEVICES=$s python scripts/finetune_thermo_head.py \
            --ckpt "$CKPT" --config "$CONFIG" \
            --train-pt "$TRAIN_PT" --test-pt "$TEST_PT" \
            --cache-dir "$CACHE" \
            --max-train "$MAX_TRAIN" --max-test "$MAX_TEST" \
            --cache-dtype bf16 \
            --epochs "$TRAIN_EPOCHS" --batch-size "$TRAIN_BS" --lr "$TRAIN_LR" \
            --seed "$s" \
            --wandb --wandb-project "$WANDB_PROJECT" \
            --wandb-group "ft_n${MAX_TRAIN}_seeds" --wandb-name "seed_${s}" \
            --device cuda > "$CACHE/train_seed${s}.log" 2>&1 &
        pids+=("$!")
        echo "   seed $s -> pid ${pids[-1]}, GPU $s"
    done
    wait "${pids[@]}"
    echo "==> [$(date +%T)] seeds DONE"
}

cmd="${1:-}"
case "$cmd" in
    extract)  stage_extract  ;;
    train)    stage_train    ;;
    continue) stage_continue ;;
    seeds)    stage_seeds    ;;
    all)      stage_extract; stage_train; stage_continue ;;
    *)
        echo "usage: bash $0 {extract|train|continue|seeds|all}" >&2
        exit 1
        ;;
esac
