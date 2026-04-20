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
# Model architecture + training hyperparameters live in YAML (edit there):
#   scripts/conf/thermo/finetune.yaml
#   scripts/conf/thermo/continuation.yaml
# This shell script only handles paths, GPU orchestration, and wandb naming.

CKPT=data/loqi.ckpt
LOQI_CONFIG=scripts/conf/loqi/loqi.yaml
FT_CFG=scripts/conf/thermo/finetune.yaml
CONT_CFG=scripts/conf/thermo/continuation.yaml

TRAIN_PT=data/chembl3d_stereo/processed/train_h_thermo.pt
VAL_PT=data/chembl3d_stereo/processed/val_h_thermo.pt
TEST_PT=data/chembl3d_stereo/processed/test_h_thermo.pt

CACHE=/tmp/ft_cache_full
CONT_OUT=/tmp/continuation_u2

# Empty string = "use all labeled samples" (scripts default to None).
MAX_TRAIN=""
MAX_VAL=""
MAX_TEST=""
# continuation typically uses a smaller subset because backbone is in the loop:
CONT_MAX_TRAIN="200000"

N_GPUS=4
SEED=0

# wandb
WANDB_PROJECT=thermogen
# ================================

# Helper: emit --max-<name> <value> only when non-empty.
_cap() { [[ -n "${2:-}" ]] && printf ' --%s %s' "$1" "$2" || true; }

mkdir -p "$CACHE" "$CONT_OUT"

_common_data_args() {
    echo "--train-pt $TRAIN_PT --val-pt $VAL_PT --test-pt $TEST_PT"
    printf '%s' "$(_cap max-train "$MAX_TRAIN")$(_cap max-val "$MAX_VAL")$(_cap max-test "$MAX_TEST")"
}

stage_extract() {
    echo "==> [$(date +%T)] Launching ${N_GPUS}-GPU H extraction"
    local pids=()
    for ((i=0; i<N_GPUS; i++)); do
        CUDA_VISIBLE_DEVICES=$i python scripts/finetune_thermo_head.py \
            --ckpt "$CKPT" --config "$LOQI_CONFIG" --thermo-config "$FT_CFG" \
            $(_common_data_args) \
            --cache-dir "$CACHE" \
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
        --ckpt "$CKPT" --config "$LOQI_CONFIG" --thermo-config "$FT_CFG" \
        $(_common_data_args) \
        --cache-dir "$CACHE" \
        --seed "$SEED" \
        --wandb --wandb-project "$WANDB_PROJECT" \
        --wandb-name "ft_$(basename ${FT_CFG%.yaml})_s${SEED}" \
        --device cuda 2>&1 | tee "$CACHE/train.log"
    echo "==> [$(date +%T)] train DONE"
}

stage_continue() {
    echo "==> [$(date +%T)] Continuation training (config: $CONT_CFG)"
    local head_init="$CACHE/heads_final.pt"
    if [[ ! -f "$head_init" ]]; then
        echo "ERROR: $head_init not found — run 'train' stage first"
        exit 1
    fi
    CUDA_VISIBLE_DEVICES=0 python scripts/continuation_training.py \
        --ckpt "$CKPT" --config "$LOQI_CONFIG" --thermo-config "$CONT_CFG" \
        --train-pt "$TRAIN_PT" --val-pt "$VAL_PT" --test-pt "$TEST_PT" \
        $(_cap max-train "$CONT_MAX_TRAIN")$(_cap max-val "$MAX_VAL")$(_cap max-test "$MAX_TEST") \
        --head-init "$head_init" \
        --out-dir "$CONT_OUT" \
        --seed "$SEED" \
        --wandb --wandb-project "$WANDB_PROJECT" \
        --wandb-name "cont_$(basename ${CONT_CFG%.yaml})_s${SEED}" \
        --device cuda 2>&1 | tee "$CONT_OUT/train.log"
    echo "==> [$(date +%T)] continue DONE"
}

stage_seeds() {
    echo "==> [$(date +%T)] Launching ensemble seeds 1,2,3 on GPUs 1,2,3"
    local pids=()
    for s in 1 2 3; do
        CUDA_VISIBLE_DEVICES=$s python scripts/finetune_thermo_head.py \
            --ckpt "$CKPT" --config "$LOQI_CONFIG" --thermo-config "$FT_CFG" \
            $(_common_data_args) \
            --cache-dir "$CACHE" \
            --seed "$s" \
            --wandb --wandb-project "$WANDB_PROJECT" \
            --wandb-group "ft_$(basename ${FT_CFG%.yaml})_seeds" \
            --wandb-name "seed_${s}" \
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
