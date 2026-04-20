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

# GPU orchestration:
#   Leave N_GPUS empty to auto-detect via nvidia-smi. Set to an integer
#   to cap usage (e.g. N_GPUS=2 to reserve 2 cards for other work).
#   CUDA_VISIBLE_DEVICES in the environment is respected — if set to
#   "1,3" we'll see 2 GPUs and use indices 0..1 of the visible set.
N_GPUS=""
SEED=0

# wandb
WANDB_PROJECT=thermogen
# ================================

# Helper: emit --max-<name> <value> only when non-empty.
_cap() { [[ -n "${2:-}" ]] && printf ' --%s %s' "$1" "$2" || true; }

# Auto-detect GPUs if N_GPUS not set.
_detect_gpus() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        # Count commas + 1 in the visible list.
        local n
        n=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
        echo "$n"
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
    else
        echo 0
    fi
}

if [[ -z "$N_GPUS" ]]; then
    N_GPUS=$(_detect_gpus)
    [[ -z "$N_GPUS" || "$N_GPUS" -lt 1 ]] && N_GPUS=1
    echo "[config] auto-detected N_GPUS=$N_GPUS"
fi

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
    echo "==> [$(date +%T)] Continuation training (config: $CONT_CFG, N_GPUS=$N_GPUS)"
    local head_init="$CACHE/heads_final.pt"
    if [[ ! -f "$head_init" ]]; then
        echo "ERROR: $head_init not found — run 'train' stage first"
        exit 1
    fi
    # Args shared between single-GPU and DDP launch paths.
    local args=(
        --ckpt "$CKPT" --config "$LOQI_CONFIG" --thermo-config "$CONT_CFG"
        --train-pt "$TRAIN_PT" --val-pt "$VAL_PT" --test-pt "$TEST_PT"
    )
    # _cap produces leading " --max-train N"; split into individual tokens.
    # shellcheck disable=SC2206
    args+=($(_cap max-train "$CONT_MAX_TRAIN") $(_cap max-val "$MAX_VAL") $(_cap max-test "$MAX_TEST"))
    args+=(
        --head-init "$head_init"
        --out-dir   "$CONT_OUT"
        --seed      "$SEED"
        --wandb --wandb-project "$WANDB_PROJECT"
        --wandb-name "cont_$(basename ${CONT_CFG%.yaml})_s${SEED}_ws${N_GPUS}"
        --device cuda
    )
    if (( N_GPUS > 1 )); then
        echo "   launching DDP across $N_GPUS GPUs via torchrun"
        torchrun --standalone --nnodes=1 --nproc_per_node="$N_GPUS" \
            scripts/continuation_training.py "${args[@]}" 2>&1 | tee "$CONT_OUT/train.log"
    else
        CUDA_VISIBLE_DEVICES=0 python scripts/continuation_training.py \
            "${args[@]}" 2>&1 | tee "$CONT_OUT/train.log"
    fi
    echo "==> [$(date +%T)] continue DONE"
}

stage_seeds() {
    # Run ensemble-seed runs on every GPU beyond GPU 0 (which we reserve
    # for the primary stage). For N_GPUS=4 we get seeds 1,2,3 in parallel;
    # for N_GPUS=1 we fall back to a single sequential seed-1 run.
    local n_seed_gpus=$((N_GPUS - 1))
    if (( n_seed_gpus < 1 )); then
        echo "==> [$(date +%T)] Only $N_GPUS GPU(s) visible; running seed 1 sequentially on GPU 0"
        CUDA_VISIBLE_DEVICES=0 python scripts/finetune_thermo_head.py \
            --ckpt "$CKPT" --config "$LOQI_CONFIG" --thermo-config "$FT_CFG" \
            $(_common_data_args) \
            --cache-dir "$CACHE" \
            --seed 1 \
            --wandb --wandb-project "$WANDB_PROJECT" \
            --wandb-group "ft_$(basename ${FT_CFG%.yaml})_seeds" \
            --wandb-name "seed_1" \
            --device cuda 2>&1 | tee "$CACHE/train_seed1.log"
        echo "==> [$(date +%T)] seeds DONE"
        return
    fi
    echo "==> [$(date +%T)] Launching $n_seed_gpus ensemble seeds on GPUs 1..$n_seed_gpus"
    local pids=()
    for ((s=1; s<=n_seed_gpus; s++)); do
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
