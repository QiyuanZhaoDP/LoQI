#!/usr/bin/env bash
# ThermoGen Phase 0 pipeline runner.
#
# Usage:
#   bash scripts/run_thermo.sh extract     # ${N_GPUS}-way parallel H extraction
#   bash scripts/run_thermo.sh train       # single-GPU head training (auto-merges shards)
#   bash scripts/run_thermo.sh seeds       # ensemble seeds 1,2,3 on GPUs 1,2,3 (optional)
#   bash scripts/run_thermo.sh sample_k5   # flow-matching K=5 conformer sampling
#                                          # for all TCIT-labeled molecules,
#                                          # sharded across ${N_GPUS} GPUs
#   bash scripts/run_thermo.sh all         # extract -> train
#
# Edit the CONFIG section below before running.

set -euo pipefail
cd "$(dirname "$0")/.."  # project root

# ============ CONFIG ============
# Model architecture + training hyperparameters live in YAML:
#   scripts/conf/thermo/finetune.yaml
# This shell script only handles paths, GPU orchestration, and wandb naming.

CKPT=data/loqi.ckpt
LOQI_CONFIG=scripts/conf/loqi/loqi.yaml
FT_CFG=scripts/conf/thermo/finetune.yaml

# Flow-matching checkpoint — used only by the `sample_k5` stage.
FLOW_CKPT=data/thermo_flow_warm.ckpt
FLOW_CONFIG=scripts/conf/loqi/loqi_thermo_flow_warm.yaml

TRAIN_PT=data/chembl3d_stereo/processed/train_h.pt
VAL_PT=data/chembl3d_stereo/processed/val_h.pt
TEST_PT=data/chembl3d_stereo/processed/test_h.pt
PROPERTY_TABLE=data/property_table.parquet

# Output directory for pre-sampled K=5 conformers (stage `sample_k5`).
K5_OUT=data/labeled_conformers_warm
K5_N=5
K5_STEPS=10
K5_BATCH=256

CACHE=/tmp/ft_cache_full

# Empty string = "use all labeled samples" (scripts default to None).
MAX_TRAIN=""
MAX_VAL=""
MAX_TEST=""

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

mkdir -p "$CACHE"

_common_data_args() {
    echo "--train-pt $TRAIN_PT --val-pt $VAL_PT --test-pt $TEST_PT --property-table $PROPERTY_TABLE"
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

stage_sample_k5() {
    # Pre-sample K=${K5_N} conformers for every TCIT-labeled molecule using
    # the flow-matching LoQI checkpoint. Each split is processed in parallel
    # with ${N_GPUS}-way sharding on the molecule list.
    echo "==> [$(date +%T)] K=${K5_N} conformer sampling on ${N_GPUS} GPU(s)"
    mkdir -p "$K5_OUT"
    if [[ ! -f "$FLOW_CKPT" ]]; then
        echo "ERROR: flow checkpoint not found at $FLOW_CKPT" >&2
        exit 1
    fi
    #local splits=(train val test)
    #local split_paths=("$TRAIN_PT" "$VAL_PT" "$TEST_PT")
    local splits=(val test)
    local split_paths=("$VAL_PT" "$TEST_PT")
    for idx in "${!splits[@]}"; do
        local split="${splits[$idx]}"
        local in_pt="${split_paths[$idx]}"
        echo "--> split=$split  input=$in_pt"
        local pids=()
        for ((i=0; i<N_GPUS; i++)); do
            local out_pkl="$K5_OUT/${split}_K${K5_N}_shard${i}.pkl"
            CUDA_VISIBLE_DEVICES=$i python scripts/sample_conformers_for_labeled.py \
                --ckpt "$FLOW_CKPT" --config "$FLOW_CONFIG" \
                --input-pt "$in_pt" \
                --property-table "$PROPERTY_TABLE" \
                --output-pkl "$out_pkl" \
                --n-confs "$K5_N" --n-steps "$K5_STEPS" \
                --batch-size "$K5_BATCH" \
                --shard-id "$i" --n-shards "$N_GPUS" \
                --device cuda \
                > "$K5_OUT/${split}_shard${i}.log" 2>&1 &
            pids+=("$!")
            echo "    shard $i -> pid ${pids[-1]}, GPU $i, out $out_pkl"
        done
        wait "${pids[@]}"
    done
    echo "==> [$(date +%T)] sample_k5 DONE  (shards in $K5_OUT)"
}

cmd="${1:-}"
case "$cmd" in
    extract)    stage_extract    ;;
    train)      stage_train      ;;
    seeds)      stage_seeds      ;;
    sample_k5)  stage_sample_k5  ;;
    all)        stage_extract; stage_train ;;
    *)
        echo "usage: bash $0 {extract|train|seeds|sample_k5|all}" >&2
        exit 1
        ;;
esac
