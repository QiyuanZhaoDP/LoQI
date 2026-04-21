#!/usr/bin/env bash
# Parallel grid search over thermo-head hyperparameters on N GPUs.
#
# Worker-pool scheduling: one job per GPU at any time, and the next
# pending cell is dispatched the moment ANY GPU finishes (no wave
# barrier). This keeps GPU utilization maxed when runtimes vary between
# cells (batch_size swings, etc.).
#
# Requires the shared H cache to already exist at $CACHE (run
# `bash scripts/run_thermo.sh extract` once first, or let a single run
# of finetune populate it). Each grid cell reuses the same H cache but
# writes its heads_final.pt / finetune_report.json to its own out-dir.
#
# Usage:
#   bash scripts/grid_search_thermo.sh
#   FORCE=1 bash scripts/grid_search_thermo.sh   # re-run complete cells
#
# Edit the GRID block below for the sweep you want.
#
# Requires bash >= 5.1 for `wait -n -p` (pid-of-finished-child capture).

set -euo pipefail
cd "$(dirname "$0")/.."

# ---- bash version guard ----------------------------------------------------
if (( BASH_VERSINFO[0] < 5 )) || { (( BASH_VERSINFO[0] == 5 )) && (( BASH_VERSINFO[1] < 1 )); }; then
    echo "ERROR: this script needs bash >= 5.1 (wait -n -p). You have $BASH_VERSION" >&2
    exit 1
fi

# ============ FIXED CONFIG ============
CKPT=data/loqi.ckpt
LOQI_CONFIG=scripts/conf/loqi/loqi.yaml
FT_CFG=scripts/conf/thermo/finetune.yaml
TRAIN_PT=data/chembl3d_stereo/processed/train_h.pt
VAL_PT=data/chembl3d_stereo/processed/val_h.pt
TEST_PT=data/chembl3d_stereo/processed/test_h.pt
PROPERTY_TABLE=data/property_table.parquet

CACHE=/tmp/ft_cache_full                # shared H cache — one copy, reused
GRID_OUT=/tmp/grid_thermo               # per-run outputs land here
SEED=0

WANDB_PROJECT=ft
WANDB_GROUP=grid_$(date +%Y%m%d_%H%M)

# GPU pool: empty => auto-detect via nvidia-smi
N_GPUS=""

# ============ GRID ============
# Each array is a dimension. Cartesian product = all combos.
# Leave an array with a single element to pin that dimension.
LAYERS=(2 4)                   # --n-mp-layers
HEADS=(4 8)                    # --mp-n-heads  (must divide 256)
HIDDEN=(128 256 512)           # --head-hidden
LRS=(3e-4 1e-4)                # --lr
BATCH_SIZES=(64 128 256 512)   # --batch-size (larger = fewer steps/epoch)
EPOCHS=100                     # constant across the grid
# ==============================

# --- Auto-detect GPUs ---
if [[ -z "$N_GPUS" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        N_GPUS=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
    elif command -v nvidia-smi >/dev/null 2>&1; then
        N_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    else
        N_GPUS=1
    fi
fi
[[ "$N_GPUS" -lt 1 ]] && N_GPUS=1
echo "[config] using $N_GPUS GPU(s), cache=$CACHE, out=$GRID_OUT"

mkdir -p "$GRID_OUT"

# --- Pre-flight: cache must exist ---
if ! ls "$CACHE"/train_H_*.pt >/dev/null 2>&1; then
    echo "ERROR: no H cache found at $CACHE"
    echo "Run 'bash scripts/run_thermo.sh extract' first to populate it."
    exit 1
fi

# --- Build cartesian product ---
ALL_CONFIGS=()
for L in "${LAYERS[@]}"; do
  for H in "${HEADS[@]}"; do
    for D in "${HIDDEN[@]}"; do
      for LR in "${LRS[@]}"; do
        for BS in "${BATCH_SIZES[@]}"; do
          ALL_CONFIGS+=("$L $H $D $LR $BS")
        done
      done
    done
  done
done

# --- Skip already-complete cells (those with a valid finetune_report.json).
#     Re-run everything regardless by setting FORCE=1.
FORCE="${FORCE:-0}"
CONFIGS=()
SKIPPED=0
for cfg in "${ALL_CONFIGS[@]}"; do
    read -r L H D LR BS <<<"$cfg"
    name="L${L}_H${H}_D${D}_LR${LR}_BS${BS}_ep${EPOCHS}_s${SEED}"
    report="$GRID_OUT/$name/finetune_report.json"
    if [[ "$FORCE" != "1" ]] \
        && [[ -f "$report" ]] \
        && python3 -c "import json,sys; json.load(open('$report'))" >/dev/null 2>&1; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    CONFIGS+=("$cfg")
done
TOTAL=${#CONFIGS[@]}
TOTAL_ALL=${#ALL_CONFIGS[@]}
echo "[grid] $TOTAL_ALL total combinations: $SKIPPED complete (skipped), $TOTAL pending"

# --- Launcher helper. Starts cell $1 on GPU $2 in the background.
_launch() {
    local cfg="$1" gpu="$2"
    read -r L H D LR BS <<<"$cfg"
    local name="L${L}_H${H}_D${D}_LR${LR}_BS${BS}_ep${EPOCHS}_s${SEED}"
    local outdir="$GRID_OUT/$name"
    mkdir -p "$outdir"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/finetune_thermo_head.py \
        --ckpt "$CKPT" --config "$LOQI_CONFIG" --thermo-config "$FT_CFG" \
        --train-pt "$TRAIN_PT" --val-pt "$VAL_PT" --test-pt "$TEST_PT" \
        --property-table "$PROPERTY_TABLE" \
        --cache-dir "$CACHE" \
        --out-dir "$outdir" \
        --n-mp-layers "$L" --mp-n-heads "$H" --head-hidden "$D" \
        --lr "$LR" --batch-size "$BS" --epochs "$EPOCHS" \
        --seed "$SEED" \
        --wandb --wandb-project "$WANDB_PROJECT" \
        --wandb-group "$WANDB_GROUP" --wandb-name "$name" \
        --device cuda \
        > "$outdir/train.log" 2>&1 &
    echo "$!|$name"
}

# --- Worker pool dispatch ---
start_time=$(date +%s)
declare -A GPU_OF_PID   # pid -> gpu index
declare -A NAME_OF_PID  # pid -> run name (for logging)
n_started=0
n_done=0
n_failed=0

# Seed the pool: one job per GPU (or fewer if fewer cells than GPUs).
for ((gpu=0; gpu<N_GPUS && n_started<TOTAL; gpu++)); do
    result="$(_launch "${CONFIGS[$n_started]}" "$gpu")"
    pid="${result%%|*}"
    name="${result#*|}"
    GPU_OF_PID[$pid]=$gpu
    NAME_OF_PID[$pid]="$name"
    echo "[$((n_started+1))/$TOTAL] launch pid=$pid gpu=$gpu  $name"
    n_started=$((n_started + 1))
done

# Drain: whenever one finishes, immediately re-launch the next pending cell
# on its freed GPU.
finished_pid=0
while (( ${#GPU_OF_PID[@]} > 0 )); do
    # wait -n -p: block until ANY child exits, capture its pid in $finished_pid.
    # Exit code is the child's status (non-zero = the job failed; we keep going).
    wait -n -p finished_pid || true
    status=$?
    if [[ -z "${GPU_OF_PID[$finished_pid]:-}" ]]; then
        continue  # spurious wake (e.g., subshell); loop
    fi
    gpu="${GPU_OF_PID[$finished_pid]}"
    name="${NAME_OF_PID[$finished_pid]}"
    unset 'GPU_OF_PID[$finished_pid]'
    unset 'NAME_OF_PID[$finished_pid]'
    n_done=$((n_done + 1))
    if (( status != 0 )); then
        n_failed=$((n_failed + 1))
        echo "[$n_done/$TOTAL] FAIL pid=$finished_pid gpu=$gpu status=$status  $name"
    else
        echo "[$n_done/$TOTAL] done pid=$finished_pid gpu=$gpu  $name"
    fi
    # If more work remains, launch the next cell on the freed GPU.
    if (( n_started < TOTAL )); then
        result="$(_launch "${CONFIGS[$n_started]}" "$gpu")"
        pid="${result%%|*}"
        name="${result#*|}"
        GPU_OF_PID[$pid]=$gpu
        NAME_OF_PID[$pid]="$name"
        echo "[$((n_started+1))/$TOTAL] launch pid=$pid gpu=$gpu  $name"
        n_started=$((n_started + 1))
    fi
done
end_time=$(date +%s)
echo "==> grid DONE in $((end_time - start_time))s  ($n_done completed, $n_failed failed)"

# --- Summary table (sorted by test MAE on enthalpy_298, best head) ---
GRID_OUT="$GRID_OUT" python3 - <<'PY'
import json, os, glob
rows = []
for path in sorted(glob.glob(os.path.join(os.environ.get("GRID_OUT","/tmp/grid_thermo"),
                                           "*", "finetune_report.json"))):
    try:
        d = json.load(open(path))
    except Exception as e:
        print(f"  skipping {path}: {e}")
        continue
    cfg = d["args"]
    rep = {r["target"]: r for r in d["rows"] if "mae_mp" in r}
    name = os.path.basename(os.path.dirname(path))
    if "enthalpy_298" not in rep:
        continue
    rows.append((rep["enthalpy_298"]["mae_mp"], name, rep, cfg))

rows.sort(key=lambda r: r[0])
print("\nGrid results (sorted by Hf_298 MAE):")
print(f"{'name':<32s} {'Hf (kJ/mol)':>12s} {'Gf':>8s} {'Cv':>6s} {'S0':>6s} {'Hf_0':>8s}")
print("-" * 80)
for hf, name, rep, cfg in rows:
    print(f"{name:<32s} {hf:>12.2f} "
          f"{rep['gibbs_298']['mae_mp']:>8.2f} "
          f"{rep['cv_gas']['mae_mp']:>6.2f} "
          f"{rep['entropy_gas']['mae_mp']:>6.2f} "
          f"{rep['enthalpy_0']['mae_mp']:>8.2f}")
PY
