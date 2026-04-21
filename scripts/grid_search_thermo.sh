#!/usr/bin/env bash
# Parallel grid search over thermo-head hyperparameters on N GPUs.
#
# Requires the shared H cache to already exist at $CACHE (run
# `bash scripts/run_thermo.sh extract` once first, or let a single run
# of finetune populate it). Each grid cell reuses the same H cache but
# writes its heads_final.pt / finetune_report.json to its own out-dir.
#
# Usage:
#   bash scripts/grid_search_thermo.sh
#
# Edit the GRID block below for the sweep you want. Default total:
#   2 × 3 × 3 × 2 = 36 combinations.

set -euo pipefail
cd "$(dirname "$0")/.."

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
echo "[grid] $TOTAL_ALL total combinations: $SKIPPED complete (skipped), "\
"$TOTAL pending"
if (( TOTAL == 0 )); then
    echo "[grid] nothing to run — jumping to summary"
else
    echo "[grid] $N_GPUS GPUs -> $(( (TOTAL + N_GPUS - 1) / N_GPUS )) waves"
fi

# --- Dispatch in waves of N_GPUS ---
start_time=$(date +%s)
for ((wave_start=0; wave_start<TOTAL; wave_start+=N_GPUS)); do
    echo "==> wave $((wave_start/N_GPUS + 1)): combos $wave_start..$((wave_start+N_GPUS-1))"
    pids=()
    for gpu in $(seq 0 $((N_GPUS - 1))); do
        slot=$((wave_start + gpu))
        [[ $slot -ge $TOTAL ]] && break
        # shellcheck disable=SC2086
        read -r L H D LR BS <<<"${CONFIGS[$slot]}"
        # Hparam-tagged run name → easy to filter/plot in wandb.
        name="L${L}_H${H}_D${D}_LR${LR}_BS${BS}_ep${EPOCHS}_s${SEED}"
        outdir="$GRID_OUT/$name"
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
        pids+=("$!")
        echo "   [$((slot+1))/$TOTAL] GPU $gpu  $name  log=$outdir/train.log"
    done
    wait "${pids[@]}"
done
end_time=$(date +%s)
echo "==> grid DONE in $((end_time - start_time))s"

# --- Summary table (sorted by test MAE on enthalpy_298, best head) ---
python3 - <<'PY'
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
