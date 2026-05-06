#!/usr/bin/env bash
# Full downstream test on the 0506 dataset collection.
#
# What it does (sequentially, all 4 GPUs):
#   Stage A  Extract SMILES from each cleaned CSV
#   Stage B  K=8  standard conformer sampling (1 trajectory per mol)
#   Stage C  K=15 multi-snapshot sampling  (5 traj × steps 7,8,9 = 15 conf)
#   Stage D  M0: 5-fold CV, K=8,  INIT_FROM_THERMO=1
#   Stage E  M2: 5-fold CV, K=15, INIT_FROM_THERMO=1
#
# Only thermo-head warm-init is tested (no random-init cold_large).
# Both K=8 ensemble and K=1 per-conformer metrics are reported automatically.
#
# Requires:
#   CKPT   path to the cold backbone checkpoint
#   CONFIG path to the matching config YAML
#
# If you only have the ckpt and can't remember the exact config, use
# loqi_thermo_flow_cold.yaml — it matches the cold backbone architecture
# (384/12/4) and the difference in head definition is ignored by
# strict=False loading.
#
# Usage:
#   CKPT=data/thermo_flow_cold.ckpt \
#   SWANLAB_SYNC=1 \
#   nohup bash scripts/run_0506_test.sh > /tmp/0506_test.log 2>&1 &
#   disown

set -uo pipefail
cd "$(dirname "$0")/.."

if (( BASH_VERSINFO[0] < 5 )) || { (( BASH_VERSINFO[0] == 5 )) && (( BASH_VERSINFO[1] < 1 )); }; then
    echo "ERROR: bash >= 5.1 required" >&2; exit 1
fi

# ============ CONFIG ============
N_GPUS=${N_GPUS:-4}
CUDA_DEVICES=${CUDA_DEVICES:-0,1,2,3}
EPOCHS=${EPOCHS:-200}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-30}
LR=${LR:-3e-4}
BATCH=${BATCH:-64}

INPUT_DIR=${INPUT_DIR:-downstream_ft/0506/clean}

# --- Checkpoint: pass via env, no default intentionally ---
# If you only have the ckpt and forgot which config was used, use the
# cold config — it specifies the same backbone architecture (384/12/4)
# and strict=False loading silently ignores any head-weight mismatches.
CKPT=${CKPT:?'Set CKPT=path/to/your.ckpt'}
CONFIG=${CONFIG:-scripts/conf/loqi/loqi_thermo_flow_cold.yaml}

# K=8 standard sampling
K_SS=8
N_STEPS_SS=10
PKL_SS=data/downstream_0506_k8
PT_SS=data/downstream_pt_0506_k8

# K=15 multi-snapshot sampling (5 traj × steps 7,8,9)
N_TRAJ=5
N_STEPS_MS=10
SNAPSHOT_STEPS="7 8 9"
PKL_MS=data/downstream_0506_k15ms
PT_MS=data/downstream_pt_0506_k15ms

# CV output
OUT_ROOT=outputs/downstream_cv_0506
OUT_M0=thermo_init_K8      # naming suffix for M0
OUT_M2=thermo_init_K15ms   # naming suffix for M2

WANDB=${WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-downstream_0506}
LOG_DIR=${LOG_DIR:-/tmp}

# Skip flags (set to 1 to skip a stage if already done)
SKIP_SMI=${SKIP_SMI:-0}
SKIP_SAMPLE_SS=${SKIP_SAMPLE_SS:-0}
SKIP_SAMPLE_MS=${SKIP_SAMPLE_MS:-0}
SKIP_M0=${SKIP_M0:-0}
SKIP_M2=${SKIP_M2:-0}
# ================================

mkdir -p "$PKL_SS" "$PKL_MS" "$PT_SS" "$PT_MS" "$OUT_ROOT" "$LOG_DIR"
[[ -f "$CKPT"   ]] || { echo "ERROR: ckpt not found: $CKPT"     >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }
[[ -d "$INPUT_DIR" ]] || { echo "ERROR: input dir not found: $INPUT_DIR" >&2; exit 1; }

_hdr() {
    echo
    echo "============================================================"
    echo "[$(date +'%F %T')]  $1"
    echo "============================================================"
}

# Auto-discover all CSVs in INPUT_DIR (skip cleaning_report files)
DATASETS_CSV=()
for f in "$INPUT_DIR"/*.csv; do
    [[ "$(basename "$f")" == processing_report* ]] && continue
    [[ "$(basename "$f")" == cleaning_report* ]]   && continue
    DATASETS_CSV+=("$f")
done
echo "Datasets found: ${#DATASETS_CSV[@]}"
for f in "${DATASETS_CSV[@]}"; do echo "  $(basename "$f")"; done

# -----------------------------------------------------------------------
# Stage A — Extract .smi for each dataset (needed for multi-snap sampling)
# -----------------------------------------------------------------------
if [[ "$SKIP_SMI" == "1" ]]; then
    _hdr "Stage A SKIPPED"
else
    _hdr "Stage A — extracting .smi files from $INPUT_DIR"
    for csv in "${DATASETS_CSV[@]}"; do
        name=$(basename "$csv" .csv)
        smi="$PKL_SS/$name.smi"
        if [[ ! -f "$smi" ]]; then
            python scripts/extract_smiles.py \
                --csv "$csv" --out "$smi" --no-dedup \
                >> "$LOG_DIR/0506_extract.log" 2>&1
            echo "  [extracted] $smi"
        else
            echo "  [skip] $smi already exists"
        fi
    done
fi

# -----------------------------------------------------------------------
# Stage B — K=8 standard conformer sampling (reuses sample_downstream_K5.sh)
# -----------------------------------------------------------------------
if [[ "$SKIP_SAMPLE_SS" == "1" ]]; then
    _hdr "Stage B SKIPPED"
elif (( $(find "$PKL_SS" -name "*.pkl" | wc -l) >= ${#DATASETS_CSV[@]} )); then
    _hdr "Stage B SKIPPED (enough pickles already in $PKL_SS)"
else
    _hdr "Stage B — K=$K_SS standard sampling"
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
    K=$K_SS N_STEPS=$N_STEPS_SS \
    OUTPUT_DIR=$PKL_SS \
    INPUT_DIR=$INPUT_DIR \
    FLOW_CKPT=$CKPT FLOW_CONFIG=$CONFIG \
    N_GPUS=$N_GPUS \
        bash scripts/sample_downstream_K5.sh \
        2>&1 | tee -a "$LOG_DIR/0506_sample_k8.log"
fi

# -----------------------------------------------------------------------
# Stage C — K=15 multi-snapshot sampling (5 traj × 3 steps)
# -----------------------------------------------------------------------
if [[ "$SKIP_SAMPLE_MS" == "1" ]]; then
    _hdr "Stage C SKIPPED"
else
    _hdr "Stage C — K=15 multi-snapshot sampling (n_traj=$N_TRAJ, steps=$SNAPSHOT_STEPS)"
    n_done=0
    for csv in "${DATASETS_CSV[@]}"; do
        name=$(basename "$csv" .csv)
        smi="$PKL_SS/$name.smi"     # already extracted in Stage A
        pkl="$PKL_MS/$name.pkl"
        if [[ -f "$pkl" ]]; then
            echo "  [skip] $pkl already exists"
            continue
        fi
        [[ -f "$smi" ]] || { echo "  [WARN] .smi missing for $name, skipping"; continue; }
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
            python scripts/sample_conformers_multistep.py \
                --ckpt "$CKPT" --config "$CONFIG" \
                --input "$smi" --output "$pkl" \
                --n_traj $N_TRAJ --n_steps $N_STEPS_MS \
                --snapshot_steps $SNAPSHOT_STEPS \
                --batch_size $BATCH \
            2>&1 | tee -a "$LOG_DIR/0506_sample_ms_${name}.log"
        n_done=$((n_done+1))
    done
    echo "Multi-snap sampling done for $n_done datasets."
fi

# -----------------------------------------------------------------------
# Helpers for the CV stages
# -----------------------------------------------------------------------

# Auto-build DATASETS array from CSV discovery (all SMILES/TARGET)
_build_datasets_env() {
    local pkl_dir="$1"
    local ds_arr=()
    for csv in "${DATASETS_CSV[@]}"; do
        local name=$(basename "$csv" .csv)
        ds_arr+=("${name}|${name}.csv|${name}.pkl|SMILES|TARGET|0")
    done
    # Return as newline-sep string via stdout (caller reads into array)
    printf '%s\n' "${ds_arr[@]}"
}

_run_cv() {
    local mode="$1"        # M0 or M2
    local pkl_dir="$2"
    local pt_dir="$3"
    local k_eff="$4"
    local out_suffix="$5"

    _hdr "Stage CV / $mode  (K=$k_eff, INIT_FROM_THERMO=1)"

    # Build a temp DATASETS variable for run_downstream_pipeline.sh.
    # We override the DATASETS array by exporting a file that the sub-
    # shell will source — simpler: use ONLY_DATASETS with each name,
    # looping dataset by dataset so the pipeline auto-discovers them.
    # Actually: just call run_downstream_pipeline.sh once with auto-
    # generated DATASETS env.  The pipeline reads the DATASETS array
    # from its own env, so we must export it.
    local ds_list=()
    while IFS= read -r row; do
        ds_list+=("$row")
    done < <(_build_datasets_env "$pkl_dir")

    export DATASETS=("${ds_list[@]}")

    SLEEP_HOURS=0 \
    K=$k_eff EPOCHS=$EPOCHS EARLY_STOP_PATIENCE=$EARLY_STOP_PATIENCE \
    LR=$LR BATCH=$BATCH N_GPUS=$N_GPUS \
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
    INPUT_DIR=$INPUT_DIR \
    PKL_DIR=$pkl_dir \
    PT_DIR=$pt_dir \
    OUT_ROOT=$OUT_ROOT \
    OUT_SUFFIX=$out_suffix \
    CKPT=$CKPT CONFIG=$CONFIG \
    INIT_FROM_THERMO=1 \
    WANDB=$WANDB WANDB_PROJECT=$WANDB_PROJECT WANDB_GROUP=$out_suffix \
        bash scripts/run_downstream_pipeline.sh \
        2>&1 | tee -a "$LOG_DIR/0506_cv_${out_suffix}.log"
}

# -----------------------------------------------------------------------
# Stage D — M0: K=8 standard, thermo warm-init head
# -----------------------------------------------------------------------
[[ "$SKIP_M0" == "1" ]] && _hdr "Stage D SKIPPED" || \
    _run_cv M0 "$PKL_SS" "$PT_SS" $K_SS "$OUT_M0"

# -----------------------------------------------------------------------
# Stage E — M2: K=15 multi-snapshot, thermo warm-init head
# -----------------------------------------------------------------------
[[ "$SKIP_M2" == "1" ]] && _hdr "Stage E SKIPPED" || \
    _run_cv M2 "$PKL_MS" "$PT_MS" 15 "$OUT_M2"

# -----------------------------------------------------------------------
# Final summary
# -----------------------------------------------------------------------
_hdr "FINAL SUMMARY"
python3 - <<PY
import glob, json, os
root = "$OUT_ROOT"
suffixes = ["$OUT_M0", "$OUT_M2"]

print(f"\n{'dataset':<14s}  {'mode':<20s}  {'ens_MAE':>9s}  {'1conf_MAE':>10s}  {'R²(ens)':>9s}  {'best_ep':>8s}")
print("-" * 82)
for sfx in suffixes:
    for rep in sorted(glob.glob(os.path.join(root, f"*_{sfx}/cv_report.json"))):
        ds = os.path.basename(os.path.dirname(rep)).replace(f"_{sfx}", "")
        try:
            d = json.load(open(rep))
            ens = d["mae_mean"]
            pc  = d.get("mae_per_conformer_mean", float("nan"))
            r2  = d["r2_mean"]
            ep  = sum(f.get("best_epoch",0) for f in d.get("folds",[])) / max(len(d.get("folds",[])),1)
            print(f"{ds:<14s}  {sfx:<20s}  {ens:>9.3f}  {pc:>10.3f}  {r2:>9.3f}  {ep:>8.0f}")
        except Exception as e:
            print(f"{ds:<14s}  {sfx:<20s}  (parse error: {e})")
PY

echo
echo "[$(date +'%F %T')] All stages complete."
echo "Results under: $OUT_ROOT/<dataset>_<suffix>/cv_report.json"
