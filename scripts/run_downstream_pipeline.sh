#!/usr/bin/env bash
# End-to-end downstream FT pipeline across all CSVs in downstream_ft/.
#
# Pipeline per dataset:
#   1. (already done by sample_downstream_K5.sh) K=5 conformer pickle
#   2. prepare_downstream_K_pt.py    pickle + CSV → PyG .pt
#   3. downstream_cv.py              5-fold CV with --ensemble-by input_id
#
# 4-GPU worker pool (one dataset per GPU; next gets dispatched the
# moment any GPU frees). Default sleeps 2h before starting so that
# the K=5 sampling has time to finish.
#
# Special handling:
#   * V_cp.csv     uses TARGET; Temperature_K column is IGNORED.
#                  Multiple rows per SMILES at different temperatures
#                  → head learns temperature-averaged prediction.
#   * gas_Hf.csv,
#     liquid_Hf.csv use `mean` as target column; std/n columns ignored.
#   * delaney_s, freesolv_s, lipo_s — pre-split datasets. We MERGE
#     train+valid+test into one CSV+pickle before 5-fold CV (uses all
#     available data; current pipeline doesn't honor the original split).
#
# Usage:
#   nohup bash scripts/run_downstream_pipeline.sh > downstream.log 2>&1 &
#   disown
#
# Override the 2h sleep with SLEEP_HOURS=0  to run immediately.

set -uo pipefail        # NOTE: no -e — we want one dataset's failure not to kill the rest
cd "$(dirname "$0")/.."

# bash 5.1 needed for `wait -n -p`
if (( BASH_VERSINFO[0] < 5 )) || { (( BASH_VERSINFO[0] == 5 )) && (( BASH_VERSINFO[1] < 1 )); }; then
    echo "ERROR: bash >= 5.1 required (you have $BASH_VERSION)" >&2
    exit 1
fi

# ============ CONFIG ============
SLEEP_HOURS=${SLEEP_HOURS:-2}

CKPT=${CKPT:-data/thermo_flow_warm.ckpt}
CONFIG=${CONFIG:-scripts/conf/loqi/loqi_thermo_flow_warm.yaml}

INPUT_DIR=${INPUT_DIR:-downstream_ft}
PKL_DIR=${PKL_DIR:-data/downstream_k5}
PT_DIR=${PT_DIR:-data/downstream_pt}
OUT_ROOT=${OUT_ROOT:-outputs/downstream_cv}

K=${K:-5}
N_GPUS=${N_GPUS:-4}
EPOCHS=${EPOCHS:-100}
LR=${LR:-3e-4}
BATCH=${BATCH:-64}
# ================================

mkdir -p "$PT_DIR" "$OUT_ROOT"

# Pre-flight
[[ -f "$CKPT"   ]] || { echo "ERROR: ckpt not found: $CKPT"     >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }

# ---- Sleep ----
if (( SLEEP_HOURS > 0 )); then
    echo "[$(date +%T)] Sleeping ${SLEEP_HOURS}h before starting (let K=5 sampling finish)..."
    sleep $((SLEEP_HOURS * 3600))
    echo "[$(date +%T)] Awake. Starting pipeline."
fi

# ---- Dataset table -----------------------------------------------------
# Each row: NAME|CSV_REL|PKL_REL|SMILES_COL|TARGET_COL|IS_PRESPLIT
#
# For pre-split (delaney_s/freesolv_s/lipo_s) the script merges
# train+valid+test before prepare. CSV_REL is the directory; PKL_REL is
# also the directory. (No glob needed — we hardcode the 3 split names.)
DATASETS=(
    "Cp|Cp.csv|Cp.pkl|SMILES|TARGET|0"
    "V_cp|V_cp.csv|V_cp.pkl|SMILES|TARGET|0"
    "de|de.csv|de.pkl|SMILES|TARGET|0"
    "gas_Hf|gas_Hf.csv|gas_Hf.pkl|smiles|mean|0"
    "k|k.csv|k.pkl|SMILES|TARGET|0"
    "liquid_Hf|liquid_Hf.csv|liquid_Hf.pkl|smiles|mean|0"
    "delaney_s|delaney_s|delaney_s|SMILES|TARGET|1"
    "freesolv_s|freesolv_s|freesolv_s|SMILES|TARGET|1"
    "lipo_s|lipo_s|lipo_s|SMILES|TARGET|1"
)

# ---- Helpers -----------------------------------------------------------

# Concatenate train/valid/test CSVs into one (single header retained).
_merge_csv() {
    local in_dir="$1" out_csv="$2"
    {
        head -1 "$in_dir/train.csv"
        tail -n +2 "$in_dir/train.csv"
        tail -n +2 "$in_dir/valid.csv"
        tail -n +2 "$in_dir/test.csv"
    } > "$out_csv"
}

# Concatenate per-split sample_conformers.py pickles in train,valid,test order.
_merge_pkl() {
    local in_dir="$1" out_pkl="$2"
    python3 - "$in_dir" "$out_pkl" <<'PY'
import sys, pickle, numpy as np
in_dir, out_path = sys.argv[1], sys.argv[2]
gen, ids = [], []
energies = None
for split in ("train", "valid", "test"):
    with open(f"{in_dir}/{split}.pkl", "rb") as f:
        d = pickle.load(f)
    gen.extend(d["generated"])
    ids.extend(d.get("ids", ["NA"] * len(d["generated"])))
    if "energies" in d:
        e = np.asarray(d["energies"])
        energies = e if energies is None else np.concatenate([energies, e])
out = {"generated": gen, "ids": ids}
if energies is not None:
    out["energies"] = energies
with open(out_path, "wb") as f:
    pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"merged {len(gen)} conformers -> {out_path}")
PY
}

# One dataset end-to-end. Run in a sub-shell with a single GPU pinned.
_run_one() {
    local name=$1 csv_rel=$2 pkl_rel=$3 smi_col=$4 tgt_col=$5 is_split=$6 gpu=$7
    local out_dir="$OUT_ROOT/${name}_warm"
    mkdir -p "$out_dir"

    local csv pkl
    if [[ "$is_split" == "1" ]]; then
        # Pre-split: merge train+valid+test
        csv="$out_dir/_merged.csv"
        pkl="$out_dir/_merged.pkl"
        _merge_csv "$INPUT_DIR/$csv_rel" "$csv"   >> "$out_dir/prep.log" 2>&1
        _merge_pkl "$PKL_DIR/$pkl_rel"   "$pkl"   >> "$out_dir/prep.log" 2>&1
    else
        csv="$INPUT_DIR/$csv_rel"
        pkl="$PKL_DIR/$pkl_rel"
    fi

    local pt="$PT_DIR/${name}.pt"
    if [[ ! -f "$csv" ]]; then
        echo "[$name] MISSING CSV: $csv" >&2; return 2
    fi
    if [[ ! -f "$pkl" ]]; then
        echo "[$name] MISSING PKL: $pkl  (did K=5 sampling finish?)" >&2; return 2
    fi

    # Step 1: build PyG .pt
    if [[ ! -f "$pt" ]]; then
        echo "[$name] preparing PyG dataset (smi_col=$smi_col target_col=$tgt_col)"
        python scripts/prepare_downstream_K_pt.py \
            --conformer-pkl "$pkl" \
            --target-csv    "$csv" \
            --smiles-col    "$smi_col" \
            --target-col    "$tgt_col" \
            --n-confs       "$K" \
            --output        "$pt" \
            >> "$out_dir/prep.log" 2>&1 \
            || { echo "[$name] prepare FAILED, see $out_dir/prep.log" >&2; return 1; }
    fi

    # Step 2: 5-fold CV
    echo "[$name] starting 5-fold CV on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python scripts/downstream_cv.py \
        --ckpt   "$CKPT"   --config "$CONFIG" \
        --dataset-pt "$pt" \
        --ensemble-by input_id \
        --out-dir "$out_dir" \
        --n-folds 5 --epochs "$EPOCHS" --lr "$LR" \
        --batch-size "$BATCH" \
        --device cuda \
        >> "$out_dir/cv.log" 2>&1 \
        || { echo "[$name] CV FAILED, see $out_dir/cv.log" >&2; return 1; }

    echo "[$name] DONE"
}

# ---- Worker pool dispatch ---------------------------------------------
echo "[$(date +%T)] Dispatching ${#DATASETS[@]} datasets across $N_GPUS GPUs"
declare -A GPU_OF_PID
declare -A NAME_OF_PID

idx=0
n_total=${#DATASETS[@]}
n_done=0
n_failed=0
n_started=0

# Seed pool
for ((gpu=0; gpu<N_GPUS && idx<n_total; gpu++)); do
    IFS='|' read -r name csv_rel pkl_rel smi_col tgt_col is_split <<< "${DATASETS[$idx]}"
    ( _run_one "$name" "$csv_rel" "$pkl_rel" "$smi_col" "$tgt_col" "$is_split" "$gpu" ) &
    pid=$!
    GPU_OF_PID[$pid]=$gpu
    NAME_OF_PID[$pid]="$name"
    n_started=$((n_started + 1))
    idx=$((idx + 1))
    echo "[$(date +%T)] [${n_started}/${n_total}] launch $name on GPU $gpu  (pid=$pid)"
done

# Drain
finished_pid=0
while (( ${#GPU_OF_PID[@]} > 0 )); do
    # Capture child exit status — note: NO `|| true`, that would mask
    # non-zero status (the original bug that reported "0 failed" when
    # all 9 datasets failed because the K=5 pickles were missing).
    wait -n -p finished_pid
    status=$?
    if [[ -z "${GPU_OF_PID[$finished_pid]:-}" ]]; then
        continue
    fi
    gpu="${GPU_OF_PID[$finished_pid]}"
    name="${NAME_OF_PID[$finished_pid]}"
    unset 'GPU_OF_PID[$finished_pid]'
    unset 'NAME_OF_PID[$finished_pid]'
    n_done=$((n_done + 1))
    if (( status != 0 )); then
        n_failed=$((n_failed + 1))
        echo "[$(date +%T)] [${n_done}/${n_total}] FAIL  $name  (gpu=$gpu status=$status)"
    else
        echo "[$(date +%T)] [${n_done}/${n_total}] done  $name  (gpu=$gpu)"
    fi
    if (( idx < n_total )); then
        IFS='|' read -r name csv_rel pkl_rel smi_col tgt_col is_split <<< "${DATASETS[$idx]}"
        ( _run_one "$name" "$csv_rel" "$pkl_rel" "$smi_col" "$tgt_col" "$is_split" "$gpu" ) &
        pid=$!
        GPU_OF_PID[$pid]=$gpu
        NAME_OF_PID[$pid]="$name"
        n_started=$((n_started + 1))
        idx=$((idx + 1))
        echo "[$(date +%T)] [${n_started}/${n_total}] launch $name on GPU $gpu  (pid=$pid)"
    fi
done

# ---- Summary ----------------------------------------------------------
echo
echo "============================================================"
echo "  Pipeline done: $n_done total, $n_failed failed"
echo "============================================================"
python3 - "$OUT_ROOT" <<'PY'
import sys, json, glob, os
root = sys.argv[1]
print(f"\n{'dataset':<14s}  {'MAE':>10s}  {'RMSE':>10s}  {'R2':>8s}  {'n_folds':>8s}")
print("-" * 60)
for rep in sorted(glob.glob(os.path.join(root, "*_warm/cv_report.json"))):
    name = os.path.basename(os.path.dirname(rep)).replace("_warm", "")
    try:
        d = json.load(open(rep))
        print(f"{name:<14s}  {d['mae_mean']:>10.4f}  {d['rmse_mean']:>10.4f}  "
              f"{d['r2_mean']:>8.3f}  {len(d['folds']):>8d}")
    except Exception as e:
        print(f"{name:<14s}  (parse error: {e})")
PY
