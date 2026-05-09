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

# INPUT_DIR defaults to downstream_ft/clean (the deduplicated/filtered
# output of scripts/clean_downstream.py). Pre-split datasets are FLAT
# in the clean dir (delaney_s/freesolv_s/lipo_s merged into one CSV
# with a `_split` column preserved for reference). Override with
# INPUT_DIR=downstream_ft to reproduce the original raw-data pipeline.
INPUT_DIR=${INPUT_DIR:-downstream_ft/clean}
PKL_DIR=${PKL_DIR:-data/downstream_k5}
PT_DIR=${PT_DIR:-data/downstream_pt}
OUT_ROOT=${OUT_ROOT:-outputs/downstream_cv}

K=${K:-5}
N_GPUS=${N_GPUS:-4}
BASE_GPU=${BASE_GPU:-0}   # physical GPU offset; outer scripts set BASE_GPU=X instead of CUDA_VISIBLE_DEVICES
EPOCHS=${EPOCHS:-100}
LR=${LR:-3e-4}

# wandb (opt-in). Set WANDB=1 to enable; one wandb run per dataset.
WANDB=${WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-downstream_cv}
WANDB_GROUP=${WANDB_GROUP:-warm}

# Set INIT_FROM_THERMO=1 to warm-start the downstream head's AtomMolMP
# from the ckpt's trained thermo head (auto-aligns head dims to the
# ckpt's thermo_head_args; final Linear stays random because output dim
# 5→1 differs). When 0, HEAD_HIDDEN / N_MP_LAYERS / MP_N_HEADS take effect.
INIT_FROM_THERMO=${INIT_FROM_THERMO:-0}
HEAD_HIDDEN=${HEAD_HIDDEN:-256}        # ignored when INIT_FROM_THERMO=1
N_MP_LAYERS=${N_MP_LAYERS:-4}          # ignored when INIT_FROM_THERMO=1
MP_N_HEADS=${MP_N_HEADS:-4}            # ignored when INIT_FROM_THERMO=1
BATCH=${BATCH:-64}

EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-0}   # 0 = disabled

# LoRA-adapter FT (backbone unfrozen via low-rank deltas). 0 = disabled
# (head-only on cached H). Useful for breaking the H ceiling without
# losing the base ckpt's generation capability — no LoRA → original
# generative behavior; with LoRA → property-tuned behavior.
LORA_R=${LORA_R:-0}
LORA_ALPHA=${LORA_ALPHA:-}                       # empty → defaults to LORA_R
LORA_TARGET=${LORA_TARGET:-qkv_proj,out_projection}

# Suffix appended to each dataset's per-mode output dir so multiple modes
# (warm vs cold-small vs cold-large) don't collide under OUT_ROOT.
OUT_SUFFIX=${OUT_SUFFIX:-warm}

# Subset filter: comma-separated dataset names. Either keeps only those
# (ONLY_DATASETS=lipo_s) or skips them (SKIP_DATASETS=lipo_s,delaney_s).
# Useful for splitting heavy jobs across multiple machines/sessions —
# e.g. lipo_s alone on one box, the other 8 on another.
ONLY_DATASETS=${ONLY_DATASETS:-}
SKIP_DATASETS=${SKIP_DATASETS:-}
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
# When INPUT_DIR=downstream_ft/clean (default): all 9 datasets are FLAT
# CSVs in INPUT_DIR (delaney_s/freesolv_s/lipo_s have already been
# merged from train+valid+test by clean_downstream.py), so IS_PRESPLIT=0
# everywhere.
#
# When INPUT_DIR=downstream_ft (raw): delaney_s/freesolv_s/lipo_s exist
# as subdirs with train/valid/test.csv. Use IS_PRESPLIT=1 in that case
# (uncomment the alt block below).
# Order by descending size after K=8 cleaning ("longest processing time
# first" / LPT scheduling). Largest job lipo_s starts in the seed pool
# and runs in parallel with the others, so by the time lipo finishes
# everyone else has already cycled through. With alphabetical order
# lipo_s used to start last, holding 1 GPU while the other 3 sat idle.
# Ensure DATASETS is initialised so set -u doesn't fire on ${#DATASETS[@]}.
# Callers can pre-populate it; if empty, auto-discovery below fills it in.
DATASETS=("${DATASETS[@]+"${DATASETS[@]}"}")

# Auto-discover datasets from INPUT_DIR when the caller hasn't already
# exported a DATASETS array. Discovers every *.csv (skips report files),
# inspects the SMILES and TARGET column names, and registers each as a
# flat dataset. Works for any clean directory (0403, 0506, …).
#
# Falls back to the hardcoded table below only when INPUT_DIR contains
# none of the known flat CSVs (e.g. raw downstream_ft/ with presplit
# subdirs — see the commented-out block at the bottom for that case).
if [[ ${#DATASETS[@]} -eq 0 ]]; then
    DATASETS=()
    # Sort CSVs by descending row count (LPT scheduling: largest job starts
    # first so it overlaps with smaller ones and all GPUs stay busy).
    while IFS= read -r csv; do
        name=$(basename "$csv" .csv)
        [[ "$name" == *report* ]] && continue
        # Detect SMILES column name (SMILES or smiles)
        _smi=$(python3 -c "
import pandas as pd, sys
df = pd.read_csv('$csv', nrows=0, encoding='utf-8-sig')
cols = [c for c in df.columns if c.lower() == 'smiles']
print(cols[0] if cols else 'SMILES')" 2>/dev/null)
        # Detect TARGET column name (TARGET or mean, etc.)
        _tgt=$(python3 -c "
import pandas as pd, sys
df = pd.read_csv('$csv', nrows=0, encoding='utf-8-sig')
skip = {c for c in df.columns if c.lower() in ('smiles','_split','split','n','std')}
pref = ['TARGET','target','mean','y','value']
tgt = next((c for c in pref if c in df.columns), None)
if tgt is None:
    tgt = next((c for c in df.columns if c not in skip), 'TARGET')
print(tgt)" 2>/dev/null)
        DATASETS+=("${name}|${name}.csv|${name}.pkl|${_smi}|${_tgt}|0")
    done < <(
        # Sort by descending row count for LPT scheduling
        for f in "$INPUT_DIR"/*.csv; do
            [[ "$(basename "$f")" == *report* ]] && continue
            n=$(wc -l < "$f" 2>/dev/null || echo 0)
            echo "$n $f"
        done | sort -rn | awk '{print $2}'
    )
fi

# Fallback hardcoded table (LPT-ordered for 0403 nine-dataset benchmark).
# Only active when INPUT_DIR has no CSVs and caller didn't export DATASETS.
if [[ ${#DATASETS[@]} -eq 0 ]]; then
DATASETS=(
    "lipo_s|lipo_s.csv|lipo_s.pkl|SMILES|TARGET|0"
    "gas_Hf|gas_Hf.csv|gas_Hf.pkl|SMILES|TARGET|0"
    "liquid_Hf|liquid_Hf.csv|liquid_Hf.pkl|SMILES|TARGET|0"
    "Cp|Cp.csv|Cp.pkl|SMILES|TARGET|0"
    "delaney_s|delaney_s.csv|delaney_s.pkl|SMILES|TARGET|0"
    "V_cp|V_cp.csv|V_cp.pkl|SMILES|TARGET|0"
    "de|de.csv|de.pkl|SMILES|TARGET|0"
    "k|k.csv|k.pkl|SMILES|TARGET|0"
    "freesolv_s|freesolv_s.csv|freesolv_s.pkl|SMILES|TARGET|0"
)
fi

# ---- Helpers -----------------------------------------------------------

# Concatenate train/valid/test CSVs into one (single header retained).
# Prefers the filtered.csv emitted by sample_downstream_K5.sh's
# extract_smiles (lives under PKL_DIR/<dataset>/). Falls back to the
# original CSV under INPUT_DIR/<dataset>/ when filtered.csv is absent
# (e.g., pickles were sampled before extract_smiles was added).
_merge_csv() {
    local pkl_dir="$1" input_dir="$2" out_csv="$3"
    local files=()
    for split in train valid test; do
        if [[ -f "$pkl_dir/${split}.filtered.csv" ]]; then
            files+=("$pkl_dir/${split}.filtered.csv")
        elif [[ -f "$input_dir/${split}.csv" ]]; then
            files+=("$input_dir/${split}.csv")
        else
            echo "ERROR: neither $pkl_dir/${split}.filtered.csv nor $input_dir/${split}.csv exists" >&2
            return 1
        fi
    done
    {
        head -1 "${files[0]}"
        for f in "${files[@]}"; do
            tail -n +2 "$f"
        done
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
    local out_dir="$OUT_ROOT/${name}_${OUT_SUFFIX}"
    mkdir -p "$out_dir"

    local csv pkl
    if [[ "$is_split" == "1" ]]; then
        # Pre-split: merge train+valid+test. `_merge_csv` prefers
        # filtered.csv from PKL_DIR (post-validation row order matches the
        # pickle's conformer order); falls back to the original CSV under
        # INPUT_DIR when filtered.csv isn't present. `csv_rel` is the
        # subdir name (e.g. "lipo_s") in both directories.
        csv="$out_dir/_merged.csv"
        pkl="$out_dir/_merged.pkl"
        _merge_csv "$PKL_DIR/$pkl_rel" "$INPUT_DIR/$csv_rel" "$csv"  >> "$out_dir/prep.log" 2>&1 \
            || { echo "[$name] _merge_csv FAILED, see $out_dir/prep.log" >&2; return 2; }
        _merge_pkl "$PKL_DIR/$pkl_rel" "$pkl"   >> "$out_dir/prep.log" 2>&1
    else
        # Flat CSV: prefer the filtered.csv emitted by sample_downstream_K5's
        # extract_smiles (positions match the pickle). Fall back to original
        # CSV if filter wasn't run.
        if [[ -f "$PKL_DIR/${csv_rel%.csv}.filtered.csv" ]]; then
            csv="$PKL_DIR/${csv_rel%.csv}.filtered.csv"
        else
            csv="$INPUT_DIR/$csv_rel"
        fi
        pkl="$PKL_DIR/$pkl_rel"
    fi

    local pt="$PT_DIR/${name}_K${K}.pt"
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
    local wandb_args=""
    if [[ "$WANDB" == "1" ]]; then
        wandb_args="--wandb --wandb-project $WANDB_PROJECT --wandb-group $WANDB_GROUP --wandb-name ${name}_${WANDB_GROUP}"
    fi
    local warm_args=""
    if [[ "$INIT_FROM_THERMO" == "1" ]]; then
        warm_args="--init-head-from-thermo"
    else
        warm_args="--head-hidden $HEAD_HIDDEN --n-mp-layers $N_MP_LAYERS --mp-n-heads $MP_N_HEADS"
    fi
    local stop_args=""
    if (( EARLY_STOP_PATIENCE > 0 )); then
        stop_args="--early-stopping-patience $EARLY_STOP_PATIENCE"
    fi
    local lora_args=""
    if (( LORA_R > 0 )); then
        lora_args="--lora-r $LORA_R --lora-target $LORA_TARGET"
        if [[ -n "$LORA_ALPHA" ]]; then
            lora_args="$lora_args --lora-alpha $LORA_ALPHA"
        fi
    fi
    # Adaptive epochs flag (set AUTO_EPOCHS=1 to enable)
    local epoch_args="--epochs $EPOCHS"
    if [[ "${AUTO_EPOCHS:-0}" == "1" ]]; then
        epoch_args="--auto-epochs --epochs-large ${EPOCHS_LARGE:-200} --epochs-small ${EPOCHS_SMALL:-150}"
    fi
    # Cap K conformers on training side (e.g., MAX_K_PER_INPUT=5 for K=5-from-K=8)
    local maxk_args=""
    if [[ -n "${MAX_K_PER_INPUT:-}" ]] && (( MAX_K_PER_INPUT > 0 )); then
        maxk_args="--max-k-per-input $MAX_K_PER_INPUT"
    fi

    CUDA_VISIBLE_DEVICES=$gpu python scripts/downstream_cv.py \
        --ckpt   "$CKPT"   --config "$CONFIG" \
        --dataset-pt "$pt" \
        --ensemble-by input_id \
        --out-dir "$out_dir" \
        --n-folds 5 --lr "$LR" \
        --batch-size "$BATCH" \
        --device cuda \
        $epoch_args $wandb_args $warm_args $stop_args $lora_args $maxk_args \
        >> "$out_dir/cv.log" 2>&1 \
        || { echo "[$name] CV FAILED, see $out_dir/cv.log" >&2; return 1; }

    echo "[$name] DONE"
}

# ---- Apply ONLY_DATASETS / SKIP_DATASETS filters ----------------------
if [[ -n "$ONLY_DATASETS" || -n "$SKIP_DATASETS" ]]; then
    _only=",$ONLY_DATASETS,"
    _skip=",$SKIP_DATASETS,"
    filtered=()
    for row in "${DATASETS[@]}"; do
        IFS='|' read -r _name _rest <<< "$row"
        if [[ -n "$ONLY_DATASETS" && "$_only" != *",$_name,"* ]]; then
            continue
        fi
        if [[ -n "$SKIP_DATASETS" && "$_skip" == *",$_name,"* ]]; then
            continue
        fi
        filtered+=("$row")
    done
    DATASETS=("${filtered[@]}")
    echo "[$(date +%T)] Filter applied — running ${#DATASETS[@]} dataset(s):"
    for row in "${DATASETS[@]}"; do
        IFS='|' read -r _name _rest <<< "$row"
        echo "  - $_name"
    done
fi
if [[ ${#DATASETS[@]} -eq 0 ]]; then
    echo "[$(date +%T)] No datasets to run after filtering. Exiting." >&2
    exit 0
fi

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
    phys_gpu=$(( BASE_GPU + gpu ))
    IFS='|' read -r name csv_rel pkl_rel smi_col tgt_col is_split <<< "${DATASETS[$idx]}"
    ( _run_one "$name" "$csv_rel" "$pkl_rel" "$smi_col" "$tgt_col" "$is_split" "$phys_gpu" ) &
    pid=$!
    GPU_OF_PID[$pid]=$phys_gpu
    NAME_OF_PID[$pid]="$name"
    n_started=$((n_started + 1))
    idx=$((idx + 1))
    echo "[$(date +%T)] [${n_started}/${n_total}] launch $name on GPU $phys_gpu  (pid=$pid)"
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
    phys_gpu="${GPU_OF_PID[$finished_pid]}"
    name="${NAME_OF_PID[$finished_pid]}"
    unset 'GPU_OF_PID[$finished_pid]'
    unset 'NAME_OF_PID[$finished_pid]'
    n_done=$((n_done + 1))
    if (( status != 0 )); then
        n_failed=$((n_failed + 1))
        echo "[$(date +%T)] [${n_done}/${n_total}] FAIL  $name  (gpu=$phys_gpu status=$status)"
    else
        echo "[$(date +%T)] [${n_done}/${n_total}] done  $name  (gpu=$phys_gpu)"
    fi
    if (( idx < n_total )); then
        IFS='|' read -r name csv_rel pkl_rel smi_col tgt_col is_split <<< "${DATASETS[$idx]}"
        ( _run_one "$name" "$csv_rel" "$pkl_rel" "$smi_col" "$tgt_col" "$is_split" "$phys_gpu" ) &
        pid=$!
        GPU_OF_PID[$pid]=$phys_gpu
        NAME_OF_PID[$pid]="$name"
        n_started=$((n_started + 1))
        idx=$((idx + 1))
        echo "[$(date +%T)] [${n_started}/${n_total}] launch $name on GPU $phys_gpu  (pid=$pid)"
    fi
done

# ---- Summary ----------------------------------------------------------
echo
echo "============================================================"
echo "  Pipeline done: $n_done total, $n_failed failed"
echo "============================================================"
python3 - "$OUT_ROOT" "$OUT_SUFFIX" <<'PY'
import sys, json, glob, os
root, suffix = sys.argv[1], sys.argv[2]
print(f"\n{'dataset':<14s}  {'MAE':>10s}  {'RMSE':>10s}  {'R2':>8s}  {'n_folds':>8s}")
print("-" * 60)
for rep in sorted(glob.glob(os.path.join(root, f"*_{suffix}/cv_report.json"))):
    name = os.path.basename(os.path.dirname(rep)).replace(f"_{suffix}", "")
    try:
        d = json.load(open(rep))
        print(f"{name:<14s}  {d['mae_mean']:>10.4f}  {d['rmse_mean']:>10.4f}  "
              f"{d['r2_mean']:>8.3f}  {len(d['folds']):>8d}")
    except Exception as e:
        print(f"{name:<14s}  (parse error: {e})")
PY
