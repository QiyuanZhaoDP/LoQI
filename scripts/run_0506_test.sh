#!/usr/bin/env bash
# Full 0506 downstream benchmark: 4 ckpts × 2 sampling modes × 13 datasets.
#
# Checkpoints:
#   loqi_flow            — 256-dim, no thermo (blank 2D→3D baseline)
#   thermo_flow_warm     — 384/12, thermo-trained (despite the name "warm")
#   thermo_flow_cold_early — 384/12, thermo-trained, early checkpoint
#   thermo_flow_cold_late  — 384/12, thermo-trained, later checkpoint
#
# Sampling modes:
#   K=8   standard   : 8 independent trajectories, 1 conformer each
#   K=12  multi-snap : 4 trajectories × snapshot steps 7,8,9 = 12 conformers
#
# Downstream FT (all modes):
#   INIT_FROM_THERMO=1, head dims 4/4/256
#   loqi_flow has no thermo head → load_thermo_head_into returns 0 →
#   automatically falls back to random init (serves as baseline).
#
# Usage (run from repo root):
#   nohup bash scripts/run_0506_test.sh > /tmp/0506_test.log 2>&1 &
#   disown

set -uo pipefail
cd "$(dirname "$0")/.."

if (( BASH_VERSINFO[0] < 5 )) || { (( BASH_VERSINFO[0] == 5 )) && (( BASH_VERSINFO[1] < 1 )); }; then
    echo "ERROR: bash >= 5.1 required" >&2; exit 1
fi

# ============ CONFIG ============
N_GPUS=${N_GPUS:-8}
CUDA_DEVICES=${CUDA_DEVICES:-0,1,2,3,4,5,6,7}
EPOCHS=${EPOCHS:-150}           # fallback if AUTO_EPOCHS=0
AUTO_EPOCHS=${AUTO_EPOCHS:-1}   # 1=enable adaptive: >2000 → 200ep, ≤2000 → 150ep
EPOCHS_LARGE=${EPOCHS_LARGE:-200}
EPOCHS_SMALL=${EPOCHS_SMALL:-150}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-100}
LR=${LR:-3e-4}
BATCH=${BATCH:-64}

INPUT_DIR=${INPUT_DIR:-downstream_ft/0506/cleaned_by_codex}

# ---- Checkpoint definitions ------------------------------------------------
# Format: "label|ckpt_path|config_path|init_from_thermo"
# init_from_thermo=1 → warm-start downstream head from ckpt's thermo_heads
# init_from_thermo=0 → random-init head (loqi_flow has no thermo head)
# 9 combinations:
#   loqi_flow            × cold init only  (no thermo head to load)
#   thermo_flow_cold_early × cold + warm init
#   thermo_flow_cold_late  × cold + warm init
#   thermo_flow_warm_large × cold + warm init
#   thermo_flow_warm_small × cold + warm init
#
# Format: "label|ckpt_path|config_path|init_from_thermo"
# Run `python scripts/inspect_ckpt.py <path>` to verify config before
# adjusting the yaml below (warm_small=256-dim, warm_large=384-dim assumed).
CKPT_DEFS=(
    # ---- loqi_flow: no thermo head, always random init ----
    "loqi_flow|data/ft_ckpts/loqi_flow.ckpt|scripts/conf/loqi/loqi_flow.yaml|0"

    # ---- cold-start backbone (384/12) × 2 init modes ----
    "cold_early_c|data/ft_ckpts/thermo_flow_cold_early.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold.yaml|0"
    "cold_early_w|data/ft_ckpts/thermo_flow_cold_early.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold.yaml|1"
    "cold_late_c|data/ft_ckpts/thermo_flow_cold_late.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold.yaml|0"
    "cold_late_w|data/ft_ckpts/thermo_flow_cold_late.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold.yaml|1"

    # ---- warm-start large backbone (384/12) × 2 init modes ----
    "warm_large_c|data/ft_ckpts/thermo_flow_warm_large.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold.yaml|0"
    "warm_large_w|data/ft_ckpts/thermo_flow_warm_large.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold.yaml|1"

    # ---- warm-start small backbone (256/10) × 2 init modes ----
    # If inspect_ckpt shows d=384, change yaml to loqi_thermo_flow_cold.yaml
    "warm_small_c|data/ft_ckpts/thermo_flow_warm_small.ckpt|scripts/conf/loqi/loqi_thermo_flow_warm.yaml|0"
    "warm_small_w|data/ft_ckpts/thermo_flow_warm_small.ckpt|scripts/conf/loqi/loqi_thermo_flow_warm.yaml|1"
)

# ---- Sampling parameters ---------------------------------------------------
# K=8 standard
K_SS=8
N_STEPS_SS=10

# K=12 multi-snapshot: 4 traj × steps 7,8,9
N_TRAJ=4
N_STEPS_MS=10
SNAPSHOT_STEPS="7 8 9"   # 3 snapshots → K = 4×3 = 12

# ---- Downstream FT ---------------------------------------------------------
HEAD_HIDDEN=256
N_MP_LAYERS=4
MP_N_HEADS=4

# ---- Output ----------------------------------------------------------------
OUT_ROOT=outputs/downstream_cv_0506

WANDB=${WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-downstream_0506}
LOG_DIR=${LOG_DIR:-/tmp}

# Skip flags
SKIP_SMI=${SKIP_SMI:-0}
SKIP_SAMPLE=${SKIP_SAMPLE:-0}
SKIP_CV=${SKIP_CV:-0}

# Tasks per GPU for CV (memory: ~3-15GB per task; 4× safe on 80GB A100)
TASKS_PER_GPU=${TASKS_PER_GPU:-1}
# ================================

mkdir -p "$OUT_ROOT" "$LOG_DIR"
[[ -d "$INPUT_DIR" ]] || { echo "ERROR: $INPUT_DIR not found" >&2; exit 1; }

# Pre-flight: verify all ckpts + configs exist
for def in "${CKPT_DEFS[@]}"; do
    IFS='|' read -r label ckpt cfg _init <<< "$def"
    [[ -f "$ckpt" ]] || { echo "ERROR: ckpt not found: $ckpt ($label)" >&2; exit 1; }
    [[ -f "$cfg"  ]] || { echo "ERROR: config not found: $cfg ($label)" >&2; exit 1; }
done

_hdr() { echo; echo "============================================================"; echo "[$(date +'%F %T')]  $1"; echo "============================================================"; }

# Auto-discover datasets
DATASETS_CSV=()
for f in "$INPUT_DIR"/*.csv; do
    [[ "$(basename "$f")" == *report* ]] && continue
    DATASETS_CSV+=("$f")
done
echo "Datasets (${#DATASETS_CSV[@]}):"
for f in "${DATASETS_CSV[@]}"; do echo "  $(basename "$f")"; done

# -----------------------------------------------------------------------
# Stage A — Extract .smi (shared across all ckpts)
# -----------------------------------------------------------------------
SMI_DIR="$LOG_DIR/0506_smi"
mkdir -p "$SMI_DIR"

if [[ "$SKIP_SMI" == "1" ]]; then
    _hdr "Stage A SKIPPED"
else
    _hdr "Stage A — extract .smi"
    for csv in "${DATASETS_CSV[@]}"; do
        name=$(basename "$csv" .csv)
        smi="$SMI_DIR/$name.smi"
        [[ -f "$smi" ]] && continue
        python scripts/extract_smiles.py --csv "$csv" --out "$smi" --no-dedup \
            >> "$LOG_DIR/0506_extract.log" 2>&1
        echo "  [ok] $smi"
    done
fi

# -----------------------------------------------------------------------
# Stage B — Flat work-queue sampling across all ckpts + datasets + modes
#
# Old design: per-ckpt sequential → 1 GPU busy while 7 idle for K=12.
# New design: collect ALL (ckpt × dataset × mode) tasks into a flat
# queue, then dispatch using a N_GPUS worker pool (one task per GPU).
# Each GPU runs one sampling job; as it finishes, the next task starts.
# -----------------------------------------------------------------------
if [[ "$SKIP_SAMPLE" == "1" ]]; then
    _hdr "Sampling SKIPPED"
else
    # Parse CUDA_DEVICES into an array of individual GPU IDs
    IFS=',' read -ra _GPU_IDS <<< "$CUDA_DEVICES"
    _N_POOL=${#_GPU_IDS[@]}

    # Build task queue: "mode|label|ckpt|cfg|smi|pkl|name"
    _QUEUE=()
    for def in "${CKPT_DEFS[@]}"; do
        IFS='|' read -r label ckpt cfg _init <<< "$def"
        pkl_ss="data/0506_pkl_${label}_k8"
        pkl_ms="data/0506_pkl_${label}_k12ms"
        mkdir -p "$pkl_ss" "$pkl_ms"
        for csv in "${DATASETS_CSV[@]}"; do
            name=$(basename "$csv" .csv)
            smi="$SMI_DIR/$name.smi"
            [[ -f "$smi" ]] || { echo "  [WARN] no .smi $name ($label)"; continue; }
            pkl_k8="$pkl_ss/$name.pkl"
            pkl_k12="$pkl_ms/$name.pkl"
            [[ -f "$pkl_k8"  ]] || _QUEUE+=("k8|$label|$ckpt|$cfg|$smi|$pkl_k8|$name")
            [[ -f "$pkl_k12" ]] || _QUEUE+=("k12|$label|$ckpt|$cfg|$smi|$pkl_k12|$name")
        done
    done

    total_tasks=${#_QUEUE[@]}
    _hdr "Stage B — dispatching $total_tasks sampling tasks across ${_N_POOL} GPUs"

    if (( total_tasks == 0 )); then
        echo "  all pickles already exist — skipping"
    else
        declare -A _PID_GPU
        declare -A _PID_TAG
        _done=0; _fail=0; _idx=0

        # Seed pool
        for (( _gi=0; _gi < _N_POOL && _idx < total_tasks; _gi++, _idx++ )); do
            IFS='|' read -r _mode _lbl _ckpt _cfg _smi _pkl _name <<< "${_QUEUE[$_idx]}"
            _gpu="${_GPU_IDS[$_gi]}"
            if [[ "$_mode" == "k8" ]]; then
                CUDA_VISIBLE_DEVICES=$_gpu python scripts/sample_conformers.py \
                    --ckpt "$_ckpt" --config "$_cfg" \
                    --input "$_smi" --output "$_pkl" \
                    --n_confs $K_SS --n_steps $N_STEPS_SS \
                    --batch_size $BATCH --postprocess none \
                    >> "$LOG_DIR/0506_${_lbl}_${_name}_k8.log" 2>&1 &
            else
                CUDA_VISIBLE_DEVICES=$_gpu python scripts/sample_conformers_multistep.py \
                    --ckpt "$_ckpt" --config "$_cfg" \
                    --input "$_smi" --output "$_pkl" \
                    --n_traj $N_TRAJ --n_steps $N_STEPS_MS \
                    --snapshot_steps $SNAPSHOT_STEPS \
                    --batch_size $BATCH \
                    >> "$LOG_DIR/0506_${_lbl}_${_name}_k12.log" 2>&1 &
            fi
            _pid=$!
            _PID_GPU[$_pid]=$_gpu
            _PID_TAG[$_pid]="${_mode}/${_lbl}/${_name}"
            echo "[$(date +%T)] [$((_idx))/$total_tasks] launch ${_mode} ${_lbl}/${_name} → GPU $_gpu (pid=$_pid)"
        done

        # Drain queue
        while (( ${#_PID_GPU[@]} > 0 )); do
            _fpid=0
            wait -n -p _fpid 2>/dev/null || true
            _fpid=${_fpid:-0}
            [[ -z "${_PID_GPU[$_fpid]:-}" ]] && continue
            _gpu="${_PID_GPU[$_fpid]}"
            _tag="${_PID_TAG[$_fpid]}"
            unset '_PID_GPU[$_fpid]' '_PID_TAG[$_fpid]'
            _done=$((_done+1))
            _wait=$?
            (( _wait != 0 )) && { _fail=$((_fail+1)); echo "[$(date +%T)] FAIL $_tag (gpu=$_gpu)"; } \
                             || echo "[$(date +%T)] done $_tag (gpu=$_gpu)"
            # dispatch next
            if (( _idx < total_tasks )); then
                IFS='|' read -r _mode _lbl _ckpt _cfg _smi _pkl _name <<< "${_QUEUE[$_idx]}"
                _idx=$((_idx+1))
                if [[ "$_mode" == "k8" ]]; then
                    CUDA_VISIBLE_DEVICES=$_gpu python scripts/sample_conformers.py \
                        --ckpt "$_ckpt" --config "$_cfg" \
                        --input "$_smi" --output "$_pkl" \
                        --n_confs $K_SS --n_steps $N_STEPS_SS \
                        --batch_size $BATCH --postprocess none \
                        >> "$LOG_DIR/0506_${_lbl}_${_name}_k8.log" 2>&1 &
                else
                    CUDA_VISIBLE_DEVICES=$_gpu python scripts/sample_conformers_multistep.py \
                        --ckpt "$_ckpt" --config "$_cfg" \
                        --input "$_smi" --output "$_pkl" \
                        --n_traj $N_TRAJ --n_steps $N_STEPS_MS \
                        --snapshot_steps $SNAPSHOT_STEPS \
                        --batch_size $BATCH \
                        >> "$LOG_DIR/0506_${_lbl}_${_name}_k12.log" 2>&1 &
                fi
                _pid=$!
                _PID_GPU[$_pid]=$_gpu
                _PID_TAG[$_pid]="${_mode}/${_lbl}/${_name}"
                echo "[$(date +%T)] [$_idx/$total_tasks] launch ${_mode} ${_lbl}/${_name} → GPU $_gpu (pid=$_pid)"
            fi
        done

        echo
        echo "Sampling done: $_done total, $_fail failed"
    fi
fi

# -----------------------------------------------------------------------
# Stage C — Flat work-queue CV: all (ckpt × mode × dataset) in parallel
#
# Old design: 18 configs run sequentially, each using all N_GPUS via
# run_downstream_pipeline.sh — GPUs idle between config transitions.
# New design: one task = one dataset's prepare + CV on ONE GPU.
# All (9 ckpts × 2 modes × N_datasets) tasks enter a flat queue
# dispatched across N_GPUS; GPUs stay fully busy throughout.
# -----------------------------------------------------------------------
if [[ "$SKIP_CV" == "1" ]]; then
    _hdr "CV SKIPPED"
else
    IFS=',' read -ra _BASE_GPU_IDS <<< "$CUDA_DEVICES"
    _CV_GPU_IDS=()
    for _g in "${_BASE_GPU_IDS[@]}"; do
        for (( _t=0; _t<TASKS_PER_GPU; _t++ )); do _CV_GPU_IDS+=("$_g"); done
    done
    _CV_N_POOL=${#_CV_GPU_IDS[@]}

    # Build flat CV task queue: "suffix|ckpt|cfg|init|pkl_dir|pt_dir|k|ds_name|csv"
    _CV_QUEUE=()
    for def in "${CKPT_DEFS[@]}"; do
        IFS='|' read -r label ckpt cfg init_thermo <<< "$def"
        for _mode_k in "K8:$K_SS:data/0506_pkl_${label}_k8:data/0506_pt_${label}_k8" \
                        "K12ms:12:data/0506_pkl_${label}_k12ms:data/0506_pt_${label}_k12ms"; do
            IFS=':' read -r _mode_tag _keff _pkl_dir _pt_dir <<< "$_mode_k"
            _suffix="${label}_${_mode_tag}"
            mkdir -p "$_pt_dir" "$OUT_ROOT"
            [[ -d "$_pkl_dir" ]] || continue
            for csv in "${DATASETS_CSV[@]}"; do
                _ds=$(basename "$csv" .csv)
                _ds_pkl=$(find "$_pkl_dir" -name "${_ds}.pkl" 2>/dev/null | head -1)
                [[ -n "$_ds_pkl" ]] || continue   # no pickle → skip
                # Skip if cv_report already exists (safe to restart)
                _cv_done_flag="$OUT_ROOT/${_ds}_${_suffix}/cv_report.json"
                [[ -f "$_cv_done_flag" ]] && { echo "  [skip CV] ${_suffix}/${_ds}"; continue; }
                _CV_QUEUE+=("$_suffix|$ckpt|$cfg|$init_thermo|$_pkl_dir|$_pt_dir|$_keff|$_ds|$csv")
            done
        done
    done

    _cv_total=${#_CV_QUEUE[@]}
    _hdr "Stage C — dispatching $_cv_total CV tasks across ${_CV_N_POOL} GPUs"

    if (( _cv_total == 0 )); then
        echo "  no CV tasks — check that sampling completed"
    else
        declare -A _CV_PID_GPU
        declare -A _CV_PID_TAG
        _cv_done=0; _cv_fail=0; _cv_idx=0

        _do_cv_task() {
            local _gpu="$1" _sfx="$2" _ck="$3" _cf="$4" _init="$5" _pkl="$6" _pt="$7" _k="$8" _ds="$9"
            local _out_ds="$OUT_ROOT/${_ds}_${_sfx}"
            mkdir -p "$_out_ds"
            BASE_GPU=$_gpu \
            SLEEP_HOURS=0 \
            K=$_k EPOCHS=$EPOCHS EARLY_STOP_PATIENCE=$EARLY_STOP_PATIENCE \
            AUTO_EPOCHS=$AUTO_EPOCHS EPOCHS_LARGE=$EPOCHS_LARGE EPOCHS_SMALL=$EPOCHS_SMALL \
            LR=$LR BATCH=$BATCH N_GPUS=1 \
            INPUT_DIR=$INPUT_DIR \
            PKL_DIR=$_pkl PT_DIR=$_pt \
            OUT_ROOT=$OUT_ROOT OUT_SUFFIX=$_sfx \
            CKPT=$_ck CONFIG=$_cf \
            INIT_FROM_THERMO=$_init \
            HEAD_HIDDEN=$HEAD_HIDDEN N_MP_LAYERS=$N_MP_LAYERS MP_N_HEADS=$MP_N_HEADS \
            ONLY_DATASETS=$_ds \
            WANDB=$WANDB WANDB_PROJECT=$WANDB_PROJECT WANDB_GROUP=$_sfx \
                bash scripts/run_downstream_pipeline.sh \
                >> "$LOG_DIR/0506_cv_${_sfx}_${_ds}.log" 2>&1 &
        }

        # Seed pool
        for (( _gi=0; _gi < _CV_N_POOL && _cv_idx < _cv_total; _gi++, _cv_idx++ )); do
            _gpu="${_CV_GPU_IDS[$_gi]}"
            IFS='|' read -r _sfx _ck _cf _init _pkl _pt _k _ds _csv <<< "${_CV_QUEUE[$_cv_idx]}"
            _do_cv_task "$_gpu" "$_sfx" "$_ck" "$_cf" "$_init" "$_pkl" "$_pt" "$_k" "$_ds"
            _pid=$!
            _CV_PID_GPU[$_pid]=$_gpu
            _CV_PID_TAG[$_pid]="${_sfx}/${_ds}"
            echo "[$(date +%T)] [$_cv_idx/$_cv_total] launch CV ${_sfx}/${_ds} → GPU $_gpu (pid=$_pid)"
        done

        # Drain
        while (( ${#_CV_PID_GPU[@]} > 0 )); do
            _fpid=0
            wait -n -p _fpid 2>/dev/null || true
            _fpid=${_fpid:-0}
            [[ -z "${_CV_PID_GPU[$_fpid]:-}" ]] && continue
            _gpu="${_CV_PID_GPU[$_fpid]}"
            _tag="${_CV_PID_TAG[$_fpid]}"
            unset '_CV_PID_GPU[$_fpid]' '_CV_PID_TAG[$_fpid]'
            _cv_done=$((_cv_done+1))
            _wst=$?
            (( _wst != 0 )) && { _cv_fail=$((_cv_fail+1)); echo "[$(date +%T)] FAIL $_tag (gpu=$_gpu)"; } \
                             || echo "[$(date +%T)] done $_tag (gpu=$_gpu)"
            if (( _cv_idx < _cv_total )); then
                _entry="${_CV_QUEUE[$_cv_idx]}"
                _cv_idx=$((_cv_idx+1))
                IFS='|' read -r _sfx _ck _cf _init _pkl _pt _k _ds _csv <<< "$_entry"
                _do_cv_task "$_gpu" "$_sfx" "$_ck" "$_cf" "$_init" "$_pkl" "$_pt" "$_k" "$_ds"
                _pid=$!
                _CV_PID_GPU[$_pid]=$_gpu
                _CV_PID_TAG[$_pid]="${_sfx}/${_ds}"
                echo "[$(date +%T)] [$_cv_idx/$_cv_total] launch CV ${_sfx}/${_ds} → GPU $_gpu (pid=$_pid)"
            fi
        done

        echo
        echo "CV done: $_cv_done total, $_cv_fail failed"
    fi
fi

# -----------------------------------------------------------------------
# Final summary
# -----------------------------------------------------------------------
_hdr "FINAL SUMMARY"
python3 - <<PY
import glob, json, os
root = "$OUT_ROOT"
from pathlib import Path

print(f"\n{'dataset':<14s}  {'ckpt_mode':<26s}  "
      f"{'best_val MAE':>12s}  {'last_stab MAE':>13s}  {'R²':>7s}  {'ep':>5s}")
print("-" * 90)
for rep in sorted(glob.glob(os.path.join(root, "*/cv_report.json"))):
    suffix = Path(rep).parent.name
    parts = suffix.rsplit("_", 2)
    ds   = parts[0] if len(parts) == 3 else suffix
    mode = "_".join(parts[1:]) if len(parts) >= 2 else ""
    try:
        d = json.load(open(rep))
        mae_bv = d["mae_mean"]
        mae_ls = d.get("mae_last_stable_mean", float("nan"))
        r2     = d["r2_mean"]
        ep     = sum(f.get("best_epoch",0) for f in d.get("folds",[])) / max(len(d.get("folds",[])),1)
        print(f"{ds:<14s}  {mode:<26s}  "
              f"{mae_bv:>12.3f}  {mae_ls:>13.3f}  {r2:>7.3f}  {ep:>5.0f}")
    except Exception as e:
        print(f"{suffix}  (err: {e})")
PY

echo; echo "[$(date +'%F %T')] Done. Outputs under $OUT_ROOT/"
