#!/usr/bin/env bash
# Downstream CV benchmark — 0509 edition
#
# Two backbones × three sampling modes × N_datasets:
#   warm_last  thermo_flow_warm_last.ckpt  (256/10 or check with inspect_ckpt.py)
#   cold_last  thermo_flow_cold_last.ckpt  (384/12)
#
# Sampling modes:
#   K=5   standard  : 5 independent trajectories, 1 conformer each
#   K=8   standard  : 8 independent trajectories, 1 conformer each
#   K=9ms multi-snap: 3 trajectories × snapshot steps 7,8,9 = 9 conformers
#
# Head: cold (random) init for all, 4/4/256.
# CV: 5-fold, seed=2, 10% val from train pool, best-val + last-stable reported.
#
# Usage:
#   nohup bash scripts/run_0509_cv.sh > /tmp/0509_cv.log 2>&1 &
#   disown

set -uo pipefail
cd "$(dirname "$0")/.."

if (( BASH_VERSINFO[0] < 5 )) || { (( BASH_VERSINFO[0] == 5 )) && (( BASH_VERSINFO[1] < 1 )); }; then
    echo "ERROR: bash >= 5.1 required" >&2; exit 1
fi

# ============ CONFIG ============
N_GPUS=${N_GPUS:-8}
CUDA_DEVICES=${CUDA_DEVICES:-0,1,2,3,4,5,6,7}

INPUT_DIR=${INPUT_DIR:-downstream_ft/0506/cleaned_by_codex}

CKPT_DEFS=(
    # "label|ckpt_path|config_path|init_from_thermo"
    # Run inspect_ckpt.py to verify config (warm_last: warm.yaml if d=256, cold.yaml if d=384)
    "warm_last|data/ft_ckpts/thermo_flow_warm_last.ckpt|scripts/conf/loqi/loqi_thermo_flow_warm.yaml|0"
    "cold_last|data/ft_ckpts/thermo_flow_cold_last.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold.yaml|0"
)

# K=5 standard
K_5=5
N_STEPS_5=10

# K=8 standard
K_8=8
N_STEPS_8=10

# K=9 multi-snapshot: 3 traj × steps 7,8,9
N_TRAJ_MS=3
N_STEPS_MS=10
SNAPSHOT_STEPS="7 8 9"

HEAD_HIDDEN=${HEAD_HIDDEN:-256}
N_MP_LAYERS=${N_MP_LAYERS:-4}
MP_N_HEADS=${MP_N_HEADS:-4}

EPOCHS=${EPOCHS:-150}
AUTO_EPOCHS=${AUTO_EPOCHS:-1}
EPOCHS_LARGE=${EPOCHS_LARGE:-200}
EPOCHS_SMALL=${EPOCHS_SMALL:-150}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-100}
LR=${LR:-3e-4}
BATCH=${BATCH:-64}

OUT_ROOT=${OUT_ROOT:-outputs/downstream_cv_0509}
WANDB=${WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-downstream_0509}
LOG_DIR=${LOG_DIR:-/tmp}

SKIP_SMI=${SKIP_SMI:-0}
SKIP_SAMPLE=${SKIP_SAMPLE:-0}
SKIP_CV=${SKIP_CV:-0}
# ================================

mkdir -p "$OUT_ROOT" "$LOG_DIR"
[[ -d "$INPUT_DIR" ]] || { echo "ERROR: $INPUT_DIR not found" >&2; exit 1; }
for def in "${CKPT_DEFS[@]}"; do
    IFS='|' read -r label ckpt cfg _init <<< "$def"
    [[ -f "$ckpt" ]] || { echo "ERROR: ckpt not found: $ckpt ($label)" >&2; exit 1; }
    [[ -f "$cfg"  ]] || { echo "ERROR: config not found: $cfg ($label)" >&2; exit 1; }
done

_hdr() { echo; echo "============================================================"; echo "[$(date +'%F %T')]  $1"; echo "============================================================"; }

# Discover datasets
DATASETS_CSV=()
for f in "$INPUT_DIR"/*.csv; do
    [[ "$(basename "$f")" == *report* ]] && continue
    DATASETS_CSV+=("$f")
done
echo "Datasets (${#DATASETS_CSV[@]}):"
for f in "${DATASETS_CSV[@]}"; do echo "  $(basename "$f")"; done

# Parse GPU list
IFS=',' read -ra _GPU_IDS <<< "$CUDA_DEVICES"
_N_POOL=${#_GPU_IDS[@]}

# -----------------------------------------------------------------------
# Stage A — extract .smi
# -----------------------------------------------------------------------
SMI_DIR="$LOG_DIR/0509_smi"
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
            >> "$LOG_DIR/0509_extract.log" 2>&1
        echo "  [ok] $smi"
    done
fi

# -----------------------------------------------------------------------
# Stage B — flat work-queue sampling (all ckpts × modes × datasets)
# -----------------------------------------------------------------------
if [[ "$SKIP_SAMPLE" == "1" ]]; then
    _hdr "Stage B SKIPPED"
else
    _QUEUE=()
    for def in "${CKPT_DEFS[@]}"; do
        IFS='|' read -r label ckpt cfg _init <<< "$def"
        for csv in "${DATASETS_CSV[@]}"; do
            name=$(basename "$csv" .csv)
            smi="$SMI_DIR/$name.smi"
            [[ -f "$smi" ]] || continue
            # K=5: no separate sampling — reuse K=8 pickles with --max-k-per-input 5
            # K=8
            pkl8="data/0509_pkl_${label}_k8"; mkdir -p "$pkl8"
            [[ -f "$pkl8/$name.pkl" ]] || _QUEUE+=("k8|$label|$ckpt|$cfg|$smi|$pkl8/$name.pkl|$name")
            # K=9ms
            pkl9="data/0509_pkl_${label}_k9ms"; mkdir -p "$pkl9"
            [[ -f "$pkl9/$name.pkl" ]] || _QUEUE+=("k9ms|$label|$ckpt|$cfg|$smi|$pkl9/$name.pkl|$name")
        done
    done

    total_tasks=${#_QUEUE[@]}
    _hdr "Stage B — $total_tasks sampling tasks across ${_N_POOL} GPUs"

    if (( total_tasks == 0 )); then
        echo "  all pickles exist — skipping"
    else
        declare -A _PID_GPU
        declare -A _PID_TAG
        _done=0; _fail=0; _idx=0

        _launch_sample() {
            local _gpu="$1"
            IFS='|' read -r _mode _lbl _ckpt _cfg _smi _pkl _name <<< "$2"
            if [[ "$_mode" == "k5" ]]; then
                CUDA_VISIBLE_DEVICES=$_gpu python scripts/sample_conformers.py \
                    --ckpt "$_ckpt" --config "$_cfg" \
                    --input "$_smi" --output "$_pkl" \
                    --n_confs $K_5 --n_steps $N_STEPS_5 \
                    --batch_size $BATCH --postprocess none \
                    >> "$LOG_DIR/0509_${_lbl}_${_name}_k5.log" 2>&1 &
            elif [[ "$_mode" == "k8" ]]; then
                CUDA_VISIBLE_DEVICES=$_gpu python scripts/sample_conformers.py \
                    --ckpt "$_ckpt" --config "$_cfg" \
                    --input "$_smi" --output "$_pkl" \
                    --n_confs $K_8 --n_steps $N_STEPS_8 \
                    --batch_size $BATCH --postprocess none \
                    >> "$LOG_DIR/0509_${_lbl}_${_name}_k8.log" 2>&1 &
            else
                CUDA_VISIBLE_DEVICES=$_gpu python scripts/sample_conformers_multistep.py \
                    --ckpt "$_ckpt" --config "$_cfg" \
                    --input "$_smi" --output "$_pkl" \
                    --n_traj $N_TRAJ_MS --n_steps $N_STEPS_MS \
                    --snapshot_steps $SNAPSHOT_STEPS \
                    --batch_size $BATCH \
                    >> "$LOG_DIR/0509_${_lbl}_${_name}_k9ms.log" 2>&1 &
            fi
            echo $!
        }

        for (( _gi=0; _gi < _N_POOL && _idx < total_tasks; _gi++, _idx++ )); do
            _gpu="${_GPU_IDS[$_gi]}"
            _pid=$(_launch_sample "$_gpu" "${_QUEUE[$_idx]}")
            _PID_GPU[$_pid]=$_gpu
            IFS='|' read -r _mode _lbl _ck _cf _si _pk _nm <<< "${_QUEUE[$_idx]}"
            _PID_TAG[$_pid]="${_mode}/${_lbl}/${_nm}"
            echo "[$(date +%T)] [$_idx/$total_tasks] launch ${_mode} ${_lbl}/${_nm} → GPU $_gpu"
        done

        while (( ${#_PID_GPU[@]} > 0 )); do
            _fpid=0; wait -n -p _fpid 2>/dev/null || true
            [[ -z "${_PID_GPU[$_fpid]:-}" ]] && continue
            _gpu="${_PID_GPU[$_fpid]}"; _tag="${_PID_TAG[$_fpid]}"
            unset '_PID_GPU[$_fpid]' '_PID_TAG[$_fpid]'
            _done=$((_done+1)); _wst=$?
            (( _wst != 0 )) && { _fail=$((_fail+1)); echo "[$(date +%T)] FAIL $_tag (gpu=$_gpu)"; } \
                             || echo "[$(date +%T)] done $_tag (gpu=$_gpu)"
            if (( _idx < total_tasks )); then
                _gpu_n=$_gpu
                _pid=$(_launch_sample "$_gpu_n" "${_QUEUE[$_idx]}")
                _PID_GPU[$_pid]=$_gpu_n
                IFS='|' read -r _mode _lbl _ck _cf _si _pk _nm <<< "${_QUEUE[$_idx]}"
                _PID_TAG[$_pid]="${_mode}/${_lbl}/${_nm}"
                echo "[$(date +%T)] [$_idx/$total_tasks] launch ${_mode} ${_lbl}/${_nm} → GPU $_gpu_n"
                _idx=$((_idx+1))
            fi
        done
        echo "Sampling done: $_done total, $_fail failed"
    fi
fi

# -----------------------------------------------------------------------
# Stage C — flat work-queue CV
# -----------------------------------------------------------------------
if [[ "$SKIP_CV" == "1" ]]; then
    _hdr "CV SKIPPED"
else
    IFS=',' read -ra _CV_GPU_IDS <<< "$CUDA_DEVICES"
    _CV_N_POOL=${#_CV_GPU_IDS[@]}

    _CV_QUEUE=()
    for def in "${CKPT_DEFS[@]}"; do
        IFS='|' read -r label ckpt cfg init_thermo <<< "$def"
        # K=5 reuses K=8 pkl/pt (no separate sampling) with max-k-per-input=5.
        # Format: "tag:keff:pkl_dir:pt_dir:max_k_cap" (max_k_cap=0 means no cap)
        for _entry in \
            "K5:5:data/0509_pkl_${label}_k8:data/0509_pt_${label}_k8:5" \
            "K8:${K_8}:data/0509_pkl_${label}_k8:data/0509_pt_${label}_k8:0" \
            "K9ms:9:data/0509_pkl_${label}_k9ms:data/0509_pt_${label}_k9ms:0"; do
            IFS=':' read -r _mode_tag _keff _pkl_dir _pt_dir _maxk <<< "$_entry"
            _suffix="${label}_${_mode_tag}"
            mkdir -p "$_pt_dir" "$OUT_ROOT"
            [[ -d "$_pkl_dir" ]] || continue
            for csv in "${DATASETS_CSV[@]}"; do
                _ds=$(basename "$csv" .csv)
                [[ -f "$_pkl_dir/$_ds.pkl" ]] || continue
                [[ -f "$OUT_ROOT/${_ds}_${_suffix}/cv_report.json" ]] && \
                    { echo "  [skip CV] ${_suffix}/${_ds}"; continue; }
                _CV_QUEUE+=("$_suffix|$ckpt|$cfg|$init_thermo|$_pkl_dir|$_pt_dir|$_keff|$_ds|$csv|$_maxk")
            done
        done
    done

    _cv_total=${#_CV_QUEUE[@]}
    _hdr "Stage C — $_cv_total CV tasks across ${_CV_N_POOL} GPUs"

    if (( _cv_total == 0 )); then
        echo "  all CV reports exist"
    else
        declare -A _CV_PID_GPU
        declare -A _CV_PID_TAG
        _cv_done=0; _cv_fail=0; _cv_idx=0

        _launch_cv() {
            local _gpu="$1"
            IFS='|' read -r _sfx _ck _cf _init _pkl _pt _k _ds _csv _maxk <<< "$2"
            CUDA_VISIBLE_DEVICES=$_gpu \
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
            MAX_K_PER_INPUT=${_maxk:-0} \
            ONLY_DATASETS=$_ds \
            WANDB=$WANDB WANDB_PROJECT=$WANDB_PROJECT WANDB_GROUP=$_sfx \
                bash scripts/run_downstream_pipeline.sh \
                >> "$LOG_DIR/0509_cv_${_sfx}_${_ds}.log" 2>&1 &
            echo $!
        }

        for (( _gi=0; _gi < _CV_N_POOL && _cv_idx < _cv_total; _gi++, _cv_idx++ )); do
            _gpu="${_CV_GPU_IDS[$_gi]}"
            _pid=$(_launch_cv "$_gpu" "${_CV_QUEUE[$_cv_idx]}")
            _CV_PID_GPU[$_pid]=$_gpu
            IFS='|' read -r _sfx _ck _cf _init _pkl _pt _k _ds _csv _maxk <<< "${_CV_QUEUE[$_cv_idx]}"
            _CV_PID_TAG[$_pid]="${_sfx}/${_ds}"
            echo "[$(date +%T)] [$_cv_idx/$_cv_total] CV ${_sfx}/${_ds} → GPU $_gpu"
        done

        while (( ${#_CV_PID_GPU[@]} > 0 )); do
            _fpid=0; wait -n -p _fpid 2>/dev/null || true
            [[ -z "${_CV_PID_GPU[$_fpid]:-}" ]] && continue
            _gpu="${_CV_PID_GPU[$_fpid]}"; _tag="${_CV_PID_TAG[$_fpid]}"
            unset '_CV_PID_GPU[$_fpid]' '_CV_PID_TAG[$_fpid]'
            _cv_done=$((_cv_done+1)); _wst=$?
            (( _wst != 0 )) && { _cv_fail=$((_cv_fail+1)); echo "[$(date +%T)] FAIL $_tag (gpu=$_gpu)"; } \
                             || echo "[$(date +%T)] done $_tag (gpu=$_gpu)"
            if (( _cv_idx < _cv_total )); then
                _entry="${_CV_QUEUE[$_cv_idx]}"
                _cv_idx=$((_cv_idx+1))
                _pid=$(_launch_cv "$_gpu" "$_entry")
                _CV_PID_GPU[$_pid]=$_gpu
                IFS='|' read -r _sfx _ck _cf _init _pkl _pt _k _ds _csv _maxk <<< "$_entry"
                _CV_PID_TAG[$_pid]="${_sfx}/${_ds}"
                echo "[$(date +%T)] [$_cv_idx/$_cv_total] CV ${_sfx}/${_ds} → GPU $_gpu"
            fi
        done
        echo "CV done: $_cv_done total, $_cv_fail failed"
    fi
fi

# -----------------------------------------------------------------------
# Final summary (both best-val and last-stable)
# -----------------------------------------------------------------------
_hdr "FINAL SUMMARY"
python3 - <<PY
import glob, json, os
from pathlib import Path

root = "$OUT_ROOT"
print(f"\n{'dataset':<14s}  {'config':<22s}  "
      f"{'best_val MAE':>12s}  {'last_stab MAE':>13s}  {'R²':>7s}  {'ep':>5s}")
print("-" * 88)
for rep in sorted(glob.glob(os.path.join(root, "*/cv_report.json"))):
    suffix = Path(rep).parent.name
    parts = suffix.split("_")
    # suffix like gas_Hf_warm_last_K8 or gas_Hf_cold_last_K9ms
    # find split: ds ends before ckpt label starts
    # simple heuristic: ckpt labels start with warm/cold
    ds = suffix; mode = ""
    for i, p in enumerate(parts):
        if p in ("warm", "cold"):
            ds = "_".join(parts[:i])
            mode = "_".join(parts[i:])
            break
    try:
        d = json.load(open(rep))
        mae_bv = d["mae_mean"]
        mae_ls = d.get("mae_last_stable_mean", float("nan"))
        r2     = d["r2_mean"]
        ep     = sum(f.get("best_epoch",0) for f in d.get("folds",[])) / max(len(d.get("folds",[])),1)
        print(f"{ds:<14s}  {mode:<22s}  "
              f"{mae_bv:>12.3f}  {mae_ls:>13.3f}  {r2:>7.3f}  {ep:>5.0f}")
    except Exception as e:
        print(f"{suffix}  (err: {e})")
PY

echo
echo "[$(date +'%F %T')] Done. Reports: $OUT_ROOT/<ds>_<config>/cv_report.json"
