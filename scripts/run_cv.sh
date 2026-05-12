#!/usr/bin/env bash
# Unified downstream CV benchmark.
#
# Configure CKPT_DEFS and SAMPLING_MODES, then run.
#
# SAMPLING_MODES format — one entry per mode:
#   "standard|tag|keff|n_steps"
#       Standard sampling: keff independent trajectories, 1 conformer each.
#   "multistep|tag|keff|n_traj:n_steps:snap1 snap2 snap3"   ← ':' inside rest
#       Multi-snapshot: n_traj trajectories × len(snaps) snapshots = keff conformers.
#   "reuse|tag|keff|source_tag"
#       No sampling. Borrows source_tag's pkl with MAX_K_PER_INPUT=keff cap.
#
# Examples:
#   # Reproduce 0509 (2 ckpts × K5+K8+K9ms):
#   CKPT_DEFS=(...) SAMPLING_MODES=("standard|K8|8|10" "reuse|K5|5|K8" "multistep|K9ms|9|3|10|7 8 9") \
#   OUT_ROOT=outputs/cv_0509 WANDB_PROJECT=cv_0509 bash scripts/run_cv.sh
#
#   # Reproduce 0506 (9 ckpts × K8+K12ms):
#   CKPT_DEFS=(...) SAMPLING_MODES=("standard|K8|8|10" "multistep|K12ms|12|4|10|7 8 9") \
#   OUT_ROOT=outputs/cv_0506 WANDB_PROJECT=cv_0506 bash scripts/run_cv.sh
#
# Usage:
#   nohup bash scripts/run_cv.sh > /tmp/cv.log 2>&1 & disown

set -uo pipefail
cd "$(dirname "$0")/.."

if (( BASH_VERSINFO[0] < 5 )) || { (( BASH_VERSINFO[0] == 5 )) && (( BASH_VERSINFO[1] < 1 )); }; then
    echo "ERROR: bash >= 5.1 required" >&2; exit 1
fi

# ============ CONFIG ============
N_GPUS=${N_GPUS:-8}
CUDA_DEVICES=${CUDA_DEVICES:-0,1,2,3,4,5,6,7}
TASKS_PER_GPU=${TASKS_PER_GPU:-1}   # concurrent CV tasks per GPU (head-only phase)

INPUT_DIR=${INPUT_DIR:-downstream_ft/0506/cleaned_by_codex}

# CKPT_DEFS: "label|ckpt_path|config_path|init_from_thermo"
# init_from_thermo=1 → warm-start downstream head from ckpt's thermo_heads
#
# Bash arrays do NOT survive `exec bash` / subprocess boundaries, so the
# canonical way to override these is to `source scripts/run_cv.sh` from
# a wrapper after defining CKPT_DEFS / SAMPLING_MODES. We guard the
# defaults below so they only apply when the array is unset / empty.
if ! declare -p CKPT_DEFS &>/dev/null || (( ${#CKPT_DEFS[@]} == 0 )); then
    CKPT_DEFS=(
        "warm_last|data/ft_ckpts/thermo_flow_warm_last.ckpt|scripts/conf/loqi/loqi_thermo_flow_warm.yaml|0"
        "cold_last|data/ft_ckpts/thermo_flow_cold_last.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold.yaml|0"
    )
fi

# SAMPLING_MODES: see format at top of file.
# K8 + K12ms (4 traj × 3 snapshots = 12 conformers).
# NOTE: multistep rest uses ':' as internal separator (not '|') to avoid
# clashing with the '|' field separator used in the task queue.
if ! declare -p SAMPLING_MODES &>/dev/null || (( ${#SAMPLING_MODES[@]} == 0 )); then
    SAMPLING_MODES=(
        "standard|K8|8|10"
        "multistep|K12ms|12|4:10:7 8 9"
    )
fi

# Run tag — used to namespace pkl/pt dirs: data/<RUN_TAG>_pkl_<label>_<mode>
# Set to match the dataset / experiment name (e.g. "0509", "0506", "0511").
RUN_TAG=${RUN_TAG:-cv}

HEAD_HIDDEN=${HEAD_HIDDEN:-256}
N_MP_LAYERS=${N_MP_LAYERS:-4}
MP_N_HEADS=${MP_N_HEADS:-4}

EPOCHS=${EPOCHS:-150}
AUTO_EPOCHS=${AUTO_EPOCHS:-1}
EPOCHS_LARGE=${EPOCHS_LARGE:-200}
EPOCHS_SMALL=${EPOCHS_SMALL:-150}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-100}
LR=${LR:-3e-4}
BATCH=${BATCH:-64}                              # CV training batch
SAMPLE_BATCH=${SAMPLE_BATCH:-$BATCH}            # conformer-sampling batch (Stage B)
EXTRACT_BATCH=${EXTRACT_BATCH:-$BATCH}          # H-cache extraction batch (Stage B.5 / C)

# Pre-computed CV split directory (for 0511 audited data). When set,
# downstream_cv.py loads fold assignments directly from
#   $SPLIT_DIR_ROOT/<dataset>/random_cv5/cv{i}_train/valid/test.csv
# instead of re-splitting internally. Leave empty to use internal KFold.
SPLIT_DIR_ROOT=${SPLIT_DIR_ROOT:-}

OUT_ROOT=${OUT_ROOT:-outputs/downstream_cv}
WANDB=${WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-downstream_cv}
LOG_DIR=${LOG_DIR:-/tmp}

SKIP_SMI=${SKIP_SMI:-0}
SKIP_SAMPLE=${SKIP_SAMPLE:-0}
SKIP_EXTRACT=${SKIP_EXTRACT:-0}
SKIP_CV=${SKIP_CV:-0}
FORCE_CV=${FORCE_CV:-0}        # 1 = ignore existing cv_report.json (re-run CV)
# Defensive: clear any inherited EXTRACT_ONLY env var so Stage C never
# accidentally gets passed --extract-only via env-var leakage.
unset EXTRACT_ONLY
# ================================

mkdir -p "$OUT_ROOT" "$LOG_DIR"
[[ -d "$INPUT_DIR" ]] || { echo "ERROR: INPUT_DIR not found: $INPUT_DIR" >&2; exit 1; }
for def in "${CKPT_DEFS[@]}"; do
    IFS='|' read -r label ckpt cfg _init <<< "$def"
    [[ -f "$ckpt" ]] || { echo "ERROR: ckpt not found: $ckpt ($label)" >&2; exit 1; }
    [[ -f "$cfg"  ]] || { echo "ERROR: config not found: $cfg ($label)" >&2; exit 1; }
done

_hdr() { echo; echo "============================================================"; echo "[$(date +'%F %T')]  $1"; echo "============================================================"; }

# Discover datasets — optional comma-separated whitelist via DATASETS_FILTER
DATASETS_FILTER=${DATASETS_FILTER:-}
DATASETS_CSV=()
for f in "$INPUT_DIR"/*.csv; do
    _bn=$(basename "$f" .csv)
    [[ "$_bn" == *report* ]] && continue
    if [[ -n "$DATASETS_FILTER" ]]; then
        case ",$DATASETS_FILTER," in
            *",$_bn,"*) ;;
            *) continue ;;
        esac
    fi
    DATASETS_CSV+=("$f")
done
echo "Datasets (${#DATASETS_CSV[@]}):"
for f in "${DATASETS_CSV[@]}"; do echo "  $(basename "$f")"; done

# Parse GPU list → pool with TASKS_PER_GPU slots each
IFS=',' read -ra _BASE_GPU_IDS <<< "$CUDA_DEVICES"
_POOL_IDS=()
for _g in "${_BASE_GPU_IDS[@]}"; do
    for (( _t=0; _t<TASKS_PER_GPU; _t++ )); do _POOL_IDS+=("$_g"); done
done
_N_POOL=${#_BASE_GPU_IDS[@]}   # for sampling (1 per GPU); CV uses TASKS_PER_GPU

# ---------------------------------------------------------------------------
# Helper: resolve pkl/pt dirs for a given label+mode_tag
# ---------------------------------------------------------------------------
_pkl_dir_for() { echo "data/${RUN_TAG}_pkl_${1}_$(echo "$2" | tr '[:upper:]' '[:lower:]')"; }
_pt_dir_for()  { echo "data/${RUN_TAG}_pt_${1}_$(echo  "$2" | tr '[:upper:]' '[:lower:]')"; }

# ---------------------------------------------------------------------------
# Stage A — extract .smi
# ---------------------------------------------------------------------------
SMI_DIR="$LOG_DIR/${RUN_TAG}_smi"
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
            >> "$LOG_DIR/${RUN_TAG}_extract.log" 2>&1
        echo "  [ok] $smi"
    done
fi

# ---------------------------------------------------------------------------
# Stage B — sample conformers (standard + multistep modes; skip reuse)
# ---------------------------------------------------------------------------
if [[ "$SKIP_SAMPLE" == "1" ]]; then
    _hdr "Stage B SKIPPED"
else
    _B_QUEUE=()
    for def in "${CKPT_DEFS[@]}"; do
        IFS='|' read -r label ckpt cfg _init <<< "$def"
        for mode_entry in "${SAMPLING_MODES[@]}"; do
            IFS='|' read -r mtype tag keff rest <<< "$mode_entry"
            [[ "$mtype" == "reuse" ]] && continue   # no sampling needed
            pkl_dir=$(_pkl_dir_for "$label" "$tag")
            mkdir -p "$pkl_dir"
            for csv in "${DATASETS_CSV[@]}"; do
                name=$(basename "$csv" .csv)
                smi="$SMI_DIR/$name.smi"
                [[ -f "$smi" ]] || continue
                [[ -f "$pkl_dir/$name.pkl" ]] && continue
                _B_QUEUE+=("$mtype|$tag|$keff|$rest|$label|$ckpt|$cfg|$smi|$pkl_dir/$name.pkl|$name")
            done
        done
    done

    _hdr "Stage B — ${#_B_QUEUE[@]} sampling tasks across ${_N_POOL} GPUs"

    if (( ${#_B_QUEUE[@]} == 0 )); then
        echo "  all pickles exist — skipping"
    else
        _do_sample() {
            local _gpu="$1" _mtype="$2" _tag="$3" _keff="$4" _rest="$5"
            local _lbl="$6" _ck="$7" _cf="$8" _smi="$9" _pkl="${10}" _name="${11}"
            if [[ "$_mtype" == "standard" ]]; then
                local _steps="$_rest"
                CUDA_VISIBLE_DEVICES=$_gpu python scripts/sample_conformers.py \
                    --ckpt "$_ck" --config "$_cf" \
                    --input "$_smi" --output "$_pkl" \
                    --n_confs "$_keff" --n_steps "$_steps" \
                    --batch_size $SAMPLE_BATCH --postprocess none \
                    >> "$LOG_DIR/${RUN_TAG}_${_lbl}_${_name}_${_tag}.log" 2>&1 &
            else  # multistep — rest uses ':' internally: "n_traj:n_steps:snap1 snap2 ..."
                IFS=':' read -r _n_traj _n_steps _snaps <<< "$_rest"
                CUDA_VISIBLE_DEVICES=$_gpu python scripts/sample_conformers_multistep.py \
                    --ckpt "$_ck" --config "$_cf" \
                    --input "$_smi" --output "$_pkl" \
                    --n_traj "$_n_traj" --n_steps "$_n_steps" \
                    --snapshot_steps $_snaps \
                    --batch_size $SAMPLE_BATCH \
                    >> "$LOG_DIR/${RUN_TAG}_${_lbl}_${_name}_${_tag}.log" 2>&1 &
            fi
        }

        declare -A _B_PID_GPU; declare -A _B_PID_TAG
        _b_done=0; _b_fail=0; _b_idx=0

        for (( _gi=0; _gi < _N_POOL && _b_idx < ${#_B_QUEUE[@]}; _gi++, _b_idx++ )); do
            _gpu="${_BASE_GPU_IDS[$_gi]}"
            IFS='|' read -r _mt _tg _ke _rs _lb _ck _cf _si _pk _nm <<< "${_B_QUEUE[$_b_idx]}"
            _do_sample "$_gpu" "$_mt" "$_tg" "$_ke" "$_rs" "$_lb" "$_ck" "$_cf" "$_si" "$_pk" "$_nm"
            _pid=$!; _B_PID_GPU[$_pid]=$_gpu; _B_PID_TAG[$_pid]="${_lb}/${_nm}/${_tg}"
            echo "[$(date +%T)] [$_b_idx/${#_B_QUEUE[@]}] sample ${_lb}/${_nm}/${_tg} → GPU $_gpu"
        done
        while (( ${#_B_PID_GPU[@]} > 0 )); do
            _fpid=0
            wait -n -p _fpid 2>/dev/null
            _wst=$?
            _fpid=${_fpid:-0}
            (( _wst == 127 )) && break  # no more children
            [[ -z "${_B_PID_GPU[$_fpid]:-}" ]] && continue
            _gpu="${_B_PID_GPU[$_fpid]}"; _tag="${_B_PID_TAG[$_fpid]}"
            unset '_B_PID_GPU[$_fpid]' '_B_PID_TAG[$_fpid]'
            _b_done=$((_b_done+1))
            (( _wst != 0 )) && { _b_fail=$((_b_fail+1)); echo "[$(date +%T)] FAIL $_tag (gpu=$_gpu) exit=$_wst — check $LOG_DIR/${RUN_TAG}_${_tag//\//_}.log"; } \
                             || echo "[$(date +%T)] done $_tag (gpu=$_gpu)"
            if (( _b_idx < ${#_B_QUEUE[@]} )); then
                IFS='|' read -r _mt _tg _ke _rs _lb _ck _cf _si _pk _nm <<< "${_B_QUEUE[$_b_idx]}"
                _b_idx=$((_b_idx+1))
                _do_sample "$_gpu" "$_mt" "$_tg" "$_ke" "$_rs" "$_lb" "$_ck" "$_cf" "$_si" "$_pk" "$_nm"
                _pid=$!; _B_PID_GPU[$_pid]=$_gpu; _B_PID_TAG[$_pid]="${_lb}/${_nm}/${_tg}"
                echo "[$(date +%T)] [$_b_idx/${#_B_QUEUE[@]}] sample ${_lb}/${_nm}/${_tg} → GPU $_gpu"
            fi
        done
        echo "Sampling done: $_b_done total, $_b_fail failed"
    fi
fi

# ---------------------------------------------------------------------------
# Stage B.5 — pre-extract backbone H embeddings
# Dedup by (pt_dir × dataset) so reuse modes don't double-extract.
# ---------------------------------------------------------------------------
if [[ "$SKIP_EXTRACT" == "1" ]]; then
    _hdr "Stage B.5 SKIPPED"
else
    declare -A _EX_SEEN
    _EX_QUEUE=()
    for def in "${CKPT_DEFS[@]}"; do
        IFS='|' read -r label ckpt cfg init_thermo <<< "$def"
        for mode_entry in "${SAMPLING_MODES[@]}"; do
            IFS='|' read -r mtype tag keff rest <<< "$mode_entry"
            if [[ "$mtype" == "reuse" ]]; then
                source_tag="$rest"
                pkl_dir=$(_pkl_dir_for "$label" "$source_tag")
            else
                pkl_dir=$(_pkl_dir_for "$label" "$tag")
            fi
            pt_dir=$(_pt_dir_for "$label" "$tag")
            mkdir -p "$pt_dir"
            [[ -d "$pkl_dir" ]] || continue
            for csv in "${DATASETS_CSV[@]}"; do
                _ds=$(basename "$csv" .csv)
                [[ -f "$pkl_dir/$_ds.pkl" ]] || continue
                _key="${pt_dir}|${_ds}"
                [[ -n "${_EX_SEEN[$_key]:-}" ]] && continue
                _EX_SEEN[$_key]=1
                [[ -f "$pt_dir/${_ds}_H.pt" ]] && { echo "  [skip H] ${label}_${tag}/${_ds}"; continue; }
                _EX_QUEUE+=("${label}_${tag}|$ckpt|$cfg|$init_thermo|$pkl_dir|$pt_dir|$keff|$_ds|$csv")
            done
        done
    done

    _hdr "Stage B.5 — ${#_EX_QUEUE[@]} H-extraction tasks across ${_N_POOL} GPUs"
    if (( ${#_EX_QUEUE[@]} == 0 )); then
        echo "  all H caches exist"
    else
        _do_extract() {
            local _gpu="$1" _sfx="$2" _ck="$3" _cf="$4" _init="$5" _pkl="$6" _pt="$7" _k="$8" _ds="$9"
            BASE_GPU=$_gpu SLEEP_HOURS=0 K=$_k N_GPUS=1 \
            INPUT_DIR=$INPUT_DIR PKL_DIR=$_pkl PT_DIR=$_pt \
            OUT_ROOT=$OUT_ROOT OUT_SUFFIX=$_sfx \
            CKPT=$_ck CONFIG=$_cf INIT_FROM_THERMO=$_init \
            HEAD_HIDDEN=$HEAD_HIDDEN N_MP_LAYERS=$N_MP_LAYERS MP_N_HEADS=$MP_N_HEADS \
            BATCH=$BATCH EXTRACT_BATCH=$EXTRACT_BATCH \
            ONLY_DATASETS=$_ds H_CACHE_DIR=$_pt WANDB=0 \
                bash scripts/run_downstream_pipeline.sh --extract-only \
                >> "$LOG_DIR/${RUN_TAG}_extract_${_sfx}_${_ds}.log" 2>&1 &
        }
        declare -A _EX_PID_GPU; declare -A _EX_PID_TAG
        _ex_done=0; _ex_fail=0; _ex_idx=0
        for (( _gi=0; _gi < _N_POOL && _ex_idx < ${#_EX_QUEUE[@]}; _gi++, _ex_idx++ )); do
            _gpu="${_BASE_GPU_IDS[$_gi]}"
            IFS='|' read -r _sfx _ck _cf _init _pkl _pt _k _ds _csv <<< "${_EX_QUEUE[$_ex_idx]}"
            _do_extract "$_gpu" "$_sfx" "$_ck" "$_cf" "$_init" "$_pkl" "$_pt" "$_k" "$_ds"
            _pid=$!; _EX_PID_GPU[$_pid]=$_gpu; _EX_PID_TAG[$_pid]="${_sfx}/${_ds}"
            echo "[$(date +%T)] [$_ex_idx/${#_EX_QUEUE[@]}] extract H ${_sfx}/${_ds} → GPU $_gpu"
        done
        while (( ${#_EX_PID_GPU[@]} > 0 )); do
            _fpid=0
            wait -n -p _fpid 2>/dev/null
            _wst=$?
            _fpid=${_fpid:-0}
            (( _wst == 127 )) && break
            [[ -z "${_EX_PID_GPU[$_fpid]:-}" ]] && continue
            _gpu="${_EX_PID_GPU[$_fpid]}"; _tag="${_EX_PID_TAG[$_fpid]}"
            unset '_EX_PID_GPU[$_fpid]' '_EX_PID_TAG[$_fpid]'
            _ex_done=$((_ex_done+1))
            (( _wst != 0 )) && { _ex_fail=$((_ex_fail+1)); echo "[$(date +%T)] FAIL extract $_tag (gpu=$_gpu) exit=$_wst — check $LOG_DIR/${RUN_TAG}_extract_${_tag//\//_}.log"; } \
                             || echo "[$(date +%T)] done extract $_tag (gpu=$_gpu)"
            if (( _ex_idx < ${#_EX_QUEUE[@]} )); then
                IFS='|' read -r _sfx _ck _cf _init _pkl _pt _k _ds _csv <<< "${_EX_QUEUE[$_ex_idx]}"
                _ex_idx=$((_ex_idx+1))
                _do_extract "$_gpu" "$_sfx" "$_ck" "$_cf" "$_init" "$_pkl" "$_pt" "$_k" "$_ds"
                _pid=$!; _EX_PID_GPU[$_pid]=$_gpu; _EX_PID_TAG[$_pid]="${_sfx}/${_ds}"
                echo "[$(date +%T)] [$_ex_idx/${#_EX_QUEUE[@]}] extract H ${_sfx}/${_ds} → GPU $_gpu"
            fi
        done
        echo "H extraction done: $_ex_done total, $_ex_fail failed"
    fi
fi

# ---------------------------------------------------------------------------
# Stage C — flat work-queue CV across all ckpts × modes × datasets
# ---------------------------------------------------------------------------
if [[ "$SKIP_CV" == "1" ]]; then
    _hdr "Stage C SKIPPED"
else
    IFS=',' read -ra _CV_BASE_IDS <<< "$CUDA_DEVICES"
    _CV_GPU_IDS=()
    for _g in "${_CV_BASE_IDS[@]}"; do
        for (( _t=0; _t<TASKS_PER_GPU; _t++ )); do _CV_GPU_IDS+=("$_g"); done
    done
    _CV_N_POOL=${#_CV_GPU_IDS[@]}

    _CV_QUEUE=()
    for def in "${CKPT_DEFS[@]}"; do
        IFS='|' read -r label ckpt cfg init_thermo <<< "$def"
        for mode_entry in "${SAMPLING_MODES[@]}"; do
            IFS='|' read -r mtype tag keff rest <<< "$mode_entry"
            if [[ "$mtype" == "reuse" ]]; then
                source_tag="$rest"
                pkl_dir=$(_pkl_dir_for "$label" "$source_tag")
                maxk="$keff"
            else
                pkl_dir=$(_pkl_dir_for "$label" "$tag")
                maxk="0"
            fi
            pt_dir=$(_pt_dir_for "$label" "$tag")
            _suffix="${label}_${tag}"
            mkdir -p "$pt_dir" "$OUT_ROOT"
            [[ -d "$pkl_dir" ]] || continue
            for csv in "${DATASETS_CSV[@]}"; do
                _ds=$(basename "$csv" .csv)
                [[ -f "$pkl_dir/$_ds.pkl" ]] || { echo "  [skip CV] ${_suffix}/${_ds} — pkl missing ($pkl_dir/$_ds.pkl)"; continue; }
                if [[ -f "$OUT_ROOT/${_ds}_${_suffix}/cv_report.json" && "$FORCE_CV" != "1" ]]; then
                    echo "  [skip CV] ${_suffix}/${_ds} — cv_report.json exists (set FORCE_CV=1 to re-run)"
                    continue
                fi
                _CV_QUEUE+=("$_suffix|$ckpt|$cfg|$init_thermo|$pkl_dir|$pt_dir|$keff|$_ds|$csv|$maxk")
            done
        done
    done

    _cv_total=${#_CV_QUEUE[@]}
    _hdr "Stage C — $_cv_total CV tasks across ${_CV_N_POOL} GPUs"

    if (( _cv_total == 0 )); then
        echo "  all CV reports exist"
    else
        _do_cv() {
            local _gpu="$1" _sfx="$2" _ck="$3" _cf="$4" _init="$5"
            local _pkl="$6" _pt="$7" _k="$8" _ds="$9" _maxk="${10}"
            BASE_GPU=$_gpu SLEEP_HOURS=0 \
            K=$_k EPOCHS=$EPOCHS EARLY_STOP_PATIENCE=$EARLY_STOP_PATIENCE \
            AUTO_EPOCHS=$AUTO_EPOCHS EPOCHS_LARGE=$EPOCHS_LARGE EPOCHS_SMALL=$EPOCHS_SMALL \
            LR=$LR BATCH=$BATCH EXTRACT_BATCH=$EXTRACT_BATCH N_GPUS=1 \
            INPUT_DIR=$INPUT_DIR PKL_DIR=$_pkl PT_DIR=$_pt \
            OUT_ROOT=$OUT_ROOT OUT_SUFFIX=$_sfx \
            CKPT=$_ck CONFIG=$_cf INIT_FROM_THERMO=$_init \
            HEAD_HIDDEN=$HEAD_HIDDEN N_MP_LAYERS=$N_MP_LAYERS MP_N_HEADS=$MP_N_HEADS \
            MAX_K_PER_INPUT=${_maxk:-0} \
            ONLY_DATASETS=$_ds H_CACHE_DIR=$_pt \
            SPLIT_DIR_ROOT="${SPLIT_DIR_ROOT:-}" \
            WANDB=$WANDB WANDB_PROJECT=$WANDB_PROJECT WANDB_GROUP=$_sfx \
                bash scripts/run_downstream_pipeline.sh \
                >> "$LOG_DIR/${RUN_TAG}_cv_${_sfx}_${_ds}.log" 2>&1 &
        }

        declare -A _CV_PID_GPU; declare -A _CV_PID_TAG
        _cv_done=0; _cv_fail=0; _cv_idx=0

        for (( _gi=0; _gi < _CV_N_POOL && _cv_idx < _cv_total; _gi++, _cv_idx++ )); do
            _gpu="${_CV_GPU_IDS[$_gi]}"
            IFS='|' read -r _sfx _ck _cf _init _pkl _pt _k _ds _csv _maxk <<< "${_CV_QUEUE[$_cv_idx]}"
            _do_cv "$_gpu" "$_sfx" "$_ck" "$_cf" "$_init" "$_pkl" "$_pt" "$_k" "$_ds" "${_maxk:-0}"
            _pid=$!; _CV_PID_GPU[$_pid]=$_gpu; _CV_PID_TAG[$_pid]="${_sfx}/${_ds}"
            echo "[$(date +%T)] [$_cv_idx/$_cv_total] CV ${_sfx}/${_ds} → GPU $_gpu (pid=$_pid)"
        done
        while (( ${#_CV_PID_GPU[@]} > 0 )); do
            _fpid=0
            wait -n -p _fpid 2>/dev/null
            _wst=$?
            _fpid=${_fpid:-0}
            if (( _wst == 127 )); then
                # No more waitable children. Could mean (a) all done, or
                # (b) all children died before wait could see them and bash
                # auto-reaped. Either way, if PID_GPU is non-empty, mark
                # all remaining as FAILED so we don't silently drop them.
                if (( ${#_CV_PID_GPU[@]} > 0 )); then
                    echo "[$(date +%T)] WARN  wait returned 127 with ${#_CV_PID_GPU[@]} tracked pids — children gone before wait could see them. Marking as FAIL:"
                    for _orphan in "${!_CV_PID_GPU[@]}"; do
                        echo "[$(date +%T)] FAIL ${_CV_PID_TAG[$_orphan]} (gpu=${_CV_PID_GPU[$_orphan]}, pid=$_orphan) — check $LOG_DIR/${RUN_TAG}_cv_${_CV_PID_TAG[$_orphan]//\//_}.log"
                        _cv_fail=$((_cv_fail+1))
                        _cv_done=$((_cv_done+1))
                    done
                fi
                break
            fi
            [[ -z "${_CV_PID_GPU[$_fpid]:-}" ]] && continue
            _gpu="${_CV_PID_GPU[$_fpid]}"; _tag="${_CV_PID_TAG[$_fpid]}"
            unset '_CV_PID_GPU[$_fpid]' '_CV_PID_TAG[$_fpid]'
            _cv_done=$((_cv_done+1))
            (( _wst != 0 )) && { _cv_fail=$((_cv_fail+1)); echo "[$(date +%T)] FAIL $_tag (gpu=$_gpu) exit=$_wst — check $LOG_DIR/${RUN_TAG}_cv_${_tag//\//_}.log"; } \
                             || echo "[$(date +%T)] done $_tag (gpu=$_gpu)"
            if (( _cv_idx < _cv_total )); then
                IFS='|' read -r _sfx _ck _cf _init _pkl _pt _k _ds _csv _maxk <<< "${_CV_QUEUE[$_cv_idx]}"
                _cv_idx=$((_cv_idx+1))
                _do_cv "$_gpu" "$_sfx" "$_ck" "$_cf" "$_init" "$_pkl" "$_pt" "$_k" "$_ds" "${_maxk:-0}"
                _pid=$!; _CV_PID_GPU[$_pid]=$_gpu; _CV_PID_TAG[$_pid]="${_sfx}/${_ds}"
                echo "[$(date +%T)] [$_cv_idx/$_cv_total] CV ${_sfx}/${_ds} → GPU $_gpu (pid=$_pid)"
            fi
        done
        echo "CV done: $_cv_done total, $_cv_fail failed"
    fi
fi

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
_hdr "FINAL SUMMARY"
python3 - <<PY
import glob, json, os, math
from pathlib import Path

root = "$OUT_ROOT"
rows = []
for rep in sorted(glob.glob(os.path.join(root, "*/cv_report.json"))):
    suffix = Path(rep).parent.name
    try:
        d = json.load(open(rep))
        rows.append((suffix,
                     d.get("mae_mean", math.nan),
                     d.get("mae_last_stable_mean", math.nan),
                     d.get("r2_mean", math.nan),
                     sum(f.get("best_epoch",0) for f in d.get("folds",[]))
                     / max(len(d.get("folds",[])), 1)))
    except Exception as e:
        rows.append((suffix, math.nan, math.nan, math.nan, 0))

if not rows:
    print("No cv_report.json found under", root)
else:
    print(f"\n{'suffix':<36s}  {'best_val_MAE':>12s}  {'last_stab_MAE':>13s}  {'R²':>7s}  {'ep':>5s}")
    print("-" * 82)
    for suffix, mae_bv, mae_ls, r2, ep in rows:
        print(f"{suffix:<36s}  {mae_bv:>12.4f}  {mae_ls:>13.4f}  {r2:>7.3f}  {ep:>5.0f}")
PY

echo
echo "[$(date +'%F %T')] Done. Reports: $OUT_ROOT/<ds>_<config>/cv_report.json"
