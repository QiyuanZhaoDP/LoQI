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
    local _N_POOL=${#_GPU_IDS[@]}

    # Build task queue: "mode|label|ckpt|cfg|smi|pkl|name"
    declare -a _QUEUE=()
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
        declare -A _PID_GPU _PID_TAG
        _done=0 _fail=0 _idx=0

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
# Stage C — Downstream CV per (ckpt, sampling_mode)
# -----------------------------------------------------------------------
_run_cv() {
    local label="$1" ckpt="$2" cfg="$3" init_thermo="$4" pkl_dir="$5" pt_dir="$6" k_eff="$7" suffix="$8"
    mkdir -p "$pt_dir"

    # Check at least one pickle exists for this ckpt+mode.
    local n_pkls
    n_pkls=$(find "$pkl_dir" -name "*.pkl" 2>/dev/null | wc -l)
    if (( n_pkls == 0 )); then
        echo "  [WARN] no pickles found in $pkl_dir — skipping $suffix"
        return
    fi

    # run_downstream_pipeline.sh auto-discovers DATASETS from INPUT_DIR
    # and reads SMILES/TARGET column names directly from each CSV header,
    # so no DATASETS export needed here.
    _hdr "CV: $suffix  (K=$k_eff, ckpt=$label)"
    SLEEP_HOURS=0 \
    K=$k_eff EPOCHS=$EPOCHS EARLY_STOP_PATIENCE=$EARLY_STOP_PATIENCE \
    AUTO_EPOCHS=$AUTO_EPOCHS EPOCHS_LARGE=$EPOCHS_LARGE EPOCHS_SMALL=$EPOCHS_SMALL \
    LR=$LR BATCH=$BATCH N_GPUS=$N_GPUS \
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
    INPUT_DIR=$INPUT_DIR \
    PKL_DIR=$pkl_dir PT_DIR=$pt_dir \
    OUT_ROOT=$OUT_ROOT OUT_SUFFIX=$suffix \
    CKPT=$ckpt CONFIG=$cfg \
    INIT_FROM_THERMO=$init_thermo \
    HEAD_HIDDEN=$HEAD_HIDDEN N_MP_LAYERS=$N_MP_LAYERS MP_N_HEADS=$MP_N_HEADS \
    WANDB=$WANDB WANDB_PROJECT=$WANDB_PROJECT WANDB_GROUP=$suffix \
        bash scripts/run_downstream_pipeline.sh \
        2>&1 | tee -a "$LOG_DIR/0506_cv_${suffix}.log"
}

if [[ "$SKIP_CV" == "1" ]]; then
    _hdr "CV SKIPPED"
else
    for def in "${CKPT_DEFS[@]}"; do
        IFS='|' read -r label ckpt cfg init_thermo <<< "$def"

        # M0: K=8
        _run_cv "$label" "$ckpt" "$cfg" "$init_thermo" \
            "data/0506_pkl_${label}_k8" \
            "data/0506_pt_${label}_k8" \
            $K_SS \
            "${label}_K8"

        # M2: K=12 multi-snap
        _run_cv "$label" "$ckpt" "$cfg" "$init_thermo" \
            "data/0506_pkl_${label}_k12ms" \
            "data/0506_pt_${label}_k12ms" \
            12 \
            "${label}_K12ms"
    done
fi

# -----------------------------------------------------------------------
# Final summary
# -----------------------------------------------------------------------
_hdr "FINAL SUMMARY"
python3 - <<PY
import glob, json, os
root = "$OUT_ROOT"
from pathlib import Path

print(f"\n{'dataset':<14s}  {'ckpt_mode':<26s}  {'ens_MAE':>9s}  {'1conf_MAE':>10s}  {'R²':>7s}  {'ep':>5s}")
print("-" * 82)
for rep in sorted(glob.glob(os.path.join(root, "*/cv_report.json"))):
    suffix = Path(rep).parent.name   # e.g. gas_Hf_cold_late_K8
    # split suffix: last two parts are <label>_<mode>
    parts = suffix.rsplit("_", 2)
    ds = parts[0] if len(parts) == 3 else suffix
    mode = "_".join(parts[1:]) if len(parts) >= 2 else ""
    try:
        d = json.load(open(rep))
        ens = d["mae_mean"]; pc = d.get("mae_per_conformer_mean", float("nan"))
        r2  = d["r2_mean"]
        ep  = sum(f.get("best_epoch",0) for f in d.get("folds",[])) / max(len(d.get("folds",[])),1)
        print(f"{ds:<14s}  {mode:<26s}  {ens:>9.3f}  {pc:>10.3f}  {r2:>7.3f}  {ep:>5.0f}")
    except Exception as e:
        print(f"{suffix}  (err: {e})")
PY

echo; echo "[$(date +'%F %T')] Done. Outputs under $OUT_ROOT/"
