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

INPUT_DIR=${INPUT_DIR:-downstream_ft/0506/cleaned_by_CC/cleaned_by_codex}

# ---- Checkpoint definitions ------------------------------------------------
# Format: "label|ckpt_path|config_path|init_from_thermo"
# init_from_thermo=1 → warm-start downstream head from ckpt's thermo_heads
# init_from_thermo=0 → random-init head (loqi_flow has no thermo head)
CKPT_DEFS=(
    "loqi_flow|data/ft_ckpts/loqi_flow.ckpt|scripts/conf/loqi/loqi_flow.yaml|0"
    "cold_warm|data/ft_ckpts/thermo_flow_warm.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold.yaml|0"
    "cold_early|data/ft_ckpts/thermo_flow_cold_early.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold.yaml|0"
    "cold_late|data/ft_ckpts/thermo_flow_cold_late.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold.yaml|0"
    # Add new ckpt here: "label|path/to/ckpt|config.yaml|0"
    # "new_model|data/ft_ckpts/new_model.ckpt|scripts/conf/loqi/loqi_thermo_flow_cold.yaml|0"
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
# Stage B — Sampling per checkpoint
# -----------------------------------------------------------------------
if [[ "$SKIP_SAMPLE" == "1" ]]; then
    _hdr "Sampling SKIPPED"
else
    for def in "${CKPT_DEFS[@]}"; do
        IFS='|' read -r label ckpt cfg _init <<< "$def"

        # K=8 standard
        pkl_ss="data/0506_pkl_${label}_k8"
        mkdir -p "$pkl_ss"
        n_ss=$(find "$pkl_ss" -name "*.pkl" | wc -l)
        if (( n_ss >= ${#DATASETS_CSV[@]} )); then
            _hdr "[$label] K=8 sampling SKIPPED (${n_ss} pkls found)"
        else
            _hdr "[$label] K=8 standard sampling"
            CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
            K=$K_SS N_STEPS=$N_STEPS_SS \
            OUTPUT_DIR=$pkl_ss \
            INPUT_DIR=$INPUT_DIR \
            FLOW_CKPT=$ckpt FLOW_CONFIG=$cfg \
            N_GPUS=$N_GPUS \
                bash scripts/sample_downstream_K5.sh \
                2>&1 | tee -a "$LOG_DIR/0506_sample_${label}_k8.log"
        fi

        # K=12 multi-snapshot
        pkl_ms="data/0506_pkl_${label}_k12ms"
        mkdir -p "$pkl_ms"
        n_ms=$(find "$pkl_ms" -name "*.pkl" | wc -l)
        if (( n_ms >= ${#DATASETS_CSV[@]} )); then
            _hdr "[$label] K=12 multi-snap SKIPPED"
        else
            _hdr "[$label] K=12 multi-snapshot (4 traj × steps 7,8,9)"
            for csv in "${DATASETS_CSV[@]}"; do
                name=$(basename "$csv" .csv)
                smi="$SMI_DIR/$name.smi"
                pkl="$pkl_ms/$name.pkl"
                [[ -f "$pkl" ]] && continue
                [[ -f "$smi" ]] || { echo "  [WARN] no .smi for $name"; continue; }
                CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
                    python scripts/sample_conformers_multistep.py \
                        --ckpt "$ckpt" --config "$cfg" \
                        --input "$smi" --output "$pkl" \
                        --n_traj $N_TRAJ --n_steps $N_STEPS_MS \
                        --snapshot_steps $SNAPSHOT_STEPS \
                        --batch_size $BATCH \
                    >> "$LOG_DIR/0506_ms_${label}_${name}.log" 2>&1
                echo "  [ok] $pkl"
            done
        fi
    done
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
