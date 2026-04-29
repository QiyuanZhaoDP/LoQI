#!/usr/bin/env bash
# End-to-end driver for the K=8 + 200-epoch + early-stopping benchmark
# across all 9 downstream datasets, in three modes:
#
#   1. warm        - head warm-init from ckpt's thermo head (current best
#                    practice). Tests "do we get extra mileage from a
#                    longer schedule + early stop on top of K=8?"
#   2. cold_small  - head trained from random init, smaller capacity
#                    (head_hidden=128, n_mp_layers=2). Tests "does a
#                    smaller head trained from scratch generalize better?"
#   3. cold_large  - head trained from random init at the same capacity
#                    as the warm thermo head (head_hidden=256, n_mp=4,
#                    mp_n_heads=4). Tests "does the warm-init from thermo
#                    head actually help, or is it equivalent to random?"
#
# Pipeline (each run gates on the next, no inter-stage sleeping):
#   Stage 1: K=8 conformer sampling for every dataset (uses
#            sample_downstream_K5.sh with K=8 OUTPUT_DIR=data/downstream_k8).
#            Skipped per-dataset if the .pkl already exists.
#   Stage 2: Three CV passes — warm / cold_small / cold_large — sequentially.
#            Each one dispatches its 9 datasets across N_GPUS GPUs.
#
# Usage:
#   nohup bash scripts/run_downstream_K8_full.sh > downstream_K8.log 2>&1 &
#   disown
#
# Env knobs (with sensible defaults):
#   N_GPUS=4  K=8  EPOCHS=200  EARLY_STOP_PATIENCE=30  LR=3e-4  BATCH=64
#   WANDB=0   (set 1 to push curves to wandb)
#   SKIP_SAMPLING=0   (set 1 to skip Stage 1 entirely; assumes pkls exist)

set -uo pipefail
cd "$(dirname "$0")/.."

# bash 5.1 needed for run_downstream_pipeline.sh's `wait -n -p`.
if (( BASH_VERSINFO[0] < 5 )) || { (( BASH_VERSINFO[0] == 5 )) && (( BASH_VERSINFO[1] < 1 )); }; then
    echo "ERROR: bash >= 5.1 required (you have $BASH_VERSION)" >&2
    exit 1
fi

# ============ CONFIG ============
N_GPUS=${N_GPUS:-4}
K=${K:-8}
EPOCHS=${EPOCHS:-200}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-30}
LR=${LR:-3e-4}
BATCH=${BATCH:-64}

CKPT=${CKPT:-data/thermo_flow_warm.ckpt}
CONFIG=${CONFIG:-scripts/conf/loqi/loqi_thermo_flow_warm.yaml}

# Default to the cleaned, deduplicated CSVs from
# scripts/clean_downstream.py. Override with INPUT_DIR=downstream_ft to
# run against raw data.
INPUT_DIR=${INPUT_DIR:-downstream_ft/clean}
PKL_DIR=${PKL_DIR:-data/downstream_k${K}}
PT_DIR=${PT_DIR:-data/downstream_pt}
OUT_ROOT=${OUT_ROOT:-outputs/downstream_cv_K${K}}

WANDB=${WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-downstream_cv}

SKIP_SAMPLING=${SKIP_SAMPLING:-0}

# Subset of modes to actually run. Default: all three. Use this to skip
# modes you don't need — e.g. for an alternate-backbone control where
# warm-init has no thermo head to load and cold_small adds little:
#   MODES="cold_large" bash scripts/run_downstream_K8_full.sh
MODES=${MODES:-"warm cold_small cold_large"}
# ================================

mkdir -p "$PKL_DIR" "$PT_DIR" "$OUT_ROOT"

[[ -f "$CKPT"   ]] || { echo "ERROR: ckpt missing: $CKPT"     >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "ERROR: config missing: $CONFIG" >&2; exit 1; }

echo "==========================================================="
echo " K=$K  EPOCHS=$EPOCHS  EARLY_STOP=$EARLY_STOP_PATIENCE  N_GPUS=$N_GPUS"
echo " ckpt    : $CKPT"
echo " pkl_dir : $PKL_DIR"
echo " out_root: $OUT_ROOT"
echo "==========================================================="

# -----------------------------------------------------------------------
# Stage 1: K=8 conformer sampling (gated; reuses sample_downstream_K5.sh
# which is already K-parametric).
# -----------------------------------------------------------------------
if [[ "$SKIP_SAMPLING" == "1" ]]; then
    echo "[$(date +%T)] Stage 1 SKIPPED (SKIP_SAMPLING=1)"
else
    echo "[$(date +%T)] Stage 1: K=$K sampling for all datasets under $INPUT_DIR"
    K=$K \
    OUTPUT_DIR=$PKL_DIR \
    FLOW_CKPT=$CKPT \
    FLOW_CONFIG=$CONFIG \
    INPUT_DIR=$INPUT_DIR \
    N_GPUS=$N_GPUS \
        bash scripts/sample_downstream_K5.sh \
        2>&1 | tee -a "$OUT_ROOT/_sampling.log"
    echo "[$(date +%T)] Stage 1 done."
fi

# -----------------------------------------------------------------------
# Stage 2: Three CV passes. Each one drives run_downstream_pipeline.sh
# with mode-specific env overrides; SLEEP_HOURS=0 so we don't pause.
# -----------------------------------------------------------------------

_run_mode() {
    local suffix="$1"            # warm | cold_small | cold_large
    local init_from_thermo="$2"  # 0 | 1
    local head_hidden="$3"
    local n_mp_layers="$4"
    local mp_n_heads="$5"

    echo
    echo "==========================================================="
    echo "[$(date +%T)] Stage 2 / mode=$suffix"
    echo "  init_from_thermo=$init_from_thermo  head=${head_hidden}/${n_mp_layers}/${mp_n_heads}"
    echo "==========================================================="

    SLEEP_HOURS=0 \
    K=$K \
    EPOCHS=$EPOCHS \
    EARLY_STOP_PATIENCE=$EARLY_STOP_PATIENCE \
    LR=$LR \
    BATCH=$BATCH \
    N_GPUS=$N_GPUS \
    CKPT=$CKPT \
    CONFIG=$CONFIG \
    INPUT_DIR=$INPUT_DIR \
    PKL_DIR=$PKL_DIR \
    PT_DIR=$PT_DIR \
    OUT_ROOT=$OUT_ROOT \
    OUT_SUFFIX=$suffix \
    INIT_FROM_THERMO=$init_from_thermo \
    HEAD_HIDDEN=$head_hidden \
    N_MP_LAYERS=$n_mp_layers \
    MP_N_HEADS=$mp_n_heads \
    WANDB=$WANDB \
    WANDB_PROJECT=$WANDB_PROJECT \
    WANDB_GROUP=$suffix \
    ONLY_DATASETS="${ONLY_DATASETS:-}" \
    SKIP_DATASETS="${SKIP_DATASETS:-}" \
        bash scripts/run_downstream_pipeline.sh \
        2>&1 | tee -a "$OUT_ROOT/_${suffix}.log"
}

# warm:        head warm-init; head dims auto-aligned to thermo_head_args
# cold_small:  no warm-init; small head 128/2/4
# cold_large:  no warm-init; same dims as warm 256/4/4 (apples-to-apples)
_modes_list=" $MODES "
[[ "$_modes_list" == *" warm "*       ]] && _run_mode warm       1 256 4 4
[[ "$_modes_list" == *" cold_small "* ]] && _run_mode cold_small 0 128 2 4
[[ "$_modes_list" == *" cold_large "* ]] && _run_mode cold_large 0 256 4 4

# -----------------------------------------------------------------------
# Stage 3: Aggregated table across all 3 modes (mae_mean ± std per dataset).
# -----------------------------------------------------------------------
echo
echo "==========================================================="
echo " Final cross-mode summary"
echo "==========================================================="
python3 - "$OUT_ROOT" <<'PY'
import sys, json, glob, os
root = sys.argv[1]
modes = ["warm", "cold_small", "cold_large"]

# Collect: results[dataset][mode] = (mae_mean, mae_std, r2_mean)
results = {}
for mode in modes:
    for rep in sorted(glob.glob(os.path.join(root, f"*_{mode}/cv_report.json"))):
        name = os.path.basename(os.path.dirname(rep)).replace(f"_{mode}", "")
        try:
            d = json.load(open(rep))
            results.setdefault(name, {})[mode] = (
                d["mae_mean"], d.get("mae_std", float("nan")), d["r2_mean"],
            )
        except Exception as e:
            results.setdefault(name, {})[mode] = ("err", "err", str(e))

if not results:
    print("(no cv_report.json found)")
    sys.exit(0)

# Header
header = f"{'dataset':<12s}"
for mode in modes:
    header += f"  {mode:>22s}"
print(header)
print("-" * len(header))

for name in sorted(results):
    line = f"{name:<12s}"
    for mode in modes:
        cell = results[name].get(mode)
        if cell is None:
            line += f"  {'—':>22s}"
        elif cell[0] == "err":
            line += f"  {'(err)':>22s}"
        else:
            mae, std, r2 = cell
            line += f"  {mae:>8.3f}±{std:<6.3f} R²={r2:>5.3f}"
    print(line)
PY

echo
echo "[$(date +%T)] All stages complete. Reports under $OUT_ROOT/<dataset>_<mode>/cv_report.json"
