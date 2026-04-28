#!/usr/bin/env bash
# End-to-end runner for the three downstream FT comparisons:
#
#   Test 1: thermo_flow_warm.ckpt  +  warm-init head  &  cold-init head (256/4/4)
#   Test 2: loqi_flow.ckpt         +  cold-init head (256/4/4)         (no thermo training)
#   Test 3: thermo_flow_warm.ckpt  +  warm-init head  +  LoRA r=16 (attn + FFN)
#
# All on the cleaned downstream tree (downstream_ft/clean/), K=8 conformer
# ensemble for training, val reports BOTH K=8 ensemble MAE and K=1
# per-conformer MAE in cv_report.json.
#
# Stages run sequentially:
#   0. clean_downstream.py            (idempotent)
#   1. K=8 sampling with thermo_flow_warm.ckpt        (~3-4h)
#   2. Test 1 — warm+cold_large CV     (~6-8h)
#   3. Test 2 — sample-with-loqi_flow + cold_large CV (~5-7h)
#   4. Test 3 — LoRA r=16 CV           (~9-15h, slowest)
#
# Total wall: roughly 24-30h on 4×A100. Each stage is independently
# resumable: a completed stage's outputs are detected and skipped.
#
# Run from repo root:
#   nohup bash scripts/run_three_tests.sh > /tmp/three_tests.log 2>&1 &
#   disown
#
# Override individual stages via SKIP_*=1 env vars.

set -uo pipefail
cd "$(dirname "$0")/.."

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

# Checkpoints
CKPT_WARM=${CKPT_WARM:-data/thermo_flow_warm.ckpt}
CONFIG_WARM=${CONFIG_WARM:-scripts/conf/loqi/loqi_thermo_flow_warm.yaml}
CKPT_FLOW=${CKPT_FLOW:-data/loqi_flow.ckpt}
CONFIG_FLOW=${CONFIG_FLOW:-scripts/conf/loqi/loqi_flow.yaml}

# Data tree (default = cleaned)
INPUT_DIR=${INPUT_DIR:-downstream_ft/clean}

# Per-test artefact roots
PKL_DIR_WARM=${PKL_DIR_WARM:-data/downstream_k${K}}
PT_DIR_WARM=${PT_DIR_WARM:-data/downstream_pt}
OUT_ROOT_WARM=${OUT_ROOT_WARM:-outputs/downstream_cv_K${K}_thermo_flow_warm}

PKL_DIR_FLOW=${PKL_DIR_FLOW:-data/downstream_k${K}_loqi}
PT_DIR_FLOW=${PT_DIR_FLOW:-data/downstream_pt_loqi}
OUT_ROOT_FLOW=${OUT_ROOT_FLOW:-outputs/downstream_cv_K${K}_loqi_flow}

# LoRA spec for Test 3
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_TARGET=${LORA_TARGET:-qkv_proj,out_projection,ffn,ffn_edge}

# WandB
WANDB=${WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-downstream_K${K}_clean}

# Skip flags (set to 1 to skip a stage)
SKIP_CLEAN=${SKIP_CLEAN:-0}
SKIP_SAMPLE_WARM=${SKIP_SAMPLE_WARM:-0}
SKIP_TEST1=${SKIP_TEST1:-0}
SKIP_TEST2=${SKIP_TEST2:-0}
SKIP_TEST3=${SKIP_TEST3:-0}

LOG_DIR=${LOG_DIR:-/tmp}
# ================================

echo "============================================================="
echo "  three-test runner"
echo "============================================================="
echo "  N_GPUS=$N_GPUS  K=$K  EPOCHS=$EPOCHS  PATIENCE=$EARLY_STOP_PATIENCE"
echo "  CKPT_WARM=$CKPT_WARM"
echo "  CKPT_FLOW=$CKPT_FLOW"
echo "  INPUT_DIR=$INPUT_DIR"
echo "  WANDB=$WANDB  WANDB_PROJECT=$WANDB_PROJECT"
echo "============================================================="

# Pre-flight
[[ -f "$CKPT_WARM"   ]] || { echo "ERROR: ckpt missing: $CKPT_WARM"   >&2; exit 1; }
[[ -f "$CONFIG_WARM" ]] || { echo "ERROR: config missing: $CONFIG_WARM" >&2; exit 1; }
[[ -f "$CKPT_FLOW"   ]] || { echo "ERROR: ckpt missing: $CKPT_FLOW"   >&2; exit 1; }
[[ -f "$CONFIG_FLOW" ]] || { echo "ERROR: config missing: $CONFIG_FLOW" >&2; exit 1; }

mkdir -p "$LOG_DIR"

_stage_header() {
    echo
    echo "============================================================="
    echo "  [$(date +'%F %T')]  $1"
    echo "============================================================="
}

# -----------------------------------------------------------------------
# STAGE 0 — clean downstream_ft → downstream_ft/clean (idempotent, fast)
# -----------------------------------------------------------------------
if [[ "$SKIP_CLEAN" == "1" ]]; then
    _stage_header "STAGE 0  SKIPPED  (SKIP_CLEAN=1)"
else
    _stage_header "STAGE 0  cleaning downstream CSVs"
    python scripts/clean_downstream.py 2>&1 | tee "$LOG_DIR/stage0_clean.log"
fi

# -----------------------------------------------------------------------
# STAGE 1 — K=8 sampling with thermo_flow_warm.ckpt
# Output: $PKL_DIR_WARM/<dataset>.pkl
# -----------------------------------------------------------------------
if [[ "$SKIP_SAMPLE_WARM" == "1" ]]; then
    _stage_header "STAGE 1  SKIPPED  (SKIP_SAMPLE_WARM=1)"
elif [[ -d "$PKL_DIR_WARM" ]] && (( $(find "$PKL_DIR_WARM" -name '*.pkl' | wc -l) >= 9 )); then
    _stage_header "STAGE 1  ALREADY DONE  ($PKL_DIR_WARM has ≥9 pickles)"
else
    _stage_header "STAGE 1  K=$K sampling (thermo_flow_warm)"
    K=$K \
    OUTPUT_DIR=$PKL_DIR_WARM \
    INPUT_DIR=$INPUT_DIR \
    FLOW_CKPT=$CKPT_WARM \
    FLOW_CONFIG=$CONFIG_WARM \
    N_GPUS=$N_GPUS \
        bash scripts/sample_downstream_K5.sh 2>&1 | tee "$LOG_DIR/stage1_sample_warm.log"
fi

# -----------------------------------------------------------------------
# STAGE 2 — Test 1: warm + cold_large head modes on thermo_flow_warm.ckpt
# -----------------------------------------------------------------------
if [[ "$SKIP_TEST1" == "1" ]]; then
    _stage_header "STAGE 2  SKIPPED  (SKIP_TEST1=1)"
else
    _stage_header "STAGE 2  Test 1 — warm + cold_large CV (thermo_flow_warm)"
    MODES="warm cold_large" \
    SKIP_SAMPLING=1 \
    K=$K EPOCHS=$EPOCHS EARLY_STOP_PATIENCE=$EARLY_STOP_PATIENCE \
    LR=$LR BATCH=$BATCH N_GPUS=$N_GPUS \
    INPUT_DIR=$INPUT_DIR \
    PKL_DIR=$PKL_DIR_WARM \
    PT_DIR=$PT_DIR_WARM \
    OUT_ROOT=$OUT_ROOT_WARM \
    CKPT=$CKPT_WARM \
    CONFIG=$CONFIG_WARM \
    WANDB=$WANDB \
    WANDB_PROJECT=$WANDB_PROJECT \
        bash scripts/run_downstream_K8_full.sh 2>&1 | tee "$LOG_DIR/stage2_test1.log"
fi

# -----------------------------------------------------------------------
# STAGE 3 — Test 2: loqi_flow.ckpt (no thermo training) + cold_large head
# Note: warm-init mode would crash here (loqi_flow's config has no
# thermo_head_args), so MODES is restricted to cold_large.
# -----------------------------------------------------------------------
if [[ "$SKIP_TEST2" == "1" ]]; then
    _stage_header "STAGE 3  SKIPPED  (SKIP_TEST2=1)"
else
    _stage_header "STAGE 3  Test 2 — sample + CV (loqi_flow, cold_large only)"
    MODES="cold_large" \
    SKIP_SAMPLING=0 \
    K=$K EPOCHS=$EPOCHS EARLY_STOP_PATIENCE=$EARLY_STOP_PATIENCE \
    LR=$LR BATCH=$BATCH N_GPUS=$N_GPUS \
    INPUT_DIR=$INPUT_DIR \
    PKL_DIR=$PKL_DIR_FLOW \
    PT_DIR=$PT_DIR_FLOW \
    OUT_ROOT=$OUT_ROOT_FLOW \
    CKPT=$CKPT_FLOW \
    CONFIG=$CONFIG_FLOW \
    WANDB=$WANDB \
    WANDB_PROJECT=$WANDB_PROJECT \
        bash scripts/run_downstream_K8_full.sh 2>&1 | tee "$LOG_DIR/stage3_test2.log"
fi

# -----------------------------------------------------------------------
# STAGE 4 — Test 3: LoRA r=16, attention + FFN, warm-init head, on
# thermo_flow_warm.ckpt. Reuses STAGE 1's pickles + STAGE 2's .pt files
# (same backbone → same conformers → same .pt; the only difference is
# the LoRA-adapted backbone forward in the FT pass).
# -----------------------------------------------------------------------
if [[ "$SKIP_TEST3" == "1" ]]; then
    _stage_header "STAGE 4  SKIPPED  (SKIP_TEST3=1)"
else
    _stage_header "STAGE 4  Test 3 — LoRA r=$LORA_R warm head"
    LORA_R=$LORA_R \
    LORA_ALPHA=$LORA_ALPHA \
    LORA_TARGET=$LORA_TARGET \
    INIT_FROM_THERMO=1 \
    OUT_SUFFIX=lora_r${LORA_R}_warm_attnffn \
    K=$K EPOCHS=$EPOCHS EARLY_STOP_PATIENCE=$EARLY_STOP_PATIENCE \
    LR=$LR BATCH=$BATCH N_GPUS=$N_GPUS \
    INPUT_DIR=$INPUT_DIR \
    PKL_DIR=$PKL_DIR_WARM \
    PT_DIR=$PT_DIR_WARM \
    OUT_ROOT=$OUT_ROOT_WARM \
    CKPT=$CKPT_WARM \
    CONFIG=$CONFIG_WARM \
    WANDB=$WANDB \
    WANDB_PROJECT=$WANDB_PROJECT \
    WANDB_GROUP=lora_r${LORA_R} \
    SLEEP_HOURS=0 \
        bash scripts/run_downstream_pipeline.sh 2>&1 | tee "$LOG_DIR/stage4_test3.log"
fi

# -----------------------------------------------------------------------
# FINAL SUMMARY — collect all results into one table
# -----------------------------------------------------------------------
_stage_header "FINAL SUMMARY"
python3 - <<PY
import glob, json, os
roots = [
    ("$OUT_ROOT_WARM", ["warm", "cold_large", "lora_r${LORA_R}_warm_attnffn"]),
    ("$OUT_ROOT_FLOW", ["cold_large"]),
]

print(f"\n{'dataset':<14s}  {'mode':<28s}  {'ens_MAE':>9s}  {'1conf_MAE':>10s}  {'R²(ens)':>9s}  {'best_ep':>8s}")
print("-" * 92)

for root, modes in roots:
    label = "thermo_flow_warm" if "warm" in root else "loqi_flow"
    for mode in modes:
        for rep_path in sorted(glob.glob(os.path.join(root, f"*_{mode}/cv_report.json"))):
            ds = os.path.basename(os.path.dirname(rep_path)).replace(f"_{mode}", "")
            try:
                d = json.load(open(rep_path))
                pc_mae  = d.get("mae_per_conformer_mean", float("nan"))
                ep_mean = float(sum(f.get("best_epoch", 0) for f in d.get("folds", []))
                                 / max(len(d.get("folds", [])), 1)) if d.get("folds") else 0.0
                print(f"{ds:<14s}  {label}/{mode:<18s}  "
                      f"{d['mae_mean']:>9.3f}  {pc_mae:>10.3f}  "
                      f"{d['r2_mean']:>9.3f}  {ep_mean:>8.0f}")
            except Exception as e:
                print(f"{ds:<14s}  {label}/{mode:<18s}  (parse error: {e})")
PY

echo
echo "[$(date +'%F %T')] all stages complete."
echo "Reports under:"
echo "  $OUT_ROOT_WARM/<dataset>_<mode>/cv_report.json"
echo "  $OUT_ROOT_FLOW/<dataset>_<mode>/cv_report.json"
