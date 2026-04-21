#!/usr/bin/env bash
# Prepare + 5-fold CV on every CSV under $CSV_DIR.
#
# Each CSV is assumed to have a SMILES column and a single target column.
# You list them as pairs in the TASKS array below.
#
# Example layout:
#     data/downstream/delaney.csv    (smiles, measured log solubility in mols per litre)
#     data/downstream/bace.csv       (mol, pIC50)
#     data/downstream/esol.csv       ...
#
# Usage:
#     bash scripts/run_downstream_all.sh
#
# Edit the CONFIG + TASKS blocks before running.

set -euo pipefail
cd "$(dirname "$0")/.."

# ============ CONFIG ============
CKPT=data/loqi.ckpt
LOQI_CONFIG=scripts/conf/loqi/loqi.yaml
OUT_ROOT=/tmp/downstream_cv
PT_DIR=data/downstream_pt           # where prepared .pt files go
N_FOLDS=5
EPOCHS=50
BATCH_SIZE=64
LR=3e-4
HEAD_HIDDEN=256
N_MP_LAYERS=2
MP_N_HEADS=4
DEVICE=cuda
SEED=42

# Each row: csv_path|smiles_col|target_col|short_name
TASKS=(
    "data/downstream/delaney.csv|smiles|measured log solubility in mols per litre|delaney"
    # "data/downstream/bace.csv|mol|pIC50|bace"
    # "data/downstream/esol.csv|smiles|measured log solubility|esol"
)
# ================================

mkdir -p "$OUT_ROOT" "$PT_DIR"

for row in "${TASKS[@]}"; do
    IFS='|' read -r CSV SMI_COL TGT_COL NAME <<<"$row"
    PT="$PT_DIR/$NAME.pt"
    OUT="$OUT_ROOT/$NAME"

    echo
    echo "==============================================================="
    echo "TASK: $NAME   (csv=$CSV, target=$TGT_COL)"
    echo "==============================================================="

    if [[ ! -f "$PT" ]]; then
        echo "[prep] $CSV -> $PT"
        python scripts/prepare_downstream_dataset.py \
            --csv "$CSV" \
            --smiles-col "$SMI_COL" \
            --target-col "$TGT_COL" \
            --output "$PT"
    else
        echo "[prep] $PT already exists, skipping conformer embedding"
    fi

    if [[ -f "$OUT/cv_report.json" ]]; then
        echo "[cv]  $OUT/cv_report.json exists, skipping"
        continue
    fi

    python scripts/downstream_cv.py \
        --ckpt "$CKPT" --config "$LOQI_CONFIG" \
        --dataset-pt "$PT" \
        --out-dir "$OUT" \
        --n-folds "$N_FOLDS" \
        --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --lr "$LR" \
        --head-hidden "$HEAD_HIDDEN" \
        --n-mp-layers "$N_MP_LAYERS" --mp-n-heads "$MP_N_HEADS" \
        --device "$DEVICE" --seed "$SEED" 2>&1 | tee "$OUT/train.log"
done

# --- Final cross-dataset summary ---
python3 - <<'PY'
import json, glob, os
print("\n" + "=" * 78)
print(f"{'dataset':<24s} {'n':>7s} {'ext MAE':>12s} {'mp MAE':>12s} {'best R²':>10s}")
print("-" * 78)
for path in sorted(glob.glob(os.path.join(os.environ.get("OUT_ROOT", "/tmp/downstream_cv"),
                                            "*/cv_report.json"))):
    name = os.path.basename(os.path.dirname(path))
    d = json.load(open(path))
    best_r2 = max(d["ext"]["r2_mean"], d["mp"]["r2_mean"])
    print(f"{name:<24s} {d['n_labeled']:>7d} "
          f"{d['ext']['mae_mean']:>6.3f}±{d['ext']['mae_std']:<4.3f} "
          f"{d['mp']['mae_mean']:>6.3f}±{d['mp']['mae_std']:<4.3f} "
          f"{best_r2:>10.3f}")
print("=" * 78)
PY
