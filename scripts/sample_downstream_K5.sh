#!/usr/bin/env bash
# Generate K conformers for every CSV in a downstream-FT directory using
# the warm-trained flow checkpoint. Auto-detects the SMILES column
# (case-insensitive: SMILES, smiles, ...) so heterogeneous CSVs from
# different sources work without per-file editing.
#
# Output layout:
#   data/downstream_k5/<csv_basename>.pkl      — sample_conformers.py pickle
#   data/downstream_k5/<csv_basename>.smi      — extracted SMILES (intermediate)
#   data/downstream_k5/<csv_basename>.log      — per-dataset log
#
# Usage:
#   bash scripts/sample_downstream_K5.sh
#
# Edit the CONFIG block below for paths / K / GPU.

set -euo pipefail
cd "$(dirname "$0")/.."

# ============ CONFIG ============
INPUT_DIR=${INPUT_DIR:-downstream_ft}              # CSVs to process
OUTPUT_DIR=${OUTPUT_DIR:-data/downstream_k5}       # where pkls / smi / logs go

FLOW_CKPT=${FLOW_CKPT:-data/thermo_flow_warm.ckpt}
FLOW_CONFIG=${FLOW_CONFIG:-scripts/conf/loqi/loqi_thermo_flow_warm.yaml}

K=${K:-5}                       # conformers per molecule
N_STEPS=${N_STEPS:-10}          # flow integration steps
BATCH_SIZE=${BATCH_SIZE:-64}    # sampling batch size
POSTPROCESS=${POSTPROCESS:-optimization}    # "none" | "optimization" | "optimization+irmsd"
                                            # ("optimization+irmsd" can drop conformers, breaking
                                            #  the prepare_downstream_K_pt position-based join —
                                            #  use plain "optimization" unless you handle that.)
OPT_MAX_NSTEP=${OPT_MAX_NSTEP:-100}
GPU_ID=${GPU_ID:-0}

FORCE=${FORCE:-0}               # set FORCE=1 to re-run already-finished CSVs
# ================================

mkdir -p "$OUTPUT_DIR"

# Pre-flight
if [[ ! -f "$FLOW_CKPT" ]]; then
    echo "ERROR: ckpt not found: $FLOW_CKPT" >&2
    echo "Hint: cp 'outputs/.../checkpoints/best-epoch=...ckpt' $FLOW_CKPT" >&2
    exit 1
fi
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "ERROR: input dir not found: $INPUT_DIR" >&2
    exit 1
fi

# Helper: extract SMILES column from a CSV → .smi (one SMILES per line, no header).
# Auto-detects the column case-insensitively. Drops empty / NaN rows.
extract_smiles() {
    local csv="$1"
    local out="$2"
    python3 - "$csv" "$out" <<'PY'
import sys, csv, pandas as pd
csv_path, out_path = sys.argv[1], sys.argv[2]
df = pd.read_csv(csv_path)
# case-insensitive smiles column lookup
matches = [c for c in df.columns if c.lower() == "smiles"]
if not matches:
    raise SystemExit(f"No SMILES column found in {csv_path}.  Columns: {list(df.columns)}")
col = matches[0]
ser = df[col].astype(str).str.strip()
ser = ser[ser.str.len() > 0]
ser = ser[~ser.str.lower().isin(("nan", "none"))]
ser.to_csv(out_path, index=False, header=False)
print(f"  {col!r}: {len(ser):,} SMILES -> {out_path}")
PY
}

# Recursively walk INPUT_DIR for all *.csv (handles flat CSVs and pre-split
# subdirs like delaney_s/{train,valid,test}.csv equally). Output paths
# mirror the input structure under OUTPUT_DIR.
mapfile -t csvs < <(find "$INPUT_DIR" -type f -name '*.csv' | sort)
if [[ ${#csvs[@]} -eq 0 ]]; then
    echo "No CSVs found under $INPUT_DIR" >&2
    exit 1
fi
echo "Found ${#csvs[@]} CSVs under $INPUT_DIR"

start=$(date +%s)
n_done=0; n_skipped=0; n_failed=0

for csv in "${csvs[@]}"; do
    # Preserve subdir structure: e.g. delaney_s/train.csv -> delaney_s/train.{smi,pkl,log}
    rel_path="${csv#$INPUT_DIR/}"            # strip "$INPUT_DIR/" prefix
    rel_dir=$(dirname "$rel_path")
    name=$(basename "$rel_path" .csv)
    out_dir="$OUTPUT_DIR"
    [[ "$rel_dir" != "." ]] && out_dir="$OUTPUT_DIR/$rel_dir"
    mkdir -p "$out_dir"
    smi="$out_dir/$name.smi"
    pkl="$out_dir/$name.pkl"
    log="$out_dir/$name.log"

    echo
    echo "============================================================"
    echo "  $rel_path"
    echo "============================================================"

    if [[ -f "$pkl" && "$FORCE" != "1" ]]; then
        echo "  [skip] $pkl already exists. Set FORCE=1 to re-run."
        n_skipped=$((n_skipped + 1))
        continue
    fi

    # Step 1: extract SMILES
    if [[ ! -f "$smi" || "$FORCE" == "1" ]]; then
        extract_smiles "$csv" "$smi"
    else
        echo "  [reuse] $smi"
    fi

    n_smi=$(wc -l < "$smi" | tr -d ' ')
    echo "  → sampling K=$K, n_steps=$N_STEPS, postprocess=$POSTPROCESS  ($n_smi mols)"

    # Step 2: sample
    if CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/sample_conformers.py \
            --ckpt   "$FLOW_CKPT" \
            --config "$FLOW_CONFIG" \
            --input  "$smi" \
            --output "$pkl" \
            --n_confs "$K" \
            --n_steps "$N_STEPS" \
            --batch_size "$BATCH_SIZE" \
            --postprocess "$POSTPROCESS" \
            --opt_max_nstep "$OPT_MAX_NSTEP" \
            > "$log" 2>&1; then
        echo "  [done] $pkl  (log: $log)"
        n_done=$((n_done + 1))
    else
        echo "  [FAIL] see $log" >&2
        n_failed=$((n_failed + 1))
    fi
done

echo
echo "============================================================"
echo "  Summary  (wall: $(( $(date +%s) - start ))s)"
echo "============================================================"
echo "  done:    $n_done"
echo "  skipped: $n_skipped"
echo "  failed:  $n_failed"
echo "  outputs: $OUTPUT_DIR/<name>.pkl"
