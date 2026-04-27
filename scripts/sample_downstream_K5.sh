#!/usr/bin/env bash
# Generate K conformers for every CSV in a downstream-FT directory using
# the warm-trained flow checkpoint. Auto-detects the SMILES column
# (case-insensitive: SMILES, smiles, ...) so heterogeneous CSVs from
# different sources work without per-file editing.
#
# Multi-GPU worker pool: dispatches one CSV per GPU; the next CSV in the
# queue is launched the moment any GPU frees. N_GPUS=1 (default) gives
# the original sequential behaviour. Auto-detects via nvidia-smi when
# N_GPUS is unset.
#
# Output layout (mirrors INPUT_DIR's subdir structure):
#   data/downstream_k5/<csv_basename>.pkl      — sample_conformers.py pickle
#   data/downstream_k5/<csv_basename>.smi      — extracted SMILES (intermediate)
#   data/downstream_k5/<csv_basename>.log      — per-dataset log
#
# Usage:
#   bash scripts/sample_downstream_K5.sh                # uses default N_GPUS
#   N_GPUS=4 bash scripts/sample_downstream_K5.sh       # 4-GPU worker pool
#
# Requires bash >= 5.1 for `wait -n -p` only when N_GPUS > 1.

set -uo pipefail        # NOTE: no -e — one CSV's failure shouldn't abort the rest
cd "$(dirname "$0")/.."

# ============ CONFIG ============
INPUT_DIR=${INPUT_DIR:-downstream_ft}              # CSVs to process (recursive)
OUTPUT_DIR=${OUTPUT_DIR:-data/downstream_k5}       # where pkls / smi / logs go

FLOW_CKPT=${FLOW_CKPT:-data/thermo_flow_warm.ckpt}
FLOW_CONFIG=${FLOW_CONFIG:-scripts/conf/loqi/loqi_thermo_flow_warm.yaml}

K=${K:-5}                       # conformers per molecule
N_STEPS=${N_STEPS:-10}          # flow integration steps
BATCH_SIZE=${BATCH_SIZE:-64}    # sampling batch size
POSTPROCESS=${POSTPROCESS:-none}            # "none" | "optimization" | "optimization+irmsd"
                                            # Default `none`: downstream FT just needs reasonable
                                            # conformers, not AIMNet2-relaxed minima. Optimization
                                            # also crashes on batches whose mols contain elements
                                            # outside AIMNet2's coverage (`torch.cat(): expected a
                                            # non-empty list`). Set POSTPROCESS=optimization to
                                            # opt-in (after running this with the element filter).
OPT_MAX_NSTEP=${OPT_MAX_NSTEP:-100}

# GPU pool. Empty → auto-detect via nvidia-smi. Leave at 1 to keep the
# original sequential behaviour. Single-GPU runs still respect GPU_ID
# (default 0) for which card to use.
N_GPUS=${N_GPUS:-}
GPU_ID=${GPU_ID:-0}             # only used when N_GPUS == 1

FORCE=${FORCE:-0}               # set FORCE=1 to re-run already-finished CSVs
# ================================

mkdir -p "$OUTPUT_DIR"

# Auto-detect N_GPUS if not set.
if [[ -z "$N_GPUS" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        N_GPUS=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
    elif command -v nvidia-smi >/dev/null 2>&1; then
        N_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    else
        N_GPUS=1
    fi
    [[ "$N_GPUS" -lt 1 ]] && N_GPUS=1
fi
echo "[config] N_GPUS=$N_GPUS  ckpt=$FLOW_CKPT"

# bash version guard (only required for parallel mode)
if (( N_GPUS > 1 )); then
    if (( BASH_VERSINFO[0] < 5 )) || { (( BASH_VERSINFO[0] == 5 )) && (( BASH_VERSINFO[1] < 1 )); }; then
        echo "ERROR: N_GPUS>1 needs bash >= 5.1 (wait -n -p). You have $BASH_VERSION" >&2
        exit 1
    fi
fi

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

# Helper: extract + validate SMILES from a CSV.
# Writes TWO files:
#   <out>.smi          — one SMILES per line (input to sample_conformers.py)
#   <out>.filtered.csv — subset of original CSV rows that survived validation,
#                        SAME ORDER as the .smi. Downstream prepare_downstream_K_pt
#                        joins targets from this filtered CSV (positions still
#                        match the K-conformer pickle ordering).
# Drops rows with: empty / NaN SMILES, RDKit parse failure, radical electrons,
# elements outside LoQI's 17-element atom encoder, disconnected fragments.
# Mirrors sample_conformers.py's validate_smiles checks so the sampler
# doesn't drop additional rows downstream of this filter.
extract_smiles() {
    local csv="$1"
    local smi_out="$2"
    local csv_out="${smi_out%.smi}.filtered.csv"
    python3 - "$csv" "$smi_out" "$csv_out" <<'PY'
import sys, pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

# LoQI's atom encoder (matches src/megalodon/metrics/molecule_evaluation_callback.py)
SUPPORTED = {
    "H","B","C","N","O","F","Al","Si","P","S","Cl",
    "As","Br","I","Hg","Bi","Se",
}

csv_path, smi_path, csv_filtered_path = sys.argv[1], sys.argv[2], sys.argv[3]
df = pd.read_csv(csv_path)
# case-insensitive smiles column
matches = [c for c in df.columns if c.lower() == "smiles"]
if not matches:
    raise SystemExit(f"No SMILES column in {csv_path}.  Columns: {list(df.columns)}")
col = matches[0]

n_raw = len(df)
kept_rows = []      # list of original CSV row indices that passed
kept_smis = []      # parallel list of SMILES strings
n_empty = 0
n_unparseable = 0
n_radical = 0
n_disconnected = 0
bad_elements = {}

for i, smi_raw in enumerate(df[col].astype(str)):
    smi = smi_raw.strip()
    if not smi or smi.lower() in ("nan", "none"):
        n_empty += 1
        continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        n_unparseable += 1
        continue
    if any(a.GetNumRadicalElectrons() > 0 for a in mol.GetAtoms()):
        n_radical += 1
        continue
    if "." in Chem.MolToSmiles(mol):
        n_disconnected += 1
        continue
    elems = {a.GetSymbol() for a in mol.GetAtoms()}
    bad = elems - SUPPORTED
    if bad:
        for e in bad:
            bad_elements[e] = bad_elements.get(e, 0) + 1
        continue
    kept_rows.append(i)
    kept_smis.append(smi)

# Write .smi (one per line)
with open(smi_path, "w") as f:
    for s in kept_smis:
        f.write(s + "\n")

# Write filtered.csv (subset of original CSV in the same order as .smi)
df.iloc[kept_rows].to_csv(csv_filtered_path, index=False)

n_dropped = n_raw - len(kept_smis)
print(f"  {col!r}: {n_raw:,} raw → {len(kept_smis):,} kept  ({n_dropped} dropped)")
if n_empty:        print(f"    empty/NaN:        {n_empty}")
if n_unparseable:  print(f"    unparseable:      {n_unparseable}")
if n_radical:      print(f"    radical:          {n_radical}")
if n_disconnected: print(f"    disconnected:     {n_disconnected}")
if bad_elements:
    pretty = ", ".join(f"{e}:{n}" for e, n in
                       sorted(bad_elements.items(), key=lambda x: -x[1]))
    print(f"    bad elements:     {pretty}")
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

# Process one CSV end-to-end on the given GPU (extract SMILES, sample,
# write pkl). Echoes a one-line status to stdout when done so the dispatch
# loop has something to grep. Returns 0 on success, non-zero on failure.
_process_one() {
    local csv="$1" gpu="$2"
    local rel_path="${csv#$INPUT_DIR/}"
    local rel_dir name out_dir smi pkl log
    rel_dir=$(dirname "$rel_path")
    name=$(basename "$rel_path" .csv)
    out_dir="$OUTPUT_DIR"
    [[ "$rel_dir" != "." ]] && out_dir="$OUTPUT_DIR/$rel_dir"
    mkdir -p "$out_dir"
    smi="$out_dir/$name.smi"
    pkl="$out_dir/$name.pkl"
    log="$out_dir/$name.log"

    if [[ ! -f "$smi" || "$FORCE" == "1" ]]; then
        extract_smiles "$csv" "$smi" >> "$log" 2>&1 \
            || { echo "[FAIL extract] $rel_path  log=$log"; return 1; }
    fi

    CUDA_VISIBLE_DEVICES=$gpu python scripts/sample_conformers.py \
        --ckpt   "$FLOW_CKPT" \
        --config "$FLOW_CONFIG" \
        --input  "$smi" \
        --output "$pkl" \
        --n_confs "$K" \
        --n_steps "$N_STEPS" \
        --batch_size "$BATCH_SIZE" \
        --postprocess "$POSTPROCESS" \
        --opt_max_nstep "$OPT_MAX_NSTEP" \
        >> "$log" 2>&1 \
        || { echo "[FAIL sample] $rel_path  log=$log"; return 1; }

    echo "[OK] $rel_path  pkl=$pkl"
    return 0
}

# Filter out already-completed CSVs so the worker-pool only schedules real work.
pending=()
n_skipped=0
for csv in "${csvs[@]}"; do
    rel_path="${csv#$INPUT_DIR/}"
    rel_dir=$(dirname "$rel_path")
    name=$(basename "$rel_path" .csv)
    out_dir="$OUTPUT_DIR"
    [[ "$rel_dir" != "." ]] && out_dir="$OUTPUT_DIR/$rel_dir"
    pkl="$out_dir/$name.pkl"
    if [[ -f "$pkl" && "$FORCE" != "1" ]]; then
        echo "  [skip] $rel_path  ($pkl exists)"
        n_skipped=$((n_skipped + 1))
        continue
    fi
    pending+=("$csv")
done
echo "[plan] pending=${#pending[@]}  skipped=$n_skipped"
if [[ ${#pending[@]} -eq 0 ]]; then
    echo "Nothing to do."
    exit 0
fi

start=$(date +%s)
n_done=0
n_failed=0

# --- Sequential single-GPU mode ---------------------------------------
if (( N_GPUS == 1 )); then
    for csv in "${pending[@]}"; do
        echo
        echo "============================================================"
        echo "  ${csv#$INPUT_DIR/}"
        echo "============================================================"
        if _process_one "$csv" "$GPU_ID"; then
            n_done=$((n_done + 1))
        else
            n_failed=$((n_failed + 1))
        fi
    done
else
# --- Multi-GPU worker pool --------------------------------------------
    declare -A GPU_OF_PID
    declare -A NAME_OF_PID
    idx=0
    n_started=0
    n_total=${#pending[@]}

    # Seed the pool: one CSV per GPU, up to whichever's smaller.
    for ((gpu=0; gpu<N_GPUS && idx<n_total; gpu++)); do
        csv="${pending[$idx]}"
        ( _process_one "$csv" "$gpu" ) &
        pid=$!
        GPU_OF_PID[$pid]=$gpu
        NAME_OF_PID[$pid]="${csv#$INPUT_DIR/}"
        n_started=$((n_started + 1))
        idx=$((idx + 1))
        echo "[$(date +%T)] [${n_started}/${n_total}] launch ${NAME_OF_PID[$pid]} on GPU $gpu  (pid=$pid)"
    done

    # Drain. As each child exits, dispatch the next pending CSV onto its GPU.
    finished_pid=0
    while (( ${#GPU_OF_PID[@]} > 0 )); do
        wait -n -p finished_pid
        status=$?
        if [[ -z "${GPU_OF_PID[$finished_pid]:-}" ]]; then
            continue
        fi
        gpu="${GPU_OF_PID[$finished_pid]}"
        name="${NAME_OF_PID[$finished_pid]}"
        unset 'GPU_OF_PID[$finished_pid]'
        unset 'NAME_OF_PID[$finished_pid]'
        if (( status == 0 )); then
            n_done=$((n_done + 1))
            echo "[$(date +%T)] done  $name  (gpu=$gpu)"
        else
            n_failed=$((n_failed + 1))
            echo "[$(date +%T)] FAIL  $name  (gpu=$gpu status=$status)"
        fi
        if (( idx < n_total )); then
            csv="${pending[$idx]}"
            ( _process_one "$csv" "$gpu" ) &
            pid=$!
            GPU_OF_PID[$pid]=$gpu
            NAME_OF_PID[$pid]="${csv#$INPUT_DIR/}"
            n_started=$((n_started + 1))
            idx=$((idx + 1))
            echo "[$(date +%T)] [${n_started}/${n_total}] launch ${NAME_OF_PID[$pid]} on GPU $gpu  (pid=$pid)"
        fi
    done
fi

echo
echo "============================================================"
echo "  Summary  (wall: $(( $(date +%s) - start ))s)"
echo "============================================================"
echo "  done:    $n_done"
echo "  skipped: $n_skipped"
echo "  failed:  $n_failed"
echo "  outputs: $OUTPUT_DIR/<name>.pkl"
