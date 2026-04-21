"""Parse TCIT thermochemistry log into a per-SMILES CSV of thermo labels.

TCIT emits five records per successful molecule:
    Hf_0    for <SMILES> is <value> kJ/mol
    Hf_298  for <SMILES> is <value> kJ/mol
    Gf_298  for <SMILES> is <value> kJ/mol
    S0_gas  for <SMILES> is <value> J/(mol*K)
    Cv_gas  for <SMILES> is <value> J/(mol*K)
All other text (TCIT banner, status, ring correction, skip notices) is ignored.

Parallel TCIT workers occasionally write two records into one line without a
newline between them, e.g.
    ...20.824 kJ/molHf_0 for CC(=O)C(Cl)Cl is -224.006 kJ/mol
The parser scans every line with a regex that captures ALL record matches, so
glued boundaries yield both records instead of being dropped.

SMILES that end up missing any of the 5 properties (TCIT crashed mid-mol,
or a glued-line tail couldn't be reassembled) are dropped from the CSV —
downstream code assumes complete records.

Example:
    python data_processing/parse_tcit_log.py \\
        --input  data_processing/batch9.log \\
        --output data_processing/tcit_thermo_labels.csv
"""
import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

PROPS = ("Hf_0", "Hf_298", "Gf_298", "S0_gas", "Cv_gas")
COLUMNS = {
    "Hf_0": "Hf_0_kJmol",
    "Hf_298": "Hf_298_kJmol",
    "Gf_298": "Gf_298_kJmol",
    "S0_gas": "S0_gas_JmolK",
    "Cv_gas": "Cv_gas_JmolK",
}
EXPECTED_UNIT = {
    "Hf_0":   "kJ/mol",
    "Hf_298": "kJ/mol",
    "Gf_298": "kJ/mol",
    "S0_gas": "J/(mol*K)",
    "Cv_gas": "J/(mol*K)",
}

# One regex handles clean lines AND glued lines.
# - `\S+?` (non-greedy) for SMILES so the shortest match wins at the first
#   ` is ` (SMILES never contain spaces, so this always aligns correctly).
# - Unit captured explicitly; we reject records where the unit doesn't match
#   the property (TCIT never mixes them; a mismatch means a corrupt tail).
RE_RECORD = re.compile(
    r"(Hf_0|Hf_298|Gf_298|S0_gas|Cv_gas)"     # prop
    r" for (\S+?)"                             # SMILES (non-greedy)
    r" is "
    r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"      # value
    r" (kJ/mol|J/\(mol\*K\))"                  # unit
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stats", default=None)
    p.add_argument("--progress-every", type=int, default=2_000_000)
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    stats_path = Path(args.stats) if args.stats else out.with_suffix(out.suffix + ".stats.json")

    labels = defaultdict(dict)
    per_prop_count = defaultdict(int)
    duplicate_writes = 0
    unit_mismatches = 0
    glued_recoveries = 0         # records found past the first one in a line

    t0 = time.time()
    n_lines = 0
    n_records = 0
    print(f"Reading {inp} ...", file=sys.stderr)
    with open(inp, "r", errors="replace") as f:
        for line in f:
            n_lines += 1
            if args.progress_every and n_lines % args.progress_every == 0:
                rate = n_lines / max(time.time() - t0, 1e-6)
                print(f"  {n_lines:>10,} lines  {n_records:>12,} records  "
                      f"({rate/1e6:.2f} M lines/s)", file=sys.stderr)
            matches = RE_RECORD.findall(line)
            if not matches:
                continue
            for mi, (prop, smi, val_str, unit) in enumerate(matches):
                if unit != EXPECTED_UNIT[prop]:
                    unit_mismatches += 1
                    continue
                try:
                    value = float(val_str)
                except ValueError:
                    continue
                if prop in labels[smi]:
                    duplicate_writes += 1
                labels[smi][prop] = value
                per_prop_count[prop] += 1
                n_records += 1
                if mi > 0:
                    glued_recoveries += 1

    # Only emit SMILES with all 5 properties present. Partial rows are
    # dropped — downstream build_property_table.py expects fully-labeled
    # records, and keeping stragglers would silently dilute thermo loss
    # with NaNs.
    n_complete = 0
    n_partial_dropped = 0
    header = ["smiles"] + [COLUMNS[p] for p in PROPS]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for smi, vals in labels.items():
            if not all(p in vals for p in PROPS):
                n_partial_dropped += 1
                continue
            w.writerow([smi] + [vals[p] for p in PROPS])
            n_complete += 1

    stats = {
        "input": str(inp),
        "output": str(out),
        "wall_seconds": round(time.time() - t0, 2),
        "total_lines": n_lines,
        "total_records": n_records,
        "unique_smiles_seen": len(labels),
        "n_complete_records_written": n_complete,
        "n_partial_records_dropped": n_partial_dropped,
        "per_property_count": dict(per_prop_count),
        "duplicate_writes": duplicate_writes,
        "unit_mismatches_skipped": unit_mismatches,
        "glued_records_recovered": glued_recoveries,
        "columns": header,
        "units": {
            "Hf_0_kJmol": "kJ/mol",
            "Hf_298_kJmol": "kJ/mol",
            "Gf_298_kJmol": "kJ/mol",
            "S0_gas_JmolK": "J/(mol*K)",
            "Cv_gas_JmolK": "J/(mol*K)",
        },
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
