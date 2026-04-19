"""Parse TCIT thermochemistry log into a per-SMILES CSV of thermo labels.

TCIT emits five lines per successful molecule:
    Hf_0    for <SMILES> is <value> kJ/mol
    Hf_298  for <SMILES> is <value> kJ/mol
    Gf_298  for <SMILES> is <value> kJ/mol
    S0_gas  for <SMILES> is <value> J/(mol*K)
    Cv_gas  for <SMILES> is <value> J/(mol*K)
All other lines (TCIT banner, status, ring correction, skip notices) are ignored.

Example:
    python data_processing/parse_tcit_log.py \\
        --input  data_processing/batch9.log \\
        --output data_processing/tcit_thermo_labels.csv
"""
import argparse
import csv
import json
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


def parse_line(line):
    """Return (prop, smiles, value) or None.

    Rejects interleaved/corrupt lines — parallel TCIT workers occasionally
    wrote two messages into one, producing bogus fragments like
    `Hf_298 for <smi> is 26.266 kJ/molHf_0 for <smi2> is -239.052 kJ/mol`.
    Valid SMILES contain no spaces, so any space in the parsed SMILES means
    corruption — discard.
    """
    for prop in PROPS:
        if line.startswith(prop) and len(line) > len(prop) and line[len(prop)] == " ":
            try:
                left, right = line.rsplit(" is ", 1)
                smi = left.split(" for ", 1)[1]
                if " " in smi:
                    return "CORRUPT", None, None
                value = float(right.split(" ", 1)[0])
                return prop, smi, value
            except (ValueError, IndexError):
                return None
    return None


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
    corrupt_lines = 0

    t0 = time.time()
    n_lines = 0
    print(f"Reading {inp} ...", file=sys.stderr)
    with open(inp, "r", errors="replace") as f:
        for line in f:
            n_lines += 1
            if args.progress_every and n_lines % args.progress_every == 0:
                rate = n_lines / max(time.time() - t0, 1e-6)
                print(f"  {n_lines:>10,} lines  ({rate/1e6:.2f} M/s)", file=sys.stderr)
            r = parse_line(line)
            if r is None:
                continue
            prop, smi, value = r
            if prop == "CORRUPT":
                corrupt_lines += 1
                continue
            if prop in labels[smi]:
                duplicate_writes += 1
            labels[smi][prop] = value
            per_prop_count[prop] += 1

    n_complete = 0
    n_partial = 0
    header = ["smiles"] + [COLUMNS[p] for p in PROPS]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for smi, vals in labels.items():
            row = [smi] + [vals.get(p, "") for p in PROPS]
            w.writerow(row)
            if all(p in vals for p in PROPS):
                n_complete += 1
            else:
                n_partial += 1

    stats = {
        "input": str(inp),
        "output": str(out),
        "wall_seconds": round(time.time() - t0, 2),
        "total_lines": n_lines,
        "unique_smiles": len(labels),
        "n_complete_records": n_complete,
        "n_partial_records": n_partial,
        "per_property_count": dict(per_prop_count),
        "duplicate_writes": duplicate_writes,
        "corrupt_lines_skipped": corrupt_lines,
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
