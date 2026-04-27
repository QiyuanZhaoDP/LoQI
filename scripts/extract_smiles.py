"""Extract a clean SMILES list from a downstream CSV.

Mirrors sample_conformers.py's validate_smiles checks plus chembl3d's
distributional constraints, so the output is safe to feed into
sample_conformers.py without per-mol failures derailing the batch:

    * empty / NaN / non-string SMILES        → drop
    * RDKit unparseable                      → drop
    * radicals (any unpaired electron)       → drop
    * disconnected (multi-fragment, "." in)  → drop
    * elements outside LoQI's 17-atom set    → drop
    * |formal_charge| > 1                    → drop  (chembl3d range)

Writes:
    <out>.smi          — one SMILES per line (input to sample_conformers.py)
    <out>.filtered.csv — original CSV rows that survived, SAME ORDER as
                         the .smi (downstream prepare_downstream_K_pt.py
                         joins by canonical SMILES, but having the row
                         order match the .smi keeps things sane).

Usage:
    python scripts/extract_smiles.py \\
        --csv downstream_ft/gas_Hf.csv \\
        --out data/downstream_k5/gas_Hf.smi
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# LoQI's atom encoder — must match
# src/megalodon/metrics/molecule_evaluation_callback.py.
SUPPORTED = {
    "H", "B", "C", "N", "O", "F", "Al", "Si", "P", "S", "Cl",
    "As", "Br", "I", "Hg", "Bi", "Se",
}


def extract(csv_path: Path, smi_path: Path, csv_filtered_path: Path) -> None:
    df = pd.read_csv(csv_path)
    matches = [c for c in df.columns if c.lower() == "smiles"]
    if not matches:
        raise SystemExit(
            f"No SMILES column in {csv_path}.  Columns: {list(df.columns)}"
        )
    col = matches[0]

    n_raw = len(df)
    kept_rows: list[int] = []
    kept_smis: list[str] = []
    n_empty = n_unparseable = n_radical = n_disconnected = n_bad_charge = 0
    bad_elements: dict[str, int] = {}

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
        if any(abs(a.GetFormalCharge()) > 1 for a in mol.GetAtoms()):
            n_bad_charge += 1
            continue
        kept_rows.append(i)
        kept_smis.append(smi)

    smi_path.parent.mkdir(parents=True, exist_ok=True)
    with open(smi_path, "w") as f:
        f.write("\n".join(kept_smis) + ("\n" if kept_smis else ""))
    df.iloc[kept_rows].to_csv(csv_filtered_path, index=False)

    n_dropped = n_raw - len(kept_smis)
    print(f"  {col!r}: {n_raw:,} raw → {len(kept_smis):,} kept  "
          f"({n_dropped} dropped)")
    if n_empty:        print(f"    empty/NaN:        {n_empty}")
    if n_unparseable:  print(f"    unparseable:      {n_unparseable}")
    if n_radical:      print(f"    radical:          {n_radical}")
    if n_disconnected: print(f"    disconnected:     {n_disconnected}")
    if n_bad_charge:   print(f"    |charge|>1:       {n_bad_charge}")
    if bad_elements:
        pretty = ", ".join(f"{e}:{n}" for e, n in
                            sorted(bad_elements.items(), key=lambda x: -x[1]))
        print(f"    bad elements:     {pretty}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Input CSV")
    p.add_argument("--out", required=True,
                   help="Output .smi path (the .filtered.csv is auto-derived "
                        "by replacing the .smi suffix)")
    args = p.parse_args()
    smi_path = Path(args.out)
    if smi_path.suffix != ".smi":
        raise SystemExit("--out must end in .smi")
    csv_filtered_path = smi_path.with_suffix("").with_suffix(".filtered.csv")
    extract(Path(args.csv), smi_path, csv_filtered_path)


if __name__ == "__main__":
    main()
