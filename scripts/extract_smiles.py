"""Extract a clean SMILES list from a downstream CSV.

Mirrors sample_conformers.py's validate_smiles checks plus chembl3d's
distributional constraints, so the output is safe to feed into
sample_conformers.py without per-mol failures derailing the batch:

    * empty / NaN / non-string SMILES        → drop
    * RDKit unparseable                      → drop
    * radicals (any unpaired electron)       → drop  (see note below)
    * disconnected (multi-fragment, "." in)  → drop
    * elements outside LoQI's 17-atom set    → drop
    * |formal_charge| > 1                    → drop  (chembl3d range)
    * canonical SMILES already seen          → drop  (default; --no-dedup)

Why radicals are dropped (and not just "be safe"):
  Our atom encoder (`ATOMIC_TO_INNER` in prepare_downstream_dataset.py)
  encodes only element + formal charge — NOT the unpaired-electron count.
  So a closed-shell methanol `CO` and the open-shell hydroxymethyl
  radical `[CH2]O` would be indistinguishable to the backbone, even
  though their Hf differs by ~190 kJ/mol. Keeping radicals means
  silently mispredicting them by tens-to-hundreds of kJ/mol — the loss
  of ~2-3% of data (mostly small species like ·OH, ·CN, ·CHO in
  gas_Hf) is the cheaper option until the encoder is upgraded to a
  (z, q, n_rad) triplet and the backbone re-pretrained. Our
  closed-shell-only restriction matches what UniMol / SchNet / MACE
  also typically do; document it as a benchmark limitation, not as
  a noise source.

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

To use as a library (clean_downstream.py does this):
    from extract_smiles import filter_and_dedup, find_smiles_column
    cleaned_df, stats = filter_and_dedup(df, smi_col, dedup_canonical=True)
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


def find_smiles_column(df: pd.DataFrame) -> str:
    matches = [c for c in df.columns if c.lower() == "smiles"]
    if not matches:
        raise SystemExit(f"No SMILES column. Columns: {list(df.columns)}")
    return matches[0]


def filter_and_dedup(df: pd.DataFrame, smi_col: str,
                     dedup_canonical: bool = True) -> tuple[pd.DataFrame, dict]:
    """Apply the 6 physical filters + optional canonical dedup, return
    (cleaned_df, stats). `cleaned_df` preserves original column ordering and
    row order (drops only). Stats counts what got removed at each step."""
    n_raw = len(df)
    kept_rows: list[int] = []
    kept_canons: list[str] = []
    n_empty = n_unparseable = n_radical = n_disconnected = n_bad_charge = 0
    bad_elements: dict[str, int] = {}

    for i, smi_raw in enumerate(df[smi_col].astype(str)):
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
        canon = Chem.MolToSmiles(mol, isomericSmiles=True)
        if "." in canon:
            n_disconnected += 1
            continue
        bad = {a.GetSymbol() for a in mol.GetAtoms()} - SUPPORTED
        if bad:
            for e in bad:
                bad_elements[e] = bad_elements.get(e, 0) + 1
            continue
        if any(abs(a.GetFormalCharge()) > 1 for a in mol.GetAtoms()):
            n_bad_charge += 1
            continue
        kept_rows.append(i)
        kept_canons.append(canon)

    n_after_phys = len(kept_rows)

    # Canonical dedup — keep first occurrence (preserves earliest row).
    n_canonical_dup = 0
    if dedup_canonical:
        seen: set = set()
        deduped: list[int] = []
        for ri, c in zip(kept_rows, kept_canons):
            if c not in seen:
                seen.add(c)
                deduped.append(ri)
            else:
                n_canonical_dup += 1
        kept_rows = deduped

    cleaned = df.iloc[kept_rows].copy().reset_index(drop=True)
    stats = {
        "smi_col":         smi_col,
        "n_raw":           n_raw,
        "n_empty":         n_empty,
        "n_unparseable":   n_unparseable,
        "n_radical":       n_radical,
        "n_disconnected":  n_disconnected,
        "n_bad_elements":  sum(bad_elements.values()),
        "bad_elements_breakdown": bad_elements,
        "n_bad_charge":    n_bad_charge,
        "n_after_phys":    n_after_phys,
        "n_canonical_dup": n_canonical_dup,
        "n_final":         len(cleaned),
    }
    return cleaned, stats


def print_stats(stats: dict) -> None:
    s = stats
    print(f"  {s['smi_col']!r}: {s['n_raw']:,} raw "
          f"→ {s['n_final']:,} kept  "
          f"({s['n_raw'] - s['n_final']} dropped)")
    if s["n_empty"]:        print(f"    empty/NaN:        {s['n_empty']}")
    if s["n_unparseable"]:  print(f"    unparseable:      {s['n_unparseable']}")
    if s["n_radical"]:      print(f"    radical:          {s['n_radical']}")
    if s["n_disconnected"]: print(f"    disconnected:     {s['n_disconnected']}")
    if s["n_bad_charge"]:   print(f"    |charge|>1:       {s['n_bad_charge']}")
    if s["bad_elements_breakdown"]:
        pretty = ", ".join(f"{e}:{n}" for e, n in
                            sorted(s["bad_elements_breakdown"].items(),
                                   key=lambda x: -x[1]))
        print(f"    bad elements:     {pretty}")
    if s["n_canonical_dup"]:
        print(f"    canonical dup:    {s['n_canonical_dup']}")


def extract(csv_path: Path, smi_path: Path, csv_filtered_path: Path,
            dedup_canonical: bool = True) -> dict:
    """CLI entry: read csv, clean, write .smi + .filtered.csv. Returns stats."""
    df = pd.read_csv(csv_path)
    smi_col = find_smiles_column(df)
    cleaned, stats = filter_and_dedup(df, smi_col, dedup_canonical=dedup_canonical)

    smi_path.parent.mkdir(parents=True, exist_ok=True)
    smis = cleaned[smi_col].astype(str).str.strip().tolist()
    with open(smi_path, "w") as f:
        f.write("\n".join(smis) + ("\n" if smis else ""))
    cleaned.to_csv(csv_filtered_path, index=False)

    print_stats(stats)
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Input CSV")
    p.add_argument("--out", required=True,
                   help="Output .smi path (the .filtered.csv is auto-derived "
                        "by replacing the .smi suffix)")
    p.add_argument("--no-dedup", action="store_true",
                   help="Skip canonical-SMILES deduplication "
                        "(default is to dedup; keeps first occurrence).")
    args = p.parse_args()
    smi_path = Path(args.out)
    if smi_path.suffix != ".smi":
        raise SystemExit("--out must end in .smi")
    csv_filtered_path = smi_path.with_suffix("").with_suffix(".filtered.csv")
    extract(Path(args.csv), smi_path, csv_filtered_path,
            dedup_canonical=not args.no_dedup)


if __name__ == "__main__":
    main()
