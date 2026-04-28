"""Audit each downstream CSV for duplicate SMILES (string-level + canonical-
level) and the same SMILES filters extract_smiles.py applies, with drop
counts at every step. Reads downstream_ft/ tree:

  downstream_ft/<flat>.csv             — single CSV
  downstream_ft/<presplit>/{train,valid,test}.csv  — three-way split, merged
                                                     into one before audit

Validation steps (mirrors scripts/extract_smiles.py):
    1. empty / NaN                          → drop
    2. RDKit unparseable                    → drop
    3. radicals (any unpaired electrons)    → drop
    4. disconnected ("." in canonical)      → drop
    5. elements outside LoQI's 17-atom set  → drop
    6. |formal_charge| > 1                  → drop

Plus deduplication:
    7a. exact-string duplicate              → counted (not auto-dropped)
    7b. canonical duplicate (different      → counted (not auto-dropped)
         input SMILES, same molecule)
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

SUPPORTED = {
    "H", "B", "C", "N", "O", "F", "Al", "Si", "P", "S", "Cl",
    "As", "Br", "I", "Hg", "Bi", "Se",
}

PRESPLIT = {"delaney_s", "freesolv_s", "lipo_s"}


def _load_dataset(name: str, root: Path) -> pd.DataFrame:
    """Return one DataFrame per dataset, merging splits when needed."""
    if name in PRESPLIT:
        parts = []
        for split in ("train", "valid", "test"):
            p = root / name / f"{split}.csv"
            if not p.exists():
                raise SystemExit(f"missing split: {p}")
            d = pd.read_csv(p)
            d["_split"] = split
            parts.append(d)
        return pd.concat(parts, ignore_index=True)
    p = root / f"{name}.csv"
    return pd.read_csv(p)


def _smi_col(df: pd.DataFrame) -> str:
    matches = [c for c in df.columns if c.lower() == "smiles"]
    if not matches:
        raise SystemExit(f"no SMILES column in {list(df.columns)}")
    return matches[0]


def audit(name: str, df: pd.DataFrame) -> dict:
    col = _smi_col(df)
    n_raw = len(df)

    # Step 1: empty / NaN
    smi_series = df[col].astype(str).str.strip()
    keep_nonempty = ~(smi_series.str.lower().isin({"", "nan", "none"}))
    n_after_empty = int(keep_nonempty.sum())
    n_empty = n_raw - n_after_empty
    smis = smi_series[keep_nonempty].tolist()

    # Step 2-6: parse + filter
    canon_list = []           # canonical SMILES of survivors
    raw_list = []             # original SMILES strings of survivors
    n_unparseable = n_radical = n_disconnected = n_bad_charge = 0
    bad_elem_counter: Counter = Counter()
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None:
            n_unparseable += 1
            continue
        if any(a.GetNumRadicalElectrons() > 0 for a in m.GetAtoms()):
            n_radical += 1
            continue
        canon = Chem.MolToSmiles(m, isomericSmiles=True)
        if "." in canon:
            n_disconnected += 1
            continue
        bad = {a.GetSymbol() for a in m.GetAtoms()} - SUPPORTED
        if bad:
            for e in bad:
                bad_elem_counter[e] += 1
            continue
        if any(abs(a.GetFormalCharge()) > 1 for a in m.GetAtoms()):
            n_bad_charge += 1
            continue
        canon_list.append(canon)
        raw_list.append(s)

    n_after_filter = len(canon_list)

    # Step 7a: exact-string duplicates (same input SMILES typed twice)
    raw_counter = Counter(raw_list)
    n_str_dup_extra = sum(c - 1 for c in raw_counter.values() if c > 1)
    n_str_unique = len(raw_counter)

    # Step 7b: canonical duplicates (different input SMILES → same mol).
    # We count "extras" beyond the first canonical occurrence.
    canon_counter = Counter(canon_list)
    n_canon_unique = len(canon_counter)
    n_canon_dup_extra = n_after_filter - n_canon_unique
    # canonical duplicates that aren't already string duplicates:
    n_canon_only_extra = n_canon_dup_extra - n_str_dup_extra

    # examples for inspection
    str_dup_examples = sorted(
        ((s, c) for s, c in raw_counter.items() if c > 1),
        key=lambda x: -x[1],
    )[:3]
    # Find canonical groups whose members include >=2 distinct raw strings.
    canon_to_raws: dict[str, set] = {}
    for r, c in zip(raw_list, canon_list):
        canon_to_raws.setdefault(c, set()).add(r)
    canon_only_examples = [
        (c, sorted(rs))
        for c, rs in canon_to_raws.items()
        if len(rs) > 1
    ][:3]

    return {
        "name": name,
        "n_raw":           n_raw,
        "n_empty":         n_empty,
        "n_unparseable":   n_unparseable,
        "n_radical":       n_radical,
        "n_disconnected":  n_disconnected,
        "n_bad_elements":  sum(bad_elem_counter.values()),
        "bad_elem_breakdown": dict(bad_elem_counter),
        "n_bad_charge":    n_bad_charge,
        "n_after_filter":  n_after_filter,
        "n_str_unique":    n_str_unique,
        "n_str_dup_extra": n_str_dup_extra,
        "n_canon_unique":  n_canon_unique,
        "n_canon_dup_extra":  n_canon_dup_extra,
        "n_canon_only_extra": n_canon_only_extra,
        "str_dup_examples":   str_dup_examples,
        "canon_only_examples": canon_only_examples,
    }


def _print_report(r: dict) -> None:
    n_raw = r["n_raw"]
    surv = r["n_after_filter"]
    print(f"\n=== {r['name']} ===")
    print(f"  raw rows:                       {n_raw:>7,}")
    print(f"   - empty / NaN:                 -{r['n_empty']:>6,}")
    print(f"   - RDKit unparseable:           -{r['n_unparseable']:>6,}")
    print(f"   - radical (any unpaired e-):   -{r['n_radical']:>6,}")
    print(f"   - disconnected ('.' present):  -{r['n_disconnected']:>6,}")
    if r["n_bad_elements"]:
        breakdown = ", ".join(f"{e}:{n}" for e, n in
                              sorted(r["bad_elem_breakdown"].items(),
                                     key=lambda x: -x[1]))
        print(f"   - elements ∉ LoQI-17:          -{r['n_bad_elements']:>6,}  ({breakdown})")
    else:
        print(f"   - elements ∉ LoQI-17:          -{r['n_bad_elements']:>6,}")
    print(f"   - |formal_charge| > 1:         -{r['n_bad_charge']:>6,}")
    print(f"  → after physical filters:       {surv:>7,}  "
          f"({100*surv/max(n_raw,1):.1f}%)")
    print()
    print(f"  unique input strings:           {r['n_str_unique']:>7,}  "
          f"(extra string-dup rows: {r['n_str_dup_extra']:,})")
    print(f"  unique canonical molecules:     {r['n_canon_unique']:>7,}  "
          f"(canonical dup rows: {r['n_canon_dup_extra']:,}; of which "
          f"different-string-but-same-mol: {r['n_canon_only_extra']:,})")
    if r["str_dup_examples"]:
        print(f"  string-dup top 3:")
        for s, c in r["str_dup_examples"]:
            print(f"     ×{c:<3d}  {s[:80]}")
    if r["canon_only_examples"]:
        print(f"  different-string→same-molecule top 3:")
        for canon, raws in r["canon_only_examples"]:
            print(f"     canonical: {canon[:70]}")
            for raw in list(raws)[:4]:
                print(f"       ←  {raw[:70]}")


def main():
    root = Path("downstream_ft")
    if not root.exists():
        sys.exit(f"missing {root}")

    flat = sorted(p.stem for p in root.glob("*.csv"))
    presplit = sorted(d.name for d in root.iterdir()
                      if d.is_dir() and d.name in PRESPLIT)
    datasets = flat + presplit

    summary_rows = []
    for name in datasets:
        df = _load_dataset(name, root)
        r = audit(name, df)
        _print_report(r)
        summary_rows.append(r)

    # ---- Aggregated table ------------------------------------------------
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'dataset':<14s}  {'raw':>6s}  {'kept':>6s}  "
          f"{'str_dup':>8s}  {'canon_dup':>10s}  {'σmol→1str':>11s}  "
          f"{'unique_mol':>11s}")
    print("-" * 100)
    for r in summary_rows:
        print(f"{r['name']:<14s}  "
              f"{r['n_raw']:>6,}  {r['n_after_filter']:>6,}  "
              f"{r['n_str_dup_extra']:>8,}  {r['n_canon_dup_extra']:>10,}  "
              f"{r['n_canon_only_extra']:>11,}  {r['n_canon_unique']:>11,}")
    print()
    print("Columns:")
    print("  raw          = total rows in the CSV (presplit: train+valid+test merged)")
    print("  kept         = rows that pass all 6 physical filters")
    print("  str_dup      = extra rows where the SAME SMILES string repeats")
    print("  canon_dup    = extra rows that map to an already-seen canonical SMILES")
    print("  σmol→1str    = canon_dup minus str_dup; i.e. different SMILES strings")
    print("                  that turn out to be the same molecule")
    print("  unique_mol   = distinct canonical SMILES after filters")


if __name__ == "__main__":
    main()
