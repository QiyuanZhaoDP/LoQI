"""Process downstream_ft/0506/ → downstream_ft/0506/clean/

Per-dataset steps:
  1. Unify columns to (SMILES, TARGET):
       - Strip UTF-8 BOM (﻿) from headers
       - AOH: TARGET = log10(OH)  [rate constant → log scale]
       - solid_Hf: rename `value` → TARGET
       - density: drop extra `compound_name` col
       - mnsol: drop Solvent / SoluteName / Ben-Naim cols; keep SMILES + TARGET
       - Others: pass-through
  2. Remove SMILES containing '.' (multi-fragment / ionic compounds)
  3. RDKit validation: empty, unparseable, radical, bad-element, |charge|>1
  4. InChIKey-based deduplication:
       - Compute InChIKey for each valid mol (robust cross-tautomer dedup)
       - For each InChIKey group:
           * 1 row          → keep as-is
           * >1 rows, spread ≤ outlier_threshold_rel × dataset_std
                            → aggregate (mean/median/first)
           * >1 rows, spread > threshold → drop entire group
             (measurement-condition mismatch, e.g., mnsol multi-solvent)

Note on mnsol: 106 solvents × solute → same InChIKey will have wildly
different solvation FEs across solvents. With default outlier_rel=0.3,
most multi-solvent entries will be dropped as outliers. This effectively
reduces mnsol to solutes whose solvation FE is similar across many
solvents — a conservative but robust choice. Raise --outlier-rel to keep
more entries, or pre-filter by solvent before running this script.

Usage:
    python scripts/process_0506.py
    python scripts/process_0506.py --outlier-rel 0.3 --aggregation mean
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import MolToInchiKey

RDLogger.DisableLog("rdApp.*")

SUPPORTED = {
    "H", "B", "C", "N", "O", "F", "Al", "Si", "P", "S", "Cl",
    "As", "Br", "I", "Hg", "Bi", "Se",
}


# ---- Per-dataset column-normalization rules --------------------------------
def normalize_columns(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Rename / transform columns → (SMILES, TARGET), drop extras."""
    # strip BOM from all column names
    df.columns = [c.lstrip("﻿").strip() for c in df.columns]

    if name == "AOH":
        df = df.rename(columns={"OH": "TARGET"})
        df["TARGET"] = np.log10(df["TARGET"].astype(float))
        return df[["SMILES", "TARGET"]]

    if name == "solid_Hf":
        df = df.rename(columns={"value": "TARGET"})
        return df[["SMILES", "TARGET"]]

    if name == "density":
        return df[["SMILES", "TARGET"]]

    if name == "mnsol":
        return df[["SMILES", "TARGET"]]

    # Generic: auto-detect SMILES + TARGET
    smi_col = next((c for c in df.columns if c.lower() == "smiles"), None)
    tgt_col = next((c for c in df.columns if c.lower() == "target"), None)
    if smi_col is None:
        raise ValueError(f"[{name}] no SMILES column in {list(df.columns)}")
    if tgt_col is None:
        raise ValueError(f"[{name}] no TARGET column in {list(df.columns)}")
    out = df[[smi_col, tgt_col]].copy()
    out.columns = ["SMILES", "TARGET"]
    return out


# ---- Physical filters + InChIKey dedup ------------------------------------
def process(df: pd.DataFrame, name: str,
            aggregation: str, outlier_rel: float,
            outlier_abs: float | None) -> tuple[pd.DataFrame, dict]:
    n_raw = len(df)
    stats = {
        "name": name, "n_raw": n_raw,
        "n_dot_smi": 0, "n_empty": 0, "n_unparseable": 0,
        "n_radical": 0, "n_disconnected": 0, "n_bad_elements": 0,
        "n_bad_charge": 0, "bad_elements": {},
        "n_after_phys": 0,
        "n_groups_aggregated": 0,
        "n_outlier_groups": 0, "n_outlier_rows": 0,
        "n_final": 0,
    }

    kept_idx: list[int] = []
    kept_inchikeys: list[str] = []
    kept_canons: list[str] = []

    for i, (smi_raw, tgt_raw) in enumerate(
            zip(df["SMILES"].astype(str), df["TARGET"])):
        smi = smi_raw.strip()
        if not smi or smi.lower() in ("nan", "none"):
            stats["n_empty"] += 1
            continue
        # Step 2: remove '.' SMILES
        if "." in smi:
            stats["n_dot_smi"] += 1
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            stats["n_unparseable"] += 1
            continue
        if any(a.GetNumRadicalElectrons() > 0 for a in mol.GetAtoms()):
            stats["n_radical"] += 1
            continue
        canon = Chem.MolToSmiles(mol, isomericSmiles=True)
        if "." in canon:
            stats["n_disconnected"] += 1
            continue
        bad = {a.GetSymbol() for a in mol.GetAtoms()} - SUPPORTED
        if bad:
            for e in bad:
                stats["bad_elements"][e] = stats["bad_elements"].get(e, 0) + 1
            stats["n_bad_elements"] += 1
            continue
        if any(abs(a.GetFormalCharge()) > 1 for a in mol.GetAtoms()):
            stats["n_bad_charge"] += 1
            continue
        # Compute InChIKey (more robust than canonical SMILES for dedup)
        ik = MolToInchiKey(mol)
        if ik is None:
            stats["n_unparseable"] += 1
            continue
        kept_idx.append(i)
        kept_inchikeys.append(ik)
        kept_canons.append(canon)

    stats["n_after_phys"] = len(kept_idx)
    if not kept_idx:
        return pd.DataFrame(columns=["SMILES", "TARGET"]), stats

    valid_df = df.iloc[kept_idx][["SMILES", "TARGET"]].copy().reset_index(drop=True)
    valid_df["__inchikey"] = kept_inchikeys
    valid_df["TARGET"] = pd.to_numeric(valid_df["TARGET"], errors="coerce")
    valid_df = valid_df.dropna(subset=["TARGET"])

    target_std = float(valid_df["TARGET"].std()) if outlier_rel else None

    kept: list = []
    for ik, group in valid_df.groupby("__inchikey", sort=False):
        if len(group) == 1:
            kept.append(group.iloc[0])
            continue

        targets = group["TARGET"].values.astype(float)
        spread = float(targets.max() - targets.min())
        is_outlier = (
            (outlier_rel is not None and target_std is not None
             and target_std > 0 and spread > outlier_rel * target_std)
            or (outlier_abs is not None and spread > outlier_abs)
        )
        if is_outlier:
            stats["n_outlier_groups"] += 1
            stats["n_outlier_rows"] += len(group)
            continue

        if aggregation == "mean":
            agg = float(np.mean(targets))
        elif aggregation == "median":
            agg = float(np.median(targets))
        else:
            agg = float(group["TARGET"].iloc[0])
        row = group.iloc[0].copy()
        row["TARGET"] = agg
        kept.append(row)
        stats["n_groups_aggregated"] += 1

    if not kept:
        return pd.DataFrame(columns=["SMILES", "TARGET"]), stats

    out = pd.DataFrame(kept).reset_index(drop=True)
    # Use canonical SMILES (consistent representation) as output SMILES
    idx_map = {ik: canon for ik, canon in zip(kept_inchikeys, kept_canons)}
    out["SMILES"] = out["__inchikey"].map(idx_map).fillna(out["SMILES"])
    out = out[["SMILES", "TARGET"]].copy()
    stats["n_bad_elements"] = sum(stats["bad_elements"].values())
    stats["n_final"] = len(out)
    return out, stats


# ---- Main ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="downstream_ft/0506")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--aggregation", choices=["mean", "median", "first"],
                    default="mean")
    ap.add_argument("--outlier-rel", type=float, default=0.3,
                    help="Drop InChIKey group if spread > this × dataset_std")
    ap.add_argument("--outlier-abs", type=float, default=None)
    args = ap.parse_args()

    root = Path(args.input_dir)
    out_dir = Path(args.output_dir) if args.output_dir else (root / "clean")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats: list[dict] = []
    print("=" * 70)
    print(f"Processing {root}/ → {out_dir}/")
    print(f"  aggregation={args.aggregation}  outlier_rel={args.outlier_rel}")
    print("=" * 70)

    for csv in sorted(root.glob("*.csv")):
        name = csv.stem
        df_raw = pd.read_csv(csv, encoding="utf-8-sig")

        try:
            df_norm = normalize_columns(name, df_raw)
        except ValueError as e:
            print(f"\n[{name}] SKIP: {e}")
            continue

        cleaned, stats = process(
            df_norm, name,
            aggregation=args.aggregation,
            outlier_rel=args.outlier_rel,
            outlier_abs=args.outlier_abs,
        )
        out_csv = out_dir / f"{name}.csv"
        cleaned.to_csv(out_csv, index=False)
        all_stats.append(stats)

        print(f"\n[{name}]")
        print(f"  raw: {stats['n_raw']:,}  →  final: {stats['n_final']:,}")
        drops = []
        if stats["n_dot_smi"]:        drops.append(f"'.' SMILES: {stats['n_dot_smi']}")
        if stats["n_empty"]:          drops.append(f"empty: {stats['n_empty']}")
        if stats["n_unparseable"]:    drops.append(f"unparse: {stats['n_unparseable']}")
        if stats["n_radical"]:        drops.append(f"radical: {stats['n_radical']}")
        if stats["n_disconnected"]:   drops.append(f"disconnect: {stats['n_disconnected']}")
        if stats["n_bad_elements"]:
            be = ", ".join(f"{e}:{n}" for e, n in
                           sorted(stats["bad_elements"].items(), key=lambda x: -x[1]))
            drops.append(f"bad-elem: {stats['n_bad_elements']} ({be})")
        if stats["n_bad_charge"]:     drops.append(f"|charge|>1: {stats['n_bad_charge']}")
        if drops:
            print(f"  dropped: " + "; ".join(drops))
        if stats["n_groups_aggregated"]:
            print(f"  InChIKey-aggregated: {stats['n_groups_aggregated']} groups → 1 row each")
        if stats["n_outlier_groups"]:
            print(f"  InChIKey-outlier dropped: {stats['n_outlier_groups']} groups ({stats['n_outlier_rows']} rows, spread > {args.outlier_rel}×σ)")
        print(f"  written: {out_csv}")

    # ---- Summary table -----
    print("\n" + "=" * 80)
    print(f"{'dataset':<14s}  {'raw':>6s}  {'dot':>5s}  {'phys':>5s}  "
          f"{'dup_agg':>8s}  {'dup_drop':>9s}  {'final':>6s}")
    print("-" * 80)
    for s in all_stats:
        phys = (s["n_empty"] + s["n_unparseable"] + s["n_radical"]
                + s["n_disconnected"] + s["n_bad_elements"] + s["n_bad_charge"])
        print(f"{s['name']:<14s}  {s['n_raw']:>6,}  {s['n_dot_smi']:>5,}  "
              f"{phys:>5,}  {s['n_groups_aggregated']:>8,}  "
              f"{s['n_outlier_groups']:>9,}  {s['n_final']:>6,}")
    print("=" * 80)

    (out_dir / "processing_report.json").write_text(
        json.dumps({
            "generated": datetime.now().isoformat(timespec="seconds"),
            "input": str(root),
            "aggregation": args.aggregation,
            "outlier_rel": args.outlier_rel,
            "datasets": all_stats,
        }, indent=2)
    )
    print(f"Report → {out_dir}/processing_report.json")


if __name__ == "__main__":
    main()
