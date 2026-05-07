"""Comprehensive DB screening for downstream property-prediction datasets.

Steps applied per CSV:
  1. Column normalisation → (SMILES, TARGET)
  2. SMILES validation: empty/NaN, unparseable, radical, disconnected,
     OOD-element, |charge|>1
  3. Physically-impossible target removal (per-property hard limits)
  4. Single-entry statistical outlier removal (robust 6-MAD from median)
  5. InChIKey deduplication:
       same molecule → group by InChIKey
       |max-min| ≤ outlier_threshold_rel × σ_global   → keep mean
       |max-min| >  outlier_threshold_rel × σ_global  → remove, log
  6. (optional) Heavy-atom count cap for conformer-gen feasibility

Outputs (under <out_dir>/<name>/):
  cleaned.csv           — ready-for-pipeline molecules
  removed_inchikey.csv  — InChIKey, SMILES, all original values, reason
  stats.json            — per-step counts

Usage:
  python scripts/screen_db.py \\
      --input-dir downstream_ft/0506/cleaned_by_CC/cleaned_by_codex \\
      --output-dir downstream_ft/0506/screened
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.inchi import MolToInchiKey

RDLogger.DisableLog("rdApp.*")

SUPPORTED = {
    "H", "B", "C", "N", "O", "F", "Al", "Si", "P", "S", "Cl",
    "As", "Br", "I", "Hg", "Bi", "Se",
}

# Per-property hard limits (physical plausibility).
# Values outside these ranges are unconditionally removed regardless of
# whether they are duplicates. Add per-dataset overrides here.
HARD_LIMITS: dict[str, tuple[float | None, float | None]] = {
    # (min_allowed, max_allowed)  — None = no limit on that side
    "Cp":         (0.0,   None),     # heat capacity J/(mol·K) must be >0
    "Vcp":        (0.0,   None),     # viscosity cP must be >0
    "de":         (0.0,   None),     # dielectric constant >0
    "density":    (0.0,   None),     # density >0
    "solid_Hf":   (-10000, 5000),    # kJ/mol; extreme values caught here
    "gas_Hf":     (-10000, 5000),
    "liquid_Hf":  (-10000, 5000),
    "BCF":        (None,  None),
    "AOH":        (-20.0, -5.0),     # log10(OH rate): physically ~-20 to -5
    "k":          (0.0,   None),
    "RI":         (1.0,   2.0),      # refractive index typical range
    "freesolv_s": (None,  None),
    "delaney_s":  (None,  None),
}


def find_col(df: pd.DataFrame, preferred: list[str]) -> str | None:
    for c in preferred:
        if c in df.columns:
            return c
    return None


def get_smiles_col(df: pd.DataFrame) -> str:
    c = next((c for c in df.columns if c.lower() == "smiles"), None)
    if c is None:
        raise ValueError(f"no SMILES column in {list(df.columns)}")
    return c


def get_target_col(df: pd.DataFrame, smi_col: str) -> str:
    pref = ["TARGET", "target", "mean", "y", "value"]
    c = find_col(df, pref)
    if c is None:
        skip = {smi_col, "_split", "split", "n", "std"}
        c = next((x for x in df.columns if x not in skip), None)
    if c is None:
        raise ValueError(f"no target column in {list(df.columns)}")
    return c


def screen_one(
    name: str,
    csv_path: Path,
    out_dir: Path,
    outlier_threshold_rel: float,
    max_heavy_atoms: int | None,
    aggregation: str,
) -> dict:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    smi_col = get_smiles_col(df)
    tgt_col = get_target_col(df, smi_col)
    n_raw = len(df)

    # ---- 1. SMILES validation ------------------------------------------
    n_empty = n_unparse = n_radical = n_disconnect = n_bad_elem = n_bad_charge = 0
    n_ha_cap = 0
    bad_elem_counter: dict[str, int] = {}
    valid_rows: list[int] = []
    inchikeys: list[str] = []
    canons:    list[str] = []

    for i, smi_raw in enumerate(df[smi_col].astype(str)):
        smi = smi_raw.strip()
        if not smi or smi.lower() in ("nan", "none"):
            n_empty += 1;  continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_unparse += 1;  continue
        if any(a.GetNumRadicalElectrons() > 0 for a in mol.GetAtoms()):
            n_radical += 1;  continue
        canon = Chem.MolToSmiles(mol, isomericSmiles=True)
        if "." in canon:
            n_disconnect += 1;  continue
        bad = {a.GetSymbol() for a in mol.GetAtoms()} - SUPPORTED
        if bad:
            for e in bad:
                bad_elem_counter[e] = bad_elem_counter.get(e, 0) + 1
            n_bad_elem += 1;  continue
        if any(abs(a.GetFormalCharge()) > 1 for a in mol.GetAtoms()):
            n_bad_charge += 1;  continue
        if max_heavy_atoms and mol.GetNumHeavyAtoms() > max_heavy_atoms:
            n_ha_cap += 1;  continue
        ik = MolToInchiKey(mol)
        if ik is None:
            n_unparse += 1;  continue
        valid_rows.append(i)
        inchikeys.append(ik)
        canons.append(canon)

    valid_df = df.iloc[valid_rows].copy().reset_index(drop=True)
    valid_df["__inchikey"] = inchikeys
    valid_df["__canon"]    = canons
    valid_df["TARGET_num"] = pd.to_numeric(valid_df[tgt_col], errors="coerce")
    valid_df = valid_df.dropna(subset=["TARGET_num"]).reset_index(drop=True)
    n_after_smi = len(valid_df)

    # ---- 2. Hard-limit filter ------------------------------------------
    lo, hi = HARD_LIMITS.get(name, (None, None))
    hard_mask = pd.Series(True, index=valid_df.index)
    if lo is not None:
        hard_mask &= (valid_df["TARGET_num"] >= lo)
    if hi is not None:
        hard_mask &= (valid_df["TARGET_num"] <= hi)
    n_hard_removed = int((~hard_mask).sum())
    hard_removed_df = valid_df[~hard_mask].copy()
    valid_df = valid_df[hard_mask].reset_index(drop=True)
    n_after_hard = len(valid_df)

    # (6-MAD statistical outlier step removed — too aggressive for datasets
    # with legitimate wide/bimodal distributions like viscosity or dielectrics.)
    n_stat_removed = 0
    stat_removed_df = valid_df.iloc[[]]
    n_after_stat = len(valid_df)

    # ---- 4. InChIKey deduplication -------------------------------------
    target_std = float(valid_df["TARGET_num"].std()) if len(valid_df) > 1 else 1.0
    threshold_abs = outlier_threshold_rel * target_std

    kept_rows: list[dict] = []
    removed_ik: list[dict] = []   # groups removed due to large spread
    n_agg = 0

    for ik, group in valid_df.groupby("__inchikey", sort=False):
        if len(group) == 1:
            kept_rows.append(group.iloc[0].to_dict())
            continue
        vals = group["TARGET_num"].values.astype(float)
        spread = float(vals.max() - vals.min())
        if spread > threshold_abs:
            # Remove entire group — log each molecule
            for _, row in group.iterrows():
                removed_ik.append({
                    "inchikey": ik,
                    "canonical_smiles": row["__canon"],
                    "original_smiles": row[smi_col],
                    "target_value": float(row["TARGET_num"]),
                    "group_spread": round(spread, 6),
                    "threshold": round(threshold_abs, 6),
                    "reason": "inchikey_dup_large_spread",
                })
        else:
            # Aggregate
            if aggregation == "median":
                agg_val = float(np.median(vals))
            elif aggregation == "first":
                agg_val = float(vals[0])
            else:
                agg_val = float(np.mean(vals))
            row = group.iloc[0].copy()
            row["TARGET_num"] = agg_val
            kept_rows.append(row.to_dict())
            n_agg += 1

    cleaned = pd.DataFrame(kept_rows)
    # Rename TARGET_num → TARGET, keep only SMILES + TARGET + __canon for reference
    if len(cleaned) == 0:
        cleaned = pd.DataFrame(columns=["SMILES", "TARGET"])
    else:
        cleaned["SMILES"]  = cleaned["__canon"]   # canonical form
        cleaned["TARGET"]  = cleaned["TARGET_num"]
        cleaned = cleaned[["SMILES", "TARGET"]].reset_index(drop=True)

    n_ik_removed = len(removed_ik)
    n_final = len(cleaned)

    # ---- Write outputs --------------------------------------------------
    ds_out = out_dir / name
    ds_out.mkdir(parents=True, exist_ok=True)

    cleaned.to_csv(ds_out / "cleaned.csv", index=False)

    # All removed entries in one CSV
    removed_all = []
    for _, row in hard_removed_df.iterrows():
        removed_all.append({
            "inchikey": row["__inchikey"],
            "canonical_smiles": row["__canon"],
            "original_smiles": row[smi_col],
            "target_value": float(row["TARGET_num"]),
            "reason": "hard_limit_violation",
        })
    for _, row in stat_removed_df.iterrows():
        removed_all.append({
            "inchikey": row["__inchikey"],
            "canonical_smiles": row["__canon"],
            "original_smiles": row[smi_col],
            "target_value": float(row["TARGET_num"]),
            "reason": "statistical_outlier_6mad",
        })
    removed_all.extend(removed_ik)
    pd.DataFrame(removed_all).to_csv(ds_out / "removed.csv", index=False)

    stats = {
        "name": name,
        "n_raw": n_raw,
        "n_empty": n_empty,
        "n_unparseable": n_unparse,
        "n_radical": n_radical,
        "n_disconnected": n_disconnect,
        "n_bad_elements": n_bad_elem,
        "n_bad_charge": n_bad_charge,
        "n_ha_cap": n_ha_cap,
        "bad_elements": bad_elem_counter,
        "n_after_smi": n_after_smi,
        "n_hard_removed": n_hard_removed,
        "hard_limits": {"lo": lo, "hi": hi},
        "inchikey_threshold_rel": outlier_threshold_rel,
        "inchikey_threshold_abs": round(threshold_abs, 6),
        "target_std": round(target_std, 6),
        "n_ik_groups_removed": n_ik_removed,
        "n_ik_groups_aggregated": n_agg,
        "n_final": n_final,
    }
    (ds_out / "stats.json").write_text(json.dumps(stats, indent=2))

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir",  default="downstream_ft/0506/cleaned_by_CC/cleaned_by_codex")
    ap.add_argument("--output-dir", default="downstream_ft/0506/screened")
    ap.add_argument("--outlier-rel", type=float, default=0.3,
                    help="Drop InChIKey group if target spread > this × σ_global "
                         "(default 0.3).")
    ap.add_argument("--max-heavy-atoms", type=int, default=None,
                    help="Drop molecules with more than this many heavy atoms. "
                         "Use ~60 to avoid OOM during conformer generation on "
                         "very large molecules (e.g. >C60 chains). "
                         "None = no limit.")
    ap.add_argument("--aggregation", choices=["mean", "median", "first"],
                    default="mean")
    args = ap.parse_args()

    root    = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats: list[dict] = []
    print("=" * 72)
    print(f"Screening {root}/  →  {out_dir}/")
    print(f"  outlier_rel={args.outlier_rel}  agg={args.aggregation}")
    print("=" * 72)

    for csv in sorted(root.glob("*.csv")):
        if "report" in csv.name or "removed" in csv.name:
            continue
        name = csv.stem
        print(f"\n[{name}]")
        try:
            st = screen_one(name, csv, out_dir,
                            outlier_threshold_rel=args.outlier_rel,
                            max_heavy_atoms=args.max_heavy_atoms,
                            aggregation=args.aggregation)
            all_stats.append(st)
            phys = st["n_empty"] + st["n_unparseable"] + st["n_radical"] + \
                   st["n_disconnected"] + st["n_bad_elements"] + st["n_bad_charge"]
            print(f"  raw={st['n_raw']:,}  smi_drop={phys}  "
                  f"hard_drop={st['n_hard_removed']}  "
                  f"ik_drop={st['n_ik_groups_removed']}  "
                  f"ik_agg={st['n_ik_groups_aggregated']}  "
                  f"→ final={st['n_final']:,}")
            if st["bad_elements"]:
                be = ", ".join(f"{e}:{n}" for e, n in
                               sorted(st["bad_elements"].items(), key=lambda x: -x[1]))
                print(f"  bad elements: {be}")
        except Exception as e:
            print(f"  FAILED: {e}")

    # ---- Cross-dataset summary -----------------------------------------
    print("\n" + "=" * 72)
    print(f"{'dataset':<14s}  {'raw':>6s}  {'smi':>5s}  {'hard':>5s}  "
          f"{'ik_rm':>6s}  {'ik_ag':>6s}  {'final':>7s}")
    print("-" * 65)
    for st in all_stats:
        phys = st["n_empty"] + st["n_unparseable"] + st["n_radical"] + \
               st["n_disconnected"] + st["n_bad_elements"] + st["n_bad_charge"]
        print(f"{st['name']:<14s}  {st['n_raw']:>6,}  {phys:>5,}  "
              f"{st['n_hard_removed']:>5,}  "
              f"{st['n_ik_groups_removed']:>6,}  {st['n_ik_groups_aggregated']:>6,}  "
              f"{st['n_final']:>7,}")
    print("=" * 72)

    # ---- Master report -------------------------------------------------
    (out_dir / "screening_report.json").write_text(
        json.dumps({
            "generated": datetime.now().isoformat(timespec="seconds"),
            "input": str(root),
            "params": vars(args),
            "datasets": all_stats,
        }, indent=2)
    )
    print(f"\nOutputs: {out_dir}/<name>/cleaned.csv  (use for downstream FT)")
    print(f"         {out_dir}/<name>/removed.csv  (removed entries with reasons)")
    print(f"         {out_dir}/screening_report.json")

    # ---- What might still be missing -----------------------------------
    print("""
Missing steps to consider:
  - Temperature normalisation: Vcp measurements may be at different T;
    mixing them will add noise. Filter to a reference temperature if known.
  - Unit verification: verify every dataset uses the same units before
    pooling from multiple sources (kJ/mol vs kcal/mol, etc.).
  - Cross-dataset leakage check: same molecule in gas_Hf and liquid_Hf
    with very different values may indicate measurement inconsistency.
  - Conformer-gen feasibility: --max-heavy-atoms 60 will drop chains
    like C78 fatty acids that caused OOM during K=8 sampling.
  - Stereoisomer handling: cis/trans isomers share InChIKey standard layer
    but have different properties; InChIKey-b encodes them. Current dedup
    ignores stereo — set INCHI_OPTION='DoNotAddH' etc. if needed.
""")


if __name__ == "__main__":
    main()
