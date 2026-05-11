"""Audit downstream_ft/0511 datasets → downstream_ft/0511_cc_audit/.

For each dataset:
  1. Select the newest Data_YYYYMMDD_*.csv in <dataset>/Clean/
     (files not starting with 'Data_' are ignored, e.g. k_jy.csv)
  2. Apply audit rules: SMILES validation → hard limits → InChIKey dedup
  3. Write Clean/<name>.csv (SMILES, TARGET)
  4. Chemistry-aware k-NN outlier flagging → suspected_outliers/<name>.csv
     (review only; NOT auto-removed)
  5. Run cv_split (random_cv5 + scaffold_cv3) → Split/<name>/

Skipped: EnthalpyofFormation_*, SurfaceTension.

Usage:
  python scripts/audit_0511.py
  python scripts/audit_0511.py --src downstream_ft/0511 --out downstream_ft/0511_cc_audit
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.inchi import MolToInchiKey

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Dataset catalogue
# Each entry:
#   folder     — relative path under 0511/ to the Clean dir
#   name       — output name (used for Clean/<name>.csv, Split/<name>/)
#   file_glob  — substring to match in filename (None = any Data_*.csv)
#   lo / hi    — hard physical limits (None = no limit)
#   tgt_col    — target column name override (None = auto-detect)
#   extra_cols — additional columns to keep (e.g. Solvent)
#   log10      — apply log10 to TARGET before auditing (e.g. AOH raw rate → log10)
# ---------------------------------------------------------------------------
DATASETS = [
    # folder                                     name                    file_glob   lo        hi      tgt_col   extra_cols  log10
    ("Aqueous_pKa/Clean",                        "pKa",                  None,       -5.0,     25.0,   None,     [],         False),
    ("AtmosphericProperties/AOH/Clean",          "AOH",                  None,       -20.0,    -5.0,   None,     [],         True),  # raw k_OH → log10
    ("AtmosphericProperties/BCF/Clean",          "BCF",                  None,       None,     None,   None,     [],         False),
    ("Cp/Clean",                                 "Cp",                   None,        0.0,     None,   None,     [],         False),
    ("CriticalPressure/Clean",                   "Pc",                   None,        100.0,   None,   None,     [],         False),
    ("CriticalTemperature/Clean",                "Tc",                   None,        0.0,     None,   None,     [],         False),
    ("Density/Clean",                            "Density",              None,        0.0,     None,   None,     [],         False),
    ("ESOL/Clean",                               "ESOL",                 None,       -15.0,    3.0,    None,     [],         False),
    ("Lipophilicity/Clean",                      "Lipophilicity",        None,       -10.0,    12.0,   None,     [],         False),
    ("RI/Clean",                                 "RI",                   None,        1.0,     2.0,    None,     [],         False),
    ("Solubility/Clean",                         "Solubility_water",     "water",    -20.0,    5.0,    None,     [],         False),
    ("Solubility/Clean",                         "Solubility_ethanol",   "ethanol",  -10.0,    5.0,    None,     [],         False),
    ("SolvationFreeEnergy/Clean",                "SolvationFreeEnergy",  None,       -100.0,   20.0,   None,     ["Solvent"], False),
    ("TriplePointTemp/Clean",                    "TPT",                  None,        0.0,     None,   None,     [],         False),
    ("Vcp/Clean",                                "Vcp",                  None,        0.0,     None,   None,     [],         False),
    ("bp/Clean",                                 "BP",                   None,        50.0,    2000.0, None,     [],         False),
    ("de/Clean",                                 "de",                   None,        0.0,     None,   None,     [],         False),
    ("freesolv/Clean",                           "freesolv",             None,       -40.0,    10.0,   None,     [],         False),
    ("k/Clean",                                  "k",                    None,        0.0,     None,   None,     [],         False),
    ("mp/Clean",                                 "MP",                   None,        20.0,    2000.0, None,     [],         False),
]

SUPPORTED_ELEMENTS = {
    "H", "B", "C", "N", "O", "F", "Al", "Si", "P", "S", "Cl",
    "As", "Br", "I", "Hg", "Bi", "Se",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def newest_data_csv(folder: Path, glob: str | None) -> Path | None:
    """Return the newest Data_YYYYMMDD_*.csv in folder.
    If glob is given, only files whose name contains that substring qualify.
    Files not starting with 'Data_' are ignored.
    """
    candidates = [
        f for f in folder.glob("Data_*.csv")
        if glob is None or glob.lower() in f.name.lower()
    ]
    if not candidates:
        return None
    # Sort by the YYYYMMDD date embedded in the filename
    def date_key(p: Path) -> str:
        m = re.search(r"Data_(\d{8})", p.name)
        return m.group(1) if m else "00000000"
    return max(candidates, key=date_key)


def get_smiles_col(df: pd.DataFrame) -> str:
    c = next((c for c in df.columns if c.lower() == "smiles"), None)
    if c is None:
        raise ValueError(f"no SMILES column in {list(df.columns)}")
    return c


def get_target_col(df: pd.DataFrame, smi_col: str, override: str | None) -> str:
    if override and override in df.columns:
        return override
    for pref in ["TARGET", "target", "mean", "y", "value"]:
        if pref in df.columns:
            return pref
    skip = {smi_col, "_split", "split", "n", "std", "Solvent"}
    c = next((x for x in df.columns if x not in skip), None)
    if c is None:
        raise ValueError(f"no target column in {list(df.columns)}")
    return c


def _morgan_fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def flag_knn_outliers(
    smiles_list: list[str],
    values: np.ndarray,
    k: int = 20,
    z_thresh: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flag molecules whose property deviates > z_thresh × σ from k-NN prediction.

    All molecules are evaluated regardless of neighbour similarity — those with
    no close relatives get uniform-weighted k-NN (all neighbours equally weighted).
    Returns (flagged_mask, predicted_values, best_sim_per_molecule).
    """
    n = len(smiles_list)
    if n < k + 1:
        return np.zeros(n, dtype=bool), np.full(n, np.nan), np.zeros(n)
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    valid_idx = [i for i, m in enumerate(mols) if m is not None]
    fps = [_morgan_fp(mols[i]) for i in valid_idx]
    if len(fps) < k + 1:
        return np.zeros(n, dtype=bool), np.full(n, np.nan), np.zeros(n)

    predicted = np.full(n, np.nan)
    best_sim  = np.zeros(n)
    for ii, i in enumerate(valid_idx):
        row = np.array(DataStructs.BulkTanimotoSimilarity(fps[ii], fps), dtype=np.float32)
        row[ii] = -1.0
        top = np.argsort(row)[::-1][:k]
        top_sims = row[top]
        top_vals = values[[valid_idx[j] for j in top]]
        w = np.maximum(top_sims, 0.0)
        # If all neighbours have zero similarity, fall back to uniform weighting
        if w.sum() > 0:
            predicted[i] = float(np.dot(w, top_vals) / w.sum())
        else:
            predicted[i] = float(top_vals.mean())
        best_sim[i] = float(top_sims[0])

    mask = ~np.isnan(predicted)
    if mask.sum() < 2:
        return np.zeros(n, dtype=bool), predicted, best_sim
    sigma = float(np.std(values[mask] - predicted[mask])) or 1.0

    flagged = np.zeros(n, dtype=bool)
    for i in range(n):
        if np.isnan(predicted[i]):
            continue
        if abs(values[i] - predicted[i]) > z_thresh * sigma:
            flagged[i] = True
    return flagged, predicted, best_sim


# ---------------------------------------------------------------------------
# Per-dataset audit
# ---------------------------------------------------------------------------

def audit_one(
    src_folder: Path,
    name: str,
    file_glob: str | None,
    lo: float | None,
    hi: float | None,
    tgt_col_override: str | None,
    extra_cols: list[str],
    log10_transform: bool,
    clean_dir: Path,
    outlier_dir: Path,
    removed_dir: Path,
    outlier_rel: float,
    knn_k: int,
    knn_z: float,
) -> dict:

    # --- Find newest Data_*.csv ---
    csv_path = newest_data_csv(src_folder, file_glob)
    if csv_path is None:
        raise FileNotFoundError(f"no Data_*.csv in {src_folder}" +
                                (f" matching '{file_glob}'" if file_glob else ""))

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    smi_col = get_smiles_col(df)
    tgt_col = get_target_col(df, smi_col, tgt_col_override)
    n_raw = len(df)

    # --- Step 1: SMILES validation ---
    n_empty = n_unparse = n_radical = n_disconnect = n_bad_elem = n_bad_charge = 0
    bad_elem_counter: dict[str, int] = {}
    valid_rows, inchikeys, canons = [], [], []

    for i, row in df.iterrows():
        smi_raw = str(row[smi_col]).strip()
        if not smi_raw or smi_raw.lower() in ("nan", "none"):
            n_empty += 1; continue
        mol = Chem.MolFromSmiles(smi_raw)
        if mol is None:
            n_unparse += 1; continue
        if any(a.GetNumRadicalElectrons() > 0 for a in mol.GetAtoms()):
            n_radical += 1; continue
        canon = Chem.MolToSmiles(mol, isomericSmiles=True)
        if "." in canon:
            n_disconnect += 1; continue
        bad = {a.GetSymbol() for a in mol.GetAtoms()} - SUPPORTED_ELEMENTS
        if bad:
            for e in bad:
                bad_elem_counter[e] = bad_elem_counter.get(e, 0) + 1
            n_bad_elem += 1; continue
        if any(abs(a.GetFormalCharge()) > 1 for a in mol.GetAtoms()):
            n_bad_charge += 1; continue
        ik = MolToInchiKey(mol)
        if ik is None:
            n_unparse += 1; continue
        valid_rows.append(i)
        inchikeys.append(ik)
        canons.append(canon)

    vdf = df.loc[valid_rows].copy().reset_index(drop=True)
    vdf["__ik"]    = inchikeys
    vdf["__canon"] = canons
    vdf["__tgt"]   = pd.to_numeric(vdf[tgt_col], errors="coerce")
    if log10_transform:
        mask_pos = vdf["__tgt"] > 0
        vdf.loc[mask_pos, "__tgt"] = np.log10(vdf.loc[mask_pos, "__tgt"])
        vdf.loc[~mask_pos, "__tgt"] = np.nan  # drop non-positive before log10
    vdf = vdf.dropna(subset=["__tgt"]).reset_index(drop=True)
    n_after_smi = len(vdf)

    # --- Step 2: Hard limits ---
    ok = pd.Series(True, index=vdf.index)
    if lo is not None: ok &= vdf["__tgt"] >= lo
    if hi is not None: ok &= vdf["__tgt"] <= hi
    hard_removed = vdf[~ok].copy()
    vdf = vdf[ok].reset_index(drop=True)
    n_hard = len(hard_removed)

    # --- Step 3: InChIKey deduplication (full 27-char key) ---
    sigma_g = float(vdf["__tgt"].std()) if len(vdf) > 1 else 1.0
    thresh  = outlier_rel * sigma_g

    # Dedup key = InChIKey + extra_cols (e.g. Solvent) so same molecule in
    # different solvents are treated independently.
    if extra_cols:
        dedup_key = vdf["__ik"].astype(str)
        for ec in extra_cols:
            if ec in vdf.columns:
                dedup_key = dedup_key + "|" + vdf[ec].astype(str)
        vdf["__dedup_key"] = dedup_key
    else:
        vdf["__dedup_key"] = vdf["__ik"]

    kept, ik_removed = [], []
    n_agg = 0
    for key, grp in vdf.groupby("__dedup_key", sort=False):
        if len(grp) == 1:
            kept.append(grp.iloc[0].to_dict()); continue
        vals = grp["__tgt"].values.astype(float)
        spread = float(vals.max() - vals.min())
        if spread > thresh:
            for _, row in grp.iterrows():
                ik_removed.append({
                    "inchikey": row["__ik"],
                    "canonical_smiles": row["__canon"],
                    "target": float(row["__tgt"]),
                    "group_spread": round(spread, 6),
                    "threshold": round(thresh, 6),
                    "reason": "inchikey_spread",
                    **{ec: row.get(ec) for ec in extra_cols if ec in row},
                })
        else:
            r = grp.iloc[0].copy()
            r["__tgt"] = float(np.mean(vals))
            kept.append(r.to_dict())
            n_agg += 1

    cdf = pd.DataFrame(kept).reset_index(drop=True)
    n_ik_removed = len(ik_removed)

    # --- Build output columns: SMILES + TARGET + any extras ---
    out_cols = {"SMILES": cdf["__canon"], "TARGET": cdf["__tgt"]}
    for ec in extra_cols:
        if ec in cdf.columns:
            out_cols[ec] = cdf[ec]
    cleaned = pd.DataFrame(out_cols)

    # --- Step 4: Chemistry-aware k-NN outlier flagging ---
    n_flagged = 0
    flagged_rows = []
    if len(cleaned) >= knn_k + 1:
        flag_mask, predicted, best_sim = flag_knn_outliers(
            cleaned["SMILES"].tolist(),
            cleaned["TARGET"].values.astype(float),
            k=knn_k, z_thresh=knn_z,
        )
        vals = cleaned["TARGET"].values.astype(float)
        sigma = float(np.std(vals[~np.isnan(predicted)] -
                              predicted[~np.isnan(predicted)])) or 1.0
        n_flagged = int(flag_mask.sum())
        for i in np.where(flag_mask)[0]:
            row_dict = {
                "inchikey": cdf.iloc[i]["__ik"],
                "canonical_smiles": cleaned.iloc[i]["SMILES"],
                "target": float(cleaned.iloc[i]["TARGET"]),
                "knn_predicted": round(float(predicted[i]), 4),
                "residual": round(float(vals[i] - predicted[i]), 4),
                "residual_over_sigma": round(float(abs(vals[i] - predicted[i]) / sigma), 2),
                "best_neighbour_tanimoto": round(float(best_sim[i]), 3),
                "reason": f"|residual|>{knn_z}σ  (k={knn_k})",
            }
            for ec in extra_cols:
                if ec in cleaned.columns:
                    row_dict[ec] = cleaned.iloc[i][ec]
            flagged_rows.append(row_dict)

    # --- Write outputs ---
    cleaned.to_csv(clean_dir / f"{name}.csv", index=False)

    removed_all = []
    for _, row in hard_removed.iterrows():
        removed_all.append({
            "inchikey": row["__ik"], "canonical_smiles": row["__canon"],
            "target": float(row["__tgt"]),
            "reason": f"hard_limit [lo={lo}, hi={hi}]",
            **{ec: row.get(ec) for ec in extra_cols if ec in row.index},
        })
    removed_all.extend(ik_removed)
    pd.DataFrame(removed_all).to_csv(removed_dir / f"{name}_removed.csv", index=False)
    pd.DataFrame(flagged_rows).to_csv(outlier_dir / f"{name}_suspected.csv", index=False)

    n_smi_drop = n_empty + n_unparse + n_radical + n_disconnect + n_bad_elem + n_bad_charge
    print(f"  source: {csv_path.name}")
    print(f"  raw={n_raw:,}  smi_drop={n_smi_drop}  hard_drop={n_hard}  "
          f"ik_drop={n_ik_removed}  ik_agg={n_agg}  "
          f"knn_flag={n_flagged}  → final={len(cleaned):,}")
    if bad_elem_counter:
        print(f"  bad elements: " +
              ", ".join(f"{e}:{c}" for e, c in
                        sorted(bad_elem_counter.items(), key=lambda x: -x[1])))
    if n_flagged:
        print(f"  ⚠  {n_flagged} chemistry outliers → suspected_outliers/{name}_suspected.csv")

    return {
        "name": name,
        "source": str(csv_path),
        "log10_transform": log10_transform,
        "n_raw": n_raw,
        "n_empty": n_empty, "n_unparseable": n_unparse,
        "n_radical": n_radical, "n_disconnected": n_disconnect,
        "n_bad_elements": n_bad_elem, "n_bad_charge": n_bad_charge,
        "bad_elements": bad_elem_counter,
        "n_after_smi": n_after_smi,
        "hard_limits": {"lo": lo, "hi": hi},
        "n_hard_removed": n_hard,
        "outlier_rel": outlier_rel,
        "sigma_global": round(sigma_g, 6),
        "threshold_abs": round(thresh, 6),
        "n_ik_removed": n_ik_removed,
        "n_ik_aggregated": n_agg,
        "knn_params": {"k": knn_k, "z_thresh": knn_z},
        "n_knn_flagged": n_flagged,
        "n_final": len(cleaned),
    }


# ---------------------------------------------------------------------------
# CV split (calls cv_split.py from the 0511 directory)
# ---------------------------------------------------------------------------

def run_cv_split(
    input_csv: Path,
    split_out_dir: Path,
    split_name: str,
    cv_split_script: Path,
    smiles_col: str = "SMILES",
    target_col: str = "TARGET",
) -> bool:
    split_out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(cv_split_script.resolve()),
        "--input",      str(input_csv.resolve()),      # absolute paths avoid cwd confusion
        "--output",     str(split_out_dir.resolve()),
        "--splits",     split_name,
        "--split-name", split_name,
        "--smiles-col", smiles_col,
        "--target-col", target_col,
    ]
    result = subprocess.run(
        cmd,
        capture_output=True, text=True,
        cwd=str(cv_split_script.parent),   # cv_split.py imports split_utils from same dir
    )
    if result.returncode != 0:
        print(f"    [cv_split FAILED] {result.stderr.strip()[-400:]}")
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="downstream_ft/0511")
    ap.add_argument("--out", default="downstream_ft/0511_cc_audit")
    ap.add_argument("--outlier-rel", type=float, default=0.3,
                    help="InChIKey spread threshold (× σ_global). Default 0.3.")
    ap.add_argument("--knn-k", type=int,   default=20)
    ap.add_argument("--knn-z", type=float, default=3.0)
    ap.add_argument("--skip-split",  action="store_true",
                    help="Skip CV split generation.")
    args = ap.parse_args()

    src     = Path(args.src)
    out     = Path(args.out)
    clean_dir   = out / "Clean"
    split_dir   = out / "Split"
    outlier_dir = out / "suspected_outliers"
    removed_dir = out / "removed"
    for d in (clean_dir, split_dir, outlier_dir, removed_dir):
        d.mkdir(parents=True, exist_ok=True)

    cv_split_script = (src / "cv_split.py").resolve()
    if not cv_split_script.exists():
        print(f"WARNING: cv_split.py not found at {cv_split_script}; splits will be skipped.")
        args.skip_split = True

    all_stats: list[dict] = []

    print("=" * 72)
    print(f"Auditing {src}  →  {out}")
    print(f"  outlier_rel={args.outlier_rel}  knn_k={args.knn_k}  knn_z={args.knn_z}")
    print("=" * 72)

    for (folder, name, file_glob, lo, hi, tgt_col_ov, extra_cols, log10_t) in DATASETS:
        src_folder = src / folder
        print(f"\n[{name}]  ({folder})")
        if not src_folder.is_dir():
            print(f"  SKIP — folder not found: {src_folder}")
            continue
        try:
            stats = audit_one(
                src_folder=src_folder,
                name=name,
                file_glob=file_glob,
                lo=lo, hi=hi,
                tgt_col_override=tgt_col_ov,
                extra_cols=extra_cols,
                log10_transform=log10_t,
                clean_dir=clean_dir,
                outlier_dir=outlier_dir,
                removed_dir=removed_dir,
                outlier_rel=args.outlier_rel,
                knn_k=args.knn_k,
                knn_z=args.knn_z,
            )
            all_stats.append(stats)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        # CV splits
        if not args.skip_split:
            input_csv = clean_dir / f"{name}.csv"
            ds_split_dir = split_dir / name
            smiles_col = "SMILES"
            target_col = "TARGET"
            for split_name in ("random_cv5", "scaffold_cv3"):
                sd = ds_split_dir / split_name
                ok = run_cv_split(input_csv, sd, split_name,
                                  cv_split_script, smiles_col, target_col)
                if ok:
                    n_files = len(list(sd.glob("*.csv")))
                    print(f"  [split] {split_name}: {n_files} files → {sd}")
                else:
                    print(f"  [split] {split_name}: FAILED")

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'name':<22s}  {'raw':>6s}  {'smi':>5s}  {'hard':>5s}  "
          f"{'ik_rm':>6s}  {'ik_ag':>5s}  {'flag':>5s}  {'final':>7s}")
    print("-" * 80)
    for st in all_stats:
        n_smi = (st["n_empty"] + st["n_unparseable"] + st["n_radical"] +
                 st["n_disconnected"] + st["n_bad_elements"] + st["n_bad_charge"])
        print(f"{st['name']:<22s}  {st['n_raw']:>6,}  {n_smi:>5,}  "
              f"{st['n_hard_removed']:>5,}  {st['n_ik_removed']:>6,}  "
              f"{st['n_ik_aggregated']:>5,}  {st['n_knn_flagged']:>5,}  "
              f"{st['n_final']:>7,}")
    print("=" * 80)

    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "src": str(src),
        "out": str(out),
        "params": {
            "outlier_rel": args.outlier_rel,
            "knn_k": args.knn_k,
            "knn_z": args.knn_z,
        },
        "datasets": all_stats,
    }
    (out / "audit_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nOutputs:")
    print(f"  {clean_dir}/             — cleaned CSVs (SMILES, TARGET)")
    print(f"  {split_dir}/             — CV splits (random_cv5 + scaffold_cv3)")
    print(f"  {outlier_dir}/           — flagged outliers for review")
    print(f"  {removed_dir}/           — hard-limit + InChIKey removals")
    print(f"  {out}/audit_report.json  — full statistics")


if __name__ == "__main__":
    main()
