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
    # SolvationFreeEnergy removed from db
    # ── Previously skipped, now included ──────────────────────────────────────
    # Enthalpy of formation (kJ/mol); broad limits, extremes are real
    ("EnthalpyofFormation_C/Clean",              "Hf_C",                 None,       -5000.0,  3000.0, None,     [],         False),
    ("EnthalpyofFormation_G/Clean",              "Hf_G",                 None,       -10000.0, 5000.0, None,     [],         False),
    ("EnthalpyofFormation_L/Clean",              "Hf_L",                 None,       -10000.0, 5000.0, None,     [],         False),
    # Surface tension (N/m); must be positive; typical organics 0.01–0.07
    ("SurfaceTension/Clean",                     "ST",                   None,        0.0,     0.15,   None,     [],         False),
    ("TriplePointTemp/Clean",                    "TPT",                  None,        0.0,     None,   None,     [],         False),
    ("Vcp/Clean",                                "Vcp",                  None,        0.0,     None,   None,     [],         False),
    ("bp/Clean",                                 "BP",                   None,        50.0,    2000.0, None,     [],         False),
    ("de/Clean",                                 "de",                   None,        0.0,     None,   None,     [],         False),
    ("freesolv/Clean",                           "freesolv",             None,       -40.0,    10.0,   None,     [],         False),
    ("k/Clean",                                  "k",                    None,        0.0,     None,   None,     [],         False),
    ("mp/Clean",                                 "MP",                   None,        20.0,    2000.0, None,     [],         False),
    # ── New ADMET datasets (added 2026-05-12) ──────────────────────────────────
    # AcuteToxicity: −log10(LD50 mol/kg); higher = more toxic
    ("AcuteToxicity/Clean",                      "AcuteToxicity",        None,        0.0,     8.0,    None,     [],         False),
    # CellEffectivePermeability: log10(Peff cm/s); typical −9 to −3
    ("CellEffectivePermeability/Clean",          "CEP",                  None,       -10.0,   -2.0,   None,     [],         False),
    # Clearance: mL/min/kg in vitro hepatic clearance; must be positive
    ("Clearance/Clean",                          "Clearance",            None,        0.0,     None,   None,     [],         False),
    # HalfLife: hours; must be positive
    ("HalfLife/Clean",                           "HalfLife",             None,        0.0,     None,   None,     [],         False),
    # PlasmaProteinBindingRate: % bound (0–100)
    ("PlasmaProteinBindingRate/Clean",           "PPBR",                 None,        0.0,     100.0,  None,     [],         False),
]

SUPPORTED_ELEMENTS = {
    "H", "B", "C", "N", "O", "F", "Al", "Si", "P", "S", "Cl",
    "As", "Br", "I", "Hg", "Bi", "Se",
}

# ---------------------------------------------------------------------------
# Manual removals — applied AFTER the automated audit.
# Each entry is a dict with exactly one of:
#   "filter"  — pandas query string on TARGET column
#   "smiles"  — list of SMILES to drop (matched via canonical form)
# Plus a required "reason" string (written to removed CSV).
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Physics-justified retain rules — kNN-flagged molecules matching these
# conditions are moved to a separate "physics_extreme" category and NOT
# counted as suspicious. All values are real ground-truth measurements that
# follow known physical laws.
#
#   Vcp          > 40 cP     polyols / heavy plasticizers (DEHP) — naturally viscous
#   de           > 100       secondary amides (NMF, NMA) — cooperative H-bond chain dipoles
#   RI           > 1.70      I/Br-rich molecules — high electron-cloud polarisability
#                < 1.35      perfluorocarbons — locked electron clouds
#   AOH          < −14       fully halogenated (CCl4) — no abstractable H → log-drop in rate
#   TPT/MP/BP/Tc extremes    O2/N2 low end; rigid polycyclics / super-long chains high end
#   Pc           > 10 000    water (22 010 kPa) follows critical-pressure physics
#   Density      > 2 000     heavy haloalkanes (CH2Br2 = 2 478 kg/m³)
#   Solubility   < −12       long-chain hydrophobic esters — LogS bottoming
#   freesolv     < −20       very hydrophobic chains — large +ΔG for transfer to water
# ---------------------------------------------------------------------------
PHYSICS_RETAIN: dict[str, callable] = {
    "Vcp":                lambda tgt: tgt > 40.0,
    "de":                 lambda tgt: tgt > 100.0,
    "RI":                 lambda tgt: tgt > 1.70 or tgt < 1.35,
    "AOH":                lambda tgt: tgt < -14.0,
    "TPT":                lambda tgt: tgt < 100.0  or tgt > 1000.0,
    "MP":                 lambda tgt: tgt < 150.0  or tgt > 600.0,
    "BP":                 lambda tgt: tgt < 200.0  or tgt > 700.0,
    "Tc":                 lambda tgt: (tgt < 200.0 or tgt > 1200.0) and tgt < 2500.0,
    "Pc":                 lambda tgt: tgt > 10000.0,
    "Density":            lambda tgt: tgt > 2000.0,
    "Solubility_water":   lambda tgt: tgt < -12.0,
    "Solubility_ethanol": lambda tgt: tgt < -4.0,
    "freesolv":           lambda tgt: tgt < -20.0,
    # Enthalpy of formation extremes — highly oxidized or strained compounds
    "Hf_G":               lambda tgt: tgt < -2000.0 or tgt > 1500.0,
    "Hf_L":               lambda tgt: tgt < -2000.0 or tgt > 800.0,
    "Hf_C":               lambda tgt: tgt < -1500.0 or tgt > 500.0,
    # Surface tension: high (>0.065 N/m, water-like polar) or low (<0.010, perfluoro)
    "ST":                 lambda tgt: tgt > 0.065 or tgt < 0.010,
    # ADMET extremes — all physically justified
    "AcuteToxicity":      lambda tgt: tgt > 5.5,          # extremely toxic (LD50 < 3 μmol/kg)
    "CEP":                lambda tgt: tgt < -7.0 or tgt > -4.0,  # very low/high permeability
    "Clearance":          lambda tgt: tgt > 100.0,         # ultra-rapid hepatic extraction
    "HalfLife":           lambda tgt: tgt > 200.0,         # unusually persistent compounds
    "PPBR":               lambda tgt: tgt > 98.0,          # near-total protein binding (e.g. warfarin)
}

# ---------------------------------------------------------------------------
# Manual removals — applied AFTER the automated audit.
MANUAL_REMOVALS: dict[str, dict] = {
    "SolvationFreeEnergy": {
        "filter": "TARGET < -50.0",
        "reason": "fatal: solvation FE < -50 kcal/mol — missing ionic charge in SMILES",
    },
    "Cp": {
        "smiles": [
            "CCCCCOCCOCCO",
            "O[C@H]1[C@H](O)[C@H](O)OC[C@H]1O",
            "CCCCCOCCOCCOCCO",
        ],
        "reason": "fatal: Cp ~4x too small — value in cal/(mol·K) not converted to J/(mol·K)",
    },
    "de": {
        "smiles": [
            "CNC(C)=O",
            "OCC(O)C(O)C(O)C(O)CO",
            "O=S(=O)(Cl)c1ccc(S(=O)(=O)C(F)(F)F)cc1",
            "CN(C)C",
            "N#CC#N",
            "CNC",
        ],
        "reason": "fatal: numeric error or wrong phase at RT (solid/gas in liquid dielectric dataset)",
    },
    "Tc": {
        "filter": "TARGET > 2500.0",
        "reason": "suspected QSPR extrapolation artifact: Tc > 2500 K exceeds reliable "
                  "experimental range for macrocycles (calixarenes) — likely unphysical",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_ANOMALY_XLSX = "downstream_ft/0511_cc_audit/combined_science_anomaly_report.xlsx"
DEFAULT_KEEP_RULES = "STAT_EXTREME_TUKEY"  # distribution-tail outliers — kept


def load_anomaly_blacklist(
    xlsx_path: Path, keep_rule_ids: set[str]
) -> dict[str, dict[tuple[str, float], list[str]]]:
    """Load `combined_science_anomaly_report.xlsx` → per-sheet blacklist.

    Returns ``{sheet: {(canonical_smiles, round(target, 6)): [rule_ids]}}``.
    Rows whose Rule_ID is in ``keep_rule_ids`` are skipped (kept in data).
    """
    if not xlsx_path.exists():
        print(f"  [anomaly] xlsx not found: {xlsx_path} — skipping anomaly removal")
        return {}
    df = pd.read_excel(xlsx_path, sheet_name="Anomaly_List")
    blacklist: dict[str, dict[tuple[str, float], list[str]]] = {}
    n_kept_rule = 0
    n_unparseable = 0
    for _, row in df.iterrows():
        rule = str(row["Rule_ID"])
        if rule in keep_rule_ids:
            n_kept_rule += 1
            continue
        smi_raw = str(row["SMILES"]).strip()
        tgt = row["TARGET"]
        if pd.isna(tgt) or not smi_raw or smi_raw.lower() in ("nan", "none"):
            continue
        mol = Chem.MolFromSmiles(smi_raw)
        if mol is None:
            n_unparseable += 1
            continue
        canon = Chem.MolToSmiles(mol, isomericSmiles=True)
        key = (canon, round(float(tgt), 6))
        sheet = str(row["Sheet"])
        blacklist.setdefault(sheet, {}).setdefault(key, []).append(rule)
    n_blacklisted = sum(len(v) for v in blacklist.values())
    print(f"  [anomaly] loaded {n_blacklisted} anomalies across {len(blacklist)} sheets "
          f"({n_kept_rule} kept as distribution-tail, {n_unparseable} unparseable)")
    return blacklist


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
    anomaly_set: dict[tuple[str, float], list[str]] | None = None,
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
    n_empty = n_unparse = n_radical = n_disconnect = n_bad_elem = n_bad_charge = n_isotope = 0
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
        # Reject any explicit isotope (e.g. [2H], [13C]) — InChIKey collapses
        # them with the parent atom so they otherwise sneak through and risk
        # introducing a near-duplicate with a conflicting label.
        if any(a.GetIsotope() != 0 for a in mol.GetAtoms()):
            n_isotope += 1; continue
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

    # --- Step 1.5: Drop xlsx anomalies (non-distribution-outlier types) ---
    # Match by (canonical SMILES, round(target,6)). Distribution-tail outliers
    # (STAT_EXTREME_TUKEY) are pre-filtered out of anomaly_set by the loader,
    # so anything that lands here is a real defect (charge, isotope, cross-
    # sheet inconsistency) flagged in combined_science_anomaly_report.xlsx.
    n_anomaly = 0
    anomaly_rows: list[dict] = []
    if anomaly_set:
        keys = list(zip(vdf["__canon"], vdf["__tgt"].round(6)))
        drop_mask = pd.Series([k in anomaly_set for k in keys], index=vdf.index)
        n_anomaly = int(drop_mask.sum())
        if n_anomaly:
            adf = vdf[drop_mask]
            for _, row in adf.iterrows():
                key = (row["__canon"], round(float(row["__tgt"]), 6))
                rules = anomaly_set.get(key, [])
                anomaly_rows.append({
                    "inchikey": row["__ik"],
                    "canonical_smiles": row["__canon"],
                    "target": float(row["__tgt"]),
                    "reason": f"anomaly_xlsx [{','.join(sorted(set(rules)))}]",
                    **{ec: row.get(ec) for ec in extra_cols if ec in row.index},
                })
            vdf = vdf[~drop_mask].reset_index(drop=True)

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
    # Two flag types written to the same suspected CSV:
    #   "high_residual"  — |actual - kNN_pred| > z_thresh × σ
    #   "isolated"       — best neighbour Tanimoto < iso_thresh (no structural context)
    iso_thresh = 0.2
    n_flagged = 0
    n_isolated = 0
    n_physics_extreme = 0
    flagged_rows = []
    physics_extreme_rows: list[dict] = []
    seen_isolated: set[int] = set()

    if len(cleaned) >= knn_k + 1:
        flag_mask, predicted, best_sim = flag_knn_outliers(
            cleaned["SMILES"].tolist(),
            cleaned["TARGET"].values.astype(float),
            k=knn_k, z_thresh=knn_z,
        )
        vals = cleaned["TARGET"].values.astype(float)
        sigma = float(np.std(vals[~np.isnan(predicted)] -
                              predicted[~np.isnan(predicted)])) or 1.0
        # Apply physics retain rules: demote to "physics_extreme" category
        retain_fn = PHYSICS_RETAIN.get(name)
        physics_extreme_rows = []
        if retain_fn is not None:
            for i in np.where(flag_mask)[0]:
                if retain_fn(float(vals[i])):
                    flag_mask[i] = False
                    physics_extreme_rows.append({
                        "inchikey": cdf.iloc[i]["__ik"],
                        "canonical_smiles": cleaned.iloc[i]["SMILES"],
                        "target": float(cleaned.iloc[i]["TARGET"]),
                        "knn_predicted": round(float(predicted[i]), 4),
                        "residual": round(float(vals[i] - predicted[i]), 4),
                        "residual_over_sigma": round(float(abs(vals[i] - predicted[i]) / sigma), 2),
                        "best_neighbour_tanimoto": round(float(best_sim[i]), 3),
                        "reason": "physics_extreme: retained — follows known physical law",
                        **{ec: cleaned.iloc[i][ec] for ec in extra_cols
                           if ec in cleaned.columns},
                    })

        n_flagged = int(flag_mask.sum())
        n_physics_extreme = len(physics_extreme_rows)

        # high-residual flags
        for i in np.where(flag_mask)[0]:
            row_dict = {
                "inchikey": cdf.iloc[i]["__ik"],
                "canonical_smiles": cleaned.iloc[i]["SMILES"],
                "target": float(cleaned.iloc[i]["TARGET"]),
                "knn_predicted": round(float(predicted[i]), 4),
                "residual": round(float(vals[i] - predicted[i]), 4),
                "residual_over_sigma": round(float(abs(vals[i] - predicted[i]) / sigma), 2),
                "best_neighbour_tanimoto": round(float(best_sim[i]), 3),
                "reason": f"high_residual: |res|>{knn_z}σ (k={knn_k})",
            }
            for ec in extra_cols:
                if ec in cleaned.columns:
                    row_dict[ec] = cleaned.iloc[i][ec]
            flagged_rows.append(row_dict)
            seen_isolated.add(i)

        # isolated flags — not already flagged by residual
        for i in range(len(cleaned)):
            if np.isnan(predicted[i]) or i in seen_isolated:
                continue
            if best_sim[i] < iso_thresh:
                n_isolated += 1
                row_dict = {
                    "inchikey": cdf.iloc[i]["__ik"],
                    "canonical_smiles": cleaned.iloc[i]["SMILES"],
                    "target": float(cleaned.iloc[i]["TARGET"]),
                    "knn_predicted": round(float(predicted[i]), 4),
                    "residual": round(float(vals[i] - predicted[i]), 4),
                    "residual_over_sigma": round(float(abs(vals[i] - predicted[i]) / sigma), 2),
                    "best_neighbour_tanimoto": round(float(best_sim[i]), 3),
                    "reason": f"isolated: best_sim={best_sim[i]:.3f}<{iso_thresh}",
                }
                for ec in extra_cols:
                    if ec in cleaned.columns:
                        row_dict[ec] = cleaned.iloc[i][ec]
                flagged_rows.append(row_dict)

    # --- Write outputs ---
    cleaned.to_csv(clean_dir / f"{name}.csv", index=False)

    removed_all = []
    removed_all.extend(anomaly_rows)
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
    if physics_extreme_rows:
        pd.DataFrame(physics_extreme_rows).to_csv(
            outlier_dir / f"{name}_physics_extreme.csv", index=False)

    n_smi_drop = n_empty + n_unparse + n_radical + n_isotope + n_disconnect + n_bad_elem + n_bad_charge
    print(f"  source: {csv_path.name}")
    print(f"  raw={n_raw:,}  smi_drop={n_smi_drop}  anom_drop={n_anomaly}  "
          f"hard_drop={n_hard}  ik_drop={n_ik_removed}  ik_agg={n_agg}  "
          f"knn_flag={n_flagged}  isolated={n_isolated}  → final={len(cleaned):,}")
    if bad_elem_counter:
        print(f"  bad elements: " +
              ", ".join(f"{e}:{c}" for e, c in
                        sorted(bad_elem_counter.items(), key=lambda x: -x[1])))
    if n_flagged or n_isolated:
        print(f"  ⚠  {n_flagged} high-residual + {n_isolated} isolated "
              f"→ suspected_outliers/{name}_suspected.csv")
    if n_physics_extreme:
        print(f"  ✓  {n_physics_extreme} physics-justified extremes retained "
              f"→ suspected_outliers/{name}_physics_extreme.csv")

    return {
        "name": name,
        "source": str(csv_path),
        "log10_transform": log10_transform,
        "n_raw": n_raw,
        "n_empty": n_empty, "n_unparseable": n_unparse,
        "n_radical": n_radical, "n_isotope": n_isotope,
        "n_disconnected": n_disconnect,
        "n_bad_elements": n_bad_elem, "n_bad_charge": n_bad_charge,
        "bad_elements": bad_elem_counter,
        "n_after_smi": n_after_smi,
        "n_anomaly_removed": n_anomaly,
        "hard_limits": {"lo": lo, "hi": hi},
        "n_hard_removed": n_hard,
        "outlier_rel": outlier_rel,
        "sigma_global": round(sigma_g, 6),
        "threshold_abs": round(thresh, 6),
        "n_ik_removed": n_ik_removed,
        "n_ik_aggregated": n_agg,
        "knn_params": {"k": knn_k, "z_thresh": knn_z, "iso_thresh": iso_thresh},
        "n_knn_flagged": n_flagged,
        "n_physics_extreme": n_physics_extreme,
        "n_isolated": n_isolated,
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
    overwrite: bool = False,
) -> bool:
    split_out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(cv_split_script.resolve()),
        "--input",      str(input_csv.resolve()),
        "--output",     str(split_out_dir.resolve()),
        "--splits",     split_name,
        "--split-name", split_name,
        "--smiles-col", smiles_col,
        "--target-col", target_col,
    ]
    if overwrite:
        cmd.append("--overwrite")
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
# Manual removal application
# ---------------------------------------------------------------------------

def _canon(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None


def apply_manual_removals(
    name: str,
    clean_csv: Path,
    removed_csv: Path,
) -> int:
    """Apply MANUAL_REMOVALS[name] to clean_csv in-place.

    Appends removed rows to removed_csv and overwrites clean_csv.
    Returns number of rows removed.
    """
    spec = MANUAL_REMOVALS.get(name)
    if spec is None:
        return 0

    df = pd.read_csv(clean_csv)
    reason = spec["reason"]

    if "filter" in spec:
        drop_mask = df.eval(spec["filter"])
    else:
        canon_set = {_canon(s) for s in spec["smiles"] if _canon(s)}
        drop_mask = df["SMILES"].apply(_canon).isin(canon_set)

    removed = df[drop_mask].copy()
    kept    = df[~drop_mask].reset_index(drop=True)

    if len(removed) == 0:
        return 0

    removed["reason"] = reason
    # Append to existing removed CSV (create if absent or empty)
    out_rows = removed[["SMILES", "TARGET", "reason"]]
    if removed_csv.exists() and removed_csv.stat().st_size > 0:
        try:
            existing = pd.read_csv(removed_csv)
            out_rows = pd.concat([existing, out_rows], ignore_index=True)
        except Exception:
            pass
    out_rows.to_csv(removed_csv, index=False)

    kept.to_csv(clean_csv, index=False)
    return len(removed)


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
    ap.add_argument("--anomaly-xlsx", default=DEFAULT_ANOMALY_XLSX,
                    help="Path to combined_science_anomaly_report.xlsx; rows in "
                         "its Anomaly_List whose Rule_ID is NOT in --keep-anomaly-"
                         "rules are dropped. Pass '' to disable.")
    ap.add_argument("--keep-anomaly-rules", default=DEFAULT_KEEP_RULES,
                    help="Comma-separated Rule_IDs from the xlsx that should be "
                         "kept (not removed). Default: STAT_EXTREME_TUKEY "
                         "(distribution-tail outliers).")
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

    # Load anomaly blacklist once
    keep_rules = {r.strip() for r in args.keep_anomaly_rules.split(",") if r.strip()}
    anomaly_blacklist: dict[str, dict[tuple[str, float], list[str]]] = {}
    if args.anomaly_xlsx:
        anomaly_blacklist = load_anomaly_blacklist(Path(args.anomaly_xlsx), keep_rules)

    print("=" * 72)
    print(f"Auditing {src}  →  {out}")
    print(f"  outlier_rel={args.outlier_rel}  knn_k={args.knn_k}  knn_z={args.knn_z}")
    if args.anomaly_xlsx:
        print(f"  anomaly_xlsx={args.anomaly_xlsx}")
        print(f"  keep_anomaly_rules={sorted(keep_rules)}")
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
                anomaly_set=anomaly_blacklist.get(name),
            )
            all_stats.append(stats)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        # CV splits — always overwrite to guarantee reproducibility
        if not args.skip_split:
            input_csv = clean_dir / f"{name}.csv"
            ds_split_dir = split_dir / name
            for sn in ("random_cv5", "scaffold_cv3"):
                sd = ds_split_dir / sn
                ok = run_cv_split(input_csv, sd, sn, cv_split_script, overwrite=True)
                if ok:
                    print(f"  [split] {sn}: {len(list(sd.glob('*.csv')))} files")
                else:
                    print(f"  [split] {sn}: FAILED")

    # --- Manual removals (applied after automated audit) ---
    if MANUAL_REMOVALS:
        print("\n" + "=" * 72)
        print("Manual removals")
        print("=" * 72)
        for mr_name, mr_spec in MANUAL_REMOVALS.items():
            clean_csv   = clean_dir   / f"{mr_name}.csv"
            removed_csv = removed_dir / f"{mr_name}_removed.csv"
            if not clean_csv.exists():
                print(f"  [{mr_name}] SKIP — Clean CSV not found")
                continue
            n_removed = apply_manual_removals(mr_name, clean_csv, removed_csv)
            n_remaining = len(pd.read_csv(clean_csv))
            print(f"  [{mr_name}] removed {n_removed} rows → {n_remaining} remaining")
            # Update stats
            for st in all_stats:
                if st["name"] == mr_name:
                    st["n_manual_removed"] = n_removed
                    st["n_final"] = n_remaining
            # Regenerate splits for this dataset
            if not args.skip_split and cv_split_script.exists():
                ds_split_dir = split_dir / mr_name
                for split_name in ("random_cv5", "scaffold_cv3"):
                    sd = ds_split_dir / split_name
                    ok = run_cv_split(clean_csv, sd, split_name,
                                      cv_split_script, overwrite=True)
                    print(f"    [split] {split_name}: {'regenerated' if ok else 'FAILED'}")

    # -----------------------------------------------------------------------
    # Comprehensive exclusion summary
    # -----------------------------------------------------------------------
    W = 90
    print("\n" + "=" * W)
    print("EXCLUSION SUMMARY  (all rows permanently removed from Clean CSVs)")
    print("=" * W)

    total_removed = 0
    for st in all_stats:
        n_smi   = (st["n_empty"] + st["n_unparseable"] + st["n_radical"] +
                   st.get("n_isotope", 0) +
                   st["n_disconnected"] + st["n_bad_elements"] + st["n_bad_charge"])
        n_anom  = st.get("n_anomaly_removed", 0)
        n_hard  = st["n_hard_removed"]
        n_ik    = st["n_ik_removed"]
        n_man   = st.get("n_manual_removed", 0)
        n_total = n_smi + n_anom + n_hard + n_ik + n_man
        total_removed += n_total
        if n_total == 0:
            continue
        print(f"\n  [{st['name']}]  raw={st['n_raw']:,}  → final={st['n_final']:,}"
              f"  (removed {n_total})")
        if n_smi:
            breakdown = []
            if st["n_empty"]:      breakdown.append(f"empty={st['n_empty']}")
            if st["n_unparseable"]:breakdown.append(f"unparseable={st['n_unparseable']}")
            if st["n_radical"]:    breakdown.append(f"radical={st['n_radical']}")
            if st.get("n_isotope"):breakdown.append(f"isotope={st['n_isotope']}")
            if st["n_disconnected"]:breakdown.append(f"disconnected={st['n_disconnected']}")
            if st["n_bad_elements"]:
                be = ",".join(f"{e}:{c}" for e,c in
                              sorted(st["bad_elements"].items(), key=lambda x:-x[1]))
                breakdown.append(f"bad_elem={st['n_bad_elements']}({be})")
            if st["n_bad_charge"]: breakdown.append(f"bad_charge={st['n_bad_charge']}")
            print(f"    SMILES validation : {n_smi:4d}  — {', '.join(breakdown)}")
        if n_anom:
            print(f"    Anomaly xlsx      : {n_anom:4d}  — rows in Anomaly_List ≠ STAT_EXTREME_TUKEY")
        if n_hard:
            lo, hi = st["hard_limits"]["lo"], st["hard_limits"]["hi"]
            print(f"    Hard limits       : {n_hard:4d}  — target outside [{lo}, {hi}]")
        if n_ik:
            print(f"    InChIKey spread   : {n_ik:4d}  — duplicate SMILES, "
                  f"|spread| > {st['threshold_abs']:.3g} ({st['outlier_rel']}×σ)")
        if n_man:
            spec = MANUAL_REMOVALS.get(st["name"], {})
            print(f"    Manual removal    : {n_man:4d}  — {spec.get('reason','')}")

    print(f"\n  TOTAL removed across all datasets: {total_removed:,}")

    print("\n" + "=" * W)
    print("AGGREGATED (same InChIKey, consistent values → averaged, kept)")
    print("=" * W)
    for st in all_stats:
        if st["n_ik_aggregated"]:
            print(f"  [{st['name']}]  {st['n_ik_aggregated']} groups averaged")

    print("\n" + "=" * W)
    print("SUSPECTED OUTLIERS  (kNN-flagged, NOT removed — for external review)")
    print("=" * W)
    for st in all_stats:
        n_f = st.get("n_knn_flagged", 0)
        n_i = st.get("n_isolated", 0)
        n_p = st.get("n_physics_extreme", 0)
        if n_f + n_i + n_p:
            print(f"  [{st['name']}]  high_residual={n_f}  isolated={n_i}  "
                  f"physics_extreme={n_p} (retained ground truth)")

    print("\n" + "=" * W)
    print("DATASET TABLE")
    print("=" * W)
    print(f"  {'name':<22s}  {'raw':>7s}  {'smi':>5s}  {'anom':>5s}  {'hard':>5s}  "
          f"{'ik_rm':>6s}  {'ik_ag':>5s}  {'manual':>7s}  {'final':>7s}")
    print("  " + "-" * 86)
    for st in all_stats:
        n_smi = (st["n_empty"] + st["n_unparseable"] + st["n_radical"] +
                 st.get("n_isotope", 0) +
                 st["n_disconnected"] + st["n_bad_elements"] + st["n_bad_charge"])
        print(f"  {st['name']:<22s}  {st['n_raw']:>7,}  {n_smi:>5,}  "
              f"{st.get('n_anomaly_removed',0):>5,}  "
              f"{st['n_hard_removed']:>5,}  {st['n_ik_removed']:>6,}  "
              f"{st['n_ik_aggregated']:>5,}  {st.get('n_manual_removed',0):>7,}  "
              f"{st['n_final']:>7,}")
    print("  " + "=" * 86)

    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "src": str(src),
        "out": str(out),
        "params": {
            "outlier_rel": args.outlier_rel,
            "knn_k": args.knn_k,
            "knn_z": args.knn_z,
            "anomaly_xlsx": args.anomaly_xlsx,
            "keep_anomaly_rules": sorted(keep_rules),
        },
        "datasets": all_stats,
    }
    (out / "audit_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nOutputs:")
    print(f"  {clean_dir}/             — 20 cleaned CSVs (SMILES, TARGET)")
    print(f"  {split_dir}/             — CV splits (random_cv5 + scaffold_cv3)")
    print(f"  {outlier_dir}/           — kNN-flagged rows for review")
    print(f"  {removed_dir}/           — all permanently removed rows + reasons")
    print(f"  {out}/audit_report.json  — full per-dataset statistics")
    print(f"\nTo reproduce exactly: python scripts/audit_0511.py")


if __name__ == "__main__":
    main()
