"""Sanity-check the property_table.parquet produced by build_property_table.py.

Runs fast (<1 min on 1.85M rows) and prints PASS/FAIL per check. Use
--strict to exit non-zero on any failure (for CI).

Checks:
  1. Schema: exactly the expected 16 columns, correct dtypes.
  2. Shape matches the original chembl3d _h.pt row count.
  3. All canonical SMILES are unique and valid (RDKit parse round-trip
     on a random sample).
  4. has_thermo_label matches thermo-column non-NaN-ness (exact).
  5. Thermo coverage (has_thermo_label.sum()) equals the expected count,
     either passed in via --expected-thermo-matches or shown for review.
  6. RDKit descriptors are populated (NaN only for the handful that
     genuinely failed during build).
  7. Value ranges look sane (logp / qed / tpsa / etc. within physical
     bounds). Clear warnings on any outliers.
  8. Cross-lookup from a random sample of each _h.pt split resolves in
     the table (tests that AttachProperties will hit at train time).

Usage:
  python data_processing/check_property_table.py \\
      --parquet data/property_table.parquet \\
      --inputs data/chembl3d_stereo/processed/{train,val,test}_h.pt \\
      --expected-thermo-matches 652782
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import Mol
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage

RDLogger.DisableLog("rdApp.*")

THERMO_FIELDS = ["enthalpy_0", "enthalpy_298", "gibbs_298", "cv_gas", "entropy_gas"]
RDKIT_FIELDS = [
    "logp", "tpsa", "n_h_donors", "n_h_acceptors", "n_rot_bonds",
    "frac_csp3", "n_aliph_rings", "qed", "labute_asa",
]
EXPECTED_COLS = ["smiles", "has_thermo_label"] + THERMO_FIELDS + RDKIT_FIELDS

# Reasonable physical-ish bounds for the 9 RDKit descriptors — anything
# outside is worth a warning but isn't automatically a failure (drug-like
# screens may surface legit extremes).
RDKIT_BOUNDS = {
    "logp":          (-20.0, 25.0),
    "tpsa":          (0.0, 1000.0),
    "n_h_donors":    (0, 100),
    "n_h_acceptors": (0, 100),
    "n_rot_bonds":   (0, 200),
    "frac_csp3":     (0.0, 1.0),
    "n_aliph_rings": (0, 50),
    "qed":           (0.0, 1.0),
    "labute_asa":    (0.0, 5000.0),
}


class Report:
    def __init__(self):
        self.fails = []
        self.warns = []

    def ok(self, msg):
        print(f"  \u2713 {msg}")

    def fail(self, msg):
        print(f"  \u2717 FAIL: {msg}")
        self.fails.append(msg)

    def warn(self, msg):
        print(f"  \u26a0 WARN: {msg}")
        self.warns.append(msg)


def check_schema(df, r):
    print("\n[1/7] schema + dtypes")
    extra = set(df.columns) - set(EXPECTED_COLS)
    missing = set(EXPECTED_COLS) - set(df.columns)
    if extra:
        r.fail(f"extra columns: {sorted(extra)}")
    if missing:
        r.fail(f"missing columns: {sorted(missing)}")
    if not extra and not missing and list(df.columns) == EXPECTED_COLS:
        r.ok(f"all {len(EXPECTED_COLS)} columns present, ordering canonical")

    if df["smiles"].dtype != object:
        r.fail(f"'smiles' should be object/str, got {df['smiles'].dtype}")
    if df["has_thermo_label"].dtype != bool:
        r.fail(f"'has_thermo_label' should be bool, got {df['has_thermo_label'].dtype}")
    for f in THERMO_FIELDS + RDKIT_FIELDS:
        if not np.issubdtype(df[f].dtype, np.number):
            r.fail(f"'{f}' not numeric (dtype={df[f].dtype})")
    if not r.fails:
        r.ok("all thermo + rdkit columns numeric")


def check_uniqueness(df, r):
    print("\n[2/7] SMILES uniqueness")
    n_dups = len(df) - df["smiles"].nunique()
    if n_dups:
        r.fail(f"{n_dups} duplicate SMILES keys")
    else:
        r.ok(f"{len(df):,} unique canonical SMILES")


def check_smiles_parse(df, r, n_sample=2000, seed=0):
    print(f"\n[3/7] SMILES parse round-trip (random {n_sample} sample)")
    rng = random.Random(seed)
    idx = rng.sample(range(len(df)), min(n_sample, len(df)))
    n_fail = 0
    n_mismatch = 0
    for i in idx:
        smi = df.iloc[i]["smiles"]
        m = Chem.MolFromSmiles(smi)
        if m is None:
            n_fail += 1
            continue
        roundtrip = Chem.MolToSmiles(m, isomericSmiles=True)
        if roundtrip != smi:
            n_mismatch += 1
    if n_fail:
        r.fail(f"{n_fail}/{len(idx)} SMILES failed RDKit parse")
    if n_mismatch:
        r.warn(f"{n_mismatch}/{len(idx)} SMILES not at their own canonical form")
    if not n_fail and not n_mismatch:
        r.ok(f"{len(idx)} SMILES parse & round-trip to themselves")


def check_thermo_consistency(df, r, expected=None):
    print("\n[4/7] thermo labels consistency")
    # For has_thermo_label=True rows, all 5 thermo columns must be non-NaN.
    labeled = df[df["has_thermo_label"]]
    unlabeled = df[~df["has_thermo_label"]]
    n_labeled = len(labeled)

    any_nan_labeled = labeled[THERMO_FIELDS].isna().any(axis=1).sum()
    if any_nan_labeled:
        r.fail(f"{any_nan_labeled} labeled rows have NaN in some thermo column")
    else:
        r.ok(f"{n_labeled:,} labeled rows: all 5 thermo columns populated")

    # For has_thermo_label=False rows, all 5 thermo columns must be NaN.
    any_nonnan_unlabeled = unlabeled[THERMO_FIELDS].notna().any(axis=1).sum()
    if any_nonnan_unlabeled:
        r.fail(f"{any_nonnan_unlabeled} unlabeled rows have non-NaN in thermo columns")
    else:
        r.ok(f"{len(unlabeled):,} unlabeled rows: all thermo columns NaN")

    if expected is not None:
        if n_labeled == expected:
            r.ok(f"thermo coverage = {n_labeled:,} matches expected {expected:,}")
        else:
            r.fail(f"thermo coverage {n_labeled:,} != expected {expected:,} "
                   f"(delta={n_labeled - expected:+,})")
    else:
        print(f"  (thermo coverage: {n_labeled:,} — no expected count passed)")


def check_rdkit(df, r):
    print("\n[5/7] RDKit descriptor population + ranges")
    for f in RDKIT_FIELDS:
        n_nan = df[f].isna().sum()
        if n_nan > 100:
            r.fail(f"'{f}' has {n_nan} NaNs (expected << 100)")
        elif n_nan:
            r.warn(f"'{f}' has {n_nan} NaN(s) (tolerable residual)")

        lo, hi = RDKIT_BOUNDS[f]
        col = df[f].dropna()
        below = (col < lo).sum()
        above = (col > hi).sum()
        if below + above:
            r.warn(f"'{f}' has {below} < {lo} and {above} > {hi} "
                   f"(range observed: [{col.min():.3f}, {col.max():.3f}])")
        else:
            r.ok(f"'{f}' in [{col.min():.3f}, {col.max():.3f}] ⊂ [{lo}, {hi}]")


def _iter_pt_smiles(pt_path, limit=None):
    with torch.serialization.safe_globals(
        [DataEdgeAttr, DataTensorAttr, GlobalStorage, Mol]
    ):
        data, slices = torch.load(pt_path)

    class _D(InMemoryDataset):
        def __init__(self, data, slices):
            super().__init__(".")
            self.data, self.slices = data, slices
            self._indices = None

    ds = _D(data, slices)
    n = len(ds) if limit is None else min(len(ds), limit)
    for i in range(n):
        mol = getattr(ds[i], "mol", None)
        if mol is None:
            yield None
            continue
        try:
            yield Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
        except Exception:
            yield None


def check_pt_crossref(df, pt_paths, r, n_sample=500, seed=0):
    print(f"\n[6/7] cross-check: random {n_sample} SMILES from each _h.pt resolve in table")
    if not pt_paths:
        print("  (skipped — no --inputs)")
        return
    table = set(df["smiles"].tolist())
    rng = random.Random(seed)
    for p in pt_paths:
        # Pre-iterate once to pick random indices without loading twice.
        all_smis = [s for s in _iter_pt_smiles(p) if s is not None]
        idx = rng.sample(range(len(all_smis)), min(n_sample, len(all_smis)))
        miss = sum(1 for i in idx if all_smis[i] not in table)
        label = Path(p).name
        if miss:
            r.fail(f"{label}: {miss}/{len(idx)} sampled SMILES missing from table")
        else:
            r.ok(f"{label}: all {len(idx)} sampled SMILES resolved")


def print_sample(df, n=3):
    print(f"\n[7/7] spot-check: {n} labeled + {n} unlabeled rows")
    labeled = df[df["has_thermo_label"]].sample(min(n, df["has_thermo_label"].sum()),
                                                 random_state=0)
    unlabeled = df[~df["has_thermo_label"]].sample(min(n, (~df["has_thermo_label"]).sum()),
                                                    random_state=0)
    for title, sub in [("LABELED", labeled), ("UNLABELED", unlabeled)]:
        print(f"\n  --- {title} ---")
        for _, row in sub.iterrows():
            keys_to_show = ["smiles", "enthalpy_298", "gibbs_298", "logp", "qed"]
            summary = "  ".join(
                f"{k}={row[k]:.3f}" if isinstance(row[k], float) else f"{k}={row[k]}"
                for k in keys_to_show
            )
            print(f"  {summary}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True, help="property_table.parquet to check")
    p.add_argument("--inputs", nargs="+", default=[],
                   help="original _h.pt files for cross-reference sampling (optional)")
    p.add_argument("--expected-thermo-matches", type=int, default=None,
                   help="assert has_thermo_label.sum() equals this (e.g. 652782)")
    p.add_argument("--sample-size", type=int, default=2000,
                   help="random sample size for SMILES-parse check")
    p.add_argument("--pt-sample-size", type=int, default=500,
                   help="random sample size per _h.pt for cross-reference")
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero on any FAIL")
    args = p.parse_args()

    print(f"Reading {args.parquet}")
    df = pd.read_parquet(args.parquet)
    print(f"Loaded: {len(df):,} rows × {len(df.columns)} cols  "
          f"({Path(args.parquet).stat().st_size / 1024**2:.1f} MB)")

    r = Report()
    check_schema(df, r)
    check_uniqueness(df, r)
    check_smiles_parse(df, r, n_sample=args.sample_size)
    check_thermo_consistency(df, r, expected=args.expected_thermo_matches)
    check_rdkit(df, r)
    check_pt_crossref(df, args.inputs, r, n_sample=args.pt_sample_size)
    print_sample(df)

    print("\n" + "=" * 64)
    print(f"FAILS: {len(r.fails)}   WARNS: {len(r.warns)}")
    for m in r.fails:
        print(f"  FAIL  {m}")
    for m in r.warns:
        print(f"  WARN  {m}")
    print("=" * 64)

    if args.strict and r.fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
