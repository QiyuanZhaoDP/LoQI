"""Build a per-molecule property table from chembl3d_stereo.

Combines:
  1. TCIT thermochemistry labels (5 targets: Hf_0, Hf_298, Gf_298, Cv, S°)
  2. RDKit 2D descriptors (9, all non-additive / intensive):
       LogP, TPSA, NumHDonors, NumHAcceptors, NumRotatableBonds,
       FractionCSP3, NumAliphaticRings, QED, LabuteASA

Keyed by RDKit canonical implicit-H SMILES. SMILES matching exactly
mirrors the retired scripts/label_thermo.py chain (direct, via neutralization
index alias, fallback to data.smiles), so the count of labeled molecules
after attach-at-load-time reproduces the retired per-split labeling exactly
(~641k matches total across train/val/test).

Output format: one parquet file with columns
  smiles, has_thermo_label,
  enthalpy_0, enthalpy_298, gibbs_298, cv_gas, entropy_gas,
  logp, tpsa, n_h_donors, n_h_acceptors, n_rot_bonds,
  frac_csp3, n_aliph_rings, qed, labute_asa
Unlabeled rows carry NaN for thermo targets (RDKit values always populated).

Usage:
  python data_processing/build_property_table.py \\
      --inputs data/chembl3d_stereo/processed/train_h.pt \\
               data/chembl3d_stereo/processed/val_h.pt \\
               data/chembl3d_stereo/processed/test_h.pt \\
      --thermo-csv data_processing/tcit_thermo_labels.csv \\
      --neutralization-index data_processing/chembl3d_neutralization_index.json \\
      --output data/property_table.parquet
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.rdchem import Mol
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from tqdm import tqdm

# Fail fast on missing parquet engine — the processing loop takes tens of
# minutes, so we want to know upfront that the final write will succeed.
try:
    import pyarrow  # noqa: F401
except ImportError:
    try:
        import fastparquet  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "No parquet engine found — install one before running this "
            "script so the ~30-min processing loop isn't wasted:\n"
            "    pip install pyarrow\n"
            "(AttachProperties at training time also needs pyarrow.)"
        ) from e

RDLogger.DisableLog("rdApp.*")


# Thermo CSV column → our output column
THERMO_COLS = {
    "Hf_0_kJmol":   "enthalpy_0",
    "Hf_298_kJmol": "enthalpy_298",
    "Gf_298_kJmol": "gibbs_298",
    "Cv_gas_JmolK": "cv_gas",
    "S0_gas_JmolK": "entropy_gas",
}
THERMO_FIELDS = list(THERMO_COLS.values())

RDKIT_FIELDS = [
    "logp", "tpsa", "n_h_donors", "n_h_acceptors", "n_rot_bonds",
    "frac_csp3", "n_aliph_rings", "qed", "labute_asa",
]


class _TempDataset(InMemoryDataset):
    def __init__(self, data, slices):
        super().__init__(".")
        self.data, self.slices = data, slices
        self._indices = None


def canonical(smi):
    """Canonical isomeric implicit-H SMILES, or None."""
    m = Chem.MolFromSmiles(smi)
    return None if m is None else Chem.MolToSmiles(m, isomericSmiles=True)


def canonical_from_mol(mol):
    """Canonical from a mol (strip explicit Hs first to match label_thermo)."""
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
    except Exception:
        return None


def compute_rdkit_descriptors(mol_impl):
    """Compute 9 RDKit 2D descriptors from an implicit-H mol. Returns NaN
    dict on exception (rare — usually radicals / weird valences)."""
    try:
        return {
            "logp":          float(Crippen.MolLogP(mol_impl)),
            "tpsa":          float(Descriptors.TPSA(mol_impl)),
            "n_h_donors":    int(Lipinski.NumHDonors(mol_impl)),
            "n_h_acceptors": int(Lipinski.NumHAcceptors(mol_impl)),
            "n_rot_bonds":   int(Lipinski.NumRotatableBonds(mol_impl)),
            "frac_csp3":     float(rdMolDescriptors.CalcFractionCSP3(mol_impl)),
            "n_aliph_rings": int(rdMolDescriptors.CalcNumAliphaticRings(mol_impl)),
            "qed":           float(QED.qed(mol_impl)),
            "labute_asa":    float(rdMolDescriptors.CalcLabuteASA(mol_impl)),
        }
    except Exception:
        return {k: float("nan") for k in RDKIT_FIELDS}


def _descriptor_worker_init():
    """Run once in each pool worker. Silence RDKit's logger so stderr
    doesn't flood during parallel descriptor calculation."""
    RDLogger.DisableLog("rdApp.*")


def _descriptors_from_canonical(canon):
    """Pool worker: reparse canonical SMILES + compute descriptors.
    MolFromSmiles on the isomeric canonical SMILES is equivalent to
    Chem.RemoveHs(source_mol) for descriptor purposes."""
    mol = Chem.MolFromSmiles(canon)
    if mol is None:
        return canon, {k: float("nan") for k in RDKIT_FIELDS}
    return canon, compute_rdkit_descriptors(mol)


def load_thermo_labels(csv_path):
    """Return dict canonical(smi) -> {enthalpy_0, ..., entropy_gas}."""
    labels = {}
    n_rows = 0
    n_canon_fail = 0
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            n_rows += 1
            canon = canonical(row["smiles"])
            if canon is None:
                n_canon_fail += 1
                continue
            rec = {}
            for csv_col, our_col in THERMO_COLS.items():
                v = row.get(csv_col, "")
                rec[our_col] = float(v) if v != "" else float("nan")
            labels[canon] = rec
    print(f"  {n_rows:,} CSV rows -> {len(labels):,} unique canonical SMILES "
          f"(canon_fail={n_canon_fail:,})")
    return labels


def expand_via_neutralization_index(labels, index_path):
    """Add ionic-form aliases that point at their neutral form's thermo record.
    Mirrors label_thermo.py's expand_labels_via_index exactly."""
    with open(index_path) as f:
        idx = json.load(f)
    n_added = 0
    n_neutral_unlabeled = 0
    for chembl_canon, neutral_canon in idx.items():
        if chembl_canon in labels:
            continue
        rec = labels.get(neutral_canon)
        if rec is None:
            n_neutral_unlabeled += 1
            continue
        labels[chembl_canon] = rec
        n_added += 1
    print(f"  {len(idx):,} index entries -> +{n_added:,} ionic aliases "
          f"(neutral_unlabeled={n_neutral_unlabeled:,})")
    return n_added


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help="List of *_h.pt files (train/val/test) to source SMILES from.")
    p.add_argument("--thermo-csv", required=True,
                   help="TCIT thermo labels CSV (from data_processing/parse_tcit_log.py).")
    p.add_argument("--neutralization-index", default=None,
                   help="JSON from build_neutralization_index.py (recommended).")
    p.add_argument("--output", required=True,
                   help="Output parquet path.")
    p.add_argument("--workers", type=int, default=None,
                   help="CPU processes for RDKit descriptor calculation "
                        "(default: os.cpu_count(); set 1 to force serial).")
    p.add_argument("--chunk-size", type=int, default=500,
                   help="Pool imap chunksize (default 500).")
    args = p.parse_args()

    print(f"Loading TCIT labels from {args.thermo_csv}")
    labels = load_thermo_labels(args.thermo_csv)

    if args.neutralization_index:
        print(f"Expanding via {args.neutralization_index}")
        expand_via_neutralization_index(labels, args.neutralization_index)
    print(f"Label lookup dict: {len(labels):,} keys")

    # ---- Phase 1: serial pass over all Data objects ------------------------
    # Builds the per-canonical thermo-match record. Must stay serial because
    # `labels` is mutated mid-loop (the fallback path registers aliases so
    # later lookups hit directly), and this mutation is what makes the
    # matched count match label_thermo.py's retired behaviour.
    canon_to_thermo: dict[str, dict | None] = {}
    n_total_rows = 0
    n_rows_matched = 0
    n_canon_fail = 0

    for pt_path in args.inputs:
        print(f"\nIterating {pt_path}")
        with torch.serialization.safe_globals(
            [DataEdgeAttr, DataTensorAttr, GlobalStorage, Mol]
        ):
            data, slices = torch.load(pt_path)
        ds = _TempDataset(data, slices)

        for i in tqdm(range(len(ds)), desc=Path(pt_path).name):
            n_total_rows += 1
            d = ds[i]
            mol = getattr(d, "mol", None)

            canon = canonical_from_mol(mol)
            if canon is None:
                raw = getattr(d, "smiles", None)
                if isinstance(raw, str):
                    canon = canonical(raw)
            if canon is None:
                n_canon_fail += 1
                continue

            # Thermo lookup matching label_thermo.py's chain exactly:
            #   primary: canonical_from_mol(mol)
            #   fallback: canonical(data.smiles)
            thermo_rec = labels.get(canon)
            if thermo_rec is None:
                raw = getattr(d, "smiles", None)
                if isinstance(raw, str):
                    fb = canonical(raw)
                    if fb is not None and fb != canon:
                        thermo_rec = labels.get(fb)
                        if thermo_rec is not None:
                            # Register the primary canonical so future
                            # lookups at load time pick up the record.
                            labels[canon] = thermo_rec

            if thermo_rec is not None:
                n_rows_matched += 1

            # First time we see this canonical → record thermo payload.
            if canon not in canon_to_thermo:
                canon_to_thermo[canon] = thermo_rec

    unique_canons = list(canon_to_thermo.keys())
    print("\n" + "=" * 64)
    print(f"Phase 1: {n_total_rows:,} Data objects  →  "
          f"{len(unique_canons):,} unique canonical SMILES")
    print(f"  canonicalization failed: {n_canon_fail:,}")
    print(f"  thermo matched:          {n_rows_matched:,}  "
          f"(count AttachProperties will reproduce at load time)")
    print("=" * 64)

    # ---- Phase 2: parallel RDKit descriptor calculation --------------------
    n_workers = args.workers or max(1, (os.cpu_count() or 1))
    n_workers = min(n_workers, len(unique_canons))
    n_rdkit_fail = 0
    descriptors: dict[str, dict] = {}
    print(f"\nPhase 2: computing 9 RDKit descriptors on {n_workers} worker(s)")

    if n_workers <= 1:
        for canon in tqdm(unique_canons, desc="descriptors"):
            _, rdkit_vals = _descriptors_from_canonical(canon)
            descriptors[canon] = rdkit_vals
            if all(isinstance(v, float) and v != v for v in rdkit_vals.values()):
                n_rdkit_fail += 1
    else:
        # Use spawn context so workers don't inherit the (large) `labels`
        # dict + `ds` tensors via fork — cleaner memory, faster startup.
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers, initializer=_descriptor_worker_init) as pool:
            for canon, rdkit_vals in tqdm(
                pool.imap_unordered(_descriptors_from_canonical,
                                    unique_canons,
                                    chunksize=args.chunk_size),
                total=len(unique_canons),
                desc="descriptors",
            ):
                descriptors[canon] = rdkit_vals
                if all(isinstance(v, float) and v != v for v in rdkit_vals.values()):
                    n_rdkit_fail += 1
    print(f"  RDKit descriptor failures: {n_rdkit_fail:,}")

    # ---- Phase 3: join + write parquet -------------------------------------
    rows = []
    for canon in unique_canons:
        thermo_rec = canon_to_thermo[canon]
        row = {"smiles": canon, "has_thermo_label": thermo_rec is not None}
        row.update(thermo_rec or {k: float("nan") for k in THERMO_FIELDS})
        row.update(descriptors.get(canon, {k: float("nan") for k in RDKIT_FIELDS}))
        rows.append(row)

    df = pd.DataFrame(rows)
    cols = (["smiles", "has_thermo_label"] + THERMO_FIELDS + RDKIT_FIELDS)
    df = df[cols]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\nSaved {len(df):,} rows to {out}")
    print(f"File size: {out.stat().st_size / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()
