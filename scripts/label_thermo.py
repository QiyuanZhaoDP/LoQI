"""Attach TCIT thermochemistry labels to a ChEMBL3D .pt dataset.

Pipeline:
    1. Precompute a neutralization index ONCE:
           python data_processing/build_neutralization_index.py \\
               --inputs  data/chembl3d_stereo/processed/{train,val,test}_h.pt \\
               --output  data_processing/chembl3d_neutralization_index.json

    2. Join TCIT labels to each split (this script):
           python scripts/label_thermo.py \\
               --input  data/chembl3d_stereo/processed/train_h.pt \\
               --output data/chembl3d_stereo/processed/train_h_thermo.pt \\
               --labels data_processing/tcit_thermo_labels.csv \\
               --neutralization-index data_processing/chembl3d_neutralization_index.json

All RDKit neutralization/canonicalization work is done once in step 1;
step 2 is just a dict lookup per row, so it stays fast and predictable.

Output fields on each Data object (NaN + thermo_has_label=False on miss):
    enthalpy_298  [1]  kJ/mol   Hf_298
    gibbs_298     [1]  kJ/mol   Gf_298
    cv_gas        [1]  J/(mol*K) Cv
    entropy_gas   [1]  J/(mol*K) S0       (bonus)
    enthalpy_0    [1]  kJ/mol    Hf_0     (bonus)
    thermo_has_label [1] bool
"""
import argparse
import csv
import json
import math
from pathlib import Path

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import Mol
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.collate import collate
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

FIELD_MAP = {
    "enthalpy_298": "Hf_298_kJmol",
    "gibbs_298":    "Gf_298_kJmol",
    "cv_gas":       "Cv_gas_JmolK",
    "entropy_gas":  "S0_gas_JmolK",
    "enthalpy_0":   "Hf_0_kJmol",
}


def canonical_impl_h(smi):
    """Canonical isomeric SMILES with implicit Hs, or None."""
    mol = Chem.MolFromSmiles(smi)
    return None if mol is None else Chem.MolToSmiles(mol, isomericSmiles=True)


def canonical_from_mol(mol):
    """Canonical isomeric SMILES from a Mol (strip explicit Hs)."""
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
    except Exception:
        return None


def load_labels(csv_path):
    """Load TCIT labels keyed by canonical (implicit-H, isomeric) SMILES."""
    labels = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            canon = canonical_impl_h(row["smiles"])
            if canon is None:
                continue
            rec = {
                k: (float(row[c]) if row[c] != "" else float("nan"))
                for k, c in FIELD_MAP.items()
            }
            if canon in labels:
                existing_n = sum(not math.isnan(v) for v in labels[canon].values())
                new_n = sum(not math.isnan(v) for v in rec.values())
                if new_n <= existing_n:
                    continue
            labels[canon] = rec
    print(f"  labels: {len(labels):,} unique canonical SMILES")
    return labels


def expand_labels_via_index(labels, index_path):
    """Precomputed index {canonical_chembl: canonical_neutral} — alias each
    ChEMBL3D-form canonical to its neutral form's thermo record (if labeled).
    Returns alias_source dict {canon: "direct"|"via_index"}.
    """
    with open(index_path) as f:
        index = json.load(f)
    alias_source = {k: "direct" for k in labels}
    n_added = 0
    n_neutral_unlabeled = 0
    for chembl_canon, neutral_canon in index.items():
        if chembl_canon in labels:
            continue
        rec = labels.get(neutral_canon)
        if rec is None:
            n_neutral_unlabeled += 1
            continue
        labels[chembl_canon] = rec
        alias_source[chembl_canon] = "via_index"
        n_added += 1
    print(f"  index: {len(index):,} entries -> "
          f"{n_added:,} new aliases "
          f"(neutral_unlabeled={n_neutral_unlabeled:,})")
    return alias_source


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--labels", required=True, help="TCIT labels CSV")
    p.add_argument("--neutralization-index", default=None,
                   help="JSON precomputed by data_processing/build_neutralization_index.py")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--debug-mismatches", type=int, default=0)
    args = p.parse_args()

    print(f"Loading labels from {args.labels}")
    labels = load_labels(args.labels)
    if args.neutralization_index:
        print(f"Expanding via index {args.neutralization_index}")
        alias_source = expand_labels_via_index(labels, args.neutralization_index)
    else:
        alias_source = {k: "direct" for k in labels}

    print(f"Loading dataset from {args.input}")
    with torch.serialization.safe_globals(
        [DataEdgeAttr, DataTensorAttr, GlobalStorage, Mol]
    ):
        data_obj, slices = torch.load(args.input)

    class _TempDataset(InMemoryDataset):
        def __init__(self, d, s):
            super().__init__(".")
            self.data, self.slices = d, s
            self._indices = None

    temp_ds = _TempDataset(data_obj, slices)
    n_molecules = len(temp_ds)
    if args.limit:
        n_molecules = min(n_molecules, args.limit)
    print(f"Dataset contains {len(temp_ds):,} molecules  "
          f"(processing {n_molecules:,})")

    n_direct = 0
    n_via_index = 0
    n_via_smiles = 0
    n_miss = 0
    miss_samples = []

    data_list = []
    for i in tqdm(range(n_molecules), desc="Joining labels"):
        d = temp_ds[i]
        record = None
        source = None
        attempted = []

        mol = getattr(d, "mol", None)
        canon = canonical_from_mol(mol)
        if canon is not None:
            attempted.append(("mol", canon))
            record = labels.get(canon)
            if record is not None:
                source = alias_source.get(canon, "direct")

        if record is None:
            raw = getattr(d, "smiles", None)
            if isinstance(raw, str):
                c2 = canonical_impl_h(raw)
                if c2 is not None:
                    attempted.append(("smiles", c2))
                    record = labels.get(c2)
                    if record is not None:
                        source = "via_smiles"

        if record is None:
            has_label = False
            record = {k: float("nan") for k in FIELD_MAP}
            n_miss += 1
            if len(miss_samples) < args.debug_mismatches:
                miss_samples.append((i, attempted))
        else:
            has_label = True
            if source == "via_index":
                n_via_index += 1
            elif source == "via_smiles":
                n_via_smiles += 1
            else:
                n_direct += 1

        for field, value in record.items():
            setattr(d, field, torch.tensor([value], dtype=torch.float32))
        d.thermo_has_label = torch.tensor([has_label], dtype=torch.bool)
        data_list.append(d)

    if miss_samples:
        print("\nFirst miss samples:")
        for idx, atts in miss_samples:
            print(f"  [{idx}]  " + "  |  ".join(f"{s}: {v}" for s, v in atts))

    n_hit = n_direct + n_via_index + n_via_smiles
    print(f"\nLabel hit rate: {n_hit:,}/{n_molecules:,} "
          f"({100*n_hit/max(n_molecules,1):.1f}%)  "
          f"direct={n_direct:,}  via_index={n_via_index:,}  "
          f"via_smiles={n_via_smiles:,}  miss={n_miss:,}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving labeled dataset to {out_path}")
    collated = collate(
        data_list[0].__class__, data_list,
        increment=False, add_batch=False,
    )
    torch.save(collated[:2], out_path)
    print("Done.")


if __name__ == "__main__":
    main()
