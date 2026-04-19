"""Build a per-dataset neutralization index.

Iterates the .pt splits of chembl3d_stereo, canonicalizes every unique
data.mol (implicit-H, isomeric), neutralizes it via RDKit's Uncharger,
and writes a JSON mapping:

    {canonical_chembl_smiles: canonical_neutral_smiles}

Why precompute: doing this once decouples the (slow, fragile) RDKit
canonicalization + neutralization step from `label_thermo.py`, which
then becomes a plain dict lookup. The index is also a complete record
of "what neutral form does each ChEMBL3D row map to" — handy for any
future joins (e.g. AIMNet2 energies).

Usage:
    python data_processing/build_neutralization_index.py \\
        --inputs data/chembl3d_stereo/processed/train_h.pt \\
                 data/chembl3d_stereo/processed/val_h.pt \\
                 data/chembl3d_stereo/processed/test_h.pt \\
        --output data_processing/chembl3d_neutralization_index.json
"""
import argparse
import json
from pathlib import Path

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdchem import Mol
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

ION_MARKERS = ("[O-]", "[N+]", "[n+]", "[S-]", "[NH+]", "[NH2+]", "[NH3+]",
               "[O+]", "[P-]", "[P+]", "[Cl-]", "[Br-]", "[I-]")


class _TempDataset(InMemoryDataset):
    def __init__(self, data, slices):
        super().__init__(".")
        self.data, self.slices = data, slices
        self._indices = None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help="List of *_h.pt files (train/val/test)")
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--limit", type=int, default=None, help="Cap rows per split (debug)")
    args = p.parse_args()

    uncharger = rdMolStandardize.Uncharger()
    seen = {}
    stats = {
        "n_rows_total": 0,
        "n_mol_none": 0,
        "n_canon_fail": 0,
        "n_uncharge_fail": 0,
        "n_unique_chembl": 0,
        "n_ionic_chembl": 0,
        "n_neutralization_changed": 0,
        "per_file": {},
    }

    for path in args.inputs:
        with torch.serialization.safe_globals(
            [DataEdgeAttr, DataTensorAttr, GlobalStorage, Mol]
        ):
            data, slices = torch.load(path)
        ds = _TempDataset(data, slices)
        n = len(ds) if args.limit is None else min(len(ds), args.limit)
        per = {"n_rows": n, "added_unique": 0}
        for i in tqdm(range(n), desc=Path(path).name):
            stats["n_rows_total"] += 1
            d = ds[i]
            mol = getattr(d, "mol", None)
            if mol is None:
                stats["n_mol_none"] += 1
                continue
            try:
                mol_impl = Chem.RemoveHs(mol)
                canon_chembl = Chem.MolToSmiles(mol_impl, isomericSmiles=True)
            except Exception:
                stats["n_canon_fail"] += 1
                continue
            if canon_chembl in seen:
                continue
            try:
                neutral = uncharger.uncharge(mol_impl)
                canon_neutral = Chem.MolToSmiles(neutral, isomericSmiles=True)
            except Exception:
                stats["n_uncharge_fail"] += 1
                continue
            seen[canon_chembl] = canon_neutral
            per["added_unique"] += 1
            stats["n_unique_chembl"] += 1
            if canon_chembl != canon_neutral:
                stats["n_neutralization_changed"] += 1
            if any(m in canon_chembl for m in ION_MARKERS):
                stats["n_ionic_chembl"] += 1
        stats["per_file"][path] = per

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(seen, f)
    with open(str(out_path) + ".stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    print(f"\nSaved index ({len(seen):,} unique entries) -> {out_path}")


if __name__ == "__main__":
    main()
