"""PyG transform that attaches per-molecule properties from a pre-built
parquet table (data_processing/build_property_table.py) to each Data object
at load time — so the original `{split}_h.pt` files stay untouched and the
property table can be swapped / extended independently.

Fields attached on each Data (all shape [1]):
    enthalpy_0, enthalpy_298, gibbs_298, cv_gas, entropy_gas    (thermo — may be NaN)
    logp, tpsa, n_h_donors, n_h_acceptors, n_rot_bonds,
    frac_csp3, n_aliph_rings, qed, labute_asa                   (RDKit — always populated)
    thermo_has_label:  bool
    has_properties:    bool   (True iff the SMILES was found in the table)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


THERMO_FIELDS = ["enthalpy_0", "enthalpy_298", "gibbs_298", "cv_gas", "entropy_gas"]
RDKIT_FIELDS = [
    "logp", "tpsa", "n_h_donors", "n_h_acceptors", "n_rot_bonds",
    "frac_csp3", "n_aliph_rings", "qed", "labute_asa",
]
ALL_PROPERTY_FIELDS = THERMO_FIELDS + RDKIT_FIELDS


def _canonical_from_mol(mol):
    """Canonical isomeric implicit-H SMILES from an RDKit Mol (strips Hs).
    Must match data_processing/build_property_table.py's convention."""
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
    except Exception:
        return None


def _canonical_from_smi(smi):
    if not isinstance(smi, str):
        return None
    m = Chem.MolFromSmiles(smi)
    return None if m is None else Chem.MolToSmiles(m, isomericSmiles=True)


class AttachProperties:
    """Callable PyG transform. Instantiate once per run, pass to PyG
    datasets via `transform=...`.

    Args:
        table_path: parquet file produced by build_property_table.py.
        fallback_smiles: if True (default), fall back to `data.smiles`
            when `data.mol` canonicalization fails or the primary key
            isn't in the table — mirrors label_thermo.py's behaviour.
    """

    def __init__(self, table_path: str | Path, fallback_smiles: bool = True):
        import pandas as pd
        df = pd.read_parquet(str(table_path))
        # Build dict of dicts keyed by canonical SMILES. set_index + to_dict
        # is fast for ~1.85M rows (<10 s and ~1 GB RAM).
        self._table = df.set_index("smiles").to_dict(orient="index")
        self._fallback_smiles = fallback_smiles

        # Pre-built NaN record for misses.
        self._nan_record = {k: float("nan") for k in ALL_PROPERTY_FIELDS}
        self._nan_record["has_thermo_label"] = False

    def __len__(self):
        return len(self._table)

    def __call__(self, data):
        canon = _canonical_from_mol(getattr(data, "mol", None))
        rec = self._table.get(canon) if canon is not None else None

        if rec is None and self._fallback_smiles:
            fb = _canonical_from_smi(getattr(data, "smiles", None))
            if fb is not None:
                rec = self._table.get(fb)

        if rec is None:
            # Unknown molecule → NaN everywhere, both flags False.
            for k in ALL_PROPERTY_FIELDS:
                data[k] = torch.tensor([float("nan")], dtype=torch.float32)
            data.thermo_has_label = torch.tensor([False], dtype=torch.bool)
            data.has_properties = torch.tensor([False], dtype=torch.bool)
            return data

        for k in ALL_PROPERTY_FIELDS:
            v = rec.get(k, float("nan"))
            data[k] = torch.tensor(
                [float(v) if v is not None else float("nan")],
                dtype=torch.float32,
            )
        data.thermo_has_label = torch.tensor(
            [bool(rec.get("has_thermo_label", False))], dtype=torch.bool
        )
        data.has_properties = torch.tensor([True], dtype=torch.bool)
        return data
