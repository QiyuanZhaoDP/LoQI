"""Build a PyG dataset (.pt) for a downstream property-prediction CSV.

Input:
    CSV with at least a SMILES column and a target column.

Output:
    <out>.pt containing a collated PyG dataset in the same format the
    LoQI pipeline expects:
        data.mol         RDKit Mol with explicit Hs + an embedded conformer
        data.smiles      original SMILES string
        data.target      float scalar (the property)
        data.has_target  bool (True if target parsed; False if NaN in CSV)

The prepared .pt is consumed by scripts/downstream_cv.py, which takes H
extraction through the frozen LoQI backbone and runs 5-fold CV.

Requirements:
    rdkit-pypi (or rdkit conda), pandas, torch, torch_geometric

Usage:
    python scripts/prepare_downstream_dataset.py \\
        --csv data/downstream/delaney.csv \\
        --smiles-col smiles --target-col logSolubility \\
        --output data/downstream/delaney.pt \\
        --n-confs 1 --embed-seed 42

Notes on atom/bond encoding
---------------------------
We delegate atom-type and bond-type indexing to
`data_processing/utils_data.py` where possible; a molecule is converted
to an RDKit Mol (with explicit Hs and a 3D conformer) and downstream
extraction re-uses the same per-atom mapping that the LoQI pre-training
data went through. The resulting dataset format matches what
MoleculeDataset would load from a `_h.pt` file.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from torch_geometric.data.collate import collate
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

# Re-use existing utility if present.
sys.path.insert(0, str(Path(__file__).parent.parent / "data_processing"))
try:
    from utils_data import mol_to_pyg_data
    HAVE_UTIL = True
except Exception:
    HAVE_UTIL = False


# --- Fallback conversion (atom classes match fn_model.py expectations) ---

# Matches the ordering used by process_geom.py / process_qm9.py in this repo
# (17 inner atom types covering chembl3d_stereo). If your CSV has atoms
# outside this set (exotic metals etc.), filter them beforehand.
ATOMIC_TO_INNER = {
    1: 0,  5: 1,  6: 2,  7: 3,  8: 4,  9: 5, 13: 6, 14: 7, 15: 8,
    16: 9, 17: 10, 33: 11, 35: 12, 53: 13, 80: 14, 83: 15, 34: 16,
}
# Bond type → inner index. 0 reserved as "no bond" in fully-connected graph.
BOND_TO_INNER = {
    Chem.BondType.SINGLE:    1,
    Chem.BondType.DOUBLE:    2,
    Chem.BondType.TRIPLE:    3,
    Chem.BondType.AROMATIC:  4,
}
# Formal charge inner index (per process_geom.py encoding: 0..5 for -2..+3
# with offset=2).
def _charge_to_inner(q):
    return max(0, min(5, int(q) + 2))


def _fallback_mol_to_data(mol):
    """Convert a 3D-embedded RDKit Mol (with explicit Hs) to a PyG Data
    matching the chembl3d_stereo format."""
    conf = mol.GetConformer()
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)

    atomic_nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
    try:
        h = torch.tensor([ATOMIC_TO_INNER[z] for z in atomic_nums], dtype=torch.long)
    except KeyError as e:
        raise ValueError(f"Unsupported atomic number {e.args[0]} "
                         f"(supported: {list(ATOMIC_TO_INNER)})")
    charges = torch.tensor(
        [_charge_to_inner(a.GetFormalCharge()) for a in mol.GetAtoms()],
        dtype=torch.long,
    )

    # Edges: all bonds as directed pairs
    e_idx_src, e_idx_dst, e_attr = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        t = BOND_TO_INNER.get(bond.GetBondType(), 1)
        e_idx_src += [i, j]
        e_idx_dst += [j, i]
        e_attr    += [t, t]
    edge_index = torch.tensor([e_idx_src, e_idx_dst], dtype=torch.long)
    edge_attr  = torch.tensor(e_attr, dtype=torch.long)

    return Data(
        pos=pos,
        x=h,                     # atom type indices — BatchPreProcessor renames x -> h
        edge_index=edge_index,
        edge_attr=edge_attr,
        charges=charges,
        mol=mol,
    )


def smiles_to_data(smiles, target, n_confs=1, embed_seed=42, mmff=True):
    """Return a Data object or None if embedding fails."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = embed_seed
    try:
        confs = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
        if not list(confs):
            # Retry with random coords fallback
            params.useRandomCoords = True
            confs = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
            if not list(confs):
                return None
        if mmff:
            try:
                AllChem.MMFFOptimizeMoleculeConfs(mol)
            except Exception:
                pass
    except Exception:
        return None

    # Keep the lowest-energy conformer if multiple
    if mol.GetNumConformers() > 1:
        try:
            props = AllChem.MMFFGetMoleculeProperties(mol)
            energies = []
            for cid in range(mol.GetNumConformers()):
                ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
                energies.append((ff.CalcEnergy(), cid) if ff else (float("inf"), cid))
            best = min(energies)[1]
            # Drop all other confs
            best_conf = mol.GetConformer(best)
            new_mol = Chem.Mol(mol)
            new_mol.RemoveAllConformers()
            new_mol.AddConformer(best_conf, assignId=0)
            mol = new_mol
        except Exception:
            # keep whatever's there
            pass

    try:
        if HAVE_UTIL:
            data = mol_to_pyg_data(mol)
        else:
            data = _fallback_mol_to_data(mol)
    except Exception as e:
        print(f"  convert fail for {smiles}: {e}")
        return None

    has_tgt = not (target is None or (isinstance(target, float)
                                       and (target != target)))   # NaN check
    data.smiles = smiles
    data.target      = torch.tensor([float(target) if has_tgt else float("nan")],
                                     dtype=torch.float32)
    data.has_target  = torch.tensor([has_tgt], dtype=torch.bool)
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--smiles-col", default="smiles")
    p.add_argument("--target-col", required=True)
    p.add_argument("--output", required=True, help=".pt to write")
    p.add_argument("--n-confs", type=int, default=1, help="conformers embedded per mol")
    p.add_argument("--embed-seed", type=int, default=42)
    p.add_argument("--no-mmff", action="store_true", help="Skip MMFF optimization.")
    p.add_argument("--max-rows", type=int, default=None)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if args.max_rows:
        df = df.iloc[:args.max_rows]
    print(f"Loaded {len(df):,} rows from {args.csv}")

    data_list = []
    n_fail = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="3D-embed"):
        smi = str(row[args.smiles_col])
        tgt = row[args.target_col]
        d = smiles_to_data(smi, tgt, args.n_confs, args.embed_seed,
                            mmff=not args.no_mmff)
        if d is None:
            n_fail += 1
            continue
        data_list.append(d)
    print(f"Embedded {len(data_list):,} / {len(df):,}  (failed: {n_fail})")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    collated = collate(
        data_list[0].__class__, data_list,
        increment=False, add_batch=False,
    )
    torch.save(collated[:2], out)
    print(f"Saved dataset -> {out}")


if __name__ == "__main__":
    main()
