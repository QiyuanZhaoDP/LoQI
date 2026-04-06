"""
Label ChEMBL3D dataset with AIMNet2 energies.

Usage:
    python scripts/label_energy.py \
        --input data/chembl3d_stereo/processed/train_h.pt \
        --output data/chembl3d_stereo/processed/train_h_energy.pt \
        --aimnet2_model src/megalodon/metrics/aimnet2/cpcm_model/wb97m_cpcms_v2_0.jpt \
        --batch_size 128 \
        --device cuda
"""

import argparse
import torch
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.data.collate import collate
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from rdkit.Chem.rdchem import Mol
from tqdm import tqdm


def prepare_for_aimnet(rdkit_molecules, device="cpu"):
    """Prepare RDKit molecules for AIMNet2 batch inference."""
    coord = [mol.GetConformer().GetPositions().tolist() for mol in rdkit_molecules]
    max_n_atoms = max(len(c) for c in coord)

    coordinates = torch.zeros((len(rdkit_molecules), max_n_atoms, 3), device=device)
    atoms = torch.full((len(rdkit_molecules), max_n_atoms), 0, device=device, dtype=torch.long)
    charges = torch.tensor(
        [Chem.GetFormalCharge(mol) for mol in rdkit_molecules],
        device=device, dtype=torch.long
    )

    for idx, mol in enumerate(rdkit_molecules):
        n_atoms = len(coord[idx])
        coordinates[idx, :n_atoms] = torch.tensor(coord[idx], device=device)
        atoms[idx, :n_atoms] = torch.tensor(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()], device=device
        )

    return {"coord": coordinates, "numbers": atoms, "charge": charges}


def main():
    parser = argparse.ArgumentParser(description="Label dataset with AIMNet2 energies")
    parser.add_argument("--input", required=True, help="Path to input .pt dataset")
    parser.add_argument("--output", required=True, help="Path to output .pt dataset")
    parser.add_argument("--aimnet2_model", required=True, help="Path to AIMNet2 .jpt model")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load AIMNet2
    print(f"Loading AIMNet2 model from {args.aimnet2_model}")
    aimnet2 = torch.jit.load(args.aimnet2_model).to(device)
    aimnet2.eval()

    # Load dataset
    print(f"Loading dataset from {args.input}")
    with torch.serialization.safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage, Mol]):
        data_tuple = torch.load(args.input)
    data_obj, slices = data_tuple[0], data_tuple[1]

    # Reconstruct individual Data objects
    from torch_geometric.data import InMemoryDataset

    class _TempDataset(InMemoryDataset):
        def __init__(self, data, slices):
            super().__init__(".")
            self.data, self.slices = data, slices
            self._indices = None

    temp_ds = _TempDataset(data_obj, slices)
    n_molecules = len(temp_ds)
    print(f"Dataset contains {n_molecules} molecules")

    # Compute energies in batches
    energies = torch.zeros(n_molecules, dtype=torch.float64)
    n_failed = 0

    with torch.no_grad():
        for start in tqdm(range(0, n_molecules, args.batch_size), desc="Computing energies"):
            end = min(start + args.batch_size, n_molecules)
            batch_mols = []
            batch_indices = []

            for i in range(start, end):
                data_i = temp_ds[i]
                mol = data_i.mol
                if mol is None:
                    n_failed += 1
                    continue
                try:
                    Chem.SanitizeMol(mol)
                    if mol.GetNumConformers() == 0:
                        n_failed += 1
                        continue
                    batch_mols.append(mol)
                    batch_indices.append(i)
                except Exception:
                    n_failed += 1
                    continue

            if not batch_mols:
                continue

            aimnet_input = prepare_for_aimnet(batch_mols, device=device)
            try:
                out = aimnet2(aimnet_input)
                for j, idx in enumerate(batch_indices):
                    energies[idx] = out["energy"][j].cpu()
            except Exception as e:
                print(f"Batch {start}-{end} failed: {e}")
                n_failed += len(batch_mols)

    print(f"Computed energies for {n_molecules - n_failed}/{n_molecules} molecules")
    print(f"Energy range: [{energies.min():.4f}, {energies.max():.4f}] eV")

    # Add energy to each Data object and re-collate
    print("Adding energy labels to dataset...")
    data_list = []
    for i in tqdm(range(n_molecules), desc="Rebuilding dataset"):
        data_i = temp_ds[i]
        data_i.energy = energies[i].unsqueeze(0).float()  # [1] tensor
        data_list.append(data_i)

    print(f"Saving labeled dataset to {args.output}")
    collated = collate(
        data_list[0].__class__,
        data_list,
        increment=False,
        add_batch=False,
    )
    torch.save(collated[:2], args.output)
    print("Done.")


if __name__ == "__main__":
    main()
