"""Build a PyG dataset (.pt) from a K-conformer sampling output joined with
a downstream property-prediction CSV.

Each input SMILES becomes K Data entries (one per sampled conformer); they
share the same `input_id`, `target`, and `has_target`. Downstream consumers
(scripts/downstream_cv.py with --ensemble-by input_id) then 5-fold split
by input_id and average predictions across the K conformers per input.

Pipeline:
    1. SMILES list (extract the smiles column from your CSV → .smi)
    2. scripts/sample_conformers.py  --input X.smi  --output X_kK.pkl
       --n_confs K  --postprocess optimization      (~30-60 min / 1k mols)
    3. THIS SCRIPT: pickle + CSV → PyG dataset .pt  (~seconds)
    4. scripts/downstream_cv.py --dataset-pt X_kK.pt --ensemble-by input_id

The pickle from sample_conformers.py contains keys:
    generated:  list[RDKit Mol]  length N*K, ordered [m0_c0, m0_c1, ...
                m0_cK-1, m1_c0, ...]
    ids:        list[str]        length N*K, "NA" if input had no _Name
                (we don't rely on this — we use position to map back to CSV)
    energies:   np.ndarray       length N*K, optional (only with
                --postprocess optimization)

Usage:
    python scripts/prepare_downstream_K_pt.py \\
        --conformer-pkl data/downstream_k5/esol_k5.pkl \\
        --target-csv data/downstream/esol.csv \\
        --target-col "measured log solubility in mols per litre" \\
        --smiles-col smiles \\
        --n-confs 5 \\
        --output data/downstream_pt/esol_k5.pt
"""
from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data.collate import collate

# Reuse conversion utilities from prepare_downstream_dataset.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_downstream_dataset import _fallback_mol_to_data, HAVE_UTIL  # noqa: E402
if HAVE_UTIL:
    from utils_data import mol_to_pyg_data  # noqa: E402


def _mol_to_data(mol):
    return mol_to_pyg_data(mol) if HAVE_UTIL else _fallback_mol_to_data(mol)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conformer-pkl", required=True,
                   help="Pickle from scripts/sample_conformers.py")
    p.add_argument("--target-csv", required=True,
                   help="Original CSV — row order MUST match the SMILES list "
                        "fed to sample_conformers.py.")
    p.add_argument("--smiles-col", default="smiles")
    p.add_argument("--target-col", required=True)
    p.add_argument("--n-confs", type=int, required=True,
                   help="K used during sampling (must match)")
    p.add_argument("--output", required=True)
    p.add_argument("--keep-energies", action="store_true",
                   help="If pickle has energies, store them on each Data "
                        "as `data.aimnet2_energy_eV` (handy for analysis).")
    args = p.parse_args()

    # --- Load pickle -----------------------------------------------------
    print(f"Loading {args.conformer_pkl}")
    with open(args.conformer_pkl, "rb") as f:
        d = pickle.load(f)
    mols = d["generated"]
    energies = d.get("energies", None)
    n_total = len(mols)
    print(f"  {n_total:,} conformers in pickle")

    # --- Load target CSV (no row-count assertion — we now join by SMILES) ---
    df = pd.read_csv(args.target_csv)
    print(f"  {len(df):,} rows in target CSV")
    smiles_list = df[args.smiles_col].astype(str).tolist()
    targets_raw = df[args.target_col].tolist()

    # --- Group pickle conformers by canonical SMILES ------------------------
    # sample_conformers.py may silently drop molecules (radicals, disconnected,
    # unsupported elements, etc.), which used to break our position-based
    # join with a single assertion. Switching to SMILES-based join makes us
    # robust to any upstream filtering + lets us emit a clean per-row report
    # at the end ("N CSV rows had no conformers in the pickle").
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    def _canon(smi):
        m = Chem.MolFromSmiles(smi)
        return None if m is None else Chem.MolToSmiles(m, isomericSmiles=True)

    by_canon: dict[str, list] = {}
    by_canon_energy: dict[str, list] = {}
    for j, mol in enumerate(mols):
        if mol is None:
            continue
        try:
            canon = Chem.MolToSmiles(mol, isomericSmiles=True)
        except Exception:
            continue
        by_canon.setdefault(canon, []).append(mol)
        if energies is not None:
            by_canon_energy.setdefault(canon, []).append(float(energies[j]))
    print(f"  unique canonical SMILES with conformers in pickle: {len(by_canon):,}")

    # --- Build Data list, joining by canonical SMILES -----------------------
    print("Building PyG Data list...")
    data_list = []
    n_skipped_mol = 0
    n_csv_rows_no_conformer = 0
    K_used = []                      # actual K observed per input mol
    for i, smi in enumerate(smiles_list):
        canon = _canon(smi)
        if canon is None or canon not in by_canon:
            n_csv_rows_no_conformer += 1
            continue

        t_raw = targets_raw[i]
        has_target = not (t_raw is None
                          or (isinstance(t_raw, float) and math.isnan(t_raw)))
        target_val = float(t_raw) if has_target else float("nan")

        group_mols = by_canon[canon]
        K_used.append(len(group_mols))
        group_energies = by_canon_energy.get(canon, [])
        for k, mol in enumerate(group_mols):
            if mol.GetNumAtoms() == 0:
                n_skipped_mol += 1
                continue
            try:
                data = _mol_to_data(mol)
            except Exception:
                n_skipped_mol += 1
                continue
            data.input_id = canon         # group key for ensemble splitting
            data.smiles = smi              # original (un-canonicalized) form
            data.conf_idx = int(k)
            data.target = torch.tensor([target_val], dtype=torch.float32)
            data.has_target = torch.tensor([bool(has_target)], dtype=torch.bool)
            if args.keep_energies and group_energies:
                data.aimnet2_energy_eV = torch.tensor(
                    [group_energies[k] if k < len(group_energies) else float("nan")],
                    dtype=torch.float32,
                )
            data_list.append(data)

    if K_used:
        import statistics as _stats
        print(f"  K observed per input: median={int(_stats.median(K_used))}, "
              f"mean={sum(K_used)/len(K_used):.2f}, "
              f"min={min(K_used)}, max={max(K_used)}")
    if n_csv_rows_no_conformer:
        print(f"  CSV rows with no matching pickle conformer: "
              f"{n_csv_rows_no_conformer:,} (dropped — likely filtered by sampler)")

    print(f"  built {len(data_list):,} Data ({n_skipped} skipped)")
    n_unique_inputs = len({d.input_id for d in data_list})
    print(f"  unique input SMILES retained: {n_unique_inputs:,} / {n_input:,}")

    # --- Save as InMemoryDataset -----------------------------------------
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    data, slices, _ = collate(
        data_list[0].__class__,
        data_list=data_list,
        increment=False,
        add_batch=False,
    )
    torch.save((data, slices), out)
    print(f"  saved -> {out}  ({out.stat().st_size / 1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
