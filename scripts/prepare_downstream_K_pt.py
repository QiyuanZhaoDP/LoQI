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
    K = args.n_confs
    if n_total % K != 0:
        raise SystemExit(
            f"pickle has {n_total} conformers, not divisible by K={K}. "
            "If --postprocess optimization+irmsd was used, K is no longer "
            "fixed — rerun upstream with plain --postprocess optimization."
        )
    n_input = n_total // K
    print(f"  {n_total:,} conformers = {n_input:,} mols × K={K}")

    # --- Load target CSV -------------------------------------------------
    df = pd.read_csv(args.target_csv)
    if len(df) != n_input:
        raise SystemExit(
            f"CSV has {len(df)} rows but pickle implies {n_input} input mols. "
            "Did the upstream sampling skip rows? Re-run sample_conformers.py "
            "with the same .smi (no filtering) so positions stay in sync."
        )

    smiles_list = df[args.smiles_col].tolist()
    targets_raw = df[args.target_col].tolist()

    # --- Build Data list -------------------------------------------------
    print("Building PyG Data list...")
    data_list = []
    n_skipped = 0
    for i in range(n_input):
        smi = str(smiles_list[i])
        t_raw = targets_raw[i]
        has_target = not (t_raw is None
                          or (isinstance(t_raw, float) and math.isnan(t_raw)))
        target_val = float(t_raw) if has_target else float("nan")
        for k in range(K):
            mol = mols[i * K + k]
            if mol is None or mol.GetNumAtoms() == 0:
                n_skipped += 1
                continue
            try:
                data = _mol_to_data(mol)
            except Exception as e:
                n_skipped += 1
                continue
            data.input_id = smi          # group key for ensemble splitting
            data.smiles = smi             # for debugging / reporting
            data.conf_idx = int(k)        # 0..K-1, useful for analysis
            data.target = torch.tensor([target_val], dtype=torch.float32)
            data.has_target = torch.tensor([bool(has_target)], dtype=torch.bool)
            if args.keep_energies and energies is not None:
                data.aimnet2_energy_eV = torch.tensor(
                    [float(energies[i * K + k])], dtype=torch.float32
                )
            data_list.append(data)

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
