"""Multi-timestep snapshot conformer sampling.

Cheap conformer ensemble: instead of running K independent flow
trajectories per molecule, run K_traj trajectories and capture multiple
snapshots from each at specified integration steps. Default
configuration captures the last 3 of 10 integration steps (steps 7, 8,
9 in 0-indexed terms = t=0.8, 0.9, 1.0), giving 3 near-clean
conformers per trajectory.

Effective conformer count = K_traj × len(snapshot_steps).

Output pickle is layout-compatible with `sample_conformers.py`'s output,
so `prepare_downstream_K_pt.py` consumes it directly. Per-input
conformer ordering:

    traj_0_step_7, traj_0_step_8, traj_0_step_9,
    traj_1_step_7, traj_1_step_8, traj_1_step_9,
    ... (K_traj × 3 mols per input)

Usage:
    python scripts/sample_conformers_multistep.py \\
        --ckpt   data/thermo_flow_warm.ckpt \\
        --config scripts/conf/loqi/loqi_thermo_flow_warm.yaml \\
        --input  data/downstream_k8/gas_Hf.smi \\
        --output data/downstream_k15/gas_Hf.pkl \\
        --n_traj 5 \\
        --n_steps 10 \\
        --snapshot_steps 7 8 9
"""
from __future__ import annotations

import pickle
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

# Make scripts/ importable so we can reuse helpers from sample_conformers.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from megalodon.data.batch_preprocessor import BatchPreProcessor
from megalodon.metrics.conformer_evaluation_callback import (
    convert_coords_to_np, write_coords_to_mol,
)
from megalodon.models.module import Graph3DInterpolantModel

# Reuse SMILES → mol → PyG-Data → DataLoader helpers from sample_conformers.py.
import sample_conformers as sc  # noqa: E402


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="One SMILES per line (.smi from extract_smiles.py).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True,
                        help="Output pickle, layout-compatible with "
                             "sample_conformers.py: dict with keys "
                             "'generated' (list[Mol]) and 'ids' (list[str]).")
    parser.add_argument("--n_traj", type=int, default=5,
                        help="How many independent trajectories to run "
                             "per input molecule.")
    parser.add_argument("--n_steps", type=int, default=10,
                        help="Total flow integration steps per trajectory.")
    parser.add_argument("--snapshot_steps", type=int, nargs="+",
                        default=[7, 8, 9],
                        help="0-indexed integration step indices at which "
                             "to capture a snapshot. With --n_steps 10, "
                             "step 9 is the clean t=1.0 endpoint.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--add-hs", action=BooleanOptionalAction,
                        default=True,
                        help="Add explicit Hs before sampling (matches "
                             "sample_conformers.py).")
    parser.add_argument("--atom-aware-batching", action=BooleanOptionalAction,
                        default=True)
    parser.add_argument("--target-molecule-size", type=int, default=50)
    args = parser.parse_args()

    if any(s < 0 or s >= args.n_steps for s in args.snapshot_steps):
        raise SystemExit(
            f"--snapshot_steps {args.snapshot_steps} must all be in "
            f"[0, {args.n_steps}). Reminder: step {args.n_steps - 1} is "
            f"the clean t=1.0 endpoint."
        )

    cfg = OmegaConf.load(args.config)
    print(f"[ckpt] {args.ckpt}")
    model = Graph3DInterpolantModel.load_from_checkpoint(
        args.ckpt,
        loss_fn=None,
        loss_params=cfg.loss,
        interpolant_params=cfg.interpolant,
        sampling_params=cfg.sample,
        batch_preprocessor=BatchPreProcessor(
            cfg.data.aug_rotations, cfg.data.scale_coords,
        ),
        strict=False,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # ---- Build the input dataset (reuse sample_conformers helpers) -----
    # load_rdkit_molecules returns (mols, errors) — unpack and surface
    # any per-line warnings.
    rdkit_mols, load_errors = sc.load_rdkit_molecules(args.input, add_hs=args.add_hs)
    if load_errors:
        n_show = min(5, len(load_errors))
        print(f"[data] {len(load_errors)} input(s) flagged by validation; "
              f"first {n_show}:")
        for err in load_errors[:n_show]:
            print(f"  - {err}")
    print(f"[data] {len(rdkit_mols):,} input molecules (post add-Hs / validation)")

    # Build a Data list with n_replicas=1 (we do K_traj passes ourselves).
    data_list = sc.mols_to_data_list(rdkit_mols, n_confs=1)
    # Tag each Data with its input index so we can scatter snapshots back
    # to the right molecule even after the AdaptiveBatchSampler reorders.
    for i, d in enumerate(data_list):
        d.input_idx = torch.tensor([i], dtype=torch.long)

    n_input = len(data_list)
    n_snap = len(args.snapshot_steps)
    expected_K = args.n_traj * n_snap

    # Per-input storage: list[list[Mol]] indexed by input idx.
    per_input_mols: list[list] = [[] for _ in range(n_input)]
    per_input_ids: list[str] = ["NA"] * n_input

    # ---- Per-trajectory sampling --------------------------------------
    for traj_i in range(args.n_traj):
        print(f"\n=== trajectory {traj_i+1}/{args.n_traj} ===")
        loader = sc.build_sampling_loader(
            data_list,
            sample_batch_size=args.batch_size,
            atom_aware_batching=args.atom_aware_batching,
            shuffle=False,                       # keep ordering for input_idx
            target_molecule_size=args.target_molecule_size,
        )
        for batch in tqdm(loader, desc=f"traj {traj_i+1}",
                          bar_format="{desc}: {n_fmt} batches | {elapsed} | {rate_fmt}"):
            batch = batch.to(model.device)
            result = model.sample(
                batch=batch,
                timesteps=args.n_steps,
                pre_format=True,
                save_snapshots_at=args.snapshot_steps,
            )
            # `result` = {"final": <samples>, "snapshots": [<state_at_step>, ...]}
            input_indices = batch.input_idx.view(-1).tolist()
            mols_in_batch = batch["mol"]   # list of RDKit Mol templates

            for snap in result["snapshots"]:
                coords_list = convert_coords_to_np(snap)
                mols_snap = [
                    write_coords_to_mol(mol, coords)
                    for mol, coords in zip(mols_in_batch, coords_list)
                ]
                for input_idx, m in zip(input_indices, mols_snap):
                    per_input_mols[input_idx].append(m)
                    if per_input_ids[input_idx] == "NA":
                        per_input_ids[input_idx] = (
                            m.GetProp("_Name") if m.HasProp("_Name") else "NA"
                        )

    # ---- Sanity check K uniformity ------------------------------------
    actual_K = [len(ms) for ms in per_input_mols]
    if min(actual_K, default=0) != expected_K:
        print(f"[WARN] non-uniform K per input: "
              f"min={min(actual_K, default=0)}, "
              f"max={max(actual_K, default=0)}, expected {expected_K}. "
              f"Some inputs may have failed sampling on some trajectories.")

    # ---- Flatten + write ----------------------------------------------
    generated = []
    ids = []
    for input_idx, mols in enumerate(per_input_mols):
        generated.extend(mols)
        ids.extend([per_input_ids[input_idx]] * len(mols))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "generated": generated,
            "ids": ids,
            "n_traj": args.n_traj,
            "snapshot_steps": list(args.snapshot_steps),
            "n_steps": args.n_steps,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n[output] {len(generated):,} conformers "
          f"(K={expected_K} = n_traj {args.n_traj} × n_snap {n_snap}) "
          f"→ {out_path}")


if __name__ == "__main__":
    main()
