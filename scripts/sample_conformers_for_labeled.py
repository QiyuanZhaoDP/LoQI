"""Pre-sample K conformers per TCIT-labeled molecule using the flow-matching
LoQI checkpoint.

For every molecule in each `{split}_h.pt` whose canonical SMILES is present
in the property table with `has_thermo_label == True`, run the flow model
with a small number of integration steps (default 10) to draw K diverse
3D conformers. Output is a pickle keyed by chemblid so downstream code can
look up conformers for any labeled molecule.

Design notes:
  * Geometry files stay untouched — we *read* them to enumerate mols, then
    build a fresh PyG Data list from RDKit for the model (the existing
    sample_conformers.py path, which we know is compatible with the flow
    checkpoint).
  * K replicas per mol → distinct 3D samples because the prior is Gaussian
    noise drawn independently per replica.
  * Resume-safe: if the output file already contains N chemblids, we skip
    those and append the rest.
  * Multi-GPU: launch with --shard-id / --n-shards; each shard processes a
    disjoint slice of (filtered) molecules and writes its own file.

Usage (single GPU):
  python scripts/sample_conformers_for_labeled.py \\
      --ckpt data/loqi_flow.ckpt \\
      --config scripts/conf/loqi/loqi_flow.yaml \\
      --input-pt data/chembl3d_stereo/processed/train_h.pt \\
      --property-table data/property_table.parquet \\
      --output-pkl data/labeled_conformers/train_K5.pkl \\
      --n-confs 5 --n-steps 10 --batch-size 256

Multi-GPU (4 shards):
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python scripts/sample_conformers_for_labeled.py \\
        ... --shard-id $i --n-shards 4 \\
        --output-pkl data/labeled_conformers/train_K5_shard${i}.pkl &
  done; wait
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from rdkit import Chem, RDLogger
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from tqdm import tqdm

from megalodon.data.batch_preprocessor import BatchPreProcessor
from megalodon.models.module import Graph3DInterpolantModel
from megalodon.metrics.conformer_evaluation_callback import convert_coords_to_np

# Reuse the graph-building path that sample_conformers.py uses so batches
# are guaranteed to match what the flow model expects.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_conformers import (  # noqa: E402
    mol_to_torch_geometric,
    build_sampling_loader,
)

RDLogger.DisableLog("rdApp.*")


def _canonical(mol) -> str | None:
    """Must match build_property_table.py / attach_properties.py convention."""
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
    except Exception:
        return None


def _load_labeled_smiles(parquet_path: str) -> set[str]:
    import pandas as pd
    df = pd.read_parquet(parquet_path, columns=["smiles", "has_thermo_label"])
    labeled = df.loc[df["has_thermo_label"].astype(bool), "smiles"].tolist()
    return set(labeled)


def _iter_pt_data(pt_path: str):
    """Yield (chemblid, smiles_canonical, rdkit_mol) for each entry in a
    chembl3d `_h.pt` file."""
    with torch.serialization.safe_globals([
        DataEdgeAttr, DataTensorAttr, GlobalStorage, Chem.rdchem.Mol
    ]):
        ds = torch.load(pt_path)
    data_blob, slices = ds[0], ds[1]

    # We only need .mol and .chemblid per sample. Rebuild via slices over mol.
    # Mol is stored as a Python list on data_blob (not sliceable); iterate
    # over indices directly.
    n = len(data_blob.mol) if hasattr(data_blob, "mol") else len(slices["x"]) - 1
    for i in range(n):
        mol = data_blob.mol[i] if hasattr(data_blob, "mol") else None
        if mol is None:
            continue
        smiles = _canonical(mol)
        cid = None
        if hasattr(data_blob, "chemblid"):
            cid = data_blob.chemblid[i] if isinstance(data_blob.chemblid, list) else ""
        if not cid:
            cid = f"idx_{i}"
        yield cid, smiles, mol


def _load_existing_output(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_output(path: str, records: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def build_filtered_mol_list(
    pt_path: str,
    labeled_smiles: set[str],
    shard_id: int,
    n_shards: int,
    already_done: set[str],
) -> list[tuple[str, str, Chem.Mol]]:
    """Returns [(chemblid, canonical_smiles, rdkit_mol), ...]."""
    kept = []
    n_scanned = 0
    n_unlabeled = 0
    for cid, smi, mol in _iter_pt_data(pt_path):
        n_scanned += 1
        if smi is None or smi not in labeled_smiles:
            n_unlabeled += 1
            continue
        # Stable sharding on chemblid — resume-safe across re-runs.
        if (hash(cid) & 0x7FFFFFFF) % n_shards != shard_id:
            continue
        if cid in already_done:
            continue
        kept.append((cid, smi, mol))
    print(f"[filter] scanned={n_scanned} unlabeled={n_unlabeled} "
          f"shard={shard_id}/{n_shards} kept={len(kept)} "
          f"(already_done={len(already_done)})")
    return kept


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="flow-matching LoQI checkpoint")
    p.add_argument("--config", required=True, help="matching YAML (e.g. loqi_flow.yaml)")
    p.add_argument("--input-pt", required=True,
                   help="chembl3d split file, e.g. train_h.pt")
    p.add_argument("--property-table", required=True,
                   help="parquet produced by build_property_table.py")
    p.add_argument("--output-pkl", required=True)
    p.add_argument("--n-confs", type=int, default=5,
                   help="K: conformers to draw per molecule (default 5)")
    p.add_argument("--n-steps", type=int, default=10,
                   help="flow-matching integration steps (default 10)")
    p.add_argument("--batch-size", type=int, default=256,
                   help="replicas per batch (sample_batch_size)")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save-every", type=int, default=500,
                   help="flush output every N molecules (default 500)")
    p.add_argument("--max-mols", type=int, default=None,
                   help="debug: cap molecules after filtering")
    args = p.parse_args()

    if args.n_shards < 1 or not (0 <= args.shard_id < args.n_shards):
        raise SystemExit(f"invalid shard_id={args.shard_id} / n_shards={args.n_shards}")

    print(f"[config] ckpt={args.ckpt} config={args.config}")
    print(f"[config] K={args.n_confs} steps={args.n_steps} "
          f"batch={args.batch_size} shard={args.shard_id}/{args.n_shards}")

    # 1. Figure out which molecules we still need to sample.
    labeled = _load_labeled_smiles(args.property_table)
    print(f"[data] labeled SMILES in parquet: {len(labeled):,}")

    existing = _load_existing_output(args.output_pkl)
    if existing:
        print(f"[resume] output already contains {len(existing):,} molecules "
              f"— will skip those")

    filtered = build_filtered_mol_list(
        args.input_pt, labeled, args.shard_id, args.n_shards,
        already_done=set(existing.keys()),
    )
    if args.max_mols is not None:
        filtered = filtered[: args.max_mols]
        print(f"[debug] capped to {len(filtered)} molecules")
    if not filtered:
        print("[done] nothing to do")
        return

    # 2. Load model.
    # Force loss_fn=None on load. Sampling doesn't need any auxiliary loss,
    # and explicitly nulling it sidesteps a nasty backward-compat pitfall:
    # ckpts saved before the torchmetrics rewrite (when CombinedAuxiliaryLoss
    # was a plain Python class) pickle a loss_fn whose __dict__ lacks the
    # nn.Module internals. With the new nn.Module-flavored class definition,
    # unpickling produces a half-initialized object and load_state_dict
    # crashes with `AttributeError: '...' object has no attribute '_buffers'`.
    # Passing loss_fn=None + strict=False ignores both the saved hparam and
    # any loss_fn.* keys in the state_dict that don't apply.
    cfg = OmegaConf.load(args.config)
    print(f"[model] loading flow checkpoint {args.ckpt}")
    model = Graph3DInterpolantModel.load_from_checkpoint(
        args.ckpt,
        loss_fn=None,
        loss_params=cfg.loss,
        interpolant_params=cfg.interpolant,
        sampling_params=cfg.sample,
        batch_preprocessor=BatchPreProcessor(cfg.data.aug_rotations, cfg.data.scale_coords),
        strict=False,
    ).to(args.device).eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={total_params/1e6:.2f}M")

    # 3. Main loop: batch through molecules, replicating each K times.
    records: dict[str, dict] = dict(existing)
    chunk_size = max(1, args.batch_size // max(args.n_confs, 1))  # mols per "build"
    # We process the filtered list in chunks of chunk_size mols at a time,
    # replicate each K times, run through the model, and store.

    start = time.time()
    n_total = len(filtered)
    n_flushed = 0
    pbar = tqdm(total=n_total, desc=f"sample K={args.n_confs}")
    for off in range(0, n_total, chunk_size):
        chunk = filtered[off: off + chunk_size]
        # Replicate K times, preserve (cid, smi) per replica so we can
        # regroup after the model call.
        data_list = []
        owners = []  # index into chunk for each data item
        for i, (cid, smi, mol) in enumerate(chunk):
            for _ in range(args.n_confs):
                d = mol_to_torch_geometric(Chem.Mol(mol), smi, use_3d_input=False)
                data_list.append(d)
                owners.append(i)

        loader = build_sampling_loader(
            data_list=data_list,
            sample_batch_size=args.batch_size,
            atom_aware_batching=True,
            shuffle=False,
            target_molecule_size=50,
        )

        # Sample for each batch; accumulate per-replica coords in chunk order.
        per_replica_coords: list[np.ndarray | None] = [None] * len(data_list)
        cursor = 0  # position in data_list processed so far
        for batch in loader:
            batch = batch.to(model.device)
            with torch.no_grad():
                sample = model.sample(
                    batch=batch, timesteps=args.n_steps, pre_format=True
                )
            coords_np = convert_coords_to_np(sample)  # list[ndarray] per mol
            for arr in coords_np:
                per_replica_coords[cursor] = arr.astype(np.float32)
                cursor += 1

        assert cursor == len(data_list), \
            f"replica coverage mismatch: {cursor} vs {len(data_list)}"

        # Regroup: K replicas per source mol -> stacked [K, N, 3].
        for i, (cid, smi, mol) in enumerate(chunk):
            group = [per_replica_coords[j] for j, o in enumerate(owners) if o == i]
            if len(group) != args.n_confs:
                # Should not happen, but guard anyway.
                continue
            # Some atom counts could differ if mol was rebuilt with/without
            # Hs downstream. Enforce consistency.
            n_atoms = group[0].shape[0]
            if any(a.shape[0] != n_atoms for a in group):
                continue
            coords = np.stack(group, axis=0)  # [K, N, 3]
            records[cid] = {
                "smiles": smi,
                "coords": coords,
                "n_atoms": int(n_atoms),
            }
        pbar.update(len(chunk))

        # Periodic flush so a crash doesn't lose hours of work.
        if (off // chunk_size) % max(1, args.save_every // chunk_size) == 0 \
                and off > 0:
            _save_output(args.output_pkl, records)
            n_flushed += 1
            tqdm.write(f"[flush #{n_flushed}] saved {len(records):,} mols "
                       f"({(time.time() - start):.0f}s elapsed)")

    pbar.close()

    _save_output(args.output_pkl, records)
    dur = time.time() - start
    print(f"[done] wrote {len(records):,} mols to {args.output_pkl} in {dur:.0f}s "
          f"(rate={len(records)/max(dur,1):.1f} mol/s)")


if __name__ == "__main__":
    main()
