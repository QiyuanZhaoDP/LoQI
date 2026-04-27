"""Inspect a sample_conformers.py output pickle: pick N input molecules,
write each one's K conformers as multi-frame .xyz files for visualization
in PyMol / VMD / Avogadro.

Reads the pickle's `generated` list (RDKit Mols, in [m0_c0, m0_c1, …,
m1_c0, …] order from the upstream sampler), groups them by canonical
SMILES, and exports the first N groups to disk.

Output layout:
    <output-dir>/manifest.csv              one row per group (smi, file)
    <output-dir>/mol_0001_<safe>.xyz       multi-frame xyz: K frames concatenated
    <output-dir>/mol_0002_<safe>.xyz
    ...

Multi-frame .xyz format: each frame is `N_atoms\\n<comment>\\n<atom xyz>×N`
back-to-back. PyMol/VMD load this as an animation; Avogadro shows them
as separate conformers via "frames".

Usage:
    python scripts/inspect_k5_pkl.py \\
        --pkl data/downstream_k5/V_cp.pkl \\
        --n-smiles 10 \\
        --output-dir /tmp/k5_inspect_vcp
"""
from __future__ import annotations

import argparse
import csv
import pickle
import re
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def _safe_name(smi: str, max_len: int = 32) -> str:
    """Filesystem-safe slug from a SMILES."""
    s = re.sub(r"[^A-Za-z0-9_-]", "_", smi)
    return s[:max_len]


def _mol_to_xyz_frame(mol, comment: str = "") -> str:
    """One xyz frame: N_atoms\\n<comment>\\n<elem x y z>×N."""
    if mol.GetNumConformers() == 0:
        return ""
    conf = mol.GetConformer(0)
    lines = [str(mol.GetNumAtoms()), comment]
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol():<3s} {p.x:>14.6f} {p.y:>14.6f} {p.z:>14.6f}")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pkl", required=True,
                   help="sample_conformers.py output pickle "
                        "(must contain 'generated': list of RDKit Mols)")
    p.add_argument("--n-smiles", type=int, default=10,
                   help="how many unique SMILES to export (default 10)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42,
                   help="random pick when n-smiles < n-unique")
    p.add_argument("--first", action="store_true",
                   help="take the FIRST n-smiles instead of random sample "
                        "(useful for reproducibility / matching CSV row order)")
    args = p.parse_args()

    print(f"Loading {args.pkl}")
    with open(args.pkl, "rb") as f:
        d = pickle.load(f)
    mols = d["generated"]
    energies = d.get("energies", None)
    print(f"  pickle has {len(mols):,} mols  (energies: {'yes' if energies is not None else 'no'})")

    # Group mols by their canonical SMILES, preserving the order in which each
    # canonical SMILES first appeared in the pickle. That ordering is the same
    # as the input .smi line order, so --first matches the CSV row order.
    by_canon: dict[str, list[tuple[int, "Chem.Mol"]]] = {}
    order: list[str] = []
    for j, mol in enumerate(mols):
        if mol is None:
            continue
        try:
            canon = Chem.MolToSmiles(mol, isomericSmiles=True)
        except Exception:
            continue
        if canon not in by_canon:
            order.append(canon)
            by_canon[canon] = []
        by_canon[canon].append((j, mol))
    print(f"  unique canonical SMILES: {len(by_canon):,}")

    n_pick = min(args.n_smiles, len(order))
    if args.first:
        picks = order[:n_pick]
    else:
        import random
        rng = random.Random(args.seed)
        picks = rng.sample(order, n_pick)
    print(f"  exporting {n_pick} groups")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = [("idx", "smiles", "n_conformers", "xyz_file", "energies_eV")]
    for i, canon in enumerate(picks, start=1):
        group = by_canon[canon]
        slug = _safe_name(canon)
        fname = f"mol_{i:04d}_{slug}.xyz"
        path = out_dir / fname
        ens_str = ""
        with open(path, "w") as f:
            for k, (j, mol) in enumerate(group):
                e = float(energies[j]) if energies is not None else None
                comment_parts = [f"smiles={canon}", f"conf={k}"]
                if e is not None:
                    comment_parts.append(f"energy_eV={e:.6f}")
                comment = " ".join(comment_parts)
                f.write(_mol_to_xyz_frame(mol, comment))
                if e is not None:
                    ens_str += f"{e:.6f},"
        manifest.append(
            (i, canon, len(group), fname, ens_str.rstrip(","))
        )

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        csv.writer(f).writerows(manifest)
    print(f"\nWrote {n_pick} multi-frame xyz files + manifest to {out_dir}")
    print(f"  view: pymol {out_dir}/mol_0001_*.xyz")
    print(f"  or:   vmd  {out_dir}/mol_0001_*.xyz")


if __name__ == "__main__":
    main()
