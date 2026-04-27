"""Sanity-check a prepared downstream .pt before feeding it to
scripts/downstream_cv.py. Catches the kinds of out-of-range / NaN
issues that surface as opaque CUDA device-side asserts mid-training.

Reports both:
  - Global stats: range of atom types, charges, edge_attr, edge_index;
    pos NaN/Inf; empty molecules.
  - Per-mol failures: which Data objects have invalid fields, with
    the offending SMILES and a short reason.

Usage:
    python scripts/check_downstream_pt.py \\
        --pt data/downstream_pt/k.pt
"""
from __future__ import annotations

import argparse
import sys

import torch
from rdkit.Chem.rdchem import Mol
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage


# Encoder bounds — match data_processing/build_property_table.py and
# scripts/prepare_downstream_dataset.py
ATOM_MAX = 16        # 17 element classes (LoQI encoder)
CHARGES_MAX = 5      # offset=2 → -2..+3 → indices 0..5
EDGE_MAX = 8         # bond classes


class _D(InMemoryDataset):
    def __init__(self, data, slices):
        super().__init__(".")
        self.data, self.slices = data, slices
        self._indices = None


def _load(path):
    with torch.serialization.safe_globals(
        [DataEdgeAttr, DataTensorAttr, GlobalStorage, Mol]
    ):
        data, slices = torch.load(path)
    return data, slices


def _global_stats(data, slices):
    print("\n=== Global field ranges ===")
    fields = [
        ("data.x         (atom type)",  data.x,         (0, ATOM_MAX),    True),
        ("data.charges                ", data.charges,   (0, CHARGES_MAX), True),
        ("data.edge_attr              ", data.edge_attr, (0, EDGE_MAX),    True),
    ]
    issues = []
    for label, t, (lo, hi), is_int in fields:
        mn, mx = int(t.min().item()), int(t.max().item()) if t.numel() else (0, 0)
        warn = ""
        if mn < lo or mx > hi:
            warn = f"  ⚠ OUT OF RANGE (expect {lo}..{hi})"
            issues.append(label.strip())
        print(f"  {label}  {mn} .. {mx}{warn}")

    print(f"\n=== Geometry ===")
    n_nan = int(torch.isnan(data.pos).any(dim=-1).sum().item())
    n_inf = int(torch.isinf(data.pos).any(dim=-1).sum().item())
    print(f"  pos NaN atoms: {n_nan}")
    print(f"  pos Inf atoms: {n_inf}")
    if n_nan: issues.append("pos has NaN")
    if n_inf: issues.append("pos has Inf")

    print(f"\n=== Edges ===")
    if data.edge_index.numel() > 0:
        em = int(data.edge_index.max().item())
        n_atoms_total = data.x.numel()
        print(f"  edge_index max: {em}   total atoms: {n_atoms_total}")
        if em >= n_atoms_total:
            print(f"  ⚠ edge_index references atom out of range")
            issues.append("edge_index out of bounds")
    else:
        print(f"  no edges")

    print(f"\n=== Molecule counts ===")
    n_mols = len(slices["x"]) - 1
    sizes = []
    for i in range(n_mols):
        a, b = int(slices["x"][i]), int(slices["x"][i + 1])
        sizes.append(b - a)
    sizes_t = torch.tensor(sizes)
    print(f"  total mols: {n_mols}")
    print(f"  atoms/mol  min={sizes_t.min().item()}  max={sizes_t.max().item()}  "
          f"mean={sizes_t.float().mean().item():.1f}")
    n_empty = int((sizes_t == 0).sum().item())
    n_one   = int((sizes_t == 1).sum().item())
    if n_empty:
        print(f"  ⚠ {n_empty} empty mols (0 atoms)")
        issues.append(f"{n_empty} empty mols")
    if n_one:
        print(f"  {n_one} single-atom mols (might confuse adaptive batching)")

    return issues, n_mols


def _per_mol_check(data, slices, max_print: int = 20):
    """Iterate every Data and flag specific problems. Slow-ish but useful."""
    n_mols = len(slices["x"]) - 1
    ds = _D(data, slices)
    bad = []
    for i in range(n_mols):
        d = ds[i]
        reasons = []
        if d.x.numel() == 0:
            reasons.append("x empty")
        elif int(d.x.max()) > ATOM_MAX or int(d.x.min()) < 0:
            reasons.append(f"x range [{int(d.x.min())}, {int(d.x.max())}]")
        if d.charges.numel() and (int(d.charges.max()) > CHARGES_MAX or int(d.charges.min()) < 0):
            reasons.append(f"charges range [{int(d.charges.min())}, {int(d.charges.max())}]")
        if d.edge_attr.numel() and (int(d.edge_attr.max()) > EDGE_MAX or int(d.edge_attr.min()) < 0):
            reasons.append(f"edge_attr range [{int(d.edge_attr.min())}, {int(d.edge_attr.max())}]")
        if d.edge_index.numel() and int(d.edge_index.max()) >= d.x.numel():
            reasons.append(f"edge_idx_oob {int(d.edge_index.max())} >= {d.x.numel()}")
        if torch.isnan(d.pos).any():
            reasons.append("pos NaN")
        if torch.isinf(d.pos).any():
            reasons.append("pos Inf")
        if reasons:
            smi = getattr(d, "smiles", None) or getattr(d, "input_id", None) or "?"
            bad.append((i, smi, reasons))

    print(f"\n=== Per-mol scan ===")
    print(f"  bad mols: {len(bad)} / {n_mols}")
    for i, smi, reasons in bad[:max_print]:
        print(f"   idx={i:>6d}  smi={smi!r}  ->  {', '.join(reasons)}")
    if len(bad) > max_print:
        print(f"   ... and {len(bad) - max_print} more (use --max-print to see more)")
    return bad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pt", required=True,
                   help="prepared dataset .pt (from prepare_downstream_dataset.py "
                        "or prepare_downstream_K_pt.py)")
    p.add_argument("--max-print", type=int, default=20,
                   help="max number of bad mols to print individually")
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero if any issue found")
    args = p.parse_args()

    print(f"Reading {args.pt}")
    data, slices = _load(args.pt)

    issues, n_mols = _global_stats(data, slices)
    bad = _per_mol_check(data, slices, max_print=args.max_print)

    print("\n" + "=" * 64)
    if not issues and not bad:
        print(f"  OK  ({n_mols:,} mols, no problems detected)")
    else:
        print(f"  PROBLEMS:")
        for s in issues:
            print(f"    - global: {s}")
        if bad:
            print(f"    - per-mol: {len(bad)} bad mols (see scan above)")
    print("=" * 64)

    if args.strict and (issues or bad):
        sys.exit(1)


if __name__ == "__main__":
    main()
