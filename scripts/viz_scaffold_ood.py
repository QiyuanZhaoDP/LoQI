#!/usr/bin/env python3
"""Visualize OOD nature of balanced scaffold splits — 3-panel layout per ds.

For each dataset, plots THREE panels side-by-side:
  (a) random_cv5  MOLECULE-level (1 pt per mol)
       -> folds should be uniformly MIXED (no OOD)
  (b) scaffold_balanced_cv5 MOLECULE-level
       -> folds look partly mixed because ECFP fingerprints capture more
          than scaffold (side-chain similarity bleeds across folds)
  (c) scaffold_balanced_cv5 SCAFFOLD-level (1 pt per unique Murcko scaffold)
       -> folds are CLEANLY SEPARATED by construction — each scaffold
          belongs to exactly one fold. This is the visualization that
          directly reflects the OOD principle.

The progression (a) -> (b) -> (c) tells the story:
  - random is mixed (no OOD)
  - scaffold-at-mol-level looks mixed (FP space is shared across scaffolds)
  - scaffold-at-scaffold-level is clean (the actual OOD partitioning)
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLIT_ROOT = REPO_ROOT / "downstream_ft" / "0515_final" / "Split"

DATASETS = [
    ("BP_K",            "PT,  N=5572"),
    ("Hf_gas_kJmol",    "TH,  N=3086"),
    ("ESOL_logS",       "SL,  N=1115"),
    ("dielectric_298K", "EL,  N=1362"),
    ("PPBR_pct",        "BX,  N=1386"),
]

FOLD_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

ECFP_RADIUS = 2
ECFP_BITS = 1024
PCA_DIM = 50
TSNE_PERPLEXITY = 30
TSNE_SEED = 42


def ecfp(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(ECFP_BITS, dtype=np.uint8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, ECFP_RADIUS, ECFP_BITS)
    return np.array(fp, dtype=np.uint8)


def get_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"__INV__:{smiles}"
    s = MurckoScaffold.GetScaffoldForMol(mol)
    canon = Chem.MolToSmiles(s)
    return canon if canon else f"__ACYC__:{Chem.MolToSmiles(mol)}"


def load_test_smiles(split_dir: Path) -> tuple[list, list]:
    """Returns (smiles_list, fold_labels)."""
    smis, labels = [], []
    for k in range(1, 6):
        fp = split_dir / f"cv{k}_test.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        smi_col = next(c for c in df.columns if c.upper() == "SMILES")
        smis.extend(df[smi_col].astype(str).tolist())
        labels.extend([k] * len(df))
    return smis, labels


def dedupe_to_scaffolds(smis, labels):
    """Return (scaffold_smis, scaffold_fold_labels) — one entry per unique scaffold."""
    seen = {}                  # scaffold -> fold
    sca_smis, sca_labels = [], []
    for smi, k in zip(smis, labels):
        sca = get_scaffold(smi)
        if sca in seen:
            continue
        seen[sca] = k
        # Use the scaffold SMILES itself for the ECFP — that way the t-SNE
        # is computed in scaffold-space, not in full-molecule FP space.
        # Fall back to the original mol if scaffold is empty (acyclic).
        canonical_for_fp = sca
        if sca.startswith("__"):
            canonical_for_fp = smi
        sca_smis.append(canonical_for_fp)
        sca_labels.append(k)
    return sca_smis, sca_labels


def reduce_to_2d(smis: list) -> np.ndarray:
    fps = np.stack([ecfp(s) for s in smis], axis=0).astype(np.float32)
    n = len(fps)
    if n < 5:
        return np.zeros((n, 2))
    pca_dim = min(PCA_DIM, fps.shape[1], n - 1)
    pca = PCA(n_components=pca_dim, random_state=TSNE_SEED)
    fps_pca = pca.fit_transform(fps)
    perp = min(TSNE_PERPLEXITY, max(5, (n - 1) // 4))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=TSNE_SEED,
                init="pca", learning_rate="auto", n_iter=1000)
    return tsne.fit_transform(fps_pca)


def _draw_hull(ax, pts, color):
    if len(pts) < 5:
        return
    center = pts.mean(axis=0)
    d = np.linalg.norm(pts - center, axis=1)
    keep = d < np.quantile(d, 0.90)
    pts_t = pts[keep] if keep.sum() >= 3 else pts
    try:
        hull = ConvexHull(pts_t)
        verts = pts_t[hull.vertices]
        verts = np.vstack([verts, verts[:1]])
        ax.fill(verts[:, 0], verts[:, 1], color=color, alpha=0.15, lw=0)
        ax.plot(verts[:, 0], verts[:, 1], color=color, alpha=0.55,
                lw=1.2, linestyle="--")
    except Exception:
        pass


def plot_panel(ax, coords, labels, title, n_total, draw_hull):
    labels = np.asarray(labels)
    coords = np.asarray(coords)
    for k in range(1, 6):
        m = labels == k
        if m.sum() == 0:
            continue
        ax.scatter(coords[m, 0], coords[m, 1],
                   c=FOLD_COLORS[k - 1], s=14, alpha=0.6,
                   edgecolors="none",
                   label=f"fold {k} (n={int(m.sum())})")
        if draw_hull:
            _draw_hull(ax, coords[m], FOLD_COLORS[k - 1])
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("t-SNE-1", fontsize=8)
    ax.set_ylabel("t-SNE-2", fontsize=8)
    ax.legend(loc="best", fontsize=7, framealpha=0.85, markerscale=1.4)
    ax.text(0.02, 0.98, f"N = {n_total}", transform=ax.transAxes,
            fontsize=8, verticalalignment="top",
            bbox=dict(facecolor="white", edgecolor="0.7", boxstyle="round,pad=0.2"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="/Users/zhao922/Desktop/scaffold_ood_viz.png")
    p.add_argument("--datasets", type=str, default=None)
    args = p.parse_args()

    ds_list = DATASETS
    if args.datasets:
        ds_list = [(n.strip(), "") for n in args.datasets.split(",")]

    n_rows = len(ds_list)
    fig, axes = plt.subplots(n_rows, 3,
                             figsize=(16, 4.2 * n_rows),
                             squeeze=False)

    for row, (ds_name, ds_meta) in enumerate(ds_list):
        ds_dir = SPLIT_ROOT / ds_name
        print(f"\n[{ds_name}]  {ds_meta}")

        # (a) random_cv5 molecule-level
        r_dir = ds_dir / "random_cv5"
        r_smis, r_labels = load_test_smiles(r_dir)
        r_coords = reduce_to_2d(r_smis)
        plot_panel(axes[row, 0], r_coords, r_labels,
                   f"{ds_name}\n(a) random_cv5",
                   n_total=len(r_smis), draw_hull=False)
        print(f"  random_cv5          done  ({len(r_smis)} mols)")

        # (b) scaffold_diverse_cv5 (Butina + LPT)
        div_dir = ds_dir / "scaffold_diverse_cv5"
        div_smis, div_labels = load_test_smiles(div_dir)
        div_coords = reduce_to_2d(div_smis)
        div_sizes = [int(np.sum(np.array(div_labels) == k)) for k in range(1, 6)]
        plot_panel(axes[row, 1], div_coords, div_labels,
                   f"{ds_name}\n(b) scaffold_diverse_cv5 (LPT, sim=0.30)  •  sizes={div_sizes}",
                   n_total=len(div_smis), draw_hull=True)
        print(f"  diverse             done  ({len(div_smis)} mols, sizes={div_sizes})")

        # (c) scaffold_hybrid_cv5 (Lloyd + rebalance + OOD-swap)
        hyb_dir = ds_dir / "scaffold_hybrid_cv5"
        if not hyb_dir.exists():
            axes[row, 2].text(0.5, 0.5, f"missing\nscaffold_hybrid_cv5",
                              ha="center", va="center",
                              transform=axes[row, 2].transAxes,
                              fontsize=11, color="red")
            print(f"  hybrid SKIP")
            continue
        hyb_smis, hyb_labels = load_test_smiles(hyb_dir)
        hyb_coords = reduce_to_2d(hyb_smis)
        hyb_sizes = [int(np.sum(np.array(hyb_labels) == k)) for k in range(1, 6)]
        plot_panel(axes[row, 2], hyb_coords, hyb_labels,
                   f"{ds_name}\n(c) scaffold_hybrid_cv5 (Lloyd→rebal→swap)  •  sizes={hyb_sizes}",
                   n_total=len(hyb_smis), draw_hull=True)
        print(f"  hybrid              done  ({len(hyb_smis)} mols, sizes={hyb_sizes})")

    fig.suptitle(
        "Three CV splits.  All plots: ECFP4(1024)→PCA(50)→t-SNE(2), test sets colored by fold.\n"
        "(a) random_cv5: folds uniformly mixed (baseline, no OOD).  "
        "(b) scaffold_diverse_cv5 (LPT): Butina-cluster scaffolds at sim=0.30, LPT bin-pack to balance "
        "fold sizes. Median test→nearest_train Tanimoto distance ≈ 0.60.  "
        "(c) scaffold_hybrid_cv5 (Lloyd→rebal→swap): iterative Lloyd assigns clusters by nearest fold centroid "
        "for stronger OOD; rebalance phase enforces balance; final OOD-swap pass refines. "
        "Median distance ≈ 0.68 (+13% vs LPT), balance 1.20-1.25× (vs LPT 1.04×).",
        fontsize=10, y=0.998, wrap=True
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"\nSaved -> {out_path}  ({out_path.stat().st_size//1024} KB)")
    print(f"Saved -> {out_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
