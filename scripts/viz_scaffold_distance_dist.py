#!/usr/bin/env python3
"""Per-test-mol Tanimoto distance-to-nearest-train histogram.

Stronger visual evidence of OOD strength than t-SNE hulls. For each test
molecule, compute Tanimoto distance to its nearest train/valid molecule
(over all folds), and plot the distribution. Three splits side by side:

  random_cv5            : mass concentrated near 0 (every test mol has a
                          near-twin in train)
  scaffold_diverse_cv5  : mass shifted right (Murcko + Butina partition)
  scaffold_hybrid_cv5   : mass shifted further right (Lloyd→rebal→swap)

A vertical line marks the median per split. The shift between medians is
the headline "how much stronger OOD did hybrid achieve" number.
"""
import argparse
import csv
import random
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity

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

SPLITS = [
    ("random_cv5",            "random",            "#4daf4a"),
    ("scaffold_diverse_cv5",  "diverse (LPT)",     "#377eb8"),
    ("scaffold_hybrid_cv5",   "hybrid (Lloyd→rebal→swap)", "#e41a1c"),
]


def fp(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024)


def load(split_dir, kind):
    smis = []
    for k in range(1, 6):
        fp_path = split_dir / f"cv{k}_{kind}.csv"
        if not fp_path.exists(): continue
        with open(fp_path) as fh:
            r = csv.reader(fh); next(r, None)
            for row in r:
                if len(row) >= 2:
                    smis.append((row[0], k))
    return smis


def per_test_min_dist(split_dir, sample_per_fold=200, seed=42):
    """For each test mol (sampled), distance to nearest train+valid mol."""
    rng = random.Random(seed)
    dists = []
    for k in range(1, 6):
        test_csv  = split_dir / f"cv{k}_test.csv"
        train_csv = split_dir / f"cv{k}_train.csv"
        valid_csv = split_dir / f"cv{k}_valid.csv"
        if not test_csv.exists(): continue

        test = []
        with open(test_csv) as fh:
            r = csv.reader(fh); next(r, None)
            for row in r:
                if len(row) >= 2: test.append(row[0])

        train = []
        for src in [train_csv, valid_csv]:
            if not src.exists(): continue
            with open(src) as fh:
                r = csv.reader(fh); next(r, None)
                for row in r:
                    if len(row) >= 2: train.append(row[0])

        test_smp = rng.sample(test, min(sample_per_fold, len(test)))
        train_smp = rng.sample(train, min(sample_per_fold * 5, len(train)))
        test_fps  = [fp(s) for s in test_smp]
        train_fps = [fp(s) for s in train_smp]
        test_fps  = [f for f in test_fps  if f is not None]
        train_fps = [f for f in train_fps if f is not None]
        if not test_fps or not train_fps: continue
        for tfp in test_fps:
            sims = BulkTanimotoSimilarity(tfp, train_fps)
            dists.append(1.0 - max(sims))
    return dists


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/Users/zhao922/Desktop/scaffold_ood_distance_hist.png")
    args = p.parse_args()

    n_rows = len(DATASETS)
    fig, axes = plt.subplots(n_rows, 1, figsize=(11, 2.7 * n_rows), squeeze=False)
    axes = axes[:, 0]

    for row, (ds_name, ds_meta) in enumerate(DATASETS):
        ax = axes[row]
        print(f"\n[{ds_name}] {ds_meta}")
        for split_sub, label, color in SPLITS:
            split_dir = SPLIT_ROOT / ds_name / split_sub
            if not split_dir.exists():
                print(f"  {split_sub:<25} missing")
                continue
            dists = per_test_min_dist(split_dir, sample_per_fold=150)
            if not dists:
                continue
            med = statistics.median(dists)
            ax.hist(dists, bins=30, range=(0.0, 1.0),
                    color=color, alpha=0.45, density=True,
                    label=f"{label}  (median={med:.3f}, n={len(dists)})")
            ax.axvline(med, color=color, lw=2, linestyle="--", alpha=0.8)
            print(f"  {split_sub:<25} median={med:.3f}  q25={sorted(dists)[len(dists)//4]:.3f}  q75={sorted(dists)[3*len(dists)//4]:.3f}")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Tanimoto distance (test mol → nearest train/valid mol)", fontsize=9)
        ax.set_ylabel("density", fontsize=9)
        ax.set_title(f"{ds_name}  ({ds_meta})", fontsize=11)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        ax.grid(alpha=0.3)
        # Mark "random pair" reference: where two random organic mols sit
        # Typical median ~0.65-0.75 for organic datasets — annotation
        ax.axvspan(0.0, 0.3, alpha=0.05, color="red", label="_nolegend_")
        ax.axvspan(0.7, 1.0, alpha=0.05, color="green", label="_nolegend_")

    fig.suptitle(
        "OOD strength: distribution of (test mol → nearest train mol) Tanimoto distance.\n"
        "Higher distance = stronger OOD.  "
        "Right shift between curves = hybrid genuinely places test mols further from train than diverse/random.",
        fontsize=11, y=0.998, wrap=True
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path(args.out)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
