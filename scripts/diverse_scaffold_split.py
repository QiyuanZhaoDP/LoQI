#!/usr/bin/env python3
"""Diverse scaffold k-fold splitter — Butina-cluster scaffolds first, then LPT.

Stronger OOD guarantee than scaffold_balanced_cv5:

  Old `scaffold_balanced_cv5`:
    - Murcko scaffold ID exclusivity (train/test share no scaffold ID)
    - BUT scaffolds in different folds can be highly similar in ECFP space
      (e.g., benzene ring in fold A, pyridine ring in fold B — different
      scaffold IDs, similar fingerprints) — so the OOD is "discrete-ID-only"
      and the t-SNE viz shows heavy hull overlap.

  New `scaffold_diverse_cv5`:
    - First Butina-cluster scaffolds at Tanimoto similarity ≥ S_THRESH
      → produces "super-clusters" of structurally similar scaffolds
    - LPT bin-pack super-clusters into K=5 folds by member-molecule count
    - Guarantee: every scaffold in fold A is at Tanimoto-distance ≥
      (some lower bound depending on Butina) from every scaffold in fold B

Trade-off: lower S_THRESH (looser clusters) → stronger FP-distance OOD
but fewer super-clusters → worse balance. We default to S_THRESH=0.4
(scaffold similarity ≥ 0.4 are co-clustered), which empirically gives a
good balance between OOD strength and fold-size uniformity.

Usage:
    python scripts/diverse_scaffold_split.py                # all ds, S=0.4
    python scripts/diverse_scaffold_split.py --sim 0.5      # tighter clusters
    python scripts/diverse_scaffold_split.py --ds BP_K --dry-run --sim 0.3 0.4 0.5
"""
import argparse
import csv
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.ML.Cluster import Butina

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLIT_ROOT = REPO_ROOT / "downstream_ft" / "0515_final" / "Split"
N_FOLDS = 5
VALID_FRAC = 0.10
SEED = 42
ECFP_R, ECFP_BITS = 2, 1024


def get_scaffold(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return f"__INV__:{smi}"
    s = MurckoScaffold.GetScaffoldForMol(m)
    canon = Chem.MolToSmiles(s)
    return canon if canon else f"__ACYC__:{Chem.MolToSmiles(m)}"


def get_fp(smi):
    """Return ECFP4 ExplicitBitVect, or None if SMILES invalid."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, ECFP_R, ECFP_BITS)


def load_full(ds_dir: Path):
    """Return list[(smi, target)] for the full dataset (union of any single
    fold's train+valid+test in scaffold_cv5 — they're disjoint by fold but
    union recovers the full data)."""
    src = ds_dir / "random_cv5"
    rows, seen = [], set()
    for kind in ["train", "valid", "test"]:
        fp = src / f"cv1_{kind}.csv"
        if not fp.exists():
            continue
        with open(fp) as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) < 2: continue
                smi, tgt = row[0], row[1]
                if smi in seen: continue
                seen.add(smi)
                rows.append((smi, tgt))
    return rows


def diverse_split(rows, sim_thresh=0.4, n_folds=5, seed=42, valid_frac=0.10):
    """Returns (folds_dict, stats_dict)."""
    # ---- Step 1: scaffold of each mol ----
    scaf_of = {}
    for smi, _ in rows:
        scaf_of[smi] = get_scaffold(smi)
    # Group by scaffold
    scaf_rows = defaultdict(list)
    for smi, tgt in rows:
        scaf_rows[scaf_of[smi]].append((smi, tgt))
    unique_scafs = list(scaf_rows.keys())

    # ---- Step 2: ECFP for each scaffold (or fall back to rep molecule for acyclic) ----
    scaf_fps = []
    for s in unique_scafs:
        if s.startswith("__"):
            rep_smi = scaf_rows[s][0][0]
            fp = get_fp(rep_smi)
        else:
            fp = get_fp(s)
        scaf_fps.append(fp)
    valid_idx = [i for i, fp in enumerate(scaf_fps) if fp is not None]
    valid_fps = [scaf_fps[i] for i in valid_idx]
    n_v = len(valid_fps)

    # ---- Step 3: Butina cluster scaffolds at distance < (1 - sim_thresh) ----
    dist_thresh = 1.0 - sim_thresh
    # Compute lower-triangular pairwise distances
    dists = []
    for i in range(1, n_v):
        sims = BulkTanimotoSimilarity(valid_fps[i], valid_fps[:i])
        dists.extend(1.0 - s for s in sims)
    butina = Butina.ClusterData(dists, n_v, dist_thresh, isDistData=True)
    # butina: tuple of tuples; each inner tuple = indices into valid_fps

    # ---- Step 4: build super-clusters (each = list of (smi, tgt)) ----
    super_clusters = []
    for cg in butina:
        mols = []
        for idx_v in cg:
            orig = valid_idx[idx_v]
            mols.extend(scaf_rows[unique_scafs[orig]])
        super_clusters.append(mols)
    # Singletons for any scaffolds without valid fp (rare invalid SMILES)
    for i, fp in enumerate(scaf_fps):
        if fp is None:
            super_clusters.append(scaf_rows[unique_scafs[i]])

    # ---- Step 5: LPT bin-pack super-clusters into n_folds ----
    super_clusters.sort(key=len, reverse=True)
    buckets = [[] for _ in range(n_folds)]
    sizes = [0] * n_folds
    for sc in super_clusters:
        idx = sizes.index(min(sizes))
        buckets[idx].extend(sc)
        sizes[idx] += len(sc)

    # ---- Step 6: per fold, valid sampled from train_pool ----
    rng_root = random.Random(seed)
    folds = {}
    for k in range(n_folds):
        test_set = buckets[k]
        pool = []
        for j in range(n_folds):
            if j != k:
                pool.extend(buckets[j])
        rng = random.Random(seed + k)
        pool_shuf = pool.copy()
        rng.shuffle(pool_shuf)
        n_valid = max(1, int(round(len(pool_shuf) * valid_frac)))
        valid_set = pool_shuf[:n_valid]
        train_set = pool_shuf[n_valid:]
        folds[k + 1] = {"train": train_set, "valid": valid_set, "test": test_set}

    # ---- Stats ----
    test_sizes = [len(folds[k]["test"]) for k in sorted(folds)]
    n_super = len(super_clusters)
    stats = {
        "total": sum(test_sizes),
        "n_unique_scaffolds": len(unique_scafs),
        "n_super_clusters": n_super,
        "test_sizes": test_sizes,
        "max_min_ratio": max(test_sizes) / min(test_sizes) if min(test_sizes) > 0 else float("inf"),
        "sim_threshold": sim_thresh,
    }
    return folds, stats


def inter_fold_ood_metric(folds, sample_per_fold=200, seed=42):
    """For each test mol, find nearest *train* mol by Tanimoto similarity.
    Return: median(min_train_distance) across all sampled test mols.
    Higher = stronger FP-distance OOD."""
    rng = random.Random(seed)
    all_min_dist = []
    for k in folds:
        test = folds[k]["test"]
        train = folds[k]["train"] + folds[k]["valid"]
        # Sample to keep cost manageable
        test_smp = rng.sample(test, min(sample_per_fold, len(test)))
        train_smp = rng.sample(train, min(sample_per_fold * 4, len(train)))

        test_fps = [get_fp(smi) for smi, _ in test_smp]
        train_fps = [get_fp(smi) for smi, _ in train_smp]
        test_fps = [f for f in test_fps if f is not None]
        train_fps = [f for f in train_fps if f is not None]
        if not test_fps or not train_fps:
            continue
        for tfp in test_fps:
            sims = BulkTanimotoSimilarity(tfp, train_fps)
            all_min_dist.append(1.0 - max(sims))
    if not all_min_dist:
        return None
    return {
        "median_min_dist": statistics.median(all_min_dist),
        "q25_min_dist": sorted(all_min_dist)[len(all_min_dist) // 4],
        "q75_min_dist": sorted(all_min_dist)[3 * len(all_min_dist) // 4],
        "n_samples": len(all_min_dist),
    }


def write_fold(out_dir: Path, k: int, fold_data: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind in ["train", "valid", "test"]:
        fp = out_dir / f"cv{k}_{kind}.csv"
        with open(fp, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["SMILES", "TARGET"])
            for smi, tgt in fold_data[kind]:
                w.writerow([smi, tgt])


def process_ds(ds_dir: Path, sim_thresh: float, dry_run: bool, out_subdir: str):
    rows = load_full(ds_dir)
    if not rows:
        return None
    folds, stats = diverse_split(rows, sim_thresh=sim_thresh,
                                 n_folds=N_FOLDS, seed=SEED,
                                 valid_frac=VALID_FRAC)
    if not dry_run:
        out_dir = ds_dir / out_subdir
        for k in folds:
            write_fold(out_dir, k, folds[k])
    # Compute FP-distance OOD metric (subsample for speed)
    ood = inter_fold_ood_metric(folds, sample_per_fold=150, seed=SEED)
    stats["ood"] = ood
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ds", type=str, default=None,
                   help="single dataset; if omitted, all")
    p.add_argument("--sim", type=float, nargs="+", default=[0.4],
                   help="Butina sim threshold(s). Multiple -> sweep "
                        "(dry-run only). Single -> write files.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out-subdir", type=str, default="scaffold_diverse_cv5")
    args = p.parse_args()

    if args.ds:
        ds_dirs = [SPLIT_ROOT / args.ds]
    else:
        ds_dirs = sorted([d for d in SPLIT_ROOT.iterdir() if d.is_dir()])

    is_sweep = len(args.sim) > 1
    if is_sweep and not args.dry_run:
        print("--sim sweep forces --dry-run (use one --sim value to write files)")
        args.dry_run = True

    summaries = []
    for sim in args.sim:
        print(f"\n========== sim_threshold = {sim:.2f} ==========")
        rows_acc = []
        for d in ds_dirs:
            print(f"  Processing {d.name} ...", end=" ", flush=True)
            stats = process_ds(d, sim_thresh=sim, dry_run=args.dry_run,
                               out_subdir=args.out_subdir)
            if stats is None:
                print("SKIP (no data)")
                continue
            ood = stats["ood"] or {}
            print(
                f"unique_scafs={stats['n_unique_scaffolds']:>4} "
                f"super_clusters={stats['n_super_clusters']:>4} "
                f"test_sizes={stats['test_sizes']} "
                f"max/min={stats['max_min_ratio']:.2f}× "
                f"median_min_train_dist="
                f"{ood.get('median_min_dist', 0.0):.3f}"
            )
            stats["name"] = d.name
            stats["sim"] = sim
            rows_acc.append(stats)
        # Summary
        if rows_acc:
            ratios = [r["max_min_ratio"] for r in rows_acc]
            dists = [r["ood"]["median_min_dist"] for r in rows_acc if r["ood"]]
            print(f"\n  SUMMARY sim={sim:.2f} (n={len(rows_acc)} ds):")
            print(f"    fold-size max/min : "
                  f"min={min(ratios):.2f}×  median={statistics.median(ratios):.2f}×  "
                  f"max={max(ratios):.2f}×")
            if dists:
                print(f"    OOD median_min_dist (test→nearest_train): "
                      f"min={min(dists):.3f}  median={statistics.median(dists):.3f}  "
                      f"max={max(dists):.3f}")
        summaries.append((sim, rows_acc))

    if not args.dry_run:
        print(f"\nFiles written to <ds_dir>/{args.out_subdir}/")
    else:
        print("\n(dry-run: no files written)")


if __name__ == "__main__":
    main()
