#!/usr/bin/env python3
"""Balanced scaffold k-fold splitter.

Algorithm (LPT bin-packing on scaffold clusters):
  1. Load full dataset (union of any single fold's train+valid+test).
  2. Compute Bemis-Murcko scaffold SMILES for each molecule.
  3. Group molecules by scaffold (= scaffold cluster).
  4. Sort scaffold clusters by size (descending).
  5. Greedily assign each cluster to the smallest current bucket.
  6. Each bucket becomes one fold's test set.
  7. Valid set = 10% randomly sampled from each fold's train pool.

Output: <ds>/scaffold_balanced_cv5/{cv1..5}_{train,valid,test}.csv
Format: SMILES,TARGET  (same as existing splits)
"""
import argparse
import csv
import random
from pathlib import Path
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


REPO_ROOT = Path(__file__).resolve().parent.parent
SPLIT_ROOT = REPO_ROOT / "downstream_ft" / "0515_final" / "Split"
N_FOLDS = 5
VALID_FRAC = 0.10
SEED = 42


def get_scaffold(smiles: str) -> str:
    """Return Bemis-Murcko scaffold SMILES, or '' for invalid/empty scaffold."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"__INVALID__:{smiles}"
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        scaf_smi = Chem.MolToSmiles(scaf)
        if not scaf_smi:
            # Acyclic molecule -> use canonical SMILES as its own cluster
            return f"__ACYCLIC__:{Chem.MolToSmiles(mol)}"
        return scaf_smi
    except Exception:
        return f"__ERR__:{smiles}"


def load_full_dataset(ds_dir: Path) -> list:
    """Recover full (SMILES, TARGET) list from any existing fold's union."""
    src = ds_dir / "random_cv5"
    rows = []
    seen = set()
    for kind in ["train", "valid", "test"]:
        fp = src / f"cv1_{kind}.csv"
        with open(fp) as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                smi, tgt = row[0], row[1]
                if smi in seen:
                    continue
                seen.add(smi)
                rows.append((smi, tgt))
    return rows


def balanced_scaffold_split(rows: list, n_folds: int = 5,
                            valid_frac: float = 0.10, seed: int = 42):
    """Returns dict {fold_idx (1..n_folds): {'train':[...], 'valid':[...], 'test':[...]}}."""
    rng = random.Random(seed)

    # 1. Cluster by scaffold
    clusters = defaultdict(list)
    for smi, tgt in rows:
        scaf = get_scaffold(smi)
        clusters[scaf].append((smi, tgt))

    # 2. Sort clusters by size descending
    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)

    # 3. LPT bin-packing: assign each cluster to smallest current bucket
    buckets = [[] for _ in range(n_folds)]
    bucket_sizes = [0] * n_folds
    for cluster in sorted_clusters:
        idx = bucket_sizes.index(min(bucket_sizes))
        buckets[idx].extend(cluster)
        bucket_sizes[idx] += len(cluster)

    # 4. For each fold: test = bucket_k, train_pool = others, valid = sample from train
    folds = {}
    for k in range(n_folds):
        test_set = buckets[k]
        train_pool = []
        for j in range(n_folds):
            if j != k:
                train_pool.extend(buckets[j])
        train_pool_shuffled = train_pool.copy()
        rng_local = random.Random(seed + k)
        rng_local.shuffle(train_pool_shuffled)
        n_valid = max(1, int(round(len(train_pool_shuffled) * valid_frac)))
        valid_set = train_pool_shuffled[:n_valid]
        train_set = train_pool_shuffled[n_valid:]
        folds[k + 1] = {
            "train": train_set,
            "valid": valid_set,
            "test": test_set,
        }
    return folds, bucket_sizes


def write_fold(out_dir: Path, fold_idx: int, fold_data: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind in ["train", "valid", "test"]:
        fp = out_dir / f"cv{fold_idx}_{kind}.csv"
        with open(fp, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["SMILES", "TARGET"])
            for smi, tgt in fold_data[kind]:
                w.writerow([smi, tgt])


def process_dataset(ds_dir: Path, verbose: bool = True) -> dict:
    src = ds_dir / "random_cv5"
    if not src.exists():
        return None
    rows = load_full_dataset(ds_dir)
    if not rows:
        return None
    folds, bucket_sizes = balanced_scaffold_split(
        rows, n_folds=N_FOLDS, valid_frac=VALID_FRAC, seed=SEED
    )
    out_dir = ds_dir / "scaffold_balanced_cv5"
    for k, fold_data in folds.items():
        write_fold(out_dir, k, fold_data)
    test_sizes = [len(folds[k]["test"]) for k in sorted(folds)]
    train_sizes = [len(folds[k]["train"]) for k in sorted(folds)]
    summary = {
        "name": ds_dir.name,
        "total": len(rows),
        "test_sizes": test_sizes,
        "train_sizes": train_sizes,
        "bucket_sizes": bucket_sizes,
        "max_min_ratio": max(test_sizes) / min(test_sizes) if min(test_sizes) > 0 else float("inf"),
    }
    if verbose:
        print(f"  {ds_dir.name:<28} total={len(rows):>5}  "
              f"test_sizes={test_sizes}  "
              f"train_sizes={train_sizes}  "
              f"max/min={summary['max_min_ratio']:.2f}x")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ds", type=str, default=None,
                   help="Single dataset name; if omitted, processes all.")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute splits and print summary but do not write files.")
    args = p.parse_args()

    if args.ds:
        ds_dirs = [SPLIT_ROOT / args.ds]
    else:
        ds_dirs = sorted([d for d in SPLIT_ROOT.iterdir() if d.is_dir()])

    print(f"Processing {len(ds_dirs)} datasets under {SPLIT_ROOT}")
    summaries = []
    for d in ds_dirs:
        if args.dry_run:
            rows = load_full_dataset(d)
            if not rows:
                continue
            folds, _ = balanced_scaffold_split(
                rows, n_folds=N_FOLDS, valid_frac=VALID_FRAC, seed=SEED
            )
            test_sizes = [len(folds[k]["test"]) for k in sorted(folds)]
            train_sizes = [len(folds[k]["train"]) for k in sorted(folds)]
            ratio = max(test_sizes) / min(test_sizes) if min(test_sizes) > 0 else float("inf")
            print(f"  [DRY] {d.name:<28} total={len(rows):>5}  "
                  f"test_sizes={test_sizes}  max/min={ratio:.2f}x")
            summaries.append({"name": d.name, "test_sizes": test_sizes,
                             "max_min_ratio": ratio, "total": len(rows)})
        else:
            s = process_dataset(d, verbose=True)
            if s:
                summaries.append(s)

    print("\n===== Summary =====")
    print(f"  Datasets processed: {len(summaries)}")
    ratios = [s["max_min_ratio"] for s in summaries]
    if ratios:
        print(f"  test-fold max/min ratio:")
        print(f"    min={min(ratios):.2f}x  median={sorted(ratios)[len(ratios)//2]:.2f}x  "
              f"max={max(ratios):.2f}x")
        worst = sorted(summaries, key=lambda s: -s["max_min_ratio"])[:5]
        print(f"  Top-5 most imbalanced:")
        for s in worst:
            print(f"    {s['name']:<28} test_sizes={s['test_sizes']}  max/min={s['max_min_ratio']:.2f}x")


if __name__ == "__main__":
    main()
