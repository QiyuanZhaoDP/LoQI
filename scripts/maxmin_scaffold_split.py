#!/usr/bin/env python3
"""Plan-C scaffold splitter: Butina cluster -> MaxMin seed -> Voronoi assign -> greedy rebalance.

Compared to scaffold_diverse_cv5 (LPT bin-packing on Butina super-clusters),
this approach EXPLICITLY MAXIMIZES inter-fold Tanimoto distance:

  Step 1.  Compute Murcko scaffold per molecule.
  Step 2.  Butina-cluster scaffold ECFP fps at similarity ≥ sim_thresh.
           Result: super-clusters where each cluster's scaffolds are
           within distance (1-sim_thresh) of the cluster centroid.
  Step 3.  Pick K=5 fold SEEDS via MaxMin:
             - Start with the largest super-cluster as seed_0 (balance hint).
             - Iteratively pick the super-cluster whose minimum Tanimoto
               distance to the existing seeds is maximized.
  Step 4.  Voronoi assignment: every non-seed super-cluster -> nearest seed.
           Now fold k contains seed_k + all super-clusters closer to seed_k
           than to any other seed.
  Step 5.  Greedy rebalance: while max_fold / min_fold > balance_thresh,
           move the smallest super-cluster in max_fold -> min_fold. Stop
           when balance is satisfied OR no swap reduces the ratio further.

Trade-off vs LPT (scaffold_diverse_cv5):
  + MaxMin DIRECTLY MAXIMIZES inter-fold separation in fingerprint space
    -> stronger continuous OOD
  - Voronoi assignment is greedy; balance can drift before rebalance step
  - Rebalance step trades OOD strength for fold-size uniformity (a small
    cluster moved into a different fold likely sits closer to the WRONG
    fold's centroid — but it's small so OOD impact is minor)

Usage:
    # Sweep over sim threshold on 5 viz datasets (dry-run, no files written)
    python scripts/maxmin_scaffold_split.py \\
        --ds BP_K Hf_gas_kJmol ESOL_logS dielectric_298K PPBR_pct \\
        --sim 0.30 --balance 1.3 --dry-run

    # Generate splits for all 43 ds
    python scripts/maxmin_scaffold_split.py --sim 0.30 --balance 1.3
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
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, ECFP_R, ECFP_BITS)


def load_full(ds_dir: Path):
    src = ds_dir / "scaffold_cv5"
    rows, seen = [], set()
    for kind in ["train", "valid", "test"]:
        fp = src / f"cv1_{kind}.csv"
        if not fp.exists(): continue
        with open(fp) as fh:
            r = csv.reader(fh); next(r, None)
            for row in r:
                if len(row) < 2: continue
                smi, tgt = row[0], row[1]
                if smi in seen: continue
                seen.add(smi); rows.append((smi, tgt))
    return rows


def maxmin_split(rows, sim_thresh=0.30, balance_thresh=1.3,
                 n_folds=N_FOLDS, seed=SEED, valid_frac=VALID_FRAC,
                 algorithm="maxmin"):
    """Splitter. algorithm in {maxmin, swap, lloyd}.

      maxmin  : Plan-C as-spec: MaxMin seed + Voronoi + greedy size rebalance.
                Strong OOD but balance can be bad.
      swap    : Fix 1: LPT-init balanced + greedy swap to improve OOD while
                respecting balance constraint.
      lloyd   : Fix 2: Iterative Lloyd with size budget. Converges to a
                balance-OOD compromise via repeated reassignment.
    """

    # ---- Step 1: scaffolds per molecule + group ----
    scaf_of = {smi: get_scaffold(smi) for smi, _ in rows}
    scaf_rows = defaultdict(list)
    for smi, tgt in rows:
        scaf_rows[scaf_of[smi]].append((smi, tgt))
    unique_scafs = list(scaf_rows.keys())

    # ---- Step 2: FP per scaffold ----
    scaf_fps = []
    for s in unique_scafs:
        if s.startswith("__"):
            rep_smi = scaf_rows[s][0][0]
            scaf_fps.append(get_fp(rep_smi))
        else:
            scaf_fps.append(get_fp(s))
    valid_mask = [fp is not None for fp in scaf_fps]
    valid_idx = [i for i, ok in enumerate(valid_mask) if ok]
    valid_fps = [scaf_fps[i] for i in valid_idx]
    n_v = len(valid_fps)

    # ---- Step 2b: Butina cluster scaffolds into super-clusters ----
    dists = []
    for i in range(1, n_v):
        sims = BulkTanimotoSimilarity(valid_fps[i], valid_fps[:i])
        dists.extend(1.0 - s for s in sims)
    butina = Butina.ClusterData(dists, n_v, 1.0 - sim_thresh, isDistData=True)
    # butina: tuple of tuples; first idx in each tuple is the cluster centroid

    super_clusters = []          # list of list-of-(smi, tgt)
    cluster_centroids = []       # list of fingerprints (centroid per cluster)
    for cg in butina:
        mols = []
        for idx_v in cg:
            orig = valid_idx[idx_v]
            mols.extend(scaf_rows[unique_scafs[orig]])
        super_clusters.append(mols)
        # Butina puts centroid first
        first_orig = valid_idx[cg[0]]
        cluster_centroids.append(scaf_fps[first_orig])
    # Add invalid-fp scaffolds as singleton clusters (rare)
    for i, fp in enumerate(scaf_fps):
        if fp is None:
            super_clusters.append(scaf_rows[unique_scafs[i]])
            cluster_centroids.append(None)
    n_clusters = len(super_clusters)
    cluster_sizes = [len(c) for c in super_clusters]

    # ---- Step 3-5: Algorithm dispatch ----
    if algorithm == "maxmin":
        fold_of_cluster, rebalance_moves, seeds = _maxmin_voronoi_rebalance(
            cluster_centroids, cluster_sizes, n_folds, balance_thresh)
        inter_seed_dists = _inter_seed_dists(cluster_centroids, seeds)
    elif algorithm == "swap":
        initial_assignment = _lpt_pack(super_clusters, n_folds)
        D = _pairwise_cluster_dist(cluster_centroids)
        fold_of_cluster, rebalance_moves = _swap_optimize(
            initial_assignment, D, cluster_sizes, n_folds, balance_thresh)
        seeds = None
        inter_seed_dists = []
    elif algorithm == "lloyd":
        fold_of_cluster, n_lloyd_iters = _lloyd_balanced(
            cluster_centroids, cluster_sizes, n_folds, balance_thresh)
        rebalance_moves = n_lloyd_iters
        seeds = None
        inter_seed_dists = []
    elif algorithm == "hybrid":
        # Run Lloyd first (high OOD, possibly imbalanced), then swap to
        # rebalance while keeping as much OOD as possible.
        lloyd_fold, n_lloyd_iters = _lloyd_balanced(
            cluster_centroids, cluster_sizes, n_folds,
            balance_thresh=max(balance_thresh, 2.0))   # loose for Lloyd
        D = _pairwise_cluster_dist(cluster_centroids)
        # Swap-rebalance towards the strict balance_thresh
        fold_of_cluster, n_swap_moves = _swap_optimize(
            lloyd_fold, D, cluster_sizes, n_folds, balance_thresh)
        rebalance_moves = n_lloyd_iters + n_swap_moves
        seeds = None
        inter_seed_dists = []
    else:
        raise ValueError(f"Unknown algorithm: {algorithm!r}")

    # ---- Step 6: Build per-fold molecule lists ----
    buckets = [[] for _ in range(n_folds)]
    for c in range(n_clusters):
        buckets[fold_of_cluster[c]].extend(super_clusters[c])

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
        folds[k + 1] = {
            "train": pool_shuf[n_valid:],
            "valid": pool_shuf[:n_valid],
            "test":  test_set,
        }

    test_sizes = [len(folds[k]["test"]) for k in sorted(folds)]
    stats = {
        "algorithm":          algorithm,
        "n_unique_scaffolds": len(unique_scafs),
        "n_super_clusters":   n_clusters,
        "n_seeds":            len(seeds) if seeds else None,
        "test_sizes":         test_sizes,
        "max_min_ratio":      max(test_sizes) / max(min(test_sizes), 1),
        "rebalance_moves":    rebalance_moves,
        "inter_seed_dist_min":     min(inter_seed_dists) if inter_seed_dists else None,
        "inter_seed_dist_median":  float(np.median(inter_seed_dists)) if inter_seed_dists else None,
        "balance_threshold":  balance_thresh,
        "sim_threshold":      sim_thresh,
    }
    return folds, stats


def _fold_sizes(fold_of_cluster, cluster_sizes, n_folds):
    sizes = [0] * n_folds
    for c, f in fold_of_cluster.items():
        sizes[f] += cluster_sizes[c]
    return sizes


def _pairwise_cluster_dist(centroids):
    """Precompute n x n Tanimoto distance matrix between cluster centroids."""
    n = len(centroids)
    D = np.ones((n, n), dtype=np.float32)
    for i in range(n):
        if centroids[i] is None: continue
        sims = BulkTanimotoSimilarity(centroids[i], [c for c in centroids if c is not None])
        # Re-index since None centroids were excluded
        valid_pos = [j for j, c in enumerate(centroids) if c is not None]
        for k, j in enumerate(valid_pos):
            D[i, j] = 1.0 - sims[k]
    np.fill_diagonal(D, 0.0)
    return D


def _ood_score(fold_of_cluster, D, cluster_sizes):
    """OOD proxy: sum over clusters of min Tanimoto distance to ANY cluster in a different fold.
    Weighted by cluster size (bigger clusters matter more)."""
    n = len(cluster_sizes)
    total = 0.0
    for c in range(n):
        if fold_of_cluster[c] is None: continue
        my_fold = fold_of_cluster[c]
        # Find min distance to any cluster in a different fold
        min_d = 2.0
        for c2 in range(n):
            if c == c2: continue
            if fold_of_cluster.get(c2) is None: continue
            if fold_of_cluster[c2] == my_fold: continue
            d = D[c, c2]
            if d < min_d: min_d = d
        total += min_d * cluster_sizes[c]
    return total


def _maxmin_voronoi_rebalance(cluster_centroids, cluster_sizes, n_folds, balance_thresh):
    """Plan-C original: MaxMin seeds + Voronoi + greedy size-only rebalance."""
    n_clusters = len(cluster_sizes)
    valid_idx = [i for i, c in enumerate(cluster_centroids) if c is not None]
    valid_centroids = [cluster_centroids[i] for i in valid_idx]

    seed_local = [int(np.argmax([cluster_sizes[i] for i in valid_idx]))]
    while len(seed_local) < n_folds:
        min_d = np.full(len(valid_centroids), np.inf)
        for s in seed_local:
            sims = np.asarray(BulkTanimotoSimilarity(valid_centroids[s], valid_centroids))
            d = 1.0 - sims
            min_d = np.minimum(min_d, d)
        for s in seed_local:
            min_d[s] = -np.inf
        seed_local.append(int(np.argmax(min_d)))
    seeds = [valid_idx[s] for s in seed_local]

    fold_of_cluster = {}
    for k, s in enumerate(seeds):
        fold_of_cluster[s] = k
    for c in range(n_clusters):
        if c in seeds: continue
        if cluster_centroids[c] is None:
            sizes_now = _fold_sizes(fold_of_cluster, cluster_sizes, n_folds)
            fold_of_cluster[c] = int(np.argmin(sizes_now)); continue
        sims = [BulkTanimotoSimilarity(cluster_centroids[c],
                                        [cluster_centroids[s]])[0] for s in seeds]
        fold_of_cluster[c] = int(np.argmax(sims))

    fold_sizes = _fold_sizes(fold_of_cluster, cluster_sizes, n_folds)
    moves = 0
    for _ in range(10 * n_clusters):
        ratio = max(fold_sizes) / max(min(fold_sizes), 1)
        if ratio <= balance_thresh: break
        max_fold = int(np.argmax(fold_sizes))
        min_fold = int(np.argmin(fold_sizes))
        src = sorted([c for c, f in fold_of_cluster.items() if f == max_fold],
                     key=lambda c: cluster_sizes[c])
        moved = False
        for cand in src:
            new_max = fold_sizes[max_fold] - cluster_sizes[cand]
            new_min = fold_sizes[min_fold] + cluster_sizes[cand]
            trial = list(fold_sizes)
            trial[max_fold] = new_max; trial[min_fold] = new_min
            if max(trial) / max(min(trial), 1) < ratio:
                fold_of_cluster[cand] = min_fold
                fold_sizes = trial
                moves += 1; moved = True; break
        if not moved: break
    return fold_of_cluster, moves, seeds


def _inter_seed_dists(cluster_centroids, seeds):
    if seeds is None: return []
    out = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            sim = BulkTanimotoSimilarity(cluster_centroids[seeds[i]],
                                          [cluster_centroids[seeds[j]]])[0]
            out.append(1.0 - sim)
    return out


def _lpt_pack(super_clusters, n_folds):
    """Standard LPT bin-pack: clusters sorted desc, assign each to smallest bucket."""
    order = sorted(range(len(super_clusters)), key=lambda i: -len(super_clusters[i]))
    fold_of_cluster = {}
    fold_sizes = [0] * n_folds
    for c in order:
        idx = int(np.argmin(fold_sizes))
        fold_of_cluster[c] = idx
        fold_sizes[idx] += len(super_clusters[c])
    return fold_of_cluster


def _rebalance_phase(fold_arr, sizes_arr, fold_sizes, cluster_sizes, n_folds,
                     balance_thresh, max_iter=2000):
    """Move clusters from largest to smallest fold until balance satisfied.
    Doesn't track OOD; just fixes balance as fast as possible."""
    fold_arr = fold_arr.copy()
    fold_sizes = fold_sizes.copy()
    n = fold_arr.size
    for _ in range(max_iter):
        ratio = fold_sizes.max() / max(fold_sizes.min(), 1)
        if ratio <= balance_thresh:
            break
        max_f = int(np.argmax(fold_sizes))
        min_f = int(np.argmin(fold_sizes))
        # Find cluster in max_f whose move to min_f reduces ratio most
        candidates = np.where(fold_arr == max_f)[0]
        if len(candidates) == 0: break
        best_idx, best_ratio = None, ratio
        for c in candidates:
            new_sizes = fold_sizes.copy()
            new_sizes[max_f] -= cluster_sizes[c]
            new_sizes[min_f] += cluster_sizes[c]
            new_ratio = new_sizes.max() / max(new_sizes.min(), 1)
            if new_ratio < best_ratio:
                best_ratio = new_ratio
                best_idx = c
        if best_idx is None: break
        fold_arr[best_idx] = min_f
        fold_sizes[max_f] -= cluster_sizes[best_idx]
        fold_sizes[min_f] += cluster_sizes[best_idx]
    return fold_arr, fold_sizes


def _swap_optimize(initial_assignment, D, cluster_sizes, n_folds, balance_thresh,
                   max_iter=80, first_improvement=True):
    """FIX 1: LPT-init + greedy swap to improve OOD while preserving balance.

    Vectorized & incremental implementation:
      - Precompute D (pairwise cluster Tanimoto distance) once.
      - Maintain fold_arr (per-cluster fold index, np.ndarray) for fast masking.
      - Score = Σ_c cluster_size[c] * (min distance from c to any cluster in a
        different fold). Compute vectorized via numpy: for each cluster, mask
        out same-fold entries in D[c] then take min.
      - first_improvement=True: apply the FIRST move that improves score
        (faster than scanning all moves for the best each iteration).

    For 1-2k super-clusters this completes in ≤30 s.
    """
    fold_arr = np.array([initial_assignment[c] for c in range(len(cluster_sizes))],
                        dtype=np.int32)
    sizes_arr = np.array(cluster_sizes, dtype=np.float64)
    n = fold_arr.size
    fold_sizes = np.array([sizes_arr[fold_arr == k].sum() for k in range(n_folds)])

    # ── Phase 0: hard rebalance if init is too imbalanced ─────────────────
    init_ratio = fold_sizes.max() / max(fold_sizes.min(), 1)
    if init_ratio > balance_thresh:
        fold_arr, fold_sizes = _rebalance_phase(
            fold_arr, sizes_arr, fold_sizes, cluster_sizes, n_folds, balance_thresh)

    # Vectorized score = sum_c w_c * min_{c': f(c')!=f(c)} D[c, c']
    def compute_nearest_other_fold():
        nod = np.empty(n, dtype=np.float32)
        for c in range(n):
            mask = fold_arr != fold_arr[c]
            mask[c] = False  # don't self-compare
            if mask.any():
                nod[c] = D[c, mask].min()
            else:
                nod[c] = 0.0
        return nod

    nod = compute_nearest_other_fold()
    cur_score = float((nod * sizes_arr).sum())
    moves = 0

    for it in range(max_iter):
        improved = False
        # Random order over clusters
        order = np.random.RandomState(SEED + it).permutation(n)
        for c in order:
            old_fold = int(fold_arr[c])
            csize = sizes_arr[c]
            for new_fold in range(n_folds):
                if new_fold == old_fold: continue
                # Check balance
                trial = fold_sizes.copy()
                trial[old_fold] -= csize
                trial[new_fold] += csize
                if trial.max() / max(trial.min(), 1) > balance_thresh:
                    continue
                # Simulate: temporarily change c's fold, recompute affected nod
                fold_arr[c] = new_fold
                # Only clusters whose nearest-other-fold neighbor was c (or whose
                # new nearest changes due to c switching fold) need recompute.
                # For safety just recompute c + all clusters in old_fold + new_fold.
                affected = np.where((fold_arr == old_fold) | (fold_arr == new_fold))[0]
                # affected includes c (now in new_fold)
                new_nod = nod.copy()
                for ca in affected:
                    mask = fold_arr != fold_arr[ca]
                    mask[ca] = False
                    new_nod[ca] = D[ca, mask].min() if mask.any() else 0.0
                trial_score = float((new_nod * sizes_arr).sum())
                if trial_score > cur_score + 1e-6:
                    nod = new_nod
                    fold_sizes = trial
                    cur_score = trial_score
                    moves += 1
                    improved = True
                    if first_improvement:
                        break
                else:
                    fold_arr[c] = old_fold  # revert
            if improved and first_improvement:
                break
        if not improved:
            break

    return {c: int(fold_arr[c]) for c in range(n)}, moves


def _lloyd_balanced(centroids, cluster_sizes, n_folds, balance_thresh, max_iter=20):
    """FIX 2: Iterative Lloyd with size budget.

    At each iteration:
      1. Compute fold centroid = mean fingerprint of clusters in fold.
      2. For each cluster, compute distance to each fold centroid.
      3. Greedy reassign:
         - Sort clusters by "preference strength" (gap between 1st-best
           and 2nd-best fold distance). Strong preferences placed first.
         - For each cluster, assign to nearest fold whose total size is
           under max_size = target * balance_thresh.
         - Fall back to smallest fold if all preferred folds are full.
      4. Repeat until assignment stable."""
    n_clusters = len(cluster_sizes)
    total = sum(cluster_sizes)
    target = total / n_folds
    max_size = target * balance_thresh

    # Initial: MaxMin seeds + Voronoi
    valid_idx = [i for i, c in enumerate(centroids) if c is not None]
    valid_centroids = [centroids[i] for i in valid_idx]
    seed_local = [int(np.argmax([cluster_sizes[i] for i in valid_idx]))]
    while len(seed_local) < n_folds:
        min_d = np.full(len(valid_centroids), np.inf)
        for s in seed_local:
            sims = np.asarray(BulkTanimotoSimilarity(valid_centroids[s], valid_centroids))
            d = 1.0 - sims
            min_d = np.minimum(min_d, d)
        for s in seed_local:
            min_d[s] = -np.inf
        seed_local.append(int(np.argmax(min_d)))
    seeds = [valid_idx[s] for s in seed_local]

    fold_of_cluster = {}
    for k, s in enumerate(seeds):
        fold_of_cluster[s] = k
    for c in range(n_clusters):
        if c in seeds: continue
        if centroids[c] is None:
            fold_of_cluster[c] = int(np.argmin(_fold_sizes(fold_of_cluster, cluster_sizes, n_folds)))
            continue
        sims = [BulkTanimotoSimilarity(centroids[c], [centroids[s]])[0] for s in seeds]
        fold_of_cluster[c] = int(np.argmax(sims))

    for it in range(max_iter):
        # 1. Compute float-valued fold centroids (mean of member fingerprints)
        fold_centroids_arr = []
        for k in range(n_folds):
            members = [np.array(centroids[c]) for c in range(n_clusters)
                       if fold_of_cluster.get(c) == k and centroids[c] is not None]
            if not members:
                fold_centroids_arr.append(None); continue
            stacked = np.stack(members).astype(np.float32)
            fold_centroids_arr.append(stacked.mean(axis=0))

        # 2. For each cluster, distance to each fold centroid (Tanimoto on float)
        cluster_dists = []
        for c in range(n_clusters):
            if centroids[c] is None:
                cluster_dists.append([0.5] * n_folds); continue
            a = np.array(centroids[c]).astype(np.float32)
            row = []
            for k in range(n_folds):
                b = fold_centroids_arr[k]
                if b is None:
                    row.append(2.0); continue
                # Float Tanimoto: dot / (||a||+||b||-dot)
                dot = float((a * b).sum())
                an = float(a.sum())
                bn = float(b.sum())
                denom = an + bn - dot
                sim = dot / denom if denom > 1e-9 else 0.0
                row.append(1.0 - sim)
            cluster_dists.append(row)

        # 3. Greedy reassignment with size budget
        priority = sorted(range(n_clusters),
                          key=lambda c: -(sorted(cluster_dists[c])[1] - min(cluster_dists[c])))
        new_assignment = {}
        fold_sizes_now = [0] * n_folds
        for c in priority:
            ranked = sorted(range(n_folds), key=lambda k: cluster_dists[c][k])
            assigned = False
            for k in ranked:
                if fold_sizes_now[k] + cluster_sizes[c] <= max_size:
                    new_assignment[c] = k
                    fold_sizes_now[k] += cluster_sizes[c]
                    assigned = True; break
            if not assigned:
                k = int(np.argmin(fold_sizes_now))
                new_assignment[c] = k
                fold_sizes_now[k] += cluster_sizes[c]

        # Check convergence
        same = all(fold_of_cluster.get(c) == new_assignment[c] for c in range(n_clusters))
        fold_of_cluster = new_assignment
        if same:
            break
    return fold_of_cluster, it + 1


def inter_fold_ood_metric(folds, sample_per_fold=150, seed=42, per_fold=False):
    """For each test mol, distance to nearest train+valid mol."""
    rng = random.Random(seed)
    all_min, per_fold_min = [], {}
    for k in folds:
        test = folds[k]["test"]
        train = folds[k]["train"] + folds[k]["valid"]
        test_smp  = rng.sample(test, min(sample_per_fold, len(test)))
        train_smp = rng.sample(train, min(sample_per_fold * 4, len(train)))
        test_fps  = [get_fp(s) for s, _ in test_smp]
        train_fps = [get_fp(s) for s, _ in train_smp]
        test_fps  = [f for f in test_fps  if f is not None]
        train_fps = [f for f in train_fps if f is not None]
        if not test_fps or not train_fps:
            continue
        fold_min = []
        for tfp in test_fps:
            sims = BulkTanimotoSimilarity(tfp, train_fps)
            fold_min.append(1.0 - max(sims))
        all_min.extend(fold_min)
        per_fold_min[k] = {
            "median": statistics.median(fold_min),
            "q25":    sorted(fold_min)[len(fold_min)//4],
            "q75":    sorted(fold_min)[3*len(fold_min)//4],
            "n":      len(fold_min),
        }
    overall = None
    if all_min:
        srt = sorted(all_min)
        overall = {
            "median": statistics.median(all_min),
            "q25":    srt[len(srt)//4],
            "q75":    srt[3*len(srt)//4],
            "n":      len(all_min),
        }
    return per_fold_min, overall


def write_fold(out_dir: Path, k: int, fold_data: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind in ["train", "valid", "test"]:
        fp = out_dir / f"cv{k}_{kind}.csv"
        with open(fp, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["SMILES", "TARGET"])
            for smi, tgt in fold_data[kind]:
                w.writerow([smi, tgt])


def process_ds(ds_dir, sim_thresh, balance_thresh, dry_run, out_subdir,
               per_fold_ood=False, algorithm="maxmin"):
    rows = load_full(ds_dir)
    if not rows: return None
    folds, stats = maxmin_split(rows, sim_thresh=sim_thresh,
                                 balance_thresh=balance_thresh,
                                 n_folds=N_FOLDS, seed=SEED,
                                 valid_frac=VALID_FRAC, algorithm=algorithm)
    if not dry_run:
        out_dir = ds_dir / out_subdir
        for k in folds:
            write_fold(out_dir, k, folds[k])
    pf, overall = inter_fold_ood_metric(folds, sample_per_fold=150,
                                         seed=SEED, per_fold=per_fold_ood)
    stats["ood"] = overall
    stats["ood_per_fold"] = pf if per_fold_ood else None
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ds", type=str, nargs="*", default=None)
    p.add_argument("--sim", type=float, nargs="+", default=[0.30],
                   help="Butina similarity threshold(s). Multiple -> sweep (dry-run).")
    p.add_argument("--balance", type=float, nargs="+", default=[1.3],
                   help="Max fold-size ratio after rebalance.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out-subdir", type=str, default="scaffold_maxmin_cv5")
    p.add_argument("--per-fold-ood", action="store_true",
                   help="Also report per-fold OOD distance.")
    p.add_argument("--algorithm", type=str, default="maxmin",
                   choices=["maxmin", "swap", "lloyd", "hybrid"],
                   help="maxmin = Plan-C (MaxMin seeds + Voronoi + size rebalance), "
                        "swap = Fix 1 (LPT init + OOD-improving swaps), "
                        "lloyd = Fix 2 (iterative Lloyd with size budget), "
                        "hybrid = Lloyd then Swap-rebalance (best of both).")
    args = p.parse_args()

    if args.ds:
        ds_dirs = [SPLIT_ROOT / d for d in args.ds]
    else:
        ds_dirs = sorted([d for d in SPLIT_ROOT.iterdir() if d.is_dir()])

    sweep = len(args.sim) > 1 or len(args.balance) > 1
    if sweep and not args.dry_run:
        print("Sweep mode forces --dry-run.")
        args.dry_run = True

    for sim in args.sim:
        for bal in args.balance:
            print(f"\n========== sim={sim:.2f}  balance_thresh={bal}  ==========")
            summaries = []
            for d in ds_dirs:
                if not d.exists(): continue
                print(f"  {d.name:<28}", end=" ", flush=True)
                stats = process_ds(d, sim_thresh=sim, balance_thresh=bal,
                                   dry_run=args.dry_run, out_subdir=args.out_subdir,
                                   per_fold_ood=args.per_fold_ood,
                                   algorithm=args.algorithm)
                if stats is None:
                    print("SKIP")
                    continue
                ood = stats["ood"] or {}
                inter = stats.get("inter_seed_dist_median")
                inter_str = f"{inter:.3f}" if inter is not None else "  n/a"
                print(
                    f"super={stats['n_super_clusters']:>4} "
                    f"sizes={stats['test_sizes']} "
                    f"max/min={stats['max_min_ratio']:.2f}× "
                    f"seed_dist={inter_str} "
                    f"median_min_train_dist={(ood.get('median') or 0.0):.3f} "
                    f"moves={stats['rebalance_moves']}"
                )
                if args.per_fold_ood and stats["ood_per_fold"]:
                    pf = stats["ood_per_fold"]
                    pf_str = "  per_fold_dist: " + " ".join(
                        f"{k}={pf[k]['median']:.3f}" for k in sorted(pf))
                    print(f"    {pf_str}")
                stats["name"] = d.name
                summaries.append(stats)

            # Summary
            if summaries:
                ratios = [r["max_min_ratio"] for r in summaries]
                dists  = [r["ood"]["median"] for r in summaries if r["ood"]]
                seed_d = [r["inter_seed_dist_median"] for r in summaries
                          if r.get("inter_seed_dist_median") is not None]
                print(f"\n  SUMMARY sim={sim:.2f} bal={bal} (n={len(summaries)} ds):")
                print(f"    fold-size max/min : min={min(ratios):.2f}× "
                      f"median={statistics.median(ratios):.2f}× max={max(ratios):.2f}×")
                if dists:
                    print(f"    test->nearest_train dist : "
                          f"min={min(dists):.3f} median={statistics.median(dists):.3f} max={max(dists):.3f}")
                if seed_d:
                    print(f"    inter-seed dist (median) : "
                          f"min={min(seed_d):.3f} median={statistics.median(seed_d):.3f} max={max(seed_d):.3f}")

    if not args.dry_run:
        print(f"\nFiles written to <ds_dir>/{args.out_subdir}/")


if __name__ == "__main__":
    main()
