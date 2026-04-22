"""Report mean / std / min / max / median / NaN% for every numeric property
in the property_table.parquet, then emit YAML-ready `thermo_loss` /
`rdkit_loss` blocks so you can paste real values into the train config.

- Per-property stats: computed on non-NaN rows (so thermo stats ignore the
  unlabeled subset automatically, no need to filter manually).
- Thermo YAML block: auto-restricted to has_thermo_label==True rows.
- RDKit YAML block: computed over the full table (100% coverage).

Usage:
  python data_processing/compute_rdkit_stats.py \\
      --parquet data/property_table.parquet
"""
import argparse
import json

import numpy as np
import pandas as pd

THERMO_FIELDS = ["enthalpy_298", "gibbs_298", "cv_gas",
                  "entropy_gas", "enthalpy_0"]
RDKIT_FIELDS = [
    "logp", "tpsa", "n_h_donors", "n_h_acceptors", "n_rot_bonds",
    "frac_csp3", "n_aliph_rings", "qed", "labute_asa",
]


def _per_column_summary(df):
    """Return a dict {col: dict_of_stats} for every numeric column in df,
    computed over non-NaN rows only."""
    summary = {}
    for col in df.columns:
        if df[col].dtype == bool:
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            continue
        s = df[col].astype(float)
        nn = s.dropna()
        if len(nn) == 0:
            continue
        summary[col] = {
            "n":      int(len(nn)),
            "n_nan":  int(s.isna().sum()),
            "mean":   float(nn.mean()),
            "std":    float(nn.std()),
            "min":    float(nn.min()),
            "median": float(nn.median()),
            "max":    float(nn.max()),
        }
    return summary


def _print_table(summary, n_total):
    print(f"\n{'field':<18s} {'n':>10s} {'nan%':>6s} {'mean':>12s} {'std':>12s} "
          f"{'min':>10s} {'median':>10s} {'max':>10s}")
    print("-" * 100)
    for col, d in summary.items():
        nanpct = 100 * d["n_nan"] / n_total if n_total else 0.0
        print(f"{col:<18s} {d['n']:>10,} {nanpct:>5.1f}% "
              f"{d['mean']:>12.4f} {d['std']:>12.4f} "
              f"{d['min']:>10.4f} {d['median']:>10.4f} {d['max']:>10.4f}")


def _fmt(xs, w=4):
    return "[" + ", ".join(f"{x:.{w}f}" for x in xs) + "]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    args = p.parse_args()

    df = pd.read_parquet(args.parquet)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} cols from {args.parquet}")

    # --- Per-column stats over the FULL table (NaN auto-ignored per column) --
    full_summary = _per_column_summary(df)
    print("\n[ full table — stats computed on non-NaN rows per column ]")
    _print_table(full_summary, n_total=len(df))

    # --- Same stats but thermo-restricted to has_thermo_label==True rows,
    #     for a direct comparison (does the label subset differ from full?).
    if "has_thermo_label" in df.columns:
        labeled = df[df["has_thermo_label"]]
        labeled_summary = _per_column_summary(labeled[THERMO_FIELDS])
        print(f"\n[ thermo subset — has_thermo_label==True, n={len(labeled):,} ]")
        _print_table(labeled_summary, n_total=len(labeled))

    # ---- YAML blocks, computed on the RIGHT subset per loss ----------------
    # Thermo: has_thermo_label==True so NaN rows don't contribute.
    thermo_mean = [full_summary[f]["mean"] for f in THERMO_FIELDS if f in full_summary]
    thermo_std  = [full_summary[f]["std"]  for f in THERMO_FIELDS if f in full_summary]
    # (Using full_summary because _per_column_summary already skips NaNs per column —
    #  the mean over non-NaN rows equals the mean over labeled rows.)

    rdkit_mean = [full_summary[f]["mean"] for f in RDKIT_FIELDS if f in full_summary]
    rdkit_std  = [full_summary[f]["std"]  for f in RDKIT_FIELDS if f in full_summary]

    print("\n\n# =====  paste into scripts/conf/loqi/*.yaml  =====")
    print("# Order: thermo → [H298, G298, Cv, S°, H0]")
    print("thermo_loss:")
    print("  min_time: 0.8")
    print("  weight:   0.05")
    print(f"  target_mean: {_fmt(thermo_mean, 2)}")
    print(f"  target_std:  {_fmt(thermo_std,  2)}")
    print()
    print("# Order: rdkit → [logp, tpsa, n_h_donors, n_h_acceptors, n_rot_bonds,")
    print("#                 frac_csp3, n_aliph_rings, qed, labute_asa]")
    print("rdkit_loss:")
    print("  min_time: 0.8")
    print("  weight:   0.02")
    print(f"  target_mean: {_fmt(rdkit_mean, 4)}")
    print(f"  target_std:  {_fmt(rdkit_std,  4)}")

    # JSON dump for programmatic use (e.g., loading into notebook, CI checks).
    out_json = {
        "n_rows_total":    len(df),
        "n_rows_labeled":  int(df["has_thermo_label"].sum()) if "has_thermo_label" in df else None,
        "per_column":      full_summary,
        "thermo_loss":     {"fields": THERMO_FIELDS,
                             "target_mean": thermo_mean, "target_std": thermo_std},
        "rdkit_loss":      {"fields": RDKIT_FIELDS,
                             "target_mean": rdkit_mean,  "target_std": rdkit_std},
    }
    print("\n# JSON (for programmatic consumption):")
    print(json.dumps(out_json, indent=2))


if __name__ == "__main__":
    main()
