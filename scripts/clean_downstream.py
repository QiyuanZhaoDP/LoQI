"""End-to-end cleaning of downstream_ft/ → downstream_ft/clean/.

Per dataset:
  * Flat CSV (Cp, V_cp, gas_Hf, ...): read, clean, write.
  * Pre-split (delaney_s, freesolv_s, lipo_s): merge train+valid+test,
    clean, write — output is FLAT (one CSV per dataset). The 5-fold CV
    pipeline merges these splits anyway, so we don't lose anything by
    dropping the official split here.

Cleaning steps (delegated to extract_smiles.filter_and_dedup):
    1. empty / NaN
    2. RDKit unparseable
    3. radicals (any unpaired electron)
    4. disconnected ('.' in canonical)
    5. elements outside LoQI's 17-atom set
    6. |formal_charge| > 1
    7. canonical-SMILES dedup (first occurrence kept)

Outputs:
    downstream_ft/clean/<name>.csv          — cleaned, deduplicated CSV
    downstream_ft/clean/cleaning_report.md  — per-dataset stats
    downstream_ft/clean/cleaning_report.json — same data, machine-readable

Run from repo root:
    python scripts/clean_downstream.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Local import — extract_smiles is in the same directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_smiles import filter_and_dedup, find_smiles_column  # noqa: E402

PRESPLIT = ["delaney_s", "freesolv_s", "lipo_s"]


def _load_dataset(name: str, root: Path) -> pd.DataFrame:
    if name in PRESPLIT:
        parts = []
        for split in ("train", "valid", "test"):
            p = root / name / f"{split}.csv"
            if not p.exists():
                raise SystemExit(f"missing split: {p}")
            d = pd.read_csv(p)
            d["_split"] = split
            parts.append(d)
        return pd.concat(parts, ignore_index=True)
    return pd.read_csv(root / f"{name}.csv")


def _row_md(name: str, s: dict, was_presplit: bool) -> str:
    bad_elem = s["bad_elements_breakdown"]
    bad_str = (", ".join(f"{e}:{n}" for e, n in
                         sorted(bad_elem.items(), key=lambda x: -x[1]))
               if bad_elem else "—")
    pct_kept = 100.0 * s["n_final"] / max(s["n_raw"], 1)
    return (
        f"| {name} | {'merged' if was_presplit else 'flat'} | "
        f"{s['n_raw']:,} | {s['n_empty']} | {s['n_unparseable']} | "
        f"{s['n_radical']} | {s['n_disconnected']} | {s['n_bad_elements']} | "
        f"{s['n_bad_charge']} | {s['n_canonical_dup']} | "
        f"**{s['n_final']:,}** ({pct_kept:.1f}%) | {bad_str} |"
    )


def main():
    root = Path("downstream_ft")
    if not root.exists():
        sys.exit(f"missing {root}")
    out_dir = root / "clean"
    out_dir.mkdir(exist_ok=True)

    flat = sorted(p.stem for p in root.glob("*.csv"))
    presplit_present = sorted(d.name for d in root.iterdir()
                              if d.is_dir() and d.name in PRESPLIT)
    datasets = flat + presplit_present

    all_stats: dict[str, dict] = {}
    md_rows: list[str] = []
    print("=" * 70)
    print(f"Cleaning downstream_ft/ → {out_dir}/")
    print("=" * 70)

    for name in datasets:
        was_presplit = name in PRESPLIT
        df = _load_dataset(name, root)
        smi_col = find_smiles_column(df)
        cleaned, stats = filter_and_dedup(df, smi_col, dedup_canonical=True)

        out_csv = out_dir / f"{name}.csv"
        cleaned.to_csv(out_csv, index=False)

        all_stats[name] = {**stats, "was_presplit": was_presplit,
                           "output": str(out_csv.relative_to(root.parent))}
        md_rows.append(_row_md(name, stats, was_presplit))

        print(f"\n[{name}] " + ("(merged train+valid+test)" if was_presplit
                                 else "(flat)"))
        print(f"  raw: {stats['n_raw']:,}  →  kept: {stats['n_final']:,}  "
              f"({stats['n_raw'] - stats['n_final']} dropped)")
        if stats["n_canonical_dup"]:
            print(f"  canonical dups removed: {stats['n_canonical_dup']}")
        if stats["bad_elements_breakdown"]:
            pretty = ", ".join(f"{e}:{n}" for e, n in
                                sorted(stats["bad_elements_breakdown"].items(),
                                       key=lambda x: -x[1]))
            print(f"  bad elements:           {pretty}")
        print(f"  written: {out_csv}")

    # ---- Markdown report ------------------------------------------------
    md = []
    md.append("# downstream_ft cleaning report")
    md.append("")
    md.append(f"Generated: `{datetime.now().isoformat(timespec='seconds')}`  ")
    md.append(f"Source:    `downstream_ft/`  ")
    md.append(f"Output:    `downstream_ft/clean/<dataset>.csv`")
    md.append("")
    md.append("## Filter pipeline")
    md.append("")
    md.append("Each row drops on the **first** condition it fails (counted at"
              " that step only; subsequent steps don't see it).")
    md.append("")
    md.append("1. **empty / NaN**")
    md.append("2. **RDKit unparseable** (`Chem.MolFromSmiles` returns None)")
    md.append("3. **radical** (any atom with `GetNumRadicalElectrons() > 0`)")
    md.append("4. **disconnected** (`.` in canonical SMILES → multi-fragment)")
    md.append("5. **bad elements** (atoms outside LoQI's 17-atom whitelist: "
              "H,B,C,N,O,F,Al,Si,P,S,Cl,As,Br,I,Hg,Bi,Se)")
    md.append("6. **|formal_charge| > 1** (chembl3d pretrain only saw -1..+1)")
    md.append("7. **canonical-SMILES dedup** (first occurrence kept across "
              "the entire dataset, including across pre-split files)")
    md.append("")
    md.append("Pre-split datasets (`delaney_s`, `freesolv_s`, `lipo_s`) are "
              "merged from `train.csv + valid.csv + test.csv` before "
              "cleaning, so canonical dedup also catches duplicates that "
              "span the official splits. The `_split` column is preserved "
              "in the cleaned CSV for downstream that wants to honor it.")
    md.append("")
    md.append("## Per-dataset summary")
    md.append("")
    md.append("| dataset | source | raw | empty | unparse | radical | "
              "disconnect | bad-elem | charge>1 | canon-dup | **kept** | "
              "bad-elem breakdown |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    md.extend(md_rows)
    md.append("")
    md.append("## Notes")
    md.append("")
    md.append("- The `kept` column equals raw − all drops; deduplication "
              "removes only canonical duplicates, not the first occurrence.")
    md.append("- For pre-split datasets, the merged size is "
              "`|train| + |valid| + |test|`. Canonical-dedup may remove rows "
              "that were already present in a different split.")
    md.append("- To use the cleaned CSVs in the FT pipeline, set "
              "`INPUT_DIR=downstream_ft/clean` AND update the DATASETS table "
              "in `run_downstream_pipeline.sh` to mark the formerly-presplit "
              "ones as flat (IS_PRESPLIT=0, CSV_REL=`<name>.csv`). The "
              "K=8 pickles will need to be regenerated against the cleaned "
              "CSVs (delete `data/downstream_k8/` to force re-sampling).")
    md_path = out_dir / "cleaning_report.md"
    md_path.write_text("\n".join(md) + "\n")

    json_path = out_dir / "cleaning_report.json"
    json_path.write_text(json.dumps(all_stats, indent=2))

    print()
    print("=" * 70)
    print(f"Markdown report: {md_path}")
    print(f"JSON report:     {json_path}")
    print(f"Cleaned CSVs:    {out_dir}/<dataset>.csv "
          f"({len(datasets)} files)")
    print("=" * 70)


if __name__ == "__main__":
    main()
