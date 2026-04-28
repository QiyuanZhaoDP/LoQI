# downstream_ft cleaning report

Generated: `2026-04-28T14:03:23`  
Source:    `downstream_ft/`  
Output:    `downstream_ft/clean/<dataset>.csv`

## Filter pipeline

Each row drops on the **first** condition it fails (counted at that step only; subsequent steps don't see it).

1. **empty / NaN**
2. **RDKit unparseable** (`Chem.MolFromSmiles` returns None)
3. **radical** (any atom with `GetNumRadicalElectrons() > 0`)
4. **disconnected** (`.` in canonical SMILES → multi-fragment)
5. **bad elements** (atoms outside LoQI's 17-atom whitelist: H,B,C,N,O,F,Al,Si,P,S,Cl,As,Br,I,Hg,Bi,Se)
6. **|formal_charge| > 1** (chembl3d pretrain only saw -1..+1)
7. **canonical-SMILES dedup** (first occurrence kept across the entire dataset, including across pre-split files)

Pre-split datasets (`delaney_s`, `freesolv_s`, `lipo_s`) are merged from `train.csv + valid.csv + test.csv` before cleaning, so canonical dedup also catches duplicates that span the official splits. The `_split` column is preserved in the cleaned CSV for downstream that wants to honor it.

## Per-dataset summary

| dataset | source | raw | empty | unparse | radical | disconnect | bad-elem | charge>1 | canon-dup | **kept** | bad-elem breakdown |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Cp | flat | 1,498 | 0 | 0 | 10 | 29 | 0 | 0 | 0 | **1,459** (97.4%) | — |
| V_cp | flat | 813 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **813** (100.0%) | — |
| de | flat | 782 | 0 | 0 | 2 | 0 | 2 | 0 | 0 | **778** (99.5%) | Sn:1, Ti:1 |
| gas_Hf | flat | 2,486 | 0 | 0 | 67 | 0 | 0 | 0 | 0 | **2,419** (97.3%) | — |
| k | flat | 756 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | **755** (99.9%) | — |
| liquid_Hf | flat | 1,628 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | **1,624** (99.8%) | — |
| delaney_s | merged | 1,128 | 0 | 0 | 0 | 0 | 0 | 0 | 11 | **1,117** (99.0%) | — |
| freesolv_s | merged | 642 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | **641** (99.8%) | — |
| lipo_s | merged | 4,200 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | **4,199** (100.0%) | — |

## Notes

- The `kept` column equals raw − all drops; deduplication removes only canonical duplicates, not the first occurrence.
- For pre-split datasets, the merged size is `|train| + |valid| + |test|`. Canonical-dedup may remove rows that were already present in a different split.
- To use the cleaned CSVs in the FT pipeline, set `INPUT_DIR=downstream_ft/clean` AND update the DATASETS table in `run_downstream_pipeline.sh` to mark the formerly-presplit ones as flat (IS_PRESPLIT=0, CSV_REL=`<name>.csv`). The K=8 pickles will need to be regenerated against the cleaned CSVs (delete `data/downstream_k8/` to force re-sampling).
