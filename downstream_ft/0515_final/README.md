# ThermoGen — Final Strict 12-Element DB

**Element set:** `{C, H, O, N, P, S, F, Cl, Br, I, Si, B}` (12 elements)
**Dropped vs LoQI 17:** Al, As, Hg, Bi, Se
**Base:** `dataset_v1/releases/3star`  filtered through this stricter element gate
**Extension:** 5 downstream tasks (freesolv / ESOL / Lipophilicity / CEP / PPBR)
**Total:** 21,276 unique molecules × 43 properties
**Seed:** 42    **Folds:** 5    **Valid fraction of train:** 0.1

## 3★ base molecules dropped by stricter element filter

51 molecules (0.2% of 3★ base) removed.

| Reason | Count |
|---|---:|
| `non_allowed_As` | 24 |
| `non_allowed_Se` | 16 |
| `non_allowed_Al` | 5 |
| `non_allowed_Hg` | 4 |
| `non_allowed_Bi` | 2 |


## Downstream task counts

| Task | n |
|---|---:|
| `Lipophilicity_logD` | 4,191 |
| `PPBR_pct` | 1,386 |
| `ESOL_logS` | 1,115 |
| `CEP_PCE` | 875 |
| `freesolv_dG_kcalmol` | 641 |


## Layout (same as `final/`)

```
final_strict/
├── README.md
├── master.csv                          inchikey | smiles | name | {prop}_value, {prop}_tier
├── per_property/{prop}.csv             inchikey | smiles | value | tier | scaffold
├── splits_summary.csv
├── downstream_dropped_records.csv
├── downstream_dropped_summary.csv
├── {property}/                          (one folder per property)
│   ├── random_5fold.json
│   └── scaffold_5fold.json
└── csv_data/{property}/Split/
    ├── random_cv5/cv{1-5}_{train,valid,test}.csv
    └── scaffold_cv5/cv{1-5}_{train,valid,test}.csv
```
