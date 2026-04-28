# ThermoGen / LoQI — Model Description

A multi-task 3D foundation model that *generates* molecular conformers and
*predicts* thermodynamic properties from a single backbone. Built on top of
the LoQI flow-matching conformer generator.

---

## At a glance

| Component | What it does | Trained on |
|---|---|---|
| **LoQI backbone** (MegaFNV3Conf) | Flow-matching denoising of 3D coords + atom types | chembl3d, ~720k molecules |
| **Thermo head** (`AtomMolMP`, 5 outputs) | Predict H₂₉₈, G₂₉₈, C_v, S°, H₀ from H | TCIT-labeled subset (~650k mols) |
| **RDKit head** (`AtomMolMP`, 9 outputs) | Predict logP, TPSA, n_HD, n_HA, n_rot, fraCSP3, n_rings, QED, LabuteASA | Full property_table.parquet (~1.85M mols, 100% coverage) |

The three tasks share the backbone; their losses are summed with task weights
during pretraining (flow=1.0, thermo=0.1, rdkit=0.02). Heads only condition on
the backbone's per-atom hidden state H at integration time t = 1.0 (clean end
of the flow), with `min_time=0.8` mask so the property losses only kick in
near the data manifold.

---

## Architecture

### Backbone (`MegaFNV3Conf`)

EMA-wrapped, equivariant graph network with attention-style message passing.
Two scale presets:

| Config name | `invariant_node_feat_dim` | `num_layers` | `num_heads` | `n_vector_features` |
|---|---:|---:|---:|---:|
| **warm** (`loqi_thermo_flow_warm.yaml`) | 256 | 10 | 4 | 64 |
| **cold** (`loqi_thermo_flow_cold.yaml`) | 384 | 14 | 8 | 96 |

"warm" / "cold" refer to model **size**, not initialization. (Both can be
trained from scratch — `cold-start` — or continued from a flow-only ckpt —
`warm-start`.)

Per layer the dominant linears are `qkv_proj` (d → 3d), `out_projection`
(d → d), and a SwiGLU FFN (~2.6× expansion). Total trainable parameters:

- warm config: ~10–12M
- cold config: ~25–30M

### Atom encoder (`ATOMIC_TO_INNER`)

17-element whitelist:

```
H, B, C, N, O, F, Al, Si, P, S, Cl, As, Br, I, Hg, Bi, Se
```

Plus `formal_charge ∈ {−1, 0, +1}` (offset +2 applied at runtime by
`pre_format_molecules` to map to one-hot indices 1–3 of `num_classes=6`).

**Not encoded:** unpaired-electron count. Open-shell radicals therefore
collide in feature space with their closed-shell counterparts and are
excluded from training and downstream evaluation. See
`downstream_ft/clean/cleaning_report.md` for details.

### Heads

Both property heads are `AtomMolMP` instances — message passing followed by
attention pooling to a global molecular vector, then an MLP to scalar
outputs.

| Head | n_mp_layers | mp_n_heads | hidden | n_targets | Loss weight |
|---|---:|---:|---:|---:|---:|
| Thermo | 4 | 4 | 256 | 5 | 0.1 |
| RDKit | 2 | 4 | 128 | 9 | 0.02 |

Thermo targets (z-score normalized; stats from
`data/property_table.parquet` after outlier screening, n=652,567):

| Index | Target | Mean (kJ/mol) | Std |
|---:|---|---:|---:|
| 0 | enthalpy_298 (H₂₉₈) | −317.36 | 346.40 |
| 1 | gibbs_298 (G₂₉₈)    | +111.66 | 344.07 |
| 2 | cv_gas (C_v)         | +390.86 |  98.00 |
| 3 | entropy_gas (S°)     | +676.34 | 160.56 |
| 4 | enthalpy_0 (H₀)      | −119.10 | 364.45 |

---

## Training pipeline

### Phase 1 — flow-matching pretrain (LoQI)

Standard flow-matching denoising of (atom types, formal charges,
coordinates) at uniformly-sampled t ∈ [0, 1]. 25 integration timesteps
(continuous time). Coord-velocity loss is clamped (`ts_coord=1.0`) after
epoch 10 for stability.

Output: `loqi_flow.ckpt` — a pure conformer generator. Loadable directly
for sampling without any property head.

### Phase 2a — joint multi-task pretrain (cold-start)

From random init, train flow + thermo + RDKit losses simultaneously. The
thermo / rdkit losses are masked to t > 0.8 so they only push H near the
clean end of the flow.

Output: `thermogen_cold.ckpt` — the canonical ThermoGen base model.

### Phase 2b — warm-start variant

From an existing `loqi_flow.ckpt`, attach random-init thermo + rdkit heads
and train the joint loss. Cheaper, but our experiments showed warm-start
locks the backbone in the "generation-optimal" basin and the thermo heads
end up doing all the work — downstream property MAE ≈ random-head baseline.
Cold-start is the recommended phase 2.

Output: `thermo_flow_warm.ckpt`.

### Phase 3 — downstream FT (per task)

For each downstream property dataset (e.g. gas_Hf, freesolv, lipo):

1. **Sample K conformers per molecule** with the (frozen) backbone:
   ```
   scripts/sample_conformers.py --n_confs 8 --postprocess none
   ```
2. **Build PyG dataset** joining conformers with target labels by canonical
   SMILES (`scripts/prepare_downstream_K_pt.py`).
3. **5-fold CV with K-conformer ensemble** (`scripts/downstream_cv.py`):
   - Split molecules into folds (group-by `input_id` so all K conformers of
     one molecule stay in the same fold — no leakage).
   - Cache backbone H once per fold.
   - Train a small head (`SingleTargetHead`, also `AtomMolMP` underneath)
     on cached H.
   - Val: mean prediction across the K conformers per molecule.

Default training: cosine LR schedule, 200 epochs, early-stopping patience
30, AdamW with weight_decay=1e-5, grad clip 1.0.

#### Optional FT modes

- **Warm-init head**: load thermo head's pretrained weights into the
  downstream head (auto-aligns dims to `thermo_head_args`). Final 5→1
  Linear stays random.
- **K-cap (`--max-k-per-input`)**: train on a subset of K conformers,
  diagnostic for ensemble vs single-conformer training.
- **Invariance loss (`--invariance-lambda`)**: explicit penalty on
  within-`input_id` prediction variance.
- **LoRA backbone FT (`--lora-r`)**: low-rank adapters on backbone Linears
  break the H ceiling without losing the base ckpt's generation capability.
  Default targets: `qkv_proj`, `out_projection`. r=8 → ~120K extra params
  per task (~1% of backbone), reversible (load adapters for property,
  ignore them for generation).

---

## Inference modes

The same checkpoint serves three modes, selected by which forward pass is
called:

### 1. Conformer generation

```python
model = Graph3DInterpolantModel.load_from_checkpoint("thermogen_cold.ckpt")
trajectory = model.sample(smiles_batch, n_steps=10, postprocess="none")
```

Equivalent to running LoQI alone.

### 2. Zero-shot thermo prediction

```python
out, _, _ = model(batch, t=t_max)        # batch = K conformers per SMILES
H = out["thermo_mp"]                      # (n_atoms, 5) z-score normalized
preds = H * cfg.thermo_loss.target_std + cfg.thermo_loss.target_mean
```

`scripts/predict_thermo_zeroshot.py` does this with K-conformer averaging
and SMILES-keyed CSV join.

### 3. Downstream FT prediction

After phase-3 FT, the per-dataset checkpoint contains a trained
`SingleTargetHead` on top of the (still frozen, or LoRA-adapted) backbone.
Standard `predict()` returns the property value.

---

## Data

### Pretrain corpus

- **chembl3d_stereo**: ~720k molecules with 3D conformers, single-component,
  closed-shell, formal_charge ∈ {−1, 0, +1}, atomic numbers in the 17-set
  above. Stored as PyG `_h.pt` files.
- **TCIT thermo labels** for ~650k of those molecules (5 thermo targets,
  outlier-screened; physically impossible S°<0 dropped + 6-MAD filter).
- **RDKit descriptors** for all ~1.85M property-table entries.

### Downstream FT corpus

9 standard property-prediction benchmarks. **Canonical source is
`downstream_ft/clean/`** — flat, deduplicated, filtered CSVs produced
by `scripts/clean_downstream.py`. The raw `downstream_ft/<flat>.csv`
and `downstream_ft/<presplit>/{train,valid,test}.csv` are kept for
reference; pipelines (`run_downstream_pipeline.sh`,
`run_downstream_K8_full.sh`, `sample_downstream_K5.sh`) all default
to `INPUT_DIR=downstream_ft/clean`. To reproduce the original raw-data
behavior, set `INPUT_DIR=downstream_ft`.

| Dataset | Target | Domain | Raw rows | Kept (cleaned) | Drops |
|---|---|---|---:|---:|---|
| Cp | heat capacity | thermo | 1,498 | 1,459 | 10 radical + 29 disconnect |
| V_cp | heat capacity at T | thermo | 813 | 813 | — |
| de | density | thermo | 782 | 778 | 2 radical + 2 OOD-elem (Sn, Ti) |
| gas_Hf | Hf at 298 K, gas phase | thermo | 2,486 | 2,419 | 67 radical |
| k | rate constant | kinetics | 756 | 755 | 1 \|charge\|>1 |
| liquid_Hf | Hf at 298 K, liquid | thermo | 1,628 | 1,624 | 4 radical |
| delaney_s | solubility | ADMET | 1,128 (merged) | 1,117 | 11 canonical-dup |
| freesolv_s | hydration free energy | ADMET | 642 (merged) | 641 | 1 \|charge\|>1 |
| lipo_s | lipophilicity | ADMET | 4,200 (merged) | 4,199 | 1 disconnect |

Pre-split datasets are merged from `train+valid+test` *before* cleaning,
so canonical-dedup catches duplicates that span the official splits.
The cleaned CSVs preserve a `_split` column for reference. The full
report is at `downstream_ft/clean/cleaning_report.md`.

---

## Scope and limitations

- **Closed-shell only.** Radicals (·OH, ·CN, methoxyl, etc.) are excluded;
  the atom encoder doesn't represent unpaired-electron count, so radicals
  would silently mispredict by 50–200 kJ/mol. Same restriction holds for
  UniMol / SchNet / MACE-MP / AIMNet2 family.
- **Single-component molecules only.** Disconnected SMILES (multi-fragment,
  `.` in canonical) are dropped.
- **Limited element coverage.** 17 atomic numbers — covers organic + main
  metalloids but excludes most transition metals, lanthanides, etc.
- **|formal_charge| ≤ 1.** Multiply-charged ions (sulfates, phosphates as
  divalent anions) excluded. Pretrain didn't see them.
- **Pretrain data scale.** ~720k molecules vs UniMol's ~209M structures;
  this is the dominant reason ThermoGen lags UniMol on pure property
  benchmarks. Conformer generation remains a unique capability.

### Future directions

- `(z, q, n_rad)` triplet atom encoder + re-pretrain with radical examples
  → unlock gas-phase radical thermochemistry as a distinct capability.
- LoRA-adapted backbone FT (already implemented; needs benchmark sweep) to
  break the frozen-H ceiling on hard-property datasets.
- Scale pretrain corpus via XTB/MMFF surrogate-label pseudo-labeling on
  larger unlabeled molecule sets (ZINC, etc.).

---

## Reproducing key numbers

```bash
# 0. Clean the downstream CSVs (one-shot; outputs downstream_ft/clean/).
#    All pipelines below default to INPUT_DIR=downstream_ft/clean, so
#    this step gates everything that follows.
python scripts/clean_downstream.py

# 1. Pretrain (cold-start, large config)
python scripts/train.py --config-name loqi_thermo_flow_cold

# 2. Sample K=8 conformers for every cleaned dataset
K=8 OUTPUT_DIR=data/downstream_k8 \
    bash scripts/sample_downstream_K5.sh

# 3. Downstream FT — three head modes × 9 datasets, sampling + CV
nohup bash scripts/run_downstream_K8_full.sh > downstream_K8.log 2>&1 &

# 4. (Optional) LoRA backbone FT — breaks the frozen-H ceiling
LORA_R=8 LORA_TARGET=qkv_proj,out_projection \
    OUT_SUFFIX=lora_r8 \
    bash scripts/run_downstream_pipeline.sh
```

To re-run against raw downstream_ft/ instead of the cleaned tree, prefix
each pipeline call with `INPUT_DIR=downstream_ft` and (for
run_downstream_pipeline.sh) edit the DATASETS table to restore
`IS_PRESPLIT=1` for delaney_s / freesolv_s / lipo_s.

---

## File map

```
src/megalodon/
  models/
    module.py              # Graph3DInterpolantModel (Lightning module)
    thermo_heads.py        # AtomMolMP head architecture
    loss_fn.py             # CombinedAuxiliaryLoss = thermo + rdkit + energy
  dynamics/
    fn_model.py            # MegaFNV3Conf backbone
  data/
    batch_preprocessor.py  # rotation aug + coord scaling

scripts/
  train.py                 # Hydra-driven training entry
  sample_conformers.py     # LoQI conformer generation
  predict_thermo_zeroshot.py
  prepare_downstream_K_pt.py
  downstream_cv.py         # 5-fold CV with K-conformer ensemble
  run_downstream_K8_full.sh   # full benchmark driver
  clean_downstream.py      # downstream CSV cleaning
  conf/loqi/
    loqi_flow.yaml             # phase 1 — flow only
    loqi_thermo_flow_warm.yaml # phase 2 warm, small backbone
    loqi_thermo_flow_cold.yaml # phase 2 cold, large backbone

downstream_ft/
  <dataset>.csv  /  <presplit>/{train,valid,test}.csv  # raw (reference)
  clean/                                               # CANONICAL
    <dataset>.csv                                      # flat, dedup'd, filtered
    cleaning_report.md                                 # per-dataset drop counts
    cleaning_report.json                               # machine-readable
```
