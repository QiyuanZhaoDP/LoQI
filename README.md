# LoQI: Scalable Low-Energy Molecular Conformer Generation with Quantum Mechanical Accuracy

<div align="center">
  <a href="https://scholar.google.com/citations?user=DOljaG8AAAAJ&hl=en" target="_blank">Filipp&nbsp;Nikitin<sup>1,2</sup></a> &emsp; <b>&middot;</b> &emsp;
  <a href="#" target="_blank">Dylan&nbsp;M.&nbsp;Anstine<sup>2,3</sup></a> &emsp; <b>&middot;</b> &emsp;
  <a href="#" target="_blank">Roman&nbsp;Zubatyuk<sup>2,5</sup></a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://scholar.google.ch/citations?user=8S0VfjoAAAAJ&hl=en" target="_blank">Saee&nbsp;Gopal&nbsp;Paliwal<sup>5</sup></a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://olexandrisayev.com/" target="_blank">Olexandr&nbsp;Isayev<sup>1,2,4*</sup></a>
  <br>
  <sup>1</sup>Ray and Stephanie Lane Computational Biology Department, Carnegie Mellon University, Pittsburgh, PA, USA
  <br>
  <sup>2</sup>Department of Chemistry, Carnegie Mellon University, Pittsburgh, PA, USA
  <br>
  <sup>3</sup>Department of Chemical Engineering and Materials Science, Michigan State University, East Lansing, MI, USA
  <br>
  <sup>4</sup>Department of Materials Science and Engineering, Carnegie Mellon University, Pittsburgh, PA, USA
  <br>
  <sup>5</sup>NVIDIA, Santa Clara, CA, USA
  <br><br>
  <a href="#" target="_blank">📄&nbsp;Paper</a> &emsp; <b>&middot;</b> &emsp;
  <a href="#citation">📖&nbsp;Citation</a> &emsp; <b>&middot;</b> &emsp;
  <a href="#setup">⚙️&nbsp;Setup</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://github.com/isayevlab/LoQI" target="_blank">🔗&nbsp;GitHub</a>
  <br><br>
  <span><sup>*</sup>Corresponding author: olexandr@olexandrisayev.com</span>
</div>

---

## Overview

<div align="center">
    <img width="700" alt="Macrocycles" src="assets/macrocycles.svg"/>
</div>

### Abstract

Molecular geometry is crucial for biological activity and chemical reactivity; however, computational methods for generating 3D structures are limited by the vast scale of conformational space and the complexities of stereochemistry. Here we present an approach that combines an expansive dataset of molecular conformers with generative diffusion models to address this problem. We introduce **ChEMBL3D**, which contains over 250 million molecular geometries for 1.8 million drug-like compounds, optimized using AIMNet2 neural network potentials to a near-quantum mechanical accuracy with implicit solvent effects included. This dataset captures complex organic molecules in various protonation states and stereochemical configurations. 

We then developed **LoQI** (Low-energy QM Informed conformer generative model), a stereochemistry-aware diffusion model that learns molecular geometry distributions directly from this data. Through graph augmentation, LoQI accurately generates molecular structures with targeted stereochemistry, representing a significant advance in modeling capabilities over previous generative methods. The model outperforms traditional approaches, achieving up to tenfold improvement in energy accuracy and effective recovery of optimal conformations. Benchmark tests on complex systems, including macrocycles and flexible molecules, as well as validation with crystal structures, show LoQI can perform low energy conformer search efficiently.

> **Note on Implementation**: LoQI is built upon the [Megalodon architecture](https://arxiv.org/pdf/2505.18392) developed, adapting it specifically for stereochemistry-aware conformer generation with the ChEMBL3D dataset.

---

## Key Features

- **ChEMBL3D Dataset**: 250+ million AIMNet2-optimized conformers for 1.8M drug-like molecules
- **Stereochemistry-Aware**: First all-atom diffusion model with explicit stereochemical encoding
- **Quantum Mechanical Accuracy**: Near-DFT accuracy with implicit solvent effects
- **Superior Performance**: Up to 10x improvement in energy accuracy over traditional methods
- **Complex Molecule Support**: Handles macrocycles, flexible molecules, and challenging stereochemistry

---

## Setup

Installation will usually take up to 20 minutes.

### System and Hardware Requirements

- OS tested by authors:
  - Ubuntu 24.04 LTS (latest stable Ubuntu LTS at time of writing)
- Other platforms:
  - Expected to work, but if installation is not out-of-the-box, use the PyTorch Geometric installation guide for your exact Python/PyTorch/CUDA combination:
    https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
- Tested inference hardware:
  - GPU: NVIDIA RTX 3090 (24 GB VRAM)
  - CPU: AMD Ryzen 9 5950X
- Recommended GPU memory:
  - 16-24 GB VRAM for comfortable inference/evaluation with larger molecules and higher batch sizes
- Minimum practical GPU memory:
  - 8 GB VRAM can run inference, but requires reduced batch sizes
- CPU-only:
  - Possible, but not recommended and not systematically studied by the authors

OOM mitigation for larger molecules:
- reduce inference batch size (`--batch_size` in sampling, or `data.inference_batch_size` in config)
- if using evaluation/optimization, also reduce optimization batch size (`evaluation.energy_metrics_args.batchsize`)

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for training)
- [Conda](https://docs.conda.io/) or [Mamba](https://mamba.readthedocs.io/) (recommended)

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/isayevlab/LoQI.git
cd LoQI

# Create and activate conda environment
conda create -n loqi python=3.10 -y
conda activate loqi

# Install core dependencies
pip install -r requirements.txt

# Install this package in editable mode (adds src to PYTHONPATH)
pip install -e .
```

If you prefer a fully conda-based setup (recommended for RDKit), you can install RDKit via conda-forge before running `pip install -r requirements.txt`.

### Data Setup

Training and evaluation use the **ChEMBL3D** data releases below.

**Release 1: Full ChEMBL3D Quantum-Accurate conformer dataset**
- URL: https://kilthub.cmu.edu/articles/dataset/_b_ChEMBL3D_Quantum-Accurate_3D_Conformers_for_ChEMBL_at_Scale_b_/31428449
- DOI: https://doi.org/10.1184/R1/31428449

**Release 2: Processed dataset + LoQI checkpoints (diffusion + flow matching)**
- URL: https://kilthub.cmu.edu/articles/dataset/LoQI_Scalable_Low-Energy_Molecular_Conformer_Generation_with_Quantum_Mechanical_Accuracy/31441570
- DOI: https://doi.org/10.1184/R1/31441570
- Includes:
  - `loqi.ckpt`
  - `loqi_flow.ckpt`
  - `chembl3d_stereo/` processed dataset

For this repository, place downloaded assets with this layout:
```text
LoQI/
  data/
    loqi.ckpt
    loqi_flow.ckpt
    chembl3d_stereo/
      processed/
        ...
```

AimNet2 model path expected by configs:
```text
src/megalodon/metrics/aimnet2/cpcm_model/wb97m_cpcms_v2_0.jpt
```

---

## Web App

The repository includes a Streamlit interface for interactive conformer generation, postprocessing, and visualization.

<div align="center">
    <img width="100%" alt="LoQI App" src="assets/app.png"/>
</div>

Use the app-specific installation and usage instructions from `app/README.md` (recommended, as app dependencies are separated from core training/inference dependencies).  
Quick start from repo root:

```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

## Usage

Make sure that `src` content is available in your `PYTHONPATH` (e.g., `export PYTHONPATH="./src:$PYTHONPATH"`) if LoQI is not installed locally (`pip install -e .`). 

### Model Training

```bash
# LoQI conformer generation model
python scripts/train.py --config-name=loqi outdir=./outputs train.gpus=1 data.dataset_root="./chembl3d_data"

# LoQI flow-matching conformer generation model
python scripts/train.py --config-name=loqi_flow outdir=./outputs train.gpus=1 data.dataset_root="data/chembl3d_stereo"

# Customize training parameters
python scripts/train.py --config-name=loqi \
    outdir=./outputs \
    train.gpus=2 \
    train.n_epochs=800 \
    train.seed=42 \
    data.batch_size=150 \
    optimizer.lr=0.0001
```

### Model Inference and Sampling

#### Conformer Generation

```bash
# Generate conformers for a single molecule
python scripts/sample_conformers.py \
    --config scripts/conf/loqi/loqi.yaml \
    --ckpt data/loqi.ckpt \
    --input "c1ccccc1" \
    --output outputs/benzene_conformers.sdf \
    --n_confs 10 \
    --batch_size 1

# Generate conformers with evaluation (requires 3D input, e.g., SDF with low energy conformer)
python scripts/sample_conformers.py \
    --config scripts/conf/loqi/loqi.yaml \
    --ckpt data/loqi.ckpt \
    --input data/ethanot_low_energy.sdf \
    --output outputs/ethanol_conformers.sdf \
    --n_confs 100 \
    --batch_size 10 \
    --eval

# Optional postprocessing: AIMNet2 optimization + iRMSD unique-set pruning
python scripts/sample_conformers.py \
    --config scripts/conf/loqi/loqi_flow.yaml \
    --ckpt data/loqi_flow.ckpt \
    --input "CC(=O)Oc1ccccc1C(=O)O" \
    --output outputs/aspirin_opt_unique.sdf \
    --n_confs 50 \
    --batch_size 50 \
    --postprocess optimization+irmsd \
    --optimization_batch_size 64 \
    --opt_fmax 0.05 \
    --opt_max_nstep 250 \
    --irmsd_rthr 0.125
```

Recent sampling updates in `scripts/sample_conformers.py`:
- input validation + SMILES revalidation (canonical roundtrip), with unsupported-element/radical checks
- atom-aware dynamic batching for inference (`--atom-aware-batching`, `--target-molecule-size`, `--shuffle`)
- optional hydrogen addition for SMILES inputs (`--add-hs` / `--no-add-hs`)
- no RDKit conformer initialization for SMILES; zero-initialized coordinates are used
- if input is SDF with conformers, existing 3D coordinates are used
- optional postprocessing (`--postprocess none|optimization|optimization+irmsd`)

On the tested setup (RTX 3090 + Ryzen 9 5950X), inference for a typical ChEMBL molecule takes approximately 0.1 seconds per conformer when processed within a batch. See **System and Hardware Requirements** above for VRAM guidance and OOM mitigation.

Note: Make sure you define correct paths for dataset and AimNet2 model in `loqi.yaml`. The relative path of AimNet2 model is `src/megalodon/metrics/aimnet2/cpcm_model/wb97m_cpcms_v2_0.jpt`.

Sampling steps: `--n_steps` defaults to 25. Diffusion models were trained with 25 steps and are not expected to work well for other values. Flow-matching models can be run with different step counts.

#### Performance Test (Fixed Molecule Sizes)

Use `scripts/performance_test.py` to:
- sample 1000 molecules each with atom counts 10, 25, 50, and 100 from `data/chembl3d_stereo/processed/train_h.pt`
- select molecules deterministically (first `N` per size in dataset order)
- export per-molecule SDF inputs
- measure per-molecule generation and optimization times

```bash
conda run -n mega env PYTHONPATH=./src TORCH_COMPILE_DISABLE=1 \
python scripts/performance_test.py \
  --dataset_pt data/chembl3d_stereo/processed/train_h.pt \
  --sizes 10,25,50,100 \
  --n_per_size 100 \
  --outdir outputs/performance_test \
  --config scripts/conf/loqi/loqi.yaml \
  --ckpt data/loqi.ckpt \
  --n_confs 100 \
  --generation_batch_size 1
```

By default, optimization settings are taken from the selected config
(`evaluation.energy_metrics_args.batchsize` and `evaluation.energy_metrics_args.opt_params`).

Outputs:
- `outputs/performance_test/selected_manifest.csv` (selected molecules + per-molecule SDF path)
- `outputs/performance_test/size_<N>/mol_*.sdf` (one input SDF per selected molecule)
- `outputs/performance_test/size_<N>_selected.sdf` (combined SDF per size)
- `outputs/performance_test/timings_per_molecule.csv` (generation/optimization timing per molecule)

#### Available Configurations

**LoQI Models:**
- `loqi.yaml` - LoQI stereochemistry-aware conformer generation model
- `nextmol.yaml` - Alternative configuration for NextMol-style generation
- `loqi_flow.yaml` - LoQI flow-matching conformer generation model

**Thermo-aware extensions (flow-matching only — diffusion variants
were superseded):**

|                              | separate heads (thermo + rdkit) | combined 14-target head |
|---|---|---|
| warm-start from `loqi_flow.ckpt` | `loqi_thermo_flow_warm.yaml` | `loqi_thermo_flow_warm_combined.yaml` |
| from-scratch scaled backbone | `loqi_thermo_flow_cold.yaml` | `loqi_thermo_flow_cold_combined.yaml` |

- **Warm** uses the 256/10/4 base backbone; **cold** scales to
  384/14/8 (~100M params).
- **Separate-head** variants attach `ThermoHeadModel` (5 targets) +
  `RDKitHeadModel` (9 targets) to `MegaFNV3Conf` and use
  `ThermoPropertyLoss` + `RDKitDescriptorLoss` separately.
- **Combined-head** variants attach a single `CombinedHeadModel`
  (14 targets, shared `AtomMolMP`) and use `CombinedPropertyLoss`.
  Forces one molecule representation to encode features useful for
  both task families; saves ~20 % head params.
- The head modes are mutually exclusive — `combined_head_args`
  silently wins if both are set in YAML.
- All four configs use **epoch-anchored cosine LR**
  (`warmup_epochs` + `decay_epochs` resolved against
  `trainer.estimated_stepping_batches` at runtime); set
  `lr_scheduler.type: linear_warmup_decay` to switch back to linear
  decay.

---

## ThermoGen: thermo-aware foundation model

This repo turns LoQI into a thermodynamics-aware foundation model for
downstream property prediction. The denoising objective and a property
auxiliary loss — semi-supervised, NaN-masked, gated to late timesteps
via `min_time` — are optimized jointly, producing a backbone whose
per-atom features `H` are shaped by both 3D structure and property
signal. Heads then sit on top of `H` for either pretraining auxiliary
loss or downstream fine-tuning.

Two head architectures coexist as YAML options (see the table in
*Available Configurations* above):

- **Separate** — `ThermoHeadModel` (5 thermo targets) +
  `RDKitHeadModel` (9 RDKit descriptors), losses summed.
- **Combined** — one `CombinedHeadModel` (14 targets, shared
  `AtomMolMP`) with `CombinedPropertyLoss`.

All heads are defined in `src/megalodon/models/thermo_heads.py`;
losses in `src/megalodon/models/loss_fn.py`.

### Data preparation

Geometry is kept in the original `{train,val,test}_h.pt` files untouched.
Per-molecule properties (5 TCIT thermo labels + 9 RDKit descriptors) live
in a single parquet table keyed by canonical implicit-H SMILES, joined
onto each `Data` at load time via the `AttachProperties` transform.

```bash
# 1. Parse TCIT log → per-SMILES CSV of (Hf_0, Hf_298, Gf_298, S0, Cv)
python data_processing/parse_tcit_log.py \
    --input  data_processing/batch9.log \
    --output data_processing/tcit_thermo_labels.csv

# 2. Build a canonical → neutral SMILES index (handles ChEMBL3D's
#    ionic species via RDKit Uncharger)
python data_processing/build_neutralization_index.py \
    --inputs data/chembl3d_stereo/processed/{train,val,test}_h.pt \
    --output data_processing/chembl3d_neutralization_index.json

# 3. Build the unified property table (thermo labels + RDKit descriptors).
python data_processing/build_property_table.py \
    --inputs data/chembl3d_stereo/processed/{train,val,test}_h.pt \
    --thermo-csv data_processing/tcit_thermo_labels.csv \
    --neutralization-index data_processing/chembl3d_neutralization_index.json \
    --output data/property_table.parquet
```

### Thermo-aware pre-training

Pick one of the four configs (warm/cold × separate/combined) and launch:

```bash
# Warm, separate heads (256/10/4 backbone, ~1 day on 4 GPUs)
python scripts/train.py --config-name=loqi_thermo_flow_warm \
    outdir=./outputs/loqi_thermo_flow_warm train.gpus=4

# Warm, combined head
python scripts/train.py --config-name=loqi_thermo_flow_warm_combined \
    outdir=./outputs/loqi_thermo_flow_warm_combined train.gpus=4

# Cold, separate heads (384/14/8 ≈ 100M params, ~3-5 days)
python scripts/train.py --config-name=loqi_thermo_flow_cold \
    outdir=./outputs/loqi_thermo_flow_cold train.gpus=4

# Cold, combined head
python scripts/train.py --config-name=loqi_thermo_flow_cold_combined \
    outdir=./outputs/loqi_thermo_flow_cold_combined train.gpus=4
```

wandb traces to watch:

- `train/x_loss` — denoising, should stay flat (preserve LoQI quality)
- `train/additional_loss_term` — thermo / rdkit / combined aux,
  should decrease
- `train/thermo/mae_*`, `train/rdkit/mae_*`, or
  `train/combined/mae_*` — per-target MAE per epoch (kJ/mol for H/G,
  J/mol/K for Cv/S°, natural units for RDKit)
- `train/thermo/labeled_active` — per-step labeled-molecule count
- `val/opt_median_relative_energy` — canary for conformer-generation
  quality
- `trainer/lr` — verify epoch-anchored cosine actually decays (look
  for the rank-0 `[lr_scheduler] epoch-based resolve: ...` log line
  at the top of training)

#### A note on "epoch" semantics under DDP

This codebase uses `MiDiDataloader`, whose custom collate path bypasses
Lightning's automatic `DistributedSampler` injection. As a result, under
multi-GPU DDP **every rank independently iterates the full dataset**, so
one `n_epochs` unit in the YAML corresponds to `n_gpus × full-dataset
passes` of actual data exposure. Concretely: `n_epochs: 50` on 4 GPUs
≈ 200 single-GPU-epoch equivalents. The epoch-anchored cosine LR
schedule uses `trainer.estimated_stepping_batches` directly, so the
resolved step count is correct under DDP — but the *informal* "1 epoch"
the YAML refers to is the DDP epoch, not a single-GPU pass.

### Downstream property prediction

Given CSVs with `(SMILES, target)` columns, the downstream pipeline:

1. **Audit + split**: `scripts/audit_0511.py` applies hard limits,
   InChIKey dedup, manual removals, and physics-aware retain rules,
   then writes 5-fold random and 3-fold scaffold splits per dataset.
2. **Sample conformers**: K = 8 conformers per molecule via
   flow-matching at 10 integration steps; K = 12 multi-snapshot
   (4 trajectories × 3 timesteps) as an alternative.
3. **Extract H once per (ckpt × sampling × dataset)** and cache to
   disk; CV then trains head-only against the cached features.
4. **5-fold CV** using the pre-partitioned audit splits for full
   reproducibility (`--split-dir`).

The unified driver `scripts/run_cv.sh` runs all four stages for a
matrix of ckpts × sampling modes × datasets:

```bash
# Run all CV tasks against the canonical 0511 audit splits
RUN_TAG=0511 \
INPUT_DIR=downstream_ft/0511_cc_audit/Clean \
SPLIT_DIR_ROOT=downstream_ft/0511_cc_audit/Split \
OUT_ROOT=outputs/cv_0511 \
WANDB=1 WANDB_PROJECT=cv_0511 SWANLAB_SYNC=1 \
nohup bash scripts/run_cv.sh > /tmp/cv_0511.log 2>&1 &
disown
```

Per-dataset reports land in `$OUT_ROOT/<name>/cv_report.json` with
both `mae_per_conformer_mean` (single-conformer, comparable to
UniMol) and `mae_last_stable_mean` (K-conformer ensemble averaged
over the last-10-epoch window). The driver prints a cross-dataset
summary table at the end.

For ad-hoc single-dataset runs, call the pipeline directly:

```bash
python scripts/downstream_cv.py \
    --ckpt data/thermo_flow_warm.ckpt \
    --config scripts/conf/loqi/loqi_thermo_flow_warm.yaml \
    --dataset-pt data/cv_pt_<tag>/<dataset>_K8.pt \
    --split-dir downstream_ft/0511_cc_audit/Split/<dataset>/random_cv5 \
    --out-dir outputs/cv_<dataset> \
    --n-folds 5 --epochs 200 --lr 3e-4 --device cuda
```

### Key scripts

| path | role |
|---|---|
| `data_processing/parse_tcit_log.py` | TCIT log → CSV of per-SMILES thermo labels |
| `data_processing/build_neutralization_index.py` | canonical → neutral SMILES map |
| `data_processing/build_property_table.py` | unified thermo + RDKit property parquet |
| `scripts/label_energy.py` | attach AIMNet2 energies to a chembl3d `.pt` |
| `scripts/sample_conformers.py` | flow-matching conformer sampler (single mol, batch) |
| `scripts/sample_conformers_multistep.py` | multi-timestep snapshot sampler (K = n_traj × n_snap) |
| `scripts/prepare_downstream_K_pt.py` | K-conformer pickle → PyG `.pt` for CV |
| `scripts/audit_0511.py` | downstream CSV audit + CV splits (canonical) |
| `scripts/extract_smiles.py` | SMILES helpers for audit / dedup |
| `scripts/downstream_cv.py` | 5-fold CV head training on cached H |
| `scripts/run_cv.sh` | unified CV driver: sample + extract H + CV + summary |
| `scripts/run_downstream_pipeline.sh` | per-dataset pipeline called by `run_cv.sh` |
| `scripts/inspect_ckpt.py` | print backbone/head config from a saved ckpt |
| `scripts/check_downstream_pt.py` | sanity-check a downstream `.pt` |
| `scripts/morgan_rf_baseline.py` | MorganFP + Random Forest baseline for CV comparison |
| `scripts/eval_loqi_loss.py` | reproduce val/x_loss for any ckpt |
| `src/megalodon/data/attach_properties.py` | PyG transform: attach thermo + RDKit by SMILES |
| `src/megalodon/models/thermo_heads.py` | `ThermoHeadModel`, `RDKitHeadModel`, `CombinedHeadModel`, `AtomMolMP` |
| `src/megalodon/models/loss_fn.py` | `ThermoPropertyLoss`, `RDKitDescriptorLoss`, `CombinedPropertyLoss`, `EnergyPredictionLoss` |
| `scripts/conf/loqi/loqi_thermo_flow_*.yaml` | 4 thermo-aware configs (warm/cold × separate/combined head) |

For design rationale (target normalization, semi-supervised masking,
expected benchmarks, etc.), see `docs/ThermoGen_Implementation_Plan.md`.

---

## Citation

If you use LoQI in your research, please cite our paper:

```bibtex
@article{nikitin2025scalable,
  title={Scalable Low-Energy Molecular Conformer Generation with Quantum Mechanical Accuracy},
  author={Nikitin, Filipp and Anstine, Dylan M and Zubatyuk, Roman and Paliwal, Saee Gopal and Isayev, Olexandr},
  year={2025}
}
```

This work builds upon the Megalodon architecture. If you use the underlying architecture, please also cite:

```bibtex
@article{reidenbach2025applications,
  title={Applications of Modular Co-Design for De Novo 3D Molecule Generation},
  author={Reidenbach, Danny and Nikitin, Filipp and Isayev, Olexandr and Paliwal, Saee},
  journal={arXiv preprint arXiv:2505.18392},
  year={2025}
}
```
