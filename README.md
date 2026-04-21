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

**Thermo-aware extensions (Phase 0 / Phase 1):**
- `loqi_thermo.yaml` — Phase 1, warm-started from `loqi.ckpt` (matches the
  256/10/4 backbone, adds a thermo head + auxiliary thermo loss)
- `loqi_thermo_50m.yaml` — Phase 1, from-scratch scaled backbone
  (384-d, 14 layers, 8 heads; ~100M params)

---

## ThermoGen: thermo-aware foundation model

Beyond pure conformer generation, this repo includes infrastructure for turning
LoQI into a thermodynamics-aware foundation model for downstream property
prediction. Two parallel tracks:

- **Phase 0** — *frozen backbone*: use `loqi.ckpt` as a feature extractor.
  Cheap, iterable, good for screening head architectures and benchmarking.
- **Phase 1** — *thermo-aware pre-training*: bake thermo prediction heads into
  `MegaFNV3Conf` and train them jointly with the denoising objective on
  ChEMBL3D. Produces a checkpoint whose every layer is shaped by thermo data.

Both paths share the same dataset preparation (gas-phase thermochemistry
labels from the TCIT group-additivity tool) and the same head architecture
(`ExtensiveSumHead` + multi-head `AtomMolMP`, defined in
`src/megalodon/models/thermo_heads.py`).

### Data preparation

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

# 3. Join labels onto each split (writes {train,val,test}_h_thermo.pt)
for split in train val test; do
  python scripts/label_thermo.py \
    --input  data/chembl3d_stereo/processed/${split}_h.pt \
    --output data/chembl3d_stereo/processed/${split}_h_thermo.pt \
    --labels data_processing/tcit_thermo_labels.csv \
    --neutralization-index data_processing/chembl3d_neutralization_index.json
done
```

### Phase 0 — frozen-backbone workflows

All Phase 0 pipelines keep the backbone frozen and train only small heads on
the per-atom features `H`. A single orchestrator exposes every stage:

```bash
# Extract H across splits on N GPUs, then train heads + (optional)
# end-to-end continuation with last-N layers unfrozen.
bash scripts/run_thermo.sh extract    # 4-GPU parallel H extraction
bash scripts/run_thermo.sh train      # head training on cached H
bash scripts/run_thermo.sh continue   # DDP continuation (unfreeze last N)
bash scripts/run_thermo.sh seeds      # ensemble seeds on remaining GPUs
bash scripts/run_thermo.sh all        # all of the above, sequentially
```

Knobs live in `scripts/conf/thermo/{finetune,continuation}.yaml`
(`n_mp_layers`, `mp_n_heads`, `head_hidden`, `lr`, `batch_size`, `lr_min`,
`unfreeze_layers`, `backbone_lr`, etc.) — edit in place, rerun.

#### Hyperparameter sweep (multi-GPU)

```bash
# Cartesian product over layers × heads × hidden × lr × batch_size;
# dispatches N-GPU-parallel waves, skips already-complete cells on re-run.
bash scripts/grid_search_thermo.sh
```

#### Reproduce val/x_loss of a checkpoint

```bash
python scripts/eval_loqi_loss.py \
    --ckpt data/loqi.ckpt \
    --config scripts/conf/loqi/loqi.yaml \
    --device cuda
# → prints the same val/x_loss that would appear in wandb during training
```

### Phase 1 — thermo-aware pre-training

Thermo heads are wired directly into `MegaFNV3Conf` (enable with
`dynamics.model_args.thermo_head_args` in the YAML). The denoising objective
and an auxiliary `ThermoPropertyLoss` — semi-supervised, NaN-masked, gated
to late timesteps via `min_time` — are optimized jointly:

```bash
# Warm-start from an existing LoQI checkpoint (same 256/10/4 architecture,
# adds heads + thermo loss, ~1 day on 4 GPUs).
python scripts/train.py --config-name=loqi_thermo \
    outdir=./outputs/loqi_thermo \
    train.gpus=4

# From-scratch scaled backbone (384/14/8 ≈ 100M params, ~3-5 days on 4 GPUs).
python scripts/train.py --config-name=loqi_thermo_50m \
    outdir=./outputs/loqi_thermo_50m \
    train.gpus=4
```

wandb traces to watch:

- `train/x_loss` — denoising, should stay flat (preserve LoQI quality)
- `train/additional_loss_term` — thermo aux, should decrease
- `train/thermo_last_labeled_active` — per-step labeled-molecule count
- `val/opt_median_relative_energy` — canary for conformer-generation quality

### Downstream property prediction (5-fold CV)

Given CSV files with `(SMILES, target)` columns, run a full pipeline that
3D-embeds each molecule, extracts H once, and performs 5-fold CV with the
thermo heads:

```bash
# Single dataset
python scripts/prepare_downstream_dataset.py \
    --csv data/downstream/delaney.csv \
    --smiles-col smiles --target-col "measured log solubility in mols per litre" \
    --output data/downstream_pt/delaney.pt

python scripts/downstream_cv.py \
    --ckpt data/loqi.ckpt --config scripts/conf/loqi/loqi.yaml \
    --dataset-pt data/downstream_pt/delaney.pt \
    --out-dir /tmp/downstream_cv/delaney \
    --n-folds 5 --epochs 50 --lr 3e-4 --device cuda

# Batch over all CSVs listed in TASKS[]
bash scripts/run_downstream_all.sh
```

Per-dataset reports land in `$OUT_ROOT/<name>/cv_report.json`; the batch
runner prints a cross-dataset summary table at the end.

### Key scripts

| path | role |
|---|---|
| `data_processing/parse_tcit_log.py` | TCIT log → CSV of per-SMILES thermo labels |
| `data_processing/build_neutralization_index.py` | canonical → neutral SMILES map |
| `scripts/label_thermo.py` | attach TCIT labels to a chembl3d `.pt` |
| `scripts/label_energy.py` | attach AIMNet2 energies to a chembl3d `.pt` (Phase 1 extension) |
| `scripts/probe_representation.py` | frozen-H Ridge probe (Phase 0 baseline) |
| `scripts/finetune_thermo_head.py` | cached-H head training |
| `scripts/continuation_training.py` | unfreeze last-N backbone layers + DDP |
| `scripts/grid_search_thermo.sh` | parallel hparam sweep with resume |
| `scripts/eval_loqi_loss.py` | reproduce val/x_loss for any ckpt |
| `scripts/prepare_downstream_dataset.py` | CSV → PyG `.pt` with 3D conformer |
| `scripts/downstream_cv.py` | 5-fold CV on a downstream dataset |
| `scripts/run_thermo.sh` | Phase 0 stage orchestrator |
| `scripts/run_downstream_all.sh` | batch runner for multiple downstream CSVs |
| `src/megalodon/models/thermo_heads.py` | shared head architectures |
| `src/megalodon/models/loss_fn.py` | `ThermoPropertyLoss`, `EnergyPredictionLoss` |
| `scripts/conf/thermo/*.yaml` | Phase 0 head/training hyperparameters |
| `scripts/conf/loqi/loqi_thermo*.yaml` | Phase 1 pre-training configs |

For a detailed walkthrough of the design choices, target normalization,
semi-supervised masking, and expected benchmark numbers, see
`docs/ThermoGen_Implementation_Plan.md`.

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
