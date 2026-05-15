"""Reproduce `val/x_loss` (and related metrics) for a LoQI checkpoint
exactly as they would be computed during training.

Uses Lightning's `trainer.validate(...)` — which calls the model's
validation_step on every batch in the val loader and returns the
aggregated metrics dict. No custom re-implementation of the loss, no
time-mode hacks; the number you get here is the same number that would
appear in wandb as `val/x_loss` on an epoch boundary.

Usage:
    python scripts/eval_loqi_loss.py \\
        --ckpt data/loqi.ckpt \\
        --config scripts/conf/loqi/loqi.yaml \\
        --device cuda
"""
import argparse
import json

import torch
import torch.multiprocessing as _torch_mp
# Use file_system sharing strategy so DataLoader workers don't exhaust
# file descriptors under high num_workers (symptom: "RuntimeError:
# received 0 items of ancdata" → pin-memory thread dies). Mirrors what
# scripts/train.py does for the same reason.
_torch_mp.set_sharing_strategy("file_system")

from lightning import pytorch as pl
from omegaconf import OmegaConf

from megalodon.data.batch_preprocessor import BatchPreProcessor
from megalodon.data.molecule_datamodule import MoleculeDataModule
from megalodon.models.module import Graph3DInterpolantModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True,
                   help="LoQI backbone config (same YAML used to train the ckpt).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit-batches", type=float, default=None,
                   help="Pass Lightning's limit_val_batches (int or float in [0,1]). "
                        "Default None = full val set.")
    p.add_argument("--n-gpus", type=int, default=1,
                   help="Number of GPUs for DDP val. >1 enables strategy=ddp "
                        "so each rank processes a shard of the val set; "
                        "metrics are sync_dist-aggregated. Note: MiDiDataloader "
                        "iterates the full dataset per rank (no DistributedSampler) "
                        "so >1 GPU only helps if Lightning auto-injects the "
                        "sampler (depends on data_loader_type). 1 GPU is usually "
                        "enough for a one-shot val.")
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)

    # --- Build auxiliary loss_fn from YAML (so per-target val metrics get logged)
    # Without this, validation_step skips `val/additional_loss_term` and the
    # per-target `val/combined/mae_*` block entirely — what the user reports
    # as "metric is missing".
    from megalodon.models.loss_fn import (
        CombinedAuxiliaryLoss,
        CombinedPropertyLoss,
        EnergyPredictionLoss,
        RDKitDescriptorLoss,
        ThermoPropertyLoss,
    )
    tl_cfg = OmegaConf.select(cfg, "thermo_loss",   default=None)
    rl_cfg = OmegaConf.select(cfg, "rdkit_loss",    default=None)
    el_cfg = OmegaConf.select(cfg, "energy_loss",   default=None)
    cl_cfg = OmegaConf.select(cfg, "combined_loss", default=None)
    thermo_loss = rdkit_loss = energy_loss = combined_loss = None
    if tl_cfg is not None:
        thermo_loss = ThermoPropertyLoss(
            min_time=tl_cfg.min_time,
            weight=float(OmegaConf.select(tl_cfg, "weight", default=0.05)),
            target_mean=list(tl_cfg.target_mean),
            target_std=list(tl_cfg.target_std),
            timesteps=cfg.interpolant.timesteps,
        )
    if rl_cfg is not None:
        rdkit_loss = RDKitDescriptorLoss(
            min_time=rl_cfg.min_time,
            weight=float(OmegaConf.select(rl_cfg, "weight", default=0.02)),
            target_mean=list(rl_cfg.target_mean),
            target_std=list(rl_cfg.target_std),
            timesteps=cfg.interpolant.timesteps,
        )
    if el_cfg is not None:
        energy_loss = EnergyPredictionLoss(
            min_time=el_cfg.min_time, weight=el_cfg.weight,
            normalize=el_cfg.get("normalize", "per_atom"),
            timesteps=cfg.interpolant.timesteps,
            target_mean=OmegaConf.select(el_cfg, "target_mean", default=None),
            target_std=OmegaConf.select(el_cfg,  "target_std",  default=None),
        )
    if cl_cfg is not None:
        combined_loss = CombinedPropertyLoss(
            min_time=cl_cfg.min_time,
            thermo_weight=float(OmegaConf.select(cl_cfg, "thermo_weight", default=0.1)),
            rdkit_weight=float(OmegaConf.select(cl_cfg, "rdkit_weight", default=0.02)),
            target_weights=OmegaConf.select(cl_cfg, "target_weights", default=None),
            target_mean=list(cl_cfg.target_mean),
            target_std=list(cl_cfg.target_std),
            timesteps=cfg.interpolant.timesteps,
        )
    _active = [x for x in (thermo_loss, rdkit_loss, energy_loss, combined_loss)
                if x is not None]
    if len(_active) > 1:
        loss_fn = CombinedAuxiliaryLoss(thermo_loss=thermo_loss,
                                         rdkit_loss=rdkit_loss,
                                         energy_loss=energy_loss,
                                         combined_loss=combined_loss)
    else:
        loss_fn = _active[0] if _active else None
    if loss_fn is not None:
        print(f"Auxiliary loss_fn: {type(loss_fn).__name__}")

    # --- Load model ---
    print(f"Loading {args.ckpt}")
    pre = BatchPreProcessor(cfg.data.aug_rotations, cfg.data.scale_coords)
    pl_module = Graph3DInterpolantModel.load_from_checkpoint(
        args.ckpt,
        loss_params=cfg.loss,
        interpolant_params=cfg.interpolant,
        sampling_params=cfg.sample,
        batch_preprocessor=pre,
        loss_fn=loss_fn,
    )

    # --- Data ---
    # Pass property_table so the AttachProperties transform injects thermo /
    # rdkit / combined target fields into each batch. Without this the val
    # step crashes inside CombinedPropertyLoss with KeyError: 'enthalpy_298'.
    dm = MoleculeDataModule(
        cfg.data.dataset_root,
        cfg.data.processed_folder,
        cfg.data.batch_size,
        cfg.data.data_loader_type,
        cfg.data.inference_batch_size,
        property_table=OmegaConf.select(cfg, "data.property_table", default=None),
    )
    val_loader = dm.val_dataloader()

    # --- Validate via Lightning (runs validation_step → calculate_loss) ---
    accelerator = "gpu" if args.device.startswith("cuda") else "cpu"
    n_gpus = max(1, int(args.n_gpus)) if accelerator == "gpu" else 1
    trainer_kwargs = dict(
        accelerator=accelerator,
        devices=n_gpus,
        strategy="ddp" if n_gpus > 1 else "auto",
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        use_distributed_sampler=True,   # try to shard the val loader
    )
    if n_gpus > 1:
        print(f"  ddp val on {n_gpus} GPUs")
    if args.limit_batches is not None:
        trainer_kwargs["limit_val_batches"] = (
            int(args.limit_batches)
            if args.limit_batches >= 1
            else args.limit_batches
        )
    trainer = pl.Trainer(**trainer_kwargs)

    results = trainer.validate(model=pl_module, dataloaders=val_loader)

    # trainer.validate returns a list of dict (one per dataloader). One loader
    # → one dict with keys like 'val/loss', 'val/x_loss', etc.
    print("\n" + "=" * 72)
    print("val metrics (same formulation as wandb's val/* during training):")
    print("-" * 72)
    for i, d in enumerate(results):
        for k, v in sorted(d.items()):
            print(f"  {k:<40s} {v:>15.6f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
