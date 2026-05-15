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
    dm = MoleculeDataModule(
        cfg.data.dataset_root,
        cfg.data.processed_folder,
        cfg.data.batch_size,
        cfg.data.data_loader_type,
        cfg.data.inference_batch_size,
    )
    val_loader = dm.val_dataloader()

    # --- Validate via Lightning (runs validation_step → calculate_loss) ---
    accelerator = "gpu" if args.device.startswith("cuda") else "cpu"
    trainer_kwargs = dict(
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=False,
    )
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
