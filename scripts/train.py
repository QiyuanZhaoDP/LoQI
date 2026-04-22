# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import logging
from pathlib import Path

def _is_rank0():
    return os.environ.get("LOCAL_RANK", "0") == "0"


def _print_param_breakdown(pl_module):
    """Detailed parameter-count breakdown of a Graph3DInterpolantModel.

    Groups the monolithic `dynamics` ModuleWithEMA into:
      - dynamics.model.<backbone pieces>  (trainable)
      - dynamics.model.thermo_heads.ext   (trainable, if present)
      - dynamics.model.thermo_heads.mp    (trainable, if present)
      - dynamics.model.energy_head        (trainable, if present)
      - dynamics.ema_model                (frozen shadow, 1× the inner model)
      - self_conditioning_module          (if present)
    and prints trainable / frozen / total roll-ups.
    """
    def count(module):
        return sum(p.numel() for p in module.parameters())

    def count_trainable(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    lines = ["", "=" * 66, "Parameter breakdown", "=" * 66]

    def _row(name, n, indent=2):
        lines.append(f"{' ' * indent}{name:<45s}{n:>15,}")

    dyn = pl_module.dynamics
    inner = dyn.model if hasattr(dyn, "model") else dyn

    lines.append("dynamics.model  (trainable base)")
    backbone_subtotal = 0
    head_rows = []
    for name, child in inner.named_children():
        n = count(child)
        if name == "thermo_heads":
            for sub_name, sub in child.named_children():
                head_rows.append((f"thermo_heads.{sub_name}", count(sub)))
        elif name == "energy_head":
            head_rows.append(("energy_head", n))
        else:
            backbone_subtotal += n
    _row("backbone (embedders + DiT/XEGNN stack + coord head)", backbone_subtotal)
    for name, n in head_rows:
        _row(name, n)
    _row("subtotal (dynamics.model)", count(inner))

    if hasattr(dyn, "ema_model") and dyn.ema_model is not None:
        lines.append("")
        lines.append("dynamics.ema_model  (frozen EMA shadow, mirrors dynamics.model)")
        _row("subtotal", count(dyn.ema_model))

    if getattr(pl_module, "self_conditioning_module", None) is not None:
        lines.append("")
        lines.append("self_conditioning_module")
        _row("subtotal", count(pl_module.self_conditioning_module))

    lines.append("-" * 66)
    total = count(pl_module)
    trainable = count_trainable(pl_module)
    lines.append(f"  {'Total params':<45s}{total:>15,}")
    lines.append(f"  {'Trainable':<45s}{trainable:>15,}  ({100*trainable/max(total,1):.1f}%)")
    lines.append(f"  {'Frozen (incl. EMA shadow)':<45s}{total-trainable:>15,}")
    lines.append("=" * 66)
    return "\n".join(lines)

import torch
import torch.multiprocessing as _torch_mp
# Use the file_system sharing strategy instead of file_descriptor so that
# DataLoader workers don't exhaust FDs under heavy num_workers × DDP setups
# (symptom: "RuntimeError: received 0 items of ancdata" from the pin-memory
# thread). Costs a small IPC overhead but is required for >8 workers/rank.
_torch_mp.set_sharing_strategy("file_system")

import hydra
from lightning import pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from megalodon.data.batch_preprocessor import BatchPreProcessor
from megalodon.data.molecule_datamodule import MoleculeDataModule
from megalodon.data.statistics import Statistics
from megalodon.metrics.molecule_evaluation_callback import MoleculeEvaluationCallback
from megalodon.metrics.conformer_evaluation_callback import ConformerEvaluationCallback
from megalodon.models.module import Graph3DInterpolantModel


@hydra.main(version_base=None, config_path="conf/loqi", config_name=None)
def main(cfg: DictConfig) -> None:
    """
    This is the main function conducting data loading and model training.
    """
    logging.info("\n\n************** Experiment Configuration ***********")
    pl.seed_everything(cfg.train.seed)
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")
    cfg.outdir = os.path.join(cfg.outdir, cfg.run_name)
    os.makedirs(cfg.outdir, exist_ok=True)
    os.makedirs(os.path.join(cfg.outdir, 'checkpoints'), exist_ok=True)
    # Optional auxiliary losses for thermo-aware / energy-aware pre-training
    # (Phase 1/2). Enabled by non-null `thermo_loss` / `energy_loss` sections
    # in the YAML. Both can be active simultaneously via CombinedAuxiliaryLoss.
    from megalodon.models.loss_fn import (
        CombinedAuxiliaryLoss,
        EnergyPredictionLoss,
        RDKitDescriptorLoss,
        ThermoPropertyLoss,
    )
    tl_cfg = OmegaConf.select(cfg, "thermo_loss", default=None)
    rl_cfg = OmegaConf.select(cfg, "rdkit_loss",  default=None)
    el_cfg = OmegaConf.select(cfg, "energy_loss", default=None)
    thermo_loss = None
    rdkit_loss = None
    energy_loss = None
    if tl_cfg is not None:
        thermo_loss = ThermoPropertyLoss(
            min_time=tl_cfg.min_time,
            weight=float(OmegaConf.select(tl_cfg, "weight", default=0.05)),
            target_mean=list(tl_cfg.target_mean),
            target_std=list(tl_cfg.target_std),
            timesteps=cfg.interpolant.timesteps,
        )
        logging.info(f"Enabled ThermoPropertyLoss (min_time={tl_cfg.min_time}, "
                     f"weight={thermo_loss.weight})")
    if rl_cfg is not None:
        rdkit_loss = RDKitDescriptorLoss(
            min_time=rl_cfg.min_time,
            weight=float(OmegaConf.select(rl_cfg, "weight", default=0.02)),
            target_mean=list(rl_cfg.target_mean),
            target_std=list(rl_cfg.target_std),
            timesteps=cfg.interpolant.timesteps,
        )
        logging.info(f"Enabled RDKitDescriptorLoss (min_time={rl_cfg.min_time}, "
                     f"weight={rdkit_loss.weight})")
    if el_cfg is not None:
        energy_loss = EnergyPredictionLoss(
            min_time=el_cfg.min_time,
            weight=el_cfg.weight,
            normalize=el_cfg.get("normalize", "per_atom"),
            timesteps=cfg.interpolant.timesteps,
            target_mean=OmegaConf.select(el_cfg, "target_mean", default=None),
            target_std=OmegaConf.select(el_cfg,  "target_std",  default=None),
        )
        logging.info(f"Enabled EnergyPredictionLoss (min_time={el_cfg.min_time}, "
                     f"weight={el_cfg.weight})")
    _active = [x for x in (thermo_loss, rdkit_loss, energy_loss) if x is not None]
    if len(_active) > 1:
        loss_fn = CombinedAuxiliaryLoss(thermo_loss=thermo_loss,
                                         rdkit_loss=rdkit_loss,
                                         energy_loss=energy_loss)
    else:
        loss_fn = _active[0] if _active else None

    batch_preprocessor = BatchPreProcessor(aug_rotations=cfg.data.aug_rotations,
                                        scale_coords=cfg.data.scale_coords)
    if cfg.resume:
        if os.path.isdir(cfg.resume):
            cfg.resume = f"{cfg.resume}/last.ckpt"
        # resume_strict controls BOTH:
        #   * state_dict strictness in load_from_checkpoint (False lets new
        #     heads initialize randomly without matching keys)
        #   * whether we hand trainer.fit the ckpt_path for full restoration:
        #     if strict=False, the architecture changed → optimizer /
        #     scheduler / epoch / grad state in the checkpoint doesn't match
        #     the current model, so we must NOT let Lightning restore them.
        #     This is warm-start semantics: weights yes, training-state no.
        resume_strict = bool(OmegaConf.select(cfg, "train.resume_strict",
                                               default=False))
        pl_module = Graph3DInterpolantModel.load_from_checkpoint(
            cfg.resume,
            loss_fn=loss_fn,
            loss_params=cfg.loss,
            dynamics_params=cfg.dynamics,
            interpolant_params=cfg.interpolant,
            sampling_params=cfg.sample,
            batch_preprocessor=batch_preprocessor,
            strict=resume_strict,
        )
        if resume_strict:
            ckpt = cfg.resume     # full resume — optimizer + epoch restored
            logging.info(f"Resumed from {cfg.resume} (strict=True, full state).")
        else:
            ckpt = None           # warm-start — weights only, fresh training state
            logging.info(f"Warm-started from {cfg.resume} (strict=False): "
                          f"pretrained weights loaded, optimizer / scheduler / "
                          f"epoch start fresh. Any new heads get random init.")
    else:
        pl_module = Graph3DInterpolantModel(
            loss_params=cfg.loss,
            optimizer_params=cfg.optimizer,
            lr_scheduler_params=cfg.lr_scheduler,
            dynamics_params=cfg.dynamics,
            interpolant_params=cfg.interpolant,
            sampling_params=cfg.sample,
            self_cond_params=OmegaConf.select(cfg, "self_conditioning", default=None),
            ema=OmegaConf.select(cfg, "ema", default=True),
            loss_fn=loss_fn,
            batch_preprocessor=batch_preprocessor
        )
        ckpt = None

    if _is_rank0():
        logging.info(_print_param_breakdown(pl_module))

    wandb_resume = cfg.wandb_params.resume if "resume" in cfg.wandb_params else "allow"
    # Decouple the wandb run ID from the human-readable run name. If a prior
    # run with the same name gets deleted on wandb.ai, the ID becomes
    # permanently blocked — reusing run_name as id would then fail every
    # future sync. Let wandb auto-generate a fresh ID unless the YAML
    # explicitly pins one via cfg.wandb_params.id (handy for resume).
    explicit_id = OmegaConf.select(cfg, "wandb_params.id", default=None)
    logger = pl.loggers.WandbLogger(
        save_dir=cfg.outdir,
        project=cfg.wandb_params.project,
        group=cfg.wandb_params.group,
        name=cfg.run_name,
        id=explicit_id,                       # None -> wandb generates an ID
        resume=wandb_resume,
        mode=cfg.wandb_params.mode,
    )
    logger.log_hyperparams(cfg)

    datamodule = MoleculeDataModule(
        cfg.data.dataset_root,
        cfg.data.processed_folder,
        cfg.data.batch_size,
        cfg.data.data_loader_type,
        cfg.data.inference_batch_size,
        data_suffix=OmegaConf.select(cfg, "data.data_suffix", default="_h"),
        property_table=OmegaConf.select(cfg, "data.property_table", default=None),
        num_workers=int(OmegaConf.select(cfg, "data.num_workers", default=8)),
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    last_checkpoint_callback = ModelCheckpoint(
        dirpath=Path(cfg.outdir, 'checkpoints'),
        save_last=True,
        every_n_train_steps=cfg.train.checkpoint_every_n_train_steps,
        save_on_train_epoch_end=True,
        filename="last-{epoch}-{step}",
    )
    metric_name = cfg.train.checkpoint_monitor.replace("/", "_")
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=Path(cfg.outdir, 'checkpoints'),
        save_top_k=5,
        monitor=cfg.train.checkpoint_monitor,
        mode=cfg.train.checkpoint_monitor_mode,
        filename="best-{epoch}-{step}--{" + metric_name + ":.3f}",
    )
    # Additional checkpoint based on train loss as backup (saves best 5 by train loss epoch avg)
    train_loss_checkpoint_callback = ModelCheckpoint(
        dirpath=Path(cfg.outdir, 'checkpoints'),
        save_top_k=5,
        monitor="train/loss_epoch",
        mode="min",
        save_on_train_epoch_end=True,
        filename="best_train-{epoch}-{step}",
    )
    if cfg.evaluation.type == "molecules":
        energy_metrics_args = OmegaConf.to_container(cfg.evaluation.energy_metrics_args,
                                                 resolve=True) if cfg.evaluation.energy_metrics_args is not None else None
        statistics = Statistics.load_statistics(
            statistics_dir=f"{cfg.data.dataset_root}/{cfg.data.processed_folder}",
            split_name="train")
        evaluation_callback = MoleculeEvaluationCallback(
            n_graphs=cfg.evaluation.n_molecules,
            batch_size=cfg.evaluation.batch_size,
            timesteps=cfg.evaluation.timesteps,
            train_smiles=datamodule.train_dataset.smiles,
            statistics=statistics,
            compute_2D_metrics=cfg.evaluation.compute_2D_metrics,
            compute_3D_metrics=cfg.evaluation.compute_3D_metrics,
            compute_train_data_metrics=cfg.evaluation.compute_train_data_metrics,
            compute_energy_metrics=cfg.evaluation.compute_energy_metrics,
            energy_metrics_args=energy_metrics_args,
            scale_coords=cfg.evaluation.scale_coords,
            preserve_aromatic=OmegaConf.select(cfg.evaluation, "preserve_aromatic", default=True)
        )

    elif cfg.evaluation.type == "conformers":
        energy_metrics_args = OmegaConf.to_container(cfg.evaluation.energy_metrics_args,
                                                 resolve=True) if cfg.evaluation.energy_metrics_args is not None else None
        statistics = Statistics.load_statistics(
            statistics_dir=f"{cfg.data.dataset_root}/{cfg.data.processed_folder}",
            split_name="train")
        evaluation_callback = ConformerEvaluationCallback(
            statistics=statistics,
            max_molecules=cfg.evaluation.max_molecules,
            timesteps=cfg.evaluation.timesteps,
            compute_3D_metrics=cfg.evaluation.compute_3D_metrics,
            compute_energy_metrics=cfg.evaluation.compute_energy_metrics,
            energy_metrics_args=energy_metrics_args,
            scale_coords=cfg.evaluation.scale_coords,
            compute_stereo_metrics=cfg.evaluation.compute_stereo_metrics
        )
    else: 
        raise NotImplementedError

    if 'num_nodes' in cfg.train:
        num_nodes = cfg.train.num_nodes
    else:
        num_nodes = 1

    # When an auxiliary loss (thermo / energy) is enabled, the corresponding
    # heads may produce outputs that ARE in the autograd graph for some
    # batches but NOT for others — e.g. ThermoPropertyLoss with min_time=0.8
    # contributes only when ~20% of the timesteps pass the gate, so ~80% of
    # batches leave the thermo head params without gradients. Plain DDP
    # raises "unused parameters" on the first such batch; we flip on the
    # tolerant variant when we know an aux loss is in play.
    if cfg.train.gpus > 1:
        needs_find_unused = loss_fn is not None
        strategy = 'ddp_find_unused_parameters_true' if needs_find_unused else 'ddp'
        if needs_find_unused:
            logging.info("Using DDP strategy 'ddp_find_unused_parameters_true' "
                         "(auxiliary loss has conditionally-used heads).")
    else:
        strategy = 'auto'

    trainer = pl.Trainer(
        max_epochs=cfg.train.n_epochs,
        logger=logger,
        callbacks=[lr_monitor, evaluation_callback, last_checkpoint_callback,
                   best_checkpoint_callback, train_loss_checkpoint_callback],
        enable_progress_bar=cfg.train.enable_progress_bar,
        accelerator='gpu',
        devices=cfg.train.gpus,
        num_nodes=num_nodes,
        strategy=strategy,
        check_val_every_n_epoch=cfg.train.val_freq,
        gradient_clip_val=cfg.train.gradient_clip_value,
        log_every_n_steps=cfg.train.log_freq,  # for train steps
        num_sanity_val_steps=0
    )

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    trainer.fit(model=pl_module, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=ckpt)


if __name__ == "__main__":
    main()