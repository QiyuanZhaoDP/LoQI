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

import torch
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
        ThermoPropertyLoss,
    )
    tl_cfg = OmegaConf.select(cfg, "thermo_loss", default=None)
    el_cfg = OmegaConf.select(cfg, "energy_loss", default=None)
    thermo_loss = None
    energy_loss = None
    if tl_cfg is not None:
        thermo_loss = ThermoPropertyLoss(
            min_time=tl_cfg.min_time,
            weights=OmegaConf.to_container(tl_cfg.weights, resolve=True),
            target_mean=list(tl_cfg.target_mean),
            target_std=list(tl_cfg.target_std),
            timesteps=cfg.interpolant.timesteps,
        )
        logging.info(f"Enabled ThermoPropertyLoss (min_time={tl_cfg.min_time}, "
                     f"weights={thermo_loss.weights})")
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
    if thermo_loss is not None and energy_loss is not None:
        loss_fn = CombinedAuxiliaryLoss(thermo_loss=thermo_loss, energy_loss=energy_loss)
    else:
        loss_fn = thermo_loss or energy_loss

    batch_preprocessor = BatchPreProcessor(aug_rotations=cfg.data.aug_rotations,
                                        scale_coords=cfg.data.scale_coords)
    if cfg.resume:
        if os.path.isdir(cfg.resume):
            cfg.resume = f"{cfg.resume}/last.ckpt"
            ckpt = "last"
        else:
            ckpt = cfg.resume
        pl_module = Graph3DInterpolantModel.load_from_checkpoint(cfg.resume,
                                                                 loss_fn=loss_fn,
                                                                 loss_params=cfg.loss,
                                                                 interpolant_params=cfg.interpolant,
                                                                 sampling_params=cfg.sample,
                                                                 batch_preprocessor=batch_preprocessor)
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
    wandb_resume = cfg.wandb_params.resume if "resume" in cfg.wandb_params else "allow"
    logger = pl.loggers.WandbLogger(
        save_dir=cfg.outdir,
        project=cfg.wandb_params.project,
        group=cfg.wandb_params.group,
        name=cfg.run_name,
        id=cfg.run_name,
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