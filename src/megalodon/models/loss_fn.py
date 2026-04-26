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

inner_to_atomic_number = {
    0: 1,    # H
    1: 5,    # B
    2: 6,    # C
    3: 7,    # N
    4: 8,    # O
    5: 9,    # F
    6: 13,   # Al
    7: 14,   # Si
    8: 15,   # P
    9: 16,   # S
    10: 17,  # Cl
    11: 33,  # As
    12: 35,  # Br
    13: 53,  # I
    14: 80,  # Hg
    15: 83,  # Bi
    16: 34   # Se
}

import torch
import torch.nn as nn
from typing import Dict
from torch import Tensor

class Forces(nn.Module):
    def __init__(self, module: nn.Module, x: str = 'coord', y: str = 'energy', key_out: str = 'forces'):
        super().__init__()
        self.module = module
        self.x = x
        self.y = y
        self.key_out = key_out

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        data[self.x].requires_grad_(True)
        data = self.module(data)
        y = data[self.y]
        g = torch.autograd.grad([y.sum()], [data[self.x]], create_graph=self.training)[0]
        assert g is not None
        data[self.key_out] = -g
        torch.set_grad_enabled(prev)
        return data

class AIMNet2ForcesLoss:
    def __init__(self, model_path, charge="charges", max_forces=0.05, min_time=0.9, atomics="h", coord="x", weight=1.0):
        super().__init__()
        self.nnip = Forces(torch.load(model_path))
        self.max_forces = max_forces
        self.min_time = min_time
        self.atomics = atomics
        self.coord = coord
        self.charge = charge
        self.weight = weight

    def __call__(self, batch, out, time, ws_t, stage="train"): 
        coord = out[f"{self.coord}_hat"]
        atomics = batch[self.atomics]
        charge = batch[self.charge]

        device = coord.device

        self.nnip.to(device)

        # Prepare AIMNet2 input batch
        aimnet2_batch = self.prepare_aimnet2batch(coord, atomics, charge, batch["batch"])

        # Calculate forces using AIMNet2
        forces = self.nnip(aimnet2_batch)["forces"]

        # Compute force loss
        loss_forces = torch.sum(torch.sum(torch.square(forces), dim=-1), dim=-1)

        # Normalize loss by the number of atoms per molecule
        num_atoms = (aimnet2_batch["numbers"] > 0).sum(dim=-1)
        loss_forces = loss_forces / num_atoms

        # Apply constraints and weighting
        loss_forces[torch.isnan(loss_forces)] = 0.0
        loss_forces[loss_forces > self.max_forces] = 0.0
        loss_forces[time < self.min_time] = 0.0

        # Final weighted loss
        loss = (loss_forces * ws_t).mean()*self.weight
        return loss

    @staticmethod
    def prepare_aimnet2batch(coord, atomics, charge, batch_idx):
        """
        Prepare data for AIMNet2 input format.

        Args:
            coord (Tensor): Tensor of atomic coordinates [n_atoms, 3].
            atomics (Tensor): Tensor of atom types in source format.
            charge (Tensor): Tensor of atomic charges.
            batch_idx (Tensor): Batch indices for the atoms.

        Returns:
            Dict[str, Tensor]: AIMNet2-compatible batch.
        """
        device = coord.device

        # Convert atomics to atomic numbers
        atomic_numbers = torch.tensor([inner_to_atomic_number[a.item()] for a in atomics], device=device)

        # Create batch tensors
        n_molecules = batch_idx.max().item() + 1
        max_n_atoms = torch.bincount(batch_idx).max().item()

        batch_coord = torch.zeros((n_molecules, max_n_atoms, 3), device=device)
        batch_atomics = torch.zeros((n_molecules, max_n_atoms), device=device).long()
        batch_charge = torch.zeros(n_molecules, device=device).long()

        for i in range(n_molecules):
            mask = batch_idx == i
            n_atoms = mask.sum().item()
            batch_coord[i, :n_atoms] = coord[mask]
            batch_atomics[i, :n_atoms] = atomic_numbers[mask]
            batch_charge[i] = charge[mask].sum() - 2 * n_atoms

        return {"coord": batch_coord, "numbers": batch_atomics, "charge": batch_charge}


# =============================================================================
# Phase 1/2 auxiliary losses for thermo-aware foundation-model pre-training.
# =============================================================================

TARGET_FIELDS_DEFAULT = ["enthalpy_298", "gibbs_298", "cv_gas",
                          "entropy_gas", "enthalpy_0"]

RDKIT_FIELDS_DEFAULT = ["logp", "tpsa", "n_h_donors", "n_h_acceptors",
                         "n_rot_bonds", "frac_csp3", "n_aliph_rings",
                         "qed", "labute_asa"]


def _t_frac(time, timesteps):
    """Normalize time to [0, 1] fraction regardless of discrete/continuous."""
    if time.is_floating_point():
        return time
    return time.float() / max(timesteps - 1, 1)


import torch.nn as _nn
from torchmetrics import (
    MeanAbsoluteError as _TM_MAE,
    MeanSquaredError as _TM_MSE,
    R2Score as _TM_R2,
    MeanMetric as _TM_Mean,
)


class _NNModuleSetstateMixin:
    """Workaround for a pickle backward-compat trap: the auxiliary-loss
    classes used to be plain Python classes; now they're nn.Module.
    Default pickle restores `__dict__` directly without calling __init__,
    so unpickling an OLD ckpt's hparams produces an instance whose
    nn.Module internals (`_buffers`, `_parameters`, `_modules`, ...) were
    never initialized — load_state_dict then crashes with
    `AttributeError: '...' object has no attribute '_buffers'`.

    Fix: run nn.Module.__init__ first to set up the internal dicts, then
    layer the saved __dict__ on top.
    """

    def __setstate__(self, state):
        _nn.Module.__init__(self)
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
            # PyTorch's serialization can hand a (slot_state, dict_state) tuple.
            slot_state, dict_state = state
            if slot_state:
                for k, v in slot_state.items():
                    setattr(self, k, v)
            self.__dict__.update(dict_state)
        else:
            self.__dict__.update(state)


def _build_per_target_metric_set(target_fields):
    """Per-target torchmetrics bundle: MAE, RMSE, R², plus running first/second
    moments of pred and target (for std diagnostics). Each batch .update()s the
    active subset; epoch-end .compute() gives proper cross-batch aggregation,
    then .reset() clears for the next epoch."""
    return _nn.ModuleDict({
        f: _nn.ModuleDict({
            "mae":         _TM_MAE(),
            "rmse":        _TM_MSE(squared=False),  # returns sqrt(mean(err²))
            "r2":          _TM_R2(),
            "pred_mean":   _TM_Mean(),
            "pred_msq":    _TM_Mean(),              # E[pred²] for std
            "target_mean": _TM_Mean(),
            "target_msq":  _TM_Mean(),
        }) for f in target_fields
    })


def _compute_and_reset_metric_set(metric_set, prefix):
    """Epoch-end: .compute() every metric, build a flat name → scalar dict,
    then .reset(). `prefix` is the namespace (e.g. 'thermo' / 'rdkit')."""
    def _safe(m):
        try:
            v = m.compute()
            return float(v.item()) if hasattr(v, "item") else float(v)
        except Exception:
            return float("nan")

    out = {}
    for f, mset in metric_set.items():
        mae  = _safe(mset["mae"])
        rmse = _safe(mset["rmse"])
        r2   = _safe(mset["r2"])
        pm   = _safe(mset["pred_mean"])
        pms  = _safe(mset["pred_msq"])
        tm   = _safe(mset["target_mean"])
        tms  = _safe(mset["target_msq"])
        # std from first + second moments; clamp at 0 to guard against
        # floating-point slop producing a tiny negative under-radical.
        pred_std   = max(pms - pm * pm,  0.0) ** 0.5
        target_std = max(tms - tm * tm,  0.0) ** 0.5

        out[f"{prefix}/mae_{f}"]        = mae
        out[f"{prefix}/rmse_{f}"]       = rmse
        out[f"{prefix}/r2_{f}"]         = r2
        out[f"{prefix}/pred_std_{f}"]   = pred_std
        out[f"{prefix}/target_std_{f}"] = target_std

        for m in mset.values():
            m.reset()
    return out


class ThermoPropertyLoss(_NNModuleSetstateMixin, _nn.Module):
    """Auxiliary thermo-prediction loss applied at late denoising timesteps.

    Reads per-molecule thermo targets (enthalpy_298, gibbs_298, cv_gas,
    entropy_gas, enthalpy_0) attached to the batch by the data pipeline's
    AttachProperties transform, plus a `thermo_has_label` bool flag.
    Semi-supervised: unlabeled molecules (NaN targets or
    `thermo_has_label == False`) contribute 0 to the loss.

    Requires the backbone to expose `out["thermo_mp"]` — obtained by
    building MegaFNV3Conf with `thermo_head_args` set.

    Per-target diagnostics exposed via compute_and_reset(stage):
        mae, rmse (physical units), r2, pred_std, target_std.
    torchmetrics accumulate sufficient statistics across every batch in
    the epoch and reduce correctly across DDP ranks on compute().

    Args:
        min_time: only apply loss when t >= min_time (as fraction in [0,1]).
        weight:   scalar weight for the MP head relative to main denoising loss.
        target_mean / target_std: per-target z-score stats (list of len 5).
        timesteps: needed to normalize discrete-time to fraction [0, 1].
    """

    _PREFIX = "thermo"

    def __init__(self, min_time=0.8,
                 weight=0.05,
                 target_fields=None,
                 target_mean=None,
                 target_std=None,
                 timesteps=25):
        super().__init__()
        self.min_time = min_time
        self.weight = float(weight)
        self.target_fields = list(target_fields) if target_fields else list(TARGET_FIELDS_DEFAULT)
        if target_mean is None or target_std is None:
            raise ValueError(
                "ThermoPropertyLoss requires target_mean and target_std "
                "(per-target z-score stats; compute via "
                "data_processing/compute_rdkit_stats.py)."
            )
        self.target_mean = list(target_mean)
        self.target_std = list(target_std)
        self.timesteps = timesteps
        # Z-score stats as registered buffers so .to(device) moves them
        # and they ride along in state_dict. Kept non-persistent to skip
        # checkpointing (they're reconstructed from the constructor args).
        self.register_buffer(
            "_mean_t",
            torch.tensor(self.target_mean, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_std_t",
            torch.tensor(self.target_std, dtype=torch.float32),
            persistent=False,
        )
        # Per-step label-density telemetry.
        self.last_batch_size = 0
        self.last_gated_in = 0
        self.last_labeled_active = 0
        # Per-target torchmetrics for proper epoch-wise MAE/RMSE/R² etc.
        # Two independent sets so train and val accumulators don't collide.
        self.train_metrics = _build_per_target_metric_set(self.target_fields)
        self.val_metrics   = _build_per_target_metric_set(self.target_fields)

    def forward(self, batch, out, time, ws_t, stage="train"):
        if "thermo_mp" not in out or self.weight <= 0:
            return torch.tensor(0.0, device=time.device)

        device = time.device

        t_frac = _t_frac(time, self.timesteps)
        time_mask = t_frac >= self.min_time
        if not time_mask.any():
            return torch.tensor(0.0, device=device)

        targets = torch.stack(
            [batch[f].view(-1).float() for f in self.target_fields], dim=1
        )  # [B, 5]

        if hasattr(batch, "thermo_has_label"):
            has_label = batch.thermo_has_label.view(-1).bool()
        else:
            has_label = ~torch.isnan(targets).any(dim=1)

        mol_mask = time_mask & has_label
        self.last_batch_size = int(time_mask.numel())
        self.last_gated_in = int(time_mask.sum().item())
        self.last_labeled_active = int(mol_mask.sum().item())
        if not mol_mask.any():
            return torch.tensor(0.0, device=device)

        targets_norm = (targets - self._mean_t) / self._std_t

        pred = out["thermo_mp"]
        valid = mol_mask.unsqueeze(-1) & ~torch.isnan(targets_norm)
        if not valid.any():
            return torch.tensor(0.0, device=device)
        diff = (pred - torch.nan_to_num(targets_norm)) * valid

        # Update torchmetrics on the physical-unit predictions/targets.
        with torch.no_grad():
            pred_phys   = pred * self._std_t + self._mean_t
            target_phys = torch.nan_to_num(targets)
            metrics = self.train_metrics if stage == "train" else self.val_metrics
            for i, f in enumerate(self.target_fields):
                col_valid = valid[:, i]
                n_v = int(col_valid.sum().item())
                if n_v == 0:
                    continue
                p = pred_phys[col_valid, i]
                t = target_phys[col_valid, i]
                metrics[f]["mae"].update(p, t)
                metrics[f]["rmse"].update(p, t)
                if n_v >= 2:
                    metrics[f]["r2"].update(p, t)
                metrics[f]["pred_mean"].update(p)
                metrics[f]["pred_msq"].update(p ** 2)
                metrics[f]["target_mean"].update(t)
                metrics[f]["target_msq"].update(t ** 2)

        return self.weight * (diff ** 2).sum() / valid.sum()

    def compute_and_reset(self, stage):
        """Called from the Lightning module at on_{train,validation}_epoch_end.
        Returns all per-target metrics as a flat {name: scalar} dict and
        resets the accumulators for the next epoch."""
        metrics = self.train_metrics if stage == "train" else self.val_metrics
        return _compute_and_reset_metric_set(metrics, self._PREFIX)

    def get_metrics_dict(self):
        """Per-step telemetry (label density). Per-target MAE/RMSE etc. are
        NOT returned here — they're computed properly at epoch end."""
        return {
            f"{self._PREFIX}/batch_size":     float(self.last_batch_size),
            f"{self._PREFIX}/gated_in":       float(self.last_gated_in),
            f"{self._PREFIX}/labeled_active": float(self.last_labeled_active),
        }


class RDKitDescriptorLoss(_NNModuleSetstateMixin, _nn.Module):
    """Auxiliary RDKit-descriptor prediction loss (9 targets, 100% coverage).

    Mirrors ThermoPropertyLoss but without the has_thermo_label gate —
    build_property_table.py drops all-NaN RDKit rows, so every mol in the
    parquet has valid descriptors.

    Default weight 0.02 (smaller than thermo 0.05) — regularizer, not
    primary objective. Same min_time gate + z-score normalization as thermo.
    """

    _PREFIX = "rdkit"

    def __init__(self, min_time=0.8,
                 weight=0.02,
                 target_fields=None,
                 target_mean=None,
                 target_std=None,
                 timesteps=25):
        super().__init__()
        self.min_time = min_time
        self.weight = float(weight)
        self.target_fields = list(target_fields) if target_fields else list(RDKIT_FIELDS_DEFAULT)
        if target_mean is None or target_std is None:
            raise ValueError(
                "RDKitDescriptorLoss requires target_mean and target_std. "
                "Compute via data_processing/compute_rdkit_stats.py."
            )
        self.target_mean = list(target_mean)
        self.target_std = list(target_std)
        self.timesteps = timesteps
        self.register_buffer(
            "_mean_t",
            torch.tensor(self.target_mean, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_std_t",
            torch.tensor(self.target_std, dtype=torch.float32),
            persistent=False,
        )
        self.last_batch_size = 0
        self.last_gated_in = 0
        self.train_metrics = _build_per_target_metric_set(self.target_fields)
        self.val_metrics   = _build_per_target_metric_set(self.target_fields)

    def forward(self, batch, out, time, ws_t, stage="train"):
        if "rdkit_mp" not in out or self.weight <= 0:
            return torch.tensor(0.0, device=time.device)

        device = time.device

        t_frac = _t_frac(time, self.timesteps)
        time_mask = t_frac >= self.min_time
        self.last_batch_size = int(time_mask.numel())
        self.last_gated_in = int(time_mask.sum().item())
        if not time_mask.any():
            return torch.tensor(0.0, device=device)

        targets = torch.stack(
            [batch[f].view(-1).float() for f in self.target_fields], dim=1
        )  # [B, 9]
        targets_norm = (targets - self._mean_t) / self._std_t

        pred = out["rdkit_mp"]
        valid = time_mask.unsqueeze(-1) & ~torch.isnan(targets_norm)
        if not valid.any():
            return torch.tensor(0.0, device=device)
        diff = (pred - torch.nan_to_num(targets_norm)) * valid

        with torch.no_grad():
            pred_phys   = pred * self._std_t + self._mean_t
            target_phys = torch.nan_to_num(targets)
            metrics = self.train_metrics if stage == "train" else self.val_metrics
            for i, f in enumerate(self.target_fields):
                col_valid = valid[:, i]
                n_v = int(col_valid.sum().item())
                if n_v == 0:
                    continue
                p = pred_phys[col_valid, i]
                t = target_phys[col_valid, i]
                metrics[f]["mae"].update(p, t)
                metrics[f]["rmse"].update(p, t)
                if n_v >= 2:
                    metrics[f]["r2"].update(p, t)
                metrics[f]["pred_mean"].update(p)
                metrics[f]["pred_msq"].update(p ** 2)
                metrics[f]["target_mean"].update(t)
                metrics[f]["target_msq"].update(t ** 2)

        return self.weight * (diff ** 2).sum() / valid.sum()

    def compute_and_reset(self, stage):
        metrics = self.train_metrics if stage == "train" else self.val_metrics
        return _compute_and_reset_metric_set(metrics, self._PREFIX)

    def get_metrics_dict(self):
        return {
            f"{self._PREFIX}/batch_size": float(self.last_batch_size),
            f"{self._PREFIX}/gated_in":   float(self.last_gated_in),
        }


class EnergyPredictionLoss(_NNModuleSetstateMixin, _nn.Module):
    """Auxiliary per-molecule energy loss (Phase 1).

    Expects `out["energy_pred"]` (build MegaFNV3Conf with energy_head=True)
    and `batch.energy` attached by scripts/label_energy.py. Optional
    per-atom normalization to decouple from molecule size.

    Args:
        min_time: only apply loss when t >= min_time (fraction in [0,1]).
        weight:   scalar weight relative to main denoising loss.
        normalize: "per_atom" | "zscore" | "none".
        timesteps: for discrete→fraction conversion.
        target_mean / target_std: required when normalize == "zscore".
    """

    def __init__(self, min_time=0.8, weight=0.1, normalize="per_atom",
                 timesteps=25, target_mean=None, target_std=None):
        super().__init__()
        assert normalize in ("per_atom", "zscore", "none")
        self.min_time = min_time
        self.weight = weight
        self.normalize = normalize
        self.timesteps = timesteps
        if normalize == "zscore" and (target_mean is None or target_std is None):
            raise ValueError("zscore normalization requires target_mean and target_std")
        self.target_mean = target_mean
        self.target_std = target_std

    def forward(self, batch, out, time, ws_t, stage="train"):
        if "energy_pred" not in out:
            return torch.tensor(0.0, device=time.device)
        if not hasattr(batch, "energy"):
            return torch.tensor(0.0, device=time.device)

        t_frac = _t_frac(time, self.timesteps)
        mask = t_frac >= self.min_time
        if not mask.any():
            return torch.tensor(0.0, device=time.device)

        pred = out["energy_pred"][mask]
        target = batch.energy.view(-1)[mask].float()
        valid = ~torch.isnan(target)
        if not valid.any():
            return torch.tensor(0.0, device=time.device)
        pred, target = pred[valid], target[valid]

        if self.normalize == "per_atom":
            counts = torch.bincount(batch.batch, minlength=pred.size(0))
            # counts is over all mols; we need the active subset
            n_atoms = counts[mask][valid].float().clamp(min=1)
            pred = pred / n_atoms
            target = target / n_atoms
        elif self.normalize == "zscore":
            m = torch.tensor(self.target_mean, device=pred.device, dtype=pred.dtype)
            s = torch.tensor(self.target_std,  device=pred.device, dtype=pred.dtype)
            pred   = (pred   - m) / s
            target = (target - m) / s

        return self.weight * torch.nn.functional.mse_loss(pred, target)


class CombinedAuxiliaryLoss(_NNModuleSetstateMixin, _nn.Module):
    """Wrap several auxiliary losses; delegate per-step telemetry and
    epoch-end metric computation to each sub-loss. Prefixes are baked
    into the sub-losses themselves (thermo/, rdkit/, energy/)."""

    def __init__(self, thermo_loss=None, rdkit_loss=None, energy_loss=None):
        super().__init__()
        # nn.Module.__setattr__ auto-registers nn.Module children, so
        # .to(device), state_dict(), and DDP all just work.
        self.thermo_loss = thermo_loss
        self.rdkit_loss = rdkit_loss
        self.energy_loss = energy_loss

    def forward(self, batch, out, time, ws_t, stage="train"):
        loss = torch.tensor(0.0, device=time.device)
        for sub in (self.thermo_loss, self.rdkit_loss, self.energy_loss):
            if sub is not None:
                loss = loss + sub(batch, out, time, ws_t, stage)
        return loss

    def get_metrics_dict(self):
        out = {}
        for sub in (self.thermo_loss, self.rdkit_loss, self.energy_loss):
            if sub is not None and hasattr(sub, "get_metrics_dict"):
                out.update(sub.get_metrics_dict())
        return out

    def compute_and_reset(self, stage):
        out = {}
        for sub in (self.thermo_loss, self.rdkit_loss, self.energy_loss):
            if sub is not None and hasattr(sub, "compute_and_reset"):
                out.update(sub.compute_and_reset(stage))
        return out
