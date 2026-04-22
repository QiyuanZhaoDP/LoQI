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


class ThermoPropertyLoss:
    """Auxiliary thermo-prediction loss applied at late denoising timesteps.

    Reads per-molecule thermo targets (enthalpy_298, gibbs_298, cv_gas,
    entropy_gas, enthalpy_0) attached to the batch by the data pipeline's
    AttachProperties transform, plus a `thermo_has_label` bool flag.
    Semi-supervised: unlabeled molecules (NaN targets or
    `thermo_has_label == False`) contribute 0 to the loss.

    Requires the backbone to expose `out["thermo_mp"]` — obtained by
    building MegaFNV3Conf with `thermo_head_args` set.

    Args:
        min_time: only apply loss when t >= min_time (as fraction in [0,1]).
                  Rationale: at early t, H encodes "direction of denoising"
                  rather than molecule identity — thermo loss would be noise.
        weight:   scalar weight for the MP head relative to main denoising loss.
        target_mean / target_std: per-target z-score stats (list of len 5);
                  typically precomputed from the training split.
        timesteps: needed to normalize discrete-time to fraction [0, 1].
    """

    def __init__(self, min_time=0.8,
                 weight=0.05,
                 target_fields=None,
                 target_mean=None,
                 target_std=None,
                 timesteps=25):
        self.min_time = min_time
        self.weight = float(weight)
        self.target_fields = list(target_fields) if target_fields else list(TARGET_FIELDS_DEFAULT)
        if target_mean is None or target_std is None:
            raise ValueError(
                "ThermoPropertyLoss requires target_mean and target_std "
                "(per-target z-score stats; use the values printed by "
                "scripts/finetune_thermo_head.py or build_property_table.py)."
            )
        self.target_mean = list(target_mean)
        self.target_std = list(target_std)
        self.timesteps = timesteps
        self._mean_t = None
        self._std_t = None
        # Counters for label-density telemetry.
        self.last_batch_size = 0
        self.last_gated_in = 0           # molecules with t >= min_time
        self.last_labeled_active = 0     # molecules contributing to loss
        # Per-target MAE / RMSE in physical units, over the active subset
        # (labeled + past min_time). Lightning logs these as epoch means.
        self.last_per_target_mae = {f: 0.0 for f in self.target_fields}
        self.last_per_target_rmse = {f: 0.0 for f in self.target_fields}

    def _ensure_stats(self, device, dtype):
        if self._mean_t is None or self._mean_t.device != device:
            self._mean_t = torch.tensor(self.target_mean, device=device, dtype=dtype)
            self._std_t  = torch.tensor(self.target_std,  device=device, dtype=dtype)

    def __call__(self, batch, out, time, ws_t, stage="train"):
        if "thermo_mp" not in out or self.weight <= 0:
            return torch.tensor(0.0, device=time.device)

        device = time.device
        self._ensure_stats(device, torch.float32)

        # Time gate — only compute loss when coords are nearly clean.
        t_frac = _t_frac(time, self.timesteps)
        time_mask = t_frac >= self.min_time                          # [B]

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

        targets_norm = (targets - self._mean_t) / self._std_t        # [B, 5]

        # Masked MSE: ignore rows that are out-of-time and any target entry
        # that is NaN (e.g., partial TCIT coverage).
        pred = out["thermo_mp"]
        valid = mol_mask.unsqueeze(-1) & ~torch.isnan(targets_norm)
        if not valid.any():
            return torch.tensor(0.0, device=device)
        diff = (pred - torch.nan_to_num(targets_norm)) * valid

        # Per-target MAE / RMSE in physical units — for wandb telemetry.
        # De-normalize BOTH sides, mask, and average per target.
        with torch.no_grad():
            pred_phys   = pred * self._std_t + self._mean_t
            target_phys = torch.nan_to_num(targets)
            err_phys    = (pred_phys - target_phys) * valid            # [B, 5]
            cnt         = valid.sum(dim=0).clamp(min=1).float()        # [5]
            mae         = err_phys.abs().sum(dim=0) / cnt              # [5]
            rmse        = (err_phys ** 2).sum(dim=0).sqrt() / cnt.sqrt()
            for i, f in enumerate(self.target_fields):
                self.last_per_target_mae[f]  = float(mae[i].item())
                self.last_per_target_rmse[f] = float(rmse[i].item())

        return self.weight * (diff ** 2).sum() / valid.sum()

    def get_metrics_dict(self):
        """Flat name → scalar dict for the Lightning module to log. Keys
        are namespaced with `thermo/` so multiple aux heads can coexist."""
        d = {
            "thermo/batch_size":     float(self.last_batch_size),
            "thermo/gated_in":       float(self.last_gated_in),
            "thermo/labeled_active": float(self.last_labeled_active),
        }
        for f in self.target_fields:
            d[f"thermo/mae_{f}"]  = self.last_per_target_mae[f]
            d[f"thermo/rmse_{f}"] = self.last_per_target_rmse[f]
        return d


class RDKitDescriptorLoss:
    """Auxiliary RDKit-descriptor prediction loss.

    Mirrors ThermoPropertyLoss but for the 9 RDKit 2D descriptors
    (LogP, TPSA, Lipinski counts, FracCSP3, NumAliphaticRings, QED,
    LabuteASA). Labels are 100% covered after build_property_table.py
    drops all-NaN rows, so there's no `has_*` flag to check — only the
    time gate and per-target NaN mask (vestigial; should never fire).

    Typical usage: small weight (~0.02) and same min_time as thermo,
    so the RDKit signal acts as a cheap regularizer on the global
    representation without dominating the thermo head's gradient.

    Requires the backbone to expose `out["rdkit_mp"]` — obtained by
    building MegaFNV3Conf with `rdkit_head_args` set.
    """

    def __init__(self, min_time=0.8,
                 weight=0.02,
                 target_fields=None,
                 target_mean=None,
                 target_std=None,
                 timesteps=25):
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
        self._mean_t = None
        self._std_t = None
        self.last_batch_size = 0
        self.last_gated_in = 0
        self.last_per_target_mae = {f: 0.0 for f in self.target_fields}
        self.last_per_target_rmse = {f: 0.0 for f in self.target_fields}

    def _ensure_stats(self, device, dtype):
        if self._mean_t is None or self._mean_t.device != device:
            self._mean_t = torch.tensor(self.target_mean, device=device, dtype=dtype)
            self._std_t  = torch.tensor(self.target_std,  device=device, dtype=dtype)

    def __call__(self, batch, out, time, ws_t, stage="train"):
        if "rdkit_mp" not in out or self.weight <= 0:
            return torch.tensor(0.0, device=time.device)

        device = time.device
        self._ensure_stats(device, torch.float32)

        t_frac = _t_frac(time, self.timesteps)
        time_mask = t_frac >= self.min_time                            # [B]
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
            err_phys    = (pred_phys - target_phys) * valid
            cnt         = valid.sum(dim=0).clamp(min=1).float()
            mae         = err_phys.abs().sum(dim=0) / cnt
            rmse        = (err_phys ** 2).sum(dim=0).sqrt() / cnt.sqrt()
            for i, f in enumerate(self.target_fields):
                self.last_per_target_mae[f]  = float(mae[i].item())
                self.last_per_target_rmse[f] = float(rmse[i].item())

        return self.weight * (diff ** 2).sum() / valid.sum()

    def get_metrics_dict(self):
        d = {
            "rdkit/batch_size": float(self.last_batch_size),
            "rdkit/gated_in":   float(self.last_gated_in),
        }
        for f in self.target_fields:
            d[f"rdkit/mae_{f}"]  = self.last_per_target_mae[f]
            d[f"rdkit/rmse_{f}"] = self.last_per_target_rmse[f]
        return d


class EnergyPredictionLoss:
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
        assert normalize in ("per_atom", "zscore", "none")
        self.min_time = min_time
        self.weight = weight
        self.normalize = normalize
        self.timesteps = timesteps
        if normalize == "zscore" and (target_mean is None or target_std is None):
            raise ValueError("zscore normalization requires target_mean and target_std")
        self.target_mean = target_mean
        self.target_std = target_std

    def __call__(self, batch, out, time, ws_t, stage="train"):
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


class CombinedAuxiliaryLoss:
    """Wrap several auxiliary losses and route metric aggregation.

    Each sub-loss can expose a get_metrics_dict(); this class prefixes
    their keys (thermo/ · rdkit/ · energy/) and unions them so the
    Lightning module can log per-target MAE/RMSE for every head it runs.
    """

    def __init__(self, thermo_loss=None, rdkit_loss=None, energy_loss=None):
        self.thermo_loss = thermo_loss
        self.rdkit_loss = rdkit_loss
        self.energy_loss = energy_loss

    def __call__(self, batch, out, time, ws_t, stage="train"):
        loss = torch.tensor(0.0, device=time.device)
        if self.thermo_loss is not None:
            loss = loss + self.thermo_loss(batch, out, time, ws_t, stage)
        if self.rdkit_loss is not None:
            loss = loss + self.rdkit_loss(batch, out, time, ws_t, stage)
        if self.energy_loss is not None:
            loss = loss + self.energy_loss(batch, out, time, ws_t, stage)
        return loss

    def get_metrics_dict(self):
        out = {}
        for loss in (self.thermo_loss, self.rdkit_loss, self.energy_loss):
            if loss is not None and hasattr(loss, "get_metrics_dict"):
                out.update(loss.get_metrics_dict())
        return out
