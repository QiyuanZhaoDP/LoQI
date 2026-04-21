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
EXTENSIVE_IDX_DEFAULT = [0, 1, 2, 4]


def _t_frac(time, timesteps):
    """Normalize time to [0, 1] fraction regardless of discrete/continuous."""
    if time.is_floating_point():
        return time
    return time.float() / max(timesteps - 1, 1)


class ThermoPropertyLoss:
    """Auxiliary thermo-prediction loss applied at late denoising timesteps.

    Reads per-molecule thermo targets attached by scripts/label_thermo.py:
    enthalpy_298, gibbs_298, cv_gas, entropy_gas, enthalpy_0 (+ a
    `thermo_has_label` bool). Semi-supervised: unlabeled molecules
    (NaN targets or `thermo_has_label == False`) contribute 0 to the loss.

    Requires the backbone to expose `out["thermo_ext"]` and/or
    `out["thermo_mp"]` — obtained by building MegaFNV3Conf with
    `thermo_head_args` set.

    Args:
        min_time: only apply loss when t >= min_time (as fraction in [0,1]).
                  Rationale: at early t, H encodes "direction of denoising"
                  rather than molecule identity — thermo loss would be noise.
        weights:  {"ext": float, "mp": float} — relative to main denoising loss.
        target_mean / target_std: per-target z-score stats (list of len 5);
                  typically precomputed from the training split.
        timesteps: needed to normalize discrete-time to fraction [0, 1].
    """

    def __init__(self, min_time=0.8,
                 weights=None,
                 target_fields=None,
                 extensive_idx=None,
                 target_mean=None,
                 target_std=None,
                 timesteps=25):
        self.min_time = min_time
        self.weights = dict(weights) if weights else {"ext": 0.05, "mp": 0.05}
        self.target_fields = list(target_fields) if target_fields else list(TARGET_FIELDS_DEFAULT)
        self.extensive_idx = list(extensive_idx) if extensive_idx else list(EXTENSIVE_IDX_DEFAULT)
        if target_mean is None or target_std is None:
            raise ValueError(
                "ThermoPropertyLoss requires target_mean and target_std "
                "(per-target z-score stats; use the values printed by "
                "scripts/label_thermo.py or finetune_thermo_head.py)."
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

    def _ensure_stats(self, device, dtype):
        if self._mean_t is None or self._mean_t.device != device:
            self._mean_t = torch.tensor(self.target_mean, device=device, dtype=dtype)
            self._std_t  = torch.tensor(self.target_std,  device=device, dtype=dtype)

    def __call__(self, batch, out, time, ws_t, stage="train"):
        if "thermo_ext" not in out and "thermo_mp" not in out:
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

        total = torch.tensor(0.0, device=device)

        def _masked_mse(pred, tgt, row_mask):
            # Per-element valid = row_mask AND target not NaN.
            valid = row_mask.unsqueeze(-1) & ~torch.isnan(tgt)
            if not valid.any():
                return torch.tensor(0.0, device=pred.device)
            diff = (pred - torch.nan_to_num(tgt)) * valid
            return (diff ** 2).sum() / valid.sum()

        if "thermo_ext" in out and self.weights.get("ext", 0) > 0:
            pred = out["thermo_ext"]
            tgt = targets_norm[:, self.extensive_idx]
            total = total + self.weights["ext"] * _masked_mse(pred, tgt, mol_mask)

        if "thermo_mp" in out and self.weights.get("mp", 0) > 0:
            pred = out["thermo_mp"]
            total = total + self.weights["mp"] * _masked_mse(pred, targets_norm, mol_mask)

        return total


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
    """Optional wrapper to run ThermoPropertyLoss + EnergyPredictionLoss together."""

    def __init__(self, thermo_loss=None, energy_loss=None):
        self.thermo_loss = thermo_loss
        self.energy_loss = energy_loss

    def __call__(self, batch, out, time, ws_t, stage="train"):
        loss = torch.tensor(0.0, device=time.device)
        if self.thermo_loss is not None:
            loss = loss + self.thermo_loss(batch, out, time, ws_t, stage)
        if self.energy_loss is not None:
            loss = loss + self.energy_loss(batch, out, time, ws_t, stage)
        return loss
