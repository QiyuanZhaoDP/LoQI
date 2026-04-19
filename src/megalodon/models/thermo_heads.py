"""Thermodynamic prediction heads on top of the LoQI per-atom representation H.

Two heads, designed for different physics:

  ExtensiveSumHead  — per-atom MLP(H) + scatter_sum. Matches the physical
                      additivity of Hf, Gf, Cv, Hf_0 (each scales with the
                      molecule's atomic composition).

  AtomMolMP         — bidirectional message passing between atoms and a
                      per-molecule virtual node with attention-based pooling.
                      Learns arbitrary (non-additive) aggregation, needed for
                      intensive / shape-dependent properties (S0).

Both run on the same frozen (or partially-unfrozen) H. They are trained
jointly; at eval time you pick the head that wins per target.
"""
import math

import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_softmax, scatter_sum


TARGET_FIELDS = ["enthalpy_298", "gibbs_298", "cv_gas", "entropy_gas", "enthalpy_0"]
# Indices into TARGET_FIELDS whose physics is extensive (additive over atoms):
EXTENSIVE_IDX = [0, 1, 2, 4]
TARGET_UNITS = {
    "enthalpy_298": "kJ/mol", "gibbs_298": "kJ/mol",
    "cv_gas": "J/(mol*K)",    "entropy_gas": "J/(mol*K)",
    "enthalpy_0": "kJ/mol",
}


class ExtensiveSumHead(nn.Module):
    """per-atom MLP → scatter_sum → extensive property."""

    def __init__(self, dim=256, hidden=128, n_targets=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_targets),
        )

    def forward(self, H, batch):
        per_atom = self.mlp(H)
        return scatter_sum(per_atom, batch, dim=0)


class AtomMolMP(nn.Module):
    """Bidirectional message passing between atoms and a per-molecule virtual
    node with attention-based atom → mol pooling.

    Each layer:
        q = W_q * mol_H,  k = W_k * H,  v = W_v * H
        alpha_i = softmax_per_mol(q_{b(i)} . k_i / sqrt(d))
        mol_H  ← mol_H + MLP1( sum_i alpha_i * v_i )
        H      ← H     + MLP2( [H | mol_H[b(i)]] )
    """

    def __init__(self, dim=256, n_layers=2, hidden=128, n_targets=5):
        super().__init__()
        self.n_layers = n_layers
        self.dim = dim
        self.q_proj = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_layers))
        self.k_proj = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_layers))
        self.v_proj = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_layers))
        self.mol_update = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim),
                          nn.SiLU(), nn.Linear(dim, dim))
            for _ in range(n_layers)
        )
        self.atom_update = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim),
                          nn.SiLU(), nn.Linear(dim, dim))
            for _ in range(n_layers)
        )
        self.final = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_targets),
        )

    def forward(self, H, batch):
        mol_H = scatter_mean(H, batch, dim=0)
        scale = 1.0 / math.sqrt(self.dim)
        for l in range(self.n_layers):
            q = self.q_proj[l](mol_H)
            k = self.k_proj[l](H)
            v = self.v_proj[l](H)
            q_at = q[batch]
            scores = (q_at * k).sum(-1) * scale
            alpha = scatter_softmax(scores, batch, dim=0)
            weighted = alpha.unsqueeze(-1) * v
            agg = scatter_sum(weighted, batch, dim=0)
            mol_H = mol_H + self.mol_update[l](agg)

            mol_at = mol_H[batch]
            H = H + self.atom_update[l](torch.cat([H, mol_at], dim=-1))
        return self.final(mol_H)


class ThermoHeadModel(nn.Module):
    """Container for both heads. Output is a dict with 'ext' and 'mp' keys."""

    def __init__(self, dim=256, n_mp_layers=2):
        super().__init__()
        self.ext = ExtensiveSumHead(dim=dim, n_targets=len(EXTENSIVE_IDX))
        self.mp = AtomMolMP(dim=dim, n_layers=n_mp_layers,
                             n_targets=len(TARGET_FIELDS))

    def forward(self, H, batch):
        return {"ext": self.ext(H, batch), "mp": self.mp(H, batch)}


def masked_mse(pred, target):
    """Mean-squared error that ignores NaN entries in target.

    pred, target: [B, K]; target may have NaNs (missing labels).
    Returns a scalar averaged over valid entries.
    """
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    diff = (pred - torch.nan_to_num(target)) * mask
    return (diff ** 2).sum() / mask.sum()
