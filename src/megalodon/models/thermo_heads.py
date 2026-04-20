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
    node with MULTI-HEAD attention-based atom → mol pooling.

    Each layer:
        q = W_q · mol_H   reshape → [N_mols,  n_heads, d_head]
        k = W_k · H       reshape → [N_atoms, n_heads, d_head]
        v = W_v · H       reshape → [N_atoms, n_heads, d_head]
        alpha_{i,h} = softmax_per_mol(q_{b(i),h} · k_{i,h} / sqrt(d_head))
        agg = concat_h( sum_i alpha_{i,h} · v_{i,h} )   [N_mols, dim]
        mol_H ← mol_H + MLP1( W_o · agg )
        H     ← H     + MLP2( [H | mol_H[b(i)]] )
    """

    def __init__(self, dim=256, n_layers=2, n_heads=4, hidden=128, n_targets=5):
        super().__init__()
        assert dim % n_heads == 0, \
            f"dim ({dim}) must be divisible by n_heads ({n_heads})"
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_layers))
        self.k_proj = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_layers))
        self.v_proj = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_layers))
        self.o_proj = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_layers))
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
        mol_H = scatter_mean(H, batch, dim=0)              # [N_mols, dim]
        N_mols = mol_H.size(0)
        N_atoms = H.size(0)
        for l in range(self.n_layers):
            q = self.q_proj[l](mol_H).view(N_mols,  self.n_heads, self.head_dim)
            k = self.k_proj[l](H    ).view(N_atoms, self.n_heads, self.head_dim)
            v = self.v_proj[l](H    ).view(N_atoms, self.n_heads, self.head_dim)
            q_at = q[batch]                                 # [N_atoms, H, d_h]
            scores = (q_at * k).sum(-1) * self.scale        # [N_atoms, H]
            alpha = scatter_softmax(scores, batch, dim=0)   # softmax per mol per head
            weighted = alpha.unsqueeze(-1) * v              # [N_atoms, H, d_h]
            agg = scatter_sum(weighted, batch, dim=0)       # [N_mols, H, d_h]
            agg = agg.reshape(N_mols, self.dim)             # concat heads
            agg = self.o_proj[l](agg)                       # learnable head-mix
            mol_H = mol_H + self.mol_update[l](agg)

            mol_at = mol_H[batch]
            H = H + self.atom_update[l](torch.cat([H, mol_at], dim=-1))
        return self.final(mol_H)


class ThermoHeadModel(nn.Module):
    """Container for both heads. Output is a dict with 'ext' and 'mp' keys."""

    def __init__(self, dim=256, n_mp_layers=2, n_mp_heads=4, hidden=128):
        super().__init__()
        self.ext = ExtensiveSumHead(dim=dim, hidden=hidden,
                                     n_targets=len(EXTENSIVE_IDX))
        self.mp = AtomMolMP(dim=dim, n_layers=n_mp_layers, n_heads=n_mp_heads,
                             hidden=hidden, n_targets=len(TARGET_FIELDS))

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


def apply_thermo_config_yaml(parser, yaml_path):
    """Override argparse defaults with values from a thermo config YAML.

    YAML may be nested or flat; leaf keys must match argparse `dest`
    names (underscores, no dashes). Explicit CLI flags still override
    the YAML values because argparse applies them after the defaults.
    """
    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True)
    flat = {}

    def _flatten(d):
        for k, v in d.items():
            if isinstance(v, dict):
                _flatten(v)
            else:
                flat[k] = v

    _flatten(cfg)
    for action in parser._actions:
        if action.dest in flat:
            action.default = flat[action.dest]
    return flat
