"""Thermodynamic prediction head on top of the LoQI per-atom representation H.

AtomMolMP — bidirectional message passing between atoms and a per-molecule
virtual node with multi-head attention-based pooling. Handles both additive
and non-additive targets through learned aggregation.

We previously paired this with an `ExtensiveSumHead` (per-atom MLP +
scatter_sum) that was physically principled for additive quantities (Hf,
Gf, Cv, Hf_0). In extensive benchmarking MP alone matches or beats Ext on
every target, so the Ext branch has been removed. The single-head design
simplifies loss computation, YAML configuration, and downstream reporting.
"""
import math

import torch
import torch.nn as nn


# ── Native scatter helpers (Dynamo / torch.compile compatible) ────────────
# torch_scatter is an external CUDA extension whose custom ops can't be
# traced by Dynamo with FakeTensors — running torch.compile() against
# AtomMolMP errors out at the first scatter_max call. The helpers below
# use only PyTorch built-ins (scatter_reduce_, index_add_), so they
# compose cleanly with Inductor.
#
# Numerical equivalence to torch_scatter:
#   * scatter_mean: include_self=False so the zero-init isn't averaged in
#   * scatter_softmax: max-subtraction for numerical stability, same as
#     torch_scatter.composite.softmax
#   * scatter_sum: index_add_ is exact equivalent (1-D index broadcasts
#     naturally across trailing dims)

def _expand_index_like(index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """[N] long index → matches src's shape by inserting singleton trailing
    dims and expanding. Required because scatter_reduce_ needs index.ndim
    == src.ndim."""
    if src.dim() == 1:
        return index
    view_shape = [index.size(0)] + [1] * (src.dim() - 1)
    return index.view(view_shape).expand_as(src)


def _scatter_mean(src: torch.Tensor, index: torch.Tensor,
                   dim_size: int) -> torch.Tensor:
    """Group-mean of `src` rows by `index` into `dim_size` buckets."""
    out_shape = [dim_size] + list(src.shape[1:])
    out = torch.zeros(out_shape, device=src.device, dtype=src.dtype)
    out.scatter_reduce_(0, _expand_index_like(index, src), src,
                         reduce="mean", include_self=False)
    return out


def _scatter_sum(src: torch.Tensor, index: torch.Tensor,
                  dim_size: int) -> torch.Tensor:
    """Group-sum of `src` rows by `index` into `dim_size` buckets."""
    out_shape = [dim_size] + list(src.shape[1:])
    out = torch.zeros(out_shape, device=src.device, dtype=src.dtype)
    # index_add_ accepts a 1-D index and broadcasts over trailing dims,
    # so it's a drop-in replacement for scatter_sum(src, index, dim=0).
    out.index_add_(0, index, src)
    return out


def _scatter_softmax(src: torch.Tensor, index: torch.Tensor,
                      dim_size: int) -> torch.Tensor:
    """Per-group softmax: for each unique value g in `index`, computes
    softmax over the slice src[index==g]. Returns a tensor of same shape
    as src."""
    # Per-group max for numerical stability.
    init = torch.full(([dim_size] + list(src.shape[1:])),
                       float("-inf"), device=src.device, dtype=src.dtype)
    init.scatter_reduce_(0, _expand_index_like(index, src), src,
                          reduce="amax", include_self=False)
    # Subtract per-group max and exponentiate.
    exp_src = (src - init[index]).exp()
    # Per-group sum of exps.
    denom_shape = [dim_size] + list(src.shape[1:])
    denom = torch.zeros(denom_shape, device=src.device, dtype=src.dtype)
    denom.index_add_(0, index, exp_src)
    return exp_src / denom[index].clamp(min=1e-12)


TARGET_FIELDS = ["enthalpy_298", "gibbs_298", "cv_gas", "entropy_gas", "enthalpy_0"]
TARGET_UNITS = {
    "enthalpy_298": "kJ/mol", "gibbs_298": "kJ/mol",
    "cv_gas": "J/(mol*K)",    "entropy_gas": "J/(mol*K)",
    "enthalpy_0": "kJ/mol",
}

# 9 RDKit 2D descriptors — 100% coverage on the property table.
RDKIT_TARGET_FIELDS = [
    "logp", "tpsa", "n_h_donors", "n_h_acceptors", "n_rot_bonds",
    "frac_csp3", "n_aliph_rings", "qed", "labute_asa",
]


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
        # N_mols is inferred from batch — costs one CPU sync (.item()) per
        # forward, which is fine outside the inner loop. Callers that
        # already know N_mols can avoid this by passing it through a thin
        # wrapper, but the current call sites don't need that.
        N_mols = int(batch.max().item()) + 1
        N_atoms = H.size(0)
        mol_H = _scatter_mean(H, batch, N_mols)            # [N_mols, dim]
        for l in range(self.n_layers):
            q = self.q_proj[l](mol_H).view(N_mols,  self.n_heads, self.head_dim)
            k = self.k_proj[l](H    ).view(N_atoms, self.n_heads, self.head_dim)
            v = self.v_proj[l](H    ).view(N_atoms, self.n_heads, self.head_dim)
            q_at = q[batch]
            scores = (q_at * k).sum(-1) * self.scale       # [N_atoms, n_heads]
            alpha = _scatter_softmax(scores, batch, N_mols)
            weighted = alpha.unsqueeze(-1) * v             # [N_atoms, n_heads, head_dim]
            agg = _scatter_sum(weighted, batch, N_mols)    # [N_mols, n_heads, head_dim]
            agg = agg.reshape(N_mols, self.dim)
            agg = self.o_proj[l](agg)
            mol_H = mol_H + self.mol_update[l](agg)

            mol_at = mol_H[batch]
            H = H + self.atom_update[l](torch.cat([H, mol_at], dim=-1))
        return self.final(mol_H)


class ThermoHeadModel(nn.Module):
    """Thin wrapper around AtomMolMP. Returns a dict {'mp': [N_mols, K]} so
    callers can extend with additional heads in the future without changing
    the interface (e.g., per-atom charge, intensive-only head, etc.)."""

    def __init__(self, dim=256, n_mp_layers=2, n_mp_heads=4, hidden=128):
        super().__init__()
        self.mp = AtomMolMP(dim=dim, n_layers=n_mp_layers, n_heads=n_mp_heads,
                             hidden=hidden, n_targets=len(TARGET_FIELDS))

    def forward(self, H, batch):
        return {"mp": self.mp(H, batch)}


class RDKitHeadModel(nn.Module):
    """Parallel head for predicting the 9 RDKit 2D descriptors.

    Structurally identical to ThermoHeadModel but with 9 outputs and
    typically smaller capacity — RDKit descriptors are cheaper targets
    so a lighter head is usually enough.
    """

    def __init__(self, dim=256, n_mp_layers=1, n_mp_heads=4, hidden=128):
        super().__init__()
        self.mp = AtomMolMP(dim=dim, n_layers=n_mp_layers, n_heads=n_mp_heads,
                             hidden=hidden, n_targets=len(RDKIT_TARGET_FIELDS))

    def forward(self, H, batch):
        return {"mp": self.mp(H, batch)}


# 14 targets total when combined: 5 thermo + 9 RDKit, in this order.
COMBINED_TARGET_FIELDS = TARGET_FIELDS + RDKIT_TARGET_FIELDS


class CombinedHeadModel(nn.Module):
    """Single AtomMolMP-based head producing all 14 targets (5 thermo + 9
    RDKit) at once. Replaces the separate ThermoHeadModel + RDKitHeadModel
    when `combined_head_args` is set in the dynamics config.

    Sharing one atom-mol attention pool forces the per-mol representation
    to encode features useful for **both** task families simultaneously,
    which acts as a multi-task regularizer on H. Saves ~20% head params
    over separate heads (one bigger AtomMolMP vs two smaller ones).

    Output ordering on dim=1: indices [0:5] = thermo targets in
    `TARGET_FIELDS` order; indices [5:14] = RDKit descriptors in
    `RDKIT_TARGET_FIELDS` order.
    """

    def __init__(self, dim=256, n_mp_layers=4, n_mp_heads=4, hidden=256):
        super().__init__()
        self.mp = AtomMolMP(
            dim=dim, n_layers=n_mp_layers, n_heads=n_mp_heads,
            hidden=hidden, n_targets=len(COMBINED_TARGET_FIELDS),
        )

    def forward(self, H, batch):
        return {"mp": self.mp(H, batch)}


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
