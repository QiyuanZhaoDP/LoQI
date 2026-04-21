# ThermoGen-FM: LoQI Multi-Task Foundation Model Implementation Plan

## Core Thesis

LoQI 的构象生成训练中，2D 分子图（原子类型、键类型、电荷）作为**固定条件**输入，模型唯一的任务是从噪声坐标去噪到 AIMNet2 优化后的低能量 3D 构象。这迫使模型从 2D 图中提取深层化学知识。

**关键物理联系**：扩散/流匹配模型学习的 score function `∇_x log p(x)` 等价于 `-(1/k_BT)∇_x E(x)`，即原子力。一个好的 3D 生成模型隐式地学习了量子势能面 (PES)。

**创新点**：在去噪训练中联合预测热力学性质（Energy, H, G, Cᵥ），迫使表征编码 PES 的**曲率信息**（Hessian / 振动模式），而非仅编码 PES 的位置（坐标）。这产生的表征天然适合下游性质预测。

---

## Architecture Overview

```
                    ┌─────────────────────────────────┐
                    │     MegaFNV3Conf Backbone        │
                    │  (10x DiTeBlock + XEGNN layers)  │
                    │                                   │
  2D Graph ────────►│  H_t (atom types, fixed)         │
  (fixed cond.)     │  E_t (bond types, fixed)         │
                    │  charges (fixed)                  │
                    │                                   │
  Noisy coords ───►│  X_t (noised coordinates)        │
  + time t          │  t   (diffusion timestep)        │
                    │                                   │
                    └──────┬──────────┬────────────────┘
                           │          │
                    ┌──────▼──┐  ┌────▼─────────┐
                    │ x_hat   │  │ H (atom repr)│ [N_atoms, 256]
                    │ (coords)│  │  (invariant) │
                    └────┬────┘  └──┬───────────┘
                         │         │
                  ┌──────▼──┐      │
                  │ Head 1  │      │
                  │ Denoise │      │
                  │ (exist) │      │
                  └─────────┘      │
                                   │
              ┌────────────────────┼────────────────────┐
              │           scatter_mean(H, batch)        │
              │         mol_repr [N_mols, 256]          │
              └────┬──────────┬──────────┬──────────────┘
                   │          │          │
            ┌──────▼──┐ ┌────▼───┐ ┌────▼─────────┐
            │ Head 2  │ │ Head 3 │ │   Head 4     │
            │ Energy  │ │ Thermo │ │   Forces     │
            │  E(x)   │ │ H,G,Cv │ │ F=-∂E/∂x    │
            │(Phase1) │ │(Phase2)│ │ (autograd)   │
            └─────────┘ └────────┘ └──────────────┘

  Key design:
  - H is SE(3)-invariant → pooling to mol_repr preserves invariance
  - Energy is a scalar (invariant) → predicted from mol_repr ✓
  - Forces are equivariant vectors → obtained via autograd: F = -∂E/∂x ✓
    (physically consistent, guarantees energy conservation)
  - Thermo properties (H, G, Cv) are scalars → from mol_repr ✓
```

---

## Phase 0: Validate Hypothesis (Zero Training Changes)

### Goal
验证 LoQI 预训练表征是否已经对性质预测有用。**零模型改动，纯 probing 实验。**

### Workflow

```
1. 加载 LoQI checkpoint (loqi.ckpt 或 loqi_flow.ckpt)
2. 准备 QM9 数据集（已有 process_qm9.py, 包含 HOMO/LUMO/Gap/U0/H/G/Cv 标签）
3. 对每个 QM9 分子，用 LoQI 跑完整去噪采样（25 steps）
4. 提取最后一步的 out["H"]  → [N_atoms, 256]
5. scatter_mean(H, batch) → [N_mols, 256] 分子级表征
6. 冻结表征，训练简单 MLP/Ridge 回归预测各性质
7. 对比 baseline（SchNet, DimeNet, random features）
```

### Pseudocode

```python
# === scripts/probe_representation.py ===

import torch
from torch_scatter import scatter_mean
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 1. Load model
model = Graph3DInterpolantModel.load_from_checkpoint("data/loqi.ckpt", ...)
model.eval()

# 2. Load QM9 dataset (with property labels)
qm9_dataset = MoleculeDataset(root="qm9_data", processed_folder="processed", split="test")

# 3. Extract representations
all_repr = []
all_labels = []

for batch in DataLoader(qm9_dataset, batch_size=64):
    batch = preprocessor(batch)
    
    # Run full denoising (25 steps) to get final H
    with torch.no_grad():
        # Option A: use sample() and capture H from the last dynamics call
        # Option B: single forward at t=0 (pure noise coords) for speed
        
        # === Option A: Full trajectory (recommended) ===
        samples = model.sample(batch=batch, timesteps=25, pre_format=False)
        H = samples["H"]  # Need to modify sample() to return H
        
        # === Option B: Single forward at t=0 (fast probe) ===
        batch_copy = batch.clone()
        batch_copy["x_t"] = torch.randn_like(batch["x"])  # pure noise
        time = torch.zeros(batch_size)
        out = model.dynamics(batch_copy, time)
        H = out["H"]
    
    mol_repr = scatter_mean(H, batch.batch, dim=0)  # [N_mols, 256]
    all_repr.append(mol_repr.cpu())
    all_labels.append(batch.homo.cpu())  # or any QM9 property

# 4. Linear probe
X = torch.cat(all_repr).numpy()
y = torch.cat(all_labels).numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

reg = Ridge(alpha=1.0)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
```

### Required Model Change (Minimal)

```python
# src/megalodon/models/module.py — sample() method, line ~585
# Add one line to preserve H in the output:

out = self.dynamics(data, time, conditional_batch=out, timesteps=timesteps)
# ... existing code ...

# After sampling loop ends, before return:
samples["H"] = out.get("H", None)  # MegaFNV3Conf already returns H
```

### Expected Outcome
- 如果 linear probe MAE 接近或优于 SchNet → 假设验证成功，值得继续
- 如果远差于 SchNet → 需要重新评估，可能表征缺少 3D 感知（考虑 Option A vs B）

---

## Phase 1: Energy Prediction Head (Multi-Task Training)

### Goal
在 LoQI 去噪训练中加入 AIMNet2 能量预测作为辅助任务，迫使表征编码能量面信息。
同时通过 autograd 得到力 `F = -∂E/∂x`，无需额外的 force head——物理一致且保证能量守恒。

### Workflow

```
Step 1: 数据准备 — 给 ChEMBL3D 每个构象计算 AIMNet2 能量标签
Step 2: 模型改动 — MegaFNV3Conf 加 energy prediction head
Step 3: 损失函数 — 新建 EnergyPredictionLoss (仅在 t > threshold 时施加)
Step 4: 训练配置 — train.py 中启用 loss_fn, yaml 中加参数
Step 5: 训练 & 评估
```

### Step 1: Data Labeling

```python
# === scripts/label_energy.py ===
"""
给每个构象计算 AIMNet2 能量，存入 PyG Data.energy 字段
"""

# Load AIMNet2
aimnet2 = torch.jit.load("src/megalodon/metrics/aimnet2/cpcm_model/wb97m_cpcms_v2_0.jpt")

# Load dataset
dataset = load("data/chembl3d_stereo/processed/train_h.pt")

# Batch compute energies
for batch_mols in batched(dataset, batch_size=128):
    aimnet_input = prepare_for_aimnet(batch_mols)  # 已有函数
    energy = aimnet2(aimnet_input)["energy"]        # [batch_size] in eV
    for mol, e in zip(batch_mols, energy):
        mol.energy = e.float()                      # 存入 Data 对象

# Re-save dataset
save("data/chembl3d_stereo/processed/train_h_energy.pt", dataset)

# --- 运行命令 ---
# python scripts/label_energy.py \
#     --input data/chembl3d_stereo/processed/train_h.pt \
#     --output data/chembl3d_stereo/processed/train_h_energy.pt \
#     --aimnet2_model src/megalodon/metrics/aimnet2/cpcm_model/wb97m_cpcms_v2_0.jpt \
#     --batch_size 128 --device cuda
```

**注意**：能量是绝对能量（eV），分子间差异很大。训练时需要归一化：
- 方案 A：按原子数归一化 `energy / n_atoms`
- 方案 B：减去原子参考能量（推荐，更物理）
- 方案 C：数据集全局 z-score 标准化

### Step 2: Model Modification

```python
# === src/megalodon/dynamics/fn_model.py ===
# MegaFNV3Conf.__init__ 中新增:

class MegaFNV3Conf(nn.Module):
    def __init__(self, ..., energy_head=False):
        super().__init__()
        # ... existing code ...
        
        self.energy_head = None
        if energy_head:
            self.energy_head = nn.Sequential(
                nn.Linear(invariant_node_feat_dim, invariant_node_feat_dim // 2),
                nn.SiLU(),
                nn.Linear(invariant_node_feat_dim // 2, 1)
            )

    def forward(self, batch, X, H, E_idx, E, t):
        # ... existing layers ...
        # After all DiTeBlock + XEGNN layers, H is [N_atoms, 256]
        
        X = self.coord_pred(pos).squeeze(-1)
        x = X - scatter_mean(X, index=batch, dim=0)[batch]

        out = {"x_hat": x, "H": H}
        
        # Energy prediction from atom representations
        if self.energy_head is not None:
            mol_repr = scatter_mean(H, batch, dim=0)  # [N_mols, 256]
            out["energy_pred"] = self.energy_head(mol_repr).squeeze(-1)  # [N_mols]
        
        return out
```

### Step 2b: Force Prediction via Autograd (方案A)

**核心思路**：不加 force head，力通过 `F = -∂E/∂x` 自动微分得到。

```python
# === src/megalodon/models/loss_fn.py (或 module.py training_step 中) ===

def compute_forces_from_energy(energy_pred, coords):
    """
    从标量能量通过 autograd 计算原子力。
    
    物理一致性保证：
    - F = -∂E/∂x → 力场无旋 (conservative)
    - 能量守恒自动满足
    - 力的等变性自动满足（E 是不变标量，对等变坐标求导得到等变向量）
    
    Args:
        energy_pred: [N_mols] 预测的分子能量（标量，不变）
        coords: [N_atoms, 3] 原子坐标（需要 requires_grad=True）
    Returns:
        forces: [N_atoms, 3] 原子力（等变向量）
    """
    forces = -torch.autograd.grad(
        energy_pred.sum(),
        coords,
        create_graph=True,   # 训练时需要二阶梯度
        retain_graph=True,
    )[0]  # [N_atoms, 3]
    return forces


# === 训练时的使用方式 ===
# 在 training_step 或 loss 计算中:

# 关键：输入坐标必须 requires_grad=True
coords = batch.x_t.requires_grad_(True)  # [N_atoms, 3]

out = model.dynamics(batch, time)  # forward pass

if "energy_pred" in out and hasattr(batch, "forces"):
    # 通过 autograd 得到力
    force_pred = compute_forces_from_energy(out["energy_pred"], coords)
    force_target = batch.forces  # AIMNet2 计算的参考力
    
    force_loss = F.mse_loss(force_pred, force_target)
    # 加权加入总 loss
```

**注意事项**：
- `coords.requires_grad_(True)` 必须在 forward 之前设置
- `create_graph=True` 使得 force loss 的梯度能回传到 energy head
- 计算开销：autograd force 比 direct prediction 多约 1x forward 的开销（反向传播）
- 这和已有的 `AIMNet2ForcesLoss` 设计完全一致（参考 `loss_fn.py` 中的 `Forces` class）

### Step 3: Loss Function

```python
# === src/megalodon/models/loss_fn.py ===
# 新增 class:

class EnergyPredictionLoss:
    """
    Auxiliary energy prediction loss, applied only when denoising is near completion.
    
    This forces the model's atom representations to encode energy-relevant information.
    Only active when t > min_time, because at early timesteps the coordinates are
    mostly noise and the representation lacks meaningful 3D information.
    """
    
    def __init__(self, min_time=0.8, weight=0.1, normalize="per_atom"):
        """
        Args:
            min_time: Only apply loss when t > min_time (0.8 = last 20% of denoising)
            weight: Loss weight relative to main denoising loss
            normalize: "per_atom" | "zscore" | "none"
        """
        self.min_time = min_time
        self.weight = weight
        self.normalize = normalize
    
    def __call__(self, batch, out, time, ws_t, stage="train"):
        # Check if energy prediction exists in output
        if "energy_pred" not in out:
            return torch.tensor(0.0, device=time.device)
        
        # Check if energy labels exist in batch
        if not hasattr(batch, "energy") or batch.energy is None:
            return torch.tensor(0.0, device=time.device)
        
        # Only apply at late timesteps
        batch_size = int(batch.batch.max()) + 1
        
        # time is [batch_size], need to determine which molecules qualify
        # For discrete time: time ranges from 0 (noise) to T (clean)
        # For continuous time: time ranges from 0 (noise) to 1 (clean)
        mask = time >= self.min_time
        
        if not mask.any():
            return torch.tensor(0.0, device=time.device)
        
        pred = out["energy_pred"][mask]       # [N_active]
        target = batch.energy[mask]           # [N_active]
        
        # Normalize target
        if self.normalize == "per_atom":
            n_atoms = torch.bincount(batch.batch)[mask].float()
            target = target / n_atoms
            pred = pred / n_atoms  # Predict per-atom energy
        
        loss = F.mse_loss(pred, target)
        
        return self.weight * loss
```

### Step 4: Training Configuration

```python
# === scripts/train.py ===
# 修改 loss_fn 初始化（约 line 46）:

from megalodon.models.loss_fn import EnergyPredictionLoss

# Replace: loss_fn = None
# With:
if OmegaConf.select(cfg, "energy_loss", default=None) is not None:
    loss_fn = EnergyPredictionLoss(
        min_time=cfg.energy_loss.min_time,
        weight=cfg.energy_loss.weight,
        normalize=cfg.energy_loss.normalize,
    )
else:
    loss_fn = None
```

```yaml
# === scripts/conf/loqi/loqi_energy.yaml ===
# 新建配置文件，基于 loqi.yaml 添加:

# ... (inherit all from loqi.yaml) ...

dynamics:
  model_args:
    energy_head: True          # 启用 energy prediction head
    # ... other args unchanged ...

energy_loss:
  min_time: 0.8                # 仅在去噪后 20% 施加
  weight: 0.1                  # 相对于主 loss 的权重
  normalize: "per_atom"        # 按原子数归一化

data:
  dataset_root: "data/chembl3d_stereo_energy"  # 带能量标签的数据集
```

### Step 5: Training

```bash
# Train with energy auxiliary task
python scripts/train.py --config-name=loqi_energy \
    outdir=./outputs/loqi_energy_v1 \
    train.gpus=1 \
    energy_loss.weight=0.1 \
    energy_loss.min_time=0.8
```

### Evaluation Strategy

```
1. 构象生成质量：与原始 LoQI 对比 (opt_median_relative_energy 等指标)
   → 辅助 loss 不应显著损害主任务

2. 表征质量：重复 Phase 0 的 linear probe
   → 对比 LoQI (无能量head) vs LoQI+Energy (有能量head)
   → 预期：能量 MAE 显著下降

3. 能量预测精度：直接看 energy_pred vs AIMNet2 ground truth
   → 这本身就是有意义的结果

4. 力预测精度（autograd forces）：
   → Force MAE (eV/Å), Force cosine similarity
   → 对比 AIMNet2 参考力 vs autograd F = -∂E/∂x
   → 验证物理一致性：∇×F ≈ 0（conservative field）
```

---

## Phase 2: Thermodynamic Multi-Task Heads (H, G, Cᵥ)

### Goal
在 Phase 1 基础上加入宏观热力学性质预测（焓 H、自由能 G、热容 Cᵥ），迫使表征编码能量面的**二阶信息**（Hessian / 振动态密度）。

### Why This Matters

```
- Energy, Forces = PES 的 0 阶和 1 阶信息
- Cᵥ = ∂²F/∂T² → 依赖振动频率 → 依赖 Hessian (PES 的 2 阶信息)
- G = H - TS → 需要熵 → 需要构象灵活性和振动态密度
- 预测这些性质迫使模型隐式学习 Hessian，产生极其丰富的表征
```

### Workflow

```
Step 1: 热力学标签生成 — 对子集分子计算 H, G, Cv（用已有工具）
Step 2: 模型改动 — 加 ThermoHead（全局池化 → MLP → 多目标回归）
Step 3: 损失函数 — 多目标 loss，支持 partial labels (semi-supervised)
Step 4: 训练 & 蒸馏
```

### Step 1: Thermodynamic Label Generation

```python
# === scripts/label_thermo.py ===
"""
计算热力学性质标签。
注意：不需要给所有 1.8M 分子都算，选多样性子集即可。
"""

# 选择子集策略：
#   1. 按骨架多样性采样 10-50 万分子
#   2. 确保覆盖不同分子量、原子数、功能团

def compute_thermo_labels(mol, aimnet2_model, temperature=298.15):
    """
    使用 AIMNet2 + RRHO 近似计算热力学性质
    
    1. AIMNet2 计算能量和 Hessian
    2. 从 Hessian 得到振动频率
    3. RRHO 近似计算 ZPE, H, S, G, Cv
    """
    # Step 1: 几何优化（确保在极小值）
    optimized_coords, energy = optimize_with_aimnet2(mol, aimnet2_model)
    
    # Step 2: 计算 Hessian（数值或解析）
    hessian = compute_hessian(optimized_coords, aimnet2_model)
    
    # Step 3: 振动分析
    frequencies = vibrational_analysis(hessian, masses)
    
    # Step 4: 统计热力学
    # ZPE = (1/2) * sum(h * nu_i)
    zpe = 0.5 * sum(PLANCK * freq for freq in real_frequencies)
    
    # Cv = sum_i R * (h*nu_i/kT)^2 * exp(h*nu_i/kT) / (exp(h*nu_i/kT) - 1)^2
    cv = sum(R * x**2 * exp(x) / (exp(x) - 1)**2 
             for x in [PLANCK * f / (KB * T) for f in real_frequencies])
    
    # H = E_0 + ZPE + thermal_correction
    enthalpy = energy + zpe + thermal_enthalpy_correction(frequencies, T)
    
    # G = H - T*S
    entropy = translational_S + rotational_S + vibrational_S
    gibbs = enthalpy - T * entropy
    
    return {"H": enthalpy, "G": gibbs, "Cv": cv, "ZPE": zpe}

# Batch processing
for mol_subset in diverse_sample(dataset, n=100000):
    labels = compute_thermo_labels(mol_subset, aimnet2)
    mol_subset.enthalpy = labels["H"]
    mol_subset.gibbs = labels["G"]
    mol_subset.cv = labels["Cv"]
```

**计算量估算**：
- Hessian 计算：~3N 次 AIMNet2 forward（数值差分），N 为原子数
- 典型 ChEMBL 分子 ~30 原子 → ~90 次 forward/molecule
- 10 万分子 → ~900 万次 forward → 单卡约 1-3 天

### Step 2: Model Modification

```python
# === src/megalodon/dynamics/fn_model.py ===
# 在 MegaFNV3Conf 中扩展:

class ThermoHead(nn.Module):
    """Multi-target thermodynamic property prediction head."""
    
    def __init__(self, node_dim=256, hidden_dim=128, n_targets=3):
        super().__init__()
        # Shared representation
        self.shared = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # Per-target heads (H, G, Cv have different scales/physics)
        self.heads = nn.ModuleDict({
            "enthalpy": nn.Linear(hidden_dim, 1),
            "gibbs": nn.Linear(hidden_dim, 1),
            "cv": nn.Linear(hidden_dim, 1),
        })
    
    def forward(self, H_atoms, batch):
        """
        Args:
            H_atoms: [N_atoms, node_dim] atom-level representations
            batch: [N_atoms] molecule assignment
        Returns:
            dict of {property_name: [N_mols, 1]}
        """
        mol_repr = scatter_mean(H_atoms, batch, dim=0)  # [N_mols, node_dim]
        shared = self.shared(mol_repr)                    # [N_mols, hidden_dim]
        return {name: head(shared) for name, head in self.heads.items()}


class MegaFNV3Conf(nn.Module):
    def __init__(self, ..., energy_head=False, thermo_head=False):
        # ... existing + Phase 1 code ...
        
        self.thermo_head_module = None
        if thermo_head:
            self.thermo_head_module = ThermoHead(
                node_dim=invariant_node_feat_dim,
                hidden_dim=invariant_node_feat_dim // 2
            )
    
    def forward(self, batch, X, H, E_idx, E, t):
        # ... existing layers ...
        
        out = {"x_hat": x, "H": H}
        
        if self.energy_head is not None:
            mol_repr = scatter_mean(H, batch, dim=0)
            out["energy_pred"] = self.energy_head(mol_repr).squeeze(-1)
        
        if self.thermo_head_module is not None:
            thermo_preds = self.thermo_head_module(H, batch)
            out["thermo_pred"] = thermo_preds  # {"enthalpy": ..., "gibbs": ..., "cv": ...}
        
        return out
```

### Step 3: Semi-Supervised Thermo Loss

```python
# === src/megalodon/models/loss_fn.py ===

class ThermoPropertyLoss:
    """
    Multi-target thermodynamic property loss with partial label support.
    
    Not all molecules have thermo labels (computed for diverse subset only).
    Loss is only applied to molecules that have labels.
    """
    
    def __init__(self, min_time=0.8, weights=None, target_keys=None):
        """
        Args:
            min_time: Only apply when t > min_time
            weights: dict of {property_name: loss_weight}
            target_keys: dict of {property_name: batch_field_name}
        """
        self.min_time = min_time
        self.weights = weights or {"enthalpy": 0.05, "gibbs": 0.05, "cv": 0.05}
        self.target_keys = target_keys or {
            "enthalpy": "enthalpy",
            "gibbs": "gibbs",
            "cv": "cv",
        }
    
    def __call__(self, batch, out, time, ws_t, stage="train"):
        if "thermo_pred" not in out:
            return torch.tensor(0.0, device=time.device)
        
        mask_time = time >= self.min_time
        if not mask_time.any():
            return torch.tensor(0.0, device=time.device)
        
        total_loss = torch.tensor(0.0, device=time.device)
        
        for prop_name, weight in self.weights.items():
            target_key = self.target_keys[prop_name]
            
            # Check if this property exists in batch
            if not hasattr(batch, target_key):
                continue
            
            target = getattr(batch, target_key)
            pred = out["thermo_pred"][prop_name].squeeze(-1)
            
            # Combined mask: time threshold + has label (non-NaN)
            has_label = ~torch.isnan(target)
            mask = mask_time & has_label
            
            if not mask.any():
                continue
            
            loss = F.mse_loss(pred[mask], target[mask])
            total_loss = total_loss + weight * loss
        
        return total_loss


class CombinedPropertyLoss:
    """Combines EnergyPredictionLoss + ThermoPropertyLoss."""
    
    def __init__(self, energy_loss_cfg=None, thermo_loss_cfg=None):
        self.energy_loss = EnergyPredictionLoss(**energy_loss_cfg) if energy_loss_cfg else None
        self.thermo_loss = ThermoPropertyLoss(**thermo_loss_cfg) if thermo_loss_cfg else None
    
    def __call__(self, batch, out, time, ws_t, stage="train"):
        loss = torch.tensor(0.0, device=time.device)
        if self.energy_loss:
            loss = loss + self.energy_loss(batch, out, time, ws_t, stage)
        if self.thermo_loss:
            loss = loss + self.thermo_loss(batch, out, time, ws_t, stage)
        return loss
```

### Step 4: Downstream Fine-Tuning (Post-Training)

```python
# === scripts/finetune_property.py ===
"""
Phase 2 产出的模型可以用于下游性质预测微调。
冻结 backbone，只训练 lightweight head。
"""

# Load pre-trained ThermoGen model
model = Graph3DInterpolantModel.load_from_checkpoint("outputs/thermogen_v1/best.ckpt")

# Freeze backbone
for param in model.dynamics.parameters():
    param.requires_grad = False

# Add downstream head (e.g., binding affinity)
downstream_head = nn.Sequential(
    nn.Linear(256, 128),
    nn.SiLU(), 
    nn.Linear(128, 1)
)

# Training loop
for batch in downstream_dataset:
    batch = preprocessor(batch)
    
    # Forward through frozen backbone at t=1 (clean coords as input)
    time = torch.ones(batch_size)
    with torch.no_grad():
        out = model.dynamics(batch, time)
    
    H = out["H"]  # [N_atoms, 256]
    mol_repr = scatter_mean(H, batch.batch, dim=0)  # [N_mols, 256]
    
    # Downstream prediction
    pred = downstream_head(mol_repr)
    loss = F.mse_loss(pred, batch.target_property)
    loss.backward()  # Only updates downstream_head
```

---

## Summary: File Changes Per Phase

### Phase 0 (Probing)
| File | Change |
|------|--------|
| `src/megalodon/models/module.py` | `sample()` 末尾加 `samples["H"] = out.get("H")` |
| `scripts/probe_representation.py` | **新建** — linear probe 脚本 |

### Phase 1 (Energy Head)
| File | Change |
|------|--------|
| `scripts/label_energy.py` | **新建** — AIMNet2 能量标注 |
| `src/megalodon/dynamics/fn_model.py` | `MegaFNV3Conf` 加 `energy_head` 参数和预测逻辑 |
| `src/megalodon/models/loss_fn.py` | **新增** `EnergyPredictionLoss` class |
| `scripts/train.py` | ~Line 46: 实例化 `loss_fn` |
| `scripts/conf/loqi/loqi_energy.yaml` | **新建** — 配置文件 |

### Phase 2 (Thermo Heads)
| File | Change |
|------|--------|
| `scripts/label_thermo.py` | **新建** — 热力学标签计算 |
| `src/megalodon/dynamics/fn_model.py` | 加 `ThermoHead` class, `MegaFNV3Conf` 加 `thermo_head` |
| `src/megalodon/models/loss_fn.py` | **新增** `ThermoPropertyLoss`, `CombinedPropertyLoss` |
| `scripts/conf/loqi/loqi_thermo.yaml` | **新建** — 完整配置文件 |
| `scripts/finetune_property.py` | **新建** — 下游微调脚本 |

---

## Key Design Decisions

### 1. Why Autograd Forces (方案A) Instead of Direct Force Head?

**决定**：力通过 `F = -∂E/∂x` autograd 得到，不加独立的 force prediction head。

**原因**：
- **物理一致性**：autograd 保证力场无旋（conservative），即能量守恒自动满足
- **等变性自动满足**：E 是 SE(3)-不变标量，对等变坐标 x 求导得到等变向量 F
- **H 是不变特征**：模型的 atom repr H 是 SE(3)-invariant 的，不能从纯不变特征直接预测等变的力向量。要做 direct force prediction 需要用模型内部的等变特征 `pos`，增加架构复杂度
- **已有先例**：`AIMNet2ForcesLoss` 已经用了同样的 autograd 方式
- **代价**：训练时多约 1x forward 开销（`create_graph=True` 的反向传播），但换来物理正确性

### 2. Why `min_time` Threshold?

去噪早期（t 小），坐标是噪声，H 编码的是"给定噪声坐标和 2D 图，预测方向"——此时的 H 不包含有意义的 3D 几何信息。强制在此时预测能量会产生错误梯度。

参考 `AIMNet2ForcesLoss` 的 `min_time=0.9` 设计。建议：
- Phase 1 起步用 `min_time=0.8`，可调
- 实验不同阈值（0.7, 0.8, 0.9, 1.0）对主任务和辅助任务的影响

### 3. Energy Normalization

AIMNet2 输出的是绝对能量（eV），不同分子差异巨大。推荐按原子数归一化作为起步，后续可以尝试减去原子参考能量（更物理）。

### 4. Semi-Supervised for Phase 2

不需要所有 1.8M 分子都有热力学标签。对无标签分子，thermo loss 自动跳过（NaN mask），只有去噪 loss 参与。这是 semi-supervised multi-task learning。

### 5. Loss Weight Tuning

辅助 loss 权重 `weight` 是关键超参。太大会损害构象生成质量，太小则表征提升不明显。建议：
- 先用小权重 (0.01-0.1) 验证不影响主任务
- 逐步增大，找到最优平衡点
- 监控 `train/x_loss` 和 `train/additional_loss_term` 的比例
