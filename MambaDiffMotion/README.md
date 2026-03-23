# MambaSformer：融合 Mamba SSM 的流化床图像识别模型

## 项目简介

本项目在 **TimeSformer**（`TimeStransformer1`）基础上，将时序 Transformer 注意力替换为 **Mamba SSM**，构建了一个混合架构的视频识别模型，专门针对流化床连续图像识别任务。

### 为什么要引入 Mamba？

| 指标 | TimeSformer（原始） | MambaSformer（本项目） |
|------|---------------------|------------------------|
| 时序建模复杂度 | O(T²)，帧数增多显存爆炸 | **O(T)** 线性，帧数可扩展 |
| 时序建模方式 | 时序 Self-Attention | Mamba 状态空间模型 (SSM) |
| 空间建模方式 | 空间 Self-Attention | 空间 Self-Attention（保持不变） |
| Windows 依赖 | 无问题 | **无需** mamba-ssm / triton |
| ViT 预训练权重 | 支持 | 支持（空间注意力部分） |

### 模型架构

```
输入: (B, T, 3, 224, 224)  T 帧视频
    ↓
[Patch Embedding]  → (B, 1+T×196, 768)
    ↓
[MambaSformer Block × 12]
   ├─ Pre-Norm
   ├─ Temporal Mamba SSM   ← 时序建模（O(T) 线性）
   ├─ Pre-Norm
   ├─ Spatial Attention    ← 空间建模（ViT 预训练）
   └─ MLP (FFN)
    ↓
[LayerNorm]
    ↓
[CLS Token]  → (B, 768)
    ↓
[head_gas]      → (B, 19)  气速分类（1-19）
[head_particle] → (B, 3)   粒子分类（0-2）
```

---

## 项目结构

```
MambaSformer/
├── mamba_block.py       # 纯 PyTorch 实现的 Mamba SSM（无需编译）
├── mamba_attention.py   # 混合时空模块（Temporal Mamba + Spatial Attention）
├── mambasformer.py      # 主模型 MambaSformer
├── config.py            # 配置文件
├── patch_embedding.py   # Patch 嵌入（复用自 TimeStransformer1）
├── dataset_txt.py       # 数据加载（复用自 TimeStransformer1）
├── train.py             # 单GPU训练
├── train_multigpu.py    # 多GPU训练（DDP）
├── requirements.txt     # 依赖库
├── alltrain_with_c.txt  # 数据索引文件（需从 TimeStransformer1 复制）
└── vit_b_16-c867db91.pth  # ViT-B/16 预训练权重（可选，提升准确率）
```

---

## 安装与环境配置

### 1. 前提条件
- Windows 10/11
- Python 3.8+
- CUDA 11.7+（用于 GPU 加速）
- Conda 或 venv

### 2. 安装依赖

```powershell
# 激活 conda 环境
conda activate <你的环境>

# 进入项目目录
cd d:\desktop\VideoModelchemical\MambaSformer

# 安装依赖（无需 mamba-ssm，无需编译 CUDA 扩展）
pip install -r requirements.txt
```

> **注意**：本项目使用纯 PyTorch 实现 Mamba SSM，**不依赖** `mamba-ssm` 库，不需要在 Windows 上编译任何 C++/CUDA 扩展，直接 `pip install` 即可运行。

### 3. 准备数据文件

```powershell
# 复制数据索引文件（与 TimeStransformer1 共用同一份数据）
copy ..\TimeStransformer1\alltrain_with_c.txt .

# （可选）复制 ViT-B/16 预训练权重
copy ..\TimeStransformer1\vit_b_16-c867db91.pth .
```

---

## 运行方式

### 单GPU训练（推荐先用此方式测试）

```powershell
cd d:\desktop\VideoModelchemical\MambaSformer
python train.py
```

### 多GPU训练（DDP，2张GPU）

```powershell
cd d:\desktop\VideoModelchemical\MambaSformer
torchrun --nproc_per_node=2 train_multigpu.py
```

### 多GPU训练（4张GPU）

```powershell
torchrun --nproc_per_node=4 train_multigpu.py
```

### 快速验证模型（不需要数据文件）

```powershell
cd d:\desktop\VideoModelchemical\MambaSformer
python -c "
from config import MambaSformerConfig
from mambasformer import create_mambasformer
import torch

cfg = MambaSformerConfig()
model = create_mambasformer(cfg)

# 模拟输入: 2个sample, 10帧, RGB, 224×224
x = torch.randn(2, cfg.num_frames, 3, cfg.img_size, cfg.img_size)
gas, part = model(x)

n = sum(p.numel() for p in model.parameters())
print(f'✅ 模型前向传播成功!')
print(f'   gas shape: {gas.shape}, particle shape: {part.shape}')
print(f'   参数量: {n:,} ({n/1e6:.1f}M)')
"
```

---

## 关键参数说明

在 `config.py` 中可调整：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_frames` | 10 | 输入视频帧数（可改为 16、20） |
| `d_state` | 16 | Mamba 状态向量维度（越大记忆越长，越慢） |
| `d_conv` | 4 | 因果卷积核大小 |
| `mamba_expand` | 2 | Mamba 内部维度扩展倍数 |
| `depth` | 12 | Block 总数 |
| `backbone_lr` | 2e-5 | ViT 预训练权重学习率（小值，微调） |
| `mamba_lr` | 5e-5 | Mamba 时序层学习率（较大，从头训练） |
| `head_lr` | 1e-3 | 分类头学习率 |
| `batch_size` | 16 | 批次大小 |
| `grad_accum_steps` | 4 | 梯度累积步数（有效 batch = 64） |

---

## Mamba SSM 原理简介

Mamba 是一种**状态空间模型（SSM）**，核心公式：

```
h_t = A · h_{t-1} + B_t · x_t   (状态更新)
y_t = C_t · h_t + D · x_t       (输出)
```

其中 `B_t`, `C_t` 由当前输入 `x_t` 动态决定（这就是"Selective"）。

**与 Transformer 时序注意力的对比：**
- Transformer: 每帧与所有其他帧做注意力 → O(T²)
- Mamba: 用 RNN 式扫描 → **O(T)**，但能通过 selective 机制动态聚焦重要帧

**在流化床识别中的优势：**
- 粒子运动具有强时序相关性（前几帧的气泡状态影响后续预测）
- Mamba 天然适合这种因果时序建模
- 帧数从 10 增加到 30 帧时，Transformer 显存暴增而 Mamba 线性增长

---

## 输出文件

训练完成后，最佳模型保存为：
```
best_mambasformer.pth
```

包含：`epoch`, `model_state_dict`, `optimizer_state_dict`, `val_acc_both`, `val_acc_gas`, `val_acc_particle`, `config`



