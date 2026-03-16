# TimeSformer 流化床多任务识别

基于TimeSformer模型的流化床视频识别系统,支持**多任务学习**:同时预测气速(Gas Velocity)和粒子类型(Particle Type)。

## 📋 项目简介

本项目实现了TimeSformer(Time-Space Transformer)模型用于流化床视频的时序分析。模型采用**双头输出架构**:

- **任务1**: 气速分类 (Gas Velocity Classification)
- **任务2**: 粒子分类 (Particle Classification)

**关键特性**:
- ✅ 多头输出:同时预测气速和粒子标签
- ✅ 联合准确率:两个预测都正确才算准确
- ✅ 联合损失:梯度更新基于两个任务的损失之和
- ✅ 多GPU训练:支持DataParallel并行训练
- ✅ 混合精度:自动混合精度(AMP)加速训练

## 🏗️ 模型架构

```
输入视频序列 (B, T, C, H, W)
    ↓
Patch Embedding + 位置编码
    ↓
Transformer Blocks (Divided Space-Time Attention)
    ↓
CLS Token特征 (B, D)
    ↓
    ├── Head 1: Gas Velocity → 气速预测
    └── Head 2: Particle Type → 粒子预测
```

### 核心组件

| 文件 | 说明 |
|------|------|
| `timesformer.py` | 主模型架构,包含双头输出 |
| `attention.py` | Divided Space-Time Attention机制 |
| `patch_embedding.py` | 视频Patch嵌入层 |
| `dataset_txt.py` | 数据加载器(支持双标签) |
| `config.py` | 模型配置 |
| `train.py` | 单GPU训练脚本 |
| `train_multigpu.py` | 多GPU训练脚本 |

## 📦 环境配置

### 依赖安装

```bash
pip install torch torchvision pillow
```

### 系统要求

- Python 3.7+
- PyTorch 1.8+
- CUDA 10.2+ (推荐使用GPU)
- 内存: 16GB+ (训练时)
- 显存: 8GB+ per GPU

## 📁 数据格式

数据文件格式为TXT文件,每行包含:

```
图像路径 气速标签 粒子标签
```

示例 (`alltest_with_c.txt`):

```
E:\data\frame_0001.png 0 1
E:\data\frame_0002.png 0 1
E:\data\frame_0003.png 1 0
```

- **气速标签**: 0或1 (二分类)
- **粒子标签**: 0或1 (二分类)

## 🚀 快速开始

### 1. 单GPU训练

```bash
python train.py
```

该脚本会:
- 从`alltest_with_c.txt`加载数据
- 创建TimeSformer双头模型
- 训练并保存最佳模型到`best_model_multitask.pth`

### 2. 多GPU训练

```bash
python train_multigpu.py
```

自动检测并使用所有可用GPU。特性包括:
- **DataParallel**: 自动数据并行
- **混合精度(AMP)**: 减少显存占用,加速训练
- **多worker加载**: 并行数据加载

## ⚙️ 配置参数

编辑`config.py`调整模型参数:

```python
class TimeSformerConfig:
    # 模型架构
    img_size = 384              # 输入图像尺寸
    patch_size = 16             # Patch大小
    num_frames = 30             # 序列帧数
    embed_dim = 768             # 嵌入维度
    depth = 12                  # Transformer层数
    num_heads = 12              # 注意力头数
    
    # 任务设置
    num_classes_gas = 2         # 气速类别数
    num_classes_particle = 2    # 粒子类别数
    
    # 训练参数
    batch_size = 4              # 每GPU批次大小
    learning_rate = 1e-4        # 学习率
    num_epochs = 100            # 训练轮数
    
    # 多GPU
    use_multi_gpu = True        # 是否启用多GPU
    num_workers = 4             # 数据加载worker数
```

## 📊 训练输出

训练过程中会显示:

```
Epoch [1/100]
------------------------------------------------------------
Batch [10/200], Loss: 1.2345, Acc(Both): 65.00%, Acc(Gas): 80.00%, Acc(Particle): 75.00%
...
Train Loss: 1.1234, Train Acc(Both): 68.50%
      Gas: 82.30%, Particle: 78.60%
Val Loss: 1.0987, Val Acc(Both): 70.20%
      Gas: 84.10%, Particle: 80.50%
✓ Saved best model (Val Acc Both: 70.20%)
```

**准确率说明**:
- **Acc(Both)**: 气速和粒子都预测正确的比例 ⚠️ **主要评估指标**
- **Acc(Gas)**: 仅气速预测正确的比例
- **Acc(Particle)**: 仅粒子预测正确的比例

## 🔬 模型细节

### 多任务学习

```python
# Forward pass
gas_logits, particle_logits = model(frames)  # 双头输出

# Loss计算
loss_gas = criterion(gas_logits, label_gas)
loss_particle = criterion(particle_logits, label_particle)
loss = loss_gas + loss_particle  # 联合损失

# 准确率计算
pred_gas = gas_logits.argmax(dim=1)
pred_particle = particle_logits.argmax(dim=1)

# 两个都对才算对
correct_both = ((pred_gas == label_gas) & (pred_particle == label_particle)).sum()
```

### Divided Space-Time Attention

模型使用分离的时空注意力机制:

1. **空间注意力**: 在每帧内的patches之间
2. **时间注意力**: 在不同帧的同一位置之间

这种设计比3D卷积更高效,能捕捉长时序依赖。

## 📈 模型加载与推理

```python
import torch
from timesformer import create_timesformer
from config import TimeSformerConfig

# 加载配置
config = TimeSformerConfig()

# 创建模型
model = create_timesformer(config)

# 加载训练好的权重
checkpoint = torch.load('best_model_multitask.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 推理
with torch.no_grad():
    gas_logits, particle_logits = model(video_frames)
    
    pred_gas = gas_logits.argmax(dim=1)
    pred_particle = particle_logits.argmax(dim=1)
    
    print(f"气速预测: {pred_gas.item()}")
    print(f"粒子预测: {pred_particle.item()}")
```

## 🛠️ 故障排除

### 常见问题

**1. CUDA内存不足**

减小batch_size或img_size:

```python
config.batch_size = 2
config.img_size = 224
```

**2. 数据加载错误**

检查`alltest_with_c.txt`格式:
- 每行3个字段:路径 标签1 标签2
- 分隔符为空格
- 图像路径存在

**3. 多GPU训练不工作**

确认:
- `torch.cuda.is_available()` 返回True
- `torch.cuda.device_count() > 1`
- 设置`config.use_multi_gpu = True`

## 📚 参考文献

```bibtex
@inproceedings{bertasius2021space,
  title={Is Space-Time Attention All You Need for Video Understanding?},
  author={Bertasius, Gedas and Wang, Heng and Torresani, Lorenzo},
  booktitle={ICML},
  year={2021}
}
```

## 📄 许可证

MIT License

## 🤝 联系方式

如有问题或建议,请提交Issue。

---

**重要提示**: 本项目专为流化床视频分析设计,采用多任务学习同时预测气速和粒子类型。梯度更新和模型评估都基于**两个任务都正确**的联合准确率。
