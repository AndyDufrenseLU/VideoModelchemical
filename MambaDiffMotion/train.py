"""
MambaSformer 单GPU训练脚本
多任务学习: 气速分类 + 粒子类型分类

每次运行自动创建独立运行目录，包含:
  - training_log.txt   (终端全量输出)
  - params.txt         (人类可读参数)
  - params.json        (机器可读参数)
  - best_model.pth     (最佳模型权重)
  - summary.txt        (训练总结)

目录名格式:
  run_YYYYMMDD_HHMMSS_f{frames}_bs{batch}x{accum}_hlr{head_lr}_do{dropout}
"""

import os
import sys
import math
import time
import json
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime

# ⚡ 方法三: 消除无法避免的 PyTorch 内部警告（不影响功能）
# FutureWarning 来自 PyTorch 内部 checkpoint.py，我们的代码无法直接修复
warnings.filterwarnings('ignore', category=FutureWarning,
                        module='torch.utils.checkpoint')
# Flash Attention 警告: Windows/该版本 PyTorch 不支持，已 fallback 到标准 SDPA，正常
warnings.filterwarnings('ignore', message='.*flash attention.*',
                        category=UserWarning)
warnings.filterwarnings('ignore', message='.*Torch was not compiled with flash.*',
                        category=UserWarning)

# CUDA 性能优化
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

from config import MambaSformerConfig
from mambasformer import create_mambasformer, load_pretrained_timesformer, load_pretrained_vit
from dataset_txt import get_txt_dataloader


# ============================================================
# 运行目录管理
# ============================================================

def create_run_dir(config):
    """
    在项目目录下创建本次运行的独立子目录
    格式: run_YYYYMMDD_HHMMSS_f{num_frames}_bs{batch}x{accum}_hlr{head_lr}_do{dropout}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"run_{timestamp}"
        f"_f{config.num_frames}"
        f"_bs{config.batch_size}x{config.grad_accum_steps}"
        f"_hlr{config.head_lr:.0e}"
        f"_do{config.head_dropout}"
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_dir = os.path.join(script_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_name


def save_params(config, run_dir, run_name):
    """将所有参数保存为 params.txt 和 params.json"""
    params = config.to_dict()

    # ---- 人类可读 params.txt ----
    txt_path = os.path.join(run_dir, "params.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("MambaSformer 训练参数\n")
        f.write(f"运行名称: {run_name}\n")
        f.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        f.write("【模型架构】\n")
        for key in ['img_size', 'patch_size', 'num_frames', 'embed_dim',
                    'depth', 'num_heads', 'mlp_ratio']:
            f.write(f"  {key}: {params.get(key, 'N/A')}\n")

        f.write("\n【Mamba SSM 参数】\n")
        for key in ['d_state', 'd_conv', 'mamba_expand', 'use_checkpoint']:
            f.write(f"  {key}: {params.get(key, 'N/A')}\n")

        f.write("\n【分类任务】\n")
        for key in ['num_classes_gas', 'num_classes_particle']:
            f.write(f"  {key}: {params.get(key, 'N/A')}\n")

        f.write("\n【正则化】\n")
        for key in ['dropout', 'attention_dropout', 'head_dropout',
                    'label_smoothing', 'use_class_weights', 'weight_decay']:
            f.write(f"  {key}: {params.get(key, 'N/A')}\n")

        f.write("\n【训练设置】\n")
        for key in ['batch_size', 'grad_accum_steps', 'backbone_lr',
                    'mamba_lr', 'head_lr', 'num_epochs', 'warmup_epochs',
                    'patience', 'num_workers', 'val_interval']:
            f.write(f"  {key}: {params.get(key, 'N/A')}\n")
        eff_bs = params.get('batch_size', 16) * params.get('grad_accum_steps', 4)
        f.write(f"  effective_batch_size: {eff_bs}\n")

    # ---- 机器可读 params.json ----
    json_path = os.path.join(run_dir, "params.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False, default=str)

    return txt_path, json_path


# ============================================================
# 双向日志（终端 + 文件同时输出）
# ============================================================

class DualLogger:
    """将 stdout/stderr 同时写入终端和日志文件"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = open(log_file, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


# ============================================================
# GPU 诊断
# ============================================================

def gpu_diagnostic():
    print("=" * 60)
    print("🖥️  GPU 诊断信息")
    print("=" * 60)
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用！将使用 CPU 训练（非常慢）")
        print(f"   PyTorch 版本: {torch.__version__}")
        return torch.device('cpu')

    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  ✅ CUDA 可用")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA:    {torch.version.cuda}")
    print(f"  cuDNN:   {torch.backends.cudnn.version()}")
    print(f"  GPU:     {gpu_name}")
    print(f"  显存:    {total_mem:.1f} GB")
    print(f"  GPU 数量: {torch.cuda.device_count()}")
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved  = torch.cuda.memory_reserved(0)  / 1024**3
    print(f"  当前显存: 已分配 {allocated:.2f} GB / 已保留 {reserved:.2f} GB")
    print()
    return device


def check_model_on_gpu(model, device):
    first_param = next(model.parameters())
    if first_param.device.type == 'cuda':
        print(f"  ✅ 模型在 GPU 上: {first_param.device}")
    else:
        print(f"  ❌ 模型不在 GPU 上！正在迁移...")
        model = model.to(device)
    return model


# ============================================================
# 训练 & 验证
# ============================================================

def train_epoch(model, dataloader, criterion_gas, criterion_particle,
                optimizer, device, scaler, epoch, grad_accum_steps, config):
    model.train()
    total_loss = correct_both = correct_gas = correct_particle = total_n = 0
    num_batches = len(dataloader)
    epoch_start = time.time()
    optimizer.zero_grad()

    for batch_idx, (frames, labels) in enumerate(dataloader):
        batch_start = time.time()
        label_gas, label_particle = labels
        frames         = frames.to(device, non_blocking=True)
        label_gas      = label_gas.to(device, non_blocking=True)
        label_particle = label_particle.to(device, non_blocking=True)

        # 第一个 batch 打印 GPU 状态
        if batch_idx == 0:
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f"  ✅ 数据设备: frames={frames.device} shape={frames.shape}")
            print(f"     GPU显存: {allocated:.2f} GB")

        with torch.amp.autocast('cuda'):
            gas_logits, particle_logits = model(frames)
            loss_gas      = criterion_gas(gas_logits, label_gas)
            loss_particle = criterion_particle(particle_logits, label_particle)
            loss = (
                config.gas_loss_weight * loss_gas +
                config.particle_loss_weight * loss_particle
            ) / grad_accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == num_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        with torch.no_grad():
            pg = gas_logits.argmax(1)
            pp = particle_logits.argmax(1)
            bs = frames.size(0)
            total_loss    += loss.item() * grad_accum_steps * bs
            correct_both  += ((pg == label_gas) & (pp == label_particle)).sum().item()
            correct_gas   += (pg == label_gas).sum().item()
            correct_particle += (pp == label_particle).sum().item()
            total_n       += bs

        if batch_idx == 0:
            print(f"  [Batch 1] 完成，用时: {time.time()-batch_start:.1f}秒")

        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == num_batches:
            elapsed  = time.time() - epoch_start
            avg_t    = elapsed / (batch_idx + 1)
            remain   = avg_t * (num_batches - batch_idx - 1)
            print(f"  [{batch_idx+1}/{num_batches}] "
                  f"Loss:{total_loss/total_n:.4f} | "
                  f"Both:{100*correct_both/total_n:.1f}% | "
                  f"Gas:{100*correct_gas/total_n:.1f}% | "
                  f"Part:{100*correct_particle/total_n:.1f}% | "
                  f"{avg_t:.1f}s/batch | 剩余:{remain/60:.0f}分钟")

    epoch_time = time.time() - epoch_start
    return (total_loss / total_n,
            100 * correct_both    / total_n,
            100 * correct_gas     / total_n,
            100 * correct_particle / total_n,
            epoch_time)


def validate(model, dataloader, criterion_gas, criterion_particle, device, config):
    model.eval()
    total_loss = correct_both = correct_gas = correct_particle = total_n = 0
    val_start = time.time()

    with torch.no_grad():
        for frames, labels in dataloader:
            label_gas, label_particle = labels
            frames         = frames.to(device, non_blocking=True)
            label_gas      = label_gas.to(device, non_blocking=True)
            label_particle = label_particle.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                gas_logits, particle_logits = model(frames)
                loss = (
                    config.gas_loss_weight * criterion_gas(gas_logits, label_gas) +
                    config.particle_loss_weight * criterion_particle(particle_logits, label_particle)
                )

            pg = gas_logits.argmax(1)
            pp = particle_logits.argmax(1)
            bs = frames.size(0)
            total_loss    += loss.item() * bs
            correct_both  += ((pg == label_gas) & (pp == label_particle)).sum().item()
            correct_gas   += (pg == label_gas).sum().item()
            correct_particle += (pp == label_particle).sum().item()
            total_n       += bs

    val_time = time.time() - val_start
    return (total_loss / total_n,
            100 * correct_both    / total_n,
            100 * correct_gas     / total_n,
            100 * correct_particle / total_n,
            val_time)


# ============================================================
# 主函数
# ============================================================

def main():
    config = MambaSformerConfig()
    config.max_images_per_group = 800

    # ---- 创建本次运行的独立目录 ----
    run_dir, run_name = create_run_dir(config)

    # ---- 保存参数文件 ----
    params_txt, params_json = save_params(config, run_dir, run_name)

    # ---- 双向日志 ----
    log_path = os.path.join(run_dir, "training_log.txt")
    logger = DualLogger(log_path)
    sys.stdout = logger
    sys.stderr = logger

    print("=" * 60)
    print("🌀 MambaSformer 流化床多任务识别 - 单GPU训练")
    print("   (Temporal Mamba SSM + Spatial Transformer)")
    print("=" * 60)
    print(f"📁 运行目录: {run_dir}")
    print(f"📄 参数文件: {params_txt}")
    print(f"📄 日志文件: {log_path}")
    print(f"配置: {config}")
    print()

    # ---- 打印完整超参数 ----
    print("📋 完整超参数:")
    for key, val in sorted(config.to_dict().items()):
        print(f"  {key:25s}: {val}")
    print(f"  {'effective_batch_size':25s}: {config.batch_size * config.grad_accum_steps}")
    print()

    device = gpu_diagnostic()

    # ---- 创建模型 ----
    print("创建 MambaSformer 模型...")
    model = create_mambasformer(config)

    # ---- 预训练权重 (优先 TimeSformer > ViT > 随机初始化) ----
    ts_weights = "TimeSformer_divST_16x16_448_K400.pyth"
    vit_weights = "vit_b_16-c867db91.pth"
    if os.path.exists(ts_weights):
        print(f"  ✅ 找到 TimeSformer 预训练权重（视频预训练，精度更高）")
        model = load_pretrained_timesformer(model, ts_weights, config)
    elif os.path.exists(vit_weights):
        print(f"  ⚠️ 未找到 TimeSformer 权重，改用 ViT-B/16")
        model = load_pretrained_vit(model, vit_weights)
    else:
        print(f"  ⚠️ 未找到任何预训练权重，使用随机初始化")
        print(f"     建议: 将 TimeSformer_divST_16x16_448_K400.pyth 放到本目录")
    model = model.to(device)
    model = check_model_on_gpu(model, device)

    # ⚡ 方法三: torch.compile（仅 Linux 可用，Windows 因无 Triton 而不支持）
    # 错误原因: torch.compile 的 inductor 后端需要 Triton CUDA 编译器
    #           Triton 官方不支持 Windows，第一次 forward 时会崩溃
    # 解决: compile_model=False (已在 config.py 设置)
    if getattr(config, 'compile_model', False):
        try:
            # ✅ 不用 import torch._dynamo（会在 main() 里创建局部 torch 变量，
            #    导致 make_weight 嵌套函数里的 torch 找不到 → NameError）
            # 直接用全局已导入的 torch 访问 _dynamo 子模块即可
            torch._dynamo.config.suppress_errors = True
            print("⚡ 尝试 torch.compile (Windows 可能不支持)...")
            model = torch.compile(model, mode='reduce-overhead')
            print("   ✅ torch.compile 已启用")
        except Exception as e:
            print(f"   ⚠️ torch.compile 失败: {e}")
            print("   → 已回退到普通模式 (在 config.py 设 compile_model=False 避免此提示)")
    else:
        print("   ℹ️  compile_model=False (Windows 不支持 Triton，跳过 torch.compile)")

    num_params      = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量:     {num_params:,} ({num_params/1e6:.1f}M)")
    print(f"  可训练参数: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    print(f"  head_gas:      {config.num_classes_gas} 类")
    print(f"  head_particle: {config.num_classes_particle} 类")
    print()

    # ---- 数据加载 ----
    print("加载数据...")
    txt_file = "alltrain_with_c.txt"
    if not os.path.exists(txt_file):
        print(f"❌ 文件不存在: {txt_file}")
        print(f"   当前目录: {os.getcwd()}")
        print("   请将 TimeStransformer1/alltrain_with_c.txt 复制到此目录")
        sys.stdout = logger.terminal; logger.close()
        return

    train_loader = get_txt_dataloader(
        txt_file, num_frames=config.num_frames, img_size=config.img_size,
        batch_size=config.batch_size, split='train',
        num_workers=config.num_workers, train_ratio=config.train_ratio,
        max_images_per_group=getattr(config, 'max_images_per_group', None)
    )
    val_loader = get_txt_dataloader(
        txt_file, num_frames=config.num_frames, img_size=config.img_size,
        batch_size=config.batch_size, split='val',
        num_workers=config.num_workers, train_ratio=config.train_ratio,
        max_images_per_group=getattr(config, 'max_images_per_group', None)
    )

    print(f"  训练样本: {len(train_loader.dataset)}")
    print(f"  验证样本: {len(val_loader.dataset)}")
    print(f"  批次大小: {config.batch_size} (有效: {config.batch_size * config.grad_accum_steps})")
    print(f"  输入尺寸: {config.num_frames} 帧 × {config.img_size}×{config.img_size}")
    print(f"  每 epoch 约 {len(train_loader.dataset) // config.batch_size} 个 batch")
    print()

    # ---- 类别权重 ----
    train_ds = train_loader.dataset
    counts_gas, counts_part = {}, {}
    for s in train_ds.samples:
        counts_gas[s['label_gas']]       = counts_gas.get(s['label_gas'], 0) + 1
        counts_part[s['label_particle']] = counts_part.get(s['label_particle'], 0) + 1

    def make_weight(counts, num_classes):
        freqs = np.array([counts.get(k, 1) for k in range(num_classes)], dtype=float)
        w = 1.0 / (freqs + 1e-6)
        return torch.tensor(w / w.mean(), dtype=torch.float)

    gas_weight  = make_weight(counts_gas,  config.num_classes_gas).to(device)
    part_weight = make_weight(counts_part, config.num_classes_particle).to(device)

    print(f"  Gas  类别分布: {dict(sorted(counts_gas.items()))}")
    print(f"  Part 类别分布: {dict(sorted(counts_part.items()))}")

    # ---- 损失函数 ----
    if config.use_class_weights:
        criterion_gas      = nn.CrossEntropyLoss(weight=gas_weight,  label_smoothing=config.label_smoothing)
        criterion_particle = nn.CrossEntropyLoss(weight=part_weight, label_smoothing=config.label_smoothing)
        print(f"  ✅ 类别权重 + Label Smoothing({config.label_smoothing})")
    else:
        criterion_gas      = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        criterion_particle = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        print(f"  ✅ Label Smoothing({config.label_smoothing})，无类别权重")

    # ---- 差异化三组学习率 ----
    mamba_ids, head_ids = set(), set()
    mamba_params, head_params = [], []

    for name, p in model.named_parameters():
        if 'temporal_mamba' in name or ('mamba' in name.lower() and 'norm' not in name):
            mamba_params.append(p); mamba_ids.add(id(p))

    for m in [model.head_gas, model.head_particle, model.head_dropout]:
        for p in m.parameters():
            head_params.append(p); head_ids.add(id(p))

    backbone_params = [p for p in model.parameters()
                       if id(p) not in mamba_ids and id(p) not in head_ids]

    print(f"\n📋 差异化三组学习率:")
    print(f"  Backbone (ViT 预训练): {config.backbone_lr}  "
          f"({sum(p.numel() for p in backbone_params):,} 参数)")
    print(f"  Mamba 时序层:          {config.mamba_lr}  "
          f"({sum(p.numel() for p in mamba_params):,} 参数)")
    print(f"  分类头:                {config.head_lr}  "
          f"({sum(p.numel() for p in head_params):,} 参数)")

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config.backbone_lr},
        {'params': mamba_params,    'lr': config.mamba_lr},
        {'params': head_params,     'lr': config.head_lr},
    ], weight_decay=config.weight_decay)

    def lr_lambda(epoch):
        if epoch < config.warmup_epochs:
            return (epoch + 1) / config.warmup_epochs
        return 0.5 * (1 + math.cos(
            math.pi * (epoch - config.warmup_epochs) /
            (config.num_epochs - config.warmup_epochs)
        ))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler('cuda')

    print(f"\n✅ AMP 混合精度已启用")
    print(f"✅ 梯度累积: {config.grad_accum_steps} 步 "
          f"(有效 batch = {config.batch_size * config.grad_accum_steps})")
    print(f"✅ 梯度检查点: {config.use_checkpoint} (显存优化)")
    print()

    # ---- 正则化总结 ----
    print("🛡️  正则化配置:")
    print(f"  Head Dropout:    {config.head_dropout}")
    print(f"  Label Smoothing: {config.label_smoothing}")
    print(f"  Weight Decay:    {config.weight_decay}")
    print(f"  Class Weights:   {config.use_class_weights}")
    print()

    torch.cuda.empty_cache()
    mem_before = torch.cuda.memory_allocated(0) / 1024**3
    print(f"GPU 显存（模型加载后）: {mem_before:.2f} GB\n")

    # ---- 训练循环 ----
    print("🚀 开始训练!")
    print("=" * 60)

    best_val_acc, best_epoch, no_improve = 0.0, 0, 0
    best_model_path = os.path.join(run_dir, "best_model.pth")
    val_interval = max(1, int(getattr(config, 'val_interval', 1)))

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   [],
        'lr_bb': [], 'lr_mamba': [], 'lr_head': []
    }

    total_start = time.time()

    for epoch in range(config.num_epochs):
        bb_lr = optimizer.param_groups[0]['lr']
        mb_lr = optimizer.param_groups[1]['lr']
        hd_lr = optimizer.param_groups[2]['lr']
        print(f"\n📌 Epoch [{epoch+1}/{config.num_epochs}]  "
              f"(BB:{bb_lr:.2e}, Mamba:{mb_lr:.2e}, Head:{hd_lr:.2e})")
        print("-" * 60)

        train_loss, train_both, train_gas, train_part, train_time = train_epoch(
            model, train_loader, criterion_gas, criterion_particle,
            optimizer, device, scaler, epoch, config.grad_accum_steps, config
        )
        should_validate = ((epoch + 1) % val_interval == 0) or (epoch + 1 == config.num_epochs)
        if should_validate:
            val_loss, val_both, val_gas, val_part, val_time = validate(
                model, val_loader, criterion_gas, criterion_particle, device, config
            )
        else:
            val_loss = val_both = val_gas = val_part = None
            val_time = 0.0
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_both)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_both)
        history['lr_bb'].append(bb_lr)
        history['lr_mamba'].append(mb_lr)
        history['lr_head'].append(hd_lr)

        allocated = torch.cuda.memory_allocated(0) / 1024**3

        print(f"\n  📊 Epoch {epoch+1} (训练:{train_time/60:.1f}分 + 验证:{val_time/60:.1f}分):")
        print(f"  Train | Loss:{train_loss:.4f} | Both:{train_both:.1f}% | "
              f"Gas:{train_gas:.1f}% | Part:{train_part:.1f}%")
        if should_validate:
            gap = train_both - val_both
            print(f"  Val   | Loss:{val_loss:.4f} | Both:{val_both:.1f}% | "
                  f"Gas:{val_gas:.1f}% | Part:{val_part:.1f}%")
            print(f"  Gap   | Train-Val: {gap:.1f}% {'⚠️ 过拟合' if gap > 15 else '✅ 正常'}")
        else:
            print(f"  Val   | 跳过 (val_interval={val_interval})")
        print(f"  GPU 显存: {allocated:.2f} GB")

        if should_validate and val_both > best_val_acc:
            best_val_acc = val_both; best_epoch = epoch + 1; no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc_both':     val_both,
                'val_acc_gas':      val_gas,
                'val_acc_particle': val_part,
                'train_acc_both':   train_both,
                'history':          history,
                'config':           config.to_dict(),
                'run_name':         run_name,
            }, best_model_path)
            print(f"  ✅ 保存最佳模型! (Val Both: {val_both:.1f}%) → {best_model_path}")
        elif should_validate:
            no_improve += 1
            print(f"  ⏸️  无改善 ({no_improve}/{config.patience}), "
                  f"最佳: Epoch {best_epoch} ({best_val_acc:.1f}%)")

        if should_validate and no_improve >= config.patience:
            print(f"\n🛑 早停! 连续 {config.patience} 个 epoch 无改善")
            break

    total_time = time.time() - total_start

    # ---- 训练总结 ----
    print("\n" + "=" * 60)
    print(f"🏁 训练完成!")
    print(f"   总训练时间:      {total_time/3600:.1f} 小时")
    print(f"   最佳 Val Both:   {best_val_acc:.1f}% (Epoch {best_epoch})")
    print(f"   最终 Train Acc:  {history['train_acc'][-1]:.1f}%")
    final_val_acc = next((v for v in reversed(history['val_acc']) if v is not None), None)
    if final_val_acc is not None:
        print(f"   最终 Val Acc:    {final_val_acc:.1f}%")
        print(f"   最终 Gap:        {history['train_acc'][-1] - final_val_acc:.1f}%")
    else:
        print(f"   最终 Val Acc:    N/A")
        print(f"   最终 Gap:        N/A")
    print(f"   📁 运行目录: {run_dir}")
    print(f"   📄 模型:  {best_model_path}")
    print(f"   📄 参数:  {params_txt}")
    print(f"   📄 日志:  {log_path}")
    print("=" * 60)

    # ---- 保存 summary.txt ----
    summary_path = os.path.join(run_dir, "summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"训练总结\n")
        f.write(f"运行名称: {run_name}\n")
        f.write(f"总训练时间: {total_time/3600:.1f} 小时\n")
        f.write(f"最佳 Val Acc: {best_val_acc:.1f}% (Epoch {best_epoch})\n")
        f.write(f"最终 Train Acc: {history['train_acc'][-1]:.1f}%\n")
        if final_val_acc is not None:
            f.write(f"最终 Val Acc:   {final_val_acc:.1f}%\n")
            f.write(f"最终 Gap:       {history['train_acc'][-1] - final_val_acc:.1f}%\n")
        else:
            f.write(f"最终 Val Acc:   N/A\n")
            f.write(f"最终 Gap:       N/A\n")
        f.write(f"\n训练历史 (epoch | train_loss | train_acc | val_loss | val_acc):\n")
        for i in range(len(history['train_loss'])):
            val_loss_str = 'skip' if history['val_loss'][i] is None else f"{history['val_loss'][i]:.4f}"
            val_acc_str = 'skip' if history['val_acc'][i] is None else f"{history['val_acc'][i]:.1f}%"
            f.write(f"  {i+1:3d} | "
                    f"train_loss={history['train_loss'][i]:.4f} "
                    f"train_acc={history['train_acc'][i]:.1f}% | "
                    f"val_loss={val_loss_str} "
                    f"val_acc={val_acc_str}\n")

    # 恢复 stdout
    sys.stdout = logger.terminal
    sys.stderr = logger.terminal
    logger.close()

    print(f"\n✅ 所有文件已保存到: {run_dir}")


if __name__ == '__main__':
    main()
