"""
TimeSformer 单GPU训练脚本 - 多任务学习（气速 + 粒子分类）
Anti-Overfitting版: DropPath, Dropout, Label Smoothing, Class Weights, 强增强

每次运行自动创建独立的运行文件夹，包含:
  - training_log.txt   (所有终端输出)
  - params.txt         (完整参数列表)
  - best_model.pth     (最佳模型)
"""

import os
import sys
import time
import math
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime

# ---- CUDA加速优化 ----
torch.backends.cudnn.benchmark = True  # 自动选择最快的卷积算法
torch.backends.cuda.matmul.allow_tf32 = True  # A5000等Ampere GPU用TF32加速矩阵乘
torch.backends.cudnn.allow_tf32 = True  # cuDNN也用TF32
torch.set_float32_matmul_precision('medium')  # 允许降低精度换取速度


from config import TimeSformerConfig
from timesformer import create_timesformer
from dataset_txt import get_txt_dataloader


# ============================================================
# 运行目录管理
# ============================================================

def create_run_dir(config):
    """
    在当前项目文件夹下创建本次运行的独立目录
    格式: run_YYYYMMDD_HHMMSS_f{num_frames}_bs{batch_size}_lr{head_lr}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (f"run_{timestamp}"
                f"_f{config.num_frames}"
                f"_bs{config.batch_size}x{config.grad_accum_steps}"
                f"_hlr{config.head_lr:.0e}"
                f"_do{config.head_dropout}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_dir = os.path.join(script_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    return run_dir, run_name


def save_params(config, run_dir):
    """将所有参数保存到 params.txt 和 params.json"""
    params = config.to_dict()
    
    # 人类可读的 params.txt
    txt_path = os.path.join(run_dir, "params.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("TimeSformer 训练参数\n")
        f.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("【模型架构】\n")
        for key in ['img_size', 'patch_size', 'num_frames', 'embed_dim', 
                     'depth', 'num_heads', 'mlp_ratio', 'attention_type']:
            f.write(f"  {key}: {params.get(key, 'N/A')}\n")
        
        f.write("\n【分类任务】\n")
        for key in ['num_classes_gas', 'num_classes_particle']:
            f.write(f"  {key}: {params.get(key, 'N/A')}\n")
        
        f.write("\n【正则化】\n")
        for key in ['dropout', 'attention_dropout', 'head_dropout', 
                     'drop_path_rate', 'label_smoothing', 'use_class_weights',
                     'weight_decay']:
            f.write(f"  {key}: {params.get(key, 'N/A')}\n")
        
        f.write("\n【训练设置】\n")
        for key in ['batch_size', 'grad_accum_steps', 'backbone_lr', 'head_lr',
                     'num_epochs', 'warmup_epochs', 'patience', 'num_workers']:
            f.write(f"  {key}: {params.get(key, 'N/A')}\n")
        effective_bs = params.get('batch_size', 8) * params.get('grad_accum_steps', 8)
        f.write(f"  effective_batch_size: {effective_bs}\n")
    
    # 机器可读的 params.json
    json_path = os.path.join(run_dir, "params.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False, default=str)
    
    return txt_path, json_path


# ============================================================
# 日志系统: 同时输出到终端和文件
# ============================================================

class DualLogger:
    """同时输出到终端和文件的日志器"""
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
    """打印详细的GPU诊断信息"""
    print("=" * 60)
    print("🖥️  GPU 诊断信息")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用！将使用CPU训练（会非常慢）")
        print(f"   PyTorch版本: {torch.__version__}")
        return torch.device('cpu')
    
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"  ✅ CUDA 可用")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.version.cuda}")
    print(f"  cuDNN: {torch.backends.cudnn.version()}")
    print(f"  GPU: {gpu_name}")
    print(f"  显存: {total_mem:.1f} GB")
    print(f"  GPU数量: {torch.cuda.device_count()}")
    
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    print(f"  当前显存: 已分配 {allocated:.2f} GB / 已保留 {reserved:.2f} GB")
    print()
    
    return device


def check_model_on_gpu(model, device):
    """检查模型是否真的在GPU上"""
    first_param = next(model.parameters())
    param_device = first_param.device
    
    if param_device.type == 'cuda':
        print(f"  ✅ 模型在GPU上: {param_device}")
    else:
        print(f"  ❌ 模型不在GPU上! 实际设备: {param_device}")
        print(f"     正在迁移到 {device}...")
        model = model.to(device)
    
    return model


def check_data_on_gpu(frames, labels, batch_idx):
    """检查数据是否在GPU上（仅第一个batch检查）"""
    if batch_idx == 0:
        label_gas, label_particle = labels
        print(f"  ✅ 数据设备检查:")
        print(f"     frames: {frames.device}, shape: {frames.shape}")
        print(f"     label_gas: {label_gas.device}, label_particle: {label_particle.device}")
        
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"     GPU显存使用: {allocated:.2f} GB")


# ============================================================
# 训练与验证
# ============================================================

def train_epoch(model, dataloader, criterion_gas, criterion_particle, optimizer, device, scaler, 
                epoch, grad_accum_steps=8):
    """训练一个epoch (AMP混合精度 + 梯度累积)"""
    model.train()
    total_loss = 0.0
    total_correct_both = 0
    total_correct_gas = 0
    total_correct_particle = 0
    total_samples = 0
    
    num_batches = len(dataloader)
    epoch_start = time.time()
    
    optimizer.zero_grad()
    
    for batch_idx, (frames, labels) in enumerate(dataloader):
        batch_start = time.time()
        
        label_gas, label_particle = labels
        
        frames = frames.to(device, non_blocking=True)
        label_gas = label_gas.to(device, non_blocking=True)
        label_particle = label_particle.to(device, non_blocking=True)
        
        check_data_on_gpu(frames, (label_gas, label_particle), batch_idx)
        
        with torch.amp.autocast('cuda'):
            gas_logits, particle_logits = model(frames)
            loss_gas = criterion_gas(gas_logits, label_gas)
            loss_particle = criterion_particle(particle_logits, label_particle)
            loss = (loss_gas + loss_particle) / grad_accum_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == num_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        with torch.no_grad():
            pred_gas = gas_logits.argmax(dim=1)
            pred_particle = particle_logits.argmax(dim=1)
            
            correct_gas = (pred_gas == label_gas).sum().item()
            correct_particle = (pred_particle == label_particle).sum().item()
            correct_both = ((pred_gas == label_gas) & (pred_particle == label_particle)).sum().item()
        
        batch_size = frames.size(0)
        total_loss += loss.item() * grad_accum_steps * batch_size
        total_correct_both += correct_both
        total_correct_gas += correct_gas
        total_correct_particle += correct_particle
        total_samples += batch_size
        
        batch_time = time.time() - batch_start
        
        if batch_idx == 0:
            print(f"  [Batch 1] 完成，用时: {batch_time:.1f}秒")
            if batch_time > 60:
                print(f"  ⚠️ 第一个batch较慢是正常的(CUDA初始化)")
        
        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == num_batches:
            elapsed = time.time() - epoch_start
            avg_time = elapsed / (batch_idx + 1)
            remaining = avg_time * (num_batches - batch_idx - 1)
            
            running_acc_both = 100 * total_correct_both / total_samples
            running_acc_gas = 100 * total_correct_gas / total_samples
            running_acc_part = 100 * total_correct_particle / total_samples
            running_loss = total_loss / total_samples
            
            print(f"  [{batch_idx+1}/{num_batches}] "
                  f"Loss: {running_loss:.4f} | "
                  f"Both: {running_acc_both:.1f}% | "
                  f"Gas: {running_acc_gas:.1f}% | "
                  f"Part: {running_acc_part:.1f}% | "
                  f"{avg_time:.1f}s/batch | "
                  f"剩余: {remaining/60:.0f}分钟")
    
    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / total_samples
    avg_acc_both = 100 * total_correct_both / total_samples
    avg_acc_gas = 100 * total_correct_gas / total_samples
    avg_acc_particle = 100 * total_correct_particle / total_samples
    
    return avg_loss, avg_acc_both, avg_acc_gas, avg_acc_particle, epoch_time


def validate(model, dataloader, criterion_gas, criterion_particle, device):
    """验证"""
    model.eval()
    total_loss = 0.0
    total_correct_both = 0
    total_correct_gas = 0
    total_correct_particle = 0
    total_samples = 0
    
    val_start = time.time()
    
    with torch.no_grad():
        for frames, labels in dataloader:
            label_gas, label_particle = labels
            
            frames = frames.to(device, non_blocking=True)
            label_gas = label_gas.to(device, non_blocking=True)
            label_particle = label_particle.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                gas_logits, particle_logits = model(frames)
                loss_gas = criterion_gas(gas_logits, label_gas)
                loss_particle = criterion_particle(particle_logits, label_particle)
                loss = loss_gas + loss_particle
            
            pred_gas = gas_logits.argmax(dim=1)
            pred_particle = particle_logits.argmax(dim=1)
            
            correct_gas = (pred_gas == label_gas).sum().item()
            correct_particle = (pred_particle == label_particle).sum().item()
            correct_both = ((pred_gas == label_gas) & (pred_particle == label_particle)).sum().item()
            
            batch_size = frames.size(0)
            total_loss += loss.item() * batch_size
            total_correct_both += correct_both
            total_correct_gas += correct_gas
            total_correct_particle += correct_particle
            total_samples += batch_size
    
    val_time = time.time() - val_start
    avg_loss = total_loss / total_samples
    avg_acc_both = 100 * total_correct_both / total_samples
    avg_acc_gas = 100 * total_correct_gas / total_samples
    avg_acc_particle = 100 * total_correct_particle / total_samples
    
    return avg_loss, avg_acc_both, avg_acc_gas, avg_acc_particle, val_time


# ============================================================
# 主函数
# ============================================================

def main():
    config = TimeSformerConfig()
    
    # ---- 创建本次运行的独立目录 ----
    run_dir, run_name = create_run_dir(config)
    
    # ---- 保存参数文件 ----
    params_txt, params_json = save_params(config, run_dir)
    
    # ---- 设置双向日志 (终端 + 文件) ----
    log_path = os.path.join(run_dir, "training_log.txt")
    logger = DualLogger(log_path)
    sys.stdout = logger
    sys.stderr = logger
    
    print("=" * 60)
    print("TimeSformer 流化床多任务识别 - Anti-Overfitting版")
    print("=" * 60)
    print(f"📁 运行目录: {run_dir}")
    print(f"📄 参数文件: {params_txt}")
    print(f"📄 日志文件: {log_path}")
    print(f"配置: {config}")
    print()
    
    # ---- 打印完整超参数 ----
    print("📋 完整超参数:")
    params = config.to_dict()
    for key, val in sorted(params.items()):
        print(f"  {key:25s}: {val}")
    effective_bs = config.batch_size * config.grad_accum_steps
    print(f"  {'effective_batch_size':25s}: {effective_bs}")
    print()
    
    # ---- GPU诊断 ----
    device = gpu_diagnostic()
    
    # ---- 创建模型 ----
    print("创建模型...")
    model = create_timesformer(config)
    
    from timesformer import load_pretrained_timesformer
    model = load_pretrained_timesformer(model, "TimeSformer_divST_16x16_448_K400.pyth", config)
    
    model = model.to(device)
    model = check_model_on_gpu(model, device)
    
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量: {num_params:,} ({num_params/1e6:.1f}M)")
    print(f"  可训练参数: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    print(f"  head_gas: {config.num_classes_gas}类 (气速1-{config.num_classes_gas})")
    print(f"  head_particle: {config.num_classes_particle}类 (粒子0-{config.num_classes_particle-1})")
    print()
    
    # ---- 数据加载 ----
    print("加载数据...")
    txt_file = "alltrain_with_c.txt"
    
    if not os.path.exists(txt_file):
        print(f"❌ 文件不存在: {txt_file}")
        print(f"   当前目录: {os.getcwd()}")
        return
    
    train_loader = get_txt_dataloader(
        txt_file,
        num_frames=config.num_frames,
        img_width=config.img_width,
        img_height=config.img_height,
        batch_size=config.batch_size,
        split='train',
        num_workers=config.num_workers,
        train_ratio=0.80
    )

    val_loader = get_txt_dataloader(
        txt_file,
        num_frames=config.num_frames,
        img_width=config.img_width,
        img_height=config.img_height,
        batch_size=config.batch_size,
        split='val',
        num_workers=config.num_workers,
        train_ratio=0.80
    )
    
    print(f"  训练样本: {len(train_loader.dataset)}")
    print(f"  验证样本: {len(val_loader.dataset)}")
    print(f"  批次大小: {config.batch_size} (有效: {config.batch_size * config.grad_accum_steps})")
    print(f"  输入尺寸: {config.num_frames}帧 × {config.img_height}×{config.img_width} (高×宽)")
    print(f"  每帧patch: {config.patches_h}×{config.patches_w} = {config.num_patches_per_frame}")
    print(f"  num_workers: {config.num_workers}")
    
    est_batches = len(train_loader.dataset) // config.batch_size
    print(f"  每epoch约 {est_batches} 个batch")
    print()
    
    # ---- 类别权重 ----
    train_ds = train_loader.dataset
    counts_gas = {}
    counts_part = {}
    for s in train_ds.samples:
        counts_gas[s['label_gas']] = counts_gas.get(s['label_gas'], 0) + 1
        counts_part[s['label_particle']] = counts_part.get(s['label_particle'], 0) + 1
    
    def make_weight(counts, num_classes):
        freqs = np.array([counts.get(k, 1) for k in range(num_classes)], dtype=float)
        w = 1.0 / (freqs + 1e-6)
        w = w / w.mean()
        return torch.tensor(w, dtype=torch.float)
    
    gas_weight = make_weight(counts_gas, config.num_classes_gas)
    part_weight = make_weight(counts_part, config.num_classes_particle)
    print(f"  Gas类别分布: {dict(sorted(counts_gas.items()))}")
    print(f"  Particle类别分布: {dict(sorted(counts_part.items()))}")
    print(f"  类别权重 gas (前5): {gas_weight[:5].tolist()}")
    print(f"  类别权重 particle: {part_weight.tolist()}")
    
    gas_weight = gas_weight.to(device)
    part_weight = part_weight.to(device)
    
    # ---- 损失函数 ----
    if config.use_class_weights:
        criterion_gas = nn.CrossEntropyLoss(
            weight=gas_weight, label_smoothing=config.label_smoothing)
        criterion_particle = nn.CrossEntropyLoss(
            weight=part_weight, label_smoothing=config.label_smoothing)
        print(f"  ✅ 使用类别权重 + Label Smoothing({config.label_smoothing})")
    else:
        criterion_gas = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        criterion_particle = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        print(f"  ✅ Label Smoothing({config.label_smoothing}), 无类别权重")
    
    # ---- 差异化学习率 ----
    head_modules = [model.head_gas, model.head_particle, model.head_dropout]
    head_params = []
    for m in head_modules:
        head_params.extend(list(m.parameters()))
    head_param_ids = set(id(p) for p in head_params)
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]
    
    print(f"\n📋 差异化学习率:")
    print(f"  Backbone LR: {config.backbone_lr} ({sum(p.numel() for p in backbone_params):,} 参数)")
    print(f"  Head LR: {config.head_lr} ({sum(p.numel() for p in head_params):,} 参数)")
    print(f"  Weight Decay: {config.weight_decay}")
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config.backbone_lr},
        {'params': head_params, 'lr': config.head_lr}
    ], weight_decay=config.weight_decay)
    
    # Warmup + Cosine
    warmup_epochs = config.warmup_epochs
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + math.cos(
            math.pi * (epoch - warmup_epochs) / (config.num_epochs - warmup_epochs)
        ))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # AMP
    scaler = torch.amp.GradScaler('cuda')
    print("✅ AMP混合精度训练已启用")
    
    grad_accum_steps = config.grad_accum_steps
    print(f"✅ 梯度累积: {grad_accum_steps}步 (有效batch={config.batch_size * grad_accum_steps})")
    
    # 正则化总结
    print(f"\n🛡️  正则化配置:")
    print(f"  Dropout (MLP/Embed): {config.dropout}")
    print(f"  Attention Dropout:   {config.attention_dropout}")
    print(f"  Head Dropout:        {config.head_dropout}")
    print(f"  DropPath Rate:       {config.drop_path_rate}")
    print(f"  Label Smoothing:     {config.label_smoothing}")
    print(f"  Weight Decay:        {config.weight_decay}")
    print(f"  Class Weights:       {config.use_class_weights}")
    print()
    
    # ---- GPU显存预热 ----
    print("GPU显存预热中...")
    torch.cuda.empty_cache()
    mem_before = torch.cuda.memory_allocated(0) / 1024**3
    print(f"  模型加载后显存: {mem_before:.2f} GB")
    print()
    
    # ---- 训练循环 ----
    print("🚀 开始训练!")
    print("=" * 60)
    
    best_val_acc = 0.0
    best_epoch = 0
    patience = config.patience
    no_improve = 0
    
    # 最佳模型保存在运行目录内
    best_model_path = os.path.join(run_dir, "best_model.pth")
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr_bb': [], 'lr_head': []
    }
    
    total_start = time.time()
    
    for epoch in range(config.num_epochs):
        bb_lr = optimizer.param_groups[0]['lr']
        hd_lr = optimizer.param_groups[1]['lr']
        print(f"\n📌 Epoch [{epoch+1}/{config.num_epochs}] (BB_LR: {bb_lr:.6f}, Head_LR: {hd_lr:.6f})")
        print("-" * 60)
        
        train_loss, train_both, train_gas, train_part, train_time = train_epoch(
            model, train_loader, criterion_gas, criterion_particle, optimizer, device, 
            scaler, epoch, grad_accum_steps
        )
        
        val_loss, val_both, val_gas, val_part, val_time = validate(
            model, val_loader, criterion_gas, criterion_particle, device
        )
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_both)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_both)
        history['lr_bb'].append(bb_lr)
        history['lr_head'].append(hd_lr)
        
        gap = train_both - val_both
        print(f"\n  📊 Epoch {epoch+1} 结果 (训练:{train_time/60:.1f}分 + 验证:{val_time/60:.1f}分):")
        print(f"  Train | Loss: {train_loss:.4f} | Both: {train_both:.1f}% | Gas: {train_gas:.1f}% | Part: {train_part:.1f}%")
        print(f"  Val   | Loss: {val_loss:.4f} | Both: {val_both:.1f}% | Gas: {val_gas:.1f}% | Part: {val_part:.1f}%")
        print(f"  Gap   | Train-Val: {gap:.1f}% {'⚠️ 过拟合' if gap > 15 else '✅ 正常'}")
        
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"  GPU显存: {allocated:.2f} GB")
        
        if val_both > best_val_acc:
            best_val_acc = val_both
            best_epoch = epoch + 1
            no_improve = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc_both': val_both,
                'val_acc_gas': val_gas,
                'val_acc_particle': val_part,
                'train_acc_both': train_both,
                'history': history,
                'config': config.to_dict(),
                'run_name': run_name,
            }, best_model_path)
            print(f"  ✅ 保存最佳模型! (Val Both: {val_both:.1f}%) → {best_model_path}")
        else:
            no_improve += 1
            print(f"  ⏸️  无改善 ({no_improve}/{patience}), 最佳: Epoch {best_epoch} ({best_val_acc:.1f}%)")
        
        if no_improve >= patience:
            print(f"\n🛑 早停! 连续{patience}个epoch无改善")
            break
    
    total_time = time.time() - total_start
    
    # ---- 训练总结 ----
    print("\n" + "=" * 60)
    print(f"🏁 训练完成!")
    print(f"   总训练时间: {total_time/3600:.1f}小时")
    print(f"   最佳Val Acc(Both): {best_val_acc:.1f}% (Epoch {best_epoch})")
    print(f"   最终Train Acc: {history['train_acc'][-1]:.1f}%")
    print(f"   最终Val Acc: {history['val_acc'][-1]:.1f}%")
    print(f"   最终Gap: {history['train_acc'][-1] - history['val_acc'][-1]:.1f}%")
    print(f"   📁 运行目录: {run_dir}")
    print(f"   📄 模型: {best_model_path}")
    print(f"   📄 参数: {params_txt}")
    print(f"   📄 日志: {log_path}")
    print("=" * 60)
    
    # 保存训练总结到单独文件
    summary_path = os.path.join(run_dir, "summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"训练总结\n")
        f.write(f"运行名称: {run_name}\n")
        f.write(f"总训练时间: {total_time/3600:.1f}小时\n")
        f.write(f"最佳Val Acc: {best_val_acc:.1f}% (Epoch {best_epoch})\n")
        f.write(f"最终Train Acc: {history['train_acc'][-1]:.1f}%\n")
        f.write(f"最终Val Acc: {history['val_acc'][-1]:.1f}%\n")
        f.write(f"最终Gap: {history['train_acc'][-1] - history['val_acc'][-1]:.1f}%\n")
        f.write(f"\n训练历史 (epoch, train_loss, train_acc, val_loss, val_acc):\n")
        for i in range(len(history['train_loss'])):
            f.write(f"  {i+1:3d} | train_loss={history['train_loss'][i]:.4f} "
                    f"train_acc={history['train_acc'][i]:.1f}% | "
                    f"val_loss={history['val_loss'][i]:.4f} "
                    f"val_acc={history['val_acc'][i]:.1f}%\n")
    
    # 恢复stdout
    sys.stdout = logger.terminal
    sys.stderr = logger.terminal
    logger.close()
    
    print(f"\n✅ 所有文件已保存到: {run_dir}")


if __name__ == '__main__':
    main()
