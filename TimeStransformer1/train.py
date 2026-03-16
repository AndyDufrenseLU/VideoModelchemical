"""
TimeSformer 单GPU训练脚本 - 多任务学习（气速 + 粒子分类）
包含: GPU诊断、AMP混合精度、梯度累积、进度显示、早停
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from config import TimeSformerConfig
from timesformer import create_timesformer
from dataset_txt import get_txt_dataloader


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
        print("   请检查: PyTorch是否安装了CUDA版本")
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
    
    # 检查显存使用
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
        
        # 显存使用
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"     GPU显存使用: {allocated:.2f} GB")


# ============================================================
# 训练与验证
# ============================================================

def train_epoch(model, dataloader, criterion_gas, criterion_particle, optimizer, device, scaler, 
                epoch, grad_accum_steps=2):
    """
    训练一个epoch (AMP混合精度 + 梯度累积)
    """
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
        
        # Move to GPU
        frames = frames.to(device, non_blocking=True)
        label_gas = label_gas.to(device, non_blocking=True)
        label_particle = label_particle.to(device, non_blocking=True)
        
        # 第一个batch检查GPU状态
        check_data_on_gpu(frames, (label_gas, label_particle), batch_idx)
        
        # AMP前向传播
        with torch.amp.autocast('cuda'):
            gas_logits, particle_logits = model(frames)
            loss_gas = criterion_gas(gas_logits, label_gas)
            loss_particle = criterion_particle(particle_logits, label_particle)
            loss = (loss_gas + loss_particle) / grad_accum_steps
        
        # AMP反向传播
        scaler.scale(loss).backward()
        
        # 梯度累积：每grad_accum_steps步才更新一次
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == num_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # 计算准确率
        with torch.no_grad():
            pred_gas = gas_logits.argmax(dim=1)
            pred_particle = particle_logits.argmax(dim=1)
            
            correct_gas = (pred_gas == label_gas).sum().item()
            correct_particle = (pred_particle == label_particle).sum().item()
            correct_both = ((pred_gas == label_gas) & (pred_particle == label_particle)).sum().item()
        
        batch_size = frames.size(0)
        # loss要乘回grad_accum_steps得到真实loss
        total_loss += loss.item() * grad_accum_steps * batch_size
        total_correct_both += correct_both
        total_correct_gas += correct_gas
        total_correct_particle += correct_particle
        total_samples += batch_size
        
        # 进度打印
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
    
    print("=" * 60)
    print("TimeSformer 流化床多任务识别 - 单GPU训练")
    print("=" * 60)
    print(f"配置: {config}")
    print()
    
    # ---- GPU诊断 ----
    device = gpu_diagnostic()
    
    # ---- 创建模型 ----
    print("创建模型...")
    model = create_timesformer(config)
    
    # 加载预训练权重(如果有)
    from timesformer import load_pretrained_vit
    model = load_pretrained_vit(model, "vit_b_16-c867db91.pth")
    
    model = model.to(device)
    model = check_model_on_gpu(model, device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {num_params:,} ({num_params/1e6:.1f}M)")
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
        img_size=config.img_size,
        batch_size=config.batch_size,
        split='train',
        num_workers=config.num_workers,
        train_ratio=0.75
    )
    
    val_loader = get_txt_dataloader(
        txt_file,
        num_frames=config.num_frames,
        img_size=config.img_size,
        batch_size=config.batch_size,
        split='val',
        num_workers=config.num_workers,
        train_ratio=0.75
    )
    
    print(f"  训练样本: {len(train_loader.dataset)}")
    print(f"  验证样本: {len(val_loader.dataset)}")
    print(f"  批次大小: {config.batch_size}")
    print(f"  输入尺寸: {config.num_frames}帧 × {config.img_size}×{config.img_size}")
    
    # 预估时间
    est_batches = len(train_loader.dataset) // config.batch_size
    print(f"  每epoch约 {est_batches} 个batch")
    print()
    
    # ---- 类别权重: 按标签频次自动平衡 ----
    train_ds = train_loader.dataset
    counts_gas = {}
    counts_part = {}
    for s in train_ds.samples:
        counts_gas[s['label_gas']] = counts_gas.get(s['label_gas'], 0) + 1
        counts_part[s['label_particle']] = counts_part.get(s['label_particle'], 0) + 1
    def make_weight(counts):
        # 1 / freq, 再归一化到均值1
        import numpy as np
        keys = sorted(counts.keys())
        freqs = np.array([counts[k] for k in keys], dtype=float)
        w = 1.0 / (freqs + 1e-6)
        w = w / w.mean()
        return torch.tensor(w, dtype=torch.float)
    gas_weight = make_weight(counts_gas)
    part_weight = make_weight(counts_part)
    print(f"  类别权重 gas: {gas_weight.tolist()}")
    print(f"  类别权重 particle: {part_weight.tolist()}")

    # 将权重移到设备 (如需重新启用)
    gas_weight = gas_weight.to(device)
    part_weight = part_weight.to(device)

    # ---- 损失函数: 暂时去掉class weight和label smoothing，先看能否起跳 ----
    criterion_gas = nn.CrossEntropyLoss(label_smoothing=0.0)
    criterion_particle = nn.CrossEntropyLoss(label_smoothing=0.0)
    
    # ---- 差异化学习率 ----
    head_lr = 1e-3
    backbone_lr = 2e-5
    
    # 分组: head(含dropout) vs backbone
    head_modules = [model.head_gas, model.head_particle, model.head_dropout]
    head_params = []
    for m in head_modules:
        head_params.extend(list(m.parameters()))
    head_param_ids = set(id(p) for p in head_params)
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]
    
    print(f"📋 差异化学习率微调:")
    print(f"  Backbone LR: {backbone_lr} ({sum(p.numel() for p in backbone_params):,} 参数)")
    print(f"  Head LR: {head_lr} ({sum(p.numel() for p in head_params):,} 参数)")
    print(f"  Label Smoothing: 0.0 (class weight 暂停)")
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params, 'lr': head_lr}
    ], weight_decay=config.weight_decay)
    
    # Warmup + Cosine
    import math
    warmup_epochs = 3
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
    
    # 梯度累积
    grad_accum_steps = 4
    print(f"✅ 梯度累积: {grad_accum_steps}步 (有效batch={config.batch_size * grad_accum_steps})")
    print()
    
    # ---- GPU显存预热 ----
    print("GPU显存预热中...")
    torch.cuda.empty_cache()
    mem_before = torch.cuda.memory_allocated(0) / 1024**3
    print(f"  模型加载后显存: {mem_before:.2f} GB")
    print()
    
    # ---- 训练循环 ----
    print("🚀 开始训练! (已关闭class weight/label smoothing用于排查)")
    print("=" * 60)
    
    best_val_acc = 0.0
    best_epoch = 0
    patience = 15
    no_improve = 0
    
    for epoch in range(config.num_epochs):
        bb_lr = optimizer.param_groups[0]['lr']
        hd_lr = optimizer.param_groups[1]['lr']
        print(f"\n📌 Epoch [{epoch+1}/{config.num_epochs}] (BB_LR: {bb_lr:.6f}, Head_LR: {hd_lr:.6f})")
        print("-" * 60)
        
        # 训练
        train_loss, train_both, train_gas, train_part, train_time = train_epoch(
            model, train_loader, criterion_gas, criterion_particle, optimizer, device, 
            scaler, epoch, grad_accum_steps
        )
        
        # 验证
        val_loss, val_both, val_gas, val_part, val_time = validate(
            model, val_loader, criterion_gas, criterion_particle, device
        )
        
        scheduler.step()
        
        # 打印结果
        print(f"\n  📊 Epoch {epoch+1} 结果 (训练:{train_time/60:.1f}分 + 验证:{val_time/60:.1f}分):")
        print(f"  Train | Loss: {train_loss:.4f} | Both: {train_both:.1f}% | Gas: {train_gas:.1f}% | Part: {train_part:.1f}%")
        print(f"  Val   | Loss: {val_loss:.4f} | Both: {val_both:.1f}% | Gas: {val_gas:.1f}% | Part: {val_part:.1f}%")
        
        # GPU显存
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"  GPU显存: {allocated:.2f} GB")
        
        # 保存最佳模型
        if val_both > best_val_acc:
            best_val_acc = val_both
            best_epoch = epoch + 1
            no_improve = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc_both': val_both,
                'val_acc_gas': val_gas,
                'val_acc_particle': val_part,
                'config': {
                    'img_size': config.img_size,
                    'num_frames': config.num_frames,
                    'num_classes_gas': config.num_classes_gas,
                    'num_classes_particle': config.num_classes_particle,
                    'depth': config.depth,
                }
            }, 'best_model_multitask.pth')
            print(f"  ✅ 保存最佳模型! (Val Both: {val_both:.1f}%)")
        else:
            no_improve += 1
            print(f"  ⏸️  无改善 ({no_improve}/{patience}), 最佳: Epoch {best_epoch} ({best_val_acc:.1f}%)")
        
        # 早停
        if no_improve >= patience:
            print(f"\n🛑 早停! 连续{patience}个epoch无改善")
            break
    
    print("\n" + "=" * 60)
    print(f"🏁 训练完成!")
    print(f"   最佳Val Acc(Both): {best_val_acc:.1f}% (Epoch {best_epoch})")
    print(f"   模型已保存: best_model_multitask.pth")
    print("=" * 60)


if __name__ == '__main__':
    main()
