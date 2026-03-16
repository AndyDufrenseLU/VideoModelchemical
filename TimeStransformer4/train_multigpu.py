"""
Multi-GPU Multi-Task Training script for TimeSformer
Simultaneously predicts gas velocity and particle labels
Both predictions must be correct for the sample to be considered correct
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from config import TimeSformerConfig
from timesformer import create_timesformer
from dataset_txt import get_txt_dataloader


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """
    Train for one epoch with multi-task learning
    
    Args:
        model: TimeSformer model with dual heads
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        scaler: GradScaler for mixed precision (optional)
    
    Returns:
        avg_loss: Average loss for the epoch
        avg_acc: Average accuracy (both tasks correct)
        avg_acc_gas: Average accuracy for gas velocity alone
        avg_acc_particle: Average accuracy for particle alone
    """
    model.train()
    total_loss = 0.0
    total_correct_both = 0  # Both tasks correct
    total_correct_gas = 0
    total_correct_particle = 0
    total_samples = 0
    
    import time
    
    for batch_idx, (frames, labels) in enumerate(dataloader):
        batch_start = time.time()
        
        # Unpack labels (tuple of two tensors)
        label_gas, label_particle = labels
        
        # Move to device
        frames = frames.to(device, non_blocking=True)  # (B, T, C, H, W)
        label_gas = label_gas.to(device, non_blocking=True)
        label_particle = label_particle.to(device, non_blocking=True)
        
        if batch_idx == 0:
            print(f"  [Batch 1] 数据加载完成，形状: {frames.shape}")
            print(f"  [Batch 1] 开始前向传播...")
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                gas_logits, particle_logits = model(frames)
                
                if batch_idx == 0:
                    print(f"  [Batch 1] 前向传播完成，输出形状: gas={gas_logits.shape}, particle={particle_logits.shape}")
                
                # Compute loss for both tasks
                loss_gas = criterion(gas_logits, label_gas)
                loss_particle = criterion(particle_logits, label_particle)
                loss = loss_gas + loss_particle  # Combined loss
            
            if batch_idx == 0:
                print(f"  [Batch 1] 开始反向传播...")
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            gas_logits, particle_logits = model(frames)
            loss_gas = criterion(gas_logits, label_gas)
            loss_particle = criterion(particle_logits, label_particle)
            loss = loss_gas + loss_particle
            loss.backward()
            optimizer.step()
        
        if batch_idx == 0:
            print(f"  [Batch 1] 反向传播完成，用时: {time.time()-batch_start:.2f}秒")
        
        # Compute accuracy
        pred_gas = gas_logits.argmax(dim=1)
        pred_particle = particle_logits.argmax(dim=1)
        
        correct_gas = (pred_gas == label_gas).sum().item()
        correct_particle = (pred_particle == label_particle).sum().item()
        
        # Both must be correct
        correct_both = ((pred_gas == label_gas) & (pred_particle == label_particle)).sum().item()
        
        # Update stats
        batch_size = frames.size(0)
        total_loss += loss.item() * batch_size
        total_correct_both += correct_both
        total_correct_gas += correct_gas
        total_correct_particle += correct_particle
        total_samples += batch_size
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch [{batch_idx+1}/{len(dataloader)}], "
                  f"Loss: {loss.item():.4f}, "
                  f"Acc(Both): {100*correct_both/batch_size:.2f}%, "
                  f"Acc(Gas): {100*correct_gas/batch_size:.2f}%, "
                  f"Acc(Particle): {100*correct_particle/batch_size:.2f}%")
    
    avg_loss = total_loss / total_samples
    avg_acc_both = 100 * total_correct_both / total_samples
    avg_acc_gas = 100 * total_correct_gas / total_samples
    avg_acc_particle = 100 * total_correct_particle / total_samples
    
    return avg_loss, avg_acc_both, avg_acc_gas, avg_acc_particle


def validate(model, dataloader, criterion, device):
    """
    Validate the model with multi-task learning
    
    Args:
        model: TimeSformer model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to run on
    
    Returns:
        avg_loss: Average validation loss
        avg_acc: Average accuracy (both tasks correct)
        avg_acc_gas: Average accuracy for gas velocity
        avg_acc_particle: Average accuracy for particle
    """
    model.eval()
    total_loss = 0.0
    total_correct_both = 0
    total_correct_gas = 0
    total_correct_particle = 0
    total_samples = 0
    
    with torch.no_grad():
        for frames, labels in dataloader:
            label_gas, label_particle = labels
            
            frames = frames.to(device, non_blocking=True)
            label_gas = label_gas.to(device, non_blocking=True)
            label_particle = label_particle.to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                gas_logits, particle_logits = model(frames)
                loss_gas = criterion(gas_logits, label_gas)
                loss_particle = criterion(particle_logits, label_particle)
                loss = loss_gas + loss_particle
            
            # Compute accuracy
            pred_gas = gas_logits.argmax(dim=1)
            pred_particle = particle_logits.argmax(dim=1)
            
            correct_gas = (pred_gas == label_gas).sum().item()
            correct_particle = (pred_particle == label_particle).sum().item()
            correct_both = ((pred_gas == label_gas) & (pred_particle == label_particle)).sum().item()
            
            # Update stats
            batch_size = frames.size(0)
            total_loss += loss.item() * batch_size
            total_correct_both += correct_both
            total_correct_gas += correct_gas
            total_correct_particle += correct_particle
            total_samples += batch_size
    
    avg_loss = total_loss / total_samples
    avg_acc_both = 100 * total_correct_both / total_samples
    avg_acc_gas = 100 * total_correct_gas / total_samples
    avg_acc_particle = 100 * total_correct_particle / total_samples
    
    return avg_loss, avg_acc_both, avg_acc_gas, avg_acc_particle


def main():
    """
    Main training function with multi-GPU and multi-task support
    """
    # Load configuration
    config = TimeSformerConfig()
    
    print("=" * 60)
    print("TimeSformer 流化床识别 - 多任务学习 + 多GPU训练")
    print("=" * 60)
    print(f"配置: {config}")
    print()
    print("📋 任务说明:")
    print("  任务1: 气速分类 (Gas Velocity)")
    print("  任务2: 粒子分类 (Particle)")
    print("  ⚠️  两个任务都预测正确才算准确!")
    print()
    
    # 检查GPU可用性
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU训练（速度会很慢）")
        device = torch.device('cpu')
        config.use_multi_gpu = False
    else:
        device = torch.device('cuda:0')  # 主GPU
        num_gpus = torch.cuda.device_count()
        print(f"检测到 {num_gpus} 个GPU:")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print()
        
        if num_gpus < 2:
            print("只有1个GPU，将使用单GPU训练")
            config.use_multi_gpu = False
        elif config.use_multi_gpu:
            print(f"✓ 启用多GPU训练 (使用 {num_gpus} 个GPU)")
            print(f"  有效批次大小: {config.batch_size} × {num_gpus} = {config.batch_size * num_gpus}")
    
    print()
    
    # 创建模型
    print("创建 TimeSformer 多任务模型...")
    model = create_timesformer(config)
    
    # 多GPU包装
    if config.use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"使用 DataParallel 包装模型 (GPU: {list(range(torch.cuda.device_count()))})")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # 计算参数量
    if isinstance(model, nn.DataParallel):
        num_params = sum(p.numel() for p in model.module.parameters())
    else:
        num_params = sum(p.numel() for p in model.parameters())
    
    print(f"模型参数量: {num_params:,}")
    print(f"  - head_gas: 气速分类 ({config.num_classes_gas}类, 原始标签1-{config.num_classes_gas})")
    print(f"  - head_particle: 粒子分类 ({config.num_classes_particle}类, 标签0-{config.num_classes_particle-1})")
    print()
    
    # 创建数据加载器
    print("设置数据加载器...")
    txt_file = "alltrain_with_c.txt"  # 数据文件（确保此文件存在）
    
    try:
        # 训练集
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

        # 验证集
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

        print(f"训练样本: {len(train_loader.dataset)}")
        print(f"验证样本: {len(val_loader.dataset)}")
        print(f"每GPU批次: {config.batch_size}")
        if config.use_multi_gpu:
            print(f"总批次大小: {config.batch_size * torch.cuda.device_count()}")
        print(f"输入尺寸: {config.num_frames} 帧 × {config.img_height}×{config.img_width}")
        print()
        
    except Exception as e:
        print(f"错误: 无法加载数据 - {e}")
        print("请确保 alltest_with_c.txt 文件存在且格式正确")
        return
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs
    )
    
    # 混合精度训练
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    if use_amp:
        print("✓ 启用混合精度训练 (AMP) 以提升性能")
    print()
    
    # 训练循环
    print("开始训练...")
    print("=" * 60)
    best_val_acc = 0.0
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch [{epoch+1}/{config.num_epochs}]")
        print("-" * 60)
        
        # 训练
        train_loss, train_acc, train_acc_gas, train_acc_particle = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # 验证
        val_loss, val_acc, val_acc_gas, val_acc_particle = validate(
            model, val_loader, criterion, device
        )
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"\n训练 - 损失: {train_loss:.4f}, 准确率(Both): {train_acc:.2f}%")
        print(f"      Gas: {train_acc_gas:.2f}%, Particle: {train_acc_particle:.2f}%")
        print(f"验证 - 损失: {val_loss:.4f}, 准确率(Both): {val_acc:.2f}%")
        print(f"      Gas: {val_acc_gas:.2f}%, Particle: {val_acc_particle:.2f}%")
        print(f"学习率: {current_lr:.6f}")
        
        # 保存最佳模型（基于两个任务都对的准确率）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            # 保存模型
            if isinstance(model, nn.DataParallel):
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc_both': val_acc,
                'val_acc_gas': val_acc_gas,
                'val_acc_particle': val_acc_particle,
                'config': config,
            }, 'best_model_multitask.pth')
            print(f"✓ 保存最佳模型 (验证准确率Both: {val_acc:.2f}%)")
    
    print("\n" + "=" * 60)
    print(f"训练完成! 最佳验证准确率(Both): {best_val_acc:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    # 设置环境变量以优化性能
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    # 启用 cuDNN 自动调优
    torch.backends.cudnn.benchmark = True
    
    main()
