"""
VideoMamba Multi-GPU Multi-Task Training script
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from config import VideoMambaConfig
from model import create_videomamba
from dataset_txt import get_txt_dataloader


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss, total_correct_both, total_correct_gas, total_correct_particle, total_samples = 0.0, 0, 0, 0, 0
    import time
    for batch_idx, (frames, labels) in enumerate(dataloader):
        label_gas, label_particle = labels
        frames = frames.to(device, non_blocking=True)
        label_gas = label_gas.to(device, non_blocking=True)
        label_particle = label_particle.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                gas_logits, particle_logits = model(frames)
                loss = criterion(gas_logits, label_gas) + criterion(particle_logits, label_particle)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            gas_logits, particle_logits = model(frames)
            loss = criterion(gas_logits, label_gas) + criterion(particle_logits, label_particle)
            loss.backward()
            optimizer.step()
        
        pred_gas, pred_particle = gas_logits.argmax(dim=1), particle_logits.argmax(dim=1)
        correct_gas = (pred_gas == label_gas).sum().item()
        correct_particle = (pred_particle == label_particle).sum().item()
        batch_size = frames.size(0)
        total_loss += loss.item() * batch_size
        total_correct_both += ((pred_gas == label_gas) & (pred_particle == label_particle)).sum().item()
        total_correct_gas += correct_gas
        total_correct_particle += correct_particle
        total_samples += batch_size
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
            
    return total_loss/total_samples, 100*total_correct_both/total_samples, 100*total_correct_gas/total_samples, 100*total_correct_particle/total_samples


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct_both, total_correct_gas, total_correct_particle, total_samples = 0.0, 0, 0, 0, 0
    with torch.no_grad():
        for frames, labels in dataloader:
            label_gas, label_particle = labels
            frames = frames.to(device, non_blocking=True)
            label_gas = label_gas.to(device, non_blocking=True)
            label_particle = label_particle.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                gas_logits, particle_logits = model(frames)
                loss = criterion(gas_logits, label_gas) + criterion(particle_logits, label_particle)
            
            pred_gas, pred_particle = gas_logits.argmax(dim=1), particle_logits.argmax(dim=1)
            batch_size = frames.size(0)
            total_loss += loss.item() * batch_size
            total_correct_both += ((pred_gas == label_gas) & (pred_particle == label_particle)).sum().item()
            total_correct_gas += (pred_gas == label_gas).sum().item()
            total_correct_particle += (pred_particle == label_particle).sum().item()
            total_samples += batch_size
            
    return total_loss/total_samples, 100*total_correct_both/total_samples, 100*total_correct_gas/total_samples, 100*total_correct_particle/total_samples


def main():
    config = VideoMambaConfig()
    print("=" * 60)
    print("VideoMamba 流化床识别 - 多任务学习 + 多GPU训练")
    print("=" * 60)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1 and config.use_multi_gpu: print(f"启用多GPU训练: {num_gpus} 个GPU")
    
    model = create_videomamba(config)
    if num_gpus > 1 and config.use_multi_gpu: model = nn.DataParallel(model)
    model = model.to(device)
    
    txt_file = "alltrain_with_c.txt"
    train_loader = get_txt_dataloader(txt_file, num_frames=config.num_frames, img_size=config.img_size, batch_size=config.batch_size, split='train', num_workers=config.num_workers)
    val_loader = get_txt_dataloader(txt_file, num_frames=config.num_frames, img_size=config.img_size, batch_size=config.batch_size, split='val', num_workers=config.num_workers)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.head_lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    best_val_acc = 0.0
    for epoch in range(config.num_epochs):
        print(f"\nEpoch [{epoch+1}/{config.num_epochs}]")
        train_loss, train_acc, _, _ = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Acc(Both): {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc(Both): {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({'epoch': epoch, 'model_state_dict': model_state, 'val_acc_both': val_acc}, 'best_model_multitask.pth')
            print(f"✓ 保存最佳模型!")

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    torch.backends.cudnn.benchmark = True
    main()
