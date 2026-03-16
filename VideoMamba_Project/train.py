"""
VideoMamba 单GPU训练脚本 - 多任务学习（气速 + 粒子分类）
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
torch.backends.cudnn.benchmark = True  
torch.backends.cuda.matmul.allow_tf32 = True  
torch.backends.cudnn.allow_tf32 = True  
torch.set_float32_matmul_precision('medium')  


from config import VideoMambaConfig
from model import create_videomamba
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
    
    txt_path = os.path.join(run_dir, "params.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("VideoMamba 训练参数\n")
        f.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("【模型架构】\n")
        for key in ['img_size', 'patch_size', 'num_frames', 'model_size']:
            f.write(f"  {key}: {params.get(key, 'N/A')}\n")
        
        f.write("\n【分类任务】\n")
        for key in ['num_classes_gas', 'num_classes_particle']:
            f.write(f"  {key}: {params.get(key, 'N/A')}\n")
        
        f.write("\n【正则化】\n")
        for key in ['head_dropout', 'label_smoothing', 'use_class_weights', 'weight_decay']:
            f.write(f"  {key}: {params.get(key, 'N/A')}\n")
        
        f.write("\n【训练设置】\n")
        for key in ['batch_size', 'grad_accum_steps', 'backbone_lr', 'head_lr',
                     'num_epochs', 'warmup_epochs', 'patience', 'num_workers']:
            f.write(f"  {key}: {params.get(key, 'N/A')}\n")
        effective_bs = params.get('batch_size', 8) * params.get('grad_accum_steps', 8)
        f.write(f"  effective_batch_size: {effective_bs}\n")
    
    json_path = os.path.join(run_dir, "params.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False, default=str)
    
    return txt_path, json_path


class DualLogger:
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


def gpu_diagnostic():
    print("=" * 60)
    print("🖥️  GPU 诊断信息")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用！将使用CPU训练（会非常慢）")
        return torch.device('cpu')
    
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"  ✅ CUDA 可用")
    print(f"  GPU: {gpu_name} ({total_mem:.1f} GB)")
    return device


def check_model_on_gpu(model, device):
    first_param = next(model.parameters())
    if first_param.device.type != 'cuda':
        model = model.to(device)
    return model


def train_epoch(model, dataloader, criterion_gas, criterion_particle, optimizer, device, scaler, 
                epoch, grad_accum_steps=8):
    model.train()
    total_loss, total_correct_both, total_correct_gas, total_correct_particle, total_samples = 0.0, 0, 0, 0, 0
    num_batches = len(dataloader)
    epoch_start = time.time()
    optimizer.zero_grad()
    
    for batch_idx, (frames, labels) in enumerate(dataloader):
        batch_start = time.time()
        label_gas, label_particle = labels
        
        frames = frames.to(device, non_blocking=True)
        label_gas = label_gas.to(device, non_blocking=True)
        label_particle = label_particle.to(device, non_blocking=True)
        
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
        
        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == num_batches:
            elapsed = time.time() - epoch_start
            avg_time = elapsed / (batch_idx + 1)
            print(f"  [{batch_idx+1}/{num_batches}] Loss: {total_loss/total_samples:.4f} | Both: {100*total_correct_both/total_samples:.1f}%")
            
    return total_loss/total_samples, 100*total_correct_both/total_samples, 100*total_correct_gas/total_samples, 100*total_correct_particle/total_samples, time.time() - epoch_start


def validate(model, dataloader, criterion_gas, criterion_particle, device):
    model.eval()
    total_loss, total_correct_both, total_correct_gas, total_correct_particle, total_samples = 0.0, 0, 0, 0, 0
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
            
    return total_loss/total_samples, 100*total_correct_both/total_samples, 100*total_correct_gas/total_samples, 100*total_correct_particle/total_samples, time.time() - val_start


def main():
    config = VideoMambaConfig()
    run_dir, run_name = create_run_dir(config)
    params_txt, params_json = save_params(config, run_dir)
    log_path = os.path.join(run_dir, "training_log.txt")
    logger = DualLogger(log_path)
    sys.stdout = logger
    sys.stderr = logger
    
    print("=" * 60)
    print("VideoMamba 流化床多任务识别")
    print("=" * 60)
    print(f"📁 运行目录: {run_dir}")
    print()
    
    device = gpu_diagnostic()
    print("创建模型...")
    model = create_videomamba(config)
    
    model = model.to(device)
    model = check_model_on_gpu(model, device)
    
    txt_file = "alltrain_with_c.txt"
    if not os.path.exists(txt_file):
        print(f"❌ 文件不存在: {txt_file}")
        return
    
    train_loader = get_txt_dataloader(txt_file, num_frames=config.num_frames, img_size=config.img_size, batch_size=config.batch_size, split='train', num_workers=config.num_workers, train_ratio=0.75)
    val_loader = get_txt_dataloader(txt_file, num_frames=config.num_frames, img_size=config.img_size, batch_size=config.batch_size, split='val', num_workers=config.num_workers, train_ratio=0.75)
    
    print(f"  训练样本: {len(train_loader.dataset)}  验证样本: {len(val_loader.dataset)}")
    
    train_ds = train_loader.dataset
    counts_gas, counts_part = {}, {}
    for s in train_ds.samples:
        counts_gas[s['label_gas']] = counts_gas.get(s['label_gas'], 0) + 1
        counts_part[s['label_particle']] = counts_part.get(s['label_particle'], 0) + 1
    
    def make_weight(counts, num_classes):
        freqs = np.array([counts.get(k, 1) for k in range(num_classes)], dtype=float)
        w = 1.0 / (freqs + 1e-6)
        w = w / w.mean()
        return torch.tensor(w, dtype=torch.float)
    
    gas_weight = make_weight(counts_gas, config.num_classes_gas).to(device)
    part_weight = make_weight(counts_part, config.num_classes_particle).to(device)
    
    if config.use_class_weights:
        criterion_gas = nn.CrossEntropyLoss(weight=gas_weight, label_smoothing=config.label_smoothing)
        criterion_particle = nn.CrossEntropyLoss(weight=part_weight, label_smoothing=config.label_smoothing)
    else:
        criterion_gas = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        criterion_particle = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    
    head_modules = [model.head_gas, model.head_particle, model.head_dropout]
    head_params = []
    for m in head_modules:
        head_params.extend(list(m.parameters()))
    head_param_ids = set(id(p) for p in head_params)
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config.backbone_lr},
        {'params': head_params, 'lr': config.head_lr}
    ], weight_decay=config.weight_decay)
    
    warmup_epochs = config.warmup_epochs
    def lr_lambda(epoch):
        if epoch < warmup_epochs: return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (config.num_epochs - warmup_epochs)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_acc, best_epoch, no_improve = 0.0, 0, 0
    best_model_path = os.path.join(run_dir, "best_model.pth")
    total_start = time.time()
    
    for epoch in range(config.num_epochs):
        print(f"\n📌 Epoch [{epoch+1}/{config.num_epochs}]")
        train_loss, train_both, train_gas, train_part, train_time = train_epoch(model, train_loader, criterion_gas, criterion_particle, optimizer, device, scaler, epoch, config.grad_accum_steps)
        val_loss, val_both, val_gas, val_part, val_time = validate(model, val_loader, criterion_gas, criterion_particle, device)
        scheduler.step()
        
        print(f"  Train | Loss: {train_loss:.4f} | Both: {train_both:.1f}% | Gas: {train_gas:.1f}% | Part: {train_part:.1f}%")
        print(f"  Val   | Loss: {val_loss:.4f} | Both: {val_both:.1f}% | Gas: {val_gas:.1f}% | Part: {val_part:.1f}%")
        
        if val_both > best_val_acc:
            best_val_acc = val_both
            best_epoch = epoch + 1
            no_improve = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_acc_both': val_both}, best_model_path)
            print("  ✅ 保存最佳模型!")
        else:
            no_improve += 1
            if no_improve >= config.patience:
                print("🛑 早停!")
                break
                
    sys.stdout = logger.terminal
    logger.close()

if __name__ == '__main__':
    main()
