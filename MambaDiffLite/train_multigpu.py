"""
MambaSformer 多GPU训练脚本（DistributedDataParallel）

每次运行自动创建独立运行目录（与单GPU版本 train.py 相同格式）:
  run_YYYYMMDD_HHMMSS_f{frames}_bs{batch}x{accum}_hlr{head_lr}_do{dropout}/
    ├── training_log.txt
    ├── params.txt
    ├── params.json
    ├── best_model.pth
    └── summary.txt

用法:
  torchrun --nproc_per_node=2 train_multigpu.py   # 2张GPU
  torchrun --nproc_per_node=4 train_multigpu.py   # 4张GPU
"""

import os
import sys
import math
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from datetime import datetime

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

from config import MambaSformerConfig
from mambasformer import create_mambasformer, load_pretrained_vit
from dataset_txt import get_txt_dataloader


# ============================================================
# DDP 工具
# ============================================================

def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


# ============================================================
# 运行目录管理（仅主进程操作）
# ============================================================

def create_run_dir(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"run_{timestamp}"
        f"_f{config.num_frames}"
        f"_bs{config.batch_size}x{config.grad_accum_steps}"
        f"_hlr{config.head_lr:.0e}"
        f"_do{config.head_dropout}"
        f"_ddp{dist.get_world_size() if dist.is_initialized() else 1}gpu"
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_dir = os.path.join(script_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_name


def save_params(config, run_dir, run_name):
    params = config.to_dict()
    if dist.is_initialized():
        params['world_size'] = dist.get_world_size()

    txt_path = os.path.join(run_dir, "params.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("MambaSformer 多GPU训练参数\n")
        f.write(f"运行名称: {run_name}\n")
        f.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if dist.is_initialized():
            f.write(f"GPU 数量: {dist.get_world_size()}\n")
        f.write("=" * 60 + "\n\n")

        f.write("【模型架构】\n")
        for key in ['img_size', 'patch_size', 'num_frames', 'embed_dim',
                    'depth', 'num_heads', 'mlp_ratio']:
            f.write(f"  {key}: {params.get(key, 'N/A')}\n")

        f.write("\n【Mamba SSM 参数】\n")
        for key in ['d_state', 'd_conv', 'mamba_expand', 'use_checkpoint']:
            f.write(f"  {key}: {params.get(key, 'N/A')}\n")

        f.write("\n【训练设置】\n")
        for key in ['batch_size', 'grad_accum_steps', 'backbone_lr',
                    'mamba_lr', 'head_lr', 'num_epochs', 'warmup_epochs',
                    'patience', 'num_workers', 'val_interval']:
            f.write(f"  {key}: {params.get(key, 'N/A')}\n")

        f.write("\n【正则化】\n")
        for key in ['head_dropout', 'label_smoothing', 'use_class_weights', 'weight_decay']:
            f.write(f"  {key}: {params.get(key, 'N/A')}\n")

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


# ============================================================
# 训练 & 验证
# ============================================================

def train_epoch(model, dataloader, criterion_gas, criterion_particle,
                optimizer, device, scaler, grad_accum_steps, config):
    model.train()
    total_loss = correct_both = correct_gas = correct_particle = total_n = 0
    num_batches = len(dataloader)
    optimizer.zero_grad()

    for idx, (frames, labels) in enumerate(dataloader):
        label_gas, label_particle = labels
        frames         = frames.to(device, non_blocking=True)
        label_gas      = label_gas.to(device, non_blocking=True)
        label_particle = label_particle.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            gas_logits, particle_logits = model(frames)
            loss = (
                config.gas_loss_weight * criterion_gas(gas_logits, label_gas) +
                config.particle_loss_weight * criterion_particle(particle_logits, label_particle)
            ) / grad_accum_steps

        scaler.scale(loss).backward()

        if (idx + 1) % grad_accum_steps == 0 or (idx + 1) == num_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

    return (total_loss / total_n,
            100 * correct_both / total_n,
            100 * correct_gas  / total_n,
            100 * correct_particle / total_n)


def validate(model, dataloader, criterion_gas, criterion_particle, device, config):
    model.eval()
    total_loss = correct_both = correct_gas = correct_particle = total_n = 0

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

    return (total_loss / total_n,
            100 * correct_both / total_n,
            100 * correct_gas  / total_n,
            100 * correct_particle / total_n)


# ============================================================
# 主函数
# ============================================================

def main():
    local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    config = MambaSformerConfig()
    config.max_images_per_group = 800

    # ---- 仅主进程创建目录和日志 ----
    run_dir = run_name = params_txt = log_path = None
    logger = None

    if is_main():
        run_dir, run_name = create_run_dir(config)
        params_txt, _ = save_params(config, run_dir, run_name)
        log_path = os.path.join(run_dir, "training_log.txt")
        logger = DualLogger(log_path)
        sys.stdout = logger
        sys.stderr = logger

        print("=" * 60)
        print("🌀 MambaSformer 流化床多任务识别 - 多GPU训练 (DDP)")
        print("=" * 60)
        print(f"📁 运行目录: {run_dir}")
        print(f"📄 日志文件: {log_path}")
        print(f"GPU 数量:    {dist.get_world_size()}")
        print(f"配置: {config}")
        print()
        for key, val in sorted(config.to_dict().items()):
            print(f"  {key:25s}: {val}")
        print()

    # ---- 模型 ----
    model = create_mambasformer(config)
    if is_main():
        model = load_pretrained_vit(model, "vit_b_16-c867db91.pth")
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    raw_model = model.module

    if is_main():
        n = sum(p.numel() for p in raw_model.parameters())
        print(f"参数量: {n:,} ({n/1e6:.1f}M)\n")

    # ---- 数据 ----
    txt_file = "alltrain_with_c.txt"
    if not os.path.exists(txt_file):
        if is_main():
            print(f"❌ 数据文件不存在: {txt_file}")
        cleanup_ddp()
        return

    train_loader = get_txt_dataloader(
        txt_file, num_frames=config.num_frames, img_size=config.img_size,
        batch_size=config.batch_size, split='train',
        num_workers=config.num_workers, train_ratio=config.train_ratio,
        distributed=True, rank=dist.get_rank(), world_size=dist.get_world_size()
    )
    val_loader = get_txt_dataloader(
        txt_file, num_frames=config.num_frames, img_size=config.img_size,
        batch_size=config.batch_size, split='val',
        num_workers=config.num_workers, train_ratio=config.train_ratio
    )

    # ---- 类别权重 ----
    train_ds = train_loader.dataset
    counts_gas, counts_part = {}, {}
    for s in train_ds.samples:
        counts_gas[s['label_gas']]       = counts_gas.get(s['label_gas'], 0) + 1
        counts_part[s['label_particle']] = counts_part.get(s['label_particle'], 0) + 1

    def make_weight(counts, nc):
        freqs = np.array([counts.get(k, 1) for k in range(nc)], dtype=float)
        w = 1.0 / (freqs + 1e-6)
        return torch.tensor(w / w.mean(), dtype=torch.float).to(device)

    gas_weight  = make_weight(counts_gas,  config.num_classes_gas)
    part_weight = make_weight(counts_part, config.num_classes_particle)

    if config.use_class_weights:
        criterion_gas      = nn.CrossEntropyLoss(weight=gas_weight,  label_smoothing=config.label_smoothing)
        criterion_particle = nn.CrossEntropyLoss(weight=part_weight, label_smoothing=config.label_smoothing)
    else:
        criterion_gas      = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        criterion_particle = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    # ---- 优化器（三组学习率）----
    mamba_ids, head_ids = set(), set()
    mamba_params, head_params = [], []

    for name, p in raw_model.named_parameters():
        if 'temporal_mamba' in name or ('mamba' in name.lower() and 'norm' not in name):
            mamba_params.append(p); mamba_ids.add(id(p))
    for m in [raw_model.head_gas, raw_model.head_particle, raw_model.head_dropout]:
        for p in m.parameters():
            head_params.append(p); head_ids.add(id(p))
    backbone_params = [p for p in raw_model.parameters()
                       if id(p) not in mamba_ids and id(p) not in head_ids]

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

    # ---- 训练循环 ----
    best_val_acc, best_epoch, no_improve = 0.0, 0, 0
    best_model_path = os.path.join(run_dir, "best_model.pth") if is_main() else None
    val_interval = max(1, int(getattr(config, 'val_interval', 1)))

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   [],
    }

    total_start = time.time()

    for epoch in range(config.num_epochs):
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        t0 = time.time()
        train_loss, train_both, train_gas, train_part = train_epoch(
            model, train_loader, criterion_gas, criterion_particle,
            optimizer, device, scaler, config.grad_accum_steps, config
        )
        should_validate = ((epoch + 1) % val_interval == 0) or (epoch + 1 == config.num_epochs)
        if should_validate:
            val_loss, val_both, val_gas, val_part = validate(
                model, val_loader, criterion_gas, criterion_particle, device, config
            )
        else:
            val_loss = val_both = val_gas = val_part = None
        scheduler.step()

        if is_main():
            elapsed = time.time() - t0
            bb_lr = optimizer.param_groups[0]['lr']

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_both)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_both)

            print(f"\n📌 Epoch [{epoch+1}/{config.num_epochs}] ({elapsed/60:.1f}分) LR={bb_lr:.2e}")
            print(f"  Train | Loss:{train_loss:.4f} | Both:{train_both:.1f}% | "
                  f"Gas:{train_gas:.1f}% | Part:{train_part:.1f}%")
            if should_validate:
                gap = train_both - val_both
                print(f"  Val   | Loss:{val_loss:.4f}  | Both:{val_both:.1f}% | "
                      f"Gas:{val_gas:.1f}% | Part:{val_part:.1f}%")
                print(f"  Gap   | {gap:.1f}% {'⚠️ 过拟合' if gap > 15 else '✅ 正常'}")
            else:
                print(f"  Val   | 跳过 (val_interval={val_interval})")

            if should_validate and val_both > best_val_acc:
                best_val_acc = val_both; best_epoch = epoch + 1; no_improve = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': raw_model.state_dict(),
                    'val_acc_both': val_both,
                    'val_acc_gas':  val_gas,
                    'val_acc_particle': val_part,
                    'history': history,
                    'config':  config.to_dict(),
                    'run_name': run_name,
                }, best_model_path)
                print(f"  ✅ 保存最佳模型 (Val Both: {val_both:.1f}%) → {best_model_path}")
            elif should_validate:
                no_improve += 1
                print(f"  ⏸️  {no_improve}/{config.patience}, 最佳 Epoch {best_epoch} ({best_val_acc:.1f}%)")

            if should_validate and no_improve >= config.patience:
                print("🛑 早停!")
                break

    if is_main():
        total_time = time.time() - total_start
        print(f"\n🏁 训练完成！最佳 Val Both: {best_val_acc:.1f}% (Epoch {best_epoch})")
        print(f"   总时间: {total_time/3600:.1f} 小时")
        print(f"   📁 运行目录: {run_dir}")

        # summary.txt
        summary_path = os.path.join(run_dir, "summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"训练总结\n运行名称: {run_name}\n")
            f.write(f"GPU 数量: {dist.get_world_size()}\n")
            f.write(f"总训练时间: {total_time/3600:.1f} 小时\n")
            f.write(f"最佳 Val Acc: {best_val_acc:.1f}% (Epoch {best_epoch})\n")
            f.write(f"\n训练历史:\n")
            for i in range(len(history['train_loss'])):
                val_acc_str = 'skip' if history['val_acc'][i] is None else f"{history['val_acc'][i]:.1f}%"
                f.write(f"  {i+1:3d} | train={history['train_acc'][i]:.1f}% "
                        f"val={val_acc_str}\n")

        sys.stdout = logger.terminal
        sys.stderr = logger.terminal
        logger.close()
        print(f"\n✅ 所有文件已保存到: {run_dir}")

    cleanup_ddp()


if __name__ == '__main__':
    main()
