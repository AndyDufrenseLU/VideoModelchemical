import argparse
import json
import math
import os
import sys
import time
import warnings
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config_combo import MambaSformerComboConfig
from dataset_combo import ComboTxtDataset, worker_init_fn
from mambasformer_combo import (
    create_mambasformer_combo,
    load_pretrained_timesformer,
    load_pretrained_vit,
)
from train_combo import PRESETS, apply_overrides


warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.checkpoint")
warnings.filterwarnings("ignore", message=".*flash attention.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash.*", category=UserWarning)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def parse_args():
    parser = argparse.ArgumentParser(description="DDP training for 38-class combo model.")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default="route4")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any config field, for example --set batch_size=6 --set use_checkpoint=True",
    )
    return parser.parse_args()


class DualLogger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = open(log_file, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def tune_config_for_ddp(config):
    if world_size() >= 2:
        config.num_workers = max(config.num_workers, 12)
        if config.img_size <= 224 and config.batch_size <= 8:
            config.use_checkpoint = False
    return config


def create_run_dir(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"run_combo_ddp_{timestamp}"
        f"_f{config.num_frames}"
        f"_bs{config.batch_size}x{config.grad_accum_steps}"
        f"_cls{config.num_classes_combo}"
        f"_hlr{config.head_lr:.0e}"
        f"_{world_size()}gpu"
    )
    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_name


def save_params(config, run_dir, run_name):
    params = config.to_dict()
    params["world_size"] = world_size()
    params["global_batch_size"] = config.batch_size * config.grad_accum_steps * world_size()
    txt_path = os.path.join(run_dir, "params.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("MambaSformer combo DDP training config\n")
        f.write(f"run_name: {run_name}\n")
        f.write(f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        for key, value in sorted(params.items()):
            f.write(f"{key}: {value}\n")

    json_path = os.path.join(run_dir, "params.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    return txt_path, json_path


def build_optimizer(model, config):
    mamba_ids, head_ids = set(), set()
    mamba_params, head_params = [], []

    for name, param in model.named_parameters():
        if "temporal_mamba" in name or ("mamba" in name.lower() and "norm" not in name):
            mamba_params.append(param)
            mamba_ids.add(id(param))

    for module in [model.head_cls, model.head_gas_aux, model.head_dropout]:
        for param in module.parameters():
            head_params.append(param)
            head_ids.add(id(param))

    backbone_params = [
        param for param in model.parameters() if id(param) not in mamba_ids and id(param) not in head_ids
    ]

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": config.backbone_lr},
            {"params": mamba_params, "lr": config.mamba_lr},
            {"params": head_params, "lr": config.head_lr},
        ],
        weight_decay=config.weight_decay,
    )
    return optimizer


def build_scheduler(optimizer, config):
    def lr_lambda(epoch):
        if epoch < config.warmup_epochs:
            return (epoch + 1) / config.warmup_epochs
        progress = (epoch - config.warmup_epochs) / max(1, config.num_epochs - config.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_loader(config, txt_file, split):
    is_train = split == "train"
    dataset = ComboTxtDataset(
        txt_file=txt_file,
        num_frames=config.num_frames,
        img_size=config.img_size,
        split=split,
        split_ratio=config.train_ratio,
        max_images_per_group=config.max_images_per_group,
        augment=config.augment if is_train else False,
        aug_crop_scale_min=config.aug_crop_scale_min,
        aug_crop_scale_max=config.aug_crop_scale_max,
        aug_hflip_prob=config.aug_hflip_prob,
        aug_affine_prob=config.aug_affine_prob,
        aug_degrees=config.aug_degrees,
        aug_translate=config.aug_translate,
        aug_brightness=config.aug_brightness,
        aug_contrast=config.aug_contrast,
        aug_blur_prob=config.aug_blur_prob,
        aug_blur_sigma_min=config.aug_blur_sigma_min,
        aug_blur_sigma_max=config.aug_blur_sigma_max,
        aug_noise_std=config.aug_noise_std,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size(), rank=dist.get_rank(), shuffle=is_train)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=max(0, config.num_workers),
        pin_memory=True,
        drop_last=is_train,
        persistent_workers=(config.num_workers > 0),
        prefetch_factor=4 if config.num_workers > 0 else None,
        worker_init_fn=worker_init_fn if config.num_workers > 0 else None,
    )
    return loader, sampler


def reduce_metrics(loss_sum, correct, total, device):
    stats = torch.tensor([loss_sum, correct, total], device=device, dtype=torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    total = max(stats[2].item(), 1.0)
    return stats[0].item() / total, 100.0 * stats[1].item() / total


def train_epoch(model, dataloader, criterion_combo, criterion_gas, optimizer, device, scaler, config):
    model.train()
    loss_sum = 0.0
    correct = 0.0
    total = 0.0
    num_batches = len(dataloader)
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (frames, labels) in enumerate(dataloader):
        frames = frames.to(device, non_blocking=True)
        label_combo = labels["combo"].to(device, non_blocking=True)
        label_gas = labels["gas"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            combo_logits, gas_logits = model(frames)
            loss_combo = criterion_combo(combo_logits, label_combo)
            loss_gas = criterion_gas(gas_logits, label_gas)
            loss = (
                config.combo_loss_weight * loss_combo +
                config.gas_aux_loss_weight * loss_gas
            ) / config.grad_accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % config.grad_accum_steps == 0 or (batch_idx + 1) == num_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            preds = combo_logits.argmax(1)
            bs = frames.size(0)
            loss_sum += loss.item() * config.grad_accum_steps * bs
            correct += (preds == label_combo).sum().item()
            total += bs

    return reduce_metrics(loss_sum, correct, total, device)


def validate(model, dataloader, criterion_combo, criterion_gas, device, config):
    model.eval()
    loss_sum = 0.0
    correct = 0.0
    total = 0.0

    with torch.no_grad():
        for frames, labels in dataloader:
            frames = frames.to(device, non_blocking=True)
            label_combo = labels["combo"].to(device, non_blocking=True)
            label_gas = labels["gas"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                combo_logits, gas_logits = model(frames)
                loss = (
                    config.combo_loss_weight * criterion_combo(combo_logits, label_combo) +
                    config.gas_aux_loss_weight * criterion_gas(gas_logits, label_gas)
                )

            preds = combo_logits.argmax(1)
            bs = frames.size(0)
            loss_sum += loss.item() * bs
            correct += (preds == label_combo).sum().item()
            total += bs

    return reduce_metrics(loss_sum, correct, total, device)


def main():
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    args = parse_args()
    config = tune_config_for_ddp(apply_overrides(MambaSformerComboConfig(), preset=args.preset, overrides=args.set))

    run_dir = None
    run_name = None
    params_txt = None
    log_path = None
    logger = None

    if is_main():
        run_dir, run_name = create_run_dir(config)
        params_txt, _ = save_params(config, run_dir, run_name)
        log_path = os.path.join(run_dir, "training_log.txt")
        logger = DualLogger(log_path)
        sys.stdout = logger
        sys.stderr = logger

    shared = [run_dir, run_name, params_txt, log_path]
    dist.broadcast_object_list(shared, src=0)
    run_dir, run_name, params_txt, log_path = shared

    if is_main():
        print("=" * 60)
        print("MambaSformer 38-class combo multi-GPU training")
        print("=" * 60)
        print(f"run_dir: {run_dir}")
        print(f"params:  {params_txt}")
        print(f"log:     {log_path}")
        print(f"preset:  {args.preset}")
        print(f"config:  {config}")
        print(f"world_size: {world_size()}")
        print(f"global_batch_size: {config.batch_size * config.grad_accum_steps * world_size()}")
        print()

    model = create_mambasformer_combo(config)

    ts_weights = "TimeSformer_divST_8x32_224_K400.pyth"
    fallback_ts_weights = "TimeSformer_divST_16x16_448_K400.pyth"
    vit_weights = "vit_b_16-c867db91.pth"
    if os.path.exists(ts_weights):
        model = load_pretrained_timesformer(model, ts_weights, config)
    elif os.path.exists(fallback_ts_weights):
        model = load_pretrained_timesformer(model, fallback_ts_weights, config)
    elif os.path.exists(vit_weights):
        model = load_pretrained_vit(model, vit_weights)

    model = model.to(device)
    raw_model = model
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    if is_main():
        print(f"Trainable params: {sum(p.numel() for p in raw_model.parameters() if p.requires_grad):,}")
        print(f"Main head classes: {config.num_classes_combo}")
        print(f"Aux gas classes:   {config.num_classes_gas}")
        print()

    txt_file = "alltrain_with_c.txt"
    if not os.path.exists(txt_file):
        if is_main():
            print(f"Missing dataset index file: {txt_file}")
        cleanup_ddp()
        return

    train_loader, train_sampler = build_loader(config, txt_file, split="train")
    val_loader, _ = build_loader(config, txt_file, split="val")

    train_ds = train_loader.dataset
    val_ds = val_loader.dataset
    if train_ds.num_classes_combo != config.num_classes_combo:
        raise RuntimeError(
            f"Expected {config.num_classes_combo} combo classes, got {train_ds.num_classes_combo} from {txt_file}"
        )

    criterion_combo = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    criterion_gas = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = build_optimizer(raw_model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler("cuda")

    if is_main():
        print("Loading data...")
        print(f"train samples: {len(train_ds)}")
        print(f"val samples:   {len(val_ds)}")
        print(f"per_gpu_batch: {config.batch_size}")
        print(f"global_batch:  {config.batch_size * config.grad_accum_steps * world_size()}")
        print(f"num_workers:   {config.num_workers}")
        print(f"use_checkpoint:{config.use_checkpoint}")
        print(
            "augment: "
            f"crop={config.aug_crop_scale_min:.2f}-{config.aug_crop_scale_max:.2f}, "
            f"hflip={config.aug_hflip_prob}, affine={config.aug_affine_prob}, "
            f"brightness={config.aug_brightness}, contrast={config.aug_contrast}, "
            f"blur={config.aug_blur_prob}, noise={config.aug_noise_std}"
        )
        print()

    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_model_path = os.path.join(run_dir, "best_model.pth")
    total_start = time.time()

    for epoch in range(config.num_epochs):
        train_sampler.set_epoch(epoch)
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion_combo, criterion_gas, optimizer, device, scaler, config
        )
        val_loss, val_acc = validate(model, val_loader, criterion_combo, criterion_gas, device, config)
        scheduler.step()

        if is_main():
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            gap = train_acc - val_acc
            elapsed = time.time() - epoch_start
            bb_lr = optimizer.param_groups[0]["lr"]
            mb_lr = optimizer.param_groups[1]["lr"]
            hd_lr = optimizer.param_groups[2]["lr"]

            print(f"\nEpoch [{epoch + 1}/{config.num_epochs}] ({elapsed/60:.1f} min)")
            print(f"LR | BB:{bb_lr:.2e} Mamba:{mb_lr:.2e} Head:{hd_lr:.2e}")
            print(f"Train | Loss:{train_loss:.4f} | Acc:{train_acc:.1f}%")
            print(f"Val   | Loss:{val_loss:.4f} | Acc:{val_acc:.1f}%")
            print(f"Gap   | Train-Val: {gap:.1f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                no_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "val_acc_combo": val_acc,
                        "train_acc_combo": train_acc,
                        "combo_to_idx": train_ds.combo_to_idx,
                        "idx_to_combo": train_ds.idx_to_combo,
                        "history": history,
                        "config": config.to_dict(),
                        "run_name": run_name,
                        "world_size": world_size(),
                    },
                    best_model_path,
                )
                print(f"Saved best model ({val_acc:.1f}%) -> {best_model_path}")
            else:
                no_improve += 1
                print(f"No improve ({no_improve}/{config.patience}), best epoch {best_epoch} ({best_val_acc:.1f}%)")

            if no_improve >= config.patience:
                print("Early stopping triggered.")
                break

        stop_now = torch.tensor(1 if (is_main() and no_improve >= config.patience) else 0, device=device)
        dist.broadcast(stop_now, src=0)
        if stop_now.item():
            break

    if is_main():
        total_time = time.time() - total_start
        print("\n" + "=" * 60)
        print("Training finished")
        print(f"best val acc: {best_val_acc:.1f}% (epoch {best_epoch})")
        print(f"total time:   {total_time / 3600:.1f} hours")
        print(f"run_dir:      {run_dir}")
        print("=" * 60)

        summary_path = os.path.join(run_dir, "summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("38-class combo DDP training summary\n")
            f.write(f"run_name: {run_name}\n")
            f.write(f"world_size: {world_size()}\n")
            f.write(f"best_val_acc: {best_val_acc:.2f}\n")
            f.write(f"best_epoch: {best_epoch}\n")
            for i in range(len(history["train_loss"])):
                f.write(
                    f"{i + 1:03d} "
                    f"train_loss={history['train_loss'][i]:.4f} "
                    f"train_acc={history['train_acc'][i]:.2f}% "
                    f"val_loss={history['val_loss'][i]:.4f} "
                    f"val_acc={history['val_acc'][i]:.2f}%\n"
                )

        sys.stdout = logger.terminal
        sys.stderr = logger.terminal
        logger.close()

    cleanup_ddp()


if __name__ == "__main__":
    main()
