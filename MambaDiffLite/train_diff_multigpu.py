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


warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.checkpoint")
warnings.filterwarnings("ignore", message=".*flash attention.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash.*", category=UserWarning)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")


PRESETS = {
    "lite_2gpu_fast": {
        "img_size": 224,
        "num_frames": 16,
        "batch_size": 4,
        "grad_accum_steps": 1,
        "backbone_lr": 3e-5,
        "mamba_lr": 1e-4,
        "head_lr": 1e-3,
        "head_dropout": 0.2,
        "motion_head_dropout": 0.2,
        "fusion_dropout": 0.08,
        "label_smoothing": 0.05,
        "weight_decay": 5e-4,
        "motion_input_scale": 2.5,
        "motion_residual_weight": 0.3,
        "motion_depth": 2,
        "combo_loss_weight": 1.0,
        "gas_aux_loss_weight": 0.3,
        "num_epochs": 40,
        "warmup_epochs": 5,
        "patience": 6,
        "val_interval": 1,
        "num_workers": 12,
        "train_ratio": 0.8,
        "max_images_per_group": None,
        "use_checkpoint": False,
    }
}


def _parse_bool(value):
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid bool value: {value}")


def _cast_override(current_value, raw_value):
    if current_value is None:
        lowered = raw_value.lower()
        if lowered == "none":
            return None
        try:
            return int(raw_value)
        except ValueError:
            try:
                return float(raw_value)
            except ValueError:
                return raw_value
    if isinstance(current_value, bool):
        return _parse_bool(raw_value)
    if isinstance(current_value, int):
        return int(raw_value)
    if isinstance(current_value, float):
        return float(raw_value)
    return raw_value


def apply_overrides(config, preset=None, overrides=None):
    if preset:
        for key, value in PRESETS[preset].items():
            setattr(config, key, value)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must use key=value syntax: {item}")
        key, raw_value = item.split("=", 1)
        if not hasattr(config, key):
            raise AttributeError(f"Unknown config key: {key}")
        setattr(config, key, _cast_override(getattr(config, key), raw_value))
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Windows-friendly multi-GPU training for MambaDiffLite.")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default="lite_2gpu_fast")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def setup_ddp():
    required = ["RANK", "WORLD_SIZE", "LOCAL_RANK"]
    if not all(name in os.environ for name in required):
        local_rank = 0
        torch.cuda.set_device(local_rank)
        return local_rank, "single"

    if os.name == "nt":
        os.environ.setdefault("USE_LIBUV", "0")
    backend = "gloo" if os.name == "nt" else ("nccl" if dist.is_nccl_available() else "gloo")
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, backend


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


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


def create_run_dir(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"run_combo_diff_ddp_{timestamp}"
        f"_f{config.num_frames}"
        f"_bs{config.batch_size}x{config.grad_accum_steps}"
        f"_cls{config.num_classes_combo}"
        f"_hlr{config.head_lr:.0e}"
        f"_{world_size()}gpu"
    )
    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_name


def save_params(config, run_dir, run_name, backend):
    params = config.to_dict()
    params["world_size"] = world_size()
    params["backend"] = backend
    params["global_batch_size"] = config.batch_size * config.grad_accum_steps * world_size()
    txt_path = os.path.join(run_dir, "params.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("MambaDiffLite multi-GPU config\n")
        f.write(f"run_name: {run_name}\n")
        f.write(f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        for key, value in sorted(params.items()):
            f.write(f"{key}: {value}\n")

    json_path = os.path.join(run_dir, "params.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    return txt_path


def build_loader(config, txt_file, split):
    dataset = ComboTxtDataset(
        txt_file=txt_file,
        num_frames=config.num_frames,
        img_size=config.img_size,
        split=split,
        split_ratio=config.train_ratio,
        max_images_per_group=config.max_images_per_group,
    )
    sampler = None
    if split == "train" and dist.is_initialized():
        sampler = DistributedSampler(dataset, num_replicas=world_size(), rank=dist.get_rank(), shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=(split == "train" and sampler is None),
        num_workers=max(0, config.num_workers),
        pin_memory=True,
        drop_last=(split == "train"),
        persistent_workers=(config.num_workers > 0),
        prefetch_factor=4 if config.num_workers > 0 else None,
        worker_init_fn=worker_init_fn if config.num_workers > 0 else None,
    )
    return loader, sampler


def build_optimizer(model, config):
    mamba_ids, head_ids = set(), set()
    mamba_params, head_params = [], []

    for name, param in model.named_parameters():
        if "temporal_mamba" in name or ("mamba" in name.lower() and "norm" not in name):
            mamba_params.append(param)
            mamba_ids.add(id(param))

    for module in [
        model.appearance_dropout,
        model.motion_dropout,
        model.residual_dropout,
        model.fusion_gate,
        model.motion_residual,
        model.head_cls,
        model.head_gas_aux,
    ]:
        for param in module.parameters():
            head_params.append(param)
            head_ids.add(id(param))

    backbone_params = [
        param for param in model.parameters() if id(param) not in mamba_ids and id(param) not in head_ids
    ]
    return optim.AdamW(
        [
            {"params": backbone_params, "lr": config.backbone_lr},
            {"params": mamba_params, "lr": config.mamba_lr},
            {"params": head_params, "lr": config.head_lr},
        ],
        weight_decay=config.weight_decay,
    )


def build_scheduler(optimizer, config):
    def lr_lambda(epoch):
        if epoch < config.warmup_epochs:
            return (epoch + 1) / config.warmup_epochs
        progress = (epoch - config.warmup_epochs) / max(1, config.num_epochs - config.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def reduce_triplet(loss_sum, correct, total, device):
    if not dist.is_initialized():
        total = max(total, 1.0)
        return loss_sum / total, 100.0 * correct / total
    stats = torch.tensor([loss_sum, correct, total], device=device, dtype=torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    total = max(stats[2].item(), 1.0)
    return stats[0].item() / total, 100.0 * stats[1].item() / total


def train_epoch(model, dataloader, criterion_combo, criterion_gas, optimizer, device, scaler, config):
    model.train()
    loss_sum = correct = total = 0.0
    optimizer.zero_grad(set_to_none=True)

    for inputs, labels in dataloader:
        frames = inputs["frames"].to(device, non_blocking=True)
        diff_frames = inputs["diff_frames"].to(device, non_blocking=True)
        label_combo = labels["combo"].to(device, non_blocking=True)
        label_gas = labels["gas"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            combo_logits, gas_logits = model(frames, diff_frames)
            loss = (
                config.combo_loss_weight * criterion_combo(combo_logits, label_combo)
                + config.gas_aux_loss_weight * criterion_gas(gas_logits, label_gas)
            ) / config.grad_accum_steps

        scaler.scale(loss).backward()
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

    return reduce_triplet(loss_sum, correct, total, device)


def validate(model, dataloader, criterion_combo, criterion_gas, device, config):
    model.eval()
    loss_sum = correct = total = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            frames = inputs["frames"].to(device, non_blocking=True)
            diff_frames = inputs["diff_frames"].to(device, non_blocking=True)
            label_combo = labels["combo"].to(device, non_blocking=True)
            label_gas = labels["gas"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                combo_logits, gas_logits = model(frames, diff_frames)
                loss = (
                    config.combo_loss_weight * criterion_combo(combo_logits, label_combo)
                    + config.gas_aux_loss_weight * criterion_gas(gas_logits, label_gas)
                )
            preds = combo_logits.argmax(1)
            bs = frames.size(0)
            loss_sum += loss.item() * bs
            correct += (preds == label_combo).sum().item()
            total += bs
    return loss_sum / max(total, 1.0), 100.0 * correct / max(total, 1.0)


def main():
    args = parse_args()
    local_rank, backend = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    config = apply_overrides(MambaSformerComboConfig(), preset=args.preset, overrides=args.set)

    run_dir = run_name = params_txt = log_path = None
    logger = None
    if is_main():
        run_dir, run_name = create_run_dir(config)
        params_txt = save_params(config, run_dir, run_name, backend)
        log_path = os.path.join(run_dir, "training_log.txt")
        logger = DualLogger(log_path)
        sys.stdout = logger
        sys.stderr = logger

    if dist.is_initialized():
        shared = [run_dir, run_name, params_txt, log_path]
        dist.broadcast_object_list(shared, src=0)
        run_dir, run_name, params_txt, log_path = shared

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
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    txt_file = "alltrain_with_c.txt"
    train_loader, train_sampler = build_loader(config, txt_file, "train")
    val_loader = None
    if is_main():
        val_loader, _ = build_loader(config, txt_file, "val")

    criterion_combo = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    criterion_gas = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = build_optimizer(raw_model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler("cuda")

    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0
    best_model_path = os.path.join(run_dir, "best_model.pth") if is_main() else None
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    total_start = time.time()

    if is_main():
        print("=" * 60)
        print("MambaDiffLite multi-GPU training")
        print("=" * 60)
        print(f"run_dir: {run_dir}")
        print(f"params:  {params_txt}")
        print(f"preset:  {args.preset}")
        print(f"backend: {backend}")
        print(f"world_size: {world_size()}")
        print(f"global_batch_size: {config.batch_size * config.grad_accum_steps * world_size()}")
        print(f"use_checkpoint: {config.use_checkpoint}")
        print()

    for epoch in range(config.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        start = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion_combo, criterion_gas, optimizer, device, scaler, config
        )
        scheduler.step()

        val_loss = None
        val_acc = None
        if is_main():
            val_loss, val_acc = validate(raw_model, val_loader, criterion_combo, criterion_gas, device, config)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            minutes = (time.time() - start) / 60.0
            print(
                f"Epoch [{epoch + 1}/{config.num_epochs}] "
                f"time={minutes:.1f}min train_acc={train_acc:.2f}% val_acc={val_acc:.2f}%"
            )

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
                        "history": history,
                        "config": config.to_dict(),
                        "run_name": run_name,
                    },
                    best_model_path,
                )
            else:
                no_improve += 1

        if dist.is_initialized():
            stop_flag = torch.tensor([1 if (is_main() and no_improve >= config.patience) else 0], device=device)
            dist.broadcast(stop_flag, src=0)
            if stop_flag.item():
                break
        elif no_improve >= config.patience:
            break

    if is_main():
        total_time = time.time() - total_start
        print(f"Best val acc: {best_val_acc:.2f}% @ epoch {best_epoch}")
        print(f"Total hours: {total_time / 3600:.2f}")
        summary_path = os.path.join(run_dir, "summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("MambaDiffLite multi-GPU summary\n")
            f.write(f"run_name: {run_name}\n")
            f.write(f"best_val_acc: {best_val_acc:.2f}\n")
            f.write(f"best_epoch: {best_epoch}\n")
            f.write(f"total_time_hours: {total_time / 3600:.2f}\n")

        sys.stdout = logger.terminal
        sys.stderr = logger.terminal
        logger.close()

    cleanup_ddp()


if __name__ == "__main__":
    main()
