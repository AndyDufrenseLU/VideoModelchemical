import argparse
import json
import math
import os
import sys
import time
import warnings
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from config_combo import MambaSformerComboConfig
from dataset_combo import get_combo_dataloader
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
    "baseline_all_20260316": {
        "img_size": 224,
        "num_frames": 16,
        "batch_size": 8,
        "grad_accum_steps": 1,
        "backbone_lr": 3e-5,
        "mamba_lr": 1e-4,
        "head_lr": 1e-3,
        "head_dropout": 0.2,
        "motion_head_dropout": 0.25,
        "fusion_dropout": 0.1,
        "label_smoothing": 0.05,
        "weight_decay": 5e-4,
        "motion_input_scale": 3.0,
        "motion_residual_weight": 0.35,
        "motion_dim": 128,
        "motion_kernel_size": 3,
        "combo_loss_weight": 1.0,
        "gas_aux_loss_weight": 0.3,
        "num_epochs": 40,
        "warmup_epochs": 5,
        "val_interval": 1,
        "num_workers": 8,
        "train_ratio": 0.8,
        "max_images_per_group": None,
        "use_checkpoint": True,
    },
    "anti_overfit_noaug": {
        "img_size": 224,
        "num_frames": 16,
        "batch_size": 8,
        "grad_accum_steps": 1,
        "backbone_lr": 2e-5,
        "mamba_lr": 7e-5,
        "head_lr": 5e-4,
        "dropout": 0.1,
        "attention_dropout": 0.05,
        "head_dropout": 0.35,
        "motion_head_dropout": 0.35,
        "fusion_dropout": 0.2,
        "label_smoothing": 0.08,
        "weight_decay": 1.2e-3,
        "motion_input_scale": 2.5,
        "motion_residual_weight": 0.25,
        "motion_dim": 96,
        "motion_kernel_size": 3,
        "combo_loss_weight": 1.0,
        "gas_aux_loss_weight": 0.25,
        "num_epochs": 50,
        "warmup_epochs": 5,
        "val_interval": 1,
        "num_workers": 12,
        "train_ratio": 0.8,
        "max_images_per_group": None,
        "use_checkpoint": False,
    },
    "route1": {
        "img_size": 224,
        "num_frames": 16,
        "batch_size": 8,
        "grad_accum_steps": 1,
        "backbone_lr": 3e-5,
        "mamba_lr": 1e-4,
        "head_lr": 1e-3,
        "head_dropout": 0.2,
        "motion_head_dropout": 0.25,
        "fusion_dropout": 0.1,
        "label_smoothing": 0.05,
        "weight_decay": 5e-4,
        "motion_input_scale": 3.0,
        "motion_residual_weight": 0.35,
        "motion_dim": 128,
        "motion_kernel_size": 3,
        "num_epochs": 40,
        "warmup_epochs": 5,
        "val_interval": 1,
        "max_images_per_group": 800,
        "use_checkpoint": True,
    },
    "route2": {
        "img_size": 320,
        "num_frames": 16,
        "batch_size": 4,
        "grad_accum_steps": 2,
        "backbone_lr": 3e-5,
        "mamba_lr": 1e-4,
        "head_lr": 1e-3,
        "head_dropout": 0.2,
        "motion_head_dropout": 0.25,
        "fusion_dropout": 0.1,
        "label_smoothing": 0.05,
        "weight_decay": 5e-4,
        "motion_input_scale": 3.0,
        "motion_residual_weight": 0.35,
        "motion_dim": 128,
        "motion_kernel_size": 3,
        "num_epochs": 40,
        "warmup_epochs": 5,
        "val_interval": 1,
        "max_images_per_group": 800,
        "use_checkpoint": True,
        "num_workers": 6,
    },
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
        current_value = getattr(config, key)
        setattr(config, key, _cast_override(current_value, raw_value))

    return config


def parse_args():
    parser = argparse.ArgumentParser(description="训练 38 类组合分类模型，可通过命令行覆盖超参数。")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()))
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="覆盖任意配置项，例如 --set batch_size=8 --set img_size=320",
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


def create_run_dir(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"run_combo_diff_{timestamp}"
        f"_f{config.num_frames}"
        f"_bs{config.batch_size}x{config.grad_accum_steps}"
        f"_cls{config.num_classes_combo}"
        f"_hlr{config.head_lr:.0e}"
    )
    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_name


def save_params(config, run_dir, run_name):
    params = config.to_dict()
    txt_path = os.path.join(run_dir, "params.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("MambaDiffMotion 38 类训练配置\n")
        f.write(f"run_name: {run_name}\n")
        f.write(f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        for key, value in sorted(params.items()):
            f.write(f"{key}: {value}\n")
        f.write(f"effective_batch_size: {config.batch_size * config.grad_accum_steps}\n")

    json_path = os.path.join(run_dir, "params.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    return txt_path, json_path


def gpu_diagnostic():
    print("=" * 60)
    print("GPU 信息")
    print("=" * 60)
    if not torch.cuda.is_available():
        print("CUDA 不可用，回退到 CPU。")
        return torch.device("cpu")

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {props.total_memory / 1024**3:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print()
    return device


def build_optimizer(model, config):
    mamba_ids, head_ids = set(), set()
    mamba_params, head_params = [], []

    for name, param in model.named_parameters():
        if "temporal_mamba" in name or ("mamba" in name.lower() and "norm" not in name):
            mamba_params.append(param)
            mamba_ids.add(id(param))

    for module in [
        model.motion_stem,
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

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": config.backbone_lr},
            {"params": mamba_params, "lr": config.mamba_lr},
            {"params": head_params, "lr": config.head_lr},
        ],
        weight_decay=config.weight_decay,
    )

    print("\n参数分组:")
    print(f"  主干网络: {config.backbone_lr} ({sum(p.numel() for p in backbone_params):,} 个参数)")
    print(f"  Mamba:    {config.mamba_lr} ({sum(p.numel() for p in mamba_params):,} 个参数)")
    print(f"  任务头:   {config.head_lr} ({sum(p.numel() for p in head_params):,} 个参数)")
    return optimizer


def build_scheduler(optimizer, config):
    def lr_lambda(epoch):
        if epoch < config.warmup_epochs:
            return (epoch + 1) / config.warmup_epochs
        progress = (epoch - config.warmup_epochs) / max(1, config.num_epochs - config.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, dataloader, criterion_combo, criterion_gas, optimizer, device, scaler, config):
    model.train()
    total_loss = correct = total = 0
    epoch_start = time.time()
    num_batches = len(dataloader)
    optimizer.zero_grad()

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        batch_start = time.time()
        frames = inputs["frames"].to(device, non_blocking=True)
        diff_frames = inputs["diff_frames"].to(device, non_blocking=True)
        label_combo = labels["combo"].to(device, non_blocking=True)
        label_gas = labels["gas"].to(device, non_blocking=True)

        if batch_idx == 0:
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f"  数据设备: frames={frames.device} shape={frames.shape}")
            print(f"  GPU显存: {allocated:.2f} GB")

        with torch.amp.autocast("cuda"):
            combo_logits, gas_logits = model(frames, diff_frames)
            loss_combo = criterion_combo(combo_logits, label_combo)
            loss_gas = criterion_gas(gas_logits, label_gas)
            loss = (
                config.combo_loss_weight * loss_combo
                + config.gas_aux_loss_weight * loss_gas
            ) / config.grad_accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % config.grad_accum_steps == 0 or (batch_idx + 1) == num_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        with torch.no_grad():
            preds = combo_logits.argmax(1)
            bs = frames.size(0)
            total_loss += loss.item() * config.grad_accum_steps * bs
            correct += (preds == label_combo).sum().item()
            total += bs

        if batch_idx == 0:
            print(f"  [Batch 1] 完成，用时: {time.time() - batch_start:.1f}秒")

        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == num_batches:
            elapsed = time.time() - epoch_start
            avg_t = elapsed / (batch_idx + 1)
            remain = avg_t * (num_batches - batch_idx - 1)
            print(
                f"  [{batch_idx + 1}/{num_batches}] "
                f"Loss:{total_loss / total:.4f} | "
                f"Acc:{100 * correct / total:.1f}% | "
                f"{avg_t:.1f}s/batch | 剩余:{remain / 60:.0f}分钟"
            )

    return total_loss / total, 100 * correct / total, time.time() - epoch_start


def validate(model, dataloader, criterion_combo, criterion_gas, device, config):
    model.eval()
    total_loss = correct = total = 0
    start = time.time()

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
            total_loss += loss.item() * bs
            correct += (preds == label_combo).sum().item()
            total += bs

    return total_loss / total, 100 * correct / total, time.time() - start


def main():
    args = parse_args()
    config = MambaSformerComboConfig()
    config = apply_overrides(config, preset=args.preset, overrides=args.set)
    run_dir, run_name = create_run_dir(config)
    params_txt, _ = save_params(config, run_dir, run_name)

    log_path = os.path.join(run_dir, "training_log.txt")
    logger = DualLogger(log_path)
    sys.stdout = logger
    sys.stderr = logger

    print("=" * 60)
    print("MambaDiffMotion 38 类双分支训练")
    print("=" * 60)
    print(f"运行目录: {run_dir}")
    print(f"参数文件: {params_txt}")
    print(f"日志文件: {log_path}")
    print(f"配置:     {config}")
    print()

    device = gpu_diagnostic()
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    print("正在创建模型...")
    model = create_mambasformer_combo(config)

    ts_weights = "TimeSformer_divST_8x32_224_K400.pyth"
    fallback_ts_weights = "TimeSformer_divST_16x16_448_K400.pyth"
    vit_weights = "vit_b_16-c867db91.pth"

    if os.path.exists(ts_weights):
        print(f"加载预训练 TimeSformer 权重: {ts_weights}")
        model = load_pretrained_timesformer(model, ts_weights, config)
    elif os.path.exists(fallback_ts_weights):
        print(f"加载备用 TimeSformer 权重: {fallback_ts_weights}")
        model = load_pretrained_timesformer(model, fallback_ts_weights, config)
    elif os.path.exists(vit_weights):
        print(f"加载预训练 ViT-B/16 权重: {vit_weights}")
        model = load_pretrained_vit(model, vit_weights)
    else:
        print("未找到预训练权重，使用随机初始化。")

    model = model.to(device)
    raw_model = model

    if config.use_multi_gpu and gpu_count > 1:
        model = nn.DataParallel(model, device_ids=list(range(gpu_count)))
        print(f"使用 DataParallel，GPU 数量: {gpu_count}")

    print(f"可训练参数量: {sum(p.numel() for p in raw_model.parameters() if p.requires_grad):,}")
    print(f"主头类别数: {config.num_classes_combo}")
    print(f"Gas辅助头类别数: {config.num_classes_gas}")
    print("模型结构: 重型外观主干 + 轻量差分运动分支 + 门控融合")
    print()

    txt_file = "alltrain_with_c.txt"
    if not os.path.exists(txt_file):
        print(f"缺少数据索引文件: {txt_file}")
        sys.stdout = logger.terminal
        sys.stderr = logger.terminal
        logger.close()
        return

    print("正在加载数据...")
    train_loader = get_combo_dataloader(
        txt_file=txt_file,
        num_frames=config.num_frames,
        img_size=config.img_size,
        batch_size=config.batch_size,
        split="train",
        num_workers=config.num_workers,
        train_ratio=config.train_ratio,
        max_images_per_group=config.max_images_per_group,
    )
    val_loader = get_combo_dataloader(
        txt_file=txt_file,
        num_frames=config.num_frames,
        img_size=config.img_size,
        batch_size=config.batch_size,
        split="val",
        num_workers=config.num_workers,
        train_ratio=config.train_ratio,
        max_images_per_group=config.max_images_per_group,
    )

    train_ds = train_loader.dataset
    val_ds = val_loader.dataset
    if train_ds.num_classes_combo != config.num_classes_combo:
        raise RuntimeError(
            f"Expected {config.num_classes_combo} combo classes, got {train_ds.num_classes_combo} from {txt_file}"
        )

    print(f"训练样本: {len(train_ds)}")
    print(f"验证样本: {len(val_ds)}")
    print(f"批次大小: {config.batch_size} (有效: {config.batch_size * config.grad_accum_steps})")
    print(f"输入尺寸: {config.num_frames} 帧 × {config.img_size}×{config.img_size}")
    print(f"每 epoch 约 {len(train_loader)} 个 batch")
    print(f"38类映射: {train_ds.idx_to_combo}")
    print()

    combo_counts = {}
    gas_counts = {}
    for sample in train_ds.samples:
        combo_counts[sample["label_combo"]] = combo_counts.get(sample["label_combo"], 0) + 1
        gas_counts[sample["label_gas"]] = gas_counts.get(sample["label_gas"], 0) + 1

    print(f"训练集 38 类分布: {dict(sorted(combo_counts.items()))}")
    print(f"训练集 Gas 分布: {dict(sorted(gas_counts.items()))}")

    criterion_combo = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    criterion_gas = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = build_optimizer(raw_model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler("cuda")

    print("\nAMP 混合精度已启用")
    print(f"梯度累积: {config.grad_accum_steps}")
    print(f"梯度检查点: {config.use_checkpoint}")
    print(f"主头损失权重: {config.combo_loss_weight}")
    print(f"Gas辅助损失权重: {config.gas_aux_loss_weight}")
    print(f"Motion head dropout: {config.motion_head_dropout}")
    print(f"Fusion dropout: {config.fusion_dropout}")
    print(f"Motion input scale: {config.motion_input_scale}")
    print(f"Motion residual weight: {config.motion_residual_weight}")
    print(f"Motion dim: {config.motion_dim}")
    print(f"Motion kernel size: {config.motion_kernel_size}")
    print()

    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_model_path = os.path.join(run_dir, "best_model.pth")
    total_start = time.time()

    print("开始训练")
    print("=" * 60)

    for epoch in range(config.num_epochs):
        bb_lr = optimizer.param_groups[0]["lr"]
        mb_lr = optimizer.param_groups[1]["lr"]
        hd_lr = optimizer.param_groups[2]["lr"]

        print(f"\nEpoch [{epoch + 1}/{config.num_epochs}] (BB:{bb_lr:.2e}, Mamba:{mb_lr:.2e}, Head:{hd_lr:.2e})")
        print("-" * 60)

        train_loss, train_acc, train_time = train_epoch(
            model, train_loader, criterion_combo, criterion_gas, optimizer, device, scaler, config
        )
        val_loss, val_acc, val_time = validate(
            model, val_loader, criterion_combo, criterion_gas, device, config
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        gap = train_acc - val_acc
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3

        print(f"\n  Epoch {epoch + 1} (训练:{train_time / 60:.1f}分 + 验证:{val_time / 60:.1f}分)")
        print(f"  Train | Loss:{train_loss:.4f} | Acc:{train_acc:.1f}%")
        print(f"  Val   | Loss:{val_loss:.4f} | Acc:{val_acc:.1f}%")
        print(f"  Gap   | Train-Val: {gap:.1f}%")
        print(f"  GPU   | allocated={allocated:.2f} GB reserved={reserved:.2f} GB")

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
                },
                best_model_path,
            )
            print(f"  保存最佳模型 (Val Acc: {val_acc:.1f}%) -> {best_model_path}")
        else:
            no_improve += 1
            print(f"  无改进 ({no_improve}/{config.patience}), 最佳 Epoch {best_epoch} ({best_val_acc:.1f}%)")

        if no_improve >= config.patience:
            print(f"\n早停: 连续 {config.patience} 个 epoch 无改进")
            break

    total_time = time.time() - total_start
    final_val_acc = history["val_acc"][-1] if history["val_acc"] else None

    print("\n" + "=" * 60)
    print("训练完成")
    print(f"总训练时间: {total_time / 3600:.1f} 小时")
    print(f"最佳 Val Acc: {best_val_acc:.1f}% (Epoch {best_epoch})")
    print(f"最终 Train Acc: {history['train_acc'][-1]:.1f}%")
    if final_val_acc is not None:
        print(f"最终 Val Acc: {final_val_acc:.1f}%")
    print(f"运行目录: {run_dir}")
    print(f"模型文件: {best_model_path}")
    print("=" * 60)

    summary_path = os.path.join(run_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("38 类双分支差分运动训练摘要\n")
        f.write(f"run_name: {run_name}\n")
        f.write(f"total_time_hours: {total_time / 3600:.2f}\n")
        f.write(f"best_val_acc: {best_val_acc:.2f}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"combo_to_idx: {train_ds.combo_to_idx}\n\n")
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
    print(f"所有文件已保存到 {run_dir}")


if __name__ == "__main__":
    main()