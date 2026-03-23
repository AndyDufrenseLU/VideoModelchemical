import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


class ComboTxtDataset(Dataset):
    """Dataset for 38-class combo classification with an auxiliary gas label."""

    def __init__(
        self,
        txt_file,
        num_frames=8,
        img_size=224,
        split="train",
        split_ratio=0.75,
        max_images_per_group=800,
    ):
        self.num_frames = num_frames
        self.img_size = img_size
        self.split = split
        self.split_ratio = split_ratio
        self.max_images_per_group = max_images_per_group
        self.is_train = split == "train"

        self.cache_dir_square = f"image_cache_{img_size}"
        self.cache_dir_ratio = f"image_cache_{img_size}w"

        if os.path.isdir(self.cache_dir_square):
            self.cache_dir = self.cache_dir_square
            self.cache_mode = "square"
            cache_count = len(list(Path(self.cache_dir).glob("*.npy")))
            print(f"  使用全图缓存: {self.cache_dir}/ ({cache_count}个)")
        elif os.path.isdir(self.cache_dir_ratio):
            self.cache_dir = self.cache_dir_ratio
            self.cache_mode = "ratio"
            cache_count = len(list(Path(self.cache_dir).glob("*.npy")))
            print(f"  使用比例缓存(自动resize): {self.cache_dir}/ ({cache_count}个)")
        else:
            self.cache_dir = None
            self.cache_mode = None
            print("  无缓存，从原始图片加载(较慢)")

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.samples = []
        self.combo_to_idx = {}
        self.idx_to_combo = {}
        self.num_classes_combo = 0
        self._load_txt_file(txt_file)

    def _load_txt_file(self, txt_file):
        print(f"Loading data from {txt_file}...")

        all_data = []
        skipped = 0
        with open(txt_file, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or line.startswith("the"):
                    skipped += 1
                    continue

                parts = line.rsplit(maxsplit=2)
                if len(parts) < 3:
                    skipped += 1
                    continue

                try:
                    label_particle = int(parts[1])
                    label_gas = int(parts[2]) - 1
                except ValueError:
                    skipped += 1
                    continue

                if label_gas < 0:
                    skipped += 1
                    continue

                all_data.append(
                    {
                        "path": parts[0],
                        "idx": len(all_data),
                        "label_particle": label_particle,
                        "label_gas": label_gas,
                    }
                )

        print(f"  读取 {len(all_data)} 张图像 (跳过{skipped}行)")

        missing = []
        for item in all_data:
            if not os.path.exists(item["path"]):
                missing.append(item["path"])
                if len(missing) >= 5:
                    break
        if missing:
            print("  检测到缺失图片路径，示例:")
            for path in missing:
                print(f"    - {path}")

        groups = []
        if all_data:
            cur_group = [all_data[0]]
            cur_key = (all_data[0]["label_particle"], all_data[0]["label_gas"])
            for item in all_data[1:]:
                key = (item["label_particle"], item["label_gas"])
                if key == cur_key:
                    cur_group.append(item)
                else:
                    groups.append((cur_key, cur_group))
                    cur_group = [item]
                    cur_key = key
            groups.append((cur_key, cur_group))

        combo_keys = [key for key, _ in groups]
        self.combo_to_idx = {key: idx for idx, key in enumerate(combo_keys)}
        self.idx_to_combo = {idx: key for key, idx in self.combo_to_idx.items()}
        self.num_classes_combo = len(self.combo_to_idx)

        print(f"  共 {len(groups)} 个连续label组")
        print(f"  粒子标签: {sorted({k[0] for k in combo_keys})} ({len({k[0] for k in combo_keys})}类)")
        print(f"  气速标签: {sorted({k[1] for k in combo_keys})} ({len({k[1] for k in combo_keys})}类, 原始值+1)")
        print(f"  组合类别: {self.num_classes_combo} 类")

        if self.max_images_per_group is not None:
            groups = [(key, grp[: self.max_images_per_group]) for key, grp in groups]
            total_images = sum(len(group) for _, group in groups)
            print(f"  每组限制 {self.max_images_per_group} 张图 (共 {len(groups)} 组, 总图片数 {total_images})")

        total_samples = 0
        for combo_key, group in groups:
            combo_label = self.combo_to_idx[combo_key]
            split_idx = max(self.num_frames, int(len(group) * self.split_ratio))
            split_idx -= split_idx % self.num_frames
            if split_idx <= 0:
                split_idx = self.num_frames
            if split_idx >= len(group):
                split_idx = len(group) - (len(group) % self.num_frames)
                split_idx = max(self.num_frames, split_idx - self.num_frames)

            selected_group = group[:split_idx] if self.split == "train" else group[split_idx:]
            usable = len(selected_group) // self.num_frames
            for i in range(usable):
                start = i * self.num_frames
                end = (i + 1) * self.num_frames
                frames = selected_group[start:end]
                self.samples.append(
                    {
                        "frame_indices": [f["idx"] for f in frames],
                        "frame_paths": [f["path"] for f in frames],
                        "label_combo": combo_label,
                        "label_particle": combo_key[0],
                        "label_gas": combo_key[1],
                    }
                )
                total_samples += 1

        print(f"  总样本数(每{self.num_frames}帧, 无滑窗): {total_samples}")
        print(f"  {self.split}集: {len(self.samples)} 样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = []
        error_count = 0

        for i, global_idx in enumerate(sample["frame_indices"]):
            try:
                if self.cache_dir and self.cache_mode == "square":
                    cache_path = os.path.join(self.cache_dir, f"{global_idx:06d}.npy")
                    arr = np.load(cache_path)
                elif self.cache_dir and self.cache_mode == "ratio":
                    cache_path = os.path.join(self.cache_dir, f"{global_idx:06d}.npy")
                    arr = np.load(cache_path)
                    img = Image.fromarray(arr)
                    img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
                    arr = np.array(img, dtype=np.uint8)
                else:
                    img = Image.open(sample["frame_paths"][i]).convert("RGB")
                    img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
                    arr = np.array(img, dtype=np.uint8)

                tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
                frames.append(tensor)
            except Exception as exc:
                error_count += 1
                if error_count <= 1:
                    print(f"  加载失败 idx={global_idx}: {exc}")
                frames.append(torch.zeros(3, self.img_size, self.img_size))

        frames = torch.stack(frames, dim=0)

        if self.is_train:
            if torch.rand(1).item() < 0.5:
                frames = frames.flip(-1)
            if torch.rand(1).item() < 0.5:
                brightness = 0.9 + torch.rand(1).item() * 0.2
                frames = frames.mul(brightness).clamp(0, 1)

        for t in range(frames.size(0)):
            frames[t] = self.normalize(frames[t])

        return frames, {
            "combo": sample["label_combo"],
            "gas": sample["label_gas"],
        }


def worker_init_fn(worker_id):
    np.random.seed(worker_id)


def get_combo_dataloader(
    txt_file,
    num_frames=8,
    img_size=224,
    batch_size=16,
    split="train",
    num_workers=4,
    train_ratio=0.75,
    max_images_per_group=800,
):
    dataset = ComboTxtDataset(
        txt_file=txt_file,
        num_frames=num_frames,
        img_size=img_size,
        split=split,
        split_ratio=train_ratio,
        max_images_per_group=max_images_per_group,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=max(0, num_workers),
        pin_memory=True,
        drop_last=(split == "train"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )
