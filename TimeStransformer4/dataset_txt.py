"""
Dataset module - 流化床图片数据加载
优先使用 image_cache.dat memmap 缓存（若存在），否则回退到直接 PIL 加载。
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# Module-level constants: reused every __getitem__ call, never reallocated
_NORM_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
_NORM_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


class TxtFileDataset(Dataset):
    """流化床图像数据集 - memmap缓存优先，PIL回退"""

    def __init__(self, txt_file, num_frames=16, img_width=448, img_height=448,
                 transform=None, split='train', split_ratio=0.80):
        self.num_frames = num_frames
        self.img_width = img_width
        self.img_height = img_height
        self.split = split
        self.split_ratio = split_ratio
        self.is_train = (split == 'train')

        self.samples = []
        self._load_txt_file(txt_file)

        # ---- Auto-detect memmap cache ----
        self.cache = None
        self.path_to_idx = None
        txt_dir = os.path.dirname(os.path.abspath(txt_file))
        cache_file = os.path.join(txt_dir, "image_cache.dat")
        idx_file   = os.path.join(txt_dir, "image_cache_idx.json")
        if os.path.exists(cache_file) and os.path.exists(idx_file):
            try:
                with open(idx_file, 'r', encoding='utf-8') as f:
                    idx_data = json.load(f)
                n_img    = idx_data["num_images"]
                cache_h  = idx_data.get("img_h", img_height)
                cache_w  = idx_data.get("img_w", img_width)
                self.path_to_idx = idx_data["path_to_idx"]
                # Open in read-only mode; all worker processes share the same
                # physical RAM pages (OS-level page sharing on Windows & Linux)
                self.cache = np.memmap(cache_file, dtype='uint8', mode='r',
                                       shape=(n_img, cache_h, cache_w, 3))
                print(f"  ✅ memmap缓存已加载: {n_img} 张图像 "
                      f"({os.path.getsize(cache_file)/1024**3:.1f} GB, {cache_h}×{cache_w})")
            except Exception as e:
                print(f"  ⚠️  缓存加载失败，回退到PIL加载: {e}")
                self.cache = None
                self.path_to_idx = None
        else:
            print(f"  ℹ️  未找到缓存文件，使用PIL直接加载"
                  f" (运行 build_cache.py 可大幅提速)")

    def _load_txt_file(self, txt_file):
        """按txt顺序加载，连续相同label分组"""
        print(f"Loading data from {txt_file}...")

        all_data = []
        skipped = 0

        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('the') or line.startswith('#'):
                    skipped += 1
                    continue

                parts = line.rsplit(maxsplit=2)
                if len(parts) < 3:
                    skipped += 1
                    continue

                try:
                    label_particle = int(parts[1])
                    label_gas = int(parts[2]) - 1
                    if label_gas < 0:
                        skipped += 1
                        continue
                except ValueError:
                    skipped += 1
                    continue

                all_data.append({
                    'path': parts[0],
                    'idx': len(all_data),
                    'label_particle': label_particle,
                    'label_gas': label_gas
                })

        print(f"  ✓ {len(all_data)} 张图像 (跳过{skipped}行)")

        # 连续相同label分组
        groups = []
        if all_data:
            cur_group = [all_data[0]]
            cur_key = (all_data[0]['label_particle'], all_data[0]['label_gas'])

            for item in all_data[1:]:
                key = (item['label_particle'], item['label_gas'])
                if key == cur_key:
                    cur_group.append(item)
                else:
                    groups.append((cur_key, cur_group))
                    cur_group = [item]
                    cur_key = key
            groups.append((cur_key, cur_group))

        print(f"  共 {len(groups)} 个连续label组")

        particle_set = set(k[0] for k, _ in groups)
        gas_set = set(k[1] for k, _ in groups)
        print(f"  粒子标签: {sorted(particle_set)} ({len(particle_set)}类)")
        print(f"  气速标签: {sorted(gas_set)} ({len(gas_set)}类, 原始值+1)")

        # 每组内每num_frames帧为一个样本（无重叠，按txt顺序顺序读取）
        all_samples = []
        for (lp, lg), group in groups:
            n = len(group) // self.num_frames
            for i in range(n):
                s, e = i * self.num_frames, (i+1) * self.num_frames
                frames = group[s:e]
                all_samples.append({
                    'frame_paths': [f['path'] for f in frames],
                    'label_gas': lg,
                    'label_particle': lp
                })

        print(f"  总样本数(每{self.num_frames}帧): {len(all_samples)}")

        # 分层split
        grouped = {}
        for s in all_samples:
            key = (s['label_particle'], s['label_gas'])
            grouped.setdefault(key, []).append(s)

        for key, samples in grouped.items():
            n = len(samples)
            idx = max(1, int(n * self.split_ratio))
            if idx >= n:
                idx = n - 1

            if self.split == 'train':
                self.samples.extend(samples[:idx])
            else:
                self.samples.extend(samples[idx:] or samples[-1:])

        print(f"  {self.split}集: {len(self.samples)} 样本")

    def _load_image(self, path):
        """Load one image as uint8 (H, W, 3) numpy array.
        Uses memmap cache when available, otherwise falls back to PIL.
        Returns a writable copy — safe for multi-process DataLoader workers.
        """
        if self.cache is not None and self.path_to_idx is not None:
            row = self.path_to_idx.get(path)
            if row is not None:
                # np.array() creates a copy from the read-only mmap slice,
                # which also makes it writable (required by torch.from_numpy)
                return np.array(self.cache[row], dtype=np.uint8)
            # path not in index — fall through to PIL

        try:
            img = Image.open(path).convert('RGB')
            img = img.resize((self.img_width, self.img_height), Image.BILINEAR)
            return np.array(img, dtype=np.uint8)
        except Exception:
            return np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        frames = []
        for path in sample['frame_paths']:
            arr = self._load_image(path)
            tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W) uint8
            frames.append(tensor)

        # Stack to (T, C, H, W) and convert to float in one op
        frames = torch.stack(frames, dim=0).float().div_(255.0)  # (T, 3, H, W)

        # 训练增强
        if self.is_train:
            if torch.rand(1).item() < 0.5:
                frames = frames.flip(-1)

            if torch.rand(1).item() < 0.8:
                brightness_factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.6
                contrast_factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.6
                saturation_factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.4

                frames = frames * brightness_factor
                mean_val = frames.mean(dim=(-2, -1), keepdim=True)
                frames = (frames - mean_val) * contrast_factor + mean_val
                gray = frames.mean(dim=-3, keepdim=True)
                frames = gray + (frames - gray) * saturation_factor
                frames = frames.clamp(0, 1)

            if torch.rand(1).item() < 0.25:
                T, C, H, W = frames.shape
                erase_h = int(H * (0.05 + torch.rand(1).item() * 0.1))
                erase_w = int(W * (0.05 + torch.rand(1).item() * 0.1))
                top = torch.randint(0, H - erase_h + 1, (1,)).item()
                left = torch.randint(0, W - erase_w + 1, (1,)).item()
                frames[:, :, top:top+erase_h, left:left+erase_w] = \
                    torch.rand(T, C, erase_h, erase_w)

        # ImageNet归一化 — 向量化，一次操作覆盖全部T帧 (比循环快T倍)
        frames = (frames - _NORM_MEAN) / _NORM_STD

        return frames, (sample['label_gas'], sample['label_particle'])


def get_txt_dataloader(txt_file, num_frames=16, img_width=448, img_height=448,
                       batch_size=16, split='train', num_workers=2,
                       train_ratio=0.80, **kwargs):
    """Create dataloader"""
    dataset = TxtFileDataset(
        txt_file, num_frames, img_width, img_height,
        split=split, split_ratio=train_ratio
    )

    actual_workers = min(num_workers, 2)

    # pin_memory=True + numpy memmap causes CUDA to register the entire
    # ~79 GB virtual address space as pinned memory, consuming all VRAM.
    # When the memmap cache is active, data is already in RAM so pinning
    # provides no transfer benefit — disable it.
    use_pin_memory = not (dataset.cache is not None)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=actual_workers,
        pin_memory=use_pin_memory,
        drop_last=(split == 'train'),
        persistent_workers=(actual_workers > 0),
        prefetch_factor=2 if actual_workers > 0 else None,
    )

    return dataloader
