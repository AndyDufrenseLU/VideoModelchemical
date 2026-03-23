"""
Dataset module - 流化床图片数据加载

策略:
1. 优先用224×224正方形缓存(image_cache_224)做全图输入
2. 如果有保持比例缓存(image_cache_224w)则从中resize到224×224
3. 全图输入保留气速的全局分布信息
4. 按txt顺序连续帧分组，不跨label组
"""

import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class TxtFileDataset(Dataset):
    """流化床图像数据集"""
    
    def __init__(self, txt_file, num_frames=10, img_size=224, 
                 transform=None, split='train', split_ratio=0.75,
                 max_images_per_group=None):
        self.num_frames = num_frames
        self.img_size = img_size
        self.split = split
        self.split_ratio = split_ratio
        # ⚡ 方法四: 每组(粒子,气速)最多使用多少张原始图片
        #   None=全部使用  800=快速实验模式
        self.max_images_per_group = max_images_per_group
        
        # 检查缓存目录
        self.cache_dir_square = f"image_cache_{img_size}"   # 224×224正方形
        self.cache_dir_ratio = f"image_cache_{img_size}w"   # 224×552保持比例
        
        if os.path.isdir(self.cache_dir_square):
            self.cache_dir = self.cache_dir_square
            self.cache_mode = 'square'
            cache_count = len(list(Path(self.cache_dir).glob("*.npy")))
            print(f"  ✅ 使用全图缓存: {self.cache_dir}/ ({cache_count}个)")
        elif os.path.isdir(self.cache_dir_ratio):
            self.cache_dir = self.cache_dir_ratio
            self.cache_mode = 'ratio'
            cache_count = len(list(Path(self.cache_dir).glob("*.npy")))
            print(f"  ✅ 使用比例缓存(自动resize): {self.cache_dir}/ ({cache_count}个)")
        else:
            self.cache_dir = None
            self.cache_mode = None
            print(f"  ⚠️ 无缓存，从原始图片加载(较慢)")
        
        # 归一化 (ImageNet标准)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # 数据增强
        self.is_train = (split == 'train')
        
        self.samples = []
        self._load_txt_file(txt_file)
        
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
                    label_gas = int(parts[2]) - 1  # 1-19 → 0-18
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

        # 快速检查路径是否存在，防止读不到图导致全零样本
        missing = []
        for item in all_data:
            if not os.path.exists(item['path']):
                missing.append(item['path'])
                if len(missing) >= 5:
                    break
        if missing:
            print("  ⚠️ 检测到缺失的图像路径，示例(前5):")
            for p in missing:
                print(f"     - {p}")
            print("  ⚠️ 请检查txt路径是否指向正确的盘符/目录")
        
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
        
        # 每组内每num_frames帧为一个样本
        # ⚡ 方法四: 如果设了 max_images_per_group，截断每组超出的图片
        if self.max_images_per_group is not None:
            original_groups = len(groups)
            groups = [(key, grp[:self.max_images_per_group]) for key, grp in groups]
            total_limited = sum(len(g) for _, g in groups)
            print(f"  ⚡ 每组限制 {self.max_images_per_group} 张图 "
                  f"(共 {original_groups} 组, 总图片数 {total_limited})")

        all_samples = []
        for (lp, lg), group in groups:
            n = len(group) // self.num_frames
            for i in range(n):
                s, e = i * self.num_frames, (i+1) * self.num_frames
                frames = group[s:e]
                all_samples.append({
                    'frame_indices': [f['idx'] for f in frames],
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
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        frames = []
        error_count = 0
        for i, global_idx in enumerate(sample['frame_indices']):
            try:
                if self.cache_dir and self.cache_mode == 'square':
                    # 224×224正方形缓存 → 直接使用
                    cache_path = os.path.join(self.cache_dir, f"{global_idx:06d}.npy")
                    arr = np.load(cache_path)  # (224, 224, 3) uint8
                elif self.cache_dir and self.cache_mode == 'ratio':
                    # 保持比例缓存 → resize全图到224×224 (保留完整空间信息)
                    cache_path = os.path.join(self.cache_dir, f"{global_idx:06d}.npy")
                    arr = np.load(cache_path)  # (H, 224, 3)
                    img = Image.fromarray(arr)
                    img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
                    arr = np.array(img, dtype=np.uint8)
                else:
                    # 无缓存 → 直接resize全图到224×224
                    img = Image.open(sample['frame_paths'][i]).convert('RGB')
                    img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
                    arr = np.array(img, dtype=np.uint8)

                tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
                frames.append(tensor)
                
            except Exception as e:
                error_count += 1
                if error_count <= 1:  # 只打印第一个错误
                    print(f"  ⚠️ 加载失败 idx={global_idx}: {e}")
                frames.append(torch.zeros(3, self.img_size, self.img_size))
        
        frames = torch.stack(frames, dim=0)  # (T, 3, H, W)
        
        # 训练增强
        if self.is_train:
            # 随机水平翻转 (整个序列一起)
            if torch.rand(1).item() < 0.5:
                frames = frames.flip(-1)
            
            # 随机亮度变化 (整个序列一起)
            if torch.rand(1).item() < 0.5:
                brightness = 0.9 + torch.rand(1).item() * 0.2  # 0.9-1.1 (减弱增强)
                frames = frames * brightness
                frames = frames.clamp(0, 1)
        
        # ImageNet归一化
        for t in range(frames.size(0)):
            frames[t] = self.normalize(frames[t])
        
        return frames, (sample['label_gas'], sample['label_particle'])


def worker_init_fn(worker_id):
    """Windows 多进程 DataLoader 初始化（避免进程间干扰）"""
    import numpy as np
    np.random.seed(worker_id)


def get_txt_dataloader(txt_file, num_frames=10, img_size=224, batch_size=16,
                       split='train', num_workers=4, train_ratio=0.75,
                       max_images_per_group=None, **kwargs):
    """
    创建 DataLoader

    速度优化（无缓存版）:
      - pin_memory=True:   CPU→GPU 传输不需要中间拷贝（快 10~30%）
      - prefetch_factor=4: 后台预取 4 个 batch，GPU 训练时 I/O 同步进行
      - persistent_workers: workers 保持常驻，避免每 epoch 重启开销
      - num_workers 不再限制到 2（原代码有错误的上限）
    """
    dataset = TxtFileDataset(
        txt_file, num_frames, img_size, split=split, split_ratio=train_ratio,
        max_images_per_group=max_images_per_group
    )

    # Windows 上 num_workers > 0 需要配合 if __name__ == '__main__' 保护
    # 建议在 train.py 的 main() 里调用，已满足此条件
    use_workers = max(0, num_workers)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=use_workers,
        pin_memory=True,                          # CPU→GPU 零拷贝传输
        drop_last=(split == 'train'),
        persistent_workers=(use_workers > 0),     # 每 epoch 不重启 workers
        prefetch_factor=4 if use_workers > 0 else None,  # 预取4个batch
        worker_init_fn=worker_init_fn if use_workers > 0 else None,
    )

    return dataloader

