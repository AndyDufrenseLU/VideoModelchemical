"""
Dataset module - 流化床图片数据加载 (直接加载版)
直接从原始图片文件加载，不使用任何缓存
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class TxtFileDataset(Dataset):
    """流化床图像数据集 - 直接加载"""
    
    def __init__(self, txt_file, num_frames=30, img_size=224, 
                 transform=None, split='train', split_ratio=0.75):
        self.num_frames = num_frames
        self.img_size = img_size
        self.split = split
        self.split_ratio = split_ratio
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
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
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        frames = []
        for path in sample['frame_paths']:
            try:
                img = Image.open(path).convert('RGB')
                img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
                arr = np.array(img, dtype=np.uint8)
                tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
                frames.append(tensor)
            except Exception as e:
                frames.append(torch.zeros(3, self.img_size, self.img_size))
        
        frames = torch.stack(frames, dim=0)  # (T, 3, H, W)
        
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
                frames[:, :, top:top+erase_h, left:left+erase_w] = torch.rand(T, C, erase_h, erase_w)
        
        # ImageNet归一化
        for t in range(frames.size(0)):
            frames[t] = self.normalize(frames[t])
        
        return frames, (sample['label_gas'], sample['label_particle'])


def get_txt_dataloader(txt_file, num_frames=30, img_size=224, batch_size=16,
                       split='train', num_workers=2, train_ratio=0.75, **kwargs):
    """Create dataloader"""
    dataset = TxtFileDataset(
        txt_file, num_frames, img_size, split=split, split_ratio=train_ratio
    )
    
    actual_workers = min(num_workers, 2)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=actual_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
        persistent_workers=(actual_workers > 0)
    )
    
    return dataloader
