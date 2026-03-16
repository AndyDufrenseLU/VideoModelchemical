"""
图片缓存生成脚本
1. 生成单个.npy缓存文件 (image_cache_224/)
2. 合并成一个大的memmap文件 (image_cache_consolidated.dat) ← 训练用这个，极快

用法: python generate_cache.py
"""

import os
import sys
import time
import numpy as np
from PIL import Image
from pathlib import Path


def count_images(txt_file="alltrain_with_c.txt"):
    """统计图片总数"""
    count = 0
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('the') or line.startswith('#'):
                continue
            parts = line.rsplit(maxsplit=2)
            if len(parts) >= 3:
                count += 1
    return count


def generate_individual_cache(txt_file="alltrain_with_c.txt", img_size=224, 
                               cache_dir="image_cache_224"):
    """Step 1: 生成单个.npy缓存文件"""
    
    print(f"Step 1: 生成单个缓存文件到 {cache_dir}/")
    
    all_paths = []
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('the') or line.startswith('#'):
                continue
            parts = line.rsplit(maxsplit=2)
            if len(parts) >= 3:
                all_paths.append(parts[0])
    
    total = len(all_paths)
    print(f"  共 {total} 张图片")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    existing = len(list(Path(cache_dir).glob("*.npy")))
    if existing >= total:
        print(f"  ✅ 单个缓存已完整 ({existing}个)，跳过")
        return total
    
    start_time = time.time()
    success = 0
    failed = 0
    skipped = 0
    
    for idx, path in enumerate(all_paths):
        cache_path = os.path.join(cache_dir, f"{idx:06d}.npy")
        
        if os.path.exists(cache_path):
            skipped += 1
            continue
        
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize((img_size, img_size), Image.BILINEAR)
            arr = np.array(img, dtype=np.uint8)
            np.save(cache_path, arr)
            success += 1
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  ⚠️ 失败 [{idx}]: {path} → {e}")
        
        processed = success + failed + skipped
        if (processed % 5000 == 0) or (processed == total):
            elapsed = time.time() - start_time
            speed = (success + failed) / max(elapsed, 0.1)
            remaining = (total - processed) / max(speed, 0.1)
            print(f"  [{processed}/{total}] "
                  f"成功:{success} 失败:{failed} 跳过:{skipped} | "
                  f"{speed:.0f}张/秒 | 剩余:{remaining/60:.1f}分钟")
    
    elapsed = time.time() - start_time
    print(f"  完成! 成功:{success} 失败:{failed} 用时:{elapsed/60:.1f}分钟")
    return total


def consolidate_cache(total_images, img_size=224, 
                       cache_dir="image_cache_224",
                       output_file="image_cache_consolidated.dat"):
    """
    Step 2: 把所有单个.npy文件合并成一个大的memmap文件
    
    这样训练时只需要1次文件映射，不再需要打开/关闭139200个文件
    速度提升: 480次文件IO/batch → 0次文件IO/batch
    """
    
    print(f"\nStep 2: 合并缓存到 {output_file}")
    
    if os.path.exists(output_file):
        # 检查大小是否正确
        expected_size = total_images * img_size * img_size * 3
        actual_size = os.path.getsize(output_file)
        if actual_size == expected_size:
            print(f"  ✅ 合并缓存已存在且大小正确 ({actual_size/1024**3:.1f} GB)")
            return
        else:
            print(f"  ⚠️ 合并缓存大小不对 (期望:{expected_size}, 实际:{actual_size})，重新生成")
    
    print(f"  创建memmap文件: ({total_images}, {img_size}, {img_size}, 3) uint8")
    expected_gb = total_images * img_size * img_size * 3 / 1024**3
    print(f"  预计大小: {expected_gb:.1f} GB")
    
    # 创建memmap
    mmap = np.memmap(output_file, dtype=np.uint8, mode='w+',
                      shape=(total_images, img_size, img_size, 3))
    
    start_time = time.time()
    success = 0
    failed = 0
    
    for idx in range(total_images):
        cache_path = os.path.join(cache_dir, f"{idx:06d}.npy")
        try:
            arr = np.load(cache_path)
            mmap[idx] = arr
            success += 1
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"  ⚠️ 读取失败 [{idx}]: {e}")
            mmap[idx] = 0  # 用黑色填充
        
        if ((idx + 1) % 10000 == 0) or (idx + 1 == total_images):
            elapsed = time.time() - start_time
            speed = (idx + 1) / elapsed
            remaining = (total_images - idx - 1) / speed
            print(f"  [{idx+1}/{total_images}] "
                  f"{speed:.0f}张/秒 | 剩余:{remaining/60:.1f}分钟")
    
    mmap.flush()
    del mmap
    
    elapsed = time.time() - start_time
    file_size = os.path.getsize(output_file) / 1024**3
    print(f"  ✅ 合并完成! {file_size:.1f} GB, 用时:{elapsed/60:.1f}分钟")


def main():
    txt_file = "alltrain_with_c.txt"
    img_size = 224
    
    print("=" * 60)
    print("图片缓存生成 (两步)")
    print("=" * 60)
    
    # Step 1: 单个文件缓存
    total = generate_individual_cache(txt_file, img_size)
    
    # Step 2: 合并成memmap
    consolidate_cache(total, img_size)
    
    print("\n" + "=" * 60)
    print("✅ 全部完成! 现在可以运行 python train.py")
    print("   训练将自动使用合并缓存 (极快加载)")
    print("=" * 60)


if __name__ == '__main__':
    main()
