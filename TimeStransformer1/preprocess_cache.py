"""
预处理脚本 v2：保持宽高比！

原图 375×925 (竖条) → resize宽度到224 → 224×552 (保持比例)
训练时再通过随机竖向crop取224×224的区域

用法: python preprocess_cache.py
"""

import os
import sys
import time
import numpy as np
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def resize_keep_ratio(args):
    """处理单张图片: resize宽度到target_w，保持宽高比"""
    idx, img_path, output_path, target_w = args
    try:
        if os.path.exists(output_path):
            return idx, True, "已存在"
        
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        
        # 保持宽高比，宽度resize到target_w
        ratio = target_w / w
        new_h = int(h * ratio)
        img = img.resize((target_w, new_h), Image.BILINEAR)
        
        arr = np.array(img, dtype=np.uint8)
        np.save(output_path, arr)
        
        return idx, True, f"{w}x{h} -> {target_w}x{new_h}"
    except Exception as e:
        return idx, False, str(e)


def main():
    txt_file = "alltrain_with_c.txt"
    target_w = 224  # 目标宽度
    cache_dir = "image_cache_224w"  # 新目录名，区分旧的正方形cache
    num_threads = 8
    
    print("=" * 60)
    print(f"📦 图像预处理 v2 (保持宽高比)")
    print(f"  策略: 宽度→{target_w}，高度按比例缩放")
    print(f"  输入: {txt_file}")
    print(f"  输出: {cache_dir}/")
    print("=" * 60)
    
    # 读取路径
    print("\n读取文件列表...")
    all_paths = []
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('the') or line.startswith('#'):
                continue
            parts = line.rsplit(maxsplit=2)
            if len(parts) >= 3:
                all_paths.append(parts[0])
    
    print(f"  共 {len(all_paths)} 张图片")
    
    # 检查第一张图的尺寸
    if all_paths:
        sample = Image.open(all_paths[0])
        w, h = sample.size
        ratio = target_w / w
        new_h = int(h * ratio)
        print(f"  原始尺寸: {w}×{h} → 缩放后: {target_w}×{new_h}")
        print(f"  训练时随机crop {target_w}×{target_w} = 224×224")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # 检查已处理数量
    existing = sum(1 for p in Path(cache_dir).glob("*.npy"))
    if existing > 0:
        print(f"  已有 {existing} 张缓存")
    
    # 准备任务
    tasks = []
    for idx, img_path in enumerate(all_paths):
        output_path = os.path.join(cache_dir, f"{idx:06d}.npy")
        if not os.path.exists(output_path):
            tasks.append((idx, img_path, output_path, target_w))
    
    if not tasks:
        print("\n✅ 所有图片已缓存完成！")
        return
    
    print(f"  需要处理: {len(tasks)} 张")
    
    # 预估磁盘
    if all_paths:
        size_per = target_w * new_h * 3 + 128
        total_gb = len(all_paths) * size_per / 1024**3
        print(f"  预估磁盘: {total_gb:.1f} GB")
    
    # 处理
    print(f"\n🚀 开始预处理...")
    start_time = time.time()
    success = fail = 0
    first_result = None
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(resize_keep_ratio, t): t for t in tasks}
        
        for future in as_completed(futures):
            idx, ok, msg = future.result()
            if ok:
                success += 1
                if first_result is None:
                    first_result = msg
                    print(f"  第一张: {msg}")
            else:
                fail += 1
                if fail <= 3:
                    print(f"  ❌ {idx}: {msg}")
            
            total = success + fail
            if total % 5000 == 0 or total == len(tasks):
                elapsed = time.time() - start_time
                speed = total / max(elapsed, 0.1)
                remain = (len(tasks) - total) / max(speed, 1)
                print(f"  进度: {total}/{len(tasks)} "
                      f"({100*total/len(tasks):.1f}%) | "
                      f"{speed:.0f}张/秒 | "
                      f"剩余: {remain/60:.1f}分")
    
    elapsed = time.time() - start_time
    print(f"\n✅ 完成! 用时: {elapsed/60:.1f}分钟")
    print(f"  成功: {success}, 失败: {fail}")
    print(f"\n现在运行: python train.py")


if __name__ == '__main__':
    main()