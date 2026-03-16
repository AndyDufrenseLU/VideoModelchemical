"""
build_cache.py — One-time image preprocessing into numpy memmap cache.

Usage:
    python build_cache.py

Reads all valid image paths from alltrain_with_c.txt, resizes each image to
448×448 uint8, and writes them into a memory-mapped file (image_cache.dat).
Also writes image_cache_idx.json mapping each absolute path to its row index.

After this script completes, dataset_txt.py will auto-detect the cache and
use direct array lookups instead of PIL I/O + resize on every training step.

Disk space required: N_images * 448 * 448 * 3 bytes
  e.g. 139200 images ≈ 79 GB
"""

import os
import sys
import json
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

# ---- Configuration ----
TXT_FILE   = "alltrain_with_c.txt"
CACHE_FILE = "image_cache.dat"
IDX_FILE   = "image_cache_idx.json"
IMG_H      = 448
IMG_W      = 448
NUM_THREADS = 8   # parallel IO threads (tunable; 8 works well for HDD/SSD)
REPORT_EVERY = 500  # print progress every N images


# -----------------------------------------------------------------------
# Step 1: Parse txt file to get all unique image paths (in order)
# -----------------------------------------------------------------------

def parse_paths(txt_file):
    """Return list of (original_line_path, abs_path) for every valid line."""
    paths = []
    skipped = 0

    txt_dir = os.path.dirname(os.path.abspath(txt_file))

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
                int(parts[1])
                label_gas = int(parts[2]) - 1
                if label_gas < 0:
                    skipped += 1
                    continue
            except ValueError:
                skipped += 1
                continue

            raw_path = parts[0]
            # Resolve relative paths relative to the txt file's directory
            if not os.path.isabs(raw_path):
                abs_path = os.path.normpath(os.path.join(txt_dir, raw_path))
            else:
                abs_path = os.path.normpath(raw_path)

            paths.append((raw_path, abs_path))

    print(f"  Parsed {len(paths)} valid image entries ({skipped} skipped)")
    return paths


def deduplicate_paths(path_pairs):
    """Return (unique_abs_paths_ordered, path_to_idx dict)."""
    seen = {}
    ordered = []
    for raw, absp in path_pairs:
        if absp not in seen:
            seen[absp] = len(ordered)
            ordered.append(absp)
    # Also need to map raw_path → idx for paths stored in dataset
    raw_to_idx = {}
    for raw, absp in path_pairs:
        raw_to_idx[raw] = seen[absp]
        raw_to_idx[absp] = seen[absp]
    return ordered, raw_to_idx, seen


# -----------------------------------------------------------------------
# Step 2: Worker — load + resize one image
# -----------------------------------------------------------------------

def load_image(row_idx_path):
    """Load image at path, resize to IMG_H x IMG_W, return (row_idx, uint8 array)."""
    row_idx, path = row_idx_path
    try:
        img = Image.open(path).convert('RGB')
        img = img.resize((IMG_W, IMG_H), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8)
        if arr.shape != (IMG_H, IMG_W, 3):
            arr = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        return row_idx, arr, None
    except Exception as e:
        return row_idx, np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8), str(e)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    txt_file   = os.path.join(script_dir, TXT_FILE)
    cache_file = os.path.join(script_dir, CACHE_FILE)
    idx_file   = os.path.join(script_dir, IDX_FILE)

    if not os.path.exists(txt_file):
        print(f"ERROR: {txt_file} not found.")
        sys.exit(1)

    print("=" * 60)
    print("  build_cache.py — TimeSformer image cache builder")
    print("=" * 60)
    print(f"  Source    : {txt_file}")
    print(f"  Cache     : {cache_file}")
    print(f"  Index     : {idx_file}")
    print(f"  Image size: {IMG_H} x {IMG_W}")
    print(f"  Threads   : {NUM_THREADS}")
    print()

    # ---- Parse paths ----
    print("Step 1: Parsing image paths...")
    path_pairs = parse_paths(txt_file)

    ordered_paths, raw_to_idx, abs_to_idx = deduplicate_paths(path_pairs)
    N = len(ordered_paths)

    size_gb = N * IMG_H * IMG_W * 3 / 1024**3
    print(f"  Unique images: {N}")
    print(f"  Cache size   : {size_gb:.1f} GB")
    print()

    # ---- Check if cache already complete ----
    if os.path.exists(cache_file) and os.path.exists(idx_file):
        existing_size = os.path.getsize(cache_file)
        expected_size = N * IMG_H * IMG_W * 3
        if existing_size == expected_size:
            print(f"  Cache already complete ({existing_size / 1024**3:.1f} GB). Nothing to do.")
            print(f"  Delete {cache_file} to rebuild.")
            return
        else:
            print(f"  Existing cache size {existing_size} != expected {expected_size}. Rebuilding...")

    # ---- Create memmap ----
    print("Step 2: Creating memmap file (may take a moment for large files)...")
    t0 = time.time()
    mmap = np.memmap(cache_file, dtype='uint8', mode='w+', shape=(N, IMG_H, IMG_W, 3))
    print(f"  Memmap created in {time.time()-t0:.1f}s")
    print()

    # ---- Load images in parallel ----
    print(f"Step 3: Loading and resizing {N} images with {NUM_THREADS} threads...")
    t_start = time.time()
    errors = []
    done = 0

    tasks = [(idx, path) for idx, path in enumerate(ordered_paths)]

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {executor.submit(load_image, task): task for task in tasks}

        for future in as_completed(futures):
            row_idx, arr, err = future.result()
            mmap[row_idx] = arr
            done += 1

            if err:
                errors.append((row_idx, ordered_paths[row_idx], err))

            if done % REPORT_EVERY == 0 or done == N:
                elapsed = time.time() - t_start
                rate    = done / elapsed
                eta     = (N - done) / rate if rate > 0 else 0
                pct     = 100 * done / N
                print(f"  [{done:6d}/{N}] {pct:5.1f}%  {rate:.0f} img/s  ETA {eta/60:.1f} min"
                      + (f"  ({len(errors)} errors)" if errors else ""))

    # Flush to disk
    print()
    print("Step 4: Flushing cache to disk...")
    t_flush = time.time()
    mmap.flush()
    del mmap
    print(f"  Flush complete in {time.time()-t_flush:.1f}s")

    # ---- Save index ----
    print("Step 5: Saving path index...")
    # Save both raw and abs keys for maximum compatibility
    idx_data = {
        "img_h": IMG_H,
        "img_w": IMG_W,
        "num_images": N,
        "path_to_idx": raw_to_idx
    }
    with open(idx_file, 'w', encoding='utf-8') as f:
        json.dump(idx_data, f, ensure_ascii=False)
    print(f"  Index saved: {idx_file}  ({len(raw_to_idx)} path entries)")

    # ---- Summary ----
    total_time = time.time() - t_start
    print()
    print("=" * 60)
    print("  Cache build complete!")
    print(f"  Total time : {total_time/60:.1f} minutes")
    print(f"  Images     : {N - len(errors)} OK, {len(errors)} errors")
    print(f"  Cache file : {cache_file}  ({os.path.getsize(cache_file)/1024**3:.1f} GB)")
    print(f"  Index file : {idx_file}")
    print()
    if errors:
        print(f"  WARNING: {len(errors)} images had load errors (replaced with zeros):")
        for i, (ridx, path, err) in enumerate(errors[:10]):
            print(f"    [{ridx}] {path}: {err}")
        if len(errors) > 10:
            print(f"    ... and {len(errors)-10} more")
    print("=" * 60)


if __name__ == '__main__':
    main()
