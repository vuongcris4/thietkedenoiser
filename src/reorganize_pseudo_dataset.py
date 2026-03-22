"""
Script reorganize OEM_v2_aDanh dataset.

Chuyển từ cấu trúc flat:
    OEM_v2_aDanh/
    ├── aachen_1.tif
    ├── tokyo_3.tif
    └── ...

Thành cấu trúc chuẩn:
    OEM_v2_aDanh/
    ├── images/          ← symlinks tới OpenEarthMap/{region}/images/{fn}
    ├── labels/          ← pseudo-labels (move từ root)
    ├── train.txt        ← 80% files
    ├── val.txt          ← 10% files
    └── test.txt         ← 10% files

Usage:
    python reorganize_pseudo_dataset.py
"""
import os
import sys
import shutil
import random

# ============================================================
# CONFIG
# ============================================================
OEM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'OpenEarthMap'))
PSEUDO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'OEM_v2_aDanh'))
SPLIT_SEED = 42
SPLIT_RATIO = (0.8, 0.1, 0.1)  # train/val/test


def _get_region(filename):
    """aachen_1.tif → aachen"""
    return filename.rsplit('_', 1)[0]


def main():
    print(f'OEM_ROOT:    {OEM_ROOT}')
    print(f'PSEUDO_ROOT: {PSEUDO_ROOT}')

    # === 1. Tìm tất cả pseudo-label .tif ở root ===
    pseudo_files = sorted([
        f for f in os.listdir(PSEUDO_ROOT)
        if f.endswith('.tif') and os.path.isfile(os.path.join(PSEUDO_ROOT, f))
    ])
    print(f'\nPseudo-label files found: {len(pseudo_files)}')

    if not pseudo_files:
        print('ERROR: No .tif files found in PSEUDO_ROOT!')
        sys.exit(1)

    # === 2. Tạo thư mục images/ và labels/ ===
    img_dir = os.path.join(PSEUDO_ROOT, 'images')
    lbl_dir = os.path.join(PSEUDO_ROOT, 'labels')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    print(f'Created: {img_dir}')
    print(f'Created: {lbl_dir}')

    # === 3. Di chuyển pseudo-labels vào labels/ + tạo symlink images/ ===
    valid_files = []
    skipped = 0

    for i, fn in enumerate(pseudo_files):
        region = _get_region(fn)
        oem_img = os.path.join(OEM_ROOT, region, 'images', fn)

        src_pseudo = os.path.join(PSEUDO_ROOT, fn)
        dst_pseudo = os.path.join(lbl_dir, fn)
        dst_img = os.path.join(img_dir, fn)

        # Kiểm tra ảnh gốc trong OpenEarthMap
        if not os.path.exists(oem_img):
            skipped += 1
            continue

        # Move pseudo-label vào labels/
        if not os.path.exists(dst_pseudo):
            shutil.move(src_pseudo, dst_pseudo)

        # Tạo symlink image
        if not os.path.exists(dst_img):
            os.symlink(oem_img, dst_img)

        valid_files.append(fn)

        if (i + 1) % 500 == 0:
            print(f'  Processed {i+1}/{len(pseudo_files)}...')

    print(f'\nValid files: {len(valid_files)} (skipped {skipped} without OEM image)')

    # === 4. Tạo split files ===
    random.seed(SPLIT_SEED)
    random.shuffle(valid_files)

    n = len(valid_files)
    n_train = int(n * SPLIT_RATIO[0])
    n_val = int(n * SPLIT_RATIO[1])

    splits = {
        'train.txt': sorted(valid_files[:n_train]),
        'val.txt': sorted(valid_files[n_train:n_train + n_val]),
        'test.txt': sorted(valid_files[n_train + n_val:]),
    }

    for fname, files in splits.items():
        path = os.path.join(PSEUDO_ROOT, fname)
        with open(path, 'w') as f:
            f.write('\n'.join(files) + '\n')
        print(f'  {fname}: {len(files)} files')

    print(f'\n✅ Done! Dataset reorganized at: {PSEUDO_ROOT}')
    print(f'   train={len(splits["train.txt"])}, '
          f'val={len(splits["val.txt"])}, '
          f'test={len(splits["test.txt"])}')


if __name__ == '__main__':
    main()
