"""
Script reorganize OEM_v2_aDanh dataset.

Chuyển từ cấu trúc flat:
    OEM_v2_aDanh/
    ├── aachen_1.tif
    ├── tokyo_3.tif
    └── ...

Thành cấu trúc chuẩn:
    OEM_v2_aDanh/
    ├── images/       ← symlinks tới OpenEarthMap/{region}/images/{fn}
    ├── pseudolabels/ ← pseudo-labels từ CISC-R (move từ root)
    └── labels/       ← ground truth từ OpenEarthMap (copy)

Ghi chú: Split files (train.txt, val.txt, test.txt) sẽ tự động tạo
khi load dataset theo tỷ lệ 80/10/10.

Usage:
    python reorganize_pseudo_dataset.py
"""
import os
import sys
import shutil

# ============================================================
# CONFIG
# ============================================================
OEM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'OpenEarthMap'))
PSEUDO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'OEM_v2_aDanh'))


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

    # === 2. Tạo thư mục images/, pseudolabels/, labels/ ===
    img_dir = os.path.join(PSEUDO_ROOT, 'images')
    pseudo_dir = os.path.join(PSEUDO_ROOT, 'pseudolabels')
    gt_dir = os.path.join(PSEUDO_ROOT, 'labels')

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(pseudo_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    print(f'Created: {img_dir}')
    print(f'Created: {pseudo_dir}')
    print(f'Created: {gt_dir}')

    # === 3. Di chuyển pseudo-labels + copy ground truth + tạo symlink images/ ===
    valid_files = []
    skipped = 0

    for i, fn in enumerate(pseudo_files):
        region = _get_region(fn)
        oem_img = os.path.join(OEM_ROOT, region, 'images', fn)
        oem_lbl = os.path.join(OEM_ROOT, region, 'labels', fn)

        src_pseudo = os.path.join(PSEUDO_ROOT, fn)
        dst_pseudo = os.path.join(pseudo_dir, fn)
        dst_img = os.path.join(img_dir, fn)
        dst_gt = os.path.join(gt_dir, fn)

        # Kiểm tra ảnh gốc trong OpenEarthMap
        if not os.path.exists(oem_img):
            skipped += 1
            continue

        # Move pseudo-label vào pseudolabels/
        if not os.path.exists(dst_pseudo):
            shutil.move(src_pseudo, dst_pseudo)

        # Tạo symlink image
        if not os.path.exists(dst_img) and not os.path.islink(dst_img):
            os.symlink(oem_img, dst_img)

        # Copy ground truth vào labels/
        if os.path.exists(oem_lbl) and not os.path.exists(dst_gt):
            shutil.copy2(oem_lbl, dst_gt)

        valid_files.append(fn)

        if (i + 1) % 500 == 0:
            print(f'  Processed {i+1}/{len(pseudo_files)}...')

    print(f'\nValid files: {len(valid_files)} (skipped {skipped} without OEM image)')

    print(f'\n✅ Done! Dataset reorganized at: {PSEUDO_ROOT}')
    print(f'   Split files will be auto-created on first dataset load (80/10/10)')


if __name__ == '__main__':
    main()
