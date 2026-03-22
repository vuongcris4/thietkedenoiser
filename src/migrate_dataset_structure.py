"""
Script migrate existing OEM_v2_aDanh from old structure to new structure.

Old structure:
    OEM_v2_aDanh/
    ├── images/
    ├── labels/        ← pseudo-labels (old name)
    └── train.txt, val.txt, test.txt

New structure:
    OEM_v2_aDanh/
    ├── images/
    ├── pseudolabels/  ← rename from labels/ (pseudo-labels from CISC-R)
    ├── labels/        ← copy ground truth from OpenEarthMap
    └── train.txt, val.txt, test.txt

Usage:
    python migrate_dataset_structure.py
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

    old_labels_dir = os.path.join(PSEUDO_ROOT, 'labels')
    pseudo_dir = os.path.join(PSEUDO_ROOT, 'pseudolabels')
    gt_dir = os.path.join(PSEUDO_ROOT, 'labels')

    if not os.path.exists(old_labels_dir):
        print(f'ERROR: Old labels/ folder not found at {old_labels_dir}')
        sys.exit(1)

    # === 1. Rename labels/ → pseudolabels/ ===
    if not os.path.exists(pseudo_dir):
        print(f'Renaming: {old_labels_dir} → {pseudo_dir}')
        os.rename(old_labels_dir, pseudo_dir)
    else:
        print(f'pseudolabels/ already exists: {pseudo_dir}')

    # === 2. Create labels/ for ground truth ===
    os.makedirs(gt_dir, exist_ok=True)
    print(f'Created: {gt_dir}')

    # === 3. Copy ground truth from OpenEarthMap ===
    pseudo_files = sorted([
        f for f in os.listdir(pseudo_dir)
        if f.endswith('.tif')
    ])
    print(f'\nPseudo-label files found: {len(pseudo_files)}')

    copied = 0
    skipped = 0

    for i, fn in enumerate(pseudo_files):
        region = _get_region(fn)
        oem_gt = os.path.join(OEM_ROOT, region, 'labels', fn)
        dst_gt = os.path.join(gt_dir, fn)

        if os.path.exists(oem_gt) and not os.path.exists(dst_gt):
            shutil.copy2(oem_gt, dst_gt)
            copied += 1
        elif os.path.exists(dst_gt):
            skipped += 1
        else:
            print(f'  Warning: GT not found for {fn}')
            skipped += 1

        if (i + 1) % 500 == 0:
            print(f'  Processed {i+1}/{len(pseudo_files)}...')

    print(f'\n✅ Done! Migrated dataset structure:')
    print(f'   - Renamed labels/ → pseudolabels/')
    print(f'   - Created labels/ with {copied} ground truth files')
    print(f'   - Skipped {skipped} (already exists or GT not found)')


if __name__ == '__main__':
    main()
