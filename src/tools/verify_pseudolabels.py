"""
Verify pseudolabels vs ground truth labels for OpenEarthMap dataset.

Usage:
    # Batch verify (default: 20 samples)
    python src/verify_pseudolabels.py --pseudo_root data/OEM_v2_aDanh

    # Verify specific sample
    python src/verify_pseudolabels.py --sample aachen_1

    # Analyze class distribution
    python src/verify_pseudolabels.py --analyze --max-samples 5

OpenEarthMap class mapping (values 2-8):
    2: Bareland     3: Rangeland    4: Developed
    5: Road         6: Tree         7: Water
    8: Agriculture  9: Building (if present)
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CLASS MAPPING
# =============================================================================

# OpenEarthMap label files dùng values 2-8 (không phải 0-7)
# Sau khi remap (trừ 2): value 2→0, 3→1, ..., 8→6
#
# CLASS_NAMES từ dataset.py (8 classes):
#   Index 0: Bareland     ← file value 2
#   Index 1: Rangeland    ← file value 3
#   Index 2: Developed    ← file value 4
#   Index 3: Road         ← file value 5
#   Index 4: Tree         ← file value 6
#   Index 5: Water        ← file value 7
#   Index 6: Agriculture  ← file value 8
#   Index 7: Building     ← file value 9 (không có trong data này)

LABEL_VALUE_OFFSET = 2  # Trừ đi 2 để được class index
NUM_CLASSES = 8  # Số classes (từ dataset.py)

# Mapping: class index → name (sau khi remap)
CLASS_VALUE_TO_NAME = {
    0: 'Bareland',      # file value 2
    1: 'Rangeland',     # file value 3
    2: 'Developed',     # file value 4
    3: 'Road',          # file value 5
    4: 'Tree',          # file value 6
    5: 'Water',         # file value 7
    6: 'Agriculture',   # file value 8
    7: 'Building',      # file value 9 (nếu có)
}

# Valid class indices after remapping (0-7)
VALID_CLASS_VALUES = list(range(8))

# OpenEarthMap official colors (from dataset documentation)
# https://github.com/open-earth-map
COLORS = {
    0: [0x80, 0x00, 0x00],   # Bareland      - #800000 (dark red/maroon)
    1: [0x00, 0xFF, 0x24],   # Rangeland     - #00FF24 (bright green)
    2: [0x94, 0x94, 0x94],   # Developed     - #949494 (gray)
    3: [0xFF, 0xFF, 0xFF],   # Road          - #FFFFFF (white)
    4: [0x22, 0x61, 0x26],   # Tree          - #226126 (dark green)
    5: [0x00, 0x45, 0xFF],   # Water         - #0045FF (blue)
    6: [0x4B, 0xB5, 0x49],   # Agriculture   - #4BB549 (medium green)
    7: [0xDE, 0x1F, 0x07],   # Building      - #DE1F07 (red)
}


def get_class_name(value: int) -> str:
    return CLASS_VALUE_TO_NAME.get(value, f'Unknown_{value}')


def remap_label(mask: np.ndarray) -> np.ndarray:
    """
    Remap label values from file format (2-8) to class indices (0-7).

    Input:
        mask: np.ndarray with values 0, 1, 2, 3, 4, 5, 6, 7, 8
    Output:
        np.ndarray with values 0-7 (values 0,1 → background → 0)
    """
    mask = mask.astype(np.int16) - LABEL_VALUE_OFFSET
    # Values 0, 1 in original become negative → set to 0 (background)
    mask[mask < 0] = 0
    # Values > 8 become > 6, clamp to 7
    mask[mask >= NUM_CLASSES] = NUM_CLASSES - 1
    return mask.astype(np.uint8)


# =============================================================================
# METRICS
# =============================================================================

def compute_iou(mask1: np.ndarray, mask2: np.ndarray, remap: bool = True) -> Dict[str, float]:
    """
    Compute per-class IoU and mIoU.

    Args:
        mask1, mask2: Label masks (raw file values or pre-remapped)
        remap: If True, remap from file values (2-8) to class indices (0-7).
               Set to False if masks are already remapped.
    """
    # Remap from file values (2-8) to class indices (0-7) if needed
    if remap:
        mask1 = remap_label(mask1)
        mask2 = remap_label(mask2)

    ious = {}

    # Tính IoU cho tất cả 8 classes (0-7)
    # Class 0 = Bareland (không phải background)
    for cls_val in range(8):  # Classes 0-7
        mask1_cls = (mask1 == cls_val)
        mask2_cls = (mask2 == cls_val)

        intersection = np.logical_and(mask1_cls, mask2_cls).sum()
        union = np.logical_or(mask1_cls, mask2_cls).sum()

        if union > 0:
            ious[get_class_name(cls_val)] = intersection / union
        else:
            ious[get_class_name(cls_val)] = np.nan

    # mIoU: trung bình các classes có data (loại trừ NaN)
    # Lưu ý: class chỉ có trong GT hoặc chỉ có trong Pred vẫn tính (IoU = 0)
    valid_ious = [v for v in ious.values() if not np.isnan(v)]
    ious['mIoU'] = np.mean(valid_ious) if valid_ious else 0.0

    return ious


def compute_pixel_accuracy(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute overall pixel accuracy."""
    return (mask1 == mask2).sum() / mask1.size


# =============================================================================
# VISUALIZATION
# =============================================================================

def label_to_color(mask: np.ndarray) -> np.ndarray:
    """Convert mask to RGB color map."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for val, color in COLORS.items():
        color_mask[mask == val] = color

    return color_mask


def visualize_sample(
    image_path: Path,
    label_path: Path,
    pseudolabel_path: Path,
    save_dir: Path = None
) -> Tuple[Dict[str, float], float]:
    """
    Visualize side-by-side: RGB, GT, Pseudo, Difference.

    Returns:
        ious: Per-class IoU and mIoU
        pixel_acc: Pixel accuracy
    """

    # Load data
    image = cv2.imread(str(image_path))
    label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    pseudolabel = cv2.imread(str(pseudolabel_path), cv2.IMREAD_UNCHANGED)

    if image is None or label is None or pseudolabel is None:
        return None, None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Remap labels from file values (2-8) to class indices (0-7)
    label = remap_label(label)
    pseudolabel = remap_label(pseudolabel)

    # Pass remap=False because labels are already remapped
    ious = compute_iou(label, pseudolabel, remap=False)
    pixel_acc = compute_pixel_accuracy(label, pseudolabel)

    # Create visualizations
    label_color = label_to_color(label)
    pseudo_color = label_to_color(pseudolabel)

    # Difference map
    diff = (label != pseudolabel).astype(float)
    diff_overlay = np.zeros((*diff.shape, 3))
    diff_overlay[diff == 1] = [1, 0, 0]
    diff_overlay = cv2.addWeighted(
        image.astype(float) / 255, 0.7,
        diff_overlay, 0.5, 0
    )
    diff_overlay = np.clip(diff_overlay, 0, 1)

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(image)
    axes[0].set_title(f'RGB Image\n{image_path.name}')
    axes[0].axis('off')

    axes[1].imshow(label_color)
    axes[1].set_title(f'Ground Truth\n{label_path.name}')
    axes[1].axis('off')

    axes[2].imshow(pseudo_color)
    axes[2].set_title(f'Pseudolabel\n{pseudolabel_path.name}')
    axes[2].axis('off')

    axes[3].imshow(diff_overlay)
    axes[3].set_title(f'Difference\nmIoU: {ious["mIoU"]:.2%} | Acc: {pixel_acc:.2%}')
    axes[3].axis('off')

    plt.tight_layout()

    if save_dir:
        save_path = save_dir / f'verify_{image_path.stem}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()

    return ious, pixel_acc


# =============================================================================
# ANALYSIS MODE
# =============================================================================

def analyze_distribution(label_path: Path, pseudo_path: Path, name: str):
    """Print detailed class distribution analysis."""

    label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    pseudo = cv2.imread(str(pseudolabel_path), cv2.IMREAD_UNCHANGED)

    if label is None or pseudo is None:
        return

    # Remap to class indices
    label = remap_label(label)
    pseudo = remap_label(pseudo)

    print(f"\n{'='*60}")
    print(f"SAMPLE: {name}")
    print('='*60)

    print("\nGT Label distribution:")
    for val in sorted(np.unique(label)):
        count = (label == val).sum()
        pct = count / label.size * 100
        name_str = get_class_name(val)
        print(f"  {val:2} ({name_str:12}): {count:8} ({pct:5.2f}%)")

    print("\nPseudolabel distribution:")
    for val in sorted(np.unique(pseudo)):
        count = (pseudo == val).sum()
        pct = count / pseudo.size * 100
        name_str = get_class_name(val)
        print(f"  {val:2} ({name_str:12}): {count:8} ({pct:5.2f}%)")

    print("\nPer-class overlap:")
    all_vals = set(np.unique(label)) | set(np.unique(pseudo))
    for val in sorted(all_vals):
        name_str = get_class_name(val)
        overlap = ((label == val) & (pseudo == val)).sum()
        union = ((label == val) | (pseudo == val)).sum()
        iou = overlap / union if union > 0 else 0
        label_has = (label == val).sum() > 0
        pseudo_has = (pseudo == val).sum() > 0
        match = "✓" if label_has == pseudo_has else "✗"
        print(f"  {val:2} ({name_str:12}): IoU={iou:.2%} {match}")


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def find_image(pseudo_root: Path, filename: str) -> Path:
    """Find RGB image path (try multiple locations)."""
    for region in ['', 'muenster', 'rio', 'khartoum', 'paris', 'palu',
                   'taipei', 'chicago', 'washington', 'aachen', 'abancay']:
        for subdir in ['images', f'{region}/images']:
            try_path = pseudo_root / subdir / filename if region else pseudo_root / 'images' / filename
            if try_path.exists():
                return try_path
    return None


def batch_verify(
    pseudo_root: Path,
    region: str = None,
    max_samples: int = 20,
    save_dir: Path = None,
    analyze: bool = False
):
    """Verify multiple samples."""

    pseudo_dir = pseudo_root / 'pseudolabels'
    label_dir = pseudo_root / 'labels'

    if not pseudo_dir.exists():
        print(f"Error: Pseudolabel directory not found: {pseudo_dir}")
        return

    pseudo_files = sorted(list(pseudo_dir.glob('*.tif')))

    if region:
        pseudo_files = [f for f in pseudo_files if region in f.name]

    print(f"\nFound {len(pseudo_files)} pseudolabels")
    if max_samples > 0:
        print(f"Verifying first {max_samples} samples...\n")
        pseudo_files = pseudo_files[:max_samples]

    all_ious = []
    all_acc = []

    for pseudo_path in pseudo_files:
        label_path = label_dir / pseudo_path.name

        if not label_path.exists():
            print(f"  Skip {pseudo_path.name}: GT not found")
            continue

        image_path = find_image(pseudo_root, pseudo_path.name)
        if image_path is None:
            print(f"  Skip {pseudo_path.name}: Image not found")
            continue

        if analyze:
            analyze_distribution(label_path, pseudo_path, pseudo_path.name)
        else:
            ious, acc = visualize_sample(image_path, label_path, pseudo_path, save_dir)
            if ious:
                all_ious.append(ious)
                all_acc.append(acc)

    # Aggregate statistics
    if all_ious and not analyze:
        print("\n" + "="*60)
        print("AGGREGATE STATISTICS")
        print("="*60)

        avg_iou = {}
        all_miou = []

        for cls_name in [get_class_name(v) for v in VALID_CLASS_VALUES]:
            values = [iou.get(cls_name, 0) for iou in all_ious]
            # Filter out NaN (class not present in any sample)
            valid_values = [v for v in values if not np.isnan(v)]
            if valid_values:
                avg_iou[cls_name] = np.mean(valid_values)
            else:
                avg_iou[cls_name] = 0.0

        # Compute average mIoU from all samples
        for iou in all_ious:
            if 'mIoU' in iou and not np.isnan(iou['mIoU']):
                all_miou.append(iou['mIoU'])

        for cls_name in [get_class_name(v) for v in VALID_CLASS_VALUES]:
            val = avg_iou.get(cls_name, 0)
            print(f"  {cls_name:15}: {val:.2%}")
        print(f"  {'mIoU':15}: {np.mean(all_miou):.2%}" if all_miou else "  {'mIoU':15}: N/A")
        print(f"  {'Pixel Acc':15}: {np.mean(all_acc):.2%}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Verify pseudolabels vs ground truth for OpenEarthMap',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python src/verify_pseudolabels.py --pseudo_root data/OEM_v2_aDanh
    python src/verify_pseudolabels.py --sample muenster_5 --output verify_output
    python src/verify_pseudolabels.py --analyze --max-samples 3
        """
    )

    parser.add_argument('--pseudo_root', type=str, default='data/OEM_v2_aDanh',
                        help='Root directory containing labels/ and pseudolabels/')
    parser.add_argument('--region', type=str, default=None,
                        help='Filter by region (e.g., muenster, rio)')
    parser.add_argument('--sample', type=str, default=None,
                        help='Verify single sample by name')
    parser.add_argument('--max-samples', type=int, default=20,
                        help='Max samples to verify (0 = all)')
    parser.add_argument('--output', type=str, default='verify_output',
                        help='Output directory for visualizations')
    parser.add_argument('--analyze', action='store_true',
                        help='Print detailed class distribution analysis')

    args = parser.parse_args()

    pseudo_root = Path(args.pseudo_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pseudo root: {pseudo_root}")
    print(f"Output dir: {output_dir}")

    if args.sample:
        # Single sample mode
        sample_name = args.sample
        if not sample_name.endswith('.tif'):
            sample_name += '.tif'

        found = False
        for region in ['', 'muenster', 'rio', 'khartoum', 'paris', 'palu',
                       'taipei', 'chicago', 'washington', 'aachen', 'abancay']:
            base = pseudo_root / region if region else pseudo_root

            image_path = base / 'images' / sample_name
            label_path = base / 'labels' / sample_name
            pseudo_path = base / 'pseudolabels' / sample_name

            if image_path.exists() and label_path.exists() and pseudo_path.exists():
                if args.analyze:
                    analyze_distribution(label_path, pseudo_path, sample_name)
                else:
                    ious, acc = visualize_sample(image_path, label_path, pseudo_path, output_dir)
                    if ious:
                        print(f"\nResults for {sample_name}:")
                        for cls, iou in ious.items():
                            if isinstance(iou, float):
                                print(f"  {cls:15}: {iou:.2%}")
                        print(f"  Pixel Accuracy: {acc:.2%}")
                found = True
                break

        if not found:
            print(f"Error: Could not find all files for {sample_name}")
            sys.exit(1)

    else:
        # Batch mode
        batch_verify(
            pseudo_root,
            args.region,
            args.max_samples,
            output_dir,
            args.analyze
        )


if __name__ == '__main__':
    main()
