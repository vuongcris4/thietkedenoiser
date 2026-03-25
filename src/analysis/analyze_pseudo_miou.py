"""
Phân tích mIoU của pseudo-labels so với ground truth theo từng class.

Usage:
    # Phân tích toàn bộ train/val/test
    python src/analysis/analyze_pseudo_miou.py --pseudo_root data/OEM_v2_aDanh

    # Chỉ phân tích tập val
    python src/analysis/analyze_pseudo_miou.py --pseudo_root data/OEM_v2_aDanh --splits val

    # Lưu kết quả ra file CSV
    python src/analysis/analyze_pseudo_miou.py --pseudo_root data/OEM_v2_aDanh --output results/pseudo_miou_analysis.csv
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import json

import cv2
import numpy as np
import pandas as pd

# =============================================================================
# CLASS MAPPING - Đồng bộ với dataset.py
# =============================================================================

# 7 classes thực tế trong dataset OEM_v2_aDanh
CLASS_NAMES = [
    'Bareland',    # value 2
    'Rangeland',   # value 3
    'Developed',   # value 4
    'Road',        # value 5
    'Tree',        # value 6
    'Water',       # value 7
    'Agriculture', # value 8
]
NUM_CLASSES = 7  # Chỉ có 7 classes (không có Building)

# Label values trong file: 0-8
#   0, 1: background/void
#   2-8: 7 classes
# Sau khi remap (trừ 2): 2→0, 3→1, ..., 8→6
LABEL_VALUE_OFFSET = 2

# Class index → name
CLASS_VALUE_TO_NAME = {
    0: 'Bareland',
    1: 'Rangeland',
    2: 'Developed',
    3: 'Road',
    4: 'Tree',
    5: 'Water',
    6: 'Agriculture',
}


def remap_label(mask: np.ndarray) -> np.ndarray:
    """
    Remap label values từ file (2-8) sang class indices (0-6).
    Values 0,1 → background (loại bỏ)
    """
    mask = mask.astype(np.int16) - LABEL_VALUE_OFFSET
    mask[mask < 0] = -1  # Values 0,1 → background → -1 (không tính IoU)
    mask[mask >= NUM_CLASSES] = -1  # Values > 8 → -1
    return mask.astype(np.int16)


def compute_per_class_iou(mask_pred: np.ndarray, mask_gt: np.ndarray) -> Dict[str, float]:
    """
    Tính IoU cho từng class.

    Args:
        mask_pred: Pseudo-label (đã remap)
        mask_gt: Ground truth (đã remap)

    Returns:
        Dict: {class_name: IoU, 'mIoU': mean IoU}
    """
    ious = {}

    for cls_idx in range(NUM_CLASSES):
        pred_mask = (mask_pred == cls_idx)
        gt_mask = (mask_gt == cls_idx)

        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()

        if union > 0:
            ious[CLASS_NAMES[cls_idx]] = intersection / union
        else:
            # Class không xuất hiện trong sample này
            ious[CLASS_NAMES[cls_idx]] = np.nan

    # mIoU: trung bình các classes có dữ liệu
    valid_ious = [v for v in ious.values() if not np.isnan(v)]
    ious['mIoU'] = np.mean(valid_ious) if valid_ious else 0.0

    return ious


def analyze_split(
    pseudo_root: Path,
    split: str,
    max_samples: int = None
) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    """
    Phân tích IoU cho một split (train/val/test).

    Args:
        pseudo_root: Root directory chứa images/, labels/, pseudolabels/
        split: 'train', 'val', hoặc 'test'
        max_samples: Số samples tối đa để phân tích (None = tất cả)

    Returns:
        class_ious: Dict {class_name: [IoU values per sample]}
        summary: Dict chứa mIoU trung bình và số samples
    """
    split_file = pseudo_root / f'{split}.txt'
    if not split_file.exists():
        print(f"  Skip {split}: split file not found")
        return {}, {}

    # Đọc danh sách files
    with open(split_file) as f:
        filenames = [l.strip() for l in f if l.strip()]

    if max_samples:
        filenames = filenames[:max_samples]

    print(f"  Analyzing {split}: {len(filenames)} samples")

    # Tích lũy IoU cho từng class
    class_ious = {name: [] for name in CLASS_NAMES}
    sample_mious = []
    valid_count = 0

    label_dir = pseudo_root / 'labels'
    pseudo_dir = pseudo_root / 'pseudolabels'

    for filename in filenames:
        label_path = label_dir / filename
        pseudo_path = pseudo_dir / filename

        if not label_path.exists() or not pseudo_path.exists():
            continue

        # Đọc và remap labels
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        pseudo = cv2.imread(str(pseudo_path), cv2.IMREAD_UNCHANGED)

        if label is None or pseudo is None:
            continue

        label = remap_label(label)
        pseudo = remap_label(pseudo)

        # Tính IoU
        ious = compute_per_class_iou(pseudo, label)

        # Tích lũy
        for cls_name in CLASS_NAMES:
            if not np.isnan(ious[cls_name]):
                class_ious[cls_name].append(ious[cls_name])

        if not np.isnan(ious['mIoU']):
            sample_mious.append(ious['mIoU'])
            valid_count += 1

    # Tính thống kê tổng hợp
    summary = {
        'split': split,
        'num_samples': len(filenames),
        'valid_samples': valid_count,
        'mean_mIoU': np.mean(sample_mious) if sample_mious else 0.0,
        'std_mIoU': np.std(sample_mious) if sample_mious else 0.0,
        'min_mIoU': np.min(sample_mious) if sample_mious else 0.0,
        'max_mIoU': np.max(sample_mious) if sample_mious else 0.0,
    }

    # Thêm mean IoU cho từng class
    for cls_name in CLASS_NAMES:
        values = class_ious[cls_name]
        summary[f'{cls_name}_mean'] = np.mean(values) if values else 0.0
        summary[f'{cls_name}_std'] = np.std(values) if values else 0.0

    return class_ious, summary


def print_results(all_summaries: List[Dict], output_file: str = None):
    """In kết quả phân tích ra màn hình và/hoặc file."""

    print("\n" + "=" * 80)
    print("PHÂN TÍCH mIoU: PSEUDO-LABELS vs GROUND TRUTH")
    print("=" * 80)

    for summary in all_summaries:
        if not summary:
            continue

        split = summary.get('split', 'unknown')
        print(f"\n{'='*80}")
        print(f"SPLIT: {split.upper()}")
        print(f"{'='*80}")
        print(f"  Số samples: {summary.get('num_samples', 0)}")
        print(f"  Valid samples: {summary.get('valid_samples', 0)}")

        print(f"\n  mIoU STATISTICS:")
        print(f"    Mean mIoU: {summary.get('mean_mIoU', 0):.2%}")
        print(f"    Std mIoU:  {summary.get('std_mIoU', 0):.2%}")
        print(f"    Min mIoU:  {summary.get('min_mIoU', 0):.2%}")
        print(f"    Max mIoU:  {summary.get('max_mIoU', 0):.2%}")

        print(f"\n  PER-CLASS IoU:")
        for cls_name in CLASS_NAMES:
            mean_iou = summary.get(f'{cls_name}_mean', 0)
            std_iou = summary.get(f'{cls_name}_std', 0)
            print(f"    {cls_name:15}: {mean_iou:6.2%} ± {std_iou:6.2%}")

    # Bảng so sánh giữa các splits
    print(f"\n{'='*80}")
    print("SO SÁNH GIỮA CÁC TẬP DATA")
    print(f"{'='*80}")

    # Header
    header = f"{'Class':15}"
    for s in all_summaries:
        if s:
            header += f" | {s.get('split', '???'):>12}"
    print(header)
    print("-" * len(header))

    # Per-class rows
    for cls_name in CLASS_NAMES:
        row = f"{cls_name:15}"
        for s in all_summaries:
            if s:
                val = s.get(f'{cls_name}_mean', 0)
                row += f" | {val:10.2%}"
        print(row)

    # mIoU row
    row = f"{'mIoU (mean)':15}"
    for s in all_summaries:
        if s:
            val = s.get('mean_mIoU', 0)
            row += f" | {val:10.2%}"
    print(row)

    # Samples row
    row = f"{'# Samples':15}"
    for s in all_summaries:
        if s:
            val = s.get('valid_samples', 0)
            row += f" | {val:12}"
    print(row)

    print("=" * 80)

    # Lưu ra CSV nếu được yêu cầu
    if output_file:
        save_to_csv(all_summaries, output_file)


def save_to_csv(all_summaries: List[Dict], output_file: str):
    """Lưu kết quả ra file CSV."""

    rows = []
    for summary in all_summaries:
        if not summary:
            continue
        split = summary.get('split', 'unknown')

        # Per-class IoU
        for cls_name in CLASS_NAMES:
            rows.append({
                'Split': split,
                'Class': cls_name,
                'Mean_IoU': summary.get(f'{cls_name}_mean', 0),
                'Std_IoU': summary.get(f'{cls_name}_std', 0),
                'Metric_Type': 'per_class'
            })

        # mIoU summary
        rows.append({
            'Split': split,
            'Class': 'mIoU',
            'Mean_IoU': summary.get('mean_mIoU', 0),
            'Std_IoU': summary.get('std_mIoU', 0),
            'Min_IoU': summary.get('min_mIoU', 0),
            'Max_IoU': summary.get('max_mIoU', 0),
            'Num_Samples': summary.get('valid_samples', 0),
            'Metric_Type': 'summary'
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"\nKết quả đã lưu vào: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Phân tích mIoU của pseudo-labels vs ground truth',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--pseudo_root',
        type=str,
        default='data/OEM_v2_aDanh',
        help='Root directory chứa labels/ và pseudolabels/'
    )
    parser.add_argument(
        '--splits',
        type=str,
        default='train,val,test',
        help='Danh sách splits cần phân tích (comma-separated)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Số samples tối đa mỗi split (None = tất cả)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='File CSV để lưu kết quả'
    )

    args = parser.parse_args()

    pseudo_root = Path(args.pseudo_root)
    splits = [s.strip() for s in args.splits.split(',')]

    print(f"Pseudo root: {pseudo_root}")
    print(f"Splits to analyze: {splits}")

    all_summaries = []
    all_class_ious = {}

    for split in splits:
        class_ious, summary = analyze_split(
            pseudo_root, split, args.max_samples
        )
        all_summaries.append(summary)
        all_class_ious[split] = class_ious

    # In kết quả
    print_results(all_summaries, args.output)

    # Gợi ý cải thiện
    print("\nNHẬN XÉT:")
    for summary in all_summaries:
        if not summary:
            continue
        split = summary.get('split', 'unknown')
        mean_miou = summary.get('mean_mIoU', 0)

        if mean_miou < 0.5:
            print(f"  - {split}: mIoU thấp ({mean_miou:.2%}), pseudo-labels có nhiều noise")
        elif mean_miou < 0.7:
            print(f"  - {split}: mIoU trung bình ({mean_miou:.2%}), có thể cải thiện với DAE")
        else:
            print(f"  - {split}: mIoU cao ({mean_miou:.2%}), pseudo-labels khá sát ground truth")


if __name__ == '__main__':
    main()
