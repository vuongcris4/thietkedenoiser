"""
Phân tích chi tiết Ground Truth vs Pseudolabel cho từng sample.
Xuất CSV với per-class IoU và statistics.
"""

import cv2
import numpy as np
from pathlib import Path
import csv
from collections import defaultdict

CLASS_NAMES = {
    0: 'Background',
    2: 'Bareland',
    3: 'Rangeland',
    4: 'Developed',
    5: 'Road',
    6: 'Tree',
    7: 'Water',
    8: 'Agriculture',
    9: 'Building'
}

def compute_per_class_iou(gt: np.ndarray, pred: np.ndarray):
    """Compute per-class IoU for all class values."""
    all_classes = [0, 2, 3, 4, 5, 6, 7, 8, 9]
    ious = {}

    for cls in all_classes:
        gt_cls = (gt == cls)
        pred_cls = (pred == cls)

        intersection = np.logical_and(gt_cls, pred_cls).sum()
        union = np.logical_or(gt_cls, pred_cls).sum()

        if union > 0:
            ious[cls] = intersection / union
        else:
            ious[cls] = np.nan

    # mIoU (excluding NaN and background)
    valid_ious = [v for k, v in ious.items() if k != 0 and not np.isnan(v)]
    ious['mIoU'] = np.mean(valid_ious) if valid_ious else 0.0

    # Pixel accuracy
    ious['pixel_acc'] = (gt == pred).sum() / gt.size

    return ious


def analyze_sample(gt: np.ndarray, pred: np.ndarray):
    """Analyze single sample: class distribution + IoU."""
    results = {
        'gt_pixels': {},
        'pred_pixels': {},
        'gt_pct': {},
        'pred_pct': {},
    }

    total = gt.size

    for cls in [0, 2, 3, 4, 5, 6, 7, 8, 9]:
        gt_count = (gt == cls).sum()
        pred_count = (pred == cls).sum()

        results['gt_pixels'][cls] = gt_count
        results['pred_pixels'][cls] = pred_count
        results['gt_pct'][cls] = gt_count / total * 100
        results['pred_pct'][cls] = pred_count / total * 100

    return results


def main():
    pseudo_root = Path('data/OEM_v2_aDanh')
    label_dir = pseudo_root / 'labels'
    pseudo_dir = pseudo_root / 'pseudolabels'
    output_dir = Path('analysis_output')
    output_dir.mkdir(exist_ok=True)

    pseudo_files = sorted(list(pseudo_dir.glob('*.tif')))

    print(f'Tìm thấy {len(pseudo_files)} pseudolabel files')
    print('Đang phân tích...')

    # Per-sample results
    sample_results = []

    # Aggregate statistics
    class_iou_sum = defaultdict(float)
    class_iou_count = defaultdict(int)
    class_gt_pct_sum = defaultdict(float)
    class_pred_pct_sum = defaultdict(float)

    total_pixels = 0

    for i, pseudo_file in enumerate(pseudo_files):
        label_file = label_dir / pseudo_file.name

        if not label_file.exists():
            continue

        gt = cv2.imread(str(label_file), cv2.IMREAD_UNCHANGED)
        pred = cv2.imread(str(pseudo_file), cv2.IMREAD_UNCHANGED)

        if gt is None or pred is None:
            continue

        # Compute metrics
        ious = compute_per_class_iou(gt, pred)
        analysis = analyze_sample(gt, pred)

        # Build row
        row = {
            'sample': pseudo_file.name,
            'height': gt.shape[0],
            'width': gt.shape[1],
            'total_pixels': gt.size,
        }

        # Add per-class IoU
        for cls in [0, 2, 3, 4, 5, 6, 7, 8, 9]:
            cls_name = CLASS_NAMES.get(cls, f'Class_{cls}')
            row[f'IoU_{cls_name}'] = ious.get(cls, np.nan)
            row[f'GT_{cls_name}_pixels'] = analysis['gt_pixels'][cls]
            row[f'Pred_{cls_name}_pixels'] = analysis['pred_pixels'][cls]
            row[f'GT_{cls_name}_pct'] = analysis['gt_pct'][cls]
            row[f'Pred_{cls_name}_pct'] = analysis['pred_pct'][cls]

        # Add overall metrics
        row['mIoU'] = ious['mIoU']
        row['Pixel_Accuracy'] = ious['pixel_acc']

        sample_results.append(row)

        # Aggregate
        total_pixels += gt.size
        for cls in [2, 3, 4, 5, 6, 7, 8, 9]:  # Exclude background
            cls_name = CLASS_NAMES.get(cls, f'Class_{cls}')
            if not np.isnan(ious.get(cls, np.nan)):
                class_iou_sum[cls_name] += ious[cls]
                class_iou_count[cls_name] += 1
            class_gt_pct_sum[cls_name] += analysis['gt_pct'][cls]
            class_pred_pct_sum[cls_name] += analysis['pred_pct'][cls]

        if (i + 1) % 100 == 0:
            print(f'  Đã xử lý {i + 1}/{len(pseudo_files)} samples...')

    print(f'\nHoàn thành! Phân tích {len(sample_results)} samples')

    # Write per-sample CSV
    csv_file = output_dir / 'per_sample_analysis.csv'
    fieldnames = list(sample_results[0].keys())

    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_results)

    print(f'Đã xuất: {csv_file}')

    # Write summary CSV
    summary_file = output_dir / 'summary_analysis.csv'

    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(['=== PER-CLASS ANALYSIS SUMMARY ==='])
        writer.writerow([])

        # Per-class statistics
        writer.writerow(['Class', 'Avg IoU', 'Avg GT %', 'Avg Pseudo %', 'Diff %', 'Samples with class'])
        writer.writerow([])

        for cls in [2, 3, 4, 5, 6, 7, 8, 9]:
            cls_name = CLASS_NAMES.get(cls, f'Class_{cls}')
            avg_iou = class_iou_sum[cls_name] / class_iou_count[cls_name] if class_iou_count[cls_name] > 0 else 0
            avg_gt_pct = class_gt_pct_sum[cls_name] / len(sample_results)
            avg_pred_pct = class_pred_pct_sum[cls_name] / len(sample_results)
            diff = avg_pred_pct - avg_gt_pct
            n_samples = class_iou_count[cls_name]

            writer.writerow([
                cls_name,
                f'{avg_iou:.2%}',
                f'{avg_gt_pct:.2f}%',
                f'{avg_pred_pct:.2f}%',
                f'{diff:+.2f}%',
                f'{n_samples}/{len(sample_results)}'
            ])

        writer.writerow([])

        # Overall statistics
        writer.writerow(['=== OVERALL STATISTICS ==='])
        writer.writerow([])

        all_miou = [r['mIoU'] for r in sample_results]
        all_acc = [r['Pixel_Accuracy'] for r in sample_results]

        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total samples', len(sample_results)])
        writer.writerow(['Total pixels', f'{total_pixels:,}'])
        writer.writerow(['Mean mIoU', f'{np.mean(all_miou):.2%}'])
        writer.writerow(['Std mIoU', f'{np.std(all_miou):.2%}'])
        writer.writerow(['Min mIoU', f'{np.min(all_miou):.2%}'])
        writer.writerow(['Max mIoU', f'{np.max(all_miou):.2%}'])
        writer.writerow(['Mean Pixel Acc', f'{np.mean(all_acc):.2%}'])
        writer.writerow(['Std Pixel Acc', f'{np.std(all_acc):.2%}'])

        writer.writerow([])

        # Top/bottom samples
        writer.writerow(['=== TOP 10 SAMPLES BY mIoU ==='])
        writer.writerow(['Sample', 'mIoU', 'Pixel Acc'])

        top_samples = sorted(sample_results, key=lambda x: x['mIoU'], reverse=True)[:10]
        for r in top_samples:
            writer.writerow([r['sample'], f"{r['mIoU']:.2%}", f"{r['Pixel_Accuracy']:.2%}"])

        writer.writerow([])
        writer.writerow(['=== BOTTOM 10 SAMPLES BY mIoU ==='])
        writer.writerow(['Sample', 'mIoU', 'Pixel Acc'])

        bottom_samples = sorted(sample_results, key=lambda x: x['mIoU'])[:10]
        for r in bottom_samples:
            writer.writerow([r['sample'], f"{r['mIoU']:.2%}", f"{r['Pixel_Accuracy']:.2%}"])

        writer.writerow([])

        # Class bias analysis
        writer.writerow(['=== CLASS BIAS ANALYSIS ==='])
        writer.writerow(['Class', 'Bias Direction', 'Avg GT %', 'Avg Pseudo %', 'Bias Magnitude'])

        for cls in [2, 3, 4, 5, 6, 7, 8, 9]:
            cls_name = CLASS_NAMES.get(cls, f'Class_{cls}')
            avg_gt = class_gt_pct_sum[cls_name] / len(sample_results)
            avg_pred = class_pred_pct_sum[cls_name] / len(sample_results)
            diff = avg_pred - avg_gt

            if abs(diff) > 1.0:
                direction = 'OVER-predicted' if diff > 0 else 'UNDER-predicted'
            else:
                direction = 'Balanced'

            writer.writerow([cls_name, direction, f'{avg_gt:.2f}%', f'{avg_pred:.2f}%', f'{diff:+.2f}%'])

    print(f'Đã xuất: {summary_file}')

    # Print summary to console
    print('\n' + '='*70)
    print('TÓM TẮT PHÂN TÍCH')
    print('='*70)
    print(f'Tổng số samples: {len(sample_results)}')
    print(f'Tổng pixels: {total_pixels:,}')
    print(f'Mean mIoU: {np.mean(all_miou):.2%}')
    print(f'Mean Pixel Accuracy: {np.mean(all_acc):.2%}')
    print()

    print('Per-class IoU:')
    for cls in [2, 3, 4, 5, 6, 7, 8, 9]:
        cls_name = CLASS_NAMES.get(cls, f'Class_{cls}')
        avg_iou = class_iou_sum[cls_name] / class_iou_count[cls_name] if class_iou_count[cls_name] > 0 else 0
        print(f'  {cls_name:15}: {avg_iou:.2%}')

    print()
    print('Class Bias:')
    for cls in [2, 3, 4, 5, 6, 7, 8, 9]:
        cls_name = CLASS_NAMES.get(cls, f'Class_{cls}')
        avg_gt = class_gt_pct_sum[cls_name] / len(sample_results)
        avg_pred = class_pred_pct_sum[cls_name] / len(sample_results)
        diff = avg_pred - avg_gt
        print(f'  {cls_name:15}: GT={avg_gt:5.2f}%  Pseudo={avg_pred:5.2f}%  Diff={diff:+.2f}%')

    print()
    print(f'Files đã xuất:')
    print(f'  - {csv_file}')
    print(f'  - {summary_file}')


if __name__ == '__main__':
    main()
