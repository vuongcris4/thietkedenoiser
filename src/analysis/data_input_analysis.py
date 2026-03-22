"""
Phân tích chi tiết dữ liệu INPUT (Ground Truth và Pseudolabel).
Xuất CSV với per-class distribution cho từng sample.
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

CLASS_VALUES = [0, 2, 3, 4, 5, 6, 7, 8, 9]  # Values used in label files


def analyze_sample(gt: np.ndarray, pred: np.ndarray):
    """Analyze single sample: class distribution for GT and Pseudolabel."""
    results = {
        'gt_pixels': {},
        'pred_pixels': {},
        'gt_pct': {},
        'pred_pct': {},
    }

    total = gt.size

    for cls in CLASS_VALUES:
        gt_count = (gt == cls).sum()
        pred_count = (pred == cls).sum()

        results['gt_pixels'][cls] = int(gt_count)
        results['pred_pixels'][cls] = int(pred_count)
        results['gt_pct'][cls] = gt_count / total * 100
        results['pred_pct'][cls] = pred_count / total * 100

    # Stats
    results['total_pixels'] = total
    results['gt_unique_count'] = len(np.unique(gt))
    results['pred_unique_count'] = len(np.unique(pred))

    return results


def main():
    pseudo_root = Path('data/OEM_v2_aDanh')
    label_dir = pseudo_root / 'labels'
    pseudo_dir = pseudo_root / 'pseudolabels'
    output_dir = Path('data_analysis_output')
    output_dir.mkdir(exist_ok=True)

    pseudo_files = sorted(list(pseudo_dir.glob('*.tif')))

    print(f'Tìm thấy {len(pseudo_files)} pseudolabel files')
    print('Đang phân tích data input...')

    # Per-sample results
    sample_results = []

    # Aggregate statistics
    gt_total_pixels = 0
    pred_total_pixels = 0
    gt_class_pixels = defaultdict(int)
    pred_class_pixels = defaultdict(int)

    for i, pseudo_file in enumerate(pseudo_files):
        label_file = label_dir / pseudo_file.name

        if not label_file.exists():
            continue

        gt = cv2.imread(str(label_file), cv2.IMREAD_UNCHANGED)
        pred = cv2.imread(str(pseudo_file), cv2.IMREAD_UNCHANGED)

        if gt is None or pred is None:
            continue

        # Analyze
        analysis = analyze_sample(gt, pred)

        # Build row
        row = {
            'sample': pseudo_file.name,
            'height': gt.shape[0],
            'width': gt.shape[1],
            'total_pixels': analysis['total_pixels'],
            'gt_unique_classes': analysis['gt_unique_count'],
            'pred_unique_classes': analysis['pred_unique_count'],
        }

        # Add per-class pixels and percentage
        for cls in CLASS_VALUES:
            cls_name = CLASS_NAMES.get(cls, f'Class_{cls}')
            row[f'GT_{cls_name}_pixels'] = analysis['gt_pixels'][cls]
            row[f'GT_{cls_name}_pct'] = analysis['gt_pct'][cls]
            row[f'Pred_{cls_name}_pixels'] = analysis['pred_pixels'][cls]
            row[f'Pred_{cls_name}_pct'] = analysis['pred_pct'][cls]

        sample_results.append(row)

        # Aggregate
        gt_total_pixels += gt.size
        pred_total_pixels += pred.size
        for cls in CLASS_VALUES:
            gt_class_pixels[cls] += (gt == cls).sum()
            pred_class_pixels[cls] += (pred == cls).sum()

        if (i + 1) % 500 == 0:
            print(f'  Đã xử lý {i + 1}/{len(pseudo_files)} samples...')

    print(f'\nHoàn thành! Phân tích {len(sample_results)} samples')

    # Write per-sample CSV
    csv_file = output_dir / 'per_sample_data.csv'
    fieldnames = list(sample_results[0].keys())

    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_results)

    print(f'Đã xuất: {csv_file}')

    # Write summary CSV
    summary_file = output_dir / 'data_summary.csv'

    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(['=== DATA INPUT ANALYSIS SUMMARY ==='])
        writer.writerow([f'Analysis of {len(sample_results)} samples'])
        writer.writerow([])

        # Dataset overview
        writer.writerow(['=== DATASET OVERVIEW ==='])
        writer.writerow(['Total samples', len(sample_results)])
        writer.writerow(['Total pixels (GT)', f'{gt_total_pixels:,}'])
        writer.writerow(['Total pixels (Pseudo)', f'{pred_total_pixels:,}'])
        writer.writerow(['Image size', f'{sample_results[0]["height"]}x{sample_results[0]["width"]}'])
        writer.writerow([])

        # Per-class distribution
        writer.writerow(['=== CLASS DISTRIBUTION (GROUND TRUTH) ==='])
        writer.writerow(['Class', 'Value', 'Total Pixels', 'Percentage', 'Samples Present'])

        gt_samples_with_class = defaultdict(int)
        for r in sample_results:
            for cls in CLASS_VALUES:
                cls_name = CLASS_NAMES.get(cls, f'Class_{cls}')
                if r.get(f'GT_{cls_name}_pixels', 0) > 0:
                    gt_samples_with_class[cls] += 1

        for cls in CLASS_VALUES:
            cls_name = CLASS_NAMES.get(cls, f'Class_{cls}')
            pixels = gt_class_pixels[cls]
            pct = pixels / gt_total_pixels * 100 if gt_total_pixels > 0 else 0
            samples_count = gt_samples_with_class[cls]
            writer.writerow([cls_name, cls, f'{pixels:,}', f'{pct:.2f}%', f'{samples_count}/{len(sample_results)}'])

        writer.writerow([])

        writer.writerow(['=== CLASS DISTRIBUTION (PSEUDOLABEL) ==='])
        writer.writerow(['Class', 'Value', 'Total Pixels', 'Percentage', 'Samples Present'])

        pred_samples_with_class = defaultdict(int)
        for r in sample_results:
            for cls in CLASS_VALUES:
                cls_name = CLASS_NAMES.get(cls, f'Class_{cls}')
                if r.get(f'Pred_{cls_name}_pixels', 0) > 0:
                    pred_samples_with_class[cls] += 1

        for cls in CLASS_VALUES:
            cls_name = CLASS_NAMES.get(cls, f'Class_{cls}')
            pixels = pred_class_pixels[cls]
            pct = pixels / pred_total_pixels * 100 if pred_total_pixels > 0 else 0
            samples_count = pred_samples_with_class[cls]
            writer.writerow([cls_name, cls, f'{pixels:,}', f'{pct:.2f}%', f'{samples_count}/{len(sample_results)}'])

        writer.writerow([])

        # Comparison
        writer.writerow(['=== GT vs PSEUDOLABEL COMPARISON ==='])
        writer.writerow(['Class', 'GT %', 'Pseudo %', 'Difference', 'Bias'])
        writer.writerow([])

        for cls in [2, 3, 4, 5, 6, 7, 8, 9]:  # Exclude background
            cls_name = CLASS_NAMES.get(cls, f'Class_{cls}')
            gt_pct = gt_class_pixels[cls] / gt_total_pixels * 100
            pred_pct = pred_class_pixels[cls] / pred_total_pixels * 100
            diff = pred_pct - gt_pct

            if abs(diff) > 1.0:
                bias = 'OVER' if diff > 0 else 'UNDER'
            else:
                bias = 'Balanced'

            writer.writerow([cls_name, f'{gt_pct:.2f}%', f'{pred_pct:.2f}%', f'{diff:+.2f}%', bias])

        writer.writerow([])

        # Per-sample statistics
        writer.writerow(['=== PER-SAMPLE STATISTICS ==='])
        writer.writerow([])

        # GT stats
        gt_pcts_per_sample = []
        pred_pcts_per_sample = []

        for cls in [2, 3, 4, 5, 6, 7, 8, 9]:
            cls_name = CLASS_NAMES.get(cls, f'Class_{cls}')
            gt_vals = [r[f'GT_{cls_name}_pct'] for r in sample_results if r[f'GT_{cls_name}_pct'] > 0]
            pred_vals = [r[f'Pred_{cls_name}_pct'] for r in sample_results if r[f'Pred_{cls_name}_pct'] > 0]

            if gt_vals:
                gt_pcts_per_sample.append({
                    'class': cls_name,
                    'mean': np.mean(gt_vals),
                    'std': np.std(gt_vals),
                    'min': np.min(gt_vals),
                    'max': np.max(gt_vals),
                    'median': np.median(gt_vals)
                })

            if pred_vals:
                pred_pcts_per_sample.append({
                    'class': cls_name,
                    'mean': np.mean(pred_vals),
                    'std': np.std(pred_vals),
                    'min': np.min(pred_vals),
                    'max': np.max(pred_vals),
                    'median': np.median(pred_vals)
                })

        writer.writerow(['=== PER-CLASS STATS ACROSS SAMPLES (GT) ==='])
        writer.writerow(['Class', 'Mean %', 'Std %', 'Min %', 'Max %', 'Median %'])
        for stats in gt_pcts_per_sample:
            writer.writerow([
                stats['class'],
                f"{stats['mean']:.2f}",
                f"{stats['std']:.2f}",
                f"{stats['min']:.2f}",
                f"{stats['max']:.2f}",
                f"{stats['median']:.2f}"
            ])

        writer.writerow([])

        writer.writerow(['=== PER-CLASS STATS ACROSS SAMPLES (PSEUDO) ==='])
        writer.writerow(['Class', 'Mean %', 'Std %', 'Min %', 'Max %', 'Median %'])
        for stats in pred_pcts_per_sample:
            writer.writerow([
                stats['class'],
                f"{stats['mean']:.2f}",
                f"{stats['std']:.2f}",
                f"{stats['min']:.2f}",
                f"{stats['max']:.2f}",
                f"{stats['median']:.2f}"
            ])

    print(f'Đã xuất: {summary_file}')

    # Print summary to console
    print('\n' + '='*70)
    print('TÓM TẮT PHÂN TÍCH DATA INPUT')
    print('='*70)
    print(f'Tổng số samples: {len(sample_results)}')
    print(f'Kích thước ảnh: {sample_results[0]["height"]}x{sample_results[0]["width"]}')
    print()

    print('GROUND TRUTH - Class Distribution:')
    for cls in [2, 3, 4, 5, 6, 7, 8, 9]:
        cls_name = CLASS_NAMES.get(cls, f'Class_{cls}')
        pixels = gt_class_pixels[cls]
        pct = pixels / gt_total_pixels * 100
        print(f'  {cls_name:15}: {pixels:>15,} px ({pct:5.2f}%)')

    print()
    print('PSEUDOLABEL - Class Distribution:')
    for cls in [2, 3, 4, 5, 6, 7, 8, 9]:
        cls_name = CLASS_NAMES.get(cls, f'Class_{cls}')
        pixels = pred_class_pixels[cls]
        pct = pixels / pred_total_pixels * 100
        print(f'  {cls_name:15}: {pixels:>15,} px ({pct:5.2f}%)')

    print()
    print('SO SÁNH (Pseudo - GT):')
    for cls in [2, 3, 4, 5, 6, 7, 8, 9]:
        cls_name = CLASS_NAMES.get(cls, f'Class_{cls}')
        gt_pct = gt_class_pixels[cls] / gt_total_pixels * 100
        pred_pct = pred_class_pixels[cls] / pred_total_pixels * 100
        diff = pred_pct - gt_pct
        bias = 'OVER ↑' if diff > 1 else ('UNDER ↓' if diff < -1 else 'Balanced')
        print(f'  {cls_name:15}: GT={gt_pct:5.2f}%  Pseudo={pred_pct:5.2f}%  Diff={diff:+.2f}%  [{bias}]')

    print()
    print(f'Files đã xuất:')
    print(f'  - {csv_file} (chi tiết từng sample)')
    print(f'  - {summary_file} (tổng hợp)')


if __name__ == '__main__':
    main()
