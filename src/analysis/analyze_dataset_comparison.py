"""
Phân tích dataset: So sánh pseudo-label với ground truth
- Tính mIoU, Precision, Recall cho từng class
- Visualize sự khác biệt giữa pseudo-label và GT
"""

import os
import sys
import numpy as np
import cv2
from tqdm import tqdm
from typing import List, Tuple, Dict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.dataset import find_oem_pairs

# =============================================================================
# CONSTANTS - OVERRIDE (dataset không có Bareland)
# =============================================================================
# 7 lớp phân loại đất/vật thể trong OpenEarthMap (KHÔNG có Bareland)
# Thứ tự: 0=Rangeland, 1=Developed, 2=Road, 3=Tree, 4=Water, 5=Agriculture, 6=Building
CLASS_NAMES = [
    'Rangeland', 'Developed', 'Road', 'Tree',
    'Water', 'Agriculture', 'Building'
]
NUM_CLASSES = 7

# =============================================================================
# METRICS FUNCTIONS
# =============================================================================

def compute_confusion_matrix(gt: np.ndarray, pred: np.ndarray, num_classes: int = 7) -> np.ndarray:
    """
    Tính confusion matrix giữa GT và prediction.

    Args:
        gt: Ground truth mask [H, W], values 0-7
        pred: Prediction mask [H, W], values 0-7
        num_classes: Số lớp (8)

    Returns:
        Confusion matrix [num_classes, num_classes]
        cm[i, j] = số pixel có GT = i, prediction = j
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    mask = (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
    gt_valid = gt[mask]
    pred_valid = pred[mask]

    for i in range(num_classes):
        for j in range(num_classes):
            cm[i, j] = np.sum((gt_valid == i) & (pred_valid == j))

    return cm


def compute_metrics_per_class(cm: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Tính IoU, Precision, Recall từ confusion matrix.

    Args:
        cm: Confusion matrix [num_classes, num_classes]

    Returns:
        Dict chứa:
            - iou: IoU cho từng class [num_classes]
            - precision: Precision cho từng class [num_classes]
            - recall: Recall cho từng class [num_classes]
            - f1: F1-score cho từng class [num_classes]
    """
    num_classes = cm.shape[0]
    iou = np.zeros(num_classes)
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)

    for cls in range(num_classes):
        # TP: gt=cls, pred=cls
        tp = cm[cls, cls]
        # FP: gt!=cls, pred=cls
        fp = np.sum(cm[:, cls]) - tp
        # FN: gt=cls, pred!=cls
        fn = np.sum(cm[cls, :]) - tp
        # TN: gt!=cls, pred!=cls
        tn = np.sum(cm) - tp - fp - fn

        # IoU = TP / (TP + FP + FN)
        denom = tp + fp + fn
        if denom > 0:
            iou[cls] = tp / denom
        else:
            iou[cls] = 0.0

        # Precision = TP / (TP + FP)
        if tp + fp > 0:
            precision[cls] = tp / (tp + fp)
        else:
            precision[cls] = 0.0

        # Recall = TP / (TP + FN)
        if tp + fn > 0:
            recall[cls] = tp / (tp + fn)
        else:
            recall[cls] = 0.0

        # F1 = 2 * (P * R) / (P + R)
        if precision[cls] + recall[cls] > 0:
            f1[cls] = 2 * (precision[cls] * recall[cls]) / (precision[cls] + recall[cls])
        else:
            f1[cls] = 0.0

    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def find_dataset_pairs(data_root: str, split_file: str) -> List[Tuple[str, str, str]]:
    """
    Tìm tất cả cặp (label, pseudo) từ file split.

    Dataset structure (flat):
        data_root/
        ├── images/
        │   └── {filename}.tif
        ├── labels/
        │   └── {filename}.tif
        ├── pseudolabels/
        │   └── {filename}.tif
        └── split.txt

    Args:
        data_root: Đường dẫn gốc dataset
        split_file: Path đến file split (train.txt, val.txt, test.txt)

    Returns:
        List[Tuple[str, str, str]]: Danh sách (label_path, pseudo_path, filename)
    """
    pairs = []
    with open(split_file) as f:
        filenames = [line.strip() for line in f if line.strip()]

    for filename in filenames:
        label_path = os.path.join(data_root, 'labels', filename)
        pseudo_path = os.path.join(data_root, 'pseudolabels', filename)
        pairs.append((label_path, pseudo_path, filename))

    return pairs


def analyze_dataset(data_root: str, split_files: List[str] = None, max_samples: int = None):
    """
    Phân tích toàn bộ dataset, so sánh pseudo-label với ground truth.

    Args:
        data_root: Đường dẫn gốc dataset (chứa images/, labels/, pseudolabels/)
        split_files: Danh sách file split cần phân tích (train.txt, val.txt, test.txt)
        max_samples: Giới hạn số mẫu phân tích (None = tất cả)
    """
    if split_files is None:
        split_files = ['train.txt', 'val.txt', 'test.txt']

    # Accumulate confusion matrix
    total_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    total_pixels = 0
    total_correct = 0

    all_files = []
    for split_file in split_files:
        split_path = os.path.join(data_root, split_file)
        if os.path.exists(split_path):
            pairs = find_dataset_pairs(data_root, split_path)
            for label_path, pseudo_path, filename in pairs:
                all_files.append((label_path, pseudo_path, filename))

    if max_samples:
        all_files = all_files[:max_samples]

    print(f"\n{'='*70}")
    print(f"PHÂN TÍCH DATASET: Pseudo-label vs Ground Truth")
    print(f"{'='*70}")
    print(f"Data root: {data_root}")
    print(f"Số lượng file: {len(all_files)}")
    print(f"CLASS_NAMES: {CLASS_NAMES}")
    print(f"{'='*70}\n")

    # Process each file
    for label_path, pseudo_path, filename in tqdm(all_files, desc="Analyzing"):
        if not os.path.exists(label_path):
            print(f"Warning: GT not found: {label_path}")
            continue
        if not os.path.exists(pseudo_path):
            print(f"Warning: Pseudo not found: {pseudo_path}")
            continue

        # Load masks
        gt_mask = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        pseudo_mask = cv2.imread(pseudo_path, cv2.IMREAD_UNCHANGED)

        if gt_mask is None or pseudo_mask is None:
            continue

        # Ensure same size
        if gt_mask.shape != pseudo_mask.shape:
            pseudo_mask = cv2.resize(pseudo_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)

        # Handle label values
        # GT va Pseudo co gia tri 1-8 (1-indexed), can chuyen ve 0-6 (0-indexed, 7 classes)
        # 0 = background/void
        gt_mask = gt_mask.astype(np.int32)
        pseudo_mask = pseudo_mask.astype(np.int32)

        # Chuyen 1-8 -> 0-7, sau do chi lay 0-6 (7 classes)
        # Neu GT co 8 class (1-8) -> tru 1 de co 0-7
        if gt_mask.max() >= 1:
            gt_mask = gt_mask - 1  # 1-8 -> 0-7

        # Pseudo tuong tu
        if pseudo_mask.max() >= 1:
            pseudo_mask = pseudo_mask - 1  # 1-8 -> 0-7

        # Clip ve 0-6 (7 classes)
        gt_mask = np.clip(gt_mask, 0, NUM_CLASSES - 1)
        pseudo_mask = np.clip(pseudo_mask, 0, NUM_CLASSES - 1)

        # Compute CM for this image
        img_cm = compute_confusion_matrix(gt_mask, pseudo_mask, NUM_CLASSES)

        # Accumulate
        total_cm += img_cm
        total_pixels += np.sum((gt_mask >= 0) & (gt_mask < NUM_CLASSES))
        total_correct += np.sum((gt_mask == pseudo_mask) & (gt_mask >= 0) & (gt_mask < NUM_CLASSES))

    # Compute overall metrics
    metrics = compute_metrics_per_class(total_cm)

    # Print results
    print_results(CLASS_NAMES, metrics, total_cm, total_pixels, total_correct)

    return {
        'confusion_matrix': total_cm,
        'metrics': metrics,
        'total_pixels': total_pixels,
        'total_correct': total_correct,
        'class_names': CLASS_NAMES
    }


def print_results(class_names: List[str], metrics: Dict, cm: np.ndarray,
                  total_pixels: int, total_correct: int):
    """In kết quả phân tích đẹp mắt"""

    print("\n" + "="*70)
    print("KẾT QUẢ PHÂN TÍCH")
    print("="*70)

    # Overall accuracy
    overall_acc = total_correct / total_pixels * 100 if total_pixels > 0 else 0
    print(f"\nTổng số pixel: {total_pixels:,}")
    print(f"Pixel đúng: {total_correct:,}")
    print(f"Overall Accuracy: {overall_acc:.2f}%")

    # Per-class metrics table
    print(f"\n{'Class':<15} {'IoU':>10} {'Precision':>12} {'Recall':>10} {'F1':>10} {'Support':>12}")
    print("-"*70)

    iou = metrics['iou']
    precision = metrics['precision']
    recall = metrics['recall']
    f1 = metrics['f1']

    for cls_idx, cls_name in enumerate(class_names):
        support = np.sum(cm[cls_idx, :])  # Số pixel GT thuộc class này
        print(f"{cls_name:<15} "
              f"{iou[cls_idx]*100:>9.2f}% "
              f"{precision[cls_idx]*100:>11.2f}% "
              f"{recall[cls_idx]*100:>9.2f}% "
              f"{f1[cls_idx]*100:>9.2f}% "
              f"{support:>11,}")

    print("-"*70)
    print(f"{'mIoU':<15} "
          f"{np.mean(iou)*100:>9.2f}% "
          f"{np.mean(precision)*100:>11.2f}% "
          f"{np.mean(recall)*100:>9.2f}% "
          f"{np.mean(f1)*100:>9.2f}%")

    # Confusion matrix (normalized)
    print("\n" + "="*70)
    print("CONFUSION MATRIX (hàng = GT, cột = Pseudo-label)")
    print("="*70)

    # Normalize by row
    cm_norm = np.zeros_like(cm, dtype=float)
    for i in range(cm.shape[0]):
        row_sum = np.sum(cm[i, :])
        if row_sum > 0:
            cm_norm[i, :] = cm[i, :] / row_sum * 100

    # Print header
    header = f"{'GT\\Pred':>10}"
    for cls in class_names:
        header += f" {cls[:4]:>7}"
    print(header)
    print("-"*header.__len__())

    # Print matrix
    for i, cls_name in enumerate(class_names):
        row_str = f"{cls_name:>10}"
        for j in range(len(class_names)):
            row_str += f" {cm_norm[i, j]:>6.1f}"
        print(row_str)

    print("\n" + "="*70)


def visualize_differences(data_root: str, output_dir: str, num_samples: int = 10):
    """
    Visualize sự khác biệt giữa pseudo-label và ground truth.

    Args:
        data_root: Đường dẫn gốc dataset
        output_dir: Thư mục lưu kết quả visual
        num_samples: Số mẫu visualize
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    os.makedirs(output_dir, exist_ok=True)

    # Color map cho 7 classes (không có Bareland)
    colors = [
        '#8B4513',  # 0 - Rangeland (brown)
        '#808080',  # 1 - Developed (gray)
        '#000000',  # 2 - Road (black)
        '#228B22',  # 3 - Tree (green)
        '#0000FF',  # 4 - Water (blue)
        '#FFFF00',  # 5 - Agriculture (yellow)
        '#FF0000',  # 6 - Building (red)
    ]
    cmap = ListedColormap(colors)

    # Get some samples
    val_path = os.path.join(data_root, 'val.txt')
    if not os.path.exists(val_path):
        print(f"Warning: {val_path} not found")
        return

    pairs = find_dataset_pairs(data_root, val_path)
    samples = pairs[:num_samples]

    print(f"\nVisualizing {len(samples)} samples to {output_dir}")

    for idx, (label_path, pseudo_path, filename) in enumerate(tqdm(samples, desc="Visualizing")):
        if not os.path.exists(pseudo_path):
            continue

        # Load images
        img_path = os.path.join(data_root, 'images', filename)
        rgb = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if rgb is not None:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        pseudo_mask = cv2.imread(pseudo_path, cv2.IMREAD_UNCHANGED)

        if rgb is None or gt_mask is None or pseudo_mask is None:
            continue

        # Process masks
        if gt_mask.shape != pseudo_mask.shape:
            pseudo_mask = cv2.resize(pseudo_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)

        # Normalize GT
        gt_display = gt_mask.copy()
        if gt_display.max() > 7:
            gt_display = gt_display - 1
        gt_display = np.clip(gt_display, 0, 7)

        pseudo_display = np.clip(pseudo_mask, 0, 7)

        # Compute difference
        diff = (gt_display != pseudo_display).astype(np.float32)

        # Create figure
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))

        axes[0].imshow(rgb)
        axes[0].set_title('RGB Image')
        axes[0].axis('off')

        axes[1].imshow(gt_display, cmap=cmap, vmin=0, vmax=7)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

        axes[2].imshow(pseudo_display, cmap=cmap, vmin=0, vmax=7)
        axes[2].set_title('Pseudo-label')
        axes[2].axis('off')

        axes[3].imshow(diff, cmap='Reds')
        axes[3].set_title(f'Difference\n(Error pixels: {int(np.sum(diff)):,})')
        axes[3].axis('off')

        # Overlay difference on GT
        overlay = np.zeros((*gt_display.shape, 3), dtype=float)
        overlay[:, :] = gt_display[:, :, None] / 7.0  # Normalized GT
        overlay[diff > 0] = [1, 0, 0]  # Red for errors
        axes[4].imshow(overlay)
        axes[4].set_title('GT with Error Overlay')
        axes[4].axis('off')

        plt.tight_layout()

        filename = os.path.basename(label_path).replace('.tif', '.png')
        save_path = os.path.join(output_dir, f'{idx:03d}_{filename}')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"Saved {num_samples} visualizations to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phân tích pseudo-label vs ground truth")
    parser.add_argument("--data_root", type=str,
                        default="data/OEM_v2_aDanh",
                        help="Đường dẫn gốc dataset")
    parser.add_argument("--output_dir", type=str,
                        default="src/analysis/dataset_analysis_output",
                        help="Thư mục lưu kết quả")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Giới hạn số mẫu phân tích")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize sự khác biệt")
    parser.add_argument("--num_vis", type=int, default=10,
                        help="Số mẫu visualize")

    args = parser.parse_args()

    # Convert to absolute path - use normpath to resolve ../../ paths correctly
    data_root = args.data_root
    if not os.path.isabs(data_root):
        data_root = os.path.normpath(os.path.join(os.path.dirname(__file__), data_root))
    else:
        data_root = os.path.normpath(data_root)

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), output_dir))
    else:
        output_dir = os.path.normpath(output_dir)

    # Run analysis
    results = analyze_dataset(data_root, max_samples=args.max_samples)

    # Visualize if requested
    if args.visualize:
        visualize_differences(data_root, output_dir, num_samples=args.num_vis)
        print(f"\nVisualizations saved to: {output_dir}")
