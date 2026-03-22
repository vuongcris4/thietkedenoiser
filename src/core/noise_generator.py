"""
Utility constants and functions for segmentation evaluation.
"""
import numpy as np


CLASS_NAMES = [
    'Bareland', 'Rangeland', 'Developed', 'Road',
    'Tree', 'Water', 'Agriculture', 'Building'
]


NUM_CLASSES = len(CLASS_NAMES)


def compute_iou(pred, gt, num_classes=NUM_CLASSES):
    """Compute IoU and mIoU between prediction and ground truth."""
    ious = {}
    for c in range(num_classes):
        inter = ((pred == c) & (gt == c)).sum()
        union = ((pred == c) | (gt == c)).sum()
        if union == 0:
            continue
        ious[CLASS_NAMES[c]] = float(inter / union)
    return {'per_class_iou': ious, 'mIoU': float(np.mean(list(ious.values()))) if ious else 0.0}
