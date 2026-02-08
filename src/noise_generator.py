"""
Noise Generator for Semantic Segmentation Labels (Optimized)
"""
import numpy as np
import cv2
from typing import Dict, Optional


CLASS_NAMES = [
    'Bareland', 'Rangeland', 'Developed', 'Road',
    'Tree', 'Water', 'Agriculture', 'Building'
]


class NoiseGenerator:
    def __init__(self, num_classes: int = 8, seed: int = 42):
        self.num_classes = num_classes
        self.rng = np.random.RandomState(seed)

    def random_flip_noise(self, label: np.ndarray, noise_rate: float = 0.1) -> np.ndarray:
        """Random Flip: doi ngau nhien class cua pixels. Dom dom, khong cau truc."""
        noisy = label.copy()
        H, W = label.shape
        mask = self.rng.random((H, W)) < noise_rate
        random_classes = self.rng.randint(0, self.num_classes - 1, size=(H, W))
        # Shift to avoid same class
        random_classes[random_classes >= label] += 1
        random_classes = np.clip(random_classes, 0, self.num_classes - 1)
        noisy[mask] = random_classes[mask]
        return noisy

    def boundary_noise(self, label: np.ndarray, noise_rate: float = 0.1) -> np.ndarray:
        """Boundary Noise: nhieu tap trung o ranh gioi giua cac class."""
        noisy = label.copy()
        kernel_size = max(3, int(10 * noise_rate / 0.3))
        kernel_size = min(kernel_size, 15)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        for c in range(self.num_classes):
            mask_c = (label == c).astype(np.uint8)
            if mask_c.sum() < 100:
                continue
            if self.rng.random() > 0.5:
                modified = cv2.dilate(mask_c, kernel, iterations=1)
                new_pixels = (modified > mask_c) & (noisy != c)
                noisy[new_pixels.astype(bool)] = c
            else:
                modified = cv2.erode(mask_c, kernel, iterations=1)
                lost_pixels = (mask_c > modified)
                if lost_pixels.any():
                    # Assign to most common neighbor
                    blurred = cv2.blur(label.astype(np.float32), (5, 5))
                    neighbor_class = np.round(blurred).astype(np.int32)
                    neighbor_class = np.clip(neighbor_class, 0, self.num_classes - 1)
                    noisy[lost_pixels.astype(bool)] = neighbor_class[lost_pixels.astype(bool)]
        return noisy

    def region_swap_noise(self, label: np.ndarray, noise_rate: float = 0.1) -> np.ndarray:
        """Region Swap: hoan class cua ca vung lon."""
        noisy = label.copy()
        H, W = label.shape
        target_pixels = int(H * W * noise_rate)
        changed = 0

        # Tao random rectangular regions
        max_attempts = 50
        for _ in range(max_attempts):
            if changed >= target_pixels:
                break
            rh = self.rng.randint(20, min(100, H // 3))
            rw = self.rng.randint(20, min(100, W // 3))
            ry = self.rng.randint(0, H - rh)
            rx = self.rng.randint(0, W - rw)

            region = noisy[ry:ry+rh, rx:rx+rw]
            dominant_class = np.bincount(region.flatten(), minlength=self.num_classes).argmax()
            # Swap to confused class
            new_class = self._get_confused_class(dominant_class)
            mask = (region == dominant_class)
            region[mask] = new_class
            noisy[ry:ry+rh, rx:rx+rw] = region
            changed += mask.sum()
        return noisy

    def confusion_based_noise(self, label: np.ndarray, noise_rate: float = 0.1) -> np.ndarray:
        """Confusion-based: nhieu theo xac suat nham thuc te giua cac class."""
        noisy = label.copy()
        H, W = label.shape
        cm = self._default_confusion_matrix()
        mask = self.rng.random((H, W)) < noise_rate

        for c in range(self.num_classes):
            class_noisy = mask & (label == c)
            if not class_noisy.any():
                continue
            n = class_noisy.sum()
            probs = cm[c].copy()
            probs[c] = 0
            probs /= probs.sum()
            new_classes = self.rng.choice(self.num_classes, size=n, p=probs)
            noisy[class_noisy] = new_classes
        return noisy

    def mixed_noise(self, label: np.ndarray, noise_rate: float = 0.1) -> np.ndarray:
        """Mixed: ket hop tat ca cac loai nhieu."""
        rate_each = noise_rate / 4
        noisy = self.boundary_noise(label, noise_rate=rate_each)
        noisy = self.confusion_based_noise(noisy, noise_rate=rate_each)
        noisy = self.region_swap_noise(noisy, noise_rate=rate_each)
        noisy = self.random_flip_noise(noisy, noise_rate=rate_each)
        return noisy

    def _get_confused_class(self, class_id: int) -> int:
        pairs = {0: 3, 3: 0, 1: 6, 6: 1, 4: 1, 7: 3, 2: 0, 5: 4}
        if class_id in pairs and self.rng.random() < 0.6:
            return pairs[class_id]
        return self.rng.choice([c for c in range(self.num_classes) if c != class_id])

    def _default_confusion_matrix(self) -> np.ndarray:
        cm = np.ones((self.num_classes, self.num_classes)) * 0.02
        cm[0, 3] = 0.35; cm[3, 0] = 0.35
        cm[1, 6] = 0.25; cm[6, 1] = 0.25
        cm[0, 1] = 0.15; cm[1, 0] = 0.15
        cm[3, 4] = 0.12; cm[4, 3] = 0.12
        cm[5, 1] = 0.08; cm[1, 5] = 0.08
        cm[7, 3] = 0.10; cm[3, 7] = 0.10
        cm[4, 7] = 0.08; cm[7, 4] = 0.08
        np.fill_diagonal(cm, 0)
        cm = cm / cm.sum(axis=1, keepdims=True)
        return cm

    def compute_noise_stats(self, clean: np.ndarray, noisy: np.ndarray) -> Dict:
        H, W = clean.shape
        changed = (clean != noisy)
        num_changed = changed.sum()

        # Boundary detection (fast)
        gx = np.abs(np.diff(clean.astype(np.int16), axis=1))
        gy = np.abs(np.diff(clean.astype(np.int16), axis=0))
        boundary = np.zeros_like(clean, dtype=bool)
        boundary[:, :-1] |= (gx > 0)
        boundary[:-1, :] |= (gy > 0)
        # Dilate boundary a bit
        boundary = cv2.dilate(boundary.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

        boundary_changed = (changed & boundary).sum()

        per_class = {}
        for c in range(self.num_classes):
            cmask = (clean == c)
            ct = cmask.sum()
            if ct == 0: continue
            cc = (changed & cmask).sum()
            per_class[CLASS_NAMES[c]] = {
                'total': int(ct), 'changed': int(cc),
                'rate': float(cc / ct)
            }

        return {
            'total_pixels': H * W,
            'changed_pixels': int(num_changed),
            'actual_noise_rate': float(num_changed / (H * W)),
            'boundary_changed': int(boundary_changed),
            'interior_changed': int(num_changed - boundary_changed),
            'boundary_ratio': float(boundary_changed / max(num_changed, 1)),
            'per_class': per_class,
        }


def compute_iou(pred, gt, num_classes=8):
    ious = {}
    for c in range(num_classes):
        inter = ((pred == c) & (gt == c)).sum()
        union = ((pred == c) | (gt == c)).sum()
        if union == 0: continue
        ious[CLASS_NAMES[c]] = float(inter / union)
    return {'per_class_iou': ious, 'mIoU': float(np.mean(list(ious.values()))) if ious else 0.0}
