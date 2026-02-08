"""
Dataset classes for OpenEarthMap + DAE training.
OEM structure: data_root/{region}/images/{region}_{id}.tif
               data_root/{region}/labels/{region}_{id}.tif
Split files: data_root/train.txt, data_root/val.txt (file names only)
"""
import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional, Tuple, List

CLASS_NAMES = [
    'Bareland', 'Rangeland', 'Developed', 'Road',
    'Tree', 'Water', 'Agriculture', 'Building'
]
NUM_CLASSES = 8


def _get_region(filename):
    """Extract region from filename like 'aachen_1.tif' -> 'aachen'"""
    parts = filename.rsplit('_', 1)
    return parts[0] if len(parts) == 2 else filename.split('.')[0]


def find_oem_pairs(data_root: str, split_file: str) -> List[Tuple[str, str]]:
    """Find image-label pairs from split file."""
    pairs = []
    with open(split_file) as f:
        filenames = [l.strip() for l in f if l.strip()]
    
    for fn in filenames:
        region = _get_region(fn)
        img_path = os.path.join(data_root, region, 'images', fn)
        lbl_path = os.path.join(data_root, region, 'labels', fn)
        if os.path.exists(img_path) and os.path.exists(lbl_path):
            pairs.append((img_path, lbl_path))
    
    return pairs


class OpenEarthMapDataset(Dataset):
    """OpenEarthMap dataset for segmentation."""
    
    def __init__(self, data_root: str, split: str = 'train',
                 img_size: int = 512, augment: bool = True):
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        
        split_file = os.path.join(data_root, f'{split}.txt')
        if os.path.exists(split_file):
            self.pairs = find_oem_pairs(data_root, split_file)
        else:
            raise FileNotFoundError(f'Split file not found: {split_file}')
        
        print(f'OpenEarthMap {split}: {len(self.pairs)} samples')
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img_path, label_path = self.pairs[idx]
        
        # Read image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is not None and img.shape[2] > 3:
                img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Read label
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if label is None:
            # Try tifffile
            try:
                import tifffile
                label = tifffile.imread(label_path)
            except:
                label = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        if label.ndim == 3:
            label = label[:, :, 0]
        
        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        # Augmentation
        if self.augment:
            img, label = self._augment(img, label)
        
        # To tensor
        img = torch.from_numpy(img.transpose(2, 0, 1).copy()).float() / 255.0
        label = torch.from_numpy(label.copy()).long()
        label = torch.clamp(label, 0, NUM_CLASSES - 1)
        
        return img, label
    
    def _augment(self, img, label):
        if np.random.random() > 0.5:
            img = img[:, ::-1].copy()
            label = label[:, ::-1].copy()
        if np.random.random() > 0.5:
            img = img[::-1, :].copy()
            label = label[::-1, :].copy()
        k = np.random.randint(0, 4)
        img = np.rot90(img, k).copy()
        label = np.rot90(label, k).copy()
        return img, label


class DAEDataset(Dataset):
    """Dataset for training DAE: generates (noisy_input, clean_target) pairs.
    
    Input:  concat(RGB[3], noisy_onehot_label[C]) = 3+C channels
    Target: clean_label (class index)
    """
    
    def __init__(self, data_root: str, split: str = 'train',
                 img_size: int = 512, noise_type: str = 'mixed',
                 noise_rate_range: Tuple[float, float] = (0.05, 0.30),
                 augment: bool = True):
        self.base_dataset = OpenEarthMapDataset(data_root, split, img_size, augment=False)
        self.noise_type = noise_type
        self.noise_rate_range = noise_rate_range
        self.augment = augment and (split == 'train')
        self.img_size = img_size
        
        from noise_generator import NoiseGenerator
        self.noise_gen = NoiseGenerator(num_classes=NUM_CLASSES)
        self.noise_funcs = {
            'random_flip': self.noise_gen.random_flip_noise,
            'boundary': self.noise_gen.boundary_noise,
            'region_swap': self.noise_gen.region_swap_noise,
            'confusion_based': self.noise_gen.confusion_based_noise,
            'mixed': self.noise_gen.mixed_noise,
        }
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, clean_label = self.base_dataset[idx]
        # img: [3, H, W] float, clean_label: [H, W] long
        
        clean_np = clean_label.numpy().astype(np.int32)
        
        # Random noise rate
        noise_rate = np.random.uniform(*self.noise_rate_range)
        
        # Apply noise
        if self.noise_type == 'all_random':
            nt = np.random.choice(list(self.noise_funcs.keys()))
            noisy_np = self.noise_funcs[nt](clean_np, noise_rate=noise_rate)
        else:
            noisy_np = self.noise_funcs[self.noise_type](clean_np, noise_rate=noise_rate)
        
        # Convert noisy label to one-hot [C, H, W]
        noisy_onehot = np.zeros((NUM_CLASSES, self.img_size, self.img_size), dtype=np.float32)
        for c in range(NUM_CLASSES):
            noisy_onehot[c] = (noisy_np == c).astype(np.float32)
        noisy_onehot = torch.from_numpy(noisy_onehot)
        
        # Concat: [3+C, H, W]
        dae_input = torch.cat([img, noisy_onehot], dim=0)
        
        # Augmentation (same transform for input and target)
        if self.augment:
            if torch.rand(1) > 0.5:
                dae_input = torch.flip(dae_input, [2])
                clean_label = torch.flip(clean_label.unsqueeze(0), [2]).squeeze(0)
            if torch.rand(1) > 0.5:
                dae_input = torch.flip(dae_input, [1])
                clean_label = torch.flip(clean_label.unsqueeze(0), [1]).squeeze(0)
        
        return dae_input, clean_label
