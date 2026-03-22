"""
Dataset classes cho OpenEarthMap + DAE training.

Cấu trúc thư mục OEM:
    data_root/{region}/images/{region}_{id}.tif   ← ảnh vệ tinh RGB
    data_root/{region}/labels/{region}_{id}.tif   ← label segmentation (0-7)
    data_root/train.txt, data_root/val.txt        ← danh sách tên file cho mỗi split

File này cung cấp 2 Dataset:
    1. OpenEarthMapDataset: đọc ảnh + label gốc → (img[3,H,W], label[H,W])
    2. RealNoiseDAEDataset: đọc ảnh + pseudo-label thật từ CISC-R → (rgb, pseudo_onehot, clean_label)
"""
import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional, Tuple, List

# 8 lớp phân loại đất/vật thể trong OpenEarthMap
CLASS_NAMES = [
    'Bareland', 'Rangeland', 'Developed', 'Road',
    'Tree', 'Water', 'Agriculture', 'Building'
]
NUM_CLASSES = 8  # Tổng số lớp, dùng cho one-hot encoding và clamp label

# suy ra thư mục từ tên file
# Trích tên region bằng cách tách dấu _ cuối cùng:
def _get_region(filename):
    """
    Trích xuất tên vùng (region) từ tên file.
    
    Input:  filename (str) - tên file, vd: 'aachen_1.tif'
    Output: str - tên region, vd: 'aachen'
    
    Mục đích: Xác định thư mục con chứa ảnh, vì cấu trúc là data_root/{region}/images/
    Cách hoạt động: Tách chuỗi bằng dấu '_' cuối cùng, lấy phần trước.
    """
    parts = filename.rsplit('_', 1)  # Tách từ phải: 'aachen_1.tif' → ['aachen', '1.tif']
    return parts[0] if len(parts) == 2 else filename.split('.')[0]

# hàm find_oem_pairs
# trả về list input/label từ dataset train/val/test 
def find_oem_pairs(data_root: str, split_file: str) -> List[Tuple[str, str]]:
    """
    Tìm tất cả cặp (ảnh, label) hợp lệ từ file split.
    
    Input:
        data_root  (str) - đường dẫn gốc chứa data, vd: 'data/OpenEarthMap_wo_xBD'
        split_file (str) - đường dẫn file .txt chứa danh sách tên file, vd: 'train.txt'
    Output:
        List[Tuple[str, str]] - danh sách cặp (image_path, label_path) đã verify tồn tại
    
    Mục đích: Ghép đường dẫn đầy đủ cho ảnh và label, chỉ giữ các cặp thực sự tồn tại.
    Ví dụ: 'aachen_1.tif' → ('data_root/aachen/images/aachen_1.tif',
                              'data_root/aachen/labels/aachen_1.tif')
    """
    pairs = []
    # Đọc danh sách tên file từ split file (mỗi dòng 1 file)
    with open(split_file) as f:
        filenames = [l.strip() for l in f if l.strip()]
    
    for fn in filenames:
        region = _get_region(fn)  # Trích region từ tên file
        img_path = os.path.join(data_root, region, 'images', fn)  # Đường dẫn ảnh RGB
        lbl_path = os.path.join(data_root, region, 'labels', fn)  # Đường dẫn label
        # Chỉ thêm vào nếu cả ảnh và label đều tồn tại
        if os.path.exists(img_path) and os.path.exists(lbl_path):
            pairs.append((img_path, lbl_path))
    
    return pairs


class OpenEarthMapDataset(Dataset):
    """
    Dataset cơ sở: đọc ảnh vệ tinh RGB + label segmentation từ OpenEarthMap.
    
    Output mỗi sample:
        img   : tensor [3, H, W] float32, giá trị [0, 1] (ảnh RGB đã normalize)
        label : tensor [H, W] long, giá trị [0, 7] (class index)
    
    Được dùng trực tiếp cho segmentation, hoặc làm base cho DAEDataset.
    """
    
    def __init__(self, data_root: str, split: str = 'train',
                 img_size: int = 512, augment: bool = True):
        """
        Khởi tạo dataset.
        
        Input:
            data_root (str)  - thư mục gốc chứa data
            split     (str)  - 'train' hoặc 'val'
            img_size  (int)  - kích thước resize ảnh (mặc định 512×512)
            augment   (bool) - bật augmentation (chỉ áp dụng khi split='train')
        """
        self.img_size = img_size
        self.augment = augment and (split == 'train')  # Chỉ augment khi training
        
        # Tìm file split (train.txt hoặc val.txt)
        split_file = os.path.join(data_root, f'{split}.txt')
        if os.path.exists(split_file):
            self.pairs = find_oem_pairs(data_root, split_file)  # Danh sách cặp (img, label)
        else:
            raise FileNotFoundError(f'Split file not found: {split_file}')
        
        print(f'OpenEarthMap {split}: {len(self.pairs)} samples')
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        Lấy 1 sample theo index.
        
        Input:  idx (int) - chỉ số sample
        Output: (img, label)
            img   : tensor [3, H, W]        float32 [0,1] - ảnh RGB đã normalize
            label : tensor [H, W]           long [0,7]    - label class index
        """
        img_path, label_path = self.pairs[idx]
        
        # === Đọc ảnh RGB ===
        # → numpy array shape [H, W, 3], dtype uint8, giá trị [0, 255]
        # → 3 kênh BGR (không phải RGB!)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Đọc BGR
        
        if img is None:
            # Fallback: thử đọc nguyên bản (cho file TIFF đặc biệt)
            # trả về numpy array nhiều chiều
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is not None and img.shape[2] > 3:
                img = img[:, :, :3]  # Cắt bỏ kênh alpha/extra
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển BGR → RGB
        
        # === Đọc label segmentation ===
        # → numpy array shape [H, W], dtype uint8, giá trị [0, 7]
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)  # Đọc grayscale
        if label is None:
            # Fallback: thử dùng tifffile cho file .tif đặc biệt
            try:
                import tifffile
                label = tifffile.imread(label_path)
            except:
                # Nếu vẫn thất bại → tạo label rỗng (toàn 0)
                label = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        if label.ndim == 3:
            label = label[:, :, 0]  # Nếu label có 3 kênh → lấy kênh đầu
        
        # === Resize về kích thước chuẩn ===
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)    # Bilinear cho ảnh
        label = cv2.resize(label, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)  # Nearest cho label (giữ nguyên class)
        
        # === Augmentation (nếu bật) ===
        if self.augment:
            img, label = self._augment(img, label)
        
        # === Chuyển numpy → tensor ===
        img = torch.from_numpy(img.transpose(2, 0, 1).copy()).float() / 255.0  # [H,W,3] → [3,H,W], normalize [0,1]
        label = torch.from_numpy(label.copy()).long()  # Label → long tensor
        label = torch.clamp(label, 0, NUM_CLASSES - 1)  # Clamp về [0,7] phòng giá trị ngoài phạm vi
        
        return img, label
    
    def _augment(self, img, label):
        """
        Data augmentation cho ảnh và label (áp dụng đồng bộ).
        
        Input:  img [H,W,3] numpy, label [H,W] numpy
        Output: (img, label) đã augment
        
        3 phép biến đổi (tất cả random):
            1. Flip ngang (50% xác suất)
            2. Flip dọc   (50% xác suất)
            3. Xoay 0°/90°/180°/270° (đều nhau)
        """
        if np.random.random() > 0.5:        # Flip ngang (trái-phải)
            img = img[:, ::-1].copy()
            label = label[:, ::-1].copy()
        if np.random.random() > 0.5:        # Flip dọc (trên-dưới)
            img = img[::-1, :].copy()
            label = label[::-1, :].copy()
        k = np.random.randint(0, 4)          # Xoay k×90°
        img = np.rot90(img, k).copy()
        label = np.rot90(label, k).copy()
        return img, label

# ============================================================
# Real Noise Dataset: dùng pseudo-label thật từ model CISC-R
# ============================================================
#
# Cấu trúc folder pseudo_root (vd: OEM_v2_aDanh/):
#   pseudo_root/
#   ├── images/      ← ảnh RGB (symlinks hoặc copy từ OpenEarthMap)
#   ├── labels/      ← pseudo-labels từ CISC-R
#   ├── train.txt    ← danh sách file train
#   ├── val.txt      ← danh sách file val
#   └── test.txt     ← danh sách file test
#
# Để tạo cấu trúc này, chạy: python reorganize_pseudo_dataset.py
# ============================================================


def find_pseudo_pairs(pseudo_root: str,
                      split_file: str) -> List[Tuple[str, str]]:
    """
    Tìm cặp (ảnh, pseudo-label) từ folder có cấu trúc flat.

    Input:
        pseudo_root (str) - thư mục gốc chứa images/ + labels/
        split_file  (str) - file .txt chứa danh sách tên file

    Output:
        List[Tuple[str, str]] - danh sách (img_path, label_path)
    """
    pairs = []
    with open(split_file) as f:
        filenames = [l.strip() for l in f if l.strip()]

    for fn in filenames:
        img_path = os.path.join(pseudo_root, 'images', fn)
        lbl_path = os.path.join(pseudo_root, 'labels', fn)
        if os.path.exists(img_path) and os.path.exists(lbl_path):
            pairs.append((img_path, lbl_path))

    return pairs


class RealNoiseDAEDataset(Dataset):
    """
    Dataset cho DAE training với pseudo-label thật từ CISC-R.

    Cấu trúc folder đơn giản (flat):
        pseudo_root/images/{fn}.tif  → ảnh RGB
        pseudo_root/labels/{fn}.tif  → pseudo-label (noisy)

    Dùng clean label từ OpenEarthMap (data_root) nếu có,
    hoặc chỉ trả về (rgb, pseudo_onehot) nếu không có clean label.

    Output:
        (rgb[3,H,W], pseudo_onehot[8,H,W], clean_label[H,W])
    """

    def __init__(self, pseudo_root: str,
                 data_root: str = None,
                 split: str = 'train',
                 img_size: int = 512,
                 augment: bool = True):
        """
        Input:
            pseudo_root (str) - thư mục dataset (chứa images/, labels/, split files)
            data_root   (str) - thư mục OpenEarthMap (lấy clean label), None = không dùng
            split       (str) - 'train', 'val', 'test'
            img_size    (int) - kích thước resize
            augment     (bool) - bật augmentation
        """
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        self.data_root = data_root

        split_file = os.path.join(pseudo_root, f'{split}.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f'Split file not found: {split_file}\n'
                f'Chạy "python reorganize_pseudo_dataset.py" để tạo.'
            )

        self.pairs = find_pseudo_pairs(pseudo_root, split_file)
        print(f'RealNoiseDAEDataset {split}: {len(self.pairs)} samples')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Output: (rgb, pseudo_onehot, clean_label)
            rgb           : [3, H, W] float32 [0,1]
            pseudo_onehot : [8, H, W] float32
            clean_label   : [H, W]    long [0,7]
        """
        img_path, pseudo_path = self.pairs[idx]

        # === Đọc ảnh RGB ===
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is not None and img.shape[2] > 3:
                img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # === Đọc pseudo-label (noise thật từ CISC-R) ===
        pseudo = cv2.imread(pseudo_path, cv2.IMREAD_UNCHANGED)
        if pseudo is None:
            try:
                import tifffile
                pseudo = tifffile.imread(pseudo_path)
            except:
                pseudo = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        if pseudo.ndim == 3:
            pseudo = pseudo[:, :, 0]

        # === Đọc clean label (từ OpenEarthMap nếu có) ===
        clean = None
        if self.data_root:
            fn = os.path.basename(pseudo_path)
            region = _get_region(fn)
            clean_path = os.path.join(self.data_root, region, 'labels', fn)
            if os.path.exists(clean_path):
                clean = cv2.imread(clean_path, cv2.IMREAD_UNCHANGED)
                if clean is None:
                    try:
                        import tifffile
                        clean = tifffile.imread(clean_path)
                    except:
                        clean = None
                if clean is not None and clean.ndim == 3:
                    clean = clean[:, :, 0]

        if clean is None:
            clean = pseudo.copy()  # fallback: pseudo = clean (no denoising)

        # === Resize ===
        img = cv2.resize(img, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_LINEAR)
        clean = cv2.resize(clean, (self.img_size, self.img_size),
                           interpolation=cv2.INTER_NEAREST)
        pseudo = cv2.resize(pseudo, (self.img_size, self.img_size),
                            interpolation=cv2.INTER_NEAREST)

        # === Chuyển tensor ===
        img_t = torch.from_numpy(
            img.transpose(2, 0, 1).copy()
        ).float() / 255.0  # [3, H, W]

        clean_label = torch.from_numpy(clean.copy()).long()
        clean_label = torch.clamp(clean_label, 0, NUM_CLASSES - 1)

        pseudo_np = np.clip(pseudo.astype(np.int32), 0, NUM_CLASSES - 1)

        # === One-hot encode pseudo-label → [8, H, W] ===
        pseudo_onehot = np.zeros(
            (NUM_CLASSES, self.img_size, self.img_size), dtype=np.float32
        )
        for c in range(NUM_CLASSES):
            pseudo_onehot[c] = (pseudo_np == c).astype(np.float32)
        pseudo_onehot = torch.from_numpy(pseudo_onehot)

        # === Augmentation (đồng bộ rgb + pseudo + target) ===
        if self.augment:
            if torch.rand(1) > 0.5:
                img_t = torch.flip(img_t, [2])
                pseudo_onehot = torch.flip(pseudo_onehot, [2])
                clean_label = torch.flip(clean_label.unsqueeze(0), [2]).squeeze(0)
            if torch.rand(1) > 0.5:
                img_t = torch.flip(img_t, [1])
                pseudo_onehot = torch.flip(pseudo_onehot, [1])
                clean_label = torch.flip(clean_label.unsqueeze(0), [1]).squeeze(0)

        return img_t, pseudo_onehot, clean_label
