import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional, Tuple, List

# =============================================================================
# CONSTANTS
# =============================================================================

# 8 lớp phân loại đất/vật thể trong OpenEarthMap
CLASS_NAMES = [
    'Bareland', 'Rangeland', 'Developed', 'Road',
    'Tree', 'Water', 'Agriculture', 'Building'
]
NUM_CLASSES = 8  # Tổng số lớp, dùng cho one-hot encoding và clamp label

# Label values trong dataset gốc: 0-8 (0=background, 1-8=classes)
# Không cần offset, giữ nguyên values


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_region(filename: str) -> str:
    """
    Trích xuất tên vùng (region) từ tên file.

    Input:
        filename (str) - Tên file ảnh, ví dụ: 'aachen_1.tif'

    Output:
        str - Tên region, ví dụ: 'aachen'

    Ví dụ:
        >>> _get_region('aachen_1.tif')
        'aachen'
        >>> _get_region('new_york_42.tif')
        'new_york'

    Mục đích:
        Xác định thư mục con chứa ảnh trong cấu trúc: data_root/{region}/images/

    Cách hoạt động:
        Tách chuỗi bằng dấu '_' cuối cùng, lấy phần trước.
        'aachen_1.tif' → ['aachen', '1.tif'] → 'aachen'
    """
    parts = filename.rsplit('_', 1)  # Tách từ phải: 'aachen_1.tif' → ['aachen', '1.tif']
    return parts[0] if len(parts) == 2 else filename.split('.')[0]

# =============================================================================
# find_oem_pairs - Helper Function
# =============================================================================

def find_oem_pairs(data_root: str, split_file: str) -> List[Tuple[str, str]]:
    """
    Tìm tất cả cặp (ảnh, label) hợp lệ từ file split.

    Input:
        data_root  (str) - Đường dẫn gốc chứa data
                         Ví dụ: 'data/OpenEarthMap_wo_xBD'
        split_file (str) - Đường dẫn file .txt chứa danh sách tên file
                         Ví dụ: 'data/OpenEarthMap_wo_xBD/train.txt'
                         Nội dung file: mỗi dòng 1 tên file
                         ```
                         aachen_1.tif
                         aachen_2.tif
                         new_york_5.tif
                         ```

    Output:
        List[Tuple[str, str]] - Danh sách cặp (image_path, label_path) đã verify tồn tại
        Ví dụ: [
            ('data/OpenEarthMap_wo_xBD/aachen/images/aachen_1.tif',
             'data/OpenEarthMap_wo_xBD/aachen/labels/aachen_1.tif'),
            ('data/OpenEarthMap_wo_xBD/aachen/images/aachen_2.tif',
             'data/OpenEarthMap_wo_xBD/aachen/labels/aachen_2.tif')
        ]

    Mục đích:
        Ghép đường dẫn đầy đủ cho ảnh và label, chỉ giữ các cặp thực sự tồn tại.

    Cấu trúc dữ liệu OpenEarthMap:
        data_root/
        ├── aachen/
        │   ├── images/
        │   │   └── aachen_1.tif      # Ảnh vệ tinh RGB
        │   └── labels/
        │       └── aachen_1.tif      # Label segmentation (grayscale, giá trị 0-7)
        ├── new_york/
        │   ├── images/
        │   └── labels/
        └── train.txt                 # Danh sách file thuộc split train
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


# =============================================================================
# OpenEarthMapDataset - Base Dataset for Satellite Image Segmentation
# =============================================================================

class OpenEarthMapDataset(Dataset):
    """
    Dataset cơ sở: đọc ảnh vệ tinh RGB + label segmentation từ OpenEarthMap.

    Data flow:
        Input (file .tif) → Read (cv2) → Resize (512x512) → Augment (optional)
                         → ToTensor → Normalize → Output

    Output mỗi sample:
        img   : tensor [3, H, W] float32, giá trị [0, 1] (ảnh RGB đã normalize)
        label : tensor [H, W] long, giá trị [0, 7] (class index)

    Ví dụ:
        >>> dataset = OpenEarthMapDataset('data/OpenEarthMap_wo_xBD', split='train')
        >>> img, label = dataset[0]
        >>> img.shape      # torch.Size([3, 512, 512])
        >>> label.shape    # torch.Size([512, 512])
        >>> label.unique() # tensor([0, 1, 2, 3, 4, 5, 6, 7]) - các class có trong sample

    Được dùng trực tiếp cho segmentation, hoặc làm base cho DAEDataset.
    """

    def __init__(self, data_root: str, split: str = 'train',
                 img_size: int = 512, augment: bool = True):
        """
        Khởi tạo dataset.

        Input:
            data_root (str)  - Thư mục gốc chứa data
                              Ví dụ: 'data/OpenEarthMap_wo_xBD'
            split     (str)  - 'train' hoặc 'val'
                              File split tương ứng: {split}.txt trong data_root
            img_size  (int)  - Kích thước resize ảnh (mặc định 512×512)
            augment   (bool) - Bật augmentation (chỉ áp dụng khi split='train')

        Augmentation bao gồm:
            - Flip ngang (50% probability)
            - Flip dọc (50% probability)
            - Rotation 0°/90°/180°/270° (random)
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

        Input:
            idx (int) - Chỉ số sample trong dataset

        Output:
            Tuple chứa:
            - img   : tensor [3, 512, 512] float32 [0, 1]
                      3 kênh RGB, đã normalize về [0, 1]
            - label : tensor [512, 512] long [0, 7]
                      Mỗi pixel chứa class ID (0-7)

        Quy trình xử lý:
            1. Đọc ảnh RGB (BGR → RGB)
               Input:  File .tif trên disk
               Output: numpy [H, W, 3], uint8 [0, 255], BGR

            2. Đọc label segmentation
               Input:  File .tif chứa class IDs
               Output: numpy [H, W], uint8 [0, 7]

            3. Resize về 512x512
               img:   Bilinear interpolation
               label: Nearest-neighbor (giữ nguyên class ID)

            4. Augmentation (nếu train)
               - Flip ngang/dọc
               - Rotation 0°/90°/180°/270°

            5. Chuyển tensor + normalize
               img:   [H, W, 3] → [3, H, W], float32 [0, 1]
               label: [H, W] → long tensor, clamp [0, 7]

        Ví dụ chi tiết:
            Sample 0 từ 'aachen_1.tif':
            - img_path:   data/OpenEarthMap/aachen/images/aachen_1.tif
            - label_path: data/OpenEarthMap/aachen/labels/aachen_1.tif

            Sau khi đọc:
            - img:   numpy (512, 512, 3), uint8, giá trị [0, 255]
            - label: numpy (512, 512),    uint8, giá trị [0, 7]

            Sau khi chuyển tensor:
            - img:   torch (3, 512, 512), float32, giá trị [0, 1]
            - label: torch (512, 512),    int64,   giá trị [0, 7]

            Class mapping:
                0: Bareland     2: Developed    4: Tree         6: Agriculture
                1: Rangeland    3: Road         5: Water        7: Building
        """
        img_path, label_path = self.pairs[idx]

        # === Đọc ảnh RGB ===
        # Input:  File .tif trên disk
        # Output: numpy array shape [H, W, 3], dtype uint8, giá trị [0, 255]
        # Lưu ý: cv2.imread() đọc ảnh ở định dạng BGR (không phải RGB!)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Đọc BGR

        if img is None:
            # Fallback: thử đọc nguyên bản (cho file TIFF đặc biệt)
            # trả về numpy array nhiều chiều
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is not None and img.shape[2] > 3:
                img = img[:, :, :3]  # Cắt bỏ kênh alpha/extra
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển BGR → RGB

        # === Đọc label segmentation ===
        # Input:  File .tif chứa class IDs (giá trị gốc 0-8 từ OpenEarthMap)
        # Output: numpy array shape [H, W], dtype uint8
        #
        # Label values giữ nguyên từ dataset:
        #   0: background/void → sẽ ignore (loss=255)
        #   1-8: 8 classes (không offset, giữ nguyên)
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
        # img:   Bilinear interpolation - mượt mà cho ảnh RGB
        # label: Nearest-neighbor - giữ nguyên giá trị class ID (không interpolate)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # === Augmentation (nếu bật) ===
        if self.augment:
            img, label = self._augment(img, label)

        # === Chuyển numpy → tensor ===
        # img:   [H, W, 3] → [3, H, W] + normalize /255 → [0, 1]
        # label: numpy → long tensor, giữ nguyên values 0-8
        img = torch.from_numpy(img.transpose(2, 0, 1).copy()).float() / 255.0
        label = torch.from_numpy(label.copy()).long()
        # Không cần clamp vì đã remap và xử lý ở trên

        return img, label
    
    def _augment(self, img: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Data augmentation cho ảnh và label (áp dụng đồng bộ).

        Input:
            img   : np.ndarray [H, W, 3], uint8/float32 - Ảnh RGB
            label : np.ndarray [H, W], uint8/int32 - Label segmentation

        Output:
            Tuple[np.ndarray, np.ndarray] - (img, label) đã augment

        3 phép biến đổi (tất cả random, áp dụng đồng bộ cho img và label):

        1. Flip ngang (50% xác suất)
           ┌─────────┐              ┌─────────┐
           │ 1 2 3   │              │ 3 2 1   │
           │ 4 5 6   │   ───→       │ 6 5 4   │
           │ 7 8 9   │              │ 9 8 7   │
           └─────────┘              └─────────┘

        2. Flip dọc (50% xác suất)
           ┌─────────┐              ┌─────────┐
           │ 1 2 3   │              │ 7 8 9   │
           │ 4 5 6   │   ───→       │ 4 5 6   │
           │ 7 8 9   │              │ 1 2 3   │
           └─────────┘              └─────────┘

        3. Xoay k×90° (k ∈ {0,1,2,3}, 25% mỗi giá trị)
           k=0: giữ nguyên
           k=1: xoay 90° ngược chiều kim đồng hồ
           k=2: xoay 180°
           k=3: xoay 270°

        Ví dụ:
            img.shape = (512, 512, 3), label.shape = (512, 512)

            Sau flip ngang:
            img.shape = (512, 512, 3), label.shape = (512, 512)

            Sau xoay 90°:
            img.shape = (512, 512, 3), label.shape = (512, 512)

        Lưu ý:
            - Các phép biến đổi được áp dụng đồng bộ cho cả img và label
            - Dùng .copy() để tránh inplace modification
            - Pixel values không thay đổi, chỉ thay đổi vị trí
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

# =============================================================================
# Real Noise Dataset - Pseudo-labels from CISC-R Model
# =============================================================================
#
# Cấu trúc folder pseudo_root (ví dụ: OEM_v2_aDanh/):
#
#   pseudo_root/
#   ├── images/          ← Ảnh RGB (symlinks từ OpenEarthMap)
#   │   └── aachen_1.tif
#   ├── pseudolabels/    ← Pseudo-labels từ model CISC-R (noisy)
#   │   └── aachen_1.tif   # Giá trị: 0-7 (dự đoán từ model)
#   ├── labels/          ← Ground truth từ OpenEarthMap (clean)
#   │   └── aachen_1.tif   # Giá trị: 0-7 (label thật)
#   ├── train.txt        # Danh sách file train
#   ├── val.txt          # Danh sách file val
#   └── test.txt         # Danh sách file test
#
# Tự động chia train/val/test theo tỷ lệ 80/10/10 nếu không có split files.
# =============================================================================


# =============================================================================
# get_split_files - Split File Manager
# =============================================================================

def get_split_files(pseudo_root: str, split: str = 'train') -> str:
    """
    Lấy path đến file split (train.txt, val.txt, test.txt).
    Nếu không tồn tại, tự động tạo từ danh sách files trong pseudolabels/.

    Input:
        pseudo_root (str) - Thư mục gốc chứa dataset
                            Ví dụ: 'data/OEM_v2_aDanh'
        split       (str) - 'train', 'val', hoặc 'test'

    Output:
        str - Đường dẫn đầy đủ đến file split

    Ví dụ:
        >>> get_split_files('data/OEM_v2_aDanh', 'train')
        'data/OEM_v2_aDanh/train.txt'

        Nếu file chưa tồn tại, tự động tạo với tỷ lệ:
        - train.txt: 80% files
        - val.txt:   10% files
        - test.txt:  10% files

    Quy trình:
        1. Kiểm tra file split đã tồn tại → trả về ngay
        2. Đọc danh sách file .tif từ pseudolabels/
        3. Shuffle với seed cố định (42) cho reproducibility
        4. Chia theo tỷ lệ 80/10/10
        5. Ghi vào 3 file train.txt, val.txt, test.txt

    Output files (ví dụ với 100 files):
        train.txt: 80 files
        val.txt:   10 files
        test.txt:  10 files
    """
    split_file = os.path.join(pseudo_root, f'{split}.txt')

    # Nếu đã có file split → dùng luôn
    if os.path.exists(split_file):
        return split_file

    # Tự động tạo split files
    pseudo_dir = os.path.join(pseudo_root, 'pseudolabels')
    if not os.path.isdir(pseudo_dir):
        raise FileNotFoundError(f'Pseudo-labels folder not found: {pseudo_dir}')

    # Lấy danh sách file .tif
    files = sorted([f for f in os.listdir(pseudo_dir) if f.endswith('.tif')])

    if not files:
        raise FileNotFoundError(f'No .tif files found in {pseudo_dir}')

    # Auto-create split files (80/10/10)
    import random
    random.seed(42)
    random.shuffle(files)

    n = len(files)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    splits = {
        'train.txt': files[:n_train],
        'val.txt': files[n_train:n_train + n_val],
        'test.txt': files[n_train + n_val:],
    }

    for fname, file_list in splits.items():
        path = os.path.join(pseudo_root, fname)
        with open(path, 'w') as f:
            f.write('\n'.join(file_list) + '\n')
        print(f'Created {fname}: {len(file_list)} files')

    return os.path.join(pseudo_root, f'{split}.txt')


# =============================================================================
# find_pseudo_pairs - Triple Pair Finder
# =============================================================================

def find_pseudo_pairs(pseudo_root: str,
                      split_file: str) -> List[Tuple[str, str, str]]:
    """
    Tìm bộ ba (ảnh, pseudo-label, ground-truth) từ folder.

    Input:
        pseudo_root (str) - Thư mục gốc chứa images/, pseudolabels/, labels/
                            Ví dụ: 'data/OEM_v2_aDanh'
        split_file  (str) - File .txt chứa danh sách tên file
                            Ví dụ: 'data/OEM_v2_aDanh/train.txt'

    Output:
        List[Tuple[str, str, str]] - Danh sách (img_path, pseudo_path, gt_path)
        Ví dụ: [
            (
                'data/OEM_v2_aDanh/images/aachen_1.tif',
                'data/OEM_v2_aDanh/pseudolabels/aachen_1.tif',
                'data/OEM_v2_aDanh/labels/aachen_1.tif'
            ),
            ...
        ]

    Cấu trúc dữ liệu:
        pseudo_root/
        ├── images/
        │   └── aachen_1.tif      # Ảnh RGB gốc từ OpenEarthMap
        ├── pseudolabels/
        │   └── aachen_1.tif      # Pseudo-label từ model CISC-R (noisy)
        │                          # Giá trị: 0-7 (dự đoán)
        └── labels/
            └── aachen_1.tif      # Ground truth từ OpenEarthMap (clean)
                                   # Giá trị: 0-7 (label thật)

    Mục đích:
        Cung cấp bộ 3 dữ liệu cho DAE training:
        - Input:  RGB + pseudo-label (noisy)
        - Target: Ground truth (clean)

    Lưu ý:
        - Chỉ thêm vào nếu cả 3 file đều tồn tại
        - Pseudo-label và ground truth có cùng kích thước và align với ảnh
    """
    pairs = []
    with open(split_file) as f:
        filenames = [l.strip() for l in f if l.strip()]

    for fn in filenames:
        img_path = os.path.join(pseudo_root, 'images', fn)
        pseudo_path = os.path.join(pseudo_root, 'pseudolabels', fn)
        gt_path = os.path.join(pseudo_root, 'labels', fn)
        if os.path.exists(img_path) and os.path.exists(pseudo_path) and os.path.exists(gt_path):
            pairs.append((img_path, pseudo_path, gt_path))

    return pairs


# =============================================================================
# RealNoiseDAEDataset - Dataset for Denoising AutoEncoder Training
# =============================================================================

class RealNoiseDAEDataset(Dataset):
    """
    Dataset cho DAE training với pseudo-label thật từ CISC-R.

    Cấu trúc folder:
        pseudo_root/
        ├── images/{fn}.tif       → Ảnh RGB
        │                           [H, W, 3], uint8 [0, 255]
        ├── pseudolabels/{fn}.tif → Pseudo-label từ CISC-R (noisy)
        │                           [H, W], uint8 [0, 8] (0=background)
        └── labels/{fn}.tif       → Ground truth từ OpenEarthMap (clean)
                                    [H, W], uint8 [0, 8] (0=background)

    Data flow cho DAE training:

    ┌─────────────────────────────────────────────────────────────────────┐
    │  Input: 3 files                                                     │
    │  - image.tif (RGB)                                                  │
    │  - pseudo_label.tif (noisy prediction from CISC-R)                  │
    │  - ground_truth.tif (clean label from OpenEarthMap)                 │
    └─────────────────────────────────────────────────────────────────────┘
                                    ↓
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Preprocessing:                                                     │
    │  - Read (cv2.imread)                                                │
    │  - Resize to 512x512                                                │
    │  - Augmentation (flip, rotation) - train only                       │
    └─────────────────────────────────────────────────────────────────────┘
                                    ↓
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Output:                                                            │
    │  - rgb:           [3, 512, 512] float32 [0, 1]                      │
    │  - pseudo_onehot: [8, 512, 512] float32 (one-hot encoded)           │
    │  - clean_label:   [512, 512] int64 [0, 7], background = 255         │
    └─────────────────────────────────────────────────────────────────────┘

    Output mỗi sample:
        (rgb[3,H,W], pseudo_onehot[8,H,W], clean_label[H,W])
        - clean_label chứa 255 ở pixels background (ignore index)

    Ví dụ sử dụng:
        >>> dataset = RealNoiseDAEDataset('data/OEM_v2_aDanh', split='train')
        >>> rgb, pseudo, clean = dataset[0]
        >>> rgb.shape        # torch.Size([3, 512, 512])
        >>> pseudo.shape     # torch.Size([8, 512, 512])
        >>> clean.shape      # torch.Size([512, 512])
        >>> pseudo.min(), pseudo.max()  # tensor(0.), tensor(1.) - one-hot
    """

    def __init__(self, pseudo_root: str,
                 data_root: str = None,
                 split: str = 'train',
                 img_size: int = 512,
                 augment: bool = True):
        """
        Input:
            pseudo_root (str) - Thư mục dataset (chứa images/, pseudolabels/, labels/)
                                Ví dụ: 'data/OEM_v2_aDanh'
            data_root   (str) - Không còn dùng (ground truth đã có sẵn trong labels/)
            split       (str) - 'train', 'val', 'test'
            img_size    (int) - Kích thước resize (mặc định 512)
            augment     (bool) - Bật augmentation (chỉ khi split='train')

        Augmentation:
            - Horizontal flip (50%)
            - Vertical flip (50%)
            - Áp dụng đồng bộ cho rgb, pseudo_onehot, clean_label
        """
        self.img_size = img_size
        self.augment = augment and (split == 'train')

        # Tự động tạo split files nếu chưa có
        split_file = get_split_files(pseudo_root, split)

        self.pairs = find_pseudo_pairs(pseudo_root, split_file)
        print(f'RealNoiseDAEDataset {split}: {len(self.pairs)} samples')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Lấy 1 sample theo index.

        Input:
            idx (int) - Chỉ số sample trong dataset

        Output:
            Tuple chứa 3 tensors:

            1. rgb (tensor): [3, H, W] float32 [0, 1]
               - 3 kênh RGB đã normalize
               - Giá trị trong khoảng [0, 1]

            2. pseudo_onehot (tensor): [8, H, W] float32
               - 8 kênh one-hot encoded pseudo-label
               - Mỗi pixel có đúng 1 kênh = 1, các kênh còn lại = 0
               - Ví dụ: pixel thuộc class 3 (Road) → [0, 0, 0, 1, 0, 0, 0, 0]

            3. clean_label (tensor): [H, W] int64
               - Ground truth clean label
               - Mỗi pixel chứa class ID (0-7) sau khi map từ 1-8
               - Pixels background có giá trị 255 (ignore index)

        Quy trình xử lý chi tiết:

        ┌─────────────────────────────────────────────────────────────────┐
        │ STEP 1: Đọc 3 files từ disk                                     │
        ├─────────────────────────────────────────────────────────────────┤
        │ rgb (ảnh RGB):                                                  │
        │   - cv2.imread() → numpy [H, W, 3], uint8 [0, 255], BGR         │
        │   - cv2.cvtColor(BGR→RGB) → numpy [H, W, 3], uint8 [0, 255]     │
        │                                                                 │
        │ pseudo (pseudo-label):                                          │
        │   - cv2.imread() → numpy [H, W], uint8 [0, 7]                   │
        │   - Giá trị: class ID dự đoán từ model CISC-R                   │
        │                                                                 │
        │ clean (ground truth):                                           │
        │   - cv2.imread() → numpy [H, W], uint8 [0, 7]                   │
        │   - Giá trị: class ID thật từ OpenEarthMap                      │
        └─────────────────────────────────────────────────────────────────┘
                                    ↓
        ┌─────────────────────────────────────────────────────────────────┐
        │ STEP 2: Resize về 512x512                                       │
        ├─────────────────────────────────────────────────────────────────┤
        │ rgb:   INTER_LINEAR (bilinear) → [512, 512, 3]                  │
        │ pseudo: INTER_NEAREST (nearest) → [512, 512]                    │
        │ clean: INTER_NEAREST (nearest) → [512, 512]                     │
        └─────────────────────────────────────────────────────────────────┘
                                    ↓
        ┌─────────────────────────────────────────────────────────────────┐
        │ STEP 3: Chuyển tensor                                           │
        ├─────────────────────────────────────────────────────────────────┤
        │ rgb:   [512, 512, 3] → [3, 512, 512] + /255 → float32 [0, 1]    │
        │ clean: numpy [512, 512] → torch long [512, 512] + map 1-8→0-7 + background=255  │
        │ pseudo: numpy [512, 512] → one-hot [8, 512, 512] float32        │
        └─────────────────────────────────────────────────────────────────┘
                                    ↓
        ┌─────────────────────────────────────────────────────────────────┐
        │ STEP 4: Augmentation (nếu train, 50% mỗi phép)                  │
        ├─────────────────────────────────────────────────────────────────┤
        │ - Flip ngang: torch.flip(..., [2])                              │
        │ - Flip dọc:   torch.flip(..., [1])                              │
        │ - Áp dụng đồng bộ cho cả 3 tensors                              │
        └─────────────────────────────────────────────────────────────────┘

        Ví dụ chi tiết cho sample từ 'aachen_1.tif':

        Sau khi đọc (STEP 1):
            img:   numpy (512, 512, 3), uint8, [0, 255]
            pseudo: numpy (512, 512), uint8, [0, 7]
            clean: numpy (512, 512), uint8, [0, 7]

        Sau khi resize (STEP 2):
            img:   numpy (512, 512, 3), uint8
            pseudo: numpy (512, 512), uint8
            clean: numpy (512, 512), uint8

        Sau khi chuyển tensor (STEP 3):
            rgb:      torch.Size([3, 512, 512]), float32, [0, 1]
            clean:    torch.Size([512, 512]), int64, [0, 7]
            pseudo:   numpy (512, 512) → one-hot encoding

        One-hot encoding (STEP 3 continued):
            Input:  pseudo_np [512, 512], giá trị [0, 7]
            Output: pseudo_onehot [8, 512, 512], float32

            Với mỗi class c ∈ {0, 1, ..., 7}:
                pseudo_onehot[c, i, j] = 1 nếu pseudo_np[i, j] == c
                pseudo_onehot[c, i, j] = 0 nếu pseudo_np[i, j] != c

            Ví dụ: pixel tại (10, 20) có pseudo_np[10, 20] = 3 (Road)
                pseudo_onehot[:, 10, 20] = [0, 0, 0, 1, 0, 0, 0, 0]
                                           class 3 = 1, others = 0

        Class mapping (OpenEarthMap labels 0-8 → sau map 0-7):
            Label gốc 0 (background) → 255 (ignore)
            Label gốc 1 (Bareland)   → 0   Label gốc 2 (Rangeland)  → 1
            Label gốc 3 (Developed)  → 2   Label gốc 4 (Road)       → 3
            Label gốc 5 (Tree)       → 4   Label gốc 6 (Water)      → 5
            Label gốc 7 (Agriculture)→ 6   Label gốc 8 (Building)   → 7
        """
        img_path, pseudo_path, gt_path = self.pairs[idx]

        # === Đọc ảnh RGB ===
        # Input:  File .tif trên disk
        # Output: numpy [H, W, 3], uint8 [0, 255], BGR → RGB
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is not None and img.shape[2] > 3:
                img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # === Đọc pseudo-label (noise thật từ CISC-R) ===
        # Input:  File .tif chứa class IDs (dự đoán từ model)
        # Output: numpy [H, W], uint8 [0, 7]
        pseudo = cv2.imread(pseudo_path, cv2.IMREAD_UNCHANGED)
        if pseudo is None:
            try:
                import tifffile
                pseudo = tifffile.imread(pseudo_path)
            except:
                pseudo = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        if pseudo.ndim == 3:
            pseudo = pseudo[:, :, 0]

        # === Đọc ground truth (clean label từ OpenEarthMap) ===
        # Input:  File .tif chứa class IDs (label thật)
        # Output: numpy [H, W], uint8 [0, 7]
        clean = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if clean is None:
            try:
                import tifffile
                clean = tifffile.imread(gt_path)
            except:
                clean = pseudo.copy()  # fallback: use pseudo as clean
        if clean.ndim == 3:
            clean = clean[:, :, 0]

        # === Resize ===
        # img:   Bilinear interpolation cho ảnh RGB
        # pseudo, clean: Nearest-neighbor để giữ nguyên class IDs
        img = cv2.resize(img, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_LINEAR)
        clean = cv2.resize(clean, (self.img_size, self.img_size),
                           interpolation=cv2.INTER_NEAREST)
        pseudo = cv2.resize(pseudo, (self.img_size, self.img_size),
                            interpolation=cv2.INTER_NEAREST)

        # === Chuyển tensor ===
        # rgb: [H, W, 3] → [3, H, W] + normalize /255 → float32 [0, 1]
        img_t = torch.from_numpy(
            img.transpose(2, 0, 1).copy()
        ).float() / 255.0  # [3, H, W]

        # clean_label: numpy → torch long + map 1-8 → 0-7 (loại bỏ background 0)
        clean_label = torch.from_numpy(clean.copy()).long()
        # Clamp về [0, 8] trước, sau đó map: 0→255 (ignore), 1-8→0-7
        clean_label = torch.clamp(clean_label, 0, 8)
        # Tạo mask cho background (class 0) → đánh dấu 255 để ignore
        background_mask = (clean_label == 0)
        # Map 1-8 → 0-7
        clean_label = clean_label - 1
        # Pixels background thành 255 (ignore index)
        clean_label[background_mask] = 255

        # pseudo_np: clip giá trị về [0, 8], sau đó map 1-8 → 0-7
        pseudo_np = np.clip(pseudo.astype(np.int32), 0, 8).astype(np.int32)
        # Map 1-8 → 0-7, background (0) thành -1 (không one-hot)
        pseudo_np = pseudo_np - 1
        pseudo_np[pseudo_np < 0] = -1  # Background thành -1 (không one-hot)

        # === One-hot encode pseudo-label → [8, H, W] ===
        # Biến pseudo-label từ class IDs (scalar) sang one-hot encoding (vector)
        #
        # Input:  pseudo_np [H, W], giá trị scalar ∈ {-1, 0, 1, ..., 7}
        #         -1: background (không encode)
        #         0-7: class IDs sau khi map từ 1-8
        # Output: pseudo_onehot [8, H, W], float32
        #
        # Ví dụ: pixel có class ID = 3 (Road sau khi map)
        #   Before: pseudo_np[i, j] = 3 (từ class 4 ban đầu)
        #   After:  pseudo_onehot[:, i, j] = [0, 0, 0, 1, 0, 0, 0, 0]
        #                                      class 3 = 1
        pseudo_onehot = np.zeros(
            (NUM_CLASSES, self.img_size, self.img_size), dtype=np.float32
        )
        for c in range(NUM_CLASSES):
            pseudo_onehot[c] = (pseudo_np == c).astype(np.float32)
        pseudo_onehot = torch.from_numpy(pseudo_onehot)

        # === Augmentation (đồng bộ rgb + pseudo + target) ===
        # Flip ngang/dọc đồng bộ cho cả 3 tensors
        if self.augment:
            # Flip ngang (horizontal flip) - axis [2] tương ứng chiều W
            if torch.rand(1) > 0.5:
                img_t = torch.flip(img_t, [2])
                pseudo_onehot = torch.flip(pseudo_onehot, [2])
                clean_label = torch.flip(clean_label.unsqueeze(0), [2]).squeeze(0)
            # Flip dọc (vertical flip) - axis [1] tương ứng chiều H
            if torch.rand(1) > 0.5:
                img_t = torch.flip(img_t, [1])
                pseudo_onehot = torch.flip(pseudo_onehot, [1])
                clean_label = torch.flip(clean_label.unsqueeze(0), [1]).squeeze(0)

        return img_t, pseudo_onehot, clean_label
