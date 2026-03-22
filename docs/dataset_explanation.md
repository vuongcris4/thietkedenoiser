# Giải thích `dataset.py`

> File: [dataset.py](file:///home/ubuntu/thietkedenoiser/src/dataset.py)
> Mục đích: Đọc dữ liệu ảnh vệ tinh OpenEarthMap (.tif) → tạo cặp (input, target) cho training DAE.

---

## Tổng quan

File cung cấp **2 class Dataset** chính:

| Class | Vai trò | Output |
|---|---|---|
| `OpenEarthMapDataset` | Đọc ảnh + label gốc | `(img[3,H,W], label[H,W])` |
| `DAEDataset` | Wrapper: thêm nhiễu vào label | `(dae_input[11,H,W], clean_label[H,W])` |

---

## Dataset gốc: File `.tif` chứa gì?

### 📷 `images/*.tif` — Ảnh vệ tinh RGB
- 3 kênh RGB, uint8, giá trị 0–255
- Kích thước gốc: thường 1024×1024 pixel
- Chụp từ vệ tinh nhìn xuống mặt đất

### 🏷️ `labels/*.tif` — Label segmentation
- 1 kênh grayscale, uint8, giá trị 0–7
- Mỗi pixel = 1 class index

| Giá trị | Class | Ý nghĩa |
|---|---|---|
| 0 | Bareland | Đất trống |
| 1 | Rangeland | Đồng cỏ |
| 2 | Developed | Khu phát triển |
| 3 | Road | Đường |
| 4 | Tree | Cây |
| 5 | Water | Mặt nước |
| 6 | Agriculture | Nông nghiệp |
| 7 | Building | Tòa nhà |

### Cấu trúc thư mục

```
data_root/
├── train.txt                    ← danh sách tên file cho training
├── val.txt                      ← danh sách tên file cho validation
├── aachen/
│   ├── images/aachen_1.tif      ← ảnh RGB
│   └── labels/aachen_1.tif      ← label segmentation
├── berlin/
│   ├── images/berlin_1.tif
│   └── labels/berlin_1.tif
└── ...
```

---

## Các hàm và class chi tiết

### `_get_region(filename)`

| | |
|---|---|
| **Input** | `filename` (str) — tên file, vd: `'aachen_1.tif'` |
| **Output** | str — tên region, vd: `'aachen'` |
| **Cách hoạt động** | Tách chuỗi bằng dấu `_` cuối cùng, lấy phần trước |
| **Mục đích** | Xác định thư mục con chứa ảnh (`data_root/{region}/images/`) |

---

### `find_oem_pairs(data_root, split_file)`

| | |
|---|---|
| **Input** | `data_root` — đường dẫn gốc data; `split_file` — file `.txt` chứa danh sách tên file |
| **Output** | `List[Tuple[str, str]]` — danh sách cặp `(image_path, label_path)` đã verify tồn tại |
| **Mục đích** | Ghép đường dẫn đầy đủ cho ảnh và label từ split file (train/val/test) |

Ví dụ: `aachen_1.tif` →
- `data_root/aachen/images/aachen_1.tif`
- `data_root/aachen/labels/aachen_1.tif`

> Lưu ý: cặp nào thiếu ảnh hoặc label sẽ bị bỏ qua (không báo lỗi).

---

### Class `OpenEarthMapDataset`

Dataset cơ sở: đọc ảnh vệ tinh RGB + label segmentation.

#### `__init__(data_root, split, img_size, augment)`

| Param | Ý nghĩa |
|---|---|
| `data_root` | Thư mục gốc chứa data |
| `split` | `'train'` hoặc `'val'` |
| `img_size` | Kích thước resize (mặc định 512×512) |
| `augment` | Bật augmentation (chỉ áp dụng khi `split='train'`) |

#### `__getitem__(idx)` — Luồng xử lý

| Bước | Biến | Type | Shape | Giá trị |
|---|---|---|---|---|
| Đọc file ảnh | `img` | numpy uint8 | `[H, W, 3]` | 0–255 (BGR) |
| cvtColor | `img` | numpy uint8 | `[H, W, 3]` | 0–255 (RGB) |
| Resize | `img` | numpy uint8 | `[512, 512, 3]` | 0–255 |
| **→ Tensor** | **`img`** | **torch float32** | **`[3, 512, 512]`** | **0–1** |
| Đọc file label | `label` | numpy uint8 | `[H, W]` | 0–7 |
| Resize | `label` | numpy uint8 | `[512, 512]` | 0–7 |
| **→ Tensor** | **`label`** | **torch long** | **`[512, 512]`** | **0–7** |

> `cv2.imread(IMREAD_COLOR)` → numpy BGR. Nếu thất bại → fallback `IMREAD_UNCHANGED` (đọc nguyên bản, trả numpy array nhiều chiều, giữ tất cả kênh/bit-depth).
> Dùng `INTER_NEAREST` cho label để giữ nguyên class index (không nội suy).

#### `_augment(img, label)` — 3 phép biến đổi (đồng bộ ảnh & label)

1. Flip ngang (50% xác suất)
2. Flip dọc (50% xác suất)
3. Xoay 0°/90°/180°/270° (random đều)

---

### Class `DAEDataset`

Wrapper bọc `OpenEarthMapDataset`, thêm **nhiễu nhân tạo** vào label.

#### `__init__(...)`

| Param | Ý nghĩa |
|---|---|
| `noise_type` | Loại nhiễu: `'random_flip'`, `'boundary'`, `'region_swap'`, `'confusion_based'`, `'mixed'` |
| `noise_rate_range` | Phạm vi tỉ lệ nhiễu, vd `(0.05, 0.30)` = 5–30% pixel bị đổi class |

5 chiến lược nhiễu:

| Key | Ý nghĩa |
|---|---|
| `random_flip` | Đổi ngẫu nhiên class của pixel |
| `boundary` | Thêm nhiễu ở biên giữa các vùng |
| `region_swap` | Hoán đổi class giữa các vùng |
| `confusion_based` | Nhiễu dựa trên confusion matrix |
| `mixed` | Kết hợp ngẫu nhiên các loại trên |

#### `__getitem__(idx)` — 6 bước xử lý

```
Bước 1: Lấy (img[3,H,W], clean_label[H,W]) từ OpenEarthMapDataset
Bước 2: Random noise_rate trong khoảng (0.05, 0.30)
Bước 3: Áp dụng nhiễu → noisy_label [H,W] (numpy)
Bước 4: One-hot encode noisy_label → [8, H, W]
Bước 5: Concat img[3,H,W] + noisy_onehot[8,H,W] → dae_input[11, H, W]
Bước 6: Augmentation đồng bộ (flip ngang/dọc trên tensor)
```

**Output**: `(dae_input[11,H,W], clean_label[H,W])`

---

## Luồng dữ liệu tổng thể

```
Split file (train.txt / val.txt)
    │
    ▼
find_oem_pairs() → list[(img_path, label_path)]
    │
    ▼
OpenEarthMapDataset.__getitem__()
    │  Đọc .tif → resize → augment → tensor
    │  Output: (img[3,H,W], label[H,W])
    │
    ▼
DAEDataset.__getitem__()
    │  Thêm nhiễu → one-hot → concat
    │  Output: (dae_input[11,H,W], clean_label[H,W])
    │
    ▼
DataLoader → batch → Model (DAE) training
```

**Ý tưởng cốt lõi**: DAE học cách "khử nhiễu" label — nhận vào label bị sai (noisy) kết hợp với ảnh gốc → dự đoán ra label đúng (clean). Ứng dụng: cải thiện chất lượng pseudo-label trong thực tế.
