# Giải thích cấu hình (Configuration) của hệ thống DAE

## Cơ chế kế thừa (`_base_`)

Tất cả config model đều kế thừa từ `default.yaml` qua trường `_base_`:

```yaml
_base_: "default.yaml"   # Kế thừa toàn bộ config từ default.yaml
```

Giá trị trong file con sẽ **ghi đè** giá trị tương ứng trong `default.yaml`.

---

## 1. `default.yaml` — Config mặc định (chia sẻ cho tất cả model)

```yaml
data_root: "../data/OpenEarthMap"   # Đường dẫn đến dataset
img_size: 512                       # Kích thước ảnh (resize về 512x512)
num_classes: 8                      # Số lớp phân đoạn (8 loại đất/vật thể)
seed: 42                            # Seed để tái lập kết quả (reproducibility)
num_workers: 2                      # Số thread load dữ liệu song song
pin_memory: true                    # Ghim bộ nhớ → tăng tốc chuyển CPU→GPU
save_dir: "../checkpoints"          # Lưu model checkpoint (.pth)
log_dir: "../results/logs"          # Lưu log huấn luyện (loss, metrics)
device: "auto"                      # auto=tự chọn GPU/CPU, hoặc "cuda"/"cpu"
```

| Tham số | Giá trị | Tại sao chọn giá trị này? |
|---------|---------|--------------------------|
| `img_size: 512` | 512×512 px | Cân bằng chi tiết ảnh vs bộ nhớ GPU |
| `seed: 42` | 42 | Quy ước phổ biến, đảm bảo chạy lại ra cùng kết quả |
| `num_workers: 2` | 2 | Phù hợp server nhỏ, tăng lên nếu CPU mạnh |
| `pin_memory: true` | true | Tăng tốc transfer dữ liệu lên GPU |

---

## 2. Config Model — Khối `model:`

### DAE Models

```yaml
# dae_lightweight.yaml
model:
  name: "lightweight"       # Tên model → build_model('lightweight')
  in_channels: 11           # 3 (RGB) + 8 (one-hot label) = 11 kênh đầu vào
  num_classes: 8            # 8 lớp đầu ra

# dae_resnet34.yaml
model:
  name: "unet_resnet34"     # U-Net + ResNet-34 backbone
  encoder_name: "resnet34"
  encoder_weights: "imagenet"  # Dùng pretrained weights
  in_channels: 11
  num_classes: 8

# dae_effnet.yaml
model:
  name: "unet_effnet"          # U-Net + EfficientNet-B4
  encoder_name: "efficientnet-b4"
  encoder_weights: "imagenet"
  in_channels: 11
  num_classes: 8
```

---

## 3. Khối `training:` — Cấu hình huấn luyện

```yaml
training:
  batch_size: 4          # Số ảnh mỗi batch (4 do giới hạn GPU memory)
  epochs: 100            # Số vòng huấn luyện tối đa
  lr: 1.0e-4             # Learning rate = 0.0001
  weight_decay: 1.0e-4   # L2 regularization, chống overfitting
  optimizer: "adamw"     # AdamW: Adam + weight decay riêng biệt
  scheduler: "cosine"    # Cosine Annealing: giảm lr theo đường cosine
  patience: 15           # Early stopping: dừng nếu 15 epoch không cải thiện
```

| Tham số | Ý nghĩa |
|---------|---------|
| `batch_size: 4` | Nhỏ vì ảnh 512×512 tốn nhiều VRAM |
| `lr: 1e-4` | Learning rate phổ biến cho fine-tuning |
| `optimizer: "adamw"` | Tốt hơn Adam gốc nhờ tách weight decay |
| `scheduler: "cosine"` | LR giảm mượt: cao lúc đầu → thấp lúc cuối |
| `patience: 15` | Tránh train quá lâu khi model không cải thiện |

---

## 4. Khối `loss:` — Trọng số hàm loss

```yaml
loss:
  ce_weight: 1.0          # Trọng số CrossEntropy Loss
  dice_weight: 1.0        # Trọng số Dice Loss
  boundary_weight: 0.5    # Trọng số Boundary Loss
```

**Công thức**: `L = 1.0 × CE + 1.0 × Dice + 0.5 × Boundary`

| Loss | Vai trò |
|------|---------|
| **CrossEntropy** | Phân loại pixel, tốt cho bài toán cơ bản |
| **Dice** | Đo overlap, xử lý tốt class imbalance |
| **Boundary** | Phạt thêm ở vùng biên, cải thiện đường viền |

---

## 5. Khối `noise:` — Cấu hình nhiễu

```yaml
noise:
  type: "mixed"       # Kiểu nhiễu: trộn ngẫu nhiên 4 loại
  rate_min: 0.05      # Tỷ lệ nhiễu tối thiểu (5% pixel bị sai)
  rate_max: 0.30      # Tỷ lệ nhiễu tối đa (30% pixel bị sai)
```

4 loại nhiễu trong `mixed`:

| Loại | Mô tả |
|------|-------|
| **Random Flip** | Đổi ngẫu nhiên nhãn của một số pixel |
| **Boundary** | Thêm nhiễu tại vùng biên giữa các đối tượng |
| **Region Swap** | Hoán đổi nhãn của một vùng liền kề |
| **Confusion** | Nhầm lẫn giữa các class giống nhau |

---

## Tổng quan: 5 file config

| File | Model | Đặc điểm |
|------|-------|----------|
| `default.yaml` | — | Config gốc, chứa data/device/path |
| `dae_lightweight.yaml` | LightweightDAE | CNN nhẹ ~12.8M params |
| `dae_resnet34.yaml` | UNet+ResNet-34 | Pretrained, ~24M params |
| `dae_effnet.yaml` | UNet+EfficientNet-B4 | Pretrained, cân bằng |
