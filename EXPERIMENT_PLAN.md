# Plan Thí Nghiệm: Thiết Kế Bộ Denoiser (DAE)

## Mục Tiêu
Thiết kế và đánh giá bộ Denoising AutoEncoder (DAE) có khả năng:
- **Input**: concat(RGB image, noisy pseudo-label) → 11 channels
- **Output**: cleaned semantic label → 8 channels
- **Mục đích**: Biến pseudo-label nhiễu thành nhãn sạch (phục chế thay vì vứt bỏ)

---

## Phase 1: Setup & Chuẩn Bị Dữ Liệu (~2h)

### 1.1 Cài đặt môi trường
```bash
bash ~/thietkedenoiser/scripts/setup.sh
```

### 1.2 Download OpenEarthMap
```bash
bash ~/thietkedenoiser/scripts/download_data.sh
```

### 1.3 Tạo dữ liệu train cho DAE
Từ tập có nhãn (3000 ảnh train), tạo cặp (noisy_label, clean_label):

| Loại nhiễu | Mô tả | Tỷ lệ |
|------------|--------|--------|
| Random Flip | Đổi ngẫu nhiên class của 5-30% pixels | Mô phỏng lỗi chung |
| Boundary Erosion/Dilation | Dilate/Erode biên 2-10px | Lỗi phổ biến nhất |
| Region Swap | Hoán class của vùng superpixel | Nhầm class tương tự |
| Confusion-based | Dựa trên cặp hay nhầm: Bareland↔Developed, Rangeland↔Agriculture | Nhiễu thực tế |

**Data format:**
```
Input:  concat(RGB[3ch], noisy_onehot_label[8ch]) = 11 channels
Output: clean_onehot_label = 8 channels
Size:   512×512 (resize từ 1024)
```

---

## Phase 2: Thiết Kế Kiến Trúc DAE

### PA1: Convolutional DAE (U-Net style)
```
[B, 11, 512, 512]  (RGB + noisy label)
       ↓
   Encoder (Conv ↓ + skip connections)
       ↓
   Bottleneck (latent z)
       ↓
   Decoder (ConvTranspose ↑ + skip)
       ↓
[B, 8, 512, 512]   (clean label, softmax)
```

**Tại sao U-Net?**
- Skip connections giữ thông tin spatial (quan trọng cho boundary)
- Đã chứng minh hiệu quả cho segmentation
- Phù hợp với bài toán image-to-image

### PA2: Conditional DAE (RGB làm condition)
```
Noisy label [B, 8, 512, 512]
       ↓
   Encoder
       + Condition features từ RGB encoder (cross-attention/concat)
       ↓
   Decoder
       ↓
[B, 8, 512, 512]   (clean label)
```

**Khác PA1**: Tách riêng RGB encoder và label encoder, kết hợp ở bottleneck

---

## Phase 3: Chạy Thí Nghiệm (~16-24h)

### Exp 3.1: U-Net DAE với ResNet-34 encoder
| Config | Value |
|--------|-------|
| Architecture | U-Net encoder-decoder + skip connections |
| Encoder | ResNet-34 (modified first conv: 11 → 64ch) |
| Input | concat(RGB, noisy_onehot) = 11ch |
| Output | 8ch (softmax) |
| Size | 512×512 |
| Batch | 8 |
| Optimizer | AdamW, lr=1e-4, wd=1e-4 |
| Scheduler | CosineAnnealing, T_max=100 |
| Loss | CE + Dice |
| Epochs | 100 |
| Noise | Mixed (5-30%) |
| **VRAM est.** | **~8-10GB** |

### Exp 3.2: U-Net DAE với EfficientNet-B4 encoder
| Config | Value |
|--------|-------|
| Encoder | EfficientNet-B4 (modified input 11ch) |
| Còn lại | Giống Exp 3.1 |

### Exp 3.3: Conditional DAE (Dual-encoder)
| Config | Value |
|--------|-------|
| RGB Encoder | ResNet-34 (3ch input, frozen/finetuned) |
| Label Encoder | Custom Conv (8ch input) |
| Fusion | Concat ở bottleneck + cross-attention |
| Còn lại | Giống Exp 3.1 |

### Exp 3.4: Lightweight DAE
| Config | Value |
|--------|-------|
| Architecture | 5 Conv blocks (64→128→256→128→64) |
| Params | ~2-3M |
| Mục đích | Kiểm tra model nhỏ có đủ hiệu quả? |

---

## Phase 4: Đánh Giá Bộ Denoiser (~4h)

### Metrics đánh giá chất lượng khử nhiễu
| Metric | Mô tả |
|--------|--------|
| **mIoU recovery** | mIoU(DAE output) vs mIoU(clean GT) |
| **Per-class IoU** | Đặc biệt Bareland, Developed (class khó) |
| **Noise robustness** | Test với noise 5%, 10%, 20%, 30% |
| **Pixel accuracy** | % pixels được sửa đúng |
| **Boundary F1** | Chất lượng biên trước/sau DAE |

### Bảng kết quả cần điền
| Model | Noise 5% | Noise 10% | Noise 20% | Noise 30% | Params |
|-------|----------|-----------|-----------|-----------|--------|
| No denoising | - | - | - | - | 0 |
| Exp 3.1 ResNet-34 | | | | | |
| Exp 3.2 EffNet-B4 | | | | | |
| Exp 3.3 Conditional | | | | | |
| Exp 3.4 Lightweight | | | | | |

### Visualization
1. So sánh: GT vs Noisy vs DAE output (grid ảnh)
2. Boundary zoom: Phóng to vùng biên để thấy hiệu quả
3. Confusion matrix trước/sau DAE
4. Error map: Highlight pixels được sửa đúng/sai

---

## Timeline

| Phase | Thời gian |
|-------|-----------|
| Phase 1: Setup + data | 2h |
| Phase 2: Code kiến trúc | 4h |
| Phase 3: Train 4 experiments | 16-24h |
| Phase 4: Đánh giá + viết báo cáo | 4h |
| **Tổng** | **~26-34h** |

---

## Cấu Hình Phần Cứng
| Spec | Value |
|------|-------|
| GPU | NVIDIA L4, 23GB VRAM |
| RAM | 16GB |
| Disk | 134GB free |
| CPU | 4 cores |
| Max VRAM/exp | ~10GB (fit thoải mái) |

## Seed & Reproducibility
- Seed: 42
- Mixed precision: fp16
- Early stopping: patience 15
