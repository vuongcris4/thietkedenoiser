# Các Phương Pháp Tạo Noise cho Label Segmentation

> **File nguồn**: [`src/noise_generator.py`](../src/noise_generator.py)
>
> **Mục đích**: Tạo nhiễu nhân tạo trên label segmentation để huấn luyện DAE (Denoising AutoEncoder).
> DAE học cách: nhận label bị sai + ảnh gốc → dự đoán label đúng.

---

## Tổng Quan

| # | Phương pháp | Đặc điểm | Mô phỏng lỗi thực tế |
|---|---|---|---|
| 1 | **Random Flip** | Đổi ngẫu nhiên class, phân bố đều | Nhiễu ngẫu nhiên từ model yếu |
| 2 | **Boundary** | Nhiễu tập trung ở biên giữa các vùng | Lỗi segmentation tại đường biên |
| 3 | **Region Swap** | Hoán đổi class của cả vùng lớn | Model nhầm toàn bộ 1 khu vực |
| 4 | **Confusion-based** | Nhiễu theo xác suất nhầm thực tế | Các cặp class thường bị nhầm |
| 5 | **Mixed** | Kết hợp cả 4 loại trên | Nhiễu đa dạng, gần thực tế nhất |

Tất cả phương pháp nhận input và trả output cùng format:

```
Input:  label [H, W] int (0-7)    ← label sạch
        noise_rate: float          ← tỉ lệ pixel bị đổi (vd: 0.1 = 10%)
Output: noisy_label [H, W] int    ← label đã bị nhiễu
```

---

## 1. Random Flip Noise

### Ý tưởng
Chọn ngẫu nhiên ~`noise_rate`% pixel, đổi class sang **class khác bất kỳ**.

### Cách hoạt động
1. Tạo mask ngẫu nhiên: pixel nào được đổi (xác suất = `noise_rate`)
2. Sinh class mới ngẫu nhiên cho mỗi pixel (đảm bảo khác class cũ)
3. Ghi đè pixel được chọn bằng class mới

### Ví dụ (noise_rate = 0.2)

```
Label sạch:          Label nhiễu:
┌─────────────┐      ┌─────────────┐
│ 0  0  5  5  │      │ 0  0  5  5  │
│ 0  3  5  5  │  →   │ 0  6★ 5  5  │   ★ pixel (1,1): 3→6
│ 7  3  4  4  │      │ 7  3  4  1★ │   ★ pixel (2,3): 4→1
│ 7  7  4  4  │      │ 7  7  2★ 4  │   ★ pixel (3,2): 4→2
└─────────────┘      └─────────────┘
```

### Đặc điểm
- ✅ Đơn giản, nhanh
- ✅ Phân bố đều trên toàn ảnh
- ❌ Không thực tế — lỗi thực ít khi phân bố đều "dạng dom dom"

---

## 2. Boundary Noise

### Ý tưởng
Nhiễu tập trung ở **ranh giới giữa các vùng class khác nhau** — mô phỏng lỗi segmentation thường gặp nhất tại đường biên.

### Cách hoạt động
1. Tạo kernel hình ellipse (kích thước phụ thuộc `noise_rate`, từ 3 đến 15 pixel)
2. Duyệt từng class `c`:
   - Tạo binary mask cho class `c`
   - Random chọn **dilate** hoặc **erode**:
     - **Dilate**: mở rộng vùng class `c` ra ngoài → pixel biên bị gán thành class `c`
     - **Erode**: thu hẹp vùng class `c` → pixel biên bị gán thành class lân cận

### Ví dụ — Dilate class 5 (Water)

```
Label sạch:          Sau dilate class 5:
┌─────────────┐      ┌─────────────┐
│ 0  0  5  5  │      │ 0  5★ 5  5  │   ★ biên bị mở rộng
│ 0  3  5  5  │  →   │ 0  5★ 5  5  │   ★ biên bị mở rộng
│ 7  3  4  4  │      │ 7  3  5★ 4  │   ★ biên bị mở rộng
│ 7  7  4  4  │      │ 7  7  4  4  │
└─────────────┘      └─────────────┘
```

### Ví dụ — Erode class 5 (Water)

```
Label sạch:          Sau erode class 5:
┌─────────────┐      ┌─────────────┐
│ 0  0  5  5  │      │ 0  0  3★ 5  │   ★ biên bị thu hẹp → gán class lân cận
│ 0  3  5  5  │  →   │ 0  3  5  5  │
│ 7  3  4  4  │      │ 7  3  4  4  │
│ 7  7  4  4  │      │ 7  7  4  4  │
└─────────────┘      └─────────────┘
```

### Đặc điểm
- ✅ Rất thực tế — segmentation model thường sai nhiều nhất ở biên
- ✅ Cấu trúc tốt (không rải rác)
- ❌ Chịu ảnh hưởng bởi kernel size

---

## 3. Region Swap Noise

### Ý tưởng
Chọn **vùng hình chữ nhật ngẫu nhiên**, tìm class chiếm đa số trong vùng đó, rồi **đổi tất cả pixel class đó sang class dễ nhầm**.

### Cách hoạt động
1. Tính số pixel cần đổi = `H × W × noise_rate`
2. Lặp tối đa 50 lần:
   - Tạo vùng chữ nhật ngẫu nhiên (20-100 pixel mỗi chiều)
   - Tìm class chiếm đa số (`dominant_class`) trong vùng
   - Lấy class dễ nhầm qua `_get_confused_class()`
   - Đổi tất cả pixel `dominant_class` → class mới
3. Dừng khi đủ số pixel cần đổi

### Bảng cặp class dễ nhầm (`_get_confused_class`)

| Class gốc | Class thường nhầm sang | Lý do |
|---|---|---|
| 0 (Bareland) | 3 (Road) | Cả hai đều là bề mặt trống |
| 3 (Road) | 0 (Bareland) | Ngược lại |
| 1 (Rangeland) | 6 (Agriculture) | Cả hai đều là thảm thực vật |
| 6 (Agriculture) | 1 (Rangeland) | Ngược lại |
| 4 (Tree) | 1 (Rangeland) | Đều xanh lá |
| 7 (Building) | 3 (Road) | Đều là công trình xây dựng |
| 2 (Developed) | 0 (Bareland) | Khu đô thị giống đất trống |
| 5 (Water) | 4 (Tree) | Phản xạ tương tự từ vệ tinh |

> 60% xác suất đổi sang class trong bảng trên, 40% đổi sang class ngẫu nhiên khác.

### Ví dụ (noise_rate = 0.2)

```
Label sạch:          Region swap (vùng [0:2, 2:4]):
┌─────────────┐      ┌─────────────┐
│ 0  0 [5  5] │      │ 0  0 [4★ 4★]│   ★ Water→Tree (cả vùng)
│ 0  3 [5  5] │  →   │ 0  3 [4★ 4★]│
│ 7  3  4  4  │      │ 7  3  4  4  │
│ 7  7  4  4  │      │ 7  7  4  4  │
└─────────────┘      └─────────────┘
```

### Đặc điểm
- ✅ Mô phỏng lỗi thực tế: model nhầm toàn bộ khu vực
- ✅ Nhiễu có cấu trúc (vùng liên tục)
- ❌ Vùng chữ nhật không tự nhiên bằng vùng polygon

---

## 4. Confusion-based Noise

### Ý tưởng
Đổi class theo **xác suất nhầm thực tế** giữa các class — class nào dễ nhầm với nhau thì xác suất bị đổi sang nhau cao hơn.

### Cách hoạt động
1. Tạo mask: chọn ~`noise_rate`% pixel ngẫu nhiên
2. Với mỗi class `c`:
   - Lấy hàng `c` từ confusion matrix → xác suất chuyển sang từng class khác
   - Đặt `prob[c→c] = 0` (không giữ nguyên class cũ)
   - Normalize lại xác suất
   - Random chọn class mới theo phân phối xác suất

### Ma trận Confusion (đã normalize)

Giá trị cao = hay nhầm với nhau:

```
              Bare  Range Devel Road  Tree  Water Agri  Build
Bareland      ─     0.15  0.02  0.35  0.02  0.02  0.02  0.02
Rangeland     0.15  ─     0.02  0.02  0.02  0.08  0.25  0.02
Developed     0.02  0.02  ─     0.02  0.02  0.02  0.02  0.02
Road          0.35  0.02  0.02  ─     0.12  0.02  0.02  0.10
Tree          0.02  0.02  0.02  0.12  ─     0.02  0.02  0.08
Water         0.02  0.08  0.02  0.02  0.02  ─     0.02  0.02
Agriculture   0.02  0.25  0.02  0.02  0.02  0.02  ─     0.02
Building      0.02  0.02  0.02  0.10  0.08  0.02  0.02  ─
```

**Cặp dễ nhầm nhất**: Bareland ↔ Road (0.35), Rangeland ↔ Agriculture (0.25)

### Ví dụ (noise_rate = 0.15)

```
Label sạch:          Confusion-based noise:
┌─────────────┐      ┌─────────────┐
│ 0  0  5  5  │      │ 3★ 0  5  5  │   ★ Bareland→Road (prob 0.35)
│ 0  3  5  5  │  →   │ 0  3  5  5  │
│ 7  3  4  4  │      │ 7  0★ 4  4  │   ★ Road→Bareland (prob 0.35)
│ 7  7  4  4  │      │ 7  7  4  7★ │   ★ Tree→Building (prob 0.08)
└─────────────┘      └─────────────┘
```

### Đặc điểm
- ✅ Thực tế nhất — mô phỏng đúng kiểu lỗi của segmentation model
- ✅ Xác suất nhầm phản ánh visual similarity giữa các class
- ❌ Vẫn rải rác pixel đơn lẻ (không có cấu trúc vùng)

---

## 5. Mixed Noise

### Ý tưởng
**Kết hợp tuần tự cả 4 loại noise**, mỗi loại với `noise_rate / 4` — tạo ra nhiễu đa dạng và gần thực tế nhất.

### Cách hoạt động

```
noise_rate_each = noise_rate / 4

label_sạch ──→ boundary_noise(rate_each)
           ──→ confusion_based_noise(rate_each)
           ──→ region_swap_noise(rate_each)
           ──→ random_flip_noise(rate_each)
           ──→ label_nhiễu_cuối_cùng
```

### Ví dụ (noise_rate = 0.2 → mỗi loại 0.05)

```
Label sạch:        Sau boundary:      Sau confusion:     Sau region_swap:   Sau random_flip:
┌───────────┐     ┌───────────┐      ┌───────────┐      ┌───────────┐      ┌───────────┐
│ 0 0 5 5   │     │ 0 5 5 5   │      │ 0 5 5 5   │      │ 0 5 5 5   │      │ 0 5 5 5   │
│ 0 3 5 5   │  →  │ 0 3 5 5   │  →   │ 0 0 5 5   │  →   │ 0 0 4 4   │  →   │ 0 0 4 4   │
│ 7 3 4 4   │     │ 7 3 4 4   │      │ 7 3 4 4   │      │ 7 3 4 4   │      │ 7 3 4 6   │
│ 7 7 4 4   │     │ 7 7 4 4   │      │ 7 7 4 4   │      │ 7 7 4 4   │      │ 7 7 4 4   │
└───────────┘     └───────────┘      └───────────┘      └───────────┘      └───────────┘
  Gốc              Biên (0,1)→5       Road(1,1)→Bare     Water→Tree          (2,3)→6
```

### Đặc điểm
- ✅ Đa dạng nhất — model phải học xử lý nhiều loại lỗi
- ✅ Robust training — DAE không overfit vào 1 pattern noise cụ thể
- ✅ **Đây là noise mặc định** được dùng cho training

---

## Hàm Phụ Trợ

### `compute_noise_stats(clean, noisy)`

Tính thống kê so sánh label sạch vs label nhiễu:

| Metric | Ý nghĩa |
|---|---|
| `total_pixels` | Tổng số pixel (H × W) |
| `changed_pixels` | Số pixel bị đổi class |
| `actual_noise_rate` | Tỉ lệ nhiễu thực tế |
| `boundary_changed` | Số pixel bị đổi **nằm ở biên** |
| `interior_changed` | Số pixel bị đổi **nằm ở nội bộ vùng** |
| `boundary_ratio` | Tỉ lệ nhiễu tại biên / tổng nhiễu |
| `per_class` | Thống kê riêng cho từng class |

### `compute_iou(pred, gt)`

Tính IoU (Intersection over Union) cho từng class và mIoU trung bình — dùng để đánh giá chất lượng prediction.

---

## Cách Sử Dụng trong Pipeline

```python
# Trong DAEDataset.__getitem__():

# 1. Lấy label sạch
img, clean_label = base_dataset[idx]              # [3,H,W], [H,W]

# 2. Tạo noise
noise_gen = NoiseGenerator(num_classes=8)
noise_rate = random.uniform(0.05, 0.30)            # Random 5-30%
noisy_label = noise_gen.mixed_noise(clean_label, noise_rate)  # [H,W]

# 3. One-hot encode
noisy_onehot = one_hot(noisy_label, num_classes=8)  # [8,H,W]

# 4. Concat
dae_input = concat(img, noisy_onehot)               # [11,H,W]

# 5. Model học: dae_input[11,H,W] → predict clean_label[H,W]
```
