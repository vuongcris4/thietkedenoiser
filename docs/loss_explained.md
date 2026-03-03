# Giải thích các hàm Loss trong DAE

## Tổng quan

Hệ thống dùng 3 hàm loss kết hợp trong class `DAELoss` ([dae_model.py](file:///home/ubuntu/thietkedenoiser/src/dae_model.py)):

```
L = ce_weight × CE + dice_weight × Dice + boundary_weight × Boundary
L = 1.0 × CE + 1.0 × Dice + 0.5 × Boundary   (giá trị mặc định)
```

---

## 1. CrossEntropy Loss (CE)

**Vai trò**: Phân loại từng pixel thuộc class nào (0–7).

**Cách hoạt động**: So sánh xác suất dự đoán với nhãn đúng, phạt nặng khi dự đoán sai.

**Công thức**:
```
CE = -log(P_correct_class)
```

**Ví dụ** (1 pixel):
```
Prediction (logits):  [0.1, 0.2, 5.0, 0.3, 0.1, 0.1, 0.1, 0.1]
                                  ↑ class 2 điểm cao nhất
Target:               class 2 ✅

→ P(class=2) = softmax(5.0) ≈ 0.95
→ CE = -log(0.95) = 0.05   (loss thấp, dự đoán đúng!)
```

```
Prediction (logits):  [0.1, 5.0, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1]
                             ↑ class 1 điểm cao nhất
Target:               class 2 ❌

→ P(class=2) = softmax(0.2) ≈ 0.01
→ CE = -log(0.01) = 4.6    (loss cao, dự đoán sai!)
```

**Ưu điểm**: Đơn giản, hiệu quả cho bài toán phân loại cơ bản.
**Nhược điểm**: Không xử lý tốt class imbalance (class có ít pixel bị "lấn át").

---

## 2. Dice Loss

**Vai trò**: Đo mức độ overlap giữa vùng dự đoán và vùng ground truth. Xử lý tốt **class imbalance**.

**Công thức**:
```
Dice_c = 2 × |Prediction_c ∩ Target_c| / (|Prediction_c| + |Target_c|)
Dice Loss = 1 - mean(Dice_c)   với c = 0..7 (8 class)
```

**Ví dụ minh họa** (ảnh 4×4, 1 class):
```
Prediction:          Target (GT):         Overlap:
┌───┬───┬───┬───┐   ┌───┬───┬───┬───┐   ┌───┬───┬───┬───┐
│ 1 │ 1 │ 0 │ 0 │   │ 1 │ 1 │ 1 │ 0 │   │ ✓ │ ✓ │   │   │
│ 1 │ 1 │ 0 │ 0 │   │ 1 │ 1 │ 1 │ 0 │   │ ✓ │ ✓ │   │   │
│ 0 │ 0 │ 0 │ 0 │   │ 0 │ 0 │ 0 │ 0 │   │   │   │   │   │
│ 0 │ 0 │ 0 │ 0 │   │ 0 │ 0 │ 0 │ 0 │   │   │   │   │   │
└───┴───┴───┴───┘   └───┴───┴───┴───┘   └───┴───┴───┴───┘
|Prediction| = 4     |Target| = 6        |Overlap| = 4

Dice = 2 × 4 / (4 + 6) = 0.8
Dice Loss = 1 - 0.8 = 0.2
```

**Ưu điểm**: Không bị ảnh hưởng bởi class imbalance (class nhỏ vẫn có trọng số công bằng).
**Nhược điểm**: Gradient không ổn định khi vùng overlap rất nhỏ.

---

## 3. Boundary Loss

**Vai trò**: Phạt thêm tại **vùng biên** giữa các đối tượng — nơi nhiễu thường xảy ra nhiều nhất.

**Cách hoạt động**:
1. Dùng **bộ lọc Sobel** phát hiện cạnh trên ground truth
2. Tính CE loss cho từng pixel
3. Chỉ giữ CE ở vùng biên (mask), bỏ qua vùng bên trong

```
Sobel X:              Sobel Y:
┌────┬────┬────┐     ┌────┬────┬────┐
│ -1 │  0 │ +1 │     │ -1 │ -2 │ -1 │
│ -2 │  0 │ +2 │     │  0 │  0 │  0 │
│ -1 │  0 │ +1 │     │ +1 │ +2 │ +1 │
└────┴────┴────┘     └────┴────┴────┘
```

**Ví dụ minh họa**:
```
Ground Truth:          Boundary Mask:         CE × Mask:
┌───┬───┬───┬───┐     ┌───┬───┬───┬───┐     ┌───┬───┬───┬───┐
│ A │ A │ B │ B │     │   │ ★ │ ★ │   │     │   │2.1│0.3│   │
│ A │ A │ B │ B │     │   │ ★ │ ★ │   │     │   │0.5│1.2│   │
│ C │ C │ C │ C │  →  │ ★ │ ★ │ ★ │ ★ │  →  │0.8│0.1│0.4│0.2│
│ C │ C │ C │ C │     │   │   │   │   │     │   │   │   │   │
└───┴───┴───┴───┘     └───┴───┴───┴───┘     └───┴───┴───┴───┘
                       ★ = pixel biên         Chỉ tính loss ở biên
```

**Ưu điểm**: Cải thiện đường viền segmentation, đặc biệt quan trọng cho ảnh vệ tinh.
**Nhược điểm**: Tốn thêm chi phí tính toán (Sobel convolution).

---

## Tại sao kết hợp cả 3?

| Loss | Giỏi ở | Yếu ở |
|------|--------|-------|
| **CE** | Phân loại chính xác từng pixel | Class imbalance |
| **Dice** | Cân bằng giữa các class | Gradient không ổn định |
| **Boundary** | Chi tiết vùng biên | Chỉ tập trung ở biên |

→ **Kết hợp 3 loss bù trừ điểm yếu của nhau**:
- CE → đảm bảo phân loại cơ bản đúng
- Dice → đảm bảo class nhỏ không bị bỏ qua
- Boundary → cải thiện chất lượng ở vùng ranh giới

---

## Code tương ứng

```python
# Trong dae_model.py - class DAELoss
def forward(self, pred, target):
    ce_loss = self.ce(pred, target)              # ← CrossEntropy
    dice_loss = self._dice_loss(pred, target)    # ← Dice
    loss = 1.0 * ce_loss + 1.0 * dice_loss

    if self.boundary_weight > 0:
        boundary_loss = self._boundary_loss(pred, target)  # ← Boundary
        loss += 0.5 * boundary_loss

    return loss
```
