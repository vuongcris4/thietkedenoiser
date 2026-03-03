# Giải thích cú pháp `smp.Unet(...)` trong `dae_model.py`

## Code gốc

```python
self.model = smp.Unet(
    encoder_name='resnet34',       # ① Chọn backbone encoder
    encoder_weights='imagenet',    # ② Dùng trọng số pretrained
    in_channels=in_channels,       # ③ Số kênh đầu vào
    classes=num_classes,           # ④ Số lớp đầu ra
    activation=None,               # ⑤ Hàm kích hoạt output
)
```

## Giải thích từng tham số

| # | Tham số | Giá trị | Ý nghĩa |
|---|---------|---------|---------|
| ① | `encoder_name` | `'resnet34'` | Chọn kiến trúc encoder (backbone). Thư viện hỗ trợ nhiều loại: `resnet18`, `resnet50`, `efficientnet-b4`, `vgg16`, ... ResNet-34 có 34 lớp, cân bằng giữa hiệu suất và tốc độ |
| ② | `encoder_weights` | `'imagenet'` | Dùng trọng số đã train sẵn trên ImageNet (1.2 triệu ảnh). Encoder đã biết nhận diện cạnh, texture, hình dạng → chỉ cần fine-tune. Đặt `None` nếu muốn train từ đầu |
| ③ | `in_channels` | `11` | Số kênh đầu vào. Ở đây **11 = 3 (RGB) + 8 (one-hot label)**. Thư viện tự điều chỉnh lớp conv đầu tiên để nhận 11 kênh |
| ④ | `classes` | `8` | Số lớp phân đoạn đầu ra. Output shape: `[B, 8, H, W]` — mỗi kênh là logit của 1 lớp |
| ⑤ | `activation` | `None` | Không thêm hàm kích hoạt ở đầu ra → output là **logits thô** (chưa qua softmax). Ta tự xử lý bên ngoài khi tính loss |

## Kiến trúc U-Net tạo ra

```
Input [B, 11, H, W]
    │
    ▼
┌──────────────────┐
│  Encoder (ResNet) │  ← Pretrained ImageNet
│  /2 → /4 → /8... │    Giảm kích thước, tăng kênh đặc trưng
└────────┬─────────┘
         │ skip connections (nối tắt)
         ▼
┌──────────────────┐
│     Decoder       │  ← Tăng kích thước về gốc
│  *2 → *2 → *2... │    Kết hợp skip từ encoder
└────────┬─────────┘
         │
         ▼
   Output [B, 8, H, W]   ← logits (chưa softmax)
```

## So sánh: tự code vs dùng `smp`

```python
# ❌ Tự code (hàng trăm dòng):
self.conv1 = nn.Conv2d(11, 64, 3, padding=1)
self.bn1 = nn.BatchNorm2d(64)
self.conv2 = ...  # rất nhiều dòng

# ✅ Dùng smp (1 lệnh, có pretrained):
self.model = smp.Unet(encoder_name='resnet34', ...)
```

> [!TIP]
> `smp.Unet(...)` tạo sẵn toàn bộ kiến trúc U-Net (encoder + decoder + skip connections) chỉ trong 1 lệnh, thay vì tự code hàng trăm dòng.

---

## U-Net + Backbone (ResNet-34)

Đây là **kiến trúc U-Net** với **phần encoder (backbone) là ResNet-34**:

```
┌─────────────────────────────────────────────┐
│                   U-Net                      │
│                                              │
│   Encoder (backbone)        Decoder          │
│  ┌──────────────┐      ┌──────────────┐     │
│  │              │─skip─→│              │     │
│  │  ResNet-34   │─skip─→│  U-Net       │     │
│  │  (pretrained)│─skip─→│  Decoder     │     │
│  │              │─skip─→│              │     │
│  └──────────────┘      └──────────────┘     │
│   encoder_name=          (smp tự tạo)        │
│   'resnet34'                                 │
└─────────────────────────────────────────────┘
```

- **U-Net** = kiến trúc tổng thể (encoder → decoder + skip connections)
- **ResNet-34** = chỉ phần encoder (backbone), trích xuất đặc trưng từ ảnh
- **Decoder** + **skip connections** = smp tự tạo theo chuẩn U-Net

### Đổi backbone chỉ cần đổi `encoder_name`

```python
smp.Unet(encoder_name='resnet34')        # U-Net + ResNet-34
smp.Unet(encoder_name='resnet50')        # U-Net + ResNet-50
smp.Unet(encoder_name='efficientnet-b4') # U-Net + EfficientNet-B4
```

Phần decoder và skip connections **giữ nguyên**, chỉ thay encoder.

---

## `smp` là gì?

`segmentation_models_pytorch` **không phải** PyTorch built-in, mà là **thư viện bên thứ 3**:

| | `torch` / `torchvision` | `smp` |
|---|---|---|
| Nguồn | Chính hãng PyTorch | Bên thứ 3 (Pavel Iakubovskii) |
| Cài đặt | Có sẵn | `pip install segmentation-models-pytorch` |
| Chức năng | Framework DL tổng quát | Chuyên cho **image segmentation** |
| Model | Vài model cơ bản | **9+ kiến trúc** + **400+ encoder** pretrained |
