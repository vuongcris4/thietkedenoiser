"""
=============================================================================
MỤC ĐÍCH CỦA FILE (File Purpose):
=============================================================================
File này định nghĩa 4 kiến trúc Denoising AutoEncoder (DAE) dùng để
"làm sạch" nhãn phân đoạn (pseudo-label) bị nhiễu trong ảnh vệ tinh.

Ý tưởng chính:
    - Đầu vào: ảnh RGB (3 kênh) + nhãn bị nhiễu dưới dạng one-hot (8 kênh)
               → tổng cộng 11 kênh
    - Đầu ra:  nhãn đã được làm sạch (logits) [B, 8, H, W]
    - Mô hình học cách "sửa lỗi" trong nhãn nhiễu dựa vào thông tin ảnh gốc

4 biến thể mô hình:
    1. UNetDAE_ResNet34   - U-Net + ResNet-34 encoder (mạnh, pretrained ImageNet)
    2. UNetDAE_EffNet     - U-Net + EfficientNet-B4 encoder (cân bằng hiệu quả)
    3. ConditionalDAE     - Dual-encoder: tách riêng RGB và label, fusion bằng attention
    4. LightweightDAE     - CNN nhẹ (~2-3M params), có skip connections

Ngoài ra file còn chứa:
    - ConvBlock, UpBlock  : các building block cơ bản
    - DAELoss             : hàm loss kết hợp (CrossEntropy + Dice + Boundary)
    - build_model()       : factory function để tạo model theo tên
    - count_params()      : đếm số lượng tham số model

Ví dụ sử dụng nhanh (Quick Usage):
    >>> from dae_model import build_model, DAELoss
    >>> model = build_model('lightweight')          # tạo model nhẹ
    >>> x = torch.randn(2, 11, 256, 256)            # batch 2, 11 kênh, 256x256
    >>> logits = model(x)                           # → [2, 8, 256, 256]
    >>> pred = logits.argmax(dim=1)                 # → [2, 256, 256] nhãn dự đoán
=============================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

# Số lượng lớp phân đoạn (8 loại đất/vật thể trong ảnh vệ tinh)
NUM_CLASSES = 8


# =============================================================================
# MODEL 1: UNetDAE_ResNet34
# =============================================================================
class UNetDAE_ResNet34(nn.Module):
    """
    U-Net DAE sử dụng ResNet-34 làm encoder (backbone).

    Kiến trúc:
        - Encoder: ResNet-34 pretrained trên ImageNet
        - Decoder: U-Net decoder với skip connections
        - Đầu vào: concat(RGB[3], noisy_onehot[8]) = 11 kênh
        - Đầu ra: logits [B, 8, H, W] (chưa qua softmax)

    Ưu điểm: Encoder mạnh nhờ pretrained, phù hợp khi có ít dữ liệu
    Nhược điểm: Nặng (~24M params), chậm hơn LightweightDAE

    Ví dụ:
        >>> model = UNetDAE_ResNet34(in_channels=11, num_classes=8)
        >>> x = torch.randn(4, 11, 256, 256)   # batch=4, 11 kênh, 256x256
        >>> out = model(x)
        >>> print(out.shape)                    # → torch.Size([4, 8, 256, 256])
    """
    def __init__(self, in_channels=11, num_classes=NUM_CLASSES):
        """
        Args:
            in_channels (int): Số kênh đầu vào. Mặc định 11 = 3(RGB) + 8(one-hot label)
            num_classes (int): Số lớp phân đoạn đầu ra. Mặc định 8
        """
        super().__init__()
        # Sử dụng thư viện segmentation_models_pytorch để tạo U-Net
        # encoder_weights='imagenet': dùng trọng số pretrained từ ImageNet
        # activation=None: không thêm hàm kích hoạt ở output (ta tự xử lý bên ngoài)
        self.model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )
    
    def forward(self, x):
        """
        Lan truyền thuận (forward pass).

        Args:
            x (Tensor): Đầu vào [B, 11, H, W] - ảnh RGB + nhãn nhiễu one-hot

        Returns:
            Tensor: Logits [B, 8, H, W] - điểm số cho mỗi lớp tại mỗi pixel
        """
        return self.model(x)


# =============================================================================
# MODEL 2: UNetDAE_EffNet
# =============================================================================
class UNetDAE_EffNet(nn.Module):
    """
    U-Net DAE sử dụng EfficientNet-B4 làm encoder.

    Tương tự UNetDAE_ResNet34 nhưng dùng EfficientNet-B4 thay vì ResNet-34.
    EfficientNet có kiến trúc tối ưu hơn (compound scaling), thường cho
    kết quả tốt hơn ResNet ở cùng mức tham số.

    Ví dụ:
        >>> model = UNetDAE_EffNet(in_channels=11, num_classes=8)
        >>> x = torch.randn(2, 11, 512, 512)
        >>> out = model(x)
        >>> print(out.shape)  # → torch.Size([2, 8, 512, 512])
    """
    def __init__(self, in_channels=11, num_classes=NUM_CLASSES):
        super().__init__()
        self.model = smp.Unet(
            encoder_name='efficientnet-b4',
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )
    
    def forward(self, x):
        """
        Args:
            x (Tensor): [B, 11, H, W]
        Returns:
            Tensor: [B, 8, H, W] logits
        """
        return self.model(x)


# =============================================================================
# MODEL 3: ConditionalDAE
# =============================================================================
class ConditionalDAE(nn.Module):
    """
    Conditional DAE: dùng 2 encoder riêng biệt cho RGB và label.

    Kiến trúc:
        ┌─────────────┐     ┌──────────────┐
        │  RGB (3ch)   │     │ Label (8ch)  │
        │  ResNet-34   │     │ Custom CNN   │
        │   Encoder    │     │  Encoder     │
        └──────┬───────┘     └──────┬───────┘
               │ [B,512,H/32,W/32] │
               └────────┬──────────┘
                   Concatenate
                   [B,1024,H/32,W/32]
                        │
                 Channel Attention
                   + Fusion Conv
                        │
                    U-Net Decoder
                        │
                   [B, 8, H, W]

    Ưu điểm: Xử lý riêng thông tin RGB và label → fusion thông minh hơn
    Nhược điểm: Phức tạp hơn, cần nhiều bộ nhớ hơn

    Ví dụ:
        >>> model = ConditionalDAE(num_classes=8)
        >>> # Lưu ý: input vẫn là 11 kênh (3 RGB + 8 label), model tự tách
        >>> x = torch.randn(2, 11, 256, 256)
        >>> out = model(x)
        >>> print(out.shape)  # → torch.Size([2, 8, 256, 256])
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        
        # ---- RGB Encoder ----
        # Dùng ResNet-34 pretrained để trích xuất đặc trưng không gian/ngữ nghĩa từ ảnh RGB
        self.rgb_encoder = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_classes,
            activation=None,
        )
        # Chỉ lấy phần encoder (bỏ decoder của smp.Unet)
        self.rgb_enc = self.rgb_encoder.encoder
        
        # ---- Label Encoder ----
        # Encoder tùy chỉnh cho nhãn one-hot (8 kênh đầu vào)
        # Giảm dần kích thước không gian: /2 → /4 → /8 → /16 → /32
        self.label_enc = nn.Sequential(
            ConvBlock(num_classes, 64, 3, stride=2),   # H/2,  W/2
            ConvBlock(64, 128, 3, stride=2),            # H/4,  W/4
            ConvBlock(128, 256, 3, stride=2),           # H/8,  W/8
            ConvBlock(256, 256, 3, stride=2),           # H/16, W/16
            ConvBlock(256, 512, 3, stride=2),           # H/32, W/32
        )
        
        # ---- Fusion Module ----
        # Kết hợp đặc trưng RGB và Label tại bottleneck
        # Channel Attention: học trọng số cho từng kênh (kênh nào quan trọng hơn)
        self.fusion = nn.Sequential(
            nn.Conv2d(512 + 512, 512, 1),       # Giảm 1024 → 512 kênh bằng conv 1x1
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),             # Global Average Pooling → [B, 512, 1, 1]
        )
        # Conv để xử lý đặc trưng fused trước khi nhân với attention
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(512 + 512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # ---- Decoder ----
        # Tăng dần kích thước không gian: /32 → /16 → /8 → /4 → /2 → /1
        self.decoder = nn.ModuleList([
            UpBlock(512, 256),   # /32 → /16
            UpBlock(256, 128),   # /16 → /8
            UpBlock(128, 64),    # /8  → /4
            UpBlock(64, 32),     # /4  → /2
            UpBlock(32, 16),     # /2  → /1 (kích thước gốc)
        ])
        
        # Lớp output: chuyển 16 kênh → num_classes (8) bằng conv 1x1
        self.head = nn.Conv2d(16, num_classes, 1)
    
    def forward(self, x):
        """
        Lan truyền thuận với fusion bằng channel attention.

        Luồng xử lý:
            1. Tách input thành RGB (3ch) và noisy_label (8ch)
            2. Encode RGB → rgb_bottleneck [B, 512, H/32, W/32]
            3. Encode label → label_feat [B, 512, H/32, W/32]
            4. Concat → [B, 1024, H/32, W/32]
            5. Channel attention: học trọng số kênh
            6. Decode → [B, 8, H, W]

        Args:
            x (Tensor): [B, 11, H, W] - 3 kênh RGB + 8 kênh nhãn nhiễu one-hot

        Returns:
            Tensor: [B, 8, H, W] logits
        """
        # Bước 1: Tách input
        rgb = x[:, :3]          # [B, 3, H, W]   - ảnh RGB gốc
        noisy_label = x[:, 3:]  # [B, 8, H, W]   - nhãn nhiễu dạng one-hot
        
        # Bước 2: Trích xuất đặc trưng RGB qua ResNet-34
        rgb_features = self.rgb_enc(rgb)
        rgb_bottleneck = rgb_features[-1]  # Lấy feature map cuối cùng [B, 512, H/32, W/32]
        
        # Bước 3: Trích xuất đặc trưng Label
        label_feat = noisy_label
        for layer in self.label_enc:
            label_feat = layer(label_feat)
        # label_feat: [B, 512, H/32, W/32]
        
        # Bước 4: Nối (concatenate) 2 nhánh đặc trưng
        fused = torch.cat([rgb_bottleneck, label_feat], dim=1)  # [B, 1024, H/32, W/32]
        
        # Bước 5: Channel Attention
        # attn: trọng số cho từng kênh, học được qua global avg pooling + sigmoid
        attn = self.fusion(fused)           # [B, 512, 1, 1]
        fused = self.fusion_conv(fused)     # [B, 512, H/32, W/32]
        fused = fused * torch.sigmoid(attn) # Nhân element-wise: kênh quan trọng → giá trị lớn
        
        # Bước 6: Decode (upsampling) về kích thước gốc
        out = fused
        for dec in self.decoder:
            out = dec(out)
        
        return self.head(out)  # [B, 8, H, W]


# =============================================================================
# MODEL 4: LightweightDAE
# =============================================================================
class LightweightDAE(nn.Module):
    """
    DAE nhẹ (~2-3M params) với kiến trúc encoder-decoder + skip connections.

    Kiến trúc:
        Encoder                      Decoder
        ┌──────────────┐            ┌──────────────┐
        │ enc1: 11→64  │──(skip)──→│ dec1: 128→64 │
        │ enc2: 64→128 │──(skip)──→│ dec2: 256→64 │
        │ enc3: 128→256│──(skip)──→│ dec3: 512→128│
        │ enc4: 256→256│──(skip)──→│ dec4: 512→256│
        │ enc5: 256→512│──────────→│ dec5: 512→256│
        └──────────────┘            └──────────────┘

    Skip connections: nối output encoder với input decoder ở cùng resolution,
    giúp giữ lại thông tin chi tiết (spatial detail) khi upsampling.

    Ưu điểm: Nhẹ, nhanh, hiệu quả tốt nhất trong benchmark (~12.8M params thực tế)
    Nhược điểm: Không dùng pretrained weights

    Ví dụ:
        >>> model = LightweightDAE(in_channels=11, num_classes=8)
        >>> x = torch.randn(8, 11, 256, 256)   # batch lớn hơn nhờ model nhẹ
        >>> out = model(x)
        >>> print(out.shape)                    # → torch.Size([8, 8, 256, 256])
        >>> pred_labels = out.argmax(dim=1)     # → [8, 256, 256] nhãn dự đoán
    """
    def __init__(self, in_channels=11, num_classes=NUM_CLASSES):
        """
        Args:
            in_channels (int): Số kênh đầu vào (mặc định 11 = RGB + one-hot label)
            num_classes (int): Số lớp output (mặc định 8)
        """
        super().__init__()
        
        # ---- Encoder: giảm dần spatial resolution ----
        self.enc1 = ConvBlock(in_channels, 64, 3, stride=1)   # Giữ nguyên kích thước
        self.enc2 = ConvBlock(64, 128, 3, stride=2)            # Giảm /2
        self.enc3 = ConvBlock(128, 256, 3, stride=2)           # Giảm /4
        self.enc4 = ConvBlock(256, 256, 3, stride=2)           # Giảm /8
        self.enc5 = ConvBlock(256, 512, 3, stride=2)           # Giảm /16
        
        # ---- Decoder: tăng dần spatial resolution + skip connections ----
        # Mỗi decoder nhận output từ decoder trước + skip từ encoder cùng level
        self.dec5 = UpBlock(512, 256)                          # /16 → /8
        self.dec4 = UpBlock(256 + 256, 256)   # 256(dec5) + 256(enc4) = 512 input
        self.dec3 = UpBlock(256 + 256, 128)   # 256(dec4) + 256(enc3) = 512 input
        self.dec2 = UpBlock(128 + 128, 64)    # 128(dec3) + 128(enc2) = 256 input
        self.dec1 = nn.Sequential(            # Lớp cuối không upsample, chỉ refine
            nn.Conv2d(64 + 64, 64, 3, padding=1),   # 64(dec2) + 64(enc1) = 128 input
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Lớp output: 64 kênh → 8 lớp (conv 1x1, chỉ thay đổi số kênh)
        self.head = nn.Conv2d(64, num_classes, 1)
    
    def forward(self, x):
        """
        Lan truyền thuận với skip connections kiểu U-Net.

        Luồng dữ liệu:
            Input [B,11,H,W]
              → enc1 [B,64,H,W]
              → enc2 [B,128,H/2,W/2]
              → enc3 [B,256,H/4,W/4]
              → enc4 [B,256,H/8,W/8]
              → enc5 [B,512,H/16,W/16]     (bottleneck)
              → dec5 [B,256,H/8,W/8]       + skip e4
              → dec4 [B,256,H/4,W/4]       + skip e3
              → dec3 [B,128,H/2,W/2]       + skip e2
              → dec2 [B,64,H,W]            + skip e1
              → dec1 [B,64,H,W]
              → head [B,8,H,W]             (output logits)

        Args:
            x (Tensor): [B, 11, H, W]

        Returns:
            Tensor: [B, 8, H, W] logits
        """
        # Encoder: trích xuất đặc trưng ở nhiều mức phân giải
        e1 = self.enc1(x)      # [B, 64, H, W]
        e2 = self.enc2(e1)     # [B, 128, H/2, W/2]
        e3 = self.enc3(e2)     # [B, 256, H/4, W/4]
        e4 = self.enc4(e3)     # [B, 256, H/8, W/8]
        e5 = self.enc5(e4)     # [B, 512, H/16, W/16]  ← bottleneck
        
        # Decoder: upsample + skip connections
        # torch.cat nối feature maps từ encoder (skip) với decoder
        d5 = self.dec5(e5)                              # [B, 256, H/8, W/8]
        d4 = self.dec4(torch.cat([d5, e4], dim=1))      # [B, 256, H/4, W/4]
        d3 = self.dec3(torch.cat([d4, e3], dim=1))      # [B, 128, H/2, W/2]
        d2 = self.dec2(torch.cat([d3, e2], dim=1))      # [B, 64, H, W]
        d1 = self.dec1(torch.cat([d2, e1], dim=1))      # [B, 64, H, W]
        
        return self.head(d1)   # [B, 8, H, W]  ← logits cho 8 lớp


# =============================================================================
# CÁC BUILDING BLOCKS (Khối xây dựng cơ bản)
# =============================================================================

class ConvBlock(nn.Module):
    """
    Khối convolution cơ bản: Conv → BN → ReLU → Conv → BN → ReLU.

    Sử dụng 2 lớp conv liên tiếp để tăng khả năng học đặc trưng.
    Có thể giảm kích thước spatial bằng stride > 1.

    Args:
        in_ch  (int): Số kênh đầu vào
        out_ch (int): Số kênh đầu ra
        kernel (int): Kích thước kernel (mặc định 3x3)
        stride (int): Bước nhảy (stride=2 → giảm kích thước /2)

    Ví dụ:
        >>> block = ConvBlock(64, 128, kernel=3, stride=2)
        >>> x = torch.randn(2, 64, 256, 256)
        >>> out = block(x)
        >>> print(out.shape)  # → torch.Size([2, 128, 128, 128])  (giảm /2)
    """
    def __init__(self, in_ch, out_ch, kernel=3, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            # Conv thứ 1: thay đổi số kênh + có thể giảm kích thước (stride)
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel//2),
            nn.BatchNorm2d(out_ch),     # Chuẩn hóa batch → ổn định huấn luyện
            nn.ReLU(inplace=True),      # Hàm kích hoạt phi tuyến
            # Conv thứ 2: giữ nguyên kích thước, refine đặc trưng
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        """
        Args:
            x (Tensor): [B, in_ch, H, W]
        Returns:
            Tensor: [B, out_ch, H/stride, W/stride]
        """
        return self.conv(x)


class UpBlock(nn.Module):
    """
    Khối upsampling: ConvTranspose (tăng gấp đôi kích thước) → BN → ReLU → Conv → BN → ReLU.

    Dùng ConvTranspose2d (deconvolution) để tăng kích thước spatial lên gấp 2,
    sau đó dùng thêm 1 lớp Conv để refine.

    Args:
        in_ch  (int): Số kênh đầu vào
        out_ch (int): Số kênh đầu ra

    Ví dụ:
        >>> up = UpBlock(512, 256)
        >>> x = torch.randn(2, 512, 8, 8)
        >>> out = up(x)
        >>> print(out.shape)  # → torch.Size([2, 256, 16, 16])  (tăng x2)
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            # ConvTranspose2d: "ngược" của Conv2d, tăng kích thước spatial x2
            # kernel=4, stride=2, padding=1 → output = input_size * 2
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # Conv thêm để refine kết quả sau upsampling
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        """
        Args:
            x (Tensor): [B, in_ch, H, W]
        Returns:
            Tensor: [B, out_ch, 2*H, 2*W]  (kích thước tăng gấp đôi)
        """
        return self.up(x)


# =============================================================================
# HÀM LOSS (Loss Functions)
# =============================================================================

class DAELoss(nn.Module):
    """
    Hàm loss kết hợp cho DAE: CrossEntropy + Dice + Boundary (tùy chọn).

    Công thức:
        L = ce_weight * CE + dice_weight * Dice + boundary_weight * Boundary

    Mỗi thành phần:
        - CrossEntropy: loss phân loại chuẩn, tốt cho pixel-wise classification
        - Dice Loss: đo overlap giữa prediction và ground truth, tốt cho class imbalance
        - Boundary Loss: tăng trọng số loss tại vùng biên → cải thiện segmentation boundary

    Args:
        ce_weight       (float): Trọng số CrossEntropy (mặc định 1.0)
        dice_weight     (float): Trọng số Dice loss (mặc định 1.0)
        boundary_weight (float): Trọng số Boundary loss (mặc định 0.5, =0 để tắt)
        num_classes     (int)  : Số lớp phân đoạn

    Ví dụ:
        >>> criterion = DAELoss(ce_weight=1.0, dice_weight=1.0, boundary_weight=0.5)
        >>> pred = torch.randn(4, 8, 256, 256)        # logits từ model
        >>> target = torch.randint(0, 8, (4, 256, 256))  # nhãn ground truth
        >>> loss = criterion(pred, target)
        >>> loss.backward()  # backpropagation
    """
    def __init__(self, ce_weight=1.0, dice_weight=1.0, boundary_weight=0.5,
                 num_classes=NUM_CLASSES):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()  # Loss phân loại chuẩn
    
    def forward(self, pred, target):
        """
        Tính tổng loss.

        Args:
            pred   (Tensor): Logits từ model [B, C, H, W] (C = num_classes)
            target (Tensor): Nhãn ground truth [B, H, W] (giá trị 0..C-1)

        Returns:
            Tensor: Scalar loss value
        """
        ce_loss = self.ce(pred, target)
        dice_loss = self._dice_loss(pred, target)
        loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        
        # Boundary loss chỉ được thêm nếu boundary_weight > 0
        if self.boundary_weight > 0:
            boundary_loss = self._boundary_loss(pred, target)
            loss += self.boundary_weight * boundary_loss
        
        return loss
    
    def _dice_loss(self, pred, target):
        """
        Dice Loss: đo mức độ overlap giữa prediction và ground truth.

        Công thức Dice cho mỗi class c:
            Dice_c = 2 * |P_c ∩ T_c| / (|P_c| + |T_c|)
            Dice Loss = 1 - mean(Dice_c)

        Giá trị Dice nằm trong [0, 1]:
            - 1 = overlap hoàn hảo → loss = 0
            - 0 = không overlap → loss = 1

        Args:
            pred   (Tensor): [B, C, H, W] logits
            target (Tensor): [B, H, W] nhãn (0..C-1)

        Returns:
            Tensor: Dice loss (scalar)
        """
        # Chuyển logits → xác suất bằng softmax
        pred_soft = F.softmax(pred, dim=1)
        # Chuyển target từ index → one-hot: [B, H, W] → [B, H, W, C] → [B, C, H, W]
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        
        # Tính intersection (phần giao) và union (tổng) cho mỗi class
        intersection = (pred_soft * target_onehot).sum(dim=(2, 3))  # [B, C]
        union = pred_soft.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))  # [B, C]
        # Dice coefficient + smoothing (1e-6 tránh chia cho 0)
        dice = (2 * intersection + 1e-6) / (union + 1e-6)  # [B, C]
        return 1 - dice.mean()  # 1 - Dice trung bình = Dice Loss
    
    def _boundary_loss(self, pred, target):
        """
        Boundary Loss: tập trung phạt thêm tại vùng biên giữa các đối tượng.

        Cách hoạt động:
            1. Dùng bộ lọc Sobel (phát hiện cạnh) trên target để tìm vùng biên
            2. Tính CE loss cho từng pixel
            3. Chỉ lấy CE loss tại vùng biên (nhân với boundary mask)

        Mục đích: Biên giữa các class thường bị nhiễu nhiều nhất, cần chú ý hơn.

        Args:
            pred   (Tensor): [B, C, H, W] logits
            target (Tensor): [B, H, W] nhãn

        Returns:
            Tensor: Boundary loss (scalar)
        """
        # Tạo bộ lọc Sobel 3x3 để phát hiện cạnh theo 2 hướng x và y
        target_float = target.float().unsqueeze(1)  # [B, 1, H, W]
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=target.device).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)  # Sobel theo hướng y
        
        # Áp dụng Sobel filter → gradient theo x và y
        gx = F.conv2d(target_float, sobel_x, padding=1)
        gy = F.conv2d(target_float, sobel_y, padding=1)
        # Boundary mask: pixel nào có gradient > 0 → là biên
        boundary = (gx.abs() + gy.abs() > 0).float().squeeze(1)  # [B, H, W]
        
        # Tính CrossEntropy cho từng pixel (không reduce)
        ce_per_pixel = F.cross_entropy(pred, target, reduction='none')  # [B, H, W]
        # Chỉ lấy loss tại vùng biên, chia cho tổng số pixel biên
        boundary_ce = (ce_per_pixel * boundary).sum() / (boundary.sum() + 1e-6)
        
        return boundary_ce


# =============================================================================
# FACTORY FUNCTION & UTILITIES
# =============================================================================

def build_model(model_name: str, **kwargs) -> nn.Module:
    """
    Factory function: tạo model DAE theo tên.

    Hàm này cho phép tạo model một cách linh hoạt từ config/command line
    mà không cần import trực tiếp từng class.

    Args:
        model_name (str): Tên model, một trong:
            - 'unet_resnet34' : UNetDAE_ResNet34
            - 'unet_effnet'   : UNetDAE_EffNet
            - 'conditional'   : ConditionalDAE
            - 'lightweight'   : LightweightDAE
        **kwargs: Các tham số bổ sung truyền vào constructor của model
                  (vd: in_channels=11, num_classes=8)

    Returns:
        nn.Module: Model DAE đã khởi tạo

    Raises:
        ValueError: Nếu model_name không hợp lệ

    Ví dụ:
        >>> model = build_model('lightweight', in_channels=11, num_classes=8)
        >>> model = build_model('unet_resnet34')  # dùng giá trị mặc định
        >>> # Dùng trong config:
        >>> # config.yaml: model_name: 'lightweight'
        >>> model = build_model(config['model_name'])
    """
    models = {
        'unet_resnet34': UNetDAE_ResNet34,
        'unet_effnet': UNetDAE_EffNet,
        'conditional': ConditionalDAE,
        'lightweight': LightweightDAE,
    }
    if model_name not in models:
        raise ValueError(f'Unknown model: {model_name}. Choose from {list(models.keys())}')
    return models[model_name](**kwargs)


def count_params(model):
    """
    Đếm tổng số tham số và số tham số trainable của model.

    Args:
        model (nn.Module): PyTorch model

    Returns:
        tuple: (total_params, trainable_params)

    Ví dụ:
        >>> model = build_model('lightweight')
        >>> total, trainable = count_params(model)
        >>> print(f"Tổng: {total/1e6:.1f}M | Trainable: {trainable/1e6:.1f}M")
        # Output: Tổng: 12.8M | Trainable: 12.8M
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# =============================================================================
# KHỐI TEST (chạy khi: python dae_model.py)
# =============================================================================
if __name__ == '__main__':
    # Tạo input giả: batch=2, 11 kênh (3 RGB + 8 one-hot), kích thước 256x256
    x = torch.randn(2, 11, 256, 256)
    
    # Test lần lượt từng model variant
    for name in ['unet_resnet34', 'unet_effnet', 'conditional', 'lightweight']:
        try:
            model = build_model(name)
            out = model(x)
            total, trainable = count_params(model)
            print(f'{name:>20s} | output: {list(out.shape)} | params: {total/1e6:.1f}M ({trainable/1e6:.1f}M trainable)')
        except Exception as e:
            print(f'{name:>20s} | ERROR: {e}')
