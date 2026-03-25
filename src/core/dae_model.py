"""
=============================================================================
MỤC ĐÍCH CỦA FILE (File Purpose):
=============================================================================
File này định nghĩa 3 kiến trúc Denoising AutoEncoder (DAE) dùng để
"làm sạch" nhãn phân đoạn (pseudo-label) bị nhiễu trong ảnh vệ tinh.

Kiến trúc: LATER FUSION (Dual Branch) với Skip Connections
    - Nhánh 1 (RGB):   đưa ảnh RGB (3 kênh) qua Pre-trained Backbone
    - Nhánh 2 (Label): đưa nhãn one-hot (8 kênh) qua Lightweight Encoder
    - Fusion:          kết hợp features tại bottleneck bằng Channel Attention
    - Decoder:         skip connections từ CẢ 2 nhánh ở 4 tầng

    Đầu vào: rgb [B, 3, H, W] + label [B, 8, H, W]  (2 tensor riêng biệt)
    Đầu ra:  logits [B, 8, H, W] (chưa qua softmax)

3 biến thể mô hình:
    1. UNetDAE_ResNet34   - RGB: ResNet-34 pretrained | Label: Lightweight CNN
    2. UNetDAE_EffNet     - RGB: EfficientNet-B4 pretrained | Label: Lightweight CNN
    3. LightweightDAE     - RGB: Custom CNN | Label: Custom CNN (nhẹ nhất)

Ví dụ sử dụng nhanh (Quick Usage):
    >>> from dae_model import build_model, DAELoss
    >>> model = build_model('unet_resnet34')
    >>> rgb = torch.randn(2, 3, 256, 256)
    >>> label = torch.randn(2, 8, 256, 256)
    >>> logits = model(rgb, label)                # → [2, 8, 256, 256]
    >>> pred = logits.argmax(dim=1)               # → [2, 256, 256]
=============================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Lazy import: smp only needed when building pretrained encoder models
# This avoids hanging on HuggingFace hub connection during module import
smp = None
def _get_smp():
    global smp
    if smp is None:
        import segmentation_models_pytorch as _smp
        smp = _smp
    return smp

# Số lượng lớp phân đoạn (8 loại đất/vật thể trong ảnh vệ tinh)
NUM_CLASSES = 8


# =============================================================================
# CÁC BUILDING BLOCKS (Khối xây dựng cơ bản)
# =============================================================================

class ConvBlock(nn.Module):
    """
    Khối convolution cơ bản: Conv → BN → ReLU → Conv → BN → ReLU.

    Args:
        in_ch  (int): Số kênh đầu vào
        out_ch (int): Số kênh đầu ra
        kernel (int): Kích thước kernel (mặc định 3x3)
        stride (int): Bước nhảy (stride=2 → giảm kích thước /2)

    Input:  [B, in_ch,  H, W]
    Output: [B, out_ch, H/stride, W/stride]
    """
    def __init__(self, in_ch, out_ch, kernel=3, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            # [B, in_ch, H, W] → [B, out_ch, H/stride, W/stride]
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel//2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # [B, out_ch, H/stride, W/stride] → [B, out_ch, H/stride, W/stride]
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: [B, in_ch, H, W] → out: [B, out_ch, H/stride, W/stride]
        return self.conv(x)


class UpBlock(nn.Module):
    """
    Khối upsampling: ConvTranspose (tăng gấp đôi kích thước) → BN → ReLU → Conv → BN → ReLU.

    Input:  [B, in_ch,  H,   W]
    Output: [B, out_ch, 2*H, 2*W]
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            # [B, in_ch, H, W] → [B, out_ch, 2*H, 2*W]
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # [B, out_ch, 2*H, 2*W] → [B, out_ch, 2*H, 2*W]
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: [B, in_ch, H, W] → out: [B, out_ch, 2*H, 2*W]
        return self.up(x)


class LabelEncoder(nn.Module):
    """
    Lightweight encoder cho nhãn one-hot (8 kênh).
    Trả về multi-scale features tại 4 tầng + bottleneck.

    Output scales (khi stride=2 cho mỗi tầng):
        f1: [B, 64,  H/2,  W/2]
        f2: [B, 64,  H/4,  W/4]
        f3: [B, 128, H/8,  W/8]
        f4: [B, 256, H/16, W/16]
        f5: [B, 512, H/32, W/32]   ← bottleneck
    """
    def __init__(self, in_ch=NUM_CLASSES, channels=[64, 64, 128, 256, 512]):
        super().__init__()
        self.enc1 = ConvBlock(in_ch,      channels[0], 3, stride=2)  # H/2
        self.enc2 = ConvBlock(channels[0], channels[1], 3, stride=2)  # H/4
        self.enc3 = ConvBlock(channels[1], channels[2], 3, stride=2)  # H/8
        self.enc4 = ConvBlock(channels[2], channels[3], 3, stride=2)  # H/16
        self.enc5 = ConvBlock(channels[3], channels[4], 3, stride=2)  # H/32

    def forward(self, x):
        """
        Returns:
            list: [f1, f2, f3, f4, f5] multi-scale features
        """
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        f5 = self.enc5(f4)
        return [f1, f2, f3, f4, f5]


class FusionDecoder(nn.Module):
    """
    Decoder với skip connections từ CẢ 2 nhánh (RGB + Label).

    Tại mỗi tầng decoder:
        input = concat(upsampled_prev, rgb_skip_i, label_skip_i)

    Kiến trúc:
        Bottleneck  [fused_ch]           H/32
            ↓ UpBlock
        D4: concat(up + skip_H/16)       H/16
            ↓ UpBlock
        D3: concat(up + skip_H/8)        H/8
            ↓ UpBlock
        D2: concat(up + skip_H/4)        H/4
            ↓ UpBlock
        D1: concat(up + skip_H/2)        H/2
            ↓ Bilinear 2x + ConvRefine
        Out: 1x1 conv → [B, 8, H, W]    H
    """
    def __init__(self, fused_ch, rgb_channels, label_channels, num_classes=NUM_CLASSES):
        """
        Args:
            fused_ch (int): Số kênh sau fusion tại bottleneck
            rgb_channels (list): [ch_H/2, ch_H/4, ch_H/8, ch_H/16] của RGB encoder
            label_channels (list): [ch_H/2, ch_H/4, ch_H/8, ch_H/16] của Label encoder
            num_classes (int): Số lớp output
        """
        super().__init__()
        # D4: fused_ch → 256 (H/32 → H/16)
        d4_out = 256
        self.up4 = UpBlock(fused_ch, d4_out)

        # D3: concat(d4_out, rgb_H/16, label_H/16) → 128 (H/16 → H/8)
        d3_in = d4_out + rgb_channels[3] + label_channels[3]
        d3_out = 128
        self.up3 = UpBlock(d3_in, d3_out)

        # D2: concat(d3_out, rgb_H/8, label_H/8) → 64 (H/8 → H/4)
        d2_in = d3_out + rgb_channels[2] + label_channels[2]
        d2_out = 64
        self.up2 = UpBlock(d2_in, d2_out)

        # D1: concat(d2_out, rgb_H/4, label_H/4) → 64 (H/4 → H/2)
        d1_in = d2_out + rgb_channels[1] + label_channels[1]
        d1_out = 64
        self.up1 = UpBlock(d1_in, d1_out)

        # Final: concat(d1_out, rgb_H/2, label_H/2) → bilinear 2x → refine → head (H/2 → H)
        final_in = d1_out + rgb_channels[0] + label_channels[0]
        self.refine = nn.Sequential(
            nn.Conv2d(final_in, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(64, num_classes, 1)

    def forward(self, fused, rgb_skips, label_skips):
        """
        Args:
            fused: [B, fused_ch, H/32, W/32] bottleneck features
            rgb_skips: list >= 4 items, indices [0..3] = [H/2, H/4, H/8, H/16]
                       (index 4 = H/32 bottleneck, not used here)
            label_skips: list of 4 items, indices [0..3] = [H/2, H/4, H/8, H/16]

        Returns:
            logits [B, num_classes, H, W]
        """
        # D4: upsample bottleneck [B, fused_ch, H/32, W/32] → [B, 256, H/16, W/16]
        d4 = self.up4(fused)                                       # [B, 256, H/16, W/16]
        d4 = self._match_and_cat(d4, rgb_skips[3], label_skips[3]) # [B, 256+rgb_ch3+lbl_ch3, H/16, W/16]

        # D3: [B, d3_in, H/16, W/16] → [B, 128, H/8, W/8]
        d3 = self.up3(d4)                                          # [B, 128, H/8, W/8]
        d3 = self._match_and_cat(d3, rgb_skips[2], label_skips[2]) # [B, 128+rgb_ch2+lbl_ch2, H/8, W/8]

        # D2: [B, d2_in, H/8, W/8] → [B, 64, H/4, W/4]
        d2 = self.up2(d3)                                          # [B, 64, H/4, W/4]
        d2 = self._match_and_cat(d2, rgb_skips[1], label_skips[1]) # [B, 64+rgb_ch1+lbl_ch1, H/4, W/4]

        # D1: [B, d1_in, H/4, W/4] → [B, 64, H/2, W/2]
        d1 = self.up1(d2)                                          # [B, 64, H/2, W/2]
        d1 = self._match_and_cat(d1, rgb_skips[0], label_skips[0]) # [B, 64+rgb_ch0+lbl_ch0, H/2, W/2]

        # Upsample [B, final_in, H/2, W/2] → [B, final_in, H, W]
        d1 = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.refine(d1)   # [B, 64, H, W]
        return self.head(out)   # [B, num_classes, H, W]

    @staticmethod
    def _match_and_cat(x, skip_a, skip_b):
        """Resize skips nếu cần rồi concat."""
        if skip_a.shape[2:] != x.shape[2:]:
            skip_a = F.interpolate(skip_a, size=x.shape[2:], mode='bilinear', align_corners=False)
        if skip_b.shape[2:] != x.shape[2:]:
            skip_b = F.interpolate(skip_b, size=x.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([x, skip_a, skip_b], dim=1)


# =============================================================================
# MODEL 1: UNetDAE_ResNet34 (Later Fusion)
# =============================================================================
class UNetDAE_ResNet34(nn.Module):
    """
    U-Net DAE với Later Fusion:
        - Nhánh RGB:   ResNet-34 pretrained (lấy 5 tầng features)
        - Nhánh Label: LabelEncoder lightweight
        - Decoder:     skip connections từ cả 2 nhánh ở 4 tầng

    Ví dụ:
        >>> model = UNetDAE_ResNet34()
        >>> rgb = torch.randn(2, 3, 256, 256)
        >>> label = torch.randn(2, 8, 256, 256)
        >>> out = model(rgb, label)
        >>> print(out.shape)  # → torch.Size([2, 8, 256, 256])
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        # RGB encoder: ResNet-34 pretrained
        # smp encoder trả về list 6 features: [conv1_out, layer1, layer2, layer3, layer4]
        # channels: [64, 64, 128, 256, 512] cho resnet34
        _tmp = _get_smp().Unet(encoder_name='resnet34', encoder_weights='imagenet',
                         in_channels=3, classes=num_classes, activation=None)
        self.rgb_encoder = _tmp.encoder
        del _tmp

        # Dynamically read encoder output channels (version-safe)
        enc_out = self.rgb_encoder.out_channels  # e.g. (3, 64, 64, 128, 256, 512)
        rgb_chs = list(enc_out[1:])  # [ch_H/2, ch_H/4, ch_H/8, ch_H/16, ch_H/32]

        # Label encoder: lightweight CNN
        label_chs = [64, 64, 128, 256, 512]
        self.label_encoder = LabelEncoder(in_ch=num_classes, channels=label_chs)

        # Bottleneck fusion: concat RGB + Label → channel attention → fused
        fused_ch = rgb_chs[4] + label_chs[4]  # 1024
        self.fusion_attn = nn.Sequential(
            nn.Conv2d(fused_ch, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fused_ch, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Decoder with dual skip connections
        #  rgb_channels for skips: [H/2=64, H/4=64, H/8=128, H/16=256]
        #  label_channels for skips: [H/2=64, H/4=64, H/8=128, H/16=256]
        self.decoder = FusionDecoder(
            fused_ch=512,
            rgb_channels=rgb_chs[:4],    # exclude bottleneck
            label_channels=label_chs[:4],
            num_classes=num_classes,
        )

    def forward(self, rgb, label):
        """
        Args:
            rgb (Tensor): [B, 3, H, W] ảnh RGB
            label (Tensor): [B, 8, H, W] nhãn nhiễu one-hot

        Returns:
            Tensor: [B, 8, H, W] logits
        """
        # RGB encoder → 6 features (index 0 = identity, 1..5 = actual features)
        rgb_feats = self.rgb_encoder(rgb)  # list of 6 tensors
        rgb_bottleneck = rgb_feats[-1]     # [B, 512, H/32, W/32]

        # Label encoder → 5 features
        label_feats = self.label_encoder(label)  # [f1..f5]
        label_bottleneck = label_feats[-1]       # [B, 512, H/32, W/32]

        # Bottleneck fusion with channel attention
        fused = torch.cat([rgb_bottleneck, label_bottleneck], dim=1)
        attn = self.fusion_attn(fused)       # [B, 512, 1, 1]
        fused = self.fusion_conv(fused)      # [B, 512, H/32, W/32]
        fused = fused * torch.sigmoid(attn)  # channel attention

        # Decode with skip connections from both branches
        # rgb_skips:   features[1..4] = [H/2, H/4, H/8, H/16] + features[4] for deepest
        # label_skips: label_feats[0..3] = [H/2, H/4, H/8, H/16]
        return self.decoder(fused, rgb_feats[1:], label_feats[:4])


# =============================================================================
# MODEL 2: UNetDAE_EffNet (Later Fusion)
# =============================================================================
class UNetDAE_EffNet(nn.Module):
    """
    U-Net DAE với Later Fusion sử dụng EfficientNet-B4 cho nhánh RGB.

    Ví dụ:
        >>> model = UNetDAE_EffNet()
        >>> rgb = torch.randn(2, 3, 256, 256)
        >>> label = torch.randn(2, 8, 256, 256)
        >>> out = model(rgb, label)
        >>> print(out.shape)  # → torch.Size([2, 8, 256, 256])
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        _tmp = _get_smp().Unet(encoder_name='efficientnet-b4', encoder_weights='imagenet',
                         in_channels=3, classes=num_classes, activation=None)
        self.rgb_encoder = _tmp.encoder
        del _tmp

        # Dynamically read encoder output channels from smp (version-safe)
        # out_channels is a tuple like (3, 48, 32, 56, 160, 448) for efficientnet-b4
        # Index 0 = identity (input channels), indices 1..5 = actual feature channels
        enc_out = self.rgb_encoder.out_channels  # tuple of 6 values
        rgb_chs = list(enc_out[1:])  # [ch_H/2, ch_H/4, ch_H/8, ch_H/16, ch_H/32]

        label_chs = [64, 64, 128, 256, 512]
        self.label_encoder = LabelEncoder(in_ch=num_classes, channels=label_chs)

        fused_ch = rgb_chs[4] + label_chs[4]
        self.fusion_attn = nn.Sequential(
            nn.Conv2d(fused_ch, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fused_ch, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.decoder = FusionDecoder(
            fused_ch=512,
            rgb_channels=rgb_chs[:4],
            label_channels=label_chs[:4],
            num_classes=num_classes,
        )

    def forward(self, rgb, label):
        """
        Args:
            rgb (Tensor): [B, 3, H, W] ảnh RGB
            label (Tensor): [B, 8, H, W] nhãn nhiễu one-hot

        Returns:
            Tensor: [B, 8, H, W] logits
        """
        # RGB encoder → 6 features (index 0 = identity, 1..5 = actual features)
        rgb_feats = self.rgb_encoder(rgb)       # list of 6 tensors
        rgb_bottleneck = rgb_feats[-1]           # [B, rgb_chs[4], H/32, W/32]

        # Label encoder → 5 features
        label_feats = self.label_encoder(label)  # [f1..f5]
        label_bottleneck = label_feats[-1]       # [B, 512, H/32, W/32]

        # Bottleneck fusion with channel attention
        fused = torch.cat([rgb_bottleneck, label_bottleneck], dim=1)  # [B, rgb_chs[4]+512, H/32, W/32]
        attn = self.fusion_attn(fused)           # [B, 512, 1, 1]
        fused = self.fusion_conv(fused)          # [B, 512, H/32, W/32]
        fused = fused * torch.sigmoid(attn)      # channel attention → [B, 512, H/32, W/32]

        # Decode with skip connections from both branches
        return self.decoder(fused, rgb_feats[1:], label_feats[:4])  # [B, 8, H, W]


# =============================================================================
# MODEL 3: LightweightDAE (Later Fusion, no pretrained)
# =============================================================================
class LightweightDAE(nn.Module):
    """
    DAE nhẹ với Later Fusion — cả 2 nhánh đều dùng custom CNN.

    Ví dụ:
        >>> model = LightweightDAE()
        >>> rgb = torch.randn(2, 3, 256, 256)
        >>> label = torch.randn(2, 8, 256, 256)
        >>> out = model(rgb, label)
        >>> print(out.shape)  # → torch.Size([2, 8, 256, 256])
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        # ---- RGB Encoder (Custom CNN, không pretrained) ----
        rgb_chs = [64, 64, 128, 256, 512]
        self.rgb_enc1 = ConvBlock(3,          rgb_chs[0], 3, stride=2)  # H/2
        self.rgb_enc2 = ConvBlock(rgb_chs[0], rgb_chs[1], 3, stride=2)  # H/4
        self.rgb_enc3 = ConvBlock(rgb_chs[1], rgb_chs[2], 3, stride=2)  # H/8
        self.rgb_enc4 = ConvBlock(rgb_chs[2], rgb_chs[3], 3, stride=2)  # H/16
        self.rgb_enc5 = ConvBlock(rgb_chs[3], rgb_chs[4], 3, stride=2)  # H/32

        # ---- Label Encoder (Lightweight CNN) ----
        label_chs = [32, 32, 64, 128, 256]  # nhẹ hơn
        self.label_encoder = LabelEncoder(in_ch=num_classes, channels=label_chs)

        # ---- Fusion tại bottleneck ----
        fused_ch = rgb_chs[4] + label_chs[4]  # 768
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fused_ch, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # ---- Decoder ----
        self.decoder = FusionDecoder(
            fused_ch=512,
            rgb_channels=rgb_chs[:4],     # [64, 64, 128, 256]
            label_channels=label_chs[:4], # [32, 32, 64, 128]
            num_classes=num_classes,
        )

    def forward(self, rgb, label):
        """
        Args:
            rgb (Tensor): [B, 3, H, W]
            label (Tensor): [B, 8, H, W]

        Returns:
            Tensor: [B, 8, H, W] logits
        """
        # Encode RGB
        r1 = self.rgb_enc1(rgb)   # [B, 64, H/2, W/2]
        r2 = self.rgb_enc2(r1)    # [B, 64, H/4, W/4]
        r3 = self.rgb_enc3(r2)    # [B, 128, H/8, W/8]
        r4 = self.rgb_enc4(r3)    # [B, 256, H/16, W/16]
        r5 = self.rgb_enc5(r4)    # [B, 512, H/32, W/32]
        rgb_skips = [r1, r2, r3, r4, r5]  # 5 features

        # Encode Label
        label_feats = self.label_encoder(label)  # [f1..f5]
        label_bottleneck = label_feats[-1]

        # Fusion at bottleneck, KHÔNG CÓ ATTENTION
        fused = torch.cat([r5, label_bottleneck], dim=1)
        fused = self.fusion_conv(fused)  # [B, 512, H/32, W/32]

        # Decode with dual skip connections
        return self.decoder(fused, rgb_skips, label_feats[:4])


# =============================================================================
# HÀM LOSS (Loss Functions)
# =============================================================================

class DAELoss(nn.Module):
    """
    Hàm loss kết hợp cho DAE: CrossEntropy + Dice + Boundary (tùy chọn).

    Công thức:
        L = ce_weight * CE + dice_weight * Dice + boundary_weight * Boundary
    """
    def __init__(self, ce_weight=1.0, dice_weight=1.0, boundary_weight=0.5,
                 num_classes=NUM_CLASSES):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        dice_loss = self._dice_loss(pred, target)
        loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss

        if self.boundary_weight > 0:
            boundary_loss = self._boundary_loss(pred, target)
            loss += self.boundary_weight * boundary_loss

        return loss

    def _dice_loss(self, pred, target):
        # Mask pixels background (255) để không tính Dice loss
        valid_mask = (target != 255).float().unsqueeze(1)  # [B, 1, H, W]

        pred_soft = F.softmax(pred, dim=1) * valid_mask
        target_onehot = F.one_hot(target.clamp(0, self.num_classes - 1), self.num_classes).permute(0, 3, 1, 2).float()
        target_onehot = target_onehot * valid_mask

        intersection = (pred_soft * target_onehot).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        return 1 - dice.mean()

    def _boundary_loss(self, pred, target):
        # Mask pixels background (255) để không tính boundary loss
        valid_mask = (target != 255).float()

        # Clamp target để tránh IndexError, sau đó mask bằng valid_mask
        target_clamped = target.clamp(0, self.num_classes - 1)

        target_float = target_clamped.float().unsqueeze(1)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32, device=target.device).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)
        gx = F.conv2d(target_float, sobel_x, padding=1)
        gy = F.conv2d(target_float, sobel_y, padding=1)
        boundary = (gx.abs() + gy.abs() > 0).float().squeeze(1) * valid_mask

        # Tính CE per pixel, mask background
        ce_per_pixel = F.cross_entropy(pred, target_clamped, reduction='none') * valid_mask
        boundary_ce = (ce_per_pixel * boundary).sum() / (boundary.sum() + 1e-6)
        return boundary_ce


# =============================================================================
# FACTORY FUNCTION & UTILITIES
# =============================================================================

def build_model(model_name: str, **kwargs) -> nn.Module:
    """
    Factory function: tạo model DAE theo tên.

    Args:
        model_name (str): Tên model, một trong:
            - 'unet_resnet34' : UNetDAE_ResNet34
            - 'unet_effnet'   : UNetDAE_EffNet
            - 'lightweight'   : LightweightDAE
        **kwargs: Tham số bổ sung (vd: num_classes=8)

    Returns:
        nn.Module: Model DAE đã khởi tạo (nhận 2 input: rgb, label)
    """
    models = {
        'unet_resnet34': UNetDAE_ResNet34,
        'unet_effnet': UNetDAE_EffNet,
        'lightweight': LightweightDAE,
    }
    if model_name not in models:
        raise ValueError(f'Unknown model: {model_name}. Choose from {list(models.keys())}')

    # Filter kwargs: chỉ truyền các tham số mà model constructor chấp nhận
    import inspect
    sig = inspect.signature(models[model_name].__init__)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return models[model_name](**valid_kwargs)


def count_params(model):
    """Đếm tổng số tham số và số tham số trainable."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# =============================================================================
# KHỐI TEST (chạy khi: python dae_model.py)
# =============================================================================
if __name__ == '__main__':
    # Tạo input giả: batch=2, 256x256
    rgb = torch.randn(2, 3, 256, 256)
    label = torch.randn(2, 8, 256, 256)

    # Test lần lượt từng model variant
    for name in ['unet_resnet34', 'unet_effnet', 'lightweight']:
        try:
            model = build_model(name)
            out = model(rgb, label)
            total, trainable = count_params(model)
            print(f'{name:>20s} | output: {list(out.shape)} | '
                  f'params: {total/1e6:.1f}M ({trainable/1e6:.1f}M trainable)')
        except Exception as e:
            import traceback
            print(f'{name:>20s} | ERROR: {e}')
            traceback.print_exc()
