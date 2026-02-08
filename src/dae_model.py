"""
Denoising AutoEncoder Models for Pseudo-label Refinement.

4 variants:
  1. UNetDAE_ResNet34   - U-Net with ResNet-34 encoder (11ch input)
  2. UNetDAE_EffNet     - U-Net with EfficientNet-B4 encoder
  3. ConditionalDAE     - Dual-encoder (separate RGB + label encoders)
  4. LightweightDAE     - Small custom CNN (~2-3M params)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

NUM_CLASSES = 8


class UNetDAE_ResNet34(nn.Module):
    """U-Net DAE with ResNet-34 encoder.
    Input: concat(RGB[3], noisy_onehot[8]) = 11 channels
    Output: clean label logits [B, 8, H, W]
    """
    def __init__(self, in_channels=11, num_classes=NUM_CLASSES):
        super().__init__()
        self.model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )
    
    def forward(self, x):
        return self.model(x)


class UNetDAE_EffNet(nn.Module):
    """U-Net DAE with EfficientNet-B4 encoder."""
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
        return self.model(x)


class ConditionalDAE(nn.Module):
    """Conditional DAE: separate RGB encoder and label encoder,
    fused at bottleneck via concatenation + attention.
    
    RGB encoder extracts spatial/semantic features.
    Label encoder processes noisy label.
    Decoder fuses both to produce clean label.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        
        # RGB encoder (pretrained ResNet-34)
        self.rgb_encoder = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_classes,
            activation=None,
        )
        # Extract just the encoder part
        self.rgb_enc = self.rgb_encoder.encoder
        
        # Label encoder (custom, 8ch input)
        self.label_enc = nn.Sequential(
            ConvBlock(num_classes, 64, 3, stride=2),   # /2
            ConvBlock(64, 128, 3, stride=2),            # /4
            ConvBlock(128, 256, 3, stride=2),           # /8
            ConvBlock(256, 256, 3, stride=2),           # /16
            ConvBlock(256, 512, 3, stride=2),           # /32
        )
        
        # Fusion at bottleneck
        self.fusion = nn.Sequential(
            nn.Conv2d(512 + 512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Channel attention
            nn.AdaptiveAvgPool2d(1),
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(512 + 512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.decoder = nn.ModuleList([
            UpBlock(512, 256),   # /16
            UpBlock(256, 128),   # /8
            UpBlock(128, 64),    # /4
            UpBlock(64, 32),     # /2
            UpBlock(32, 16),     # /1
        ])
        
        self.head = nn.Conv2d(16, num_classes, 1)
    
    def forward(self, x):
        # Split input
        rgb = x[:, :3]          # [B, 3, H, W]
        noisy_label = x[:, 3:]  # [B, 8, H, W]
        
        # RGB features
        rgb_features = self.rgb_enc(rgb)
        rgb_bottleneck = rgb_features[-1]  # [B, 512, H/32, W/32]
        
        # Label features
        label_feat = noisy_label
        for layer in self.label_enc:
            label_feat = layer(label_feat)
        # label_feat: [B, 512, H/32, W/32]
        
        # Fuse
        fused = torch.cat([rgb_bottleneck, label_feat], dim=1)  # [B, 1024, H/32, W/32]
        
        # Channel attention
        attn = self.fusion(fused)  # [B, 512, 1, 1]
        fused = self.fusion_conv(fused)  # [B, 512, H/32, W/32]
        fused = fused * torch.sigmoid(attn)
        
        # Decode
        out = fused
        for dec in self.decoder:
            out = dec(out)
        
        return self.head(out)


class LightweightDAE(nn.Module):
    """Lightweight DAE with ~2-3M params.
    Simple encoder-decoder with skip connections.
    """
    def __init__(self, in_channels=11, num_classes=NUM_CLASSES):
        super().__init__()
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, 64, 3, stride=1)
        self.enc2 = ConvBlock(64, 128, 3, stride=2)
        self.enc3 = ConvBlock(128, 256, 3, stride=2)
        self.enc4 = ConvBlock(256, 256, 3, stride=2)
        self.enc5 = ConvBlock(256, 512, 3, stride=2)
        
        # Decoder with skip connections
        self.dec5 = UpBlock(512, 256)
        self.dec4 = UpBlock(256 + 256, 256)   # skip from enc4
        self.dec3 = UpBlock(256 + 256, 128)   # skip from enc3
        self.dec2 = UpBlock(128 + 128, 64)    # skip from enc2
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.head = nn.Conv2d(64, num_classes, 1)
    
    def forward(self, x):
        e1 = self.enc1(x)      # [B, 64, H, W]
        e2 = self.enc2(e1)     # [B, 128, H/2, W/2]
        e3 = self.enc3(e2)     # [B, 256, H/4, W/4]
        e4 = self.enc4(e3)     # [B, 256, H/8, W/8]
        e5 = self.enc5(e4)     # [B, 512, H/16, W/16]
        
        d5 = self.dec5(e5)                              # [B, 256, H/8, W/8]
        d4 = self.dec4(torch.cat([d5, e4], dim=1))      # [B, 256, H/4, W/4]
        d3 = self.dec3(torch.cat([d4, e3], dim=1))      # [B, 128, H/2, W/2]
        d2 = self.dec2(torch.cat([d3, e2], dim=1))      # [B, 64, H, W]
        d1 = self.dec1(torch.cat([d2, e1], dim=1))      # [B, 64, H, W]
        
        return self.head(d1)


# === Building blocks ===

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel//2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.up(x)


# === Loss functions ===

class DAELoss(nn.Module):
    """Combined loss for DAE: CrossEntropy + Dice + optional boundary."""
    def __init__(self, ce_weight=1.0, dice_weight=1.0, boundary_weight=0.5,
                 num_classes=NUM_CLASSES):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, pred, target):
        # pred: [B, C, H, W], target: [B, H, W]
        ce_loss = self.ce(pred, target)
        dice_loss = self._dice_loss(pred, target)
        loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        
        if self.boundary_weight > 0:
            boundary_loss = self._boundary_loss(pred, target)
            loss += self.boundary_weight * boundary_loss
        
        return loss
    
    def _dice_loss(self, pred, target):
        pred_soft = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        
        intersection = (pred_soft * target_onehot).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        return 1 - dice.mean()
    
    def _boundary_loss(self, pred, target):
        """Extra loss at boundary regions."""
        # Find boundary in target
        target_float = target.float().unsqueeze(1)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=target.device).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)
        
        gx = F.conv2d(target_float, sobel_x, padding=1)
        gy = F.conv2d(target_float, sobel_y, padding=1)
        boundary = (gx.abs() + gy.abs() > 0).float().squeeze(1)
        
        # Weighted CE at boundaries
        ce_per_pixel = F.cross_entropy(pred, target, reduction='none')
        boundary_ce = (ce_per_pixel * boundary).sum() / (boundary.sum() + 1e-6)
        
        return boundary_ce


# === Factory ===

def build_model(model_name: str, **kwargs) -> nn.Module:
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
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    # Test all models
    x = torch.randn(2, 11, 256, 256)
    
    for name in ['unet_resnet34', 'unet_effnet', 'conditional', 'lightweight']:
        try:
            model = build_model(name)
            out = model(x)
            total, trainable = count_params(model)
            print(f'{name:>20s} | output: {list(out.shape)} | params: {total/1e6:.1f}M ({trainable/1e6:.1f}M trainable)')
        except Exception as e:
            print(f'{name:>20s} | ERROR: {e}')
