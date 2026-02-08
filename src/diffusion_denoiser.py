"""
Conditional Diffusion Model for Pseudo-label Denoising.

Theo bai bao goc:
  - Forward: Them nhieu Gaussian vao GT label (semantic map CxHxW)
  - Reverse: Hoc khu nhieu voi conditioning = RGB image
  - Input x: noisy semantic map (one-hot CxHxW)
  - Condition: RGB image (3xHxW)
  - Output: clean semantic map (CxHxW)

Architecture: Conditional U-Net (DDPM-style)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

NUM_CLASSES = 8


# === Diffusion Schedule ===

class DiffusionSchedule:
    """Linear beta schedule for DDPM."""
    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02,
                 device: str = 'cuda'):
        self.T = T
        self.device = device
        
        # Linear schedule
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0)
        
        # Pre-compute for q(x_t | x_0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        
        # Pre-compute for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)
        )
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """Forward process: q(x_t | x_0) = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*eps"""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha = self.sqrt_alpha_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[t][:, None, None, None]
        
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
    
    def p_sample(self, model_output: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        """Reverse step: sample x_{t-1} from p(x_{t-1} | x_t)"""
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[t][:, None, None, None]
        sqrt_recip_alpha = torch.rsqrt(self.alphas[t])[:, None, None, None]
        
        # Predicted mean
        mean = sqrt_recip_alpha * (x_t - betas_t * model_output / sqrt_one_minus_alpha)
        
        # Add noise (except for t=0)
        if (t > 0).any():
            noise = torch.randn_like(x_t)
            variance = torch.sqrt(self.posterior_variance[t])[:, None, None, None]
            # Zero noise for t=0
            mask = (t > 0).float()[:, None, None, None]
            return mean + variance * noise * mask
        return mean


# === Building Blocks ===

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timestep t."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        
        self.time_mlp = nn.Linear(time_dim, out_ch) if time_dim else None
        
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x, t_emb=None):
        h = self.act(self.norm1(self.conv1(x)))
        
        if self.time_mlp is not None and t_emb is not None:
            t = self.act(self.time_mlp(t_emb))[:, :, None, None]
            h = h + t
        
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, time_dim)
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)
    
    def forward(self, x, t):
        h = self.conv(x, t)
        return self.down(h), h  # downsampled, skip (out_ch channels)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.conv = ConvBlock(out_ch + in_ch, out_ch, time_dim)  # concat: up(out_ch) + skip(in_ch)
    
    def forward(self, x, skip, t):
        x = self.up(x)
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x, t)


# === Main Model ===

class ConditionalDiffusionUNet(nn.Module):
    """
    Conditional U-Net for diffusion denoising.
    
    Predicts noise epsilon given:
      - x_t: noisy label map (C channels, one-hot float)
      - t: diffusion timestep
      - cond: RGB image (3 channels) as conditioning
    
    Architecture follows DDPM U-Net with:
      - Time embedding (sinusoidal)
      - Condition injection via concatenation
      - GroupNorm + SiLU activation
    """
    def __init__(self, in_channels=NUM_CLASSES, cond_channels=3,
                 base_dim=64, dim_mults=(1, 2, 4, 8), time_dim=256):
        super().__init__()
        
        self.time_dim = time_dim
        dims = [base_dim * m for m in dim_mults]
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        
        # Condition encoder (RGB -> feature maps)
        self.cond_encoder = nn.Sequential(
            nn.Conv2d(cond_channels, base_dim, 3, padding=1),
            nn.GroupNorm(8, base_dim),
            nn.SiLU(),
            nn.Conv2d(base_dim, base_dim, 3, padding=1),
            nn.GroupNorm(8, base_dim),
            nn.SiLU(),
        )
        
        # Input projection: x_t (C) + cond_features (base_dim) -> base_dim
        self.input_proj = nn.Conv2d(in_channels + base_dim, dims[0], 3, padding=1)
        
        # Encoder (down)
        self.down_blocks = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.down_blocks.append(DownBlock(dims[i], dims[i+1], time_dim))
        
        # Bottleneck
        self.mid = ConvBlock(dims[-1], dims[-1], time_dim)
        
        # Decoder (up)
        self.up_blocks = nn.ModuleList()
        for i in range(len(dims) - 1, 0, -1):
            self.up_blocks.append(UpBlock(dims[i], dims[i-1], time_dim))
        
        # Output projection
        self.output = nn.Sequential(
            nn.GroupNorm(8, dims[0]),
            nn.SiLU(),
            nn.Conv2d(dims[0], in_channels, 1),
        )
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, 
                cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: [B, C, H, W] noisy label map at timestep t
            t:   [B] timestep indices
            cond: [B, 3, H, W] RGB conditioning image
        Returns:
            predicted noise epsilon [B, C, H, W]
        """
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Condition features
        cond_feat = self.cond_encoder(cond)
        
        # Concat x_t with condition features
        x = torch.cat([x_t, cond_feat], dim=1)
        x = self.input_proj(x)
        
        # Encoder
        skips = []
        for down in self.down_blocks:
            x, skip = down(x, t_emb)
            skips.append(skip)
        
        # Bottleneck
        x = self.mid(x, t_emb)
        
        # Decoder
        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip, t_emb)
        
        return self.output(x)


# === Diffusion Denoiser Wrapper ===

class DiffusionDenoiser(nn.Module):
    """
    Complete Diffusion Denoiser pipeline.
    
    Training:
      1. Take clean label (one-hot) as x_0
      2. Sample random t, add noise: x_t = sqrt(a_t)*x_0 + sqrt(1-a_t)*eps
      3. Predict eps with model(x_t, t, rgb_condition)
      4. Loss = MSE(eps_pred, eps)
    
    Inference:
      1. Start from noisy pseudo-label (or pure noise)
      2. Iteratively denoise: x_{t-1} = reverse_step(x_t, t, rgb)
      3. After T steps, x_0 = clean label
    """
    def __init__(self, num_classes=NUM_CLASSES, T=1000,
                 base_dim=64, dim_mults=(1, 2, 4, 8)):
        super().__init__()
        self.num_classes = num_classes
        self.T = T
        
        self.model = ConditionalDiffusionUNet(
            in_channels=num_classes,
            cond_channels=3,
            base_dim=base_dim,
            dim_mults=dim_mults,
        )
        self.schedule = None  # Initialized on first forward
    
    def _init_schedule(self, device):
        if self.schedule is None or self.schedule.device != device:
            self.schedule = DiffusionSchedule(T=self.T, device=device)
    
    def training_step(self, clean_label_onehot: torch.Tensor, 
                      rgb: torch.Tensor) -> torch.Tensor:
        """
        One training step.
        Args:
            clean_label_onehot: [B, C, H, W] clean one-hot label
            rgb: [B, 3, H, W] RGB conditioning image
        Returns:
            loss (MSE between predicted and actual noise)
        """
        self._init_schedule(clean_label_onehot.device)
        B = clean_label_onehot.shape[0]
        
        # Sample random timestep
        t = torch.randint(0, self.T, (B,), device=clean_label_onehot.device)
        
        # Sample noise
        noise = torch.randn_like(clean_label_onehot)
        
        # Forward diffusion: x_t = noisy version of clean label
        x_t = self.schedule.q_sample(clean_label_onehot, t, noise)
        
        # Predict noise
        eps_pred = self.model(x_t, t, rgb)
        
        # MSE loss
        loss = F.mse_loss(eps_pred, noise)
        return loss
    
    @torch.no_grad()
    def denoise(self, noisy_label_onehot: torch.Tensor, rgb: torch.Tensor,
                num_steps: int = 50, start_t: Optional[int] = None) -> torch.Tensor:
        """
        Denoise a noisy pseudo-label.
        
        Instead of starting from pure noise (t=T), we can start from
        the noisy pseudo-label at an intermediate timestep, since
        the pseudo-label is partially correct (not pure noise).
        
        Args:
            noisy_label_onehot: [B, C, H, W] noisy pseudo-label (one-hot float)
            rgb: [B, 3, H, W] RGB conditioning
            num_steps: number of denoising steps (DDIM-like subsampling)
            start_t: starting timestep (default: T//4, since pseudo-labels aren't pure noise)
        Returns:
            cleaned label onehot [B, C, H, W]
        """
        self._init_schedule(noisy_label_onehot.device)
        
        if start_t is None:
            start_t = self.T // 4  # Pseudo-labels are partially correct
        
        # Subsample timesteps for faster inference
        timesteps = torch.linspace(start_t, 0, num_steps, dtype=torch.long,
                                   device=noisy_label_onehot.device)
        
        # Add noise to pseudo-label up to start_t
        t_start = torch.full((noisy_label_onehot.shape[0],), start_t,
                            device=noisy_label_onehot.device, dtype=torch.long)
        x_t = self.schedule.q_sample(noisy_label_onehot, t_start)
        
        # Iterative denoising
        for i in range(len(timesteps) - 1):
            t = timesteps[i].unsqueeze(0).expand(x_t.shape[0])
            eps_pred = self.model(x_t, t, rgb)
            x_t = self.schedule.p_sample(eps_pred, x_t, t)
        
        return x_t
    
    def get_clean_prediction(self, noisy_label_onehot: torch.Tensor,
                             rgb: torch.Tensor, **kwargs) -> torch.Tensor:
        """Get argmax class prediction after denoising."""
        cleaned = self.denoise(noisy_label_onehot, rgb, **kwargs)
        return cleaned.argmax(dim=1)  # [B, H, W]


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test model
    model = DiffusionDenoiser(num_classes=8, T=1000, base_dim=64, dim_mults=(1, 2, 4, 8))
    model = model.to(device)
    
    total, trainable = count_params(model)
    print(f'Diffusion Denoiser: {total/1e6:.1f}M params ({trainable/1e6:.1f}M trainable)')
    
    # Test training step
    B = 2
    clean_onehot = torch.randn(B, 8, 256, 256, device=device)
    rgb = torch.randn(B, 3, 256, 256, device=device)
    
    loss = model.training_step(clean_onehot, rgb)
    print(f'Training loss: {loss.item():.4f}')
    
    # Test inference
    noisy = torch.randn(B, 8, 128, 128, device=device)
    rgb_small = torch.randn(B, 3, 128, 128, device=device)
    pred = model.get_clean_prediction(noisy, rgb_small, num_steps=10)
    print(f'Prediction shape: {pred.shape}')
    
    print('\nAll tests passed!')
