# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pseudo-label Denoiser for Satellite Image Segmentation. Trains Denoising AutoEncoders (DAE) to clean noisy pseudo-labels in semantic segmentation using OpenEarthMap dataset.

## Quick Commands

```bash
# Setup environment
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install segmentation-models-pytorch opencv-python matplotlib pyyaml tqdm

# Train DAE models
cd src/
python train_dae.py --config ../configs/dae_lightweight.yaml  # Best model (97.78% mIoU)
python train_dae.py --config ../configs/dae_resnet34.yaml
python train_dae.py --config ../configs/dae_effnet.yaml
python train_dae.py --config ../configs/dae_conditional.yaml

# Override config from CLI
python train_dae.py --config ../configs/dae_lightweight.yaml --override training.lr=0.0005

# Inference & visualization
python demo_inference_real.py --checkpoint checkpoints/dae_lightweight_..._best.pth --pseudo_root data/OEM_v2_aDanh

# Evaluate
python evaluate_dae.py --checkpoint checkpoints/..._best.pth --pseudo_root data/OEM_v2_aDanh
```

## Architecture Overview

### Core Components

```
src/
├── config.py           # YAML config loader with inheritance (_base_) and CLI overrides
├── dae_model.py        # 4 DAE architectures + DAELoss (CE + Dice + Boundary)
├── noise_generator.py  # CLASS_NAMES, compute_iou utility
├── dataset.py          # OpenEarthMapDataset + RealNoiseDAEDataset (pseudo-labels from CISC-R)
├── train_dae.py        # Training loop with AMP, early stopping, W&B logging
├── evaluate_dae.py     # Metrics computation (mIoU, per-class IoU)
├── demo_inference_real.py  # Inference with real pseudo-labels
└── upload_to_wandb_run.py  # Upload results to existing W&B run
```

### Model Architecture (Later Fusion)

All DAE models follow dual-branch design:
- **RGB branch**: Pretrained encoder (ResNet34/EfficientNet) or custom CNN
- **Label branch**: Lightweight encoder for one-hot labels (8 channels)
- **Fusion**: Channel attention at bottleneck (H/32)
- **Decoder**: Skip connections from BOTH branches at 4 scales (H/2, H/4, H/8, H/16)

Input: `rgb [B,3,H,W]` + `noisy_label [B,8,H,W]` → Output: `logits [B,8,H,W]`

### Models

| Model | Encoder | Params | Best mIoU |
|-------|---------|--------|-----------|
| `lightweight` | Custom CNN | 12.82M | 97.78% |
| `unet_resnet34` | ResNet34 | 24.46M | 94.88% |
| `unet_effnet` | EfficientNet-B4 | 20.23M | 96.00% |
| `conditional` | ResNet34 + Attn | 39.10M | 89.22% |

### Config System

Configs use inheritance via `_base_` field:
```yaml
# configs/dae_lightweight.yaml
_base_: default.yaml  # Inherits data_root, img_size, wandb, pseudo_root
model:
  name: lightweight
training:
  batch_size: 8
  lr: 0.0001
  epochs: 100
```

CLI overrides use dot notation: `--override training.lr=0.001`

### Loss Function

```
Total Loss = CE×1.0 + Dice×1.0 + Boundary×0.5
```
- `DAELoss` in `dae_model.py:557`

### Key Findings

1. **Smaller model wins**: Lightweight DAE (12.82M) outperforms larger models
2. **Pretrained weights not helpful**: Input domain mismatch (11 channels vs ImageNet 3 channels)
3. **Dual-encoder overhead**: Complex architecture (39.1M) underperforms simple one
4. **Diffusion not pursued**: Complex approach, slow training

## Data

**OpenEarthMap**: 2303 train / 500 val images, 512×512, 8 classes
- Structure: `data_root/{region}/images/{id}.tif` + `labels/{id}.tif`
- Split files: `train.txt`, `val.txt` at data_root

**Pseudo-labels**: From CISC-R model, organized via `reorganize_pseudo_dataset.py`

## Checkpoints & Logs

- Checkpoints: `checkpoints/{model}_real_{timestamp}_best.pth`
- History: `results/logs/{exp}_history.json`
- W&B: Auto-logs metrics, inference images, checkpoints as artifacts
