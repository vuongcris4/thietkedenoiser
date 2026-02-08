# Thiet Ke Denoiser - DAE & Diffusion for Pseudo-label Refinement

Denoising AutoEncoder (DAE) and Conditional Diffusion models for refining noisy pseudo-labels in semantic segmentation on the OpenEarthMap dataset.

## Overview

This project implements pseudo-label denoising for satellite image segmentation:
- **4 DAE variants**: ResNet34 UNet, EfficientNet-B4 UNet, Conditional DAE, Lightweight DAE
- **1 Diffusion-based denoiser**: Conditional Diffusion with DDPM
- **4 noise types**: Random Flip, Boundary, Region Swap, Confusion-based (+ Mixed)

## Dataset

- **OpenEarthMap**: 3000 train / 500 val images, 512x512, 8 classes
- Classes: Bareland, Rangeland, Developed, Road, Tree, Water, Agriculture, Building
- GSD: 0.25-0.5m

## Results

| Model | Params | Best mIoU | Epochs |
|-------|--------|-----------|--------|
| ResNet34 UNet | 24.46M | 94.88% | 58/100 (early stop) |
| EfficientNet-B4 UNet | 20.23M | 96.00% | 95/100 (early stop) |
| **Lightweight DAE** | **12.82M** | **97.78%** | 89/100 |
| Conditional Diffusion | 22.25M | 18.86% | 20/20 |

## Project Structure

```
src/
  dae_model.py              # 4 DAE model architectures + DAELoss
  diffusion_denoiser.py     # Conditional Diffusion model
  noise_generator.py        # 4 noise types + mixed noise
  dataset.py                # DAEDataset with noise injection
  train_dae.py              # DAE training script
  train_diffusion.py        # Diffusion training script
  train_diffusion_resume.py # Resume training from checkpoint
  train_diffusion_resume_v2.py # Resume v2 (saves latest every epoch)
  evaluate_dae.py           # DAE evaluation
  evaluate_noise.py         # Noise analysis & statistics
  run_eval.py               # Run full evaluation
  plot_eval.py              # Plot evaluation results
```

## Training

```bash
# Train DAE models
python src/train_dae.py --model lightweight --epochs 100 --batch_size 8
python src/train_dae.py --model unet_resnet34 --epochs 100 --batch_size 8
python src/train_dae.py --model unet_effnet --epochs 100 --batch_size 8

# Train Diffusion
python src/train_diffusion.py --epochs 20 --batch_size 4 --T 1000 --dim 64
```

## Loss Function (DAE)

- **CrossEntropy Loss** (weight=1.0)
- **Dice Loss** (weight=1.0)
- **Boundary Loss** using Sobel edge detection (weight=0.5)

## Hardware

- AWS EC2 with NVIDIA L4 GPU (23GB VRAM)
- AMD EPYC 7R13, 16GB RAM
- CUDA 12.1, PyTorch 2.5.1

## Author

Tran Duy Vuong - HCMUTE (22139078)
