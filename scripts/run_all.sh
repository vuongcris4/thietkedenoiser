#!/bin/bash
set -e
cd ~/thietkedenoiser
source venv/bin/activate

DATA_ROOT="data/OpenEarthMap"
IMG_SIZE=512
EPOCHS=100
PATIENCE=15

echo "============================================"
echo "  ALL EXPERIMENTS - $(date)"
echo "============================================"

# === DAE EXPERIMENTS ===

# Exp 1: U-Net ResNet-34 (DAE baseline)
echo ""
echo ">>> EXP 1: DAE UNet-ResNet34 + Mixed noise"
python3 src/train_dae.py \
    --model unet_resnet34 \
    --data_root $DATA_ROOT \
    --noise_type mixed \
    --img_size $IMG_SIZE \
    --batch_size 4 \
    --epochs $EPOCHS \
    --patience $PATIENCE

# Exp 2: U-Net EfficientNet-B4
echo ""
echo ">>> EXP 2: DAE UNet-EffNetB4 + Mixed noise"
python3 src/train_dae.py \
    --model unet_effnet \
    --data_root $DATA_ROOT \
    --noise_type mixed \
    --img_size $IMG_SIZE \
    --batch_size 4 \
    --epochs $EPOCHS \
    --patience $PATIENCE

# Exp 3: Lightweight DAE
echo ""
echo ">>> EXP 3: Lightweight DAE + Mixed noise"
python3 src/train_dae.py \
    --model lightweight \
    --data_root $DATA_ROOT \
    --noise_type mixed \
    --img_size $IMG_SIZE \
    --batch_size 8 \
    --epochs $EPOCHS \
    --patience $PATIENCE

# === DIFFUSION EXPERIMENT (bai bao goc) ===

# Exp 4: Conditional Diffusion Denoiser
echo ""
echo ">>> EXP 4: Conditional Diffusion Denoiser (T=1000)"
python3 src/train_diffusion.py \
    --data_root $DATA_ROOT \
    --img_size $IMG_SIZE \
    --batch_size 4 \
    --epochs $EPOCHS \
    --T 1000 \
    --base_dim 64 \
    --patience 20 \
    --val_every 5 \
    --denoise_steps 50

echo ""
echo "============================================"
echo "  ALL DONE - $(date)"
echo "============================================"
