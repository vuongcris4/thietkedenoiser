#!/bin/bash
set -e
cd ~/thietkedenoiser
source venv/bin/activate

DATA_ROOT="data/OpenEarthMap"
IMG_SIZE=512
BATCH=4
EPOCHS=100
PATIENCE=15

echo "============================================"
echo "  DAE EXPERIMENTS - $(date)"
echo "============================================"

# Exp 1: U-Net ResNet-34 + Mixed noise
echo ""
echo ">>> EXP 1: UNet-ResNet34 + Mixed noise"
python3 src/train_dae.py \
    --model unet_resnet34 \
    --data_root $DATA_ROOT \
    --noise_type mixed \
    --img_size $IMG_SIZE \
    --batch_size $BATCH \
    --epochs $EPOCHS \
    --patience $PATIENCE

# Exp 2: U-Net EfficientNet-B4 + Mixed noise  
echo ""
echo ">>> EXP 2: UNet-EffNetB4 + Mixed noise"
python3 src/train_dae.py \
    --model unet_effnet \
    --data_root $DATA_ROOT \
    --noise_type mixed \
    --img_size $IMG_SIZE \
    --batch_size $BATCH \
    --epochs $EPOCHS \
    --patience $PATIENCE

# Exp 3: Conditional DAE + Mixed noise
echo ""
echo ">>> EXP 3: Conditional DAE + Mixed noise"
python3 src/train_dae.py \
    --model conditional \
    --data_root $DATA_ROOT \
    --noise_type mixed \
    --img_size $IMG_SIZE \
    --batch_size 2 \
    --epochs $EPOCHS \
    --patience $PATIENCE

# Exp 4: Lightweight DAE + Mixed noise
echo ""
echo ">>> EXP 4: Lightweight DAE + Mixed noise"
python3 src/train_dae.py \
    --model lightweight \
    --data_root $DATA_ROOT \
    --noise_type mixed \
    --img_size $IMG_SIZE \
    --batch_size 8 \
    --epochs $EPOCHS \
    --patience $PATIENCE

echo ""
echo "============================================"
echo "  ALL EXPERIMENTS DONE - $(date)"
echo "============================================"
