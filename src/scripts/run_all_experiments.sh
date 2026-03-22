#!/bin/bash
# =============================================================================
# Run All DAE Experiments - 30 epochs each
# =============================================================================
# Usage: ./run_all_experiments.sh
#
# This script runs 4 DAE models:
#   1. lightweight    - Best model candidate (12.82M params)
#   2. unet_resnet34  - ResNet-34 encoder (24.46M params)
#   3. unet_effnet    - EfficientNet-B4 encoder (20.23M params)
#   4. conditional    - Attention fusion (39.10M params)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_DIR="$PROJECT_ROOT/configs"

cd "$SCRIPT_DIR"

echo "============================================================"
echo "Starting DAE Experiments - 30 epochs each"
echo "Started at: $(date)"
echo "============================================================"

# Override epochs to 30 for all experiments
EPOCHS=30

# Experiment 1: Lightweight DAE
echo ""
echo ">>> Experiment 1/4: Lightweight DAE (${EPOCHS} epochs)"
echo "Started at: $(date)"
python train_dae.py \
    --config "$CONFIG_DIR/dae_lightweight.yaml" \
    --override training.epochs=${EPOCHS} \
    --override training.patience=10

# Experiment 2: UNet ResNet-34
echo ""
echo ">>> Experiment 2/4: UNet ResNet-34 DAE (${EPOCHS} epochs)"
echo "Started at: $(date)"
python train_dae.py \
    --config "$CONFIG_DIR/dae_resnet34.yaml" \
    --override training.epochs=${EPOCHS} \
    --override training.patience=10

# Experiment 3: UNet EfficientNet-B4
echo ""
echo ">>> Experiment 3/4: UNet EfficientNet-B4 DAE (${EPOCHS} epochs)"
echo "Started at: $(date)"
python train_dae.py \
    --config "$CONFIG_DIR/dae_effnet.yaml" \
    --override training.epochs=${EPOCHS} \
    --override training.patience=10

# Experiment 4: Conditional DAE
echo ""
echo ">>> Experiment 4/4: Conditional DAE (${EPOCHS} epochs)"
echo "Started at: $(date)"
python train_dae.py \
    --config "$CONFIG_DIR/dae_conditional.yaml" \
    --override training.epochs=${EPOCHS} \
    --override training.patience=10

echo ""
echo "============================================================"
echo "All experiments completed!"
echo "Finished at: $(date)"
echo "============================================================"
