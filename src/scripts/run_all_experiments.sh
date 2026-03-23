#!/bin/bash
# =============================================================================
# Run All DAE Experiments - 30 epochs each
# =============================================================================
# Usage: ./run_all_experiments.sh
#
# This script runs 3 DAE models:
#   1. lightweight    - Best model candidate (12.82M params)
#   2. unet_resnet34  - ResNet-34 encoder (24.46M params)
#   3. unet_effnet    - EfficientNet-B4 encoder (20.23M params)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_DIR="$PROJECT_ROOT/configs"

# cd to project root so relative paths in config files work correctly
cd "$PROJECT_ROOT"

echo "============================================================"
echo "Starting DAE Experiments - 30 epochs each"
echo "Started at: $(date)"
echo "============================================================"
echo "Project root: $PROJECT_ROOT"
echo "Config dir: $CONFIG_DIR"
echo "============================================================"

# Override epochs to 30 for all experiments
EPOCHS=30

# Experiment 1: Lightweight DAE
echo ""
echo ">>> Experiment 1/4: Lightweight DAE (${EPOCHS} epochs)"
echo "Started at: $(date)"
python src/scripts/train_dae.py \
    --config "$CONFIG_DIR/dae_lightweight.yaml" \
    --override training.epochs=${EPOCHS} \
    --override training.patience=10

# Experiment 2: UNet ResNet-34
echo ""
echo ">>> Experiment 2/4: UNet ResNet-34 DAE (${EPOCHS} epochs)"
echo "Started at: $(date)"
python src/scripts/train_dae.py \
    --config "$CONFIG_DIR/dae_resnet34.yaml" \
    --override training.epochs=${EPOCHS} \
    --override training.patience=10

# Experiment 3: UNet EfficientNet-B4
echo ""
echo ">>> Experiment 3/3: UNet EfficientNet-B4 DAE (${EPOCHS} epochs)"
echo "Started at: $(date)"
python src/scripts/train_dae.py \
    --config "$CONFIG_DIR/dae_effnet.yaml" \
    --override training.epochs=${EPOCHS} \
    --override training.patience=10

echo ""
echo "============================================================"
echo "All experiments completed!"
echo "Finished at: $(date)"
echo "============================================================"
