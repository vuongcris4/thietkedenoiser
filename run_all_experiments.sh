#!/bin/bash
set -e
cd /home/ubuntu/thietkedenoiser
export OPENCV_LOG_LEVEL=ERROR

echo "=========================================="
echo "  DAE Later Fusion Experiments"
echo "  Started: $(date)"
echo "=========================================="

for cfg in dae_lightweight dae_resnet34 dae_effnet dae_conditional; do
  echo ""
  echo "=========================================="
  echo "  Running: $cfg — $(date)"
  echo "=========================================="
  python src/train_dae.py --config "configs/${cfg}.yaml"
done

echo ""
echo "=========================================="
echo "  All experiments completed! — $(date)"
echo "=========================================="
