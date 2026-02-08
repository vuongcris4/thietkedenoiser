#!/bin/bash
set -e
echo "=== Setting up Denoiser Environment ==="

# Create venv
python3 -m venv ~/thietkedenoiser/venv
source ~/thietkedenoiser/venv/bin/activate

# Install PyTorch with CUDA
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install ML packages
pip install segmentation-models-pytorch timm albumentations opencv-python-headless
pip install wandb tensorboard tqdm pyyaml einops scikit-learn matplotlib
pip install rasterio  # for GeoTIFF support

echo "=== Setup complete ==="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
