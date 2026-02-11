# 🛰️ Pseudo-label Denoiser for Satellite Image Segmentation

> Denoising AutoEncoder (DAE) và Conditional Diffusion models để tinh chỉnh pseudo-labels nhiễu trong bài toán semantic segmentation ảnh vệ tinh, sử dụng dataset OpenEarthMap.

---

## 📋 Tổng quan

Trong pipeline semi-supervised semantic segmentation, pseudo-labels từ model teacher thường chứa nhiều loại lỗi: pixel lẻ sai class, biên giữa các vùng bị nhòe, hoặc cả vùng lớn bị nhầm class. Project này xây dựng các model **denoiser** để tự động sửa các lỗi đó.

### Approach

1. **Tạo nhiễu nhân tạo** trên ground truth labels (4 loại noise mô phỏng lỗi thực tế)
2. **Train denoiser** học mapping: noisy label → clean label (có điều kiện trên RGB image)
3. **Inference**: Áp dụng denoiser lên pseudo-labels từ model segmentation

### Models đã thí nghiệm

| # | Model | Params | Kiến trúc | Best mIoU | Epochs |
|---|-------|--------|-----------|-----------|--------|
| 1 | UNet-ResNet34 | 24.46M | UNet + pretrained ResNet34 encoder | 94.88% | 58 (early stop) |
| 2 | UNet-EfficientNet-B4 | 20.23M | UNet + pretrained EfficientNet-B4 encoder | 96.00% | 95 (early stop) |
| 3 | **Lightweight DAE** ⭐ | **12.82M** | Custom U-Net (tự thiết kế) | **97.78%** | 89/100 |
| 4 | Conditional Diffusion | 22.25M | U-Net + time embedding (DDPM) | 23.81% | 50 |
| 5 | Conditional DAE | 39.10M | Dual Encoder + Channel Attention | 89.22% | 90/100 |

> **Kết luận:** Lightweight DAE nhỏ nhất nhưng đạt kết quả tốt nhất. Pretrained encoders không giúp ích vì input domain (11 channels) khác ImageNet (3 channels).

---

## 📁 Cấu trúc project

```
thietkedenoiser/
├── configs/                    # YAML configs cho từng model
│   ├── default.yaml            # Config chung (data, device, paths)
│   ├── dae_resnet34.yaml       # UNet-ResNet34
│   ├── dae_effnet.yaml         # UNet-EfficientNet-B4
│   ├── dae_lightweight.yaml    # Lightweight DAE
│   ├── dae_conditional.yaml    # Conditional DAE (Dual Encoder)
│   └── diffusion.yaml          # Conditional Diffusion
├── src/
│   ├── config.py               # Config loader (YAML + CLI overrides)
│   ├── dae_model.py            # 4 DAE architectures + DAELoss
│   ├── diffusion_denoiser.py   # Conditional Diffusion model
│   ├── noise_generator.py      # 4 loại noise + mixed
│   ├── dataset.py              # DAEDataset với noise injection on-the-fly
│   ├── train_dae.py            # Training script (hỗ trợ tất cả DAE models)
│   ├── train_diffusion.py      # Diffusion training
│   ├── evaluate_dae.py         # Evaluation metrics
│   ├── evaluate_noise.py       # Phân tích noise statistics
│   ├── demo_inference.py       # Demo visualization
│   ├── plot_eval.py            # Plot kết quả
│   └── run_eval.py             # Run full evaluation
├── checkpoints/                # Model weights (Git LFS)
├── results/
│   ├── logs/                   # Training history (JSON)
│   ├── metrics/                # Evaluation metrics
│   └── visualizations/         # Demo output images
├── scripts/
│   ├── setup.sh                # Setup environment
│   ├── download_data.sh        # Download OpenEarthMap
│   ├── run_experiments.sh      # Run tất cả experiments
│   └── run_all.sh              # Full pipeline
└── data/                       # Dataset (không track trong git)
    └── OpenEarthMap/
```

---

## 🗂️ Dataset — OpenEarthMap

- **Source:** [OpenEarthMap](https://open-earth-map.org/)
- **Split:** 2303 train / 500 val images
- **Size:** 512×512 pixels
- **GSD:** 0.25–0.5m
- **8 classes:** Bareland, Rangeland, Developed space, Road, Tree, Water, Agriculture, Building

---

## 🔧 Cài đặt

```bash
# Clone repo
git clone https://github.com/vuongcris4/thietkedenoiser.git
cd thietkedenoiser

# Pull checkpoints (Git LFS)
git lfs pull

# Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install segmentation-models-pytorch opencv-python matplotlib pyyaml tqdm

# Download dataset
bash scripts/download_data.sh
```

---

## 🚀 Training

Sử dụng config-based training — mỗi model có file YAML riêng:

```bash
cd src/

# Lightweight DAE (recommended — best results)
python train_dae.py --config ../configs/dae_lightweight.yaml

# UNet-ResNet34
python train_dae.py --config ../configs/dae_resnet34.yaml

# UNet-EfficientNet-B4
python train_dae.py --config ../configs/dae_effnet.yaml

# Conditional DAE (Dual Encoder + Attention)
python train_dae.py --config ../configs/dae_conditional.yaml

# Conditional Diffusion
python train_diffusion.py --config ../configs/diffusion.yaml
```

### Override config từ CLI

```bash
# Đổi batch size và learning rate
python train_dae.py --config ../configs/dae_lightweight.yaml \
    --training.batch_size 16 --training.lr 0.0005

# Đổi noise rate range
python train_dae.py --config ../configs/dae_lightweight.yaml \
    --noise.rate_min 0.10 --noise.rate_max 0.40
```

### Output

- **Checkpoints:** `checkpoints/{model}_{noise}_{timestamp}_best.pth`
- **Training history:** `results/logs/{model}_{noise}_{timestamp}_history.json`
- **Console log:** Real-time metrics mỗi epoch

---

## 🔍 Inference & Demo

Visualize kết quả denoising: **RGB → Noisy Label → DAE Output → Ground Truth**

```bash
# Quick demo với Lightweight DAE
python src/demo_inference.py \
    --checkpoint checkpoints/dae_lightweight_mixed_20260208_122512_best.pth

# Custom demo
python src/demo_inference.py \
    --checkpoint checkpoints/dae_lightweight_mixed_20260208_122512_best.pth \
    --model lightweight \
    --noise_type mixed \
    --noise_rates 0.10 0.20 0.30 \
    --num_samples 4 \
    --output_dir results/visualizations/demo
```

| Parameter | Default | Mô tả |
|-----------|---------|-------|
| `--checkpoint` | *(bắt buộc)* | Path tới file `.pth` |
| `--model` | `lightweight` | `lightweight` / `unet_resnet34` / `unet_effnet` / `conditional` |
| `--noise_type` | `mixed` | `random_flip` / `boundary` / `region_swap` / `confusion` / `mixed` |
| `--noise_rates` | `0.10 0.20 0.30` | Các tỷ lệ noise cần test |
| `--num_samples` | `4` | Số samples mỗi noise rate |
| `--split` | `val` | `train` / `val` |
| `--seed` | `2026` | Random seed |

---

## 🎯 Cơ chế tạo nhiễu (Noise Generation)

Mỗi sample được inject noise **on-the-fly** với `noise_rate` random trong [5%, 30%]. Mixed noise kết hợp cả 4 loại, mỗi loại chiếm `noise_rate/4`:

| Loại | Mô tả | Mô phỏng lỗi |
|------|--------|---------------|
| **Random Flip** | Đổi ngẫu nhiên class pixel lẻ | Lỗi prediction rải rác (salt-and-pepper) |
| **Boundary** | Dilate/Erode ranh giới bằng morphological ops | Lỗi phổ biến nhất — sai ở biên giữa 2 class |
| **Region Swap** | Hoán class cả vùng lớn (20-100px) | Lỗi nghiêm trọng — nhầm toàn bộ 1 vùng |
| **Confusion-based** | Đổi class theo confusion matrix giả lập | Class giống nhau visual có xác suất nhầm cao hơn |

---

## 📊 Kết quả chi tiết

### Lightweight DAE ⭐ (Best model — 97.78% mIoU)

```
Bareland: 98.4%  |  Rangeland: 97.2%  |  Developed: 96.8%  |  Road: 96.5%
Tree:     98.0%  |  Water:     98.1%  |  Agriculture: 97.9% |  Building: 99.3%
```

### Conditional DAE (Exp 5 — 89.22% mIoU)

```
Bareland: 96.7%  |  Rangeland: 91.4%  |  Developed: 85.1%  |  Road: 82.5%
Tree:     87.1%  |  Water:     86.7%  |  Agriculture: 91.8% |  Building: 92.5%
```

### Key findings

1. **Model nhỏ > model lớn** — Lightweight DAE (12.82M) thắng tất cả models lớn hơn
2. **Pretrained không giúp ích** — Input 11 channels khác ImageNet, pretrained weights bị mismatch
3. **Dual-encoder phức tạp hóa** — Tách RGB/label encoder (39.1M) kém hơn single-encoder (12.82M)
4. **Diffusion chưa hiệu quả** — 50 epochs vẫn không hội tụ, approach phức tạp hơn nhiều so với DAE
5. **CE + Dice + Boundary loss** — Kết hợp 3 loss giúp bảo toàn ranh giới class tốt

---

## ⚙️ Loss Function

```
Total Loss = CE Loss × 1.0 + Dice Loss × 1.0 + Boundary Loss × 0.5
```

- **CrossEntropy Loss:** Pixel-wise classification
- **Dice Loss:** Overlap-based, xử lý class imbalance
- **Boundary Loss:** Sobel edge detection trên prediction vs target, bảo toàn biên

---

## 💻 Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA L4 (23GB VRAM) |
| CPU | AMD EPYC 7R13 |
| RAM | 16GB |
| CUDA | 12.1 |
| PyTorch | 2.5.1 |
| Platform | AWS EC2 |

---

## 📄 License

MIT License

