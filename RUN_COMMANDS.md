# RUN_COMMANDS.md — Toàn Bộ Lệnh Thí Nghiệm

> Tài liệu ghi lại **tất cả các lệnh** đã chạy từ setup đến ra kết quả cuối cùng.
> Phần cứng: **NVIDIA L4 (23 GB VRAM)**, 4 CPU cores, 16 GB RAM.

---

## Phase 0: Chuẩn Bị Dữ Liệu

### 0.1 Tải OpenEarthMap từ Google Drive (rclone)

```bash
# Cài rclone & cấu hình Google Drive remote tên "gdrive"
rclone copy gdrive:OpenEarthMap_extracted/OpenEarthMap_wo_xBD \
    /home/ubuntu/vuongvy/openearthimage_extract \
    -P --transfers=8

# Kiểm tra dữ liệu
find /home/ubuntu/vuongvy/openearthimage_extract -type f | wc -l
du -sh /home/ubuntu/vuongvy/openearthimage_extract
```

### 0.2 Tạo symlink dữ liệu cho project

```bash
# Dữ liệu nằm ở data/OpenEarthMap_wo_xBD (cũng được map thành data/OpenEarthMap)
ln -sf /home/ubuntu/vuongvy/openearthimage_extract data/OpenEarthMap_wo_xBD
```

### 0.3 Kiểm tra split dữ liệu

```bash
wc -l data/OpenEarthMap_wo_xBD/train.txt
wc -l data/OpenEarthMap_wo_xBD/val.txt
wc -l data/OpenEarthMap_wo_xBD/test.txt
# Kết quả: Train 2303 / Val 384 / Test ~1500
```

---

## Phase 1: Setup Môi Trường

### 1.1 Tạo virtual environment & cài packages

```bash
cd ~/thietkedenoiser
bash scripts/setup.sh
```

**Nội dung `scripts/setup.sh`:**
```bash
python3 -m venv ~/thietkedenoiser/venv
source ~/thietkedenoiser/venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install segmentation-models-pytorch timm albumentations opencv-python-headless
pip install wandb tensorboard tqdm pyyaml einops scikit-learn matplotlib
pip install rasterio
```

### 1.2 Kiểm tra CUDA

```bash
source venv/bin/activate
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

---

## Phase 2: Đánh Giá Noise (Trước Khi Train)

### 2.1 Phân tích ảnh hưởng của từng loại noise

```bash
cd ~/thietkedenoiser
source venv/bin/activate

# Chạy đánh giá 5 loại noise × 5 mức noise rate
python3 src/run_eval.py
# Output: Bảng mIoU, boundary ratio, per-class IoU

# Đánh giá chi tiết hơn với ảnh thật
python3 src/evaluate_noise.py \
    --data_dir data/OpenEarthMap_wo_xBD \
    --output_dir results \
    --num_samples 50 \
    --seed 42
# Output: results/NOISE_EVALUATION_REPORT.md
#         results/visualizations/noise_comparison.png
#         results/visualizations/miou_comparison.png
#         results/visualizations/per_class_impact.png
#         results/visualizations/noise_characteristics.png
```

---

## Phase 3: Huấn Luyện DAE (3 Mô Hình)

### 3.1 Chạy toàn bộ 3 DAE + 1 Diffusion (script tự động)

```bash
cd ~/thietkedenoiser
source venv/bin/activate

# Chạy toàn bộ thí nghiệm, log ra file
nohup bash scripts/run_all.sh > results/logs/run_all_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

> **Log file:** `results/logs/run_all_20260207_160648.log` (~747 MB)

### 3.2 Chi tiết từng thí nghiệm DAE

#### EXP 1: U-Net ResNet-34 (24.5M params)

```bash
python3 src/train_dae.py \
    --model unet_resnet34 \
    --data_root data/OpenEarthMap \
    --noise_type mixed \
    --img_size 512 \
    --batch_size 4 \
    --epochs 100 \
    --patience 15
```

| Config | Value |
|--------|-------|
| Encoder | ResNet-34 (pretrained, modified 11ch input) |
| Input | concat(RGB[3ch], noisy_onehot[8ch]) = 11ch |
| Output | 8ch softmax |
| Optimizer | AdamW, lr=1e-4, wd=1e-4 |
| Scheduler | CosineAnnealing, T_max=100 |
| Loss | CE + Dice + Boundary |
| Noise | Mixed (5-30%) |

#### EXP 2: U-Net EfficientNet-B4 (20.2M params)

```bash
python3 src/train_dae.py \
    --model unet_effnet \
    --data_root data/OpenEarthMap \
    --noise_type mixed \
    --img_size 512 \
    --batch_size 4 \
    --epochs 100 \
    --patience 15
```

#### EXP 3: Lightweight DAE (12.8M params) ⭐ Best

```bash
python3 src/train_dae.py \
    --model lightweight \
    --data_root data/OpenEarthMap \
    --noise_type mixed \
    --img_size 512 \
    --batch_size 8 \
    --epochs 100 \
    --patience 15
```

> **Checkpoint:** `checkpoints/dae_lightweight_mixed_20260208_122512_best.pth`
> **Training log:** `results/logs/dae_lightweight_mixed_20260208_122512_history.json`

### 3.3 Conditional DAE (39.1M params)

```bash
# Chạy riêng vì có cấu hình khác (đọc từ config YAML)
python3 src/train_dae.py \
    --config configs/dae_conditional.yaml
# Hoặc qua script:
python3 src/train_dae.py \
    --model conditional \
    --data_root data/OpenEarthMap \
    --noise_type mixed \
    --img_size 512 \
    --batch_size 4 \
    --epochs 100 \
    --patience 15
```

> **Checkpoint:** `checkpoints/dae_conditional_mixed_20260208_153934_best.pth`
> **Training log:** `results/train_conditional.log` (~287 MB)
> **History:** `results/logs/dae_conditional_mixed_20260208_153934_history.json`

---

## Phase 4: Huấn Luyện Diffusion Model

### 4.1 Train Conditional Diffusion Denoiser (22.2M params)

```bash
python3 src/train_diffusion.py \
    --data_root data/OpenEarthMap \
    --img_size 512 \
    --batch_size 4 \
    --epochs 50 \
    --T 1000 \
    --base_dim 64 \
    --patience 20 \
    --val_every 5 \
    --denoise_steps 50
```

| Config | Value |
|--------|-------|
| T (timesteps) | 1000 |
| Base dim | 64 |
| Dim mults | (1, 2, 4, 8) |
| β schedule | linear 0.0001 → 0.02 |
| Noise rate | Mixed (10-25%) |
| Optimizer | AdamW, lr=2e-4, wd=1e-4 |
| Scheduler | CosineAnnealing |
| Denoise steps (inference) | 50 (DDPM) |

> **Checkpoint:** `checkpoints/diffusion_T1000_dim64_20260209_060507_best.pth`
> **Training log:** `results/train_diffusion_50ep.log` (~128 MB)
> **History:** `results/logs/diffusion_T1000_dim64_20260209_060507_history.json`

### 4.2 Các lần resume training (gặp lỗi, phải restart)

```bash
# Log các lần resume:
# results/logs/diffusion_restart_20260208_021448.log
# results/logs/diffusion_restart_v2.log
# results/logs/diffusion_resume.log
# results/logs/diffusion_resume2.log
# results/logs/diffusion_resume_e20.log
# results/logs/diffusion_resume_e20_v2.log
```

---

## Phase 5: Đánh Giá Mô Hình

### 5.1 Đánh giá DAE (Lightweight — Best model)

```bash
cd ~/thietkedenoiser
source venv/bin/activate

python3 src/evaluate_dae.py \
    --checkpoint checkpoints/dae_lightweight_mixed_20260208_122512_best.pth \
    --model lightweight \
    --data_root data/OpenEarthMap_wo_xBD \
    --output_dir results/metrics \
    --img_size 512 \
    --batch_size 4
# Output: results/metrics/dae_evaluation.json
# Đánh giá 5 noise types × 5 noise rates = 25 combinations
```

### 5.2 Đánh giá Diffusion Model

```bash
python3 src/evaluate_diffusion.py \
    --checkpoint checkpoints/diffusion_T1000_dim64_20260209_060507_best.pth \
    --data_root data/OpenEarthMap_wo_xBD \
    --output_dir results/metrics \
    --img_size 512 \
    --batch_size 4 \
    --denoise_steps 50 \
    --T 1000 \
    --base_dim 64 \
    > results/eval_diffusion.log 2>&1
# Output: results/metrics/diffusion_evaluation.json
# Đánh giá 5 noise types × 5 noise rates = 25 combinations
```

---

## Phase 6: Visualization & Báo Cáo

### 6.1 Demo inference — Tạo ảnh so sánh

```bash
# Trên tập val
source venv/bin/activate
python3 src/demo_inference.py \
    --checkpoint checkpoints/dae_lightweight_mixed_20260208_122512_best.pth \
    --data_root data/OpenEarthMap_wo_xBD \
    --output_dir results/visualizations/random_test_samples \
    --model lightweight \
    --noise_type mixed \
    --noise_rates 0.10 0.20 0.30 \
    --num_samples 5 \
    --seed 42 \
    --dpi 150

# Trên tập test
python3 src/demo_inference.py \
    --checkpoint checkpoints/dae_lightweight_mixed_20260208_122512_best.pth \
    --data_root data/OpenEarthMap_wo_xBD \
    --output_dir results/visualizations/test_samples \
    --model lightweight \
    --noise_type mixed \
    --noise_rates 0.10 0.20 0.30 \
    --num_samples 5 \
    --split test \
    --seed 42 \
    --dpi 150
# Output: results/visualizations/*/demo_noise_*pct.png
```

### 6.2 Tạo biểu đồ báo cáo

```bash
# Phiên bản 1
python3 src/plot_report.py
# Output: results/report_charts/fig1_dae_training_curves.png
#         results/report_charts/fig2_diffusion_training.png
#         results/report_charts/fig3_perclass_iou.png
#         results/report_charts/fig4_model_comparison.png
#         results/report_charts/fig5_training_time.png
#         results/report_charts/fig6_lr_schedule.png

# Phiên bản 2 (cải tiến, thêm diffusion eval chart)
python3 src/plot_report_v2.py
# Output thêm: results/report_charts/fig7_diffusion_eval.png
```

---

## Phase 7: Latent Diffusion Model (Bài Báo Gốc — So Sánh)

> Đây là Latent Diffusion Model theo kiến trúc gốc của CompVis, chạy trên repo riêng (`/home/ubuntu/vuongvy/latent-diffusion`), dùng conda env `ldm`.

### 7.1 Setup Latent Diffusion

```bash
cd /home/ubuntu/vuongvy
git clone https://github.com/CompVis/latent-diffusion.git
cd latent-diffusion

# Tạo conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda env create -f environment.yaml
conda activate ldm
pip install torchmetrics==0.6.0
pip install packaging==21.3
pip install kornia

# Download pretrained VAE (KL-f8)
wget -O /tmp/kl-f8.zip https://ommer-lab.com/files/latent-diffusion/kl-f8.zip
sudo apt install -y unzip
unzip -o /tmp/kl-f8.zip -d models/first_stage_models/kl-f8/

# Symlink dữ liệu
ln -sf /home/ubuntu/vuongvy/openearthimage_extract data/earth
```

### 7.2 Train Latent Diffusion

```bash
cd /home/ubuntu/vuongvy/latent-diffusion
conda activate ldm

# Train trong tmux
tmux new -s latent-diffusion
CUDA_VISIBLE_DEVICES=0 python main.py \
    --base configs/latent-diffusion/earth-denoiser-bsr-sr.yaml \
    -t --gpus 0, --logdir logs
```

### 7.3 Evaluate Latent Diffusion

```bash
conda activate ldm
cd /home/ubuntu/vuongvy/latent-diffusion

python evaluate_denoising.py \
    --config configs/latent-diffusion/earth-denoiser-bsr-sr.yaml \
    --checkpoint logs/2026-02-08T12-17-56_earth-denoiser-bsr-sr/checkpoints/epoch=000000.ckpt \
    --output results/denoising_eval_epoch0 \
    --num_samples 20 \
    --ddim_steps 50
```

---

## Tóm Tắt Checkpoints & Kết Quả

| Model | Checkpoint | Params | Training Time |
|-------|-----------|--------|---------------|
| DAE Lightweight ⭐ | `dae_lightweight_mixed_20260208_122512_best.pth` | 12.8M | ~2-3h |
| DAE Conditional | `dae_conditional_mixed_20260208_153934_best.pth` | 39.1M | ~8-10h |
| Diffusion | `diffusion_T1000_dim64_20260209_060507_best.pth` | 22.2M | ~24h (50 epochs) |
| DAE UNet-ResNet34 | *(trained qua run_all.sh, xem log)* | 24.5M | ~3-4h |
| DAE UNet-EffNetB4 | *(trained qua run_all.sh, xem log)* | 20.2M | ~3-4h |

## Cấu Trúc Output

```
results/
├── logs/                              # Training logs & history JSON
│   ├── run_all_20260207_160648.log    # Log chạy toàn bộ DAE
│   ├── dae_*_history.json             # Training history (loss, mIoU, per-class IoU)
│   └── diffusion_*_history.json       # Diffusion training history
├── train_conditional.log              # Log DAE Conditional
├── train_diffusion_50ep.log           # Log Diffusion 50 epochs
├── eval_diffusion.log                 # Log evaluate Diffusion
├── metrics/
│   ├── dae_evaluation.json            # DAE eval: 5 noise types × 5 rates
│   └── diffusion_evaluation.json      # Diffusion eval: 5 noise types × 5 rates
├── visualizations/
│   ├── random_test_samples/           # Demo ảnh val
│   ├── test_samples/                  # Demo ảnh test
│   ├── noise_comparison.png           # So sánh noise types
│   └── error_maps.png                 # Error maps
├── NOISE_EVALUATION_REPORT.md         # Báo cáo noise
└── noise_eval_data.json               # Dữ liệu noise eval
```
