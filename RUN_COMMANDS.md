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

## Phase 3: Huấn Luyện DAE (4 Mô Hình)

### 3.1 Chạy toàn bộ 3 DAE (script tự động)

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

## Phase 4: Đánh Giá Mô Hình

### 4.1 Đánh giá DAE (Lightweight — Best model)

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

### 4.2 Demo inference — Tạo ảnh so sánh

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

### 4.2 Tạo biểu đồ báo cáo

```bash
python3 src/plot_report.py
# Output: results/report_charts/fig1_dae_training_curves.png
#         results/report_charts/fig2_perclass_iou.png
#         results/report_charts/fig3_model_comparison.png
#         results/report_charts/fig4_training_time.png
#         results/report_charts/fig5_lr_schedule.png
```

---

## Phase 5: Tóm Tắt Checkpoints & Kết Quả

| Model | Checkpoint | Params | Training Time |
|-------|-----------|--------|---------------|
| DAE Lightweight ⭐ | `dae_lightweight_mixed_20260208_122512_best.pth` | 12.8M | ~2-3h |
| DAE Conditional | `dae_conditional_mixed_20260208_153934_best.pth` | 39.1M | ~8-10h |
| DAE UNet-ResNet34 | *(trained qua run_all.sh, xem log)* | 24.5M | ~3-4h |
| DAE UNet-EffNetB4 | *(trained qua run_all.sh, xem log)* | 20.2M | ~3-4h |

## Cấu Trúc Output

```
results/
├── logs/                              # Training logs & history JSON
│   ├── run_all_20260207_160648.log    # Log chạy toàn bộ DAE
│   ├── dae_*_history.json             # Training history (loss, mIoU, per-class IoU)
│   └── train_conditional.log          # Log DAE Conditional
├── metrics/
│   └── dae_evaluation.json            # DAE eval: 5 noise types × 5 rates
├── visualizations/
│   ├── random_test_samples/           # Demo ảnh val
│   ├── test_samples/                  # Demo ảnh test
│   ├── noise_comparison.png           # So sánh noise types
│   └── error_maps.png                 # Error maps
├── NOISE_EVALUATION_REPORT.md         # Báo cáo noise
└── noise_eval_data.json               # Dữ liệu noise eval
```
