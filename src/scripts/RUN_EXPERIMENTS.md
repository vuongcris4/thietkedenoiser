# HƯỚNG DẪN CHẠY THÍ NGHIỆM

## Quick Start

### 1. Chạy tất cả 3 experiments (30 epochs mỗi model)

```bash
cd src/scripts/
./run_all_experiments.sh
```

Script sẽ chạy tuần tự:
1. **Lightweight DAE** (12.82M params) - 30 epochs
2. **UNet ResNet-34** (24.46M params) - 30 epochs
3. **UNet EfficientNet-B4** (20.23M params) - 30 epochs

### 2. Check tiến độ bằng sub-agent

```bash
# Check tất cả experiments
./check_agent.sh

# Check model cụ thể
./check_agent.sh lightweight
```

### 3. Auto-monitor (check liên tục)

```bash
# Check mỗi 5 phút
python auto_monitor.py --interval 300

# Check một lần rồi thoát
python auto_monitor.py --once
```

---

## Chạy từng experiment riêng lẻ

```bash
cd src/scripts/

# Lightweight DAE
python train_dae.py \
    --config ../configs/dae_lightweight.yaml \
    --override training.epochs=30

# UNet ResNet-34
python train_dae.py \
    --config ../configs/dae_resnet34.yaml \
    --override training.epochs=30

# UNet EfficientNet-B4
python train_dae.py \
    --config ../configs/dae_effnet.yaml \
    --override training.epochs=30
```

---

## Xem kết quả trên W&B

Tất cả metrics được log tự động lên Weights & Biases:
- Training loss, accuracy
- Validation mIoU, per-class IoU
- Inference results (val + test sets)
- Visualizations (RGB, pseudo-label, output, ground truth)
- Checkpoint artifacts

Truy cập: https://wandb.ai/{entity}/thietkedenoiser

---

## Output Structure

```
checkpoints/
├── dae_lightweight_real_YYYYMMDD_HHMMSS_best.pth
├── dae_unet_resnet34_real_YYYYMMDD_HHMMSS_best.pth
└── dae_unet_effnet_real_YYYYMMDD_HHMMSS_best.pth

results/logs/
├── dae_lightweight_real_YYYYMMDD_HHMMSS_history.json
├── dae_unet_resnet34_real_YYYYMMDD_HHMMSS_history.json
└── dae_unet_effnet_real_YYYYMMDD_HHMMSS_history.json
```

---

## Expected Results (tham khảo)

| Model | Params | Expected mIoU (30 epochs) |
|-------|--------|---------------------------|
| lightweight | 12.82M | ~95-97% |
| unet_resnet34 | 24.46M | ~92-95% |
| unet_effnet | 20.23M | ~94-96% |

*3 models, all using later fusion dual-branch architecture.*
