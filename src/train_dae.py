"""
MỤC ĐÍCH FILE: Huấn luyện mô hình Denoising AutoEncoder (DAE).

File này chứa toàn bộ pipeline huấn luyện DAE để khử nhiễu pseudo-label
trong bài toán segmentation ảnh vệ tinh (OpenEarthMap). Quy trình bao gồm:
  1. Đọc cấu hình từ file YAML (model, noise, loss, training hyperparams).
  2. Tạo dataset (ảnh + label nhiễu) và DataLoader cho train/val.
  3. Khởi tạo model, loss function (CE + Dice + Boundary), optimizer (AdamW),
     scheduler (CosineAnnealing), và GradScaler (mixed precision - FP16).
  4. Vòng lặp huấn luyện theo epoch: train → validate → ghi log → lưu checkpoint tốt nhất.
  5. Hỗ trợ resume từ checkpoint và early stopping.

Usage:
    python train_dae.py --config ../configs/dae_resnet34.yaml
    python train_dae.py --config ../configs/dae_lightweight.yaml --override training.lr=0.001
    python train_dae.py --config ../configs/dae_effnet.yaml --resume ../checkpoints/best.pth
"""
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

sys.path.insert(0, os.path.dirname(__file__))
from config import load_config_from_args, print_config, cfg_to_flat
from dae_model import build_model, count_params, DAELoss
from dataset import RealNoiseDAEDataset, NUM_CLASSES
from noise_generator import compute_iou, CLASS_NAMES


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """
    Huấn luyện model qua 1 epoch.

    Chức năng:
      - Duyệt qua toàn bộ batch trong train_loader.
      - Với mỗi batch: forward pass (mixed precision FP16) → tính loss → backward → cập nhật trọng số.
      - Tích lũy loss và pixel accuracy để theo dõi tiến trình.
      - In log mỗi 20 batch.

    Args:
        model: Mô hình DAE.
        loader: DataLoader tập train (trả về cặp inputs=label_nhiễu, targets=label_sạch).
        criterion: Hàm loss (DAELoss = CE + Dice + Boundary).
        optimizer: Bộ tối ưu (AdamW).
        scaler: GradScaler cho mixed precision training.
        device: 'cuda' hoặc 'cpu'.

    Returns:
        avg_loss (float): Loss trung bình trên toàn epoch.
        avg_acc (float): Pixel accuracy trung bình trên toàn epoch.
    """
    model.train()
    total_loss = 0
    total_correct = 0
    total_pixels = 0

    for batch_idx, (rgb, noisy_label, targets) in enumerate(loader):
        rgb = rgb.to(device)
        noisy_label = noisy_label.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(rgb, noisy_label)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * rgb.size(0)
        pred = outputs.argmax(dim=1)
        total_correct += (pred == targets).sum().item()
        total_pixels += targets.numel()

        if (batch_idx + 1) % 20 == 0:
            print(f'  Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f} | Acc: {total_correct/total_pixels:.4f}')

    avg_loss = total_loss / len(loader.dataset)
    avg_acc = total_correct / total_pixels
    return avg_loss, avg_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Đánh giá model trên tập validation (không tính gradient).

    Chức năng:
      - Duyệt qua toàn bộ batch trong val_loader.
      - Forward pass (FP16) → tính loss → thu thập prediction và ground truth.
      - Tính IoU (Intersection over Union) cho từng class.
      - Tính mIoU (mean IoU) trung bình trên tất cả các class.

    Args:
        model: Mô hình DAE.
        loader: DataLoader tập validation.
        criterion: Hàm loss (DAELoss).
        device: 'cuda' hoặc 'cpu'.

    Returns:
        avg_loss (float): Loss trung bình trên tập val.
        miou (float): Mean IoU trung bình các class.
        ious (dict): IoU riêng cho từng class, ví dụ {'Bareland': 0.85, 'Forest': 0.92, ...}.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    for rgb, noisy_label, targets in loader:
        rgb = rgb.to(device)
        noisy_label = noisy_label.to(device)
        targets = targets.to(device)

        with autocast():
            outputs = model(rgb, noisy_label)
            loss = criterion(outputs, targets)

        total_loss += loss.item() * rgb.size(0)
        pred = outputs.argmax(dim=1)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Compute mIoU
    ious = {}
    for c in range(NUM_CLASSES):
        inter = ((all_preds == c) & (all_targets == c)).sum()
        union = ((all_preds == c) | (all_targets == c)).sum()
        if union > 0:
            ious[CLASS_NAMES[c]] = float(inter / union)
    miou = np.mean(list(ious.values())) if ious else 0.0

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, miou, ious


def main():
    """
    Hàm chính điều phối toàn bộ quá trình huấn luyện DAE.

    Chức năng theo thứ tự:
      1. Đọc cấu hình: load file YAML (model, training, loss, ...) qua load_config_from_args().
      2. Thiết lập môi trường: seed, device (auto-detect GPU), tạo thư mục lưu checkpoint/log.
      3. Tạo dataset & dataloader: RealNoiseDAEDataset cho train/val với pseudo-label thật từ CISC-R.
      4. Khởi tạo model: build_model() theo tên model trong config (resnet34, lightweight, effnet, ...).
      5. Thiết lập training:
         - Loss: DAELoss (kết hợp CrossEntropy + Dice + Boundary loss với trọng số từ config).
         - Optimizer: AdamW với learning rate và weight decay từ config.
         - Scheduler: CosineAnnealingLR giảm dần LR theo cosine.
         - Scaler: GradScaler cho mixed precision (FP16).
      6. Resume (tùy chọn): nạp lại model, optimizer, epoch, best_miou từ checkpoint.
      7. Vòng lặp huấn luyện:
         - Mỗi epoch: train_one_epoch() → validate() → scheduler.step().
         - Ghi log (loss, acc, mIoU, IoU từng class, LR, thời gian).
         - Lưu checkpoint khi mIoU cải thiện.
         - Early stopping nếu mIoU không cải thiện sau 'patience' epoch.
      8. Lưu lịch sử training ra file JSON.
    """
    # Load config from YAML
    cfg = load_config_from_args()
    print_config(cfg, "DAE Training Configuration")

    # Extract config values
    model_name = cfg["model"]["name"]
    data_root = cfg.get("data_root", "../data/OpenEarthMap")
    pseudo_root = cfg.get("pseudo_root", None)  # Đường dẫn pseudo-labels từ CISC-R
    img_size = cfg.get("img_size", 512)
    seed = cfg.get("seed", 42)
    num_workers = cfg.get("num_workers", 2)
    save_dir = cfg.get("save_dir", "../checkpoints")
    log_dir = cfg.get("log_dir", "../results/logs")

    train_cfg = cfg.get("training", {})
    batch_size = train_cfg.get("batch_size", 4)
    epochs = train_cfg.get("epochs", 100)
    lr = train_cfg.get("lr", 1e-4)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    patience = train_cfg.get("patience", 15)

    loss_cfg = cfg.get("loss", {})
    ce_weight = loss_cfg.get("ce_weight", 1.0)
    dice_weight = loss_cfg.get("dice_weight", 1.0)
    boundary_weight = loss_cfg.get("boundary_weight", 0.5)

    augment = cfg.get("augment", True)
    resume_path = cfg.get("resume", None)

    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_str = cfg.get("device", "auto")
    if device_str == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    exp_name = f'dae_{model_name}_real_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    print(f'\n{"="*60}')
    print(f'Experiment: {exp_name}')
    print(f'Model: {model_name}')
    print(f'Dataset: REAL pseudo-labels from CISC-R')
    print(f'Device: {device}')
    print(f'{"="*60}\n')

    # --- W&B Init ---
    wandb_cfg = cfg.get("wandb", {})
    use_wandb = HAS_WANDB and wandb_cfg.get("enabled", True)
    if use_wandb:
        wandb.init(
            project=wandb_cfg.get("project", "thietkedenoiser"),
            entity=wandb_cfg.get("entity", None),
            name=exp_name,
            config=cfg_to_flat(cfg),
            tags=["dae", model_name, "real_pseudo"],
            reinit=True,
        )
        print(f'W&B run: {wandb.run.url}')
    else:
        print('W&B disabled or not installed.')

    # Data
    if not pseudo_root or not os.path.isdir(pseudo_root):
        raise ValueError(
            f'pseudo_root phải được cấu hình và tồn tại!\n'
            f'Hiện tại: pseudo_root={pseudo_root}\n'
            f'\n'
            f'Để tạo dataset pseudo-label, chạy:\n'
            f'  python src/reorganize_pseudo_dataset.py\n'
        )

    # Dùng pseudo-label thật từ CISC-R
    # pseudo_root phải có: images/, labels/, train.txt, val.txt
    print(f'Using REAL pseudo-labels from: {pseudo_root}')
    train_dataset = RealNoiseDAEDataset(
        pseudo_root, data_root=data_root, split='train',
        img_size=img_size, augment=augment
    )
    val_dataset = RealNoiseDAEDataset(
        pseudo_root, data_root=data_root, split='val',
        img_size=img_size, augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)

    # Model (Later Fusion: nhận rgb + label riêng biệt)
    model = build_model(
        model_name,
        num_classes=cfg["model"].get("num_classes", NUM_CLASSES),
    ).to(device)
    total_p, train_p = count_params(model)
    print(f'Model params: {total_p/1e6:.1f}M total, {train_p/1e6:.1f}M trainable')
    if use_wandb:
        wandb.config.update({
            'total_params': total_p,
            'trainable_params': train_p,
            'total_params_M': round(total_p / 1e6, 2),
        })

    # Training setup
    criterion = DAELoss(ce_weight=ce_weight, dice_weight=dice_weight,
                        boundary_weight=boundary_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    start_epoch = 1
    best_miou = 0

    # Resume from checkpoint
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        if 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_miou = ckpt.get('best_miou', 0)
        # Fast-forward scheduler
        for _ in range(start_epoch - 1):
            scheduler.step()
        print(f'Resumed from {resume_path} (epoch {ckpt["epoch"]}, best_miou={best_miou:.4f})')

    # Training loop
    patience_counter = 0
    history = []

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_miou, val_ious = validate(model, val_loader, criterion, device)

        scheduler.step()
        dt = time.time() - t0

        # Log
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_miou': val_miou,
            'val_ious': val_ious,
            'lr': optimizer.param_groups[0]['lr'],
            'time': dt,
        }
        history.append(log_entry)

        # W&B log
        if use_wandb:
            wandb_log = {
                'epoch': epoch,
                'train/loss': train_loss,
                'train/acc': train_acc,
                'val/loss': val_loss,
                'val/mIoU': val_miou,
                'lr': optimizer.param_groups[0]['lr'],
                'epoch_time': dt,
            }
            for cls_name, iou_val in val_ious.items():
                wandb_log[f'val/IoU_{cls_name}'] = iou_val
            wandb.log(wandb_log)

        # Print
        ious_str = ' '.join([f'{k[:4]}:{v:.3f}' for k, v in val_ious.items()])
        print(f'Epoch {epoch:3d}/{epochs} | '
              f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} mIoU: {val_miou:.4f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f} | {dt:.1f}s')
        print(f'  IoU: {ious_str}')

        # Save best
        if val_miou > best_miou:
            best_miou = val_miou
            patience_counter = 0
            ckpt_path = os.path.join(save_dir, f'{exp_name}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_miou': best_miou,
                'val_ious': val_ious,
                'config': cfg,
            }, ckpt_path)
            print(f'  >>> New best mIoU: {best_miou:.4f}, saved to {ckpt_path}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'  Early stopping at epoch {epoch} (patience={patience})')
                break

    # Save history
    history_path = os.path.join(log_dir, f'{exp_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f'\n{"="*60}')
    print(f'Training complete!')
    print(f'Best mIoU: {best_miou:.4f}')
    print(f'History saved to: {history_path}')
    print(f'{"="*60}')

    # --- W&B: Log inference samples into this run ---
    if use_wandb:
        wandb.log({'best_miou': best_miou})

        # Upload checkpoint as artifact
        ckpt_path = os.path.join(save_dir, f'{exp_name}_best.pth')
        if os.path.exists(ckpt_path):
            print('Uploading checkpoint to W&B...')
            artifact = wandb.Artifact(
                name=f'checkpoint-{model_name}',
                type='model',
                description=f'{model_name} best checkpoint (mIoU: {best_miou:.4f})',
                metadata={'model': model_name, 'best_miou': best_miou,
                          'epoch': epoch, 'data_source': 'real_pseudo_cisc_r'}
            )
            artifact.add_file(ckpt_path)
            wandb.log_artifact(artifact)
            print(f'  Uploaded: {ckpt_path}')

        print('Generating inference table for W&B...')
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            COLORS = np.array([
                [128, 0, 0], [0, 255, 36], [148, 148, 148], [255, 255, 255],
                [34, 97, 38], [0, 69, 255], [75, 181, 73], [222, 31, 7],
            ], dtype=np.uint8)

            # Log class legend
            fig_legend, ax_legend = plt.subplots(figsize=(4, 3))
            ax_legend.axis('off')
            ax_legend.set_title('Class Legend', fontsize=14, fontweight='bold', pad=10)
            patches = [mpatches.Patch(color=COLORS[c]/255., label=CLASS_NAMES[c])
                       for c in range(NUM_CLASSES)]
            ax_legend.legend(handles=patches, loc='center', fontsize=11, frameon=False,
                             ncol=2, columnspacing=1.5, handlelength=2, handleheight=1.5)
            fig_legend.tight_layout()
            wandb.log({"class_legend": wandb.Image(fig_legend, caption="Class Color Legend")})
            plt.close(fig_legend)

            def _label_to_rgb(label):
                h, w = label.shape
                rgb = np.zeros((h, w, 3), dtype=np.uint8)
                for c in range(NUM_CLASSES):
                    rgb[label == c] = COLORS[c]
                return rgb

            # Load best model for inference
            best_ckpt_path = os.path.join(save_dir, f'{exp_name}_best.pth')
            if os.path.exists(best_ckpt_path):
                best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(best_ckpt['model_state'])
            model.eval()

            # Build W&B Table with real pseudo-labels
            columns = ["sample", "rgb", "noisy_label", "dae_output",
                        "ground_truth", "noisy_mIoU", "dae_mIoU", "improvement"]
            table = wandb.Table(columns=columns)

            n_samples = 10
            infer_dataset = RealNoiseDAEDataset(
                pseudo_root, data_root=data_root, split='val',
                img_size=img_size, augment=False
            )
            indices = np.random.RandomState(seed).choice(
                len(infer_dataset), min(n_samples, len(infer_dataset)), replace=False
            )

            for i, idx in enumerate(indices):
                rgb_t, noisy_onehot, clean_label = infer_dataset[idx]
                rgb_inp = rgb_t.unsqueeze(0).to(device)
                label_inp = noisy_onehot.unsqueeze(0).to(device)
                with torch.no_grad():
                    with autocast():
                        output = model(rgb_inp, label_inp)
                pred = output.argmax(dim=1).squeeze(0).cpu().numpy()

                rgb_img = rgb_t.permute(1, 2, 0).numpy()
                rgb_img = ((rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-6) * 255).astype(np.uint8)
                noisy_label = noisy_onehot.argmax(dim=0).numpy()
                clean_np = clean_label.numpy()

                noisy_iou = compute_iou(noisy_label, clean_np)
                dae_iou = compute_iou(pred, clean_np)

                table.add_data(
                    f"sample_{i}",
                    wandb.Image(rgb_img),
                    wandb.Image(_label_to_rgb(noisy_label)),
                    wandb.Image(_label_to_rgb(pred)),
                    wandb.Image(_label_to_rgb(clean_np)),
                    round(noisy_iou["mIoU"], 4),
                    round(dae_iou["mIoU"], 4),
                    round(dae_iou["mIoU"] - noisy_iou["mIoU"], 4),
                )

            wandb.log({"inference_results": table})
            print(f'  Logged inference table ({len(table.data)} rows) to W&B')

        except Exception as e:
            print(f'  Warning: inference visualization failed: {e}')

        wandb.finish()


if __name__ == '__main__':
    main()
