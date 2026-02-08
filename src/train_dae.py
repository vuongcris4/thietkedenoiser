"""
Train Denoising AutoEncoder.

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

sys.path.insert(0, os.path.dirname(__file__))
from config import load_config_from_args, print_config
from dae_model import build_model, count_params, DAELoss
from dataset import DAEDataset, NUM_CLASSES
from noise_generator import compute_iou, CLASS_NAMES


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_pixels = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * inputs.size(0)
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
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
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
    # Load config from YAML
    cfg = load_config_from_args()
    print_config(cfg, "DAE Training Configuration")

    # Extract config values
    model_name = cfg["model"]["name"]
    data_root = cfg.get("data_root", "../data/OpenEarthMap")
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

    noise_cfg = cfg.get("noise", {})
    noise_type = noise_cfg.get("type", "mixed")
    noise_rate_min = noise_cfg.get("rate_min", 0.05)
    noise_rate_max = noise_cfg.get("rate_max", 0.30)

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

    exp_name = f'dae_{model_name}_{noise_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    print(f'\n{"="*60}')
    print(f'Experiment: {exp_name}')
    print(f'Model: {model_name}')
    print(f'Noise: {noise_type} ({noise_rate_min}-{noise_rate_max})')
    print(f'Device: {device}')
    print(f'{"="*60}\n')

    # Data
    train_dataset = DAEDataset(
        data_root, split='train', img_size=img_size,
        noise_type=noise_type,
        noise_rate_range=(noise_rate_min, noise_rate_max),
        augment=augment
    )
    val_dataset = DAEDataset(
        data_root, split='val', img_size=img_size,
        noise_type=noise_type,
        noise_rate_range=(noise_rate_min, noise_rate_max),
        augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)

    # Model
    model = build_model(model_name).to(device)
    total_p, train_p = count_params(model)
    print(f'Model params: {total_p/1e6:.1f}M total, {train_p/1e6:.1f}M trainable')

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


if __name__ == '__main__':
    main()
