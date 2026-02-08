"""
Train Denoising AutoEncoder.
Usage:
    python train_dae.py --model unet_resnet34 --noise_type mixed --epochs 100
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='unet_resnet34',
                        choices=['unet_resnet34', 'unet_effnet', 'conditional', 'lightweight'])
    parser.add_argument('--data_root', type=str, default='../data/OpenEarthMap')
    parser.add_argument('--noise_type', type=str, default='mixed',
                        choices=['random_flip', 'boundary', 'region_swap', 'confusion_based', 'mixed', 'all_random'])
    parser.add_argument('--noise_rate_min', type=float, default=0.05)
    parser.add_argument('--noise_rate_max', type=float, default=0.30)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--save_dir', type=str, default='../checkpoints')
    parser.add_argument('--log_dir', type=str, default='../results/logs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    exp_name = f'dae_{args.model}_{args.noise_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    print(f'\n{"="*60}')
    print(f'Experiment: {exp_name}')
    print(f'Model: {args.model}')
    print(f'Noise: {args.noise_type} ({args.noise_rate_min}-{args.noise_rate_max})')
    print(f'Device: {device}')
    print(f'{"="*60}\n')
    
    # Data
    train_dataset = DAEDataset(
        args.data_root, split='train', img_size=args.img_size,
        noise_type=args.noise_type,
        noise_rate_range=(args.noise_rate_min, args.noise_rate_max),
        augment=True
    )
    val_dataset = DAEDataset(
        args.data_root, split='val', img_size=args.img_size,
        noise_type=args.noise_type,
        noise_rate_range=(args.noise_rate_min, args.noise_rate_max),
        augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    
    # Model
    model = build_model(args.model).to(device)
    total_p, train_p = count_params(model)
    print(f'Model params: {total_p/1e6:.1f}M total, {train_p/1e6:.1f}M trainable')
    
    # Training setup
    criterion = DAELoss(ce_weight=1.0, dice_weight=1.0, boundary_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    
    # Training loop
    best_miou = 0
    patience_counter = 0
    history = []
    
    for epoch in range(1, args.epochs + 1):
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
        print(f'Epoch {epoch:3d}/{args.epochs} | '
              f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} mIoU: {val_miou:.4f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f} | {dt:.1f}s')
        print(f'  IoU: {ious_str}')
        
        # Save best
        if val_miou > best_miou:
            best_miou = val_miou
            patience_counter = 0
            ckpt_path = os.path.join(args.save_dir, f'{exp_name}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_miou': best_miou,
                'val_ious': val_ious,
                'args': vars(args),
            }, ckpt_path)
            print(f'  >>> New best mIoU: {best_miou:.4f}, saved to {ckpt_path}')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'  Early stopping at epoch {epoch} (patience={args.patience})')
                break
    
    # Save history
    history_path = os.path.join(args.log_dir, f'{exp_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f'\n{"="*60}')
    print(f'Training complete!')
    print(f'Best mIoU: {best_miou:.4f}')
    print(f'History saved to: {history_path}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
