"""
Train Conditional Diffusion Denoiser theo bai bao goc.

Quy trinh:
  B1: Lay GT label (one-hot CxHxW) lam x_0
  B2: Forward diffusion: them nhieu Gaussian -> x_t
  B3: Model predict noise epsilon voi conditioning = RGB image
  B4: Loss = MSE(epsilon_pred, epsilon)

Usage:
    python train_diffusion.py --epochs 100 --T 1000
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from diffusion_denoiser import DiffusionDenoiser, count_params
from dataset import OpenEarthMapDataset, NUM_CLASSES
from noise_generator import NoiseGenerator, compute_iou, CLASS_NAMES


class DiffusionDAEDataset(torch.utils.data.Dataset):
    """Dataset wrapper that provides (rgb, clean_label_onehot) for diffusion training."""
    def __init__(self, data_root, split='train', img_size=512, augment=True):
        self.base = OpenEarthMapDataset(data_root, split, img_size, augment)
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        img, label = self.base[idx]  # img:[3,H,W], label:[H,W] long
        # Convert label to one-hot float
        label_onehot = F.one_hot(label.clamp(0, NUM_CLASSES-1), NUM_CLASSES)
        label_onehot = label_onehot.permute(2, 0, 1).float()  # [C, H, W]
        return img, label_onehot, label  # rgb, onehot, class_idx


def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (rgb, label_onehot, _) in enumerate(loader):
        rgb = rgb.to(device)
        label_onehot = label_onehot.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            loss = model.training_step(label_onehot, rgb)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * rgb.size(0)
        
        if (batch_idx + 1) % 20 == 0:
            print(f'  Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}')
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, noise_gen, device, num_steps=50):
    model.eval()
    noisy_ious = []
    clean_ious = []
    
    noise_funcs = {
        'mixed': noise_gen.mixed_noise,
    }
    
    for rgb, label_onehot, label_idx in loader:
        rgb = rgb.to(device)
        label_onehot = label_onehot.to(device)
        label_np = label_idx.numpy()
        
        B = rgb.shape[0]
        for b in range(B):
            # Create noisy pseudo-label
            clean_np = label_np[b]
            noise_rate = np.random.uniform(0.10, 0.25)
            noisy_np = noise_gen.mixed_noise(clean_np.astype(np.int32), noise_rate=noise_rate)
            
            # Noisy IoU (before denoising)
            noisy_iou = compute_iou(noisy_np, clean_np)
            noisy_ious.append(noisy_iou['mIoU'])
            
            # Denoise
            noisy_onehot = F.one_hot(torch.from_numpy(noisy_np).long().clamp(0, NUM_CLASSES-1),
                                     NUM_CLASSES).permute(2, 0, 1).float().unsqueeze(0).to(device)
            rgb_single = rgb[b:b+1]
            
            with autocast():
                pred_class = model.get_clean_prediction(noisy_onehot, rgb_single, num_steps=num_steps)
            
            pred_np = pred_class.cpu().numpy()[0]
            clean_iou = compute_iou(pred_np, clean_np)
            clean_ious.append(clean_iou['mIoU'])
    
    avg_noisy = np.mean(noisy_ious)
    avg_clean = np.mean(clean_ious)
    improvement = avg_clean - avg_noisy
    
    return avg_noisy, avg_clean, improvement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../data/OpenEarthMap')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--T', type=int, default=1000, help='Diffusion timesteps')
    parser.add_argument('--base_dim', type=int, default=64)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--save_dir', type=str, default='../checkpoints')
    parser.add_argument('--log_dir', type=str, default='../results/logs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--val_every', type=int, default=5, help='Validate every N epochs')
    parser.add_argument('--denoise_steps', type=int, default=50, help='Denoising steps for inference')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    exp_name = f'diffusion_T{args.T}_dim{args.base_dim}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    print(f'\n{"="*60}')
    print(f'Experiment: {exp_name}')
    print(f'Diffusion T={args.T}, base_dim={args.base_dim}')
    print(f'Device: {device}')
    print(f'{"="*60}\n')
    
    # Data
    train_dataset = DiffusionDAEDataset(args.data_root, 'train', args.img_size, augment=True)
    val_dataset = DiffusionDAEDataset(args.data_root, 'val', args.img_size, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, num_workers=1, pin_memory=True)
    
    # Model
    model = DiffusionDenoiser(
        num_classes=NUM_CLASSES, T=args.T,
        base_dim=args.base_dim, dim_mults=(1, 2, 4, 8)
    ).to(device)
    total_p, train_p = count_params(model)
    print(f'Model: {total_p/1e6:.1f}M params ({train_p/1e6:.1f}M trainable)')
    
    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    noise_gen = NoiseGenerator(num_classes=NUM_CLASSES, seed=99)
    
    best_improvement = -float('inf')
    patience_counter = 0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        scheduler.step()
        dt = time.time() - t0
        
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'lr': optimizer.param_groups[0]['lr'],
            'time': dt,
        }
        
        # Validate periodically
        if epoch % args.val_every == 0 or epoch == 1:
            noisy_miou, clean_miou, improvement = validate(
                model, val_loader, noise_gen, device, num_steps=args.denoise_steps
            )
            log_entry.update({
                'noisy_miou': noisy_miou,
                'clean_miou': clean_miou,
                'improvement': improvement,
            })
            
            print(f'Epoch {epoch:3d}/{args.epochs} | '
                  f'Loss: {train_loss:.4f} | '
                  f'Noisy mIoU: {noisy_miou:.4f} → Clean mIoU: {clean_miou:.4f} '
                  f'(+{improvement:+.4f}) | LR: {optimizer.param_groups[0]["lr"]:.6f} | {dt:.1f}s')
            
            if improvement > best_improvement:
                best_improvement = improvement
                patience_counter = 0
                ckpt_path = os.path.join(args.save_dir, f'{exp_name}_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'best_improvement': best_improvement,
                    'clean_miou': clean_miou,
                    'args': vars(args),
                }, ckpt_path)
                print(f'  >>> New best improvement: {best_improvement:+.4f}, saved')
            else:
                patience_counter += args.val_every
        else:
            print(f'Epoch {epoch:3d}/{args.epochs} | Loss: {train_loss:.4f} | '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f} | {dt:.1f}s')
        
        history.append(log_entry)
        
        if patience_counter >= args.patience:
            print(f'Early stopping at epoch {epoch}')
            break
    
    # Save history
    with open(os.path.join(args.log_dir, f'{exp_name}_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f'\nTraining complete! Best improvement: {best_improvement:+.4f}')


if __name__ == '__main__':
    main()
