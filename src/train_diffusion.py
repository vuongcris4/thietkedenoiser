"""
Train Conditional Diffusion Denoiser.

Consolidated script: supports both fresh training and resuming.

Usage:
    python train_diffusion.py --config ../configs/diffusion.yaml
    python train_diffusion.py --config ../configs/diffusion.yaml --resume ../checkpoints/best.pth
    python train_diffusion.py --config ../configs/diffusion.yaml --override training.lr=0.001
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from config import load_config_from_args, print_config
from diffusion_denoiser import DiffusionDenoiser, count_params
from dataset import OpenEarthMapDataset, NUM_CLASSES
from noise_generator import NoiseGenerator, compute_iou, CLASS_NAMES


class DiffusionDAEDataset(torch.utils.data.Dataset):
    """Dataset wrapper that provides (rgb, clean_label_onehot, label_idx) for diffusion training."""
    def __init__(self, data_root, split='train', img_size=512, augment=True):
        self.base = OpenEarthMapDataset(data_root, split, img_size, augment)
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        img, label = self.base[idx]
        label_onehot = F.one_hot(label.clamp(0, NUM_CLASSES-1), NUM_CLASSES)
        label_onehot = label_onehot.permute(2, 0, 1).float()
        return img, label_onehot, label


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
            print(f"  Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}")
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, noise_gen, device, noise_cfg, denoise_steps=50):
    model.eval()
    noisy_ious, clean_ious = [], []
    rate_min = noise_cfg.get("rate_min", 0.10)
    rate_max = noise_cfg.get("rate_max", 0.25)

    for rgb, label_onehot, label_idx in loader:
        rgb = rgb.to(device)
        label_onehot = label_onehot.to(device)
        label_np = label_idx.numpy()
        B = rgb.shape[0]
        for b in range(B):
            clean_np = label_np[b]
            noise_rate = np.random.uniform(rate_min, rate_max)
            noisy_np = noise_gen.mixed_noise(clean_np.astype(np.int32), noise_rate=noise_rate)
            noisy_iou = compute_iou(noisy_np, clean_np)
            noisy_ious.append(noisy_iou["mIoU"])
            noisy_onehot = F.one_hot(torch.from_numpy(noisy_np).long().clamp(0, NUM_CLASSES-1),
                                     NUM_CLASSES).permute(2, 0, 1).float().unsqueeze(0).to(device)
            rgb_single = rgb[b:b+1]
            with autocast():
                pred_class = model.get_clean_prediction(noisy_onehot, rgb_single, num_steps=denoise_steps)
            pred_np = pred_class.cpu().numpy()[0]
            clean_iou = compute_iou(pred_np, clean_np)
            clean_ious.append(clean_iou["mIoU"])
    avg_noisy = np.mean(noisy_ious)
    avg_clean = np.mean(clean_ious)
    return avg_noisy, avg_clean, avg_clean - avg_noisy


def main():
    # Load config from YAML
    cfg = load_config_from_args()
    print_config(cfg, "Diffusion Training Configuration")

    # Extract config values
    model_cfg = cfg.get("model", {})
    data_root = cfg.get("data_root", "../data/OpenEarthMap")
    img_size = cfg.get("img_size", 512)
    seed = cfg.get("seed", 42)
    num_workers = cfg.get("num_workers", 2)
    save_dir = cfg.get("save_dir", "../checkpoints")
    log_dir = cfg.get("log_dir", "../results/logs")

    train_cfg = cfg.get("training", {})
    batch_size = train_cfg.get("batch_size", 4)
    epochs = train_cfg.get("epochs", 100)
    lr = train_cfg.get("lr", 2e-4)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    patience = train_cfg.get("patience", 20)
    val_every = train_cfg.get("val_every", 5)

    T = model_cfg.get("T", 1000)
    base_dim = model_cfg.get("base_dim", 64)
    dim_mults = tuple(model_cfg.get("dim_mults", [1, 2, 4, 8]))

    infer_cfg = cfg.get("inference", {})
    denoise_steps = infer_cfg.get("denoise_steps", 50)

    noise_cfg = cfg.get("noise", {})
    noise_gen_seed = noise_cfg.get("noise_gen_seed", 99)

    augment = cfg.get("augment", True)
    resume_path = cfg.get("resume", None)

    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_str = cfg.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    exp_name = f"diffusion_T{T}_dim{base_dim}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Data
    train_dataset = DiffusionDAEDataset(data_root, "train", img_size, augment=augment)
    val_dataset = DiffusionDAEDataset(data_root, "val", img_size, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    # Model
    model = DiffusionDenoiser(num_classes=NUM_CLASSES, T=T, base_dim=base_dim,
                              dim_mults=dim_mults).to(device)

    start_epoch = 1
    best_improvement = -float("inf")

    # Resume
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt["epoch"] + 1
        best_improvement = ckpt.get("best_improvement", -float("inf"))
        print(f"Resumed from {resume_path} (epoch {ckpt['epoch']}, best_improvement={best_improvement:+.4f})")

    total_p, train_p = count_params(model)
    print(f"Model: {total_p/1e6:.1f}M params | Start epoch: {start_epoch} | Device: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # Fast-forward scheduler to resume epoch
    for _ in range(start_epoch - 1):
        scheduler.step()
    scaler = GradScaler()
    noise_gen = NoiseGenerator(num_classes=NUM_CLASSES, seed=noise_gen_seed)

    patience_counter = 0
    history = []

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        scheduler.step()
        dt = time.time() - t0

        log_entry = {"epoch": epoch, "train_loss": train_loss,
                     "lr": optimizer.param_groups[0]["lr"], "time": dt}

        if epoch % val_every == 0 or epoch == start_epoch:
            noisy_miou, clean_miou, improvement = validate(
                model, val_loader, noise_gen, device, noise_cfg, denoise_steps=denoise_steps)
            log_entry.update({"noisy_miou": noisy_miou, "clean_miou": clean_miou,
                              "improvement": improvement})
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {train_loss:.4f} | "
                  f"Noisy mIoU: {noisy_miou:.4f} -> Clean mIoU: {clean_miou:.4f} "
                  f"(+{improvement:+.4f}) | LR: {optimizer.param_groups[0]['lr']:.6f} | {dt:.1f}s")
            if improvement > best_improvement:
                best_improvement = improvement
                patience_counter = 0
                ckpt_path = os.path.join(save_dir, f"{exp_name}_best.pth")
                torch.save({"epoch": epoch, "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "best_improvement": best_improvement,
                            "clean_miou": clean_miou, "config": cfg}, ckpt_path)
                print(f"  >>> New best improvement: {best_improvement:+.4f}, saved")
            else:
                patience_counter += val_every
        else:
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {train_loss:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f} | {dt:.1f}s")

        history.append(log_entry)
        # Save latest checkpoint every epoch for safe resume
        latest_path = os.path.join(save_dir, f"{exp_name}_latest.pth")
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_improvement": best_improvement,
                    "config": cfg}, latest_path)
        # Save history after each epoch (safe resume)
        with open(os.path.join(log_dir, f"{exp_name}_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"\nTraining complete! Best improvement: {best_improvement:+.4f}")


if __name__ == "__main__":
    main()
