"""
Upload inference images + checkpoint to an existing W&B run.

Usage:
    python3 src/upload_to_wandb_run.py \
        --run_id vrylq49n \
        --checkpoint checkpoints/dae_unet_resnet34_mixed_20260319_022703_best.pth \
        --model unet_resnet34 \
        --data_root data/OpenEarthMap
"""
import argparse, os, sys, torch, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
import wandb

sys.path.insert(0, os.path.dirname(__file__))
from dae_model import build_model, NUM_CLASSES
from dataset import DAEDataset
from noise_generator import CLASS_NAMES, compute_iou

COLORS = np.array([
    [128, 0, 0], [0, 255, 36], [148, 148, 148], [255, 255, 255],
    [34, 97, 38], [0, 69, 255], [75, 181, 73], [222, 31, 7],
], dtype=np.uint8)

def label_to_rgb(label):
    h, w = label.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        rgb[label == c] = COLORS[c]
    return rgb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', required=True, help='W&B run ID to resume')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--model', required=True, choices=['lightweight', 'unet_resnet34', 'unet_effnet', 'conditional'])
    parser.add_argument('--data_root', default='data/OpenEarthMap')
    parser.add_argument('--project', default='thietkedenoiser')
    parser.add_argument('--entity', default='vuongcris4-hcmute')
    parser.add_argument('--noise_type', default='mixed')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--noise_rates', type=float, nargs='+', default=[0.10, 0.20, 0.30])
    parser.add_argument('--n_samples', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Resume existing W&B run
    wandb.init(
        project=args.project,
        entity=args.entity,
        id=args.run_id,
        resume="must",
    )
    print(f'Resumed W&B run: {wandb.run.url}')

    # Load model
    model = build_model(args.model).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f'Loaded {args.model} (best mIoU: {ckpt.get("best_miou", "?")})')

    # --- Upload checkpoint as artifact ---
    print('Uploading checkpoint to W&B...')
    artifact = wandb.Artifact(
        name=f'checkpoint-{args.model}',
        type='model',
        description=f'{args.model} best checkpoint (mIoU: {ckpt.get("best_miou", "?")})',
        metadata={
            'model': args.model,
            'best_miou': ckpt.get('best_miou'),
            'epoch': ckpt.get('epoch'),
        }
    )
    artifact.add_file(args.checkpoint)
    wandb.log_artifact(artifact)
    print(f'  Uploaded: {args.checkpoint}')

    # --- Generate & log inference images ---
    print('Generating inference images...')
    for nr in args.noise_rates:
        dataset = DAEDataset(
            args.data_root, split='val', img_size=args.img_size,
            noise_type=args.noise_type,
            noise_rate_range=(nr, nr),
            augment=False
        )
        indices = np.random.RandomState(args.seed).choice(
            len(dataset), min(args.n_samples, len(dataset)), replace=False
        )

        fig, axes = plt.subplots(args.n_samples, 4, figsize=(20, 5 * args.n_samples))
        fig.suptitle(f'{args.model} | {args.noise_type} noise {nr:.0%}', fontsize=16, fontweight='bold')

        for i, idx in enumerate(indices):
            dae_input, clean_label = dataset[idx]
            inp = dae_input.unsqueeze(0).to(device)
            with torch.no_grad():
                with autocast():
                    output = model(inp)
            pred = output.argmax(dim=1).squeeze(0).cpu().numpy()

            rgb_img = dae_input[:3].permute(1, 2, 0).numpy()
            rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-6)
            noisy_label = dae_input[3:].argmax(dim=0).numpy()
            clean_np = clean_label.numpy()

            noisy_iou = compute_iou(noisy_label, clean_np)
            dae_iou = compute_iou(pred, clean_np)

            axes[i, 0].imshow(rgb_img); axes[i, 0].set_title('RGB'); axes[i, 0].axis('off')
            axes[i, 1].imshow(label_to_rgb(noisy_label)); axes[i, 1].set_title(f'Noisy\nmIoU: {noisy_iou["mIoU"]:.3f}'); axes[i, 1].axis('off')
            axes[i, 2].imshow(label_to_rgb(pred)); axes[i, 2].set_title(f'DAE Output\nmIoU: {dae_iou["mIoU"]:.3f}'); axes[i, 2].axis('off')
            axes[i, 3].imshow(label_to_rgb(clean_np)); axes[i, 3].set_title('Ground Truth'); axes[i, 3].axis('off')

        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        wandb.log({f'inference/noise_{int(nr*100)}pct': wandb.Image(fig,
            caption=f'{args.model} | {args.noise_type} noise {nr:.0%}')})
        plt.close()
        print(f'  Logged inference noise_{int(nr*100)}pct')

    wandb.finish()
    print('Done!')

if __name__ == '__main__':
    main()
