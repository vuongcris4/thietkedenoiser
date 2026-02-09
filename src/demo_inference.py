"""Demo: visualize DAE inference on test samples."""
import argparse, os, sys, torch, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.cuda.amp import autocast

sys.path.insert(0, os.path.dirname(__file__))
from dae_model import build_model, NUM_CLASSES
from dataset import DAEDataset
from noise_generator import CLASS_NAMES, compute_iou

# Color palette for 8 classes
COLORS = np.array([
    [128, 0, 0],     # Bareland - dark red
    [0, 255, 36],    # Rangeland - green
    [148, 148, 148], # Developed - gray
    [255, 255, 255], # Road - white
    [34, 97, 38],    # Tree - dark green
    [0, 69, 255],    # Water - blue
    [75, 181, 73],   # Agriculture - light green
    [222, 31, 7],    # Building - red
], dtype=np.uint8)

def label_to_rgb(label):
    h, w = label.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        rgb[label == c] = COLORS[c]
    return rgb

def parse_args():
    parser = argparse.ArgumentParser(description='Demo: visualize DAE inference on test samples')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--data_root', type=str, default='data/OpenEarthMap_wo_xBD',
                        help='Path to dataset root (default: data/OpenEarthMap_wo_xBD)')
    parser.add_argument('--output_dir', type=str, default='results/visualizations/demo_latest',
                        help='Output directory for visualizations (default: results/visualizations/demo_latest)')
    parser.add_argument('--model', type=str, default='lightweight',
                        choices=['lightweight', 'unet_resnet34', 'unet_effnet', 'conditional'],
                        help='Model architecture (default: lightweight)')
    parser.add_argument('--noise_type', type=str, default='mixed',
                        choices=['random_flip', 'boundary', 'region_swap', 'confusion', 'mixed'],
                        help='Noise type to apply (default: mixed)')
    parser.add_argument('--noise_rates', type=float, nargs='+', default=[0.10, 0.20, 0.30],
                        help='Noise rates to test (default: 0.10 0.20 0.30)')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of samples per noise rate (default: 4)')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Image size (default: 512)')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Dataset split (default: val)')
    parser.add_argument('--seed', type=int, default=2026,
                        help='Random seed for sample selection (default: 2026)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for saved figures (default: 150)')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = build_model(args.model).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f'Loaded {args.model} DAE (best mIoU: {ckpt.get("best_miou", "?")})')

    for nr in args.noise_rates:
        dataset = DAEDataset(
            args.data_root, split=args.split, img_size=args.img_size,
            noise_type=args.noise_type,
            noise_rate_range=(nr, nr),
            augment=False
        )
        indices = np.random.RandomState(args.seed).choice(
            len(dataset), args.num_samples, replace=False
        )

        fig, axes = plt.subplots(args.num_samples, 4, figsize=(20, 5*args.num_samples))
        fig.suptitle(
            f'{args.model.replace("_", " ").title()} DAE - {args.noise_type.replace("_", " ").title()} Noise Rate: {nr:.0%}',
            fontsize=16, fontweight='bold'
        )

        for i, idx in enumerate(indices):
            dae_input, clean_label = dataset[idx]
            inp = dae_input.unsqueeze(0).to(device)

            with torch.no_grad():
                with autocast():
                    output = model(inp)
            pred = output.argmax(dim=1).squeeze(0).cpu().numpy()

            # Extract components
            rgb_img = dae_input[:3].permute(1, 2, 0).numpy()
            rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-6)
            noisy_label = dae_input[3:].argmax(dim=0).numpy()
            clean_np = clean_label.numpy()

            # Compute IoU
            noisy_iou = compute_iou(noisy_label, clean_np)
            dae_iou = compute_iou(pred, clean_np)

            # Plot
            axes[i, 0].imshow(rgb_img)
            axes[i, 0].set_title('RGB Image', fontsize=12)
            axes[i, 0].axis('off')

            axes[i, 1].imshow(label_to_rgb(noisy_label))
            axes[i, 1].set_title(f'Noisy Label\nmIoU: {noisy_iou["mIoU"]:.3f}', fontsize=12)
            axes[i, 1].axis('off')

            axes[i, 2].imshow(label_to_rgb(pred))
            axes[i, 2].set_title(f'DAE Output\nmIoU: {dae_iou["mIoU"]:.3f}', fontsize=12)
            axes[i, 2].axis('off')

            axes[i, 3].imshow(label_to_rgb(clean_np))
            axes[i, 3].set_title('Ground Truth', fontsize=12)
            axes[i, 3].axis('off')

        # Legend
        patches = [mpatches.Patch(color=COLORS[c]/255., label=CLASS_NAMES[c]) for c in range(NUM_CLASSES)]
        fig.legend(handles=patches, loc='lower center', ncol=8, fontsize=10, frameon=True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        fname = f'{args.output_dir}/demo_noise_{int(nr*100)}pct.png'
        plt.savefig(fname, dpi=args.dpi, bbox_inches='tight')
        plt.close()
        print(f'Saved: {fname}')

    print('Done!')

if __name__ == '__main__':
    main()
