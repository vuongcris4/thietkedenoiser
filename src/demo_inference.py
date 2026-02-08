"""Demo: visualize DAE inference on test samples."""
import os, sys, torch, numpy as np
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = '/home/ubuntu/checkpoints/dae_lightweight_mixed_20260207_200336_best.pth'
    data_root = '/home/ubuntu/thietkedenoiser/data/OpenEarthMap_wo_xBD'
    out_dir = '/home/ubuntu/thietkedenoiser/results/visualizations/demo'
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    model = build_model('lightweight').to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f'Loaded Lightweight DAE (best mIoU: {ckpt.get("best_miou", "?")})')

    # Dataset with mixed noise at different rates
    noise_rates = [0.10, 0.20, 0.30]
    num_samples = 4

    for nr in noise_rates:
        dataset = DAEDataset(
            data_root, split='val', img_size=512,
            noise_type='mixed',
            noise_rate_range=(nr, nr),
            augment=False
        )
        indices = np.random.RandomState(42).choice(len(dataset), num_samples, replace=False)

        fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
        fig.suptitle(f'Lightweight DAE - Mixed Noise Rate: {nr:.0%}', fontsize=16, fontweight='bold')

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
        fname = f'{out_dir}/demo_noise_{int(nr*100)}pct.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {fname}')

    print('Done!')

if __name__ == '__main__':
    main()
