"""Demo: visualize DAE inference on REAL pseudo-labels from CISC-R."""
import argparse, os, sys, torch, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.cuda.amp import autocast

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

sys.path.insert(0, os.path.dirname(__file__))
from dae_model import build_model, NUM_CLASSES
from dataset import RealNoiseDAEDataset
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
    parser = argparse.ArgumentParser(
        description='Demo: visualize DAE inference on REAL pseudo-labels from CISC-R'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--pseudo_root', type=str, required=True,
                        help='Path to pseudo-label dataset root (e.g., data/OEM_v2_aDanh)')
    parser.add_argument('--output_dir', type=str, default='results/visualizations/demo_real_latest',
                        help='Output directory for visualizations (default: results/visualizations/demo_real_latest)')
    parser.add_argument('--model', type=str, default='lightweight',
                        choices=['lightweight', 'unet_resnet34', 'unet_effnet', 'conditional'],
                        help='Model architecture (default: lightweight)')
    parser.add_argument('--num_samples', type=int, default=12,
                        help='Number of samples to visualize (default: 12)')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Image size (default: 512)')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Dataset split (default: val)')
    parser.add_argument('--seed', type=int, default=2026,
                        help='Random seed for sample selection (default: 2026)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for saved figures (default: 150)')
    parser.add_argument('--wandb_run_id', type=str, default=None,
                        help='W&B run ID to log results (optional)')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # W&B init
    use_wandb = HAS_WANDB
    if use_wandb:
        wandb.init(
            project="thietkedenoiser",
            name=f"infer_real_{args.model}_{args.split}",
            job_type="inference",
            tags=["inference", args.model, "real_pseudo_labels", args.split],
            config=vars(args),
            reinit=True,
        )
        print(f'W&B run: {wandb.run.url}')

    # Load model
    model = build_model(args.model).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f'Loaded {args.model} DAE (best mIoU: {ckpt.get("best_miou", "?")})')

    # Load dataset with REAL pseudo-labels
    dataset = RealNoiseDAEDataset(
        pseudo_root=args.pseudo_root,
        split=args.split,
        img_size=args.img_size,
        augment=False
    )

    # Select random samples
    num_samples = min(args.num_samples, len(dataset))
    indices = np.random.RandomState(args.seed).choice(
        len(dataset), num_samples, replace=False
    )

    # Prepare results table for W&B
    wandb_table = wandb.Table(
        columns=["sample_id", "rgb", "pseudo_label", "dae_output", "ground_truth",
                 "pseudo_mIoU", "dae_mIoU", "improvement"]
    )

    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(
        f'{args.model.replace("_", " ").title()} DAE - REAL Pseudo-Labels from CISC-R',
        fontsize=16, fontweight='bold'
    )

    all_improvements = []

    for i, idx in enumerate(indices):
        rgb_t, pseudo_onehot, clean_label = dataset[idx]
        rgb_inp = rgb_t.unsqueeze(0).to(device)
        label_inp = pseudo_onehot.unsqueeze(0).to(device)

        with torch.no_grad():
            with autocast():
                output = model(rgb_inp, label_inp)
        pred = output.argmax(dim=1).squeeze(0).cpu().numpy()

        # Extract components
        rgb_img = rgb_t.permute(1, 2, 0).numpy()
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-6)
        pseudo_label = pseudo_onehot.argmax(dim=0).numpy()
        clean_np = clean_label.numpy()

        # Compute IoU
        pseudo_iou = compute_iou(pseudo_label, clean_np)
        dae_iou = compute_iou(pred, clean_np)
        improvement = dae_iou["mIoU"] - pseudo_iou["mIoU"]
        all_improvements.append(improvement)

        # Plot
        axes[i, 0].imshow(rgb_img)
        axes[i, 0].set_title('RGB Image', fontsize=12)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(label_to_rgb(pseudo_label))
        axes[i, 1].set_title(f'Pseudo-Label (CISC-R)\nmIoU: {pseudo_iou["mIoU"]:.3f}', fontsize=12)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(label_to_rgb(pred))
        axes[i, 2].set_title(f'DAE Output\nmIoU: {dae_iou["mIoU"]:.3f}', fontsize=12)
        axes[i, 2].axis('off')

        axes[i, 3].imshow(label_to_rgb(clean_np))
        axes[i, 3].set_title('Ground Truth (OEM)', fontsize=12)
        axes[i, 3].axis('off')

        # Add to W&B table
        wandb_table.add_data(
            f"sample_{i}",
            wandb.Image(rgb_img),
            wandb.Image(label_to_rgb(pseudo_label)),
            wandb.Image(label_to_rgb(pred)),
            wandb.Image(label_to_rgb(clean_np)),
            round(pseudo_iou["mIoU"], 4),
            round(dae_iou["mIoU"], 4),
            round(improvement, 4)
        )

    # Legend
    patches = [mpatches.Patch(color=COLORS[c]/255., label=CLASS_NAMES[c]) for c in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc='lower center', ncol=8, fontsize=10, frameon=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fname = f'{args.output_dir}/demo_real_pseudo_labels.png'
    plt.savefig(fname, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fname}')

    # Log to W&B
    if use_wandb:
        wandb.log({
            "inference_results_real": wandb.Image(fname,
                caption=f"{args.model} | Real pseudo-labels from CISC-R"),
            "inference_table_real": wandb_table,
            "mean_improvement": np.mean(all_improvements),
            "std_improvement": np.std(all_improvements),
        })
        print(f'Mean IoU improvement: {np.mean(all_improvements):.4f} (+/- {np.std(all_improvements):.4f})')

    if use_wandb:
        wandb.finish()
    print('Done!')

if __name__ == '__main__':
    main()
