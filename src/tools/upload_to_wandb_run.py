"""
Upload inference images (as W&B Table) + checkpoint to an existing W&B run.

Usage:
    python3 src/upload_to_wandb_run.py \
        --run_id vrylq49n \
        --checkpoint checkpoints/dae_unet_resnet34_real_20260320_134113_best.pth \
        --model unet_resnet34 \
        --pseudo_root data/OEM_v2_aDanh
"""
import argparse, os, sys, torch, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.cuda.amp import autocast
import wandb

sys.path.insert(0, os.path.dirname(__file__))
from dae_model import build_model, NUM_CLASSES
from dataset import RealNoiseDAEDataset
from noise_generator import CLASS_NAMES, compute_iou

# Color palette for 8 classes (HEX from dataset spec)
COLORS = np.array([
    [128, 0, 0],      # Bareland - #800000
    [0, 255, 36],     # Rangeland - #00FF24
    [148, 148, 148],  # Developed space - #949494
    [255, 255, 255],  # Road - #FFFFFF
    [34, 97, 38],     # Tree - #226126
    [0, 69, 255],     # Water - #0045FF
    [75, 181, 73],    # Agriculture land - #4BB549
    [222, 31, 7],     # Building - #DE1F07
], dtype=np.uint8)

def label_to_rgb(label):
    h, w = label.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        rgb[label == c] = COLORS[c]
    return rgb


def make_legend_image():
    """Create a color legend image mapping class names to their colors."""
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.axis('off')
    ax.set_title('Class Legend', fontsize=14, fontweight='bold', pad=10)
    patches = [mpatches.Patch(color=COLORS[c] / 255., label=CLASS_NAMES[c])
               for c in range(NUM_CLASSES)]
    ax.legend(handles=patches, loc='center', fontsize=11, frameon=False,
              ncol=2, columnspacing=1.5, handlelength=2, handleheight=1.5)
    fig.tight_layout()
    return fig


def run_inference_table(model, pseudo_root, img_size,
                        n_samples, seed, device):
    """Run inference and return a wandb.Table with individual images per row."""
    columns = ["sample_id", "rgb", "noisy_label", "dae_output",
               "ground_truth", "noisy_mIoU", "dae_mIoU", "improvement"]
    table = wandb.Table(columns=columns)

    dataset = RealNoiseDAEDataset(
        pseudo_root, split='val',
        img_size=img_size, augment=False
    )
    indices = np.random.RandomState(seed).choice(
        len(dataset), min(n_samples, len(dataset)), replace=False
    )

    for i, idx in enumerate(indices):
        rgb_t, noisy_onehot, clean_label = dataset[idx]
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
            wandb.Image(label_to_rgb(noisy_label)),
            wandb.Image(label_to_rgb(pred)),
            wandb.Image(label_to_rgb(clean_np)),
            round(noisy_iou["mIoU"], 4),
            round(dae_iou["mIoU"], 4),
            round(dae_iou["mIoU"] - noisy_iou["mIoU"], 4),
        )

    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', required=True, help='W&B run ID to resume')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--model', required=True,
                        choices=['lightweight', 'unet_resnet34', 'unet_effnet'])
    parser.add_argument('--pseudo_root', required=True, help='Path to pseudo-label dataset')
    parser.add_argument('--project', default='thietkedenoiser')
    parser.add_argument('--entity', default='vuongcris4-hcmute')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--n_samples', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Resume existing W&B run
    wandb.init(
        project=args.project, entity=args.entity,
        id=args.run_id, resume="must",
    )
    print(f'Resumed W&B run: {wandb.run.url}')

    # Load model
    model = build_model(args.model).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    best_miou = ckpt.get("best_miou", "?")
    print(f'Loaded {args.model} (best mIoU: {best_miou})')

    # Upload checkpoint
    print('Uploading checkpoint...')
    artifact = wandb.Artifact(
        name=f'checkpoint-{args.model}', type='model',
        description=f'{args.model} best checkpoint (mIoU: {best_miou})',
        metadata={'model': args.model, 'best_miou': best_miou, 'epoch': ckpt.get('epoch')}
    )
    artifact.add_file(args.checkpoint)
    wandb.log_artifact(artifact)
    print(f'  Uploaded: {args.checkpoint}')

    # Log class legend
    print('Logging class legend...')
    legend_fig = make_legend_image()
    wandb.log({"class_legend": wandb.Image(legend_fig, caption="Class Color Legend")})
    plt.close(legend_fig)

    # Generate inference table
    print('Generating inference table...')
    table = run_inference_table(
        model, args.pseudo_root, args.img_size,
        args.n_samples, args.seed, device
    )
    wandb.log({"inference_results": table})
    print(f'  Logged table with {len(table.data)} rows')

    wandb.finish()
    print('Done!')


if __name__ == '__main__':
    main()
