"""
Evaluate trained DAE model on real pseudo-labels from CISC-R.
Usage:
    python evaluate_dae.py --checkpoint ../checkpoints/dae_best.pth --pseudo_root data/OEM_v2_aDanh
"""
import os, sys, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from dae_model import build_model, NUM_CLASSES
from dataset import RealNoiseDAEDataset
from noise_generator import CLASS_NAMES, compute_iou


@torch.no_grad()
def evaluate_model(model, pseudo_root, device, img_size=512, batch_size=4):
    """Evaluate DAE on real pseudo-labels from CISC-R."""
    dataset = RealNoiseDAEDataset(
        pseudo_root, split='val',
        img_size=img_size, augment=False
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model.eval()
    all_preds = []
    all_targets = []
    all_noisy_ious = []

    for rgb, noisy_label, targets in loader:
        rgb = rgb.to(device)
        noisy_label = noisy_label.to(device)

        with autocast():
            outputs = model(rgb, noisy_label)

        pred = outputs.argmax(dim=1).cpu().numpy()
        target = targets.cpu().numpy()

        all_preds.append(pred)
        all_targets.append(target)

        # Compute noisy input IoU (before DAE)
        noisy_label_cls = noisy_label.argmax(dim=1).cpu().numpy()
        for b in range(len(target)):
            noisy_iou = compute_iou(noisy_label_cls[b], target[b])
            all_noisy_ious.append(noisy_iou['mIoU'])

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Compute overall IoU
    overall_iou = compute_iou(all_preds.reshape(-1, all_preds.shape[-2], all_preds.shape[-1]),
                              all_targets.reshape(-1, all_targets.shape[-2], all_targets.shape[-1]))

    # Per-sample mIoU
    sample_ious = []
    per_class_ious = {name: [] for name in CLASS_NAMES}

    for i in range(len(all_preds)):
        iou = compute_iou(all_preds[i], all_targets[i])
        sample_ious.append(iou['mIoU'])
        for cls_name, cls_iou in iou['per_class_iou'].items():
            per_class_ious[cls_name].append(cls_iou)

    # Average per-class IoU
    avg_per_class_iou = {name: np.mean(ious) if ious else 0.0 for name, ious in per_class_ious.items()}

    return {
        'dae_miou': float(np.mean(sample_ious)),
        'dae_std': float(np.std(sample_ious)),
        'noisy_miou': float(np.mean(all_noisy_ious)),
        'improvement': float(np.mean(sample_ious) - np.mean(all_noisy_ious)),
        'overall_iou': overall_iou,
        'per_class_iou': avg_per_class_iou,
        'num_samples': len(sample_ious),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='unet_resnet34',
                        choices=['lightweight', 'unet_resnet34', 'unet_effnet'])
    parser.add_argument('--pseudo_root', type=str, required=True,
                        help='Path to pseudo-label dataset (e.g., data/OEM_v2_aDanh)')
    parser.add_argument('--output_dir', type=str, default='../results/metrics')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(args.model).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    print(f'Loaded checkpoint: {args.checkpoint}')
    print(f'Best mIoU from training: {ckpt.get("best_miou", "N/A"):.4f}')

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'\nEvaluating on real pseudo-labels from: {args.pseudo_root}')
    results = evaluate_model(
        model, args.pseudo_root, device,
        img_size=args.img_size, batch_size=args.batch_size
    )

    # Save results
    with open(f'{args.output_dir}/dae_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f'\n{"="*60}')
    print('DAE EVALUATION RESULTS (Real Pseudo-Labels)')
    print(f'{"="*60}')
    print(f'Samples evaluated: {results["num_samples"]}')
    print(f'Noisy input mIoU:  {results["noisy_miou"]:.4f}')
    print(f'DAE output mIoU:   {results["dae_miou"]:.4f}')
    print(f'Improvement:       +{results["improvement"]:.4f}')
    print(f'DAE std:           {results["dae_std"]:.4f}')
    print(f'\nPer-class IoU:')
    for name, iou in sorted(results['per_class_iou'].items()):
        print(f'  {name}: {iou:.4f}')
    print(f'\nResults saved to: {args.output_dir}/dae_evaluation.json')
    print(f'{"="*60}')
