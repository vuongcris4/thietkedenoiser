"""
Evaluate trained DAE model on different noise types and rates.
Usage:
    python evaluate_dae.py --checkpoint ../checkpoints/best.pth --model unet_resnet34
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from dae_model import build_model, NUM_CLASSES
from dataset import DAEDataset
from noise_generator import CLASS_NAMES, compute_iou


@torch.no_grad()
def evaluate_model(model, data_root, noise_type, noise_rate, device,
                   img_size=512, batch_size=4, num_samples=100):
    """Evaluate DAE on specific noise type and rate."""
    dataset = DAEDataset(
        data_root, split='val', img_size=img_size,
        noise_type=noise_type,
        noise_rate_range=(noise_rate, noise_rate),  # Fixed rate
        augment=False
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    model.eval()
    all_preds = []
    all_targets = []
    all_noisy_ious = []
    
    for inputs, targets in loader:
        inputs = inputs.to(device)
        
        with autocast():
            outputs = model(inputs)
        
        pred = outputs.argmax(dim=1).cpu().numpy()
        target = targets.numpy()
        
        all_preds.append(pred)
        all_targets.append(target)
        
        # Also compute noisy input IoU (before DAE)
        noisy_label = inputs[:, 3:].argmax(dim=1).cpu().numpy()
        for b in range(len(target)):
            noisy_iou = compute_iou(noisy_label[b], target[b])
            all_noisy_ious.append(noisy_iou['mIoU'])
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Compute metrics
    cleaned_iou = compute_iou(all_preds.reshape(-1, img_size, img_size)[0],
                               all_targets.reshape(-1, img_size, img_size)[0])
    
    # Per-sample mIoU
    sample_ious = []
    for i in range(len(all_preds)):
        iou = compute_iou(all_preds[i], all_targets[i])
        sample_ious.append(iou['mIoU'])
    
    return {
        'noise_type': noise_type,
        'noise_rate': noise_rate,
        'dae_miou': float(np.mean(sample_ious)),
        'noisy_miou': float(np.mean(all_noisy_ious)),
        'improvement': float(np.mean(sample_ious) - np.mean(all_noisy_ious)),
        'dae_std': float(np.std(sample_ious)),
    }


def run_full_evaluation(model, data_root, device, output_dir, **kwargs):
    """Run evaluation across all noise types and rates."""
    os.makedirs(output_dir, exist_ok=True)
    
    noise_types = ['random_flip', 'boundary', 'region_swap', 'confusion_based', 'mixed']
    noise_rates = [0.05, 0.10, 0.15, 0.20, 0.30]
    
    results = []
    
    for nt in noise_types:
        for nr in noise_rates:
            print(f'Evaluating {nt} @ {nr:.0%}...')
            r = evaluate_model(model, data_root, nt, nr, device, **kwargs)
            results.append(r)
            print(f'  Noisy: {r["noisy_miou"]:.3f} -> DAE: {r["dae_miou"]:.3f} '
                  f'(+{r["improvement"]:+.3f})')
    
    # Save results
    with open(f'{output_dir}/dae_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary table
    print(f'\n{"="*80}')
    print('DAE EVALUATION SUMMARY')
    print(f'{"="*80}')
    header = f'{"Noise Type":>18s}'
    for nr in noise_rates:
        header += f' | {nr:.0%}  '
    print(header)
    print('-' * 80)
    
    for nt in noise_types:
        row = f'{nt:>18s}'
        for nr in noise_rates:
            r = [x for x in results if x['noise_type'] == nt and x['noise_rate'] == nr][0]
            row += f' | {r["improvement"]:+.2f}'
        print(row)
    
    print(f'\n(Positive = DAE improved over noisy input)')
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model', type=str, default='unet_resnet34')
    parser.add_argument('--data_root', type=str, default='../data/OpenEarthMap')
    parser.add_argument('--output_dir', type=str, default='../results/metrics')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = build_model(args.model).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f'Loaded checkpoint: {args.checkpoint}')
    print(f'Best mIoU from training: {ckpt.get("best_miou", "N/A")}')
    
    run_full_evaluation(model, args.data_root, device, args.output_dir,
                        img_size=args.img_size, batch_size=args.batch_size)
