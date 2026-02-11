# Numpy compat fix for Python 3.8 + numpy<2.0
import numpy, numpy.core, sys
sys.modules["numpy._core"] = numpy.core
if hasattr(numpy.core, "multiarray"): sys.modules["numpy._core.multiarray"] = numpy.core.multiarray

"""
Evaluate trained Diffusion model on different noise types and rates.
Usage:
    python evaluate_diffusion.py --checkpoint ../checkpoints/diffusion_best.pth
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

sys.path.insert(0, os.path.dirname(__file__))
from diffusion_denoiser import DiffusionDenoiser, count_params, NUM_CLASSES
from dataset import OpenEarthMapDataset
from noise_generator import NoiseGenerator, compute_iou, CLASS_NAMES


@torch.no_grad()
def evaluate_diffusion(model, data_root, noise_type, noise_rate, device,
                       noise_gen, img_size=512, batch_size=4, denoise_steps=50):
    """Evaluate Diffusion on specific noise type and rate."""
    dataset = OpenEarthMapDataset(data_root, split='val', img_size=img_size, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    noise_funcs = {
        'random_flip': noise_gen.random_flip_noise,
        'boundary': noise_gen.boundary_noise,
        'region_swap': noise_gen.region_swap_noise,
        'confusion_based': noise_gen.confusion_based_noise,
        'mixed': noise_gen.mixed_noise,
    }
    noise_fn = noise_funcs[noise_type]

    model.eval()
    noisy_ious = []
    clean_ious = []
    per_class_ious = {c: [] for c in CLASS_NAMES}
    count = 0

    for rgb, label_idx in loader:
        rgb = rgb.to(device)
        label_np = label_idx.numpy()
        B = rgb.shape[0]

        for b in range(B):
            clean_np = label_np[b].astype(np.int32)
            noisy_np = noise_fn(clean_np, noise_rate=noise_rate)

            # Noisy mIoU (before denoising)
            noisy_iou = compute_iou(noisy_np, clean_np)
            noisy_ious.append(noisy_iou['mIoU'])

            # Convert noisy to one-hot
            noisy_onehot = F.one_hot(
                torch.from_numpy(noisy_np).long().clamp(0, NUM_CLASSES - 1),
                NUM_CLASSES
            ).permute(2, 0, 1).float().unsqueeze(0).to(device)

            rgb_single = rgb[b:b+1]

            with autocast():
                pred_class = model.get_clean_prediction(
                    noisy_onehot, rgb_single, num_steps=denoise_steps)

            pred_np = pred_class.cpu().numpy()[0]
            clean_iou = compute_iou(pred_np, clean_np)
            clean_ious.append(clean_iou['mIoU'])

            # Per-class IoU
            for cls_name in CLASS_NAMES:
                if cls_name in clean_iou:
                    per_class_ious[cls_name].append(clean_iou[cls_name])

            count += 1
            if count % 10 == 0:
                print(f'    {noise_type}@{noise_rate:.0%}: {count} samples done')

    avg_noisy = float(np.mean(noisy_ious))
    avg_clean = float(np.mean(clean_ious))

    per_class_avg = {}
    for cls_name, vals in per_class_ious.items():
        if vals:
            per_class_avg[cls_name] = float(np.mean(vals))

    return {
        'noise_type': noise_type,
        'noise_rate': noise_rate,
        'noisy_miou': avg_noisy,
        'dae_miou': avg_clean,
        'improvement': avg_clean - avg_noisy,
        'std': float(np.std(clean_ious)),
        'num_samples': count,
        'per_class_iou': per_class_avg,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='../data/OpenEarthMap')
    parser.add_argument('--output_dir', type=str, default='../results/metrics')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--denoise_steps', type=int, default=50)
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--base_dim', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model
    model = DiffusionDenoiser(
        num_classes=NUM_CLASSES, T=args.T,
        base_dim=args.base_dim, dim_mults=(1, 2, 4, 8)
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    total_p, train_p = count_params(model)
    print(f'Loaded: {args.checkpoint}')
    print(f'Model: {total_p/1e6:.2f}M params | Epoch: {ckpt.get("epoch", "?")} | Best improvement: {ckpt.get("best_improvement", "?")}')
    print(f'Device: {device} | Denoise steps: {args.denoise_steps}')

    noise_gen = NoiseGenerator(num_classes=NUM_CLASSES, seed=42)
    os.makedirs(args.output_dir, exist_ok=True)

    noise_types = ['random_flip', 'boundary', 'region_swap', 'confusion_based', 'mixed']
    noise_rates = [0.05, 0.10, 0.15, 0.20, 0.30]

    all_results = []
    t0_total = time.time()

    for nt in noise_types:
        for nr in noise_rates:
            print(f'\nEvaluating {nt} @ {nr:.0%}...')
            t0 = time.time()
            r = evaluate_diffusion(model, args.data_root, nt, nr, device,
                                   noise_gen, img_size=args.img_size,
                                   batch_size=args.batch_size,
                                   denoise_steps=args.denoise_steps)
            dt = time.time() - t0
            r['eval_time'] = dt
            all_results.append(r)
            print(f'  Noisy: {r["noisy_miou"]:.4f} -> Denoised: {r["dae_miou"]:.4f} '
                  f'(improvement: {r["improvement"]:+.4f}) | {dt:.1f}s')

    total_time = time.time() - t0_total

    # Save results
    output_path = f'{args.output_dir}/diffusion_evaluation.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to {output_path}')

    # Print summary table
    print(f'\n{"="*90}')
    print('DIFFUSION EVALUATION SUMMARY')
    print(f'{"="*90}')
    header = f'{"Noise Type":>18s}'
    for nr in noise_rates:
        header += f' | {nr:.0%}'.rjust(8)
    print(header)
    print('-' * 90)

    for nt in noise_types:
        row = f'{nt:>18s}'
        for nr in noise_rates:
            r = [x for x in all_results if x['noise_type'] == nt and x['noise_rate'] == nr][0]
            row += f' | {r["dae_miou"]*100:>7.2f}%'
        print(row)

    print(f'\nTotal evaluation time: {total_time:.1f}s ({total_time/60:.1f}m)')


if __name__ == '__main__':
    main()
