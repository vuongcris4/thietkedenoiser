import os, sys, numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
from noise_generator import NoiseGenerator, compute_iou, CLASS_NAMES

gen = NoiseGenerator(num_classes=8, seed=42)
noise_types = ['random_flip', 'boundary', 'region_swap', 'confusion_based', 'mixed']
noise_funcs = {
    'random_flip': gen.random_flip_noise,
    'boundary': gen.boundary_noise,
    'region_swap': gen.region_swap_noise,
    'confusion_based': gen.confusion_based_noise,
    'mixed': gen.mixed_noise,
}
noise_rates = [0.05, 0.10, 0.15, 0.20, 0.30]

def make_label(H=256, W=256):
    rng = gen.rng
    label = np.full((H, W), rng.choice([1, 4, 6]), dtype=np.int32)
    for _ in range(rng.randint(5, 12)):
        cx, cy = rng.randint(0, W), rng.randint(0, H)
        rx, ry = rng.randint(15, 60), rng.randint(15, 60)
        y, x = np.ogrid[:H, :W]
        mask = ((x-cx)**2/(rx**2+1e-6) + (y-cy)**2/(ry**2+1e-6)) < 1
        label[mask] = rng.randint(0, 8)
    for _ in range(rng.randint(1, 3)):
        pt1 = (rng.randint(0, W), rng.randint(0, H))
        pt2 = (rng.randint(0, W), rng.randint(0, H))
        rm = np.zeros((H, W), dtype=np.uint8)
        cv2.line(rm, pt1, pt2, 1, thickness=rng.randint(2, 5))
        label[rm > 0] = 3
    return label

print('Generating 20 samples for evaluation...')
results = {nt: {nr: [] for nr in noise_rates} for nt in noise_types}
stats_all = {nt: {nr: [] for nr in noise_rates} for nt in noise_types}

for i in range(20):
    clean = make_label()
    for nt in noise_types:
        for nr in noise_rates:
            noisy = noise_funcs[nt](clean, noise_rate=nr)
            iou = compute_iou(noisy, clean)
            st = gen.compute_noise_stats(clean, noisy)
            results[nt][nr].append(iou['mIoU'])
            stats_all[nt][nr].append(st)
    if (i+1) % 5 == 0:
        print(f'  {i+1}/20 done')

print()
print('=' * 90)
print('TABLE 1: mIoU (%) - Higher = less damage from noise')
print('=' * 90)
header = f'{"Noise Type":>18s}'
for nr in noise_rates:
    header += f'  | {nr:.0%} noise'
print(header)
print('-' * 90)
for nt in noise_types:
    row = f'{nt:>18s}'
    for nr in noise_rates:
        m = np.mean(results[nt][nr]) * 100
        row += f'  | {m:6.2f}% '
    print(row)

print()
print('=' * 90)
print('TABLE 2: BOUNDARY RATIO (%) - Higher = noise concentrated at boundaries')
print('=' * 90)
for nt in noise_types:
    row = f'{nt:>18s}'
    for nr in noise_rates:
        br = np.mean([s['boundary_ratio'] for s in stats_all[nt][nr]]) * 100
        row += f'  | {br:6.1f}% '
    print(row)

print()
print('=' * 90)
print('TABLE 3: ACTUAL NOISE RATE (%) vs REQUESTED')
print('=' * 90)
for nt in noise_types:
    row = f'{nt:>18s}'
    for nr in noise_rates:
        ar = np.mean([s['actual_noise_rate'] for s in stats_all[nt][nr]]) * 100
        row += f'  | {ar:6.2f}% '
    print(row)

# Per-class analysis at 15%
print()
print('=' * 90)
print('TABLE 4: PER-CLASS IoU (%) at 15% noise')
print('=' * 90)
nr_target = 0.15
header2 = f'{"Class":>12s}'
for nt in noise_types:
    header2 += f' | {nt[:8]:>8s}'
print(header2)
print('-' * 90)

gen2 = NoiseGenerator(num_classes=8, seed=99)
noise_funcs2 = {
    'random_flip': gen2.random_flip_noise,
    'boundary': gen2.boundary_noise,
    'region_swap': gen2.region_swap_noise,
    'confusion_based': gen2.confusion_based_noise,
    'mixed': gen2.mixed_noise,
}
class_results = {nt: {c: [] for c in CLASS_NAMES} for nt in noise_types}
for i in range(20):
    clean = make_label()
    for nt in noise_types:
        noisy = noise_funcs2[nt](clean, noise_rate=nr_target)
        iou = compute_iou(noisy, clean)
        for cn in CLASS_NAMES:
            if cn in iou['per_class_iou']:
                class_results[nt][cn].append(iou['per_class_iou'][cn])

for cn in CLASS_NAMES:
    row = f'{cn:>12s}'
    for nt in noise_types:
        vals = class_results[nt][cn]
        if vals:
            row += f' | {np.mean(vals)*100:7.1f}%'
        else:
            row += f' |     N/A'
    print(row)

print()
print('=' * 90)
print('ANALYSIS SUMMARY')
print('=' * 90)
print()

# Find which noise type damages most/least
for nr in [0.15]:
    scores = {}
    for nt in noise_types:
        scores[nt] = np.mean(results[nt][nr]) * 100
    sorted_nt = sorted(scores.items(), key=lambda x: x[1])
    print(f'At {nr:.0%} noise rate:')
    print(f'  Most damaging:  {sorted_nt[0][0]} (mIoU={sorted_nt[0][1]:.1f}%)')
    print(f'  Least damaging: {sorted_nt[-1][0]} (mIoU={sorted_nt[-1][1]:.1f}%)')
    print()
    
    # Boundary analysis
    for nt in noise_types:
        br = np.mean([s['boundary_ratio'] for s in stats_all[nt][nr]]) * 100
        ar = np.mean([s['actual_noise_rate'] for s in stats_all[nt][nr]]) * 100
        desc = ''
        if br > 50: desc = '(boundary-concentrated)'
        elif br > 30: desc = '(moderate boundary)'
        else: desc = '(distributed/interior)'
        print(f'  {nt:>18s}: boundary={br:.1f}%, actual={ar:.1f}% {desc}')

print()
print('RECOMMENDATIONS for DAE training:')
print('  1. Use MIXED noise for robust training (diverse error patterns)')
print('  2. Focus on BOUNDARY noise - most realistic for segmentation')
print('  3. CONFUSION-BASED noise simulates real model errors best')
print('  4. Test DAE with all noise types separately for ablation study')
print()
print('DONE')
