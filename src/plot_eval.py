import os, sys, numpy as np, cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))
from noise_generator import NoiseGenerator, compute_iou, CLASS_NAMES

OEM_PALETTE = {
    0: (128, 0, 0), 1: (0, 255, 36), 2: (148, 148, 148), 3: (255, 255, 255),
    4: (34, 97, 38), 5: (0, 69, 255), 6: (75, 181, 73), 7: (222, 31, 7),
}

def label_to_color(label):
    H, W = label.shape
    color = np.zeros((H, W, 3), dtype=np.uint8)
    for cid, rgb in OEM_PALETTE.items():
        color[label == cid] = rgb
    return color

def make_label(rng, H=256, W=256):
    label = np.full((H, W), rng.choice([1, 4, 6]), dtype=np.int32)
    for _ in range(rng.randint(6, 14)):
        cx, cy = rng.randint(0, W), rng.randint(0, H)
        rx, ry = rng.randint(15, 60), rng.randint(15, 60)
        y, x = np.ogrid[:H, :W]
        mask = ((x-cx)**2/(rx**2+1e-6) + (y-cy)**2/(ry**2+1e-6)) < 1
        label[mask] = rng.randint(0, 8)
    for _ in range(rng.randint(1, 4)):
        pt1, pt2 = (rng.randint(0,W), rng.randint(0,H)), (rng.randint(0,W), rng.randint(0,H))
        rm = np.zeros((H, W), dtype=np.uint8)
        cv2.line(rm, pt1, pt2, 1, thickness=rng.randint(2, 6))
        label[rm > 0] = 3
    if rng.random() > 0.4:
        cx, cy = rng.randint(0, W), rng.randint(0, H)
        rx, ry = rng.randint(20, 50), rng.randint(20, 50)
        y, x = np.ogrid[:H, :W]
        label[((x-cx)**2/(rx**2+1e-6) + (y-cy)**2/(ry**2+1e-6)) < 1] = 5
    return label

outdir = os.path.expanduser('~/thietkedenoiser/results/visualizations')
os.makedirs(outdir, exist_ok=True)

gen = NoiseGenerator(num_classes=8, seed=42)
noise_funcs = {
    'Random Flip': gen.random_flip_noise,
    'Boundary': gen.boundary_noise,
    'Region Swap': gen.region_swap_noise,
    'Confusion-based': gen.confusion_based_noise,
    'Mixed': gen.mixed_noise,
}

clean = make_label(gen.rng, 256, 256)

# ===== FIGURE 1: Visual comparison =====
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('So sanh cac loai nhieu (noise_rate=15%)', fontsize=16, fontweight='bold')

axes[0, 0].imshow(label_to_color(clean))
axes[0, 0].set_title('Clean Ground Truth', fontsize=13, fontweight='bold')
axes[0, 0].axis('off')

desc = {
    'Random Flip': 'Dom dom, khong cau truc\nMo phong: loi random',
    'Boundary': 'Tap trung o bien\nMo phong: loi boundary',
    'Region Swap': 'Hoan ca vung lon\nMo phong: nham class',
    'Confusion-based': 'Theo confusion matrix\nMo phong: loi thuc te',
    'Mixed': 'Ket hop tat ca\nMo phong: tong hop',
}

for i, (name, func) in enumerate(noise_funcs.items()):
    row, col = (i + 1) // 3, (i + 1) % 3
    noisy = func(clean, noise_rate=0.15)
    st = gen.compute_noise_stats(clean, noisy)
    axes[row, col].imshow(label_to_color(noisy))
    axes[row, col].set_title(f'{name}\n{desc[name]}', fontsize=10)
    axes[row, col].axis('off')
    axes[row, col].text(0.02, 0.98, 
        f'Changed: {st["actual_noise_rate"]:.1%}\nBoundary: {st["boundary_ratio"]:.1%}',
        transform=axes[row, col].transAxes, va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

patches = [mpatches.Patch(color=np.array(c)/255, label=n) for n, c in zip(CLASS_NAMES, OEM_PALETTE.values())]
fig.legend(handles=patches, loc='lower center', ncol=8, fontsize=9)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig(f'{outdir}/noise_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved noise_comparison.png')

# ===== FIGURE 2: Error maps =====
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
fig.suptitle('Error Maps - Vi tri pixels bi thay doi (do = thay doi)', fontsize=14)
for i, (name, func) in enumerate(noise_funcs.items()):
    noisy = func(clean, noise_rate=0.15)
    diff = (clean != noisy).astype(np.uint8) * 255
    axes[i].imshow(diff, cmap='Reds')
    axes[i].set_title(name, fontsize=11)
    axes[i].axis('off')
plt.tight_layout()
plt.savefig(f'{outdir}/error_maps.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved error_maps.png')

# ===== FIGURE 3: mIoU comparison line chart =====
noise_rates = [0.05, 0.10, 0.15, 0.20, 0.30]
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

miou_data = {}
for name, func in noise_funcs.items():
    miou_data[name] = []
    for nr in noise_rates:
        mious = []
        for _ in range(15):
            cl = make_label(np.random.RandomState(42 + _), 256, 256)
            ny = func(cl, noise_rate=nr)
            mious.append(compute_iou(ny, cl)['mIoU'])
        miou_data[name].append(np.mean(mious) * 100)

fig, ax = plt.subplots(figsize=(12, 7))
markers = ['o', 's', '^', 'D', 'v']
for i, (name, vals) in enumerate(miou_data.items()):
    ax.plot([nr*100 for nr in noise_rates], vals, color=colors[i], 
            marker=markers[i], linewidth=2.5, markersize=9, label=name)
ax.set_xlabel('Noise Rate (%)', fontsize=13)
ax.set_ylabel('mIoU (%)', fontsize=13)
ax.set_title('mIoU vs Noise Rate - Impact cua tung loai nhieu', fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='lower left')
ax.grid(True, alpha=0.3)
ax.set_xticks([nr*100 for nr in noise_rates])
plt.tight_layout()
plt.savefig(f'{outdir}/miou_comparison.png', dpi=150)
plt.close()
print('Saved miou_comparison.png')

# ===== FIGURE 4: Per-class bar chart =====
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
fig.suptitle('Per-class IoU at 15% noise - Anh huong theo class', fontsize=14)
nr_eval = 0.15
nt_names = list(noise_funcs.keys())

for ci, cn in enumerate(CLASS_NAMES):
    row, col = ci // 4, ci % 4
    ax = axes[row, col]
    vals = []
    for ni, (name, func) in enumerate(noise_funcs.items()):
        ious = []
        for s in range(15):
            cl = make_label(np.random.RandomState(42+s))
            ny = func(cl, noise_rate=nr_eval)
            iou = compute_iou(ny, cl)
            if cn in iou['per_class_iou']:
                ious.append(iou['per_class_iou'][cn])
        vals.append(np.mean(ious)*100 if ious else 0)
    bars = ax.bar(range(len(nt_names)), vals, color=colors)
    ax.set_title(cn, fontsize=12, fontweight='bold')
    ax.set_ylabel('IoU (%)')
    ax.set_xticks(range(len(nt_names)))
    ax.set_xticklabels([n[:7] for n in nt_names], rotation=45, fontsize=8)
    ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(f'{outdir}/per_class_impact.png', dpi=150)
plt.close()
print('Saved per_class_impact.png')

print('\nAll visualizations saved to', outdir)
