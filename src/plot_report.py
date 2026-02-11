import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

outdir = os.path.expanduser('~/results/report_charts')
os.makedirs(outdir, exist_ok=True)
logdir = os.path.expanduser('~/results/logs')

# Load all histories
files = {
    'Lightweight DAE': 'dae_lightweight_mixed_20260207_200336_history.json',
    'UNet-ResNet34': 'dae_unet_resnet34_mixed_20260207_160653_history.json',
    'UNet-EfficientNet-B4': 'dae_unet_effnet_mixed_20260207_172800_history.json',
}
diff_files = {
    'Diffusion (resume 1)': 'diffusion_T1000_dim64_resume_20260208_022150_history.json',
    'Diffusion (resume 2)': 'diffusion_T1000_dim64_resume_20260208_052829_history.json',
}

histories = {}
for name, f in files.items():
    histories[name] = json.load(open(os.path.join(logdir, f)))

diff_histories = {}
for name, f in diff_files.items():
    diff_histories[name] = json.load(open(os.path.join(logdir, f)))

colors = {'Lightweight DAE': '#e74c3c', 'UNet-ResNet34': '#3498db', 'UNet-EfficientNet-B4': '#2ecc71'}
diff_color = '#9b59b6'

# ============================================================
# FIGURE 1: Training Loss curves (all 3 DAE)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Training Curves — 3 DAE Models', fontsize=16, fontweight='bold')

for name, h in histories.items():
    epochs = [e['epoch'] for e in h]
    train_loss = [e['train_loss'] for e in h]
    val_loss = [e['val_loss'] for e in h]
    ax1.plot(epochs, train_loss, color=colors[name], linewidth=2, label=f'{name} (train)', alpha=0.8)
    ax1.plot(epochs, val_loss, color=colors[name], linewidth=2, linestyle='--', label=f'{name} (val)', alpha=0.5)

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training & Validation Loss')
ax1.legend(fontsize=8, loc='upper right')
ax1.set_ylim(bottom=0)

# Val mIoU curves
for name, h in histories.items():
    epochs = [e['epoch'] for e in h]
    val_miou = [e['val_miou'] * 100 for e in h]
    best_idx = np.argmax(val_miou)
    ax2.plot(epochs, val_miou, color=colors[name], linewidth=2.5, label=name)
    ax2.scatter([epochs[best_idx]], [val_miou[best_idx]], color=colors[name], s=100, zorder=5, edgecolors='black')
    ax2.annotate(f'{val_miou[best_idx]:.1f}%', (epochs[best_idx], val_miou[best_idx]),
                textcoords='offset points', xytext=(10, -5), fontsize=9, fontweight='bold', color=colors[name])

ax2.set_xlabel('Epoch')
ax2.set_ylabel('Validation mIoU (%)')
ax2.set_title('Validation mIoU')
ax2.legend(loc='lower right')
ax2.set_ylim(70, 100)

plt.tight_layout()
plt.savefig(f'{outdir}/fig1_dae_training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig1_dae_training_curves.png')

# ============================================================
# FIGURE 2: Diffusion training curves
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Training Curves — Conditional Diffusion Model', fontsize=16, fontweight='bold')

# Combine both resume runs into continuous timeline
all_diff = []
for name, h in diff_histories.items():
    for e in h:
        all_diff.append(e)
# Sort and deduplicate by epoch
seen = set()
merged_diff = []
for e in sorted(all_diff, key=lambda x: x['epoch']):
    if e['epoch'] not in seen:
        seen.add(e['epoch'])
        merged_diff.append(e)

epochs_d = [e['epoch'] for e in merged_diff]
train_loss_d = [e['train_loss'] for e in merged_diff]
ax1.plot(epochs_d, train_loss_d, color=diff_color, linewidth=2.5, marker='o', markersize=5)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Diffusion Loss (MSE)')
ax1.set_title('Training Loss')
ax1.set_ylim(bottom=0)

# Improvement over epochs
epochs_imp = [e['epoch'] for e in merged_diff if 'improvement' in e and e['improvement'] is not None]
imp_vals = [e['improvement'] * 100 for e in merged_diff if 'improvement' in e and e['improvement'] is not None]
noisy_vals = [e['noisy_miou'] * 100 for e in merged_diff if 'noisy_miou' in e and e['noisy_miou'] is not None]
clean_vals = [e['clean_miou'] * 100 for e in merged_diff if 'clean_miou' in e and e['clean_miou'] is not None]

ax2.plot(epochs_imp, noisy_vals, color='gray', linewidth=2, linestyle='--', label='Noisy input mIoU', marker='s', markersize=4)
ax2.plot(epochs_imp, clean_vals, color=diff_color, linewidth=2.5, label='Denoised mIoU', marker='o', markersize=5)
ax2.fill_between(epochs_imp, clean_vals, noisy_vals, alpha=0.15, color='red', label='Gap (negative = worse)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('mIoU (%)')
ax2.set_title('Denoising Performance')
ax2.legend(loc='center right')

plt.tight_layout()
plt.savefig(f'{outdir}/fig2_diffusion_training.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig2_diffusion_training.png')

# ============================================================
# FIGURE 3: Per-class IoU comparison (best epoch each model)
# ============================================================
fig, ax = plt.subplots(figsize=(14, 7))
fig.suptitle('Per-class IoU Comparison — Best Epoch', fontsize=16, fontweight='bold')

class_names = ['Bareland', 'Rangeland', 'Developed', 'Road', 'Tree', 'Water', 'Agriculture', 'Building']
bar_width = 0.25
x = np.arange(len(class_names))

for i, (name, h) in enumerate(histories.items()):
    best = max(h, key=lambda e: e['val_miou'])
    ious = [best['val_ious'].get(c, 0) * 100 for c in class_names]
    bars = ax.bar(x + i * bar_width, ious, bar_width, label=f"{name} (ep{best['epoch']}, mIoU={best['val_miou']*100:.1f}%)",
                  color=colors[name], alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, ious):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

ax.set_xlabel('Class')
ax.set_ylabel('IoU (%)')
ax.set_xticks(x + bar_width)
ax.set_xticklabels(class_names, rotation=30, ha='right')
ax.set_ylim(60, 102)
ax.legend(loc='lower left', fontsize=9)
plt.tight_layout()
plt.savefig(f'{outdir}/fig3_perclass_iou.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig3_perclass_iou.png')

# ============================================================
# FIGURE 4: Model comparison summary (bar chart)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Model Comparison Summary', fontsize=16, fontweight='bold')

model_names = ['Lightweight\nDAE', 'UNet-\nResNet34', 'UNet-\nEfficientNet-B4', 'Conditional\nDiffusion']
params = [12.82, 24.46, 20.23, 22.25]
best_mious = []
for name in ['Lightweight DAE', 'UNet-ResNet34', 'UNet-EfficientNet-B4']:
    best = max(histories[name], key=lambda e: e['val_miou'])
    best_mious.append(best['val_miou'] * 100)
# Diffusion best
best_diff = max(merged_diff, key=lambda e: e.get('clean_miou', 0))
best_mious.append(best_diff.get('clean_miou', 0) * 100)

bar_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

# mIoU comparison
bars1 = ax1.bar(model_names, best_mious, color=bar_colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars1, best_mious):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax1.set_ylabel('Best mIoU (%)')
ax1.set_title('Best Validation mIoU')
ax1.set_ylim(0, 105)
ax1.axhline(y=90, color='gray', linestyle=':', alpha=0.5)

# Params comparison
bars2 = ax2.bar(model_names, params, color=bar_colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars2, params):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}M', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax2.set_ylabel('Parameters (Millions)')
ax2.set_title('Model Size')

plt.tight_layout()
plt.savefig(f'{outdir}/fig4_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig4_model_comparison.png')

# ============================================================
# FIGURE 5: Training time per epoch
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Training Time per Epoch', fontsize=16, fontweight='bold')

for name, h in histories.items():
    epochs = [e['epoch'] for e in h]
    times = [e['time'] / 60 for e in h]  # to minutes
    ax.plot(epochs, times, color=colors[name], linewidth=2, label=f'{name} (avg {np.mean(times):.1f} min/ep)')

# Diffusion
times_d = [e['time'] / 60 for e in merged_diff]
epochs_d_t = [e['epoch'] for e in merged_diff]
ax.plot(epochs_d_t, times_d, color=diff_color, linewidth=2, marker='o', markersize=4,
        label=f'Diffusion (avg {np.mean(times_d):.1f} min/ep)')

ax.set_xlabel('Epoch')
ax.set_ylabel('Time (minutes)')
ax.legend()
plt.tight_layout()
plt.savefig(f'{outdir}/fig5_training_time.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig5_training_time.png')

# ============================================================
# FIGURE 6: Learning rate schedule
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle('Learning Rate Schedule (Cosine Annealing)', fontsize=14, fontweight='bold')

for name, h in histories.items():
    epochs = [e['epoch'] for e in h]
    lrs = [e['lr'] for e in h]
    ax.plot(epochs, lrs, color=colors[name], linewidth=2, label=name)

ax.set_xlabel('Epoch')
ax.set_ylabel('Learning Rate')
ax.legend()
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(f'{outdir}/fig6_lr_schedule.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig6_lr_schedule.png')

print(f'\nAll charts saved to {outdir}/')
os.system(f'ls -lh {outdir}/')
