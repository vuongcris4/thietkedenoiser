import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

# Load all histories
histories = {
    'Lightweight DAE': json.load(open(os.path.expanduser('~/results/logs/dae_lightweight_mixed_20260207_200336_history.json'))),
    'UNet-ResNet34': json.load(open(os.path.expanduser('~/results/logs/dae_unet_resnet34_mixed_20260207_160653_history.json'))),
    'UNet-EfficientNet-B4': json.load(open(os.path.expanduser('~/results/logs/dae_unet_effnet_mixed_20260207_172800_history.json'))),
}

# Full 50-epoch diffusion history
diff_history = json.load(open(os.path.expanduser('~/thietkedenoiser/results/logs/diffusion_T1000_dim64_20260209_060507_history.json')))

# Full 25-combo evaluation results
diff_eval = json.load(open(os.path.expanduser('~/thietkedenoiser/results/metrics/diffusion_evaluation.json')))

colors = {'Lightweight DAE': '#e74c3c', 'UNet-ResNet34': '#3498db', 'UNet-EfficientNet-B4': '#2ecc71'}
diff_color = '#9b59b6'

# ============================================================
# FIGURE 1: DAE Training Curves
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Training Curves — 3 DAE Models', fontsize=16, fontweight='bold')

for name, h in histories.items():
    epochs = [e['epoch'] for e in h]
    ax1.plot(epochs, [e['train_loss'] for e in h], color=colors[name], linewidth=2, label=f'{name} (train)', alpha=0.8)
    ax1.plot(epochs, [e['val_loss'] for e in h], color=colors[name], linewidth=2, linestyle='--', label=f'{name} (val)', alpha=0.5)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('Training & Validation Loss')
ax1.legend(fontsize=8, loc='upper right'); ax1.set_ylim(bottom=0)

for name, h in histories.items():
    epochs = [e['epoch'] for e in h]
    val_miou = [e['val_miou'] * 100 for e in h]
    best_idx = np.argmax(val_miou)
    ax2.plot(epochs, val_miou, color=colors[name], linewidth=2.5, label=name)
    ax2.scatter([epochs[best_idx]], [val_miou[best_idx]], color=colors[name], s=100, zorder=5, edgecolors='black')
    ax2.annotate(f'{val_miou[best_idx]:.1f}%', (epochs[best_idx], val_miou[best_idx]),
                textcoords='offset points', xytext=(10, -5), fontsize=9, fontweight='bold', color=colors[name])
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Validation mIoU (%)'); ax2.set_title('Validation mIoU')
ax2.legend(loc='lower right'); ax2.set_ylim(70, 100)
plt.tight_layout()
plt.savefig(f'{outdir}/fig1_dae_training_curves.png', dpi=150, bbox_inches='tight')
plt.close(); print('Saved fig1')

# ============================================================
# FIGURE 2: Diffusion Training Curves (FULL 50 epochs)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Training Curves — Conditional Diffusion Model (50 Epochs)', fontsize=16, fontweight='bold')

epochs_d = [e['epoch'] for e in diff_history]
train_loss_d = [e['train_loss'] for e in diff_history]
ax1.plot(epochs_d, train_loss_d, color=diff_color, linewidth=2.5, marker='o', markersize=3)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Diffusion Loss (MSE)'); ax1.set_title('Training Loss')
ax1.set_ylim(bottom=0)

# mIoU over epochs (only epochs with eval data)
eval_epochs = [e for e in diff_history if e.get('clean_miou') is not None]
ep_e = [e['epoch'] for e in eval_epochs]
noisy_v = [e['noisy_miou'] * 100 for e in eval_epochs]
clean_v = [e['clean_miou'] * 100 for e in eval_epochs]
imp_v = [e['improvement'] * 100 for e in eval_epochs]

ax2.plot(ep_e, noisy_v, color='gray', linewidth=2, linestyle='--', label='Noisy input mIoU', marker='s', markersize=4)
ax2.plot(ep_e, clean_v, color=diff_color, linewidth=2.5, label='Denoised mIoU', marker='o', markersize=5)
ax2.fill_between(ep_e, clean_v, noisy_v, alpha=0.15, color='red')
# Annotate improvement trend
ax2.annotate(f'Imp: {imp_v[0]:.1f}%', (ep_e[0], clean_v[0]), textcoords='offset points', xytext=(15, -10), fontsize=8, color='red')
ax2.annotate(f'Imp: {imp_v[-1]:.1f}%', (ep_e[-1], clean_v[-1]), textcoords='offset points', xytext=(-60, 10), fontsize=8, color='red')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('mIoU (%)'); ax2.set_title('Denoising Performance')
ax2.legend(loc='center right')
plt.tight_layout()
plt.savefig(f'{outdir}/fig2_diffusion_training.png', dpi=150, bbox_inches='tight')
plt.close(); print('Saved fig2')

# ============================================================
# FIGURE 3: Per-class IoU comparison
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
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
ax.set_xlabel('Class'); ax.set_ylabel('IoU (%)'); ax.set_xticks(x + bar_width)
ax.set_xticklabels(class_names, rotation=30, ha='right'); ax.set_ylim(60, 102)
ax.legend(loc='lower left', fontsize=9)
plt.tight_layout()
plt.savefig(f'{outdir}/fig3_perclass_iou.png', dpi=150, bbox_inches='tight')
plt.close(); print('Saved fig3')

# ============================================================
# FIGURE 4: Model Comparison Summary
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Model Comparison Summary', fontsize=16, fontweight='bold')
model_names = ['Lightweight\nDAE', 'UNet-\nResNet34', 'UNet-\nEfficientNet-B4', 'Conditional\nDiffusion']
params = [12.82, 24.46, 20.23, 22.25]
best_mious = []
for name in ['Lightweight DAE', 'UNet-ResNet34', 'UNet-EfficientNet-B4']:
    best = max(histories[name], key=lambda e: e['val_miou'])
    best_mious.append(best['val_miou'] * 100)
# Diffusion best from 50-epoch run
best_diff_ep = max([e for e in diff_history if e.get('clean_miou') is not None], key=lambda e: e['clean_miou'])
best_mious.append(best_diff_ep['clean_miou'] * 100)

bar_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
bars1 = ax1.bar(model_names, best_mious, color=bar_colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars1, best_mious):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax1.set_ylabel('Best mIoU (%)'); ax1.set_title('Best Validation mIoU'); ax1.set_ylim(0, 105)
ax1.axhline(y=90, color='gray', linestyle=':', alpha=0.5)

bars2 = ax2.bar(model_names, params, color=bar_colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars2, params):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}M', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax2.set_ylabel('Parameters (Millions)'); ax2.set_title('Model Size')
plt.tight_layout()
plt.savefig(f'{outdir}/fig4_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close(); print('Saved fig4')

# ============================================================
# FIGURE 5: Training Time per Epoch
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Training Time per Epoch', fontsize=16, fontweight='bold')
for name, h in histories.items():
    epochs = [e['epoch'] for e in h]
    times = [e['time'] / 60 for e in h]
    ax.plot(epochs, times, color=colors[name], linewidth=2, label=f'{name} (avg {np.mean(times):.1f} min/ep)')
times_d = [e['time'] / 60 for e in diff_history]
ax.plot(epochs_d, times_d, color=diff_color, linewidth=2, marker='o', markersize=3,
        label=f'Diffusion (avg {np.mean(times_d):.1f} min/ep)')
ax.set_xlabel('Epoch'); ax.set_ylabel('Time (minutes)'); ax.legend()
plt.tight_layout()
plt.savefig(f'{outdir}/fig5_training_time.png', dpi=150, bbox_inches='tight')
plt.close(); print('Saved fig5')

# ============================================================
# FIGURE 6: LR Schedule
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle('Learning Rate Schedule (Cosine Annealing)', fontsize=14, fontweight='bold')
for name, h in histories.items():
    ax.plot([e['epoch'] for e in h], [e['lr'] for e in h], color=colors[name], linewidth=2, label=name)
ax.plot(epochs_d, [e['lr'] for e in diff_history], color=diff_color, linewidth=2, label='Diffusion')
ax.set_xlabel('Epoch'); ax.set_ylabel('Learning Rate'); ax.legend(); ax.set_yscale('log')
plt.tight_layout()
plt.savefig(f'{outdir}/fig6_lr_schedule.png', dpi=150, bbox_inches='tight')
plt.close(); print('Saved fig6')

# ============================================================
# FIGURE 7 (NEW): Diffusion Evaluation — mIoU by noise type & rate
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Diffusion Evaluation — 25 Combinations (50 Epochs)', fontsize=16, fontweight='bold')

noise_types = ['random_flip', 'boundary', 'region_swap', 'confusion_based', 'mixed']
noise_colors = {'random_flip': '#e74c3c', 'boundary': '#f39c12', 'region_swap': '#3498db', 'confusion_based': '#2ecc71', 'mixed': '#9b59b6'}
rates = [0.05, 0.10, 0.15, 0.20, 0.30]

for nt in noise_types:
    entries = sorted([e for e in diff_eval if e['noise_type'] == nt], key=lambda e: e['noise_rate'])
    rs = [e['noise_rate'] * 100 for e in entries]
    noisy = [e['noisy_miou'] * 100 for e in entries]
    denoised = [e['dae_miou'] * 100 for e in entries]
    imp = [e['improvement'] * 100 for e in entries]
    ax1.plot(rs, denoised, color=noise_colors[nt], linewidth=2, marker='o', markersize=5, label=nt)
    ax2.plot(rs, imp, color=noise_colors[nt], linewidth=2, marker='s', markersize=5, label=nt)

ax1.set_xlabel('Noise Rate (%)'); ax1.set_ylabel('Denoised mIoU (%)'); ax1.set_title('Denoised mIoU by Noise Type')
ax1.legend(fontsize=9)

ax2.axhline(y=0, color='black', linewidth=1, linestyle='-')
ax2.set_xlabel('Noise Rate (%)'); ax2.set_ylabel('Improvement (%)'); ax2.set_title('Improvement (Positive = Better)')
ax2.legend(fontsize=9)
ax2.fill_between([5, 30], 0, -40, alpha=0.05, color='red')
ax2.text(17, -2, 'worse than input', fontsize=9, color='red', ha='center', style='italic')
plt.tight_layout()
plt.savefig(f'{outdir}/fig7_diffusion_eval.png', dpi=150, bbox_inches='tight')
plt.close(); print('Saved fig7')

print(f'\nDone! All charts at {outdir}/')
os.system(f'ls -lh {outdir}/')
