"""
Visualize RGB images, pseudo-labels, and ground truth from OEM_v2_aDanh dataset.
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse

# 8 classes in OpenEarthMap
CLASS_NAMES = [
    'Bareland', 'Rangeland', 'Developed', 'Road',
    'Tree', 'Water', 'Agriculture', 'Building'
]

# Colors for each class (RGB)
CLASS_COLORS = [
    [255, 255, 255],  # Bareland - White
    [200, 150, 100],  # Rangeland - Light brown
    [255, 0, 0],      # Developed - Red
    [128, 128, 128],  # Road - Gray
    [0, 128, 0],      # Tree - Green
    [0, 0, 255],      # Water - Blue
    [128, 255, 0],    # Agriculture - Light green
    [255, 128, 0],    # Building - Orange
]

def create_color_legend():
    """Create a color legend for the classes."""
    fig, ax = plt.subplots(figsize=(10, 1.5))
    ax.set_xlim(0, len(CLASS_NAMES))
    ax.set_ylim(0, 1)
    ax.axis('off')

    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=np.array(color) / 255))
        ax.text(i + 0.5, 0.5, name, ha='center', va='center',
                fontsize=9, fontweight='bold', color='black')

    plt.tight_layout()
    return fig


def label_to_color(label):
    """Convert label map [H, W] to RGB color image [H, W, 3]."""
    h, w = label.shape
    color_label = np.zeros((h, w, 3), dtype=np.uint8)

    for c in range(len(CLASS_NAMES)):
        mask = (label == c)
        color_label[mask] = CLASS_COLORS[c]

    return color_label


def visualize_samples(data_root, num_samples=6, output_path=None):
    """
    Visualize RGB images, pseudo-labels, and ground truth.

    Since OEM_v2_aDanh has flat structure (images/, labels/),
    we read directly from those folders.
    """
    # Read list of files from train.txt
    split_file = os.path.join(data_root, 'train.txt')
    with open(split_file) as f:
        filenames = [l.strip() for l in f if l.strip()][:num_samples]

    if not filenames:
        print(f"No files found in {split_file}")
        return

    # Create figure for visualization
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    print(f"Visualizing {len(filenames)} samples...")

    for i, filename in enumerate(filenames):
        # Read RGB image
        img_path = os.path.join(data_root, 'images', filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Read pseudo-label
        label_path = os.path.join(data_root, 'labels', filename)
        pseudo_label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if pseudo_label is None:
            print(f"Failed to read label: {label_path}")
            continue
        if pseudo_label.ndim == 3:
            pseudo_label = pseudo_label[:, :, 0]

        # For OEM_v2_aDanh, pseudo-label IS the label (no separate GT)
        # The GT would be from original OpenEarthMap if available
        gt_label = pseudo_label.copy()  # In this dataset, pseudo = label

        # Convert labels to color
        pseudo_color = label_to_color(pseudo_label)
        gt_color = label_to_color(gt_label)

        # Display RGB
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'RGB Image\n{filename}', fontsize=10, fontweight='bold')
        axes[i, 0].axis('off')

        # Display pseudo-label
        axes[i, 1].imshow(pseudo_color)
        axes[i, 1].set_title('Pseudo-Label (CISC-R)', fontsize=10, fontweight='bold')
        axes[i, 1].axis('off')

        # Display ground truth
        axes[i, 2].imshow(gt_color)
        axes[i, 2].set_title('Ground Truth', fontsize=10, fontweight='bold')
        axes[i, 2].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")

    plt.show()

    # Also save legend
    legend_fig = create_color_legend()
    legend_path = output_path.replace('.png', '_legend.png') if output_path else 'class_legend.png'
    legend_fig.savefig(legend_path, dpi=150, bbox_inches='tight')
    print(f"Saved class legend to: {legend_path}")
    plt.close(legend_fig)


def visualize_single_sample(data_root, filename):
    """Visualize a single sample in detail."""
    # Read RGB image
    img_path = os.path.join(data_root, 'images', filename)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Failed to read image: {img_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Read label
    label_path = os.path.join(data_root, 'labels', filename)
    label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
    if label is None:
        print(f"Failed to read label: {label_path}")
        return
    if label.ndim == 3:
        label = label[:, :, 0]

    # Convert label to color
    label_color = label_to_color(label)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(img)
    axes[0].set_title(f'RGB Image\n{filename}', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(label_color)
    axes[1].set_title('Semantic Segmentation Label', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # Show class distribution
    unique, counts = np.unique(label, return_counts=True)
    class_dist = np.zeros(len(CLASS_NAMES))
    for u, c in zip(unique, counts):
        if u < len(CLASS_NAMES):
            class_dist[u] = c

    axes[2].bar(range(len(CLASS_NAMES)), class_dist,
                color=[np.array(c) / 255 for c in CLASS_COLORS])
    axes[2].set_title('Class Distribution', fontsize=12, fontweight='bold')
    axes[2].set_xticks(range(len(CLASS_NAMES)))
    axes[2].set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    axes[2].set_ylabel('Pixel Count')

    plt.tight_layout()
    plt.show()

    # Print statistics
    print(f"\nImage: {filename}")
    print(f"Image shape: {img.shape}")
    print(f"Label shape: {label.shape}")
    print(f"\nClass distribution:")
    total_pixels = label.shape[0] * label.shape[1]
    for c in range(len(CLASS_NAMES)):
        pct = class_dist[c] / total_pixels * 100
        print(f"  {CLASS_NAMES[c]:15s}: {class_dist[c]:6d} pixels ({pct:5.2f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize OEM_v2_aDanh dataset')
    parser.add_argument('--data-root', type=str,
                        default='/home/ubuntu/thietkedenoiser/data/OEM_v2_aDanh',
                        help='Path to OEM_v2_aDanh dataset folder')
    parser.add_argument('--num-samples', type=int, default=6,
                        help='Number of samples to visualize')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for saved visualization')
    parser.add_argument('--single', type=str, default=None,
                        help='Visualize a single sample by filename')

    args = parser.parse_args()

    if args.single:
        visualize_single_sample(args.data_root, args.single)
    else:
        visualize_samples(args.data_root, args.num_samples, args.output)
