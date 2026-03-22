"""
Quick viewer - mỗi lần chạy sẽ show random samples từ OEM_v2_aDanh.
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

CLASS_NAMES = [
    'Bareland', 'Rangeland', 'Developed', 'Road',
    'Tree', 'Water', 'Agriculture', 'Building'
]

CLASS_COLORS = [
    [255, 255, 255], [200, 150, 100], [255, 0, 0], [128, 128, 128],
    [0, 128, 0], [0, 0, 255], [128, 255, 0], [255, 128, 0],
]

def label_to_color(label):
    h, w = label.shape
    color_label = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(len(CLASS_NAMES)):
        mask = (label == c)
        color_label[mask] = CLASS_COLORS[c]
    return color_label

def show_random_samples(data_root, num_samples=6):
    # Đọc tất cả filenames
    split_file = os.path.join(data_root, 'train.txt')
    with open(split_file) as f:
        filenames = [l.strip() for l in f if l.strip()]

    # Random chọn samples
    selected = np.random.choice(filenames, min(num_samples, len(filenames)), replace=False)

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i, filename in enumerate(selected):
        img_path = os.path.join(data_root, 'images', filename)
        label_path = os.path.join(data_root, 'labels', filename)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.zeros((512, 512, 3), dtype=np.uint8)

        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if label is None:
            label = np.zeros((512, 512), dtype=np.uint8)
        if label.ndim == 3:
            label = label[:, :, 0]

        label_color = label_to_color(label)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'RGB: {filename}', fontsize=9)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(label_color)
        axes[i, 1].set_title('Pseudo Label', fontsize=9)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(label_color)
        axes[i, 2].set_title('Ground Truth', fontsize=9)
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    data_root = '/home/ubuntu/thietkedenoiser/data/OEM_v2_aDanh'
    show_random_samples(data_root, num_samples=6)
    print("\nRun lại lệnh này để xem random samples khác!")
