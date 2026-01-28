import os
import numpy as np
import matplotlib.pyplot as plt

IMG_DIR = "../../data_in/train/images"
MASK_DIR = "../../data_in/train/masks"
SAVE_DIR = "data_photos"

os.makedirs(SAVE_DIR, exist_ok=True)

img_names = {f for f in os.listdir(IMG_DIR) if f.endswith('.npy')}
mask_names = {f for f in os.listdir(MASK_DIR) if f.endswith('.npy')}
common_names = sorted(img_names & mask_names)

print(f"Найдено пар: {len(common_names)}")

for name in common_names:
    img_path = os.path.join(IMG_DIR, name)
    mask_path = os.path.join(MASK_DIR, name)
    save_path = os.path.join(SAVE_DIR, name.replace('.npy', '.png'))

    img = np.load(img_path)
    mask = np.load(mask_path)

    if img.max() > 1:
        img = img / 255.0
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img.squeeze(-1)

    mask = mask.astype(float)
    if mask.max() > 1:
        mask = mask / mask.max()

    fig, (ax_img, ax_mask) = plt.subplots(1, 2, figsize=(9, 4))

    ax_img.imshow(img, cmap='gray' if img.ndim == 2 else None)
    ax_img.set_title(f"Image\n{name}", fontsize=10)
    ax_img.axis('off')

    ax_mask.imshow(mask, cmap='gray', vmin=0, vmax=1)
    ax_mask.set_title(f"Mask\n{name}", fontsize=10)
    ax_mask.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Сохранено: {save_path}")