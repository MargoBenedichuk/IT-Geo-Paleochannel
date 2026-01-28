import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import cv2
from augmentation_tests.utils import load_npy, apply_brightness, _generate_perlin_noise_2d

IMG_PATH = r"..\data_in\train\images\6_0.npy"
MASK_PATH = r'..\data_in\train\images\6_0.npy'

def test_single_sample():
    orig_img = load_npy(IMG_PATH)
    orig_mask = load_npy(MASK_PATH)

    if orig_img.ndim == 3 and orig_img.shape[0] == 3:
        orig_img = np.transpose(orig_img, (1, 2, 0))
    orig_img = orig_img.astype(np.uint8)
    orig_mask = orig_mask.astype(np.uint8)

    results = []

    for _ in range(5):

        img_processed = orig_img.copy()
        mask_processed = orig_mask.copy()

        delta = np.random.uniform(-0.35, 0.35)
        img_processed = apply_brightness(img_processed, delta)

        noise_type = np.random.choice(['perlin', 'gauss', 'blur'])
        desc_parts = [f"яркость {delta:+.2f}"]

        if noise_type == 'perlin':
            h, w = img_processed.shape[:2]
            if len(img_processed.shape) == 3:
                noise = np.stack([_generate_perlin_noise_2d((h, w), (4, 4)) for _ in range(img_processed.shape[2])],
                                 axis=-1)
            else:
                noise = _generate_perlin_noise_2d((h, w), (4, 4))
            noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
            intensity = np.random.uniform(0.2, 0.5)
            img_processed = np.clip(img_processed.astype(np.float32) + intensity * 255 * (noise - 0.5), 0, 255).astype(
                np.uint8)
            desc_parts.append(f"Перлин {intensity:.2f}")

        elif noise_type == 'gauss':
            std = np.random.uniform(10, 50)
            noise = np.random.normal(0, std, img_processed.shape).astype(np.float32)
            img_processed = np.clip(img_processed.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            desc_parts.append(f"Гаусс σ={std:.1f}")

        elif noise_type == 'blur':
            sigma = np.random.choice([i for i in range(3, 20, 2)])
            img_processed = cv2.GaussianBlur(img_processed, (sigma, sigma), 0)
            desc_parts.append(f"Блюр {sigma}")

        angle = np.random.uniform(-30, 30)
        pipe = A.Compose([
            A.Rotate(limit=(angle, angle), p=1.0, border_mode=cv2.BORDER_CONSTANT)
        ], additional_targets={'mask': 'mask'})
        res = pipe(image=img_processed, mask=mask_processed)

        img_out = res["image"]
        mask_out = res["mask"]
        desc_parts.insert(0, f"поворот {angle:.1f}")

        results.append((img_out, mask_out, "; ".join(desc_parts)))

    n = len(results)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols * 2, figsize=(3 * cols, 2.2 * rows))
    if rows == 1 and cols * 2 == 2:
        axes = [axes]
    elif rows == 1 or cols * 2 == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for idx, (img_i, mask_i, desc) in enumerate(results):
        ax_img = axes[idx * 2]
        ax_mask = axes[idx * 2 + 1]

        ax_img.imshow(img_i)
        ax_img.set_title(desc, fontsize=8)
        ax_img.axis('off')

        ax_mask.imshow(mask_i, cmap='gray')
        ax_mask.axis('off')

    for j in range(n * 2, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_single_sample()