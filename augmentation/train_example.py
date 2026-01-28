import random
import matplotlib.pyplot as plt
from augmentation.dataset import RiverDataset

def visualize_random_samples(images_dir: str, masks_dir: str, n_samples: int = 15):
    ds = RiverDataset(images_dir, masks_dir, augment=True)

    if len(ds) == 0:
        raise ValueError("Нет .npy файлов в указанных папках!")

    indices = random.sample(range(len(ds)), min(n_samples, len(ds)))
    samples = [ds[i] for i in indices]

    fig, axes = plt.subplots(n_samples, 2, figsize=(11, 2.2 * n_samples))
    fig.suptitle(f"{n_samples} случайных аугментированных пар", fontsize=14, y=0.98)

    for i, (img, mask, fname, _) in enumerate(samples):  # ← игнорируем aug_str
        img_np = img.permute(1, 2, 0).numpy()
        mask_np = mask.squeeze().numpy()

        ax_img = axes[i, 0] if n_samples > 1 else axes[0]
        ax_mask = axes[i, 1] if n_samples > 1 else axes[1]

        ax_img.imshow(img_np)
        ax_img.set_title(fname, fontsize=9, pad=2)  # ← без описания аугментаций
        ax_img.axis('off')

        ax_mask.imshow(mask_np, cmap='gray')
        ax_mask.set_title("Маска", fontsize=9, pad=2)
        ax_mask.axis('off')

    if n_samples > 1:
        for j in range(i + 1, n_samples):
            axes[j, 0].axis('off')
            axes[j, 1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    IMAGES_DIR = r"data_in/train/images"
    MASKS_DIR = r"data_in/train/masks"

    visualize_random_samples(IMAGES_DIR, MASKS_DIR, n_samples=5)