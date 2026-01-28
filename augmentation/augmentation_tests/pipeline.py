import os
import numpy as np
from tqdm import tqdm
import albumentations as A
import cv2
from src.utils import load_npy, save_npy, apply_brightness, _generate_perlin_noise_2d

def augment_single_file(img_path: str, mask_path: str, out_img_dir: str, out_mask_dir: str):
    orig_img = load_npy(img_path)
    orig_mask = load_npy(mask_path).astype(np.uint8)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    aug_id = 1

    for _ in range(8):
        angle = np.random.uniform(-30, 30)
        pipe = A.Compose([
            A.Rotate(limit=(angle, angle), p=1.0, border_mode=cv2.BORDER_CONSTANT)
        ], additional_targets={'mask': 'mask'})
        res = pipe(image=orig_img, mask=orig_mask)
        img_rotated = res["image"]
        mask_rotated = res["mask"]

        img_out = img_rotated.copy()
        delta = np.random.uniform(-0.4, 0.4)
        img_out = apply_brightness(img_out, delta)

        noise_type = np.random.choice(['perlin', 'gauss', 'blur'])

        if noise_type == 'perlin':
            h, w = img_out.shape[:2]
            if len(img_out.shape) == 3:
                noise = np.stack([_generate_perlin_noise_2d((h, w), (4, 4)) for _ in range(img_out.shape[2])], axis=-1)
            else:
                noise = _generate_perlin_noise_2d((h, w), (4, 4))
            noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
            intensity = np.random.uniform(0.05, 0.2)
            img_out = np.clip(img_out.astype(np.float32) + intensity * 255 * (noise - 0.5), 0, 255).astype(np.uint8)

        elif noise_type == 'gauss':
            std = np.random.uniform(10, 50)
            noise = np.random.normal(0, std, img_out.shape).astype(np.float32)
            img_out = np.clip(img_out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        elif noise_type == 'blur':
            sigma = np.random.choice([i for i in range(3, 20, 2)])
            img_out = cv2.GaussianBlur(img_out, (sigma, sigma), 0)

        save_npy(img_out, os.path.join(out_img_dir, f"{base_name}_aug{aug_id:03d}.npy"))
        save_npy(mask_rotated, os.path.join(out_mask_dir, f"{base_name}_aug{aug_id:03d}.npy"))
        aug_id += 1

def augment_all_files(images_dir: str, masks_dir: str, output_img_dir: str, output_mask_dir: str):
    img_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.npy')])
    print(f"Найдено {len(img_files)} изображений")

    for fname in tqdm(img_files, desc="Аугментация"):
        img_path = os.path.join(images_dir, fname)
        mask_path = os.path.join(masks_dir, fname)
        if not os.path.exists(mask_path):
            continue
        augment_single_file(img_path, mask_path, output_img_dir, output_mask_dir)