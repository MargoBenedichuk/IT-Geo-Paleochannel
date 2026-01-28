import os
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import cv2


def _generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * np.pi * np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)

    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)

    t = f(grid)
    n0 = n00 * (1 - t[:,:,0]) + t[:,:,0] * n10
    n1 = n01 * (1 - t[:,:,0]) + t[:,:,0] * n11
    return np.sqrt(2) * ((1 - t[:,:,1]) * n0 + t[:,:,1] * n1)


class PerlinNoise(ImageOnlyTransform):
    def __init__(self, intensity=0.15, p=0.5):
        super().__init__(p=p)
        self.intensity = intensity

    def apply(self, img, **params):
        h, w = img.shape[:2]
        if len(img.shape) == 3:
            noise = np.stack([_generate_perlin_noise_2d((h, w), (4, 4)) for _ in range(img.shape[2])], axis=-1)
        else:
            noise = _generate_perlin_noise_2d((h, w), (4, 4))
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        noisy = img.astype(np.float32) + self.intensity * 255 * (noise - 0.5)
        return np.clip(noisy, 0, 255).astype(np.uint8)


class RiverDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, augment: bool = False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.augment = augment

        image_files = {f for f in os.listdir(images_dir) if f.endswith('.npy')}
        mask_files = {f for f in os.listdir(masks_dir) if f.endswith('.npy')}
        self.filenames = sorted(list(image_files & mask_files))
        print(f"Загружено {len(self.filenames)} пар")

        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(
                    limit=30,
                    p=0.4,
                    border_mode=cv2.BORDER_CONSTANT
                ),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, p=0.4),
                A.GaussNoise(std_range=(10/255.0, 50/255.0), p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),
                PerlinNoise(intensity=0.15, p=0.2),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([ToTensorV2()])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        img_path = os.path.join(self.images_dir, name)
        mask_path = os.path.join(self.masks_dir, name)

        image = np.load(img_path)
        mask = np.load(mask_path)

        # Приводим к (H, W, C), если нужно
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)

        image = image.astype(np.uint8)
        mask = (mask > 0).astype(np.uint8)

        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return image, mask, name, ""