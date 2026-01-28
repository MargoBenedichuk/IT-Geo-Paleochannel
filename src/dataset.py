# src/dataset.py
import os
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class RiverDataset(Dataset):

    def __init__(self, images_dir: str, masks_dir: str, augment: bool = False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.augment = augment

        image_files = {f for f in os.listdir(images_dir) if f.endswith('.npy')}
        mask_files = {f for f in os.listdir(masks_dir) if f.endswith('.npy')}
        self.filenames = sorted(list(image_files & mask_files))
        print(f"Загружено {len(self.filenames)} пар")

        # Аугментации
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, p=0.4),
                A.GaussNoise(
                    std_range=(0.04, 0.2), # типо нормализация 10/255 = 0.039, 50/255 = 0.196
                    per_channel=True,
                    p=0.3
                ),
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),
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

        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)

        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return image, mask