# src/augmentations.py
import cv2
import albumentations as A


def get_augmentation_pipeline(aug_type: str):
    if aug_type == "h_flip":
        return A.Compose([A.HorizontalFlip(p=1.0)], additional_targets={'mask': 'mask'})

    elif aug_type == "v_flip":
        return A.Compose([A.VerticalFlip(p=1.0)], additional_targets={'mask': 'mask'})

    elif aug_type == "transpose":
        return A.Compose([A.Transpose(p=1.0)], additional_targets={'mask': 'mask'})

    elif aug_type == "brighter":
        return A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0, p=1.0)
        ], additional_targets={'mask': 'mask'})

    elif aug_type == "darker":
        return A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(-0.4, -0.1), contrast_limit=0, p=1.0)
        ], additional_targets={'mask': 'mask'})

    elif aug_type == "crop_center":
        return A.Compose([
            A.CenterCrop(height=448, width=448, p=1.0),
            A.Resize(height=512, width=512, p=1.0)
        ], additional_targets={'mask': 'mask'})

    elif aug_type == "gauss_blur":
        return A.Compose([
            A.GaussianBlur(blur_limit=(3, 19), p=1.0)
        ], additional_targets={'mask': 'mask'})

    else:
        raise ValueError(f"Неизвестная аугментация: {aug_type}")