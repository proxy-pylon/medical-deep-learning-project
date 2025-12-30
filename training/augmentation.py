import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(img_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=0.1, scale=(1-0.2, 1+0.2), rotate=30, p=0.5),
        A.OneOf([
            A.GaussNoise(std_range=(0.1, 0.2), p=1.0),
            A.GaussianBlur(),
            A.MotionBlur(),
        ], p=0.3),
        A.OneOf([
            A.OpticalDistortion(),
            A.GridDistortion(),
            A.ElasticTransform(),
        ], p=0.3),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CoarseDropout(
            num_holes_range=(2, 5),
            hole_height_range=(10, 20),
            hole_width_range=(10, 20),
            fill="random_uniform",
            p=0.3
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transform(img_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])