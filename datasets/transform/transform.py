import albumentations as A
from albumentations.pytorch import ToTensorV2

def build_transform(is_train, img_size):
    if is_train:
        transform = A.Compose([
            A.OneOf([
                A.ImageCompression(quality_lower=50, quality_upper=100),
                A.GaussianBlur(blur_limit=(3, 29)),
                A.RandomScale(scale_limit=(0.5, 1.0)),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5)
            ], p=0.5),
            A.Resize(img_size[0], img_size[1], p=1),
            # A.RandomCrop(img_size[0], img_size[1], p=1),
            A.Flip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            # A.RandomContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.Normalize(),
            ToTensorV2()
        ])
        return transform

    transform = A.Compose([
        # A.Resize(int(img_size[0]*0.25), int(img_size[1] * 0.25), p=1),
        # A.GaussianBlur(15),
        # A.GaussNoise(5),
        A.Resize(img_size[0], img_size[1], p=1),
        A.Normalize(),
        ToTensorV2()
    ])

    return transform