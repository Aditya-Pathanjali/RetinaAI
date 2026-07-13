
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any, Optional


def get_train_transforms(config: Dict[str, Any]) -> A.Compose:
    prep = config["preprocessing"]
    aug = config["augmentation"]
    
    img_size = prep.get("image_size", 512)

    transforms_list = [
        A.Resize(height=img_size, width=img_size, interpolation=1),
    ]

    if aug.get("enabled", True):
        transforms_list.extend([
            # ── Spatial transforms (applied to image + mask) ──
            A.HorizontalFlip(p=aug.get("p_hflip", 0.5)),
            A.Rotate(
                limit=aug.get("rotate_limit", 45),
                border_mode=0,
                p=aug.get("p_rotate", 0.5),
            ),
            A.ElasticTransform(
                alpha=aug.get("elastic_alpha", 120),
                sigma=aug.get("elastic_sigma", 6),
                border_mode=0,
                p=aug.get("p_elastic", 0.3),
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                border_mode=0,
                p=aug.get("p_grid", 0.3),
            ),

            # ── Photometric transforms (image only) ──
            A.RandomBrightnessContrast(
                brightness_limit=aug.get("brightness_limit", 0.2),
                contrast_limit=aug.get("contrast_limit", 0.2),
                p=aug.get("p_brightness", 0.4),
            ),
            A.RandomGamma(
                gamma_limit=tuple(aug.get("gamma_limit", [70, 130])),
                p=aug.get("p_gamma", 0.4),
            ),
            A.GaussNoise(
                p=aug.get("p_gauss_noise", 0.2),
            ),
            # Sharpen — enhances fine structures like MA edges
            A.Sharpen(
                alpha=(0.2, 0.5),
                lightness=(0.5, 1.0),
                p=aug.get("p_sharpen", 0.3),
            ),
            # HueSaturationValue — colour variation for domain robustness
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=aug.get("p_hue_sat", 0.2),
            ),
            # CoarseDropout — acts as regularizer, forces model to use
            # multiple spatial cues instead of relying on single regions
            A.CoarseDropout(
                max_holes=6,
                max_height=int(img_size * 0.08),
                max_width=int(img_size * 0.08),
                min_holes=2,
                fill_value=0,
                p=aug.get("p_coarse_dropout", 0.2),
            ),
        ])

    transforms_list.extend([
        A.Normalize(
            mean=prep.get("normalize_mean", [0.485, 0.456, 0.406]),
            std=prep.get("normalize_std", [0.229, 0.224, 0.225]),
        ),
        ToTensorV2(),
    ])

    return A.Compose(transforms_list)


def get_val_transforms(config: Dict[str, Any]) -> A.Compose:
    
    prep = config["preprocessing"]
    img_size = prep.get("image_size", 512)

    return A.Compose([
        A.Resize(height=img_size, width=img_size, interpolation=1),
        A.Normalize(
            mean=prep.get("normalize_mean", [0.485, 0.456, 0.406]),
            std=prep.get("normalize_std", [0.229, 0.224, 0.225]),
        ),
        ToTensorV2(),
    ])


def get_augmentation_preview_transforms(config: Dict[str, Any]) -> A.Compose:
    
    prep = config["preprocessing"]
    aug = config["augmentation"]
    size = prep["image_size"]

    return A.Compose([
        A.Resize(size, size, interpolation=1),
        A.HorizontalFlip(p=aug.get("p_hflip", 0.5)),
        A.Rotate(
            limit=aug.get("rotate_limit", 45),
            border_mode=0,
            p=aug.get("p_rotate", 0.5),
        ),
        A.ElasticTransform(
            alpha=aug.get("elastic_alpha", 120),
            sigma=aug.get("elastic_sigma", 6),
            border_mode=0,
            p=aug.get("p_elastic", 0.3),
        ),
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            border_mode=0,
            p=aug.get("p_grid", 0.3),
        ),
        A.RandomBrightnessContrast(
            brightness_limit=aug.get("brightness_limit", 0.2),
            contrast_limit=aug.get("contrast_limit", 0.2),
            p=aug.get("p_brightness", 0.4),
        ),
        A.RandomGamma(
            gamma_limit=tuple(aug.get("gamma_limit", [70, 130])),
            p=aug.get("p_gamma", 0.4),
        ),
    ])
