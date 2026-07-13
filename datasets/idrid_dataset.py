
import os
import cv2
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from preprocessing.enhancer import RetinalEnhancer


class IDRiDDataset(Dataset):

    def __init__(
        self,
        image_ids: List[str],
        config: Dict[str, Any],
        transform=None,
        enhancer: Optional[RetinalEnhancer] = None,
        is_training: bool = True,
    ):

        self.image_ids = image_ids
        self.config = config
        self.transform = transform
        self.enhancer = enhancer
        self.is_training = is_training

        # Resolve paths
        ds = config["dataset"]
        self.root = Path(ds["root"])

        if is_training:
            self.image_dir = self.root / ds["train_images"]
            self.mask_root = self.root / ds["train_masks"]
        else:
            self.image_dir = self.root / ds["test_images"]
            self.mask_root = self.root / ds["test_masks"]

        self.class_names = ds["class_names"]            # ["MA", "HE", "EX", "SE"]
        self.lesion_classes = ds["lesion_classes"]       # {"MA": "1. Microaneurysms", ...}
        self.mask_suffixes = ds["mask_suffixes"]         # {"MA": "_MA", ...}
        self.image_ext = ds.get("image_ext", ".jpg")
        self.mask_ext = ds.get("mask_ext", ".tif")

        self.num_classes = len(self.class_names)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        image_id = self.image_ids[idx]

        image_path = self.image_dir / f"{image_id}{self.image_ext}"
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        h, w = image.shape[:2]
        masks = []
        for cls_name in self.class_names:
            mask = self._load_single_mask(image_id, cls_name, h, w)
            masks.append(mask)

        # Stack into (H, W, C) for Albumentations compatibility
        multi_mask = np.stack(masks, axis=-1)  # (H, W, 4)

        # preprocessing (CLAHE, crop, etc.) ---
        if self.enhancer is not None:
            image, multi_mask = self.enhancer.process_pair(image, multi_mask)

        # BGR → RGB for PyTorch / Albumentations
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #augmentation + normalisation
        if self.transform is not None:
            transformed = self.transform(image=image, mask=multi_mask)
            image = transformed["image"]          # (3, H, W) float32
            multi_mask = transformed["mask"]       # (H, W, 4) float32
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            multi_mask = torch.from_numpy(multi_mask).float()

        # Albumentations returns (H, W, C)
        if isinstance(multi_mask, torch.Tensor):
            if multi_mask.ndim == 3 and multi_mask.shape[-1] == self.num_classes:
                multi_mask = multi_mask.permute(2, 0, 1)  # (H,W,C) → (C,H,W)
        elif isinstance(multi_mask, np.ndarray):
            if multi_mask.ndim == 3 and multi_mask.shape[-1] == self.num_classes:
                multi_mask = torch.from_numpy(multi_mask.transpose(2, 0, 1)).float()

    
        multi_mask = (multi_mask > 0).float()

        meta = {
            "image_id": image_id,
            "image_path": str(image_path),
        }

        return image, multi_mask, meta

    def _load_single_mask(
        self, image_id: str, cls_name: str, h: int, w: int
    ) -> np.ndarray:
        
        subfolder = self.lesion_classes[cls_name]
        suffix = self.mask_suffixes[cls_name]
        mask_path = self.mask_root / subfolder / f"{image_id}{suffix}{self.mask_ext}"

        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
               
                return np.zeros((h, w), dtype=np.uint8)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask
        else:
            
            return np.zeros((h, w), dtype=np.uint8)

    def compute_class_statistics(self) -> Dict[str, Dict[str, float]]:
        stats = {cls: {"present_count": 0, "total_pixels": 0, "ratios": []}
                 for cls in self.class_names}

        for image_id in self.image_ids:
            # Read one image to get dimensions
            image_path = self.image_dir / f"{image_id}{self.image_ext}"
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            h, w = image.shape[:2]
            total_pixels = h * w

            for cls_name in self.class_names:
                mask = self._load_single_mask(image_id, cls_name, h, w)
                lesion_pixels = int((mask > 0).sum())

                if lesion_pixels > 0:
                    stats[cls_name]["present_count"] += 1
                    stats[cls_name]["total_pixels"] += lesion_pixels
                    stats[cls_name]["ratios"].append(lesion_pixels / total_pixels)

        # Summarise
        results = {}
        for cls_name, s in stats.items():
            ratios = s["ratios"]
            results[cls_name] = {
                "present_count": s["present_count"],
                "absent_count": len(self.image_ids) - s["present_count"],
                "total_pixels": s["total_pixels"],
                "mean_ratio": float(np.mean(ratios)) if ratios else 0.0,
                "max_ratio": float(np.max(ratios)) if ratios else 0.0,
                "min_ratio": float(np.min(ratios)) if ratios else 0.0,
            }
        return results



def split_dataset(config: Dict[str, Any]) -> Dict[str, List[str]]:
    ds = config["dataset"]
    sp = config["split"]

    root = Path(ds["root"])
    image_dir = root / ds["train_images"]
    ext = ds.get("image_ext", ".jpg")

    # Discover all image IDs
    all_ids = sorted([
        f.stem for f in image_dir.iterdir()
        if f.suffix.lower() == ext.lower()
    ])

    seed = sp["random_seed"]
    train_ratio = sp["train_ratio"]
    val_ratio = sp["val_ratio"]
    # test_ratio = sp["test_ratio"]  # implied as remainder

    # First split: train vs (val + test)
    train_ids, valtest_ids = train_test_split(
        all_ids,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
    )

    # Second split: val vs test from the remaining
    relative_val = val_ratio / (1.0 - train_ratio)
    val_ids, test_ids = train_test_split(
        valtest_ids,
        train_size=relative_val,
        random_state=seed,
        shuffle=True,
    )

    split = {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }

    # Save metadata
    split_file = sp.get("split_file", "outputs/split_metadata.json")
    save_path = Path(split_file)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "ratios": {"train": train_ratio, "val": val_ratio, "test": 1 - train_ratio - val_ratio},
            "counts": {k: len(v) for k, v in split.items()},
            "ids": split,
        }, f, indent=2)

    return split


def build_dataloaders(
    config: Dict[str, Any],
    split: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, DataLoader]:
    
    from preprocessing.transforms import get_train_transforms, get_val_transforms

    if split is None:
        split = split_dataset(config)

    enhancer = RetinalEnhancer(config["preprocessing"])
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)

    train_cfg = config["training"]

    loaders = {}
    for phase, ids in split.items():
        is_train = (phase == "train")
        transform = train_transform if is_train else val_transform

        dataset = IDRiDDataset(
            image_ids=ids,
            config=config,
            transform=transform,
            enhancer=enhancer,
            is_training=True,  # All splits come from the training folder
        )

        loaders[phase] = DataLoader(
            dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=is_train,
            num_workers=train_cfg.get("num_workers", 4),
            pin_memory=train_cfg.get("pin_memory", True),
            drop_last=is_train, 
        )

    return loaders
