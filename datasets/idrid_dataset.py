
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
        patch_config: Optional[Dict[str, Any]] = None,
    ):

        self.image_ids = image_ids
        self.config = config
        self.transform = transform
        self.enhancer = enhancer
        self.is_training = is_training
        self.patch_config = patch_config or {}
        self.patch_enabled = bool(self.patch_config.get("enabled", False) and is_training)
        self.patch_size = int(self.patch_config.get("size", 512))
        self.lesion_patch_prob = float(self.patch_config.get("lesion_patch_prob", 0.5))
        self.priority_classes = self.patch_config.get("priority_classes", ["MA", "HE"])
        self.samples_per_image = int(self.patch_config.get("samples_per_image", 4 if self.patch_enabled else 1))

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
        return len(self.image_ids) * self.samples_per_image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        image_id = self.image_ids[idx % len(self.image_ids)]

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

        if self.patch_enabled:
            image, multi_mask = self._crop_patch(image, multi_mask)

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
            "valid_classes": torch.tensor([1, 1, 1, 1], dtype=torch.float32),
        }

        return image, multi_mask, meta

    def _crop_patch(self, image: np.ndarray, multi_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = multi_mask.shape[:2]
        ps = min(self.patch_size, h, w)
        lesion_mask = np.zeros((h, w), dtype=bool)

        use_lesion = np.random.rand() < self.lesion_patch_prob
        if use_lesion:
            priority_indices = [
                self.class_names.index(cls)
                for cls in self.priority_classes
                if cls in self.class_names
            ]
            for class_idx in priority_indices:
                lesion_mask |= multi_mask[:, :, class_idx] > 0
            if not lesion_mask.any():
                lesion_mask = np.any(multi_mask > 0, axis=2)

        if use_lesion and lesion_mask.any():
            ys, xs = np.where(lesion_mask)
            pick = np.random.randint(0, len(xs))
            cx, cy = int(xs[pick]), int(ys[pick])
        else:
            cx = np.random.randint(0, w)
            cy = np.random.randint(0, h)

        x1 = int(np.clip(cx - ps // 2, 0, max(w - ps, 0)))
        y1 = int(np.clip(cy - ps // 2, 0, max(h - ps, 0)))
        return image[y1:y1 + ps, x1:x1 + ps], multi_mask[y1:y1 + ps, x1:x1 + ps]

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



class EOphthaDataset(Dataset):
    def __init__(
        self,
        image_ids: List[str],
        config: Dict[str, Any],
        transform=None,
        enhancer: Optional[RetinalEnhancer] = None,
        is_training: bool = True,
        patch_config: Optional[Dict[str, Any]] = None,
    ):
        self.image_ids = image_ids
        self.config = config
        self.transform = transform
        self.enhancer = enhancer
        self.is_training = is_training
        self.patch_config = patch_config or {}
        self.patch_enabled = bool(self.patch_config.get("enabled", False) and is_training)
        self.patch_size = int(self.patch_config.get("size", 512))
        self.lesion_patch_prob = float(self.patch_config.get("lesion_patch_prob", 0.5))
        self.priority_classes = self.patch_config.get("priority_classes", ["MA", "HE"])
        self.samples_per_image = int(self.patch_config.get("samples_per_image", 4 if self.patch_enabled else 1))

        ds = config["dataset"]
        self.root = Path(ds["root"])
        self.class_names = ds["class_names"]  # ["MA", "HE", "EX", "SE"]
        self.num_classes = len(self.class_names)

        # Pre-scan directories to build image path map and mask path map
        self.image_path_map = {}
        self.mask_path_map = {}

        root_ma = self.root / "e_ophtha_MA" / "e_optha_MA"
        if (root_ma / "MA").exists():
            for root_dir, _, files in os.walk(root_ma / "MA"):
                for file in files:
                    if file.lower().endswith((".jpg", ".png", ".jpeg")):
                        stem = Path(file).stem
                        key = f"eophtha_MA:{stem}"
                        self.image_path_map[key] = Path(root_dir) / file
                        self.mask_path_map[key] = root_ma / "Annotation_MA" / Path(root_dir).name / f"{stem}.png"

        root_ex = self.root / "e_ophtha_EX" / "e_optha_EX"
        if (root_ex / "EX").exists():
            for root_dir, _, files in os.walk(root_ex / "EX"):
                for file in files:
                    if file.lower().endswith((".jpg", ".png", ".jpeg")):
                        stem = Path(file).stem
                        key = f"eophtha_EX:{stem}"
                        self.image_path_map[key] = Path(root_dir) / file
                        self.mask_path_map[key] = root_ex / "Annotation_EX" / Path(root_dir).name / f"{stem}_EX.png"

    def __len__(self) -> int:
        return len(self.image_ids) * self.samples_per_image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        image_id = self.image_ids[idx % len(self.image_ids)]

        image_path = self.image_path_map[image_id]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        h, w = image.shape[:2]
        
        # Initialize masks for all classes as zeros
        masks = [np.zeros((h, w), dtype=np.uint8) for _ in range(self.num_classes)]

        if image_id.startswith("eophtha_MA:"):
            # MA is class index 0
            mask_path = self.mask_path_map[image_id]
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    masks[0] = (mask > 0).astype(np.uint8) * 255
            valid_classes = torch.tensor([1, 0, 0, 0], dtype=torch.float32)

        elif image_id.startswith("eophtha_EX:"):
            # EX is class index 2
            mask_path = self.mask_path_map[image_id]
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    masks[2] = (mask > 0).astype(np.uint8) * 255
            valid_classes = torch.tensor([0, 0, 1, 0], dtype=torch.float32)
        else:
            valid_classes = torch.tensor([0, 0, 0, 0], dtype=torch.float32)

        # Stack into (H, W, C) for Albumentations compatibility
        multi_mask = np.stack(masks, axis=-1)

        if self.patch_enabled:
            image, multi_mask = self._crop_patch(image, multi_mask)

        # Preprocessing
        if self.enhancer is not None:
            image, multi_mask = self.enhancer.process_pair(image, multi_mask)

        # BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Augmentation + Normalization
        if self.transform is not None:
            transformed = self.transform(image=image, mask=multi_mask)
            image = transformed["image"]
            multi_mask = transformed["mask"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            multi_mask = torch.from_numpy(multi_mask).float()

        if isinstance(multi_mask, torch.Tensor):
            if multi_mask.ndim == 3 and multi_mask.shape[-1] == self.num_classes:
                multi_mask = multi_mask.permute(2, 0, 1)
        elif isinstance(multi_mask, np.ndarray):
            if multi_mask.ndim == 3 and multi_mask.shape[-1] == self.num_classes:
                multi_mask = torch.from_numpy(multi_mask.transpose(2, 0, 1)).float()

        multi_mask = (multi_mask > 0).float()

        meta = {
            "image_id": image_id,
            "image_path": str(image_path),
            "valid_classes": valid_classes,
        }

        return image, multi_mask, meta

    def _crop_patch(self, image: np.ndarray, multi_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = multi_mask.shape[:2]
        ps = min(self.patch_size, h, w)
        lesion_mask = np.zeros((h, w), dtype=bool)

        use_lesion = np.random.rand() < self.lesion_patch_prob
        if use_lesion:
            priority_indices = [
                self.class_names.index(cls)
                for cls in self.priority_classes
                if cls in self.class_names
            ]
            for class_idx in priority_indices:
                lesion_mask |= multi_mask[:, :, class_idx] > 0
            if not lesion_mask.any():
                lesion_mask = np.any(multi_mask > 0, axis=2)

        if use_lesion and lesion_mask.any():
            ys, xs = np.where(lesion_mask)
            pick = np.random.randint(0, len(xs))
            cx, cy = int(xs[pick]), int(ys[pick])
        else:
            cx = np.random.randint(0, w)
            cy = np.random.randint(0, h)

        x1 = int(np.clip(cx - ps // 2, 0, max(w - ps, 0)))
        y1 = int(np.clip(cy - ps // 2, 0, max(h - ps, 0)))
        return image[y1:y1 + ps, x1:x1 + ps], multi_mask[y1:y1 + ps, x1:x1 + ps]





def split_dataset(config: Dict[str, Any]) -> Dict[str, List[str]]:
    ds = config["dataset"]
    sp = config["split"]

    # Support combined datasets mode
    if ds.get("name") == "combined":
        combined_split = {"train": [], "val": [], "test": []}
        for idx, d_cfg in enumerate(ds.get("datasets", [])):
            sub_ds_cfg = d_cfg.copy()
            if "class_names" not in sub_ds_cfg:
                sub_ds_cfg["class_names"] = ds["class_names"]
                
            temp_config = {
                "dataset": sub_ds_cfg,
                "split": config["split"],
                "preprocessing": config["preprocessing"]
            }
            # Modify split_file path dynamically per dataset to avoid overwrites
            temp_config["split"] = temp_config["split"].copy()
            temp_config["split"]["split_file"] = str(Path(sp["split_file"]).parent / f"split_metadata_{d_cfg.get('name', idx)}.json")
            
            d_split = split_dataset(temp_config)
            for phase in ["train", "val", "test"]:
                combined_split[phase].extend(d_split[phase])
        
        # Save combined metadata
        split_file = sp.get("split_file", "outputs/split_metadata.json")
        save_path = Path(split_file)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({
                "seed": sp["random_seed"],
                "ratios": {"train": sp["train_ratio"], "val": sp["val_ratio"], "test": "combined"},
                "counts": {k: len(v) for k, v in combined_split.items()},
                "ids": combined_split,
            }, f, indent=2)
        return combined_split

    root = Path(ds["root"])

    if ds.get("name") == "eophtha":
        all_train_ids = []
        root_ma = root / "e_ophtha_MA" / "e_optha_MA"
        if (root_ma / "MA").exists():
            for root_dir, _, files in os.walk(root_ma / "MA"):
                for file in files:
                    if file.lower().endswith((".jpg", ".png", ".jpeg")):
                        all_train_ids.append(f"eophtha_MA:{Path(file).stem}")

        root_ex = root / "e_ophtha_EX" / "e_optha_EX"
        if (root_ex / "EX").exists():
            for root_dir, _, files in os.walk(root_ex / "EX"):
                for file in files:
                    if file.lower().endswith((".jpg", ".png", ".jpeg")):
                        all_train_ids.append(f"eophtha_EX:{Path(file).stem}")
        all_train_ids = sorted(list(set(all_train_ids)))
    else:
        image_dir = root / ds["train_images"]
        ext = ds.get("image_ext", ".jpg")

        # Discover all training image IDs
        all_train_ids = sorted([
            f.stem for f in image_dir.iterdir()
            if f.suffix.lower() == ext.lower()
        ])

    seed = sp["random_seed"]
    train_ratio = sp["train_ratio"]
    val_ratio = sp["val_ratio"]

    # Check if we should use the official test set (located in ds["test_images"])
    use_official_test = ds.get("use_official_test", False)

    if use_official_test:
        # Train and Val are split from train_images
        total_train_val = train_ratio + val_ratio
        train_ids, val_ids = train_test_split(
            all_train_ids,
            train_size=train_ratio / total_train_val,
            random_state=seed,
            shuffle=True,
        )
        
        # Discover all official test image IDs
        test_image_dir = root / ds["test_images"]
        test_ids = sorted([
            f.stem for f in test_image_dir.iterdir()
            if f.suffix.lower() == ext.lower()
        ])
    else:
        # First split: train vs (val + test)
        train_ids, valtest_ids = train_test_split(
            all_train_ids,
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
            "ratios": {"train": train_ratio, "val": val_ratio, "test": 1 - train_ratio - val_ratio if not use_official_test else "official"},
            "counts": {k: len(v) for k, v in split.items()},
            "ids": split,
        }, f, indent=2)

    return split


def build_dataloaders(
    config: Dict[str, Any],
    split: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, DataLoader]:
    
    from preprocessing.transforms import get_train_transforms, get_val_transforms
    from torch.utils.data import ConcatDataset

    if split is None:
        split = split_dataset(config)

    enhancer = RetinalEnhancer(config["preprocessing"])
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)

    train_cfg = config["training"]
    ds_cfg = config["dataset"]

    # Support combined datasets mode
    if ds_cfg.get("name") == "combined":
        phase_datasets = {"train": [], "val": [], "test": []}
        for idx, d_cfg in enumerate(ds_cfg.get("datasets", [])):
            sub_ds_cfg = d_cfg.copy()
            if "class_names" not in sub_ds_cfg:
                sub_ds_cfg["class_names"] = ds_cfg["class_names"]
                
            temp_config = {
                "dataset": sub_ds_cfg,
                "split": config["split"],
                "preprocessing": config["preprocessing"],
                "training": config["training"],
                "patch_training": config.get("patch_training", {}),
            }
            # Perform split dataset lookup for this single config
            d_split = split_dataset(temp_config)
            
            for phase in ["train", "val", "test"]:
                is_train = (phase == "train")
                transform = train_transform if is_train else val_transform
                patch_config = config.get("patch_training", {}) if is_train else None
                
                is_training_split = True
                if phase == "test" and d_cfg.get("use_official_test", False):
                    is_training_split = False
                    
                if d_cfg.get("name") == "eophtha":
                    ds_instance = EOphthaDataset(
                        image_ids=d_split[phase],
                        config=temp_config,
                        transform=transform,
                        enhancer=enhancer,
                        is_training=is_training_split,
                        patch_config=patch_config,
                    )
                else:
                    ds_instance = IDRiDDataset(
                        image_ids=d_split[phase],
                        config=temp_config,
                        transform=transform,
                        enhancer=enhancer,
                        is_training=is_training_split,
                        patch_config=patch_config,
                    )
                phase_datasets[phase].append(ds_instance)
        
        loaders = {}
        for phase in ["train", "val", "test"]:
            is_train = (phase == "train")
            concated_ds = ConcatDataset(phase_datasets[phase])
            
            loaders[phase] = DataLoader(
                concated_ds,
                batch_size=train_cfg["batch_size"],
                shuffle=is_train,
                num_workers=train_cfg.get("num_workers", 4),
                pin_memory=train_cfg.get("pin_memory", True),
                drop_last=is_train,
            )
        return loaders

    loaders = {}
    for phase, ids in split.items():
        is_train = (phase == "train")
        transform = train_transform if is_train else val_transform
        patch_config = config.get("patch_training", {}) if is_train else None

        is_training_split = True
        if phase == "test" and config["dataset"].get("use_official_test", False):
            is_training_split = False

        if config["dataset"].get("name") == "eophtha":
            dataset = EOphthaDataset(
                image_ids=ids,
                config=config,
                transform=transform,
                enhancer=enhancer,
                is_training=is_training_split,
                patch_config=patch_config,
            )
        else:
            dataset = IDRiDDataset(
                image_ids=ids,
                config=config,
                transform=transform,
                enhancer=enhancer,
                is_training=is_training_split,
                patch_config=patch_config,
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
