import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from preprocessing.enhancer import RetinalEnhancer
from preprocessing.transforms import get_train_transforms, get_val_transforms


IDRID_CLASS_DIRS = {
    "MA": "1. Microaneurysms",
    "HE": "2. Haemorrhages",
    "EX": "3. Hard Exudates",
    "SE": "4. Soft Exudates",
}

IDRID_SUFFIXES = {"MA": "_MA", "HE": "_HE", "EX": "_EX", "SE": "_SE"}


class BinaryLesionDataset(Dataset):
    def __init__(
        self,
        image_ids: List[str],
        dataset_config: Dict[str, Any],
        lesion_class: str,
        transform=None,
        enhancer: Optional[RetinalEnhancer] = None,
        is_training: bool = True,
        patch_config: Optional[Dict[str, Any]] = None,
    ):
        self.image_ids = image_ids
        self.dataset_config = dataset_config
        self.lesion_class = lesion_class
        self.transform = transform
        self.enhancer = enhancer
        self.is_training = is_training
        self.patch_config = patch_config or {}
        self.patch_enabled = bool(self.patch_config.get("enabled", False) and is_training)
        self.patch_size = int(self.patch_config.get("size", 512))
        self.lesion_patch_prob = float(self.patch_config.get("lesion_patch_prob", 0.5))
        self.samples_per_image = int(self.patch_config.get("samples_per_image", 4 if self.patch_enabled else 1))

        self.name = dataset_config["name"]
        self.root = Path(dataset_config["root"])

        if self.name == "eophtha":
            self.records = self._build_eophtha_records()
        else:
            self.records = self._build_regular_records()

    def __len__(self) -> int:
        return len(self.records) * self.samples_per_image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        record = self.records[idx % len(self.records)]
        image = cv2.imread(str(record["image_path"]))
        if image is None:
            raise FileNotFoundError(f"Image not found: {record['image_path']}")

        h, w = image.shape[:2]
        mask = self._read_mask(record["mask_path"], h, w)

        if self.patch_enabled:
            image, mask = self._crop_patch(image, mask)

        multi_mask = mask[..., None]

        if self.enhancer is not None:
            image, multi_mask = self.enhancer.process_pair(image, multi_mask)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=multi_mask)
            image = transformed["image"]
            multi_mask = transformed["mask"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            multi_mask = torch.from_numpy(multi_mask).float()

        if isinstance(multi_mask, torch.Tensor):
            if multi_mask.ndim == 2:
                multi_mask = multi_mask.unsqueeze(0)
            elif multi_mask.ndim == 3 and multi_mask.shape[-1] == 1:
                multi_mask = multi_mask.permute(2, 0, 1)
        else:
            multi_mask = torch.from_numpy(multi_mask.transpose(2, 0, 1)).float()

        multi_mask = (multi_mask > 0).float()
        meta = {
            "image_id": record["image_id"],
            "source": self.name,
            "valid_classes": torch.tensor([1.0], dtype=torch.float32),
        }
        return image, multi_mask, meta

    def _build_regular_records(self) -> List[Dict[str, Any]]:
        image_dir_key = "train_images" if self.is_training else "test_images"
        mask_dir_key = "train_masks" if self.is_training else "test_masks"
        image_dir = self.root / self.dataset_config[image_dir_key]
        mask_root = self.root / self.dataset_config[mask_dir_key]
        image_ext = self.dataset_config.get("image_ext", ".jpg")
        mask_ext = self.dataset_config.get("mask_ext", ".tif")

        lesion_dirs = self.dataset_config.get("lesion_classes", {})
        suffixes = self.dataset_config.get("mask_suffixes", {})
        subdir = lesion_dirs.get(self.lesion_class, IDRID_CLASS_DIRS.get(self.lesion_class, self.lesion_class))
        suffix = suffixes.get(self.lesion_class, IDRID_SUFFIXES.get(self.lesion_class, ""))

        records = []
        for image_id in self.image_ids:
            records.append(
                {
                    "image_id": f"{self.name}:{image_id}",
                    "image_path": image_dir / f"{image_id}{image_ext}",
                    "mask_path": mask_root / subdir / f"{image_id}{suffix}{mask_ext}",
                }
            )
        return records

    def _build_eophtha_records(self) -> List[Dict[str, Any]]:
        if self.lesion_class not in {"MA", "EX"}:
            return []

        lesion_folder = f"e_ophtha_{self.lesion_class}"
        inner_folder = f"e_optha_{self.lesion_class}"
        image_root = self.root / lesion_folder / inner_folder / self.lesion_class
        annotation_root = self.root / lesion_folder / inner_folder / f"Annotation_{self.lesion_class}"

        id_set = set(self.image_ids)
        records = []
        if not image_root.exists():
            return records

        for root_dir, _, files in os.walk(image_root):
            visit_name = Path(root_dir).name
            for file in files:
                if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                stem = Path(file).stem
                full_id = f"eophtha_{self.lesion_class}:{stem}"
                if id_set and full_id not in id_set:
                    continue
                mask_name = f"{stem}.png" if self.lesion_class == "MA" else f"{stem}_EX.png"
                records.append(
                    {
                        "image_id": full_id,
                        "image_path": Path(root_dir) / file,
                        "mask_path": annotation_root / visit_name / mask_name,
                    }
                )
        return records

    def _read_mask(self, mask_path: Path, h: int, w: int) -> np.ndarray:
        if not mask_path.exists():
            return np.zeros((h, w), dtype=np.uint8)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return np.zeros((h, w), dtype=np.uint8)
        return (mask > 0).astype(np.uint8) * 255

    def _crop_patch(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = mask.shape[:2]
        ps = min(self.patch_size, h, w)
        use_lesion = np.random.rand() < self.lesion_patch_prob and np.any(mask > 0)

        if use_lesion:
            ys, xs = np.where(mask > 0)
            pick = np.random.randint(0, len(xs))
            px, py = int(xs[pick]), int(ys[pick])
            
            # Place the chosen pixel randomly within the patch, leaving a margin
            margin = 10
            x_min = max(0, px - ps + margin)
            x_max = min(w - ps, px - margin)
            if x_max >= x_min:
                x1 = np.random.randint(x_min, x_max + 1)
            else:
                x1 = max(0, px - ps // 2)
                
            y_min = max(0, py - ps + margin)
            y_max = min(h - ps, py - margin)
            if y_max >= y_min:
                y1 = np.random.randint(y_min, y_max + 1)
            else:
                y1 = max(0, py - ps // 2)
        else:
            x1 = np.random.randint(0, w - ps + 1) if w >= ps else 0
            y1 = np.random.randint(0, h - ps + 1) if h >= ps else 0

        return image[y1:y1 + ps, x1:x1 + ps], mask[y1:y1 + ps, x1:x1 + ps]


def _discover_ids(dataset_config: Dict[str, Any], lesion_class: str) -> List[str]:
    root = Path(dataset_config["root"])
    if dataset_config["name"] == "eophtha":
        if lesion_class not in {"MA", "EX"}:
            return []
        image_root = root / f"e_ophtha_{lesion_class}" / f"e_optha_{lesion_class}" / lesion_class
        ids = []
        if image_root.exists():
            for _, _, files in os.walk(image_root):
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        ids.append(f"eophtha_{lesion_class}:{Path(file).stem}")
        return sorted(set(ids))

    image_dir = root / dataset_config["train_images"]
    image_ext = dataset_config.get("image_ext", ".jpg")
    return sorted(f.stem for f in image_dir.iterdir() if f.suffix.lower() == image_ext.lower())


def split_binary_dataset(config: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    lesion_class = config["binary"]["target_class"]
    seed = config["training"].get("random_seed", 42)
    train_ratio = config["split"].get("train_ratio", 0.85)
    val_ratio = config["split"].get("val_ratio", 0.15)

    split = {}
    for dataset_config in config["dataset"]["datasets"]:
        name = dataset_config["name"]
        ids = _discover_ids(dataset_config, lesion_class)
        if not ids:
            split[name] = {"train": [], "val": [], "test": []}
            continue

        total_train_val = train_ratio + val_ratio
        train_ids, val_ids = train_test_split(
            ids,
            train_size=train_ratio / total_train_val,
            random_state=seed,
            shuffle=True,
        )

        if dataset_config.get("use_official_test", False):
            test_root = Path(dataset_config["root"]) / dataset_config["test_images"]
            image_ext = dataset_config.get("image_ext", ".jpg")
            test_ids = sorted(f.stem for f in test_root.iterdir() if f.suffix.lower() == image_ext.lower())
        else:
            test_ids = []

        split[name] = {
            "train": sorted(train_ids),
            "val": sorted(val_ids),
            "test": sorted(test_ids),
        }
    return split


def build_binary_dataloaders(config: Dict[str, Any]) -> Dict[str, DataLoader]:
    lesion_class = config["binary"]["target_class"]
    split = split_binary_dataset(config)
    enhancer = RetinalEnhancer(config["preprocessing"])
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)
    train_cfg = config["training"]
    patch_cfg = config.get("patch_training", {})

    phase_datasets = {"train": [], "val": [], "test": []}
    for dataset_config in config["dataset"]["datasets"]:
        name = dataset_config["name"]
        for phase in ["train", "val", "test"]:
            ids = split[name][phase]
            if not ids:
                continue
            is_train = phase == "train"
            is_training_split = not (phase == "test" and dataset_config.get("use_official_test", False))
            ds = BinaryLesionDataset(
                image_ids=ids,
                dataset_config=dataset_config,
                lesion_class=lesion_class,
                transform=train_transform if is_train else val_transform,
                enhancer=enhancer,
                is_training=is_training_split,
                patch_config=patch_cfg if is_train else None,
            )
            if len(ds) > 0:
                phase_datasets[phase].append(ds)

    loaders = {}
    for phase, datasets in phase_datasets.items():
        if not datasets:
            raise ValueError(f"No {phase} samples found for {lesion_class}")
        dataset = ConcatDataset(datasets)
        loaders[phase] = DataLoader(
            dataset,
            batch_size=train_cfg["batch_size"] if phase == "train" else 1,
            shuffle=(phase == "train"),
            num_workers=train_cfg.get("num_workers", 4),
            pin_memory=train_cfg.get("pin_memory", True),
            drop_last=(phase == "train"),
        )
    return loaders
