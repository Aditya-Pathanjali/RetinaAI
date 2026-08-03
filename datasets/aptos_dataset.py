import os
import json
import cv2
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader

from preprocessing.enhancer import RetinalEnhancer
from preprocessing.transforms import get_train_transforms, get_val_transforms


class APTOSDataset(Dataset):

    def __init__(
        self,
        csv_file: Path,
        image_dir: Path,
        config: Dict[str, Any],
        transform=None,
        enhancer: Optional[RetinalEnhancer] = None,
    ):
        self.csv_file = Path(csv_file)
        self.image_dir = Path(image_dir)
        self.config = config
        self.transform = transform
        self.enhancer = enhancer

        if not self.csv_file.exists():
            raise FileNotFoundError(f"APTOS metadata CSV file not found: {self.csv_file}")
        
        self.df = pd.read_csv(self.csv_file)
        
        required_cols = {"id_code", "diagnosis"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"CSV metadata must contain {required_cols} columns. Got: {self.df.columns.tolist()}")

        aptos_cfg = config.get("aptos_dataset", {})
        self.image_ext = aptos_cfg.get("image_ext", ".png")
        
        self.precomputed_counts = None
        counts_path = Path(aptos_cfg["root"]) / "lesion_counts.json"
        if counts_path.exists():
            with open(counts_path, "r") as f:
                self.precomputed_counts = json.load(f)
                
        self.use_pre_enhanced = False
        enhanced_dir = Path(aptos_cfg["root"]) / "enhanced_images" / image_dir.relative_to(Path(aptos_cfg["root"]))
        if enhanced_dir.exists():
            self.image_dir = enhanced_dir
            self.use_pre_enhanced = True

        self.available_files = set()
        if self.image_dir.exists():
            self.available_files = {f.name for f in self.image_dir.iterdir() if f.is_file()}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        row = self.df.iloc[idx]
        image_id = str(row["id_code"])
        label = int(row["diagnosis"])

        if self.use_pre_enhanced:
            jpg_name = f"{image_id}.jpg"
            if jpg_name in self.available_files:
                image_path = self.image_dir / jpg_name
            else:
                image_path = self.image_dir / f"{image_id}.png"
        else:
            image_path = self.image_dir / f"{image_id}{self.image_ext}"

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Failed to read APTOS image: {image_path}")

        if not self.use_pre_enhanced and self.enhancer is not None:
            image = self.enhancer.process(image)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            transformed = self.transform(image=image_rgb)
            image_tensor = transformed["image"]
        else:
            image_tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float() / 255.0

        meta = {
            "image_id": image_id,
            "image_path": str(image_path),
        }
        
        if self.precomputed_counts is not None and image_id in self.precomputed_counts:
            class_names = self.config["dataset"]["class_names"]
            counts_list = [float(self.precomputed_counts[image_id].get(cls, 0)) for cls in class_names]
            meta["lesion_counts"] = torch.tensor(counts_list, dtype=torch.float32)

        return image_tensor, label, meta


def build_aptos_dataloaders(config: Dict[str, Any]) -> Dict[str, DataLoader]:
    
    aptos_cfg = config.get("aptos_dataset", {})
    train_cfg = config.get("training", {})

    root = Path(aptos_cfg["root"])
    
    train_csv = root / aptos_cfg["train_csv"]
    val_csv = root / aptos_cfg["val_csv"]
    test_csv = root / aptos_cfg["test_csv"]

    train_image_dir = root / aptos_cfg["train_images"]
    val_image_dir = root / aptos_cfg["val_images"]
    test_image_dir = root / aptos_cfg["test_images"]

    enhancer = RetinalEnhancer(config["preprocessing"])
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)
    train_dataset = APTOSDataset(
        csv_file=train_csv,
        image_dir=train_image_dir,
        config=config,
        transform=train_transform,
        enhancer=enhancer,
    )

    val_dataset = APTOSDataset(
        csv_file=val_csv,
        image_dir=val_image_dir,
        config=config,
        transform=val_transform,
        enhancer=enhancer,
    )

    test_dataset = APTOSDataset(
        csv_file=test_csv,
        image_dir=test_image_dir,
        config=config,
        transform=val_transform,
        enhancer=enhancer,
    )

    # Dataloaders
    batch_size = train_cfg.get("batch_size", 4)
    num_workers = train_cfg.get("num_workers", 4)
    pin_memory = train_cfg.get("pin_memory", True)

    use_weighted_sampler = train_cfg.get("use_weighted_sampler", False)
    sampler = None
    shuffle = True
    if use_weighted_sampler:
        from torch.utils.data import WeightedRandomSampler
        targets = train_dataset.df["diagnosis"].values
        class_sample_counts = np.bincount(targets)
        weight = 1.0 / (class_sample_counts + 1e-6)
        samples_weight = torch.from_numpy(weight[targets]).float()
        sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)
        shuffle = False

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
    }

    return loaders
