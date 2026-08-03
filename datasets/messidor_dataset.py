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
from preprocessing.transforms import get_val_transforms


class MessidorDataset(Dataset):
    """
    Dataset class for MESSIDOR-2 cross-dataset evaluation.
    Loads fundus images from C:/Dissertation/Datasets/MESSIDOR-2/IMAGES,
    applies CLAHE green-channel enhancement and 512x512 resizing.
    """

    def __init__(
        self,
        image_dir: Path,
        config: Dict[str, Any],
        transform=None,
        enhancer: Optional[RetinalEnhancer] = None,
        csv_file: Optional[Path] = None,
    ):
        self.image_dir = Path(image_dir)
        self.config = config
        self.transform = transform or get_val_transforms(config)
        self.enhancer = enhancer or RetinalEnhancer(config["preprocessing"])

        if not self.image_dir.exists():
            raise FileNotFoundError(f"MESSIDOR-2 image directory not found: {self.image_dir}")

        # Find all images (.png, .jpg, .tif)
        valid_exts = {".png", ".jpg", ".jpeg", ".tif"}
        self.image_files = sorted([
            f for f in self.image_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts
        ])

        if len(self.image_files) == 0:
            raise ValueError(f"No valid image files found in {self.image_dir}")

        # Optional metadata CSV loading
        self.labels = {}
        if csv_file and Path(csv_file).exists():
            try:
                df = pd.read_csv(csv_file)
                # Map column names if present
                id_col = [c for c in df.columns if "image" in c.lower() or "id" in c.lower()]
                grade_col = [c for c in df.columns if "dr" in c.lower() or "grade" in c.lower() or "diagnosis" in c.lower()]
                if id_col and grade_col:
                    for _, row in df.iterrows():
                        self.labels[str(row[id_col[0]])] = int(row[grade_col[0]])
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        image_path = self.image_files[idx]
        image_id = image_path.stem

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Failed to read MESSIDOR-2 image: {image_path}")

        # Apply CLAHE green-channel enhancement
        if self.enhancer is not None:
            image = self.enhancer.process(image)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            transformed = self.transform(image=image_rgb)
            image_tensor = transformed["image"]
        else:
            image_tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float() / 255.0

        label = self.labels.get(image_id, 0)
        meta = {
            "image_id": image_id,
            "image_path": str(image_path),
        }

        return image_tensor, label, meta


def build_messidor_dataloader(config: Dict[str, Any], batch_size: int = 8) -> DataLoader:
    messidor_cfg = config.get("messidor_dataset", {})
    root = Path(messidor_cfg.get("root", "C:/Dissertation/Datasets/MESSIDOR-2"))
    image_dir = root / messidor_cfg.get("images_dir", "IMAGES")
    csv_file = root / messidor_cfg.get("csv_file", "messidor-2.csv")

    dataset = MessidorDataset(
        image_dir=image_dir,
        config=config,
        csv_file=csv_file,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=messidor_cfg.get("num_workers", 2),
        pin_memory=True,
        drop_last=False,
    )

    return loader
