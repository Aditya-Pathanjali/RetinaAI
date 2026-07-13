
import cv2
import numpy as np
import torch
from typing import Union, Dict, List, Optional


def count_lesions_single_mask(
    mask: Union[np.ndarray, torch.Tensor],
    min_area: int = 0,
    connectivity: int = 8
) -> int:
    
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = mask

    mask_np = (mask_np > 0).astype(np.uint8) * 255

    if mask_np.sum() == 0:
        return 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_np, connectivity=connectivity
    )

    count = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            count += 1

    return count


def count_lesions_multiclass(
    masks: Union[np.ndarray, torch.Tensor],
    class_names: List[str],
    min_areas: Optional[Dict[str, int]] = None,
    connectivity: int = 8
) -> Dict[str, int]:
    
    if isinstance(masks, torch.Tensor):
        masks_np = masks.detach().cpu().numpy()
    else:
        masks_np = masks

    if masks_np.ndim != 3:
        raise ValueError(f"Expected masks to be of shape (C, H, W), got shape {masks_np.shape}")

    C = masks_np.shape[0]
    if C != len(class_names):
        raise ValueError(f"Number of mask channels ({C}) must match number of class names ({len(class_names)})")

    if min_areas is None:
        min_areas = {}

    counts = {}
    for c, cls_name in enumerate(class_names):
        min_a = min_areas.get(cls_name, 0)
        counts[cls_name] = count_lesions_single_mask(
            masks_np[c], min_area=min_a, connectivity=connectivity
        )

    return counts


def count_lesions_batch(
    masks: Union[np.ndarray, torch.Tensor],
    class_names: List[str],
    min_areas: Optional[Dict[str, int]] = None,
    connectivity: int = 8
) -> List[Dict[str, int]]:
    
    if isinstance(masks, torch.Tensor):
        masks_np = masks.detach().cpu().numpy()
    else:
        masks_np = masks

    if masks_np.ndim != 4:
        raise ValueError(f"Expected masks to be of shape (B, C, H, W), got shape {masks_np.shape}")

    B = masks_np.shape[0]

    batch_counts = []
    for b in range(B):
        counts = count_lesions_multiclass(
            masks_np[b], class_names=class_names, min_areas=min_areas, connectivity=connectivity
        )
        batch_counts.append(counts)

    return batch_counts
