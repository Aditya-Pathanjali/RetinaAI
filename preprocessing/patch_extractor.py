import random
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

class HighResPatchExtractor:
    """
    Extracts high-resolution patches from full-size retinal fundus images
    centered around lesion annotations without downsampling blurs.
    """
    def __init__(self, patch_size: int = 512, lesion_bias: float = 0.75):
        self.patch_size = patch_size
        self.lesion_bias = lesion_bias

    def extract_patches_from_image_and_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        num_patches: int = 4
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        h, w = image.shape[:2]
        patches = []
        
        # Find positive lesion pixel coordinates (nonzero coordinates in mask)
        if len(mask.shape) == 3:
            lesion_pixels = np.argwhere(mask.sum(axis=-1) > 0)
        else:
            lesion_pixels = np.argwhere(mask > 0)
            
        has_lesions = len(lesion_pixels) > 0
        
        for _ in range(num_patches):
            sample_lesion = has_lesions and (random.random() < self.lesion_bias)
            
            if sample_lesion:
                # Sample center coordinate around a random lesion pixel
                cy, cx = lesion_pixels[random.randint(0, len(lesion_pixels) - 1)]
                # Add slight random jitter
                cy += random.randint(-self.patch_size // 4, self.patch_size // 4)
                cx += random.randint(-self.patch_size // 4, self.patch_size // 4)
            else:
                # Random crop
                cy = random.randint(self.patch_size // 2, max(self.patch_size // 2 + 1, h - self.patch_size // 2))
                cx = random.randint(self.patch_size // 2, max(self.patch_size // 2 + 1, w - self.patch_size // 2))
                
            # Clamp boundaries
            y1 = max(0, min(h - self.patch_size, cy - self.patch_size // 2))
            x1 = max(0, min(w - self.patch_size, cx - self.patch_size // 2))
            y2 = y1 + self.patch_size
            x2 = x1 + self.patch_size
            
            patch_img = image[y1:y2, x1:x2]
            patch_mask = mask[y1:y2, x1:x2]
            
            # Handle potential edge padding if image is smaller than patch size
            if patch_img.shape[0] < self.patch_size or patch_img.shape[1] < self.patch_size:
                patch_img = cv2.resize(patch_img, (self.patch_size, self.patch_size))
                if len(patch_mask.shape) == 3:
                    patch_mask = cv2.resize(patch_mask, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
                else:
                    patch_mask = cv2.resize(patch_mask, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
                    
            patches.append((patch_img, patch_mask))
            
        return patches
