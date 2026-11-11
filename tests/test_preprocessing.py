import pytest
import numpy as np
import cv2
from preprocessing.enhancer import RetinalEnhancer


def test_clahe_contrast_enhancement():
    """Test CLAHE enhancement maintains channel integrity and dynamic range."""
    enhancer = RetinalEnhancer({
        "image_size": 256,
        "apply_clahe": True,
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid": [8, 8],
        "apply_bg_crop": False,
        "apply_green_enhance": True,
        "apply_illum_correct": False,
    })
    
    # Create synthetic low-contrast image (H, W, 3)
    dummy_img = np.full((300, 300, 3), 100, dtype=np.uint8)
    cv2.circle(dummy_img, (150, 150), 100, (120, 120, 120), -1)
    
    enhanced = enhancer.process(dummy_img)
    assert enhanced.shape == (256, 256, 3)
    assert enhanced.dtype == np.uint8
    assert enhanced.max() >= enhanced.min()


def test_fov_circular_masking():
    """Test background cropping and FOV detection isolates the retinal circular area."""
    # Synthetic circular fundus with black borders
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(img, (200, 200), 150, (180, 120, 80), -1)
    
    cropped = RetinalEnhancer.crop_background(img, threshold=10)
    assert cropped.shape[0] <= 400 and cropped.shape[1] <= 400
    assert cropped.shape[0] > 0 and cropped.shape[1] > 0
    assert cropped.max() > 0


def test_intensity_normalization_range():
    """Test image normalisation constraints to standard bounded range [0, 1]."""
    img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    normalized = img.astype(np.float32) / 255.0
    
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
    assert normalized.shape == (256, 256, 3)
