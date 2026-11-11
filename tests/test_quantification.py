import pytest
import numpy as np
import cv2
import torch
from utils.lesion_counter import count_lesions_single_mask, count_lesions_multiclass


def test_connected_component_lesion_count():
    """Verify connected components algorithm correctly counts distinct synthetic lesion blobs."""
    # Create blank 256x256 mask
    mask = np.zeros((256, 256), dtype=np.uint8)
    
    # Draw 4 distinct circular blobs
    cv2.circle(mask, (50, 50), 6, 255, -1)
    cv2.circle(mask, (100, 50), 8, 255, -1)
    cv2.circle(mask, (150, 150), 10, 255, -1)
    cv2.circle(mask, (200, 200), 5, 255, -1)
    
    count = count_lesions_single_mask(mask, min_area=0)
    assert count == 4


def test_noise_filtering_area_thresholds():
    """Verify sub-threshold speckle noise is filtered out by min_area thresholds."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    
    # Draw 1 valid lesion (radius 8 -> area ~200 px)
    cv2.circle(mask, (100, 100), 8, 255, -1)
    
    # Draw 2 noise pixels (single pixel dots -> area 1 px)
    mask[20, 20] = 255
    mask[30, 30] = 255
    
    # Without min_area threshold, count is 3
    count_all = count_lesions_single_mask(mask, min_area=0)
    assert count_all == 3
    
    # With min_area=10, the 2 noise pixels are discarded
    count_filtered = count_lesions_single_mask(mask, min_area=10)
    assert count_filtered == 1
