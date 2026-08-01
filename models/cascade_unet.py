import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import cv2
import numpy as np

from models.attention_unet import AttentionUNet, build_model

class DualStageCascadeUNet(nn.Module):
    """
    Dual-Stage Cascade Architecture for Retinal Lesion Segmentation & DR Grading.
    
    Stage 1: Global Context Proposal Network operating at 1024x1024 to generate coarse RoI proposals.
    Stage 2: High-Resolution Refinement Network operating on 512x512 un-downsampled crops.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Stage 1: Global Proposal Model
        self.global_proposal_net = build_model(config)
        
        # Stage 2: High-Res Lesion Refiner Model
        self.lesion_refiner_net = build_model(config)
        
    def extract_roi_crops(
        self,
        images: torch.Tensor,
        coarse_logits: torch.Tensor,
        crop_size: int = 512,
        top_k_crops: int = 4
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        Extracts high-resolution crop proposals centered on RoI activations.
        """
        B, C, H, W = images.shape
        probs = torch.sigmoid(coarse_logits)
        crops_list = []
        coords_list = []
        
        for b in range(B):
            # Sum probability map across classes to find candidate lesion locations
            prob_map = probs[b].sum(dim=0).detach().cpu().numpy()
            
            # Find coordinates of top activation peaks
            h_img, w_img = prob_map.shape
            peaks = np.argwhere(prob_map > 0.15)
            
            if len(peaks) == 0:
                # Fallback to center crops if no candidate proposals
                center_y, center_x = h_img // 2, w_img // 2
                coords = [(center_y, center_x)]
            else:
                # Sample top-k spread out peaks
                np.random.shuffle(peaks)
                coords = [(peaks[i][0], peaks[i][1]) for i in range(min(top_k_crops, len(peaks)))]
                
            for cy, cx in coords:
                y1 = max(0, min(H - crop_size, cy - crop_size // 2))
                x1 = max(0, min(W - crop_size, cx - crop_size // 2))
                y2 = y1 + crop_size
                x2 = x1 + crop_size
                
                crop = images[b:b+1, :, y1:y2, x1:x2]
                if crop.shape[2] != crop_size or crop.shape[3] != crop_size:
                    crop = F.interpolate(crop, size=(crop_size, crop_size), mode='bilinear', align_corners=False)
                    
                crops_list.append(crop)
                coords_list.append((y1, x1))
                
        batch_crops = torch.cat(crops_list, dim=0) if crops_list else images
        return batch_crops, coords_list

    def forward(
        self,
        images: torch.Tensor,
        is_training: bool = True
    ) -> torch.Tensor:
        # Stage 1: Global Context Coarse Pass
        global_logits = self.global_proposal_net(images)
        return global_logits

def build_cascade_model(config: Dict[str, Any]) -> DualStageCascadeUNet:
    return DualStageCascadeUNet(config)
