import os
import sys
import gc
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import load_config, get_device
from preprocessing.enhancer import RetinalEnhancer
from preprocessing.transforms import get_val_transforms
from models.attention_unet import build_model
from models.hybrid_classifier import build_classifier
from train_classifier import get_lesion_count_features


class RetinaAIInferenceEngine:
    """
    Production-grade, thread-safe inference engine for RetinaAI.
    Encapsulates Stage 1 Attention U-Net (Segmentation) and
    Stage 2 Hybrid DR Classifier (Severity Grading).
    """

    def __init__(
        self,
        config_path: str = "configs/config_messidor.yaml",
        seg_ckpt: Optional[str] = None,
        cls_ckpt: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            # Fallback to default configs
            self.config_path = PROJECT_ROOT / "configs" / "config_messidor.yaml"

        self.config = load_config(str(self.config_path))
        self.device = device or get_device()

        # Resolve model checkpoints
        ckpts = self.config.get("checkpoints", {})
        self.seg_ckpt_path = Path(seg_ckpt or ckpts.get("seg_checkpoint", "experiments/exp_ddr_attention_unet/checkpoints/best.pth"))
        self.cls_ckpt_path = Path(cls_ckpt or ckpts.get("cls_checkpoint", "experiments/exp_12_cls_hybrid_high_recall/checkpoints/best.pth"))

        if not self.seg_ckpt_path.is_absolute():
            self.seg_ckpt_path = PROJECT_ROOT / self.seg_ckpt_path
        if not self.cls_ckpt_path.is_absolute():
            self.cls_ckpt_path = PROJECT_ROOT / self.cls_ckpt_path

        # Initialize models
        self.enhancer = RetinalEnhancer(self.config["preprocessing"])
        self.val_transform = get_val_transforms(self.config)
        self.class_names = self.config["dataset"]["class_names"]

        # Load Stage 1 Attention U-Net
        self.seg_model = build_model(self.config).to(self.device)
        if self.seg_ckpt_path.exists():
            ckpt = torch.load(str(self.seg_ckpt_path), map_location=self.device, weights_only=False)
            self.seg_model.load_state_dict(ckpt["model_state_dict"])
            self.seg_model.eval()

        # Load Stage 2 Hybrid DR Classifier
        self.cls_model = build_classifier(self.config, in_channels=3).to(self.device)
        if self.cls_ckpt_path.exists():
            ckpt = torch.load(str(self.cls_ckpt_path), map_location=self.device, weights_only=False)
            self.cls_model.load_state_dict(ckpt["model_state_dict"])
            self.cls_model.eval()

        self.grade_labels = {
            0: ("Grade 0 — Healthy / No DR", "🟢 Non-Referable", "No clinical signs of diabetic retinopathy. Schedule routine 12-month re-screening."),
            1: ("Grade 1 — Mild Non-Proliferative DR", "🟢 Non-Referable", "Pinpoint microaneurysms detected. Annual monitoring recommended."),
            2: ("Grade 2 — Moderate Non-Proliferative DR", "🔴 Referable DR", "Hemorrhages and exudates present. Clinical ophthalmology referral recommended."),
            3: ("Grade 3 — Severe Non-Proliferative DR", "🔴 Referable DR", "Multiple hemorrhages and soft exudates. Urgent specialist referral required."),
            4: ("Grade 4 — Proliferative DR", "🔴 Referable DR", "Neovascularization and severe clinical risk. Immediate hospital intervention required."),
        }

    def validate_input_image(self, image_bgr: np.ndarray) -> Tuple[bool, str]:
        """Validates that uploaded image is a non-corrupted fundus photograph."""
        if image_bgr is None or image_bgr.size == 0:
            return False, "Uploaded image file is empty or corrupted."

        h, w, c = image_bgr.shape
        if c != 3:
            return False, f"Expected 3-channel RGB image, got {c} channels."

        if h < 256 or w < 256:
            return False, f"Image resolution ({w}x{h}) is too low for DR analysis. Minimum required is 256x256."

        # Check green-channel intensity presence (fundus images have strong green-channel structures)
        green_mean = np.mean(image_bgr[:, :, 1])
        if green_mean < 5 or green_mean > 250:
            return False, "Image does not appear to be a valid retinal fundus photograph."

        return True, "Valid fundus photograph."

    @torch.inference_mode()
    def process_image(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """Runs two-stage hybrid inference pipeline on input fundus image."""
        is_valid, err_msg = self.validate_input_image(image_bgr)
        if not is_valid:
            raise ValueError(err_msg)

        # 1. CLAHE Green-Channel Enhancement
        enhanced_bgr = self.enhancer.process(image_bgr)
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        raw_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # 2. Transform to PyTorch Tensor
        transformed = self.val_transform(image=enhanced_rgb)
        input_tensor = transformed["image"].unsqueeze(0).to(self.device)

        # 3. Stage 1 Segmentation Inference
        seg_logits = self.seg_model(input_tensor)
        seg_probs = torch.sigmoid(seg_logits)

        # 4. Extract 4D Lesion Counts
        thresholds = self.config["evaluation"]["threshold"]
        min_areas = self.config["evaluation"]["min_area"]
        counts_tensor = get_lesion_count_features(
            seg_probs=seg_probs,
            class_names=self.class_names,
            thresholds=thresholds,
            min_areas=min_areas,
            device=self.device,
        )
        counts = counts_tensor[0].cpu().numpy().astype(int).tolist()

        # 5. Generate 4-Color Lesion Mask Overlay
        overlay_bgr = self._create_lesion_overlay(
            base_bgr=cv2.resize(image_bgr, (512, 512)),
            seg_probs=seg_probs[0].cpu().numpy(),
            thresholds=thresholds,
        )

        # 6. Stage 2 Hybrid DR Classifier Inference
        cls_logits = self.cls_model(input_tensor, counts_tensor)
        probs = torch.softmax(cls_logits, dim=1)[0]

        # Apply class calibration weights if configured
        calib_w = self.config.get("evaluation", {}).get("class_calibration_weights", None)
        if calib_w is not None:
            w_tensor = torch.tensor(calib_w, device=self.device)
            adj_probs = probs * w_tensor
            pred_grade = int(torch.argmax(adj_probs).item())
        else:
            pred_grade = int(torch.argmax(probs).item())

        prob_dist = probs.cpu().numpy().tolist()

        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        label_title, referable_status, recommendation = self.grade_labels.get(
            pred_grade, (f"Grade {pred_grade}", "Referable DR", "Consult ophthalmologist.")
        )

        return {
            "predicted_grade": pred_grade,
            "grade_title": label_title,
            "referable_status": referable_status,
            "is_referable": pred_grade >= 1,
            "recommendation": recommendation,
            "probabilities": {f"Grade_{i}": float(prob_dist[i]) for i in range(5)},
            "confidence_pct": float(prob_dist[pred_grade] * 100),
            "lesion_counts": {
                "Microaneurysms (MA)": counts[0],
                "Hemorrhages (HE)": counts[1],
                "Hard Exudates (EX)": counts[2],
                "Soft Exudates (SE)": counts[3],
            },
            "raw_rgb": raw_rgb,
            "enhanced_rgb": enhanced_rgb,
            "overlay_bgr": overlay_bgr,
        }

    def _create_lesion_overlay(
        self,
        base_bgr: np.ndarray,
        seg_probs: np.ndarray,
        thresholds: Dict[str, float],
    ) -> np.ndarray:
        """Creates a color-coded 4-class lesion mask overlay on base fundus image."""
        overlay = base_bgr.copy().astype(np.float32)

        # Class colors in BGR: MA=Yellow, HE=Red, EX=Green, SE=Blue
        colors = {
            0: (0, 255, 255),  # MA - Yellow
            1: (0, 0, 255),    # HE - Red
            2: (0, 255, 0),    # EX - Green
            3: (255, 100, 0),  # SE - Light Blue / Cyan
        }

        for c, cls_name in enumerate(self.class_names):
            thresh = thresholds.get(cls_name, 0.25)
            mask = (seg_probs[c] >= thresh).astype(np.uint8)
            if mask.sum() == 0:
                continue

            color = colors[c]
            color_mask = np.zeros_like(base_bgr, dtype=np.uint8)
            color_mask[mask > 0] = color

            # Blend with 50% opacity
            overlay[mask > 0] = cv2.addWeighted(
                overlay[mask > 0], 0.5, color_mask[mask > 0].astype(np.float32), 0.5, 0
            )

        return np.clip(overlay, 0, 255).astype(np.uint8)
