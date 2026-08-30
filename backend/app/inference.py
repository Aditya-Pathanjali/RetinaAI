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

    def assess_image_quality(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """Calculates quantitative image sharpness (Laplacian variance) and exposure quality."""
        h, w = image_bgr.shape[:2]
        scale = 512.0 / max(h, w)
        img_thumb = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if max(h, w) > 512 else image_bgr
        gray = cv2.cvtColor(img_thumb, cv2.COLOR_BGR2GRAY)

        # 1. Laplacian Variance for Focus/Sharpness
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 2. Exposure / Brightness Check
        mean_brightness = float(np.mean(gray))

        # Categorize Quality Status
        if lap_var < 80.0:
            status = "Low Sharpness / Blurry"
            color = "#DC2626"
        elif mean_brightness < 25.0:
            status = "Underexposed"
            color = "#D97706"
        elif mean_brightness > 220.0:
            status = "Overexposed"
            color = "#D97706"
        elif lap_var < 180.0:
            status = "Acceptable"
            color = "#D97706"
        else:
            status = "Good"
            color = "#16A34A"

        return {
            "score": round(lap_var, 1),
            "brightness": round(mean_brightness, 1),
            "status": status,
            "status_color": color,
            "details": f"Focus Score: {lap_var:.1f} | Brightness: {mean_brightness:.1f}"
        }

    def validate_input_image(self, image_bgr: np.ndarray) -> Tuple[bool, str, Dict[str, Any]]:
        """Validates that uploaded image is a valid non-corrupted retinal fundus photograph using fast thumbnail analysis."""
        if image_bgr is None or image_bgr.size == 0:
            return False, "Uploaded image file is empty or corrupted.", {}

        h, w, c = image_bgr.shape
        if c != 3:
            return False, f"Invalid format: Expected 3-channel RGB color image, got {c} channels.", {}

        if h < 256 or w < 256:
            return False, f"Low resolution ({w}x{h}): Minimum required resolution for diagnostic analysis is 256x256.", {}

        # Downsample to 512px max dimension for fast validation checks (<5ms)
        scale = 512.0 / max(h, w)
        img_thumb = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if max(h, w) > 512 else image_bgr
        h_t, w_t = img_thumb.shape[:2]

        # 1. Circular Field-of-View (FOV) Geometry Verification
        gray_t = cv2.cvtColor(img_thumb, cv2.COLOR_BGR2GRAY)
        _, fov_mask = cv2.threshold(gray_t, 15, 255, cv2.THRESH_BINARY)

        total_pixels = h_t * w_t
        fov_pixels = np.count_nonzero(fov_mask)
        fov_ratio = fov_pixels / float(total_pixels)

        border_margin = int(min(h_t, w_t) * 0.05)
        border_pixels = np.concatenate([
            gray_t[:border_margin, :].flatten(),
            gray_t[-border_margin:, :].flatten(),
            gray_t[:, :border_margin].flatten(),
            gray_t[:, -border_margin:].flatten()
        ])
        border_mean = np.mean(border_pixels)

        # 2. Spectral Color Ratio Verification
        b, g, r = img_thumb[:, :, 0], img_thumb[:, :, 1], img_thumb[:, :, 2]
        if fov_pixels > 0:
            b_mean = np.mean(b[fov_mask > 0])
            g_mean = np.mean(g[fov_mask > 0])
            r_mean = np.mean(r[fov_mask > 0])
        else:
            b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)

        if r_mean < b_mean * 1.05 and g_mean < b_mean * 1.05:
            return False, "Non-fundus image rejected: Image spectral profile lacks characteristic retinal red/green illumination.", {}

        blue_red_ratio = b_mean / (r_mean + 1e-5)
        if blue_red_ratio > 0.70:
            return False, "Non-fundus image rejected: Excessive blue/cyan spectrum detected (natural photo or non-retinal image).", {}

        if fov_ratio > 0.96 and border_mean > 50:
            return False, "Non-fundus image rejected: Image lacks circular retinal field-of-view aperture mask.", {}

        g_std = np.std(g[fov_mask > 0]) if fov_pixels > 0 else np.std(g)
        if g_std < 8:
            return False, "Non-fundus image rejected: Insufficient retinal vascular texture detail.", {}

        quality_info = self.assess_image_quality(image_bgr)
        return True, "Valid retinal fundus photograph.", quality_info

    @torch.inference_mode()
    def process_image(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """Runs two-stage hybrid inference pipeline on input fundus image with exact timing measurement."""
        import time
        start_time = time.perf_counter()

        is_valid, err_msg, quality_info = self.validate_input_image(image_bgr)
        if not is_valid:
            raise ValueError(err_msg)

        # 1. CLAHE Green-Channel Enhancement
        enhanced_bgr = self.enhancer.process(image_bgr)
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        raw_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # 2. Transform to PyTorch Tensor
        transformed = self.val_transform(image=enhanced_rgb)
        input_tensor = transformed["image"].unsqueeze(0).to(self.device)

        # 3. Stage 1 Segmentation Inference (with optional FP16 autocast)
        use_cuda = (self.device.type == "cuda")
        with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
            seg_logits = self.seg_model(input_tensor)
            seg_probs = torch.sigmoid(seg_logits).float()

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

        # Compute binary Retinal Fundus Field-of-View (FOV) mask
        gray_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, fov_mask = cv2.threshold(gray_img, 15, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        fov_mask = cv2.erode(fov_mask, kernel, iterations=2)

        # 5. Resize segmentation probabilities to original image dimensions (h, w) and apply FOV mask
        h_orig, w_orig = image_bgr.shape[:2]
        seg_probs_np = seg_probs[0].cpu().numpy()
        seg_probs_full = np.zeros((4, h_orig, w_orig), dtype=np.float32)
        for c_idx in range(4):
            resized_prob = cv2.resize(seg_probs_np[c_idx], (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
            resized_prob[fov_mask == 0] = 0.0
            seg_probs_full[c_idx] = resized_prob

        # 6. Generate 4-Color Lesion Mask Overlay
        overlay_bgr = self._create_lesion_overlay(
            base_bgr=image_bgr,
            seg_probs=seg_probs_full,
            thresholds=thresholds,
        )

        # 7. Generate Pure Binary Segmentation View
        segmentation_bgr = np.zeros_like(image_bgr)
        colors_bgr = [(0, 255, 255), (0, 0, 255), (0, 255, 0), (255, 100, 0)]
        for c_idx, cls_name in enumerate(self.class_names):
            thresh = thresholds.get(cls_name, 0.30)
            mask = (seg_probs_full[c_idx] >= thresh)
            segmentation_bgr[mask] = colors_bgr[c_idx]
        segmentation_rgb = cv2.cvtColor(segmentation_bgr, cv2.COLOR_BGR2RGB)

        # 8. Generate Heatmap View
        max_prob = np.max(seg_probs_full, axis=0)
        max_prob[fov_mask == 0] = 0.0
        heatmap_norm = np.uint8(np.clip(max_prob * 255, 0, 255))
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        heatmap_bgr = cv2.addWeighted(image_bgr, 0.4, heatmap_color, 0.6, 0)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # 9. Stage 2 Hybrid DR Classifier Inference
        with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
            cls_logits = self.cls_model(input_tensor, counts_tensor)
            probs = torch.softmax(cls_logits.float(), dim=1)[0]

        pred_grade = int(torch.argmax(probs).item())
        prob_dist = probs.cpu().numpy().tolist()
        confidence_pct = float(prob_dist[pred_grade] * 100.0)

        elapsed_sec = round(time.perf_counter() - start_time, 2)

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
            "is_referable": pred_grade >= 2,
            "recommendation": recommendation,
            "probabilities": {f"Grade_{i}": float(prob_dist[i]) for i in range(5)},
            "confidence_pct": confidence_pct,
            "inference_time_sec": max(0.01, elapsed_sec),
            "quality_score": quality_info.get("score", 0.0),
            "quality_status": quality_info.get("status", "Good"),
            "quality_color": quality_info.get("status_color", "#16A34A"),
            "quality_details": quality_info.get("details", ""),
            "lesion_counts": {
                "Microaneurysms (MA)": counts[0],
                "Hemorrhages (HE)": counts[1],
                "Hard Exudates (EX)": counts[2],
                "Soft Exudates (SE)": counts[3],
            },
            "raw_rgb": raw_rgb,
            "enhanced_rgb": enhanced_rgb,
            "overlay_bgr": overlay_bgr,
            "segmentation_rgb": segmentation_rgb,
            "heatmap_rgb": heatmap_rgb,
            "seg_probs_full": seg_probs_full,
        }

    def _create_lesion_overlay(
        self,
        base_bgr: np.ndarray,
        seg_probs: np.ndarray,
        thresholds: Dict[str, float],
    ) -> np.ndarray:
        """Creates a color-coded 4-class lesion mask overlay on base fundus image maintaining exact aspect ratio."""
        overlay = base_bgr.copy().astype(np.float32)

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

            overlay[mask > 0] = cv2.addWeighted(
                overlay[mask > 0], 0.5, color_mask[mask > 0].astype(np.float32), 0.5, 0
            )

        return np.clip(overlay, 0, 255).astype(np.uint8)

