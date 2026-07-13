
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/headless use
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch


# Colour-blind-friendly palette for the 4 lesion classes
CLASS_COLOURS = {
    "MA": (1.0, 0.2, 0.2, 0.7),   # Red — microaneurysms
    "HE": (0.2, 0.6, 1.0, 0.7),   # Blue — haemorrhages
    "EX": (1.0, 0.85, 0.0, 0.7),  # Yellow — hard exudates
    "SE": (0.3, 0.9, 0.3, 0.7),   # Green — soft exudates
}

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


class Visualizer:
    
    def __init__(
        self,
        output_dir: str = "outputs/plots",
        class_names: List[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names or ["MA", "HE", "EX", "SE"]

    def plot_class_distribution(
        self,
        stats: Dict[str, Dict[str, float]],
        filename: str = "class_distribution.png",
    ) -> str:
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        classes = list(stats.keys())
        colours = [CLASS_COLOURS.get(c, (0.5, 0.5, 0.5, 0.7))[:3] for c in classes]

        # 1. Present vs absent count
        present = [stats[c]["present_count"] for c in classes]
        absent = [stats[c]["absent_count"] for c in classes]
        x = np.arange(len(classes))
        width = 0.35
        axes[0].bar(x - width / 2, present, width, label="Present", color=colours, edgecolor="black")
        axes[0].bar(x + width / 2, absent, width, label="Absent", color="lightgrey", edgecolor="black")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(classes)
        axes[0].set_ylabel("Number of Images")
        axes[0].set_title("Lesion Presence per Class")
        axes[0].legend()

        # 2. Mean lesion pixel ratio
        mean_ratios = [stats[c]["mean_ratio"] * 100 for c in classes]
        axes[1].bar(classes, mean_ratios, color=colours, edgecolor="black")
        axes[1].set_ylabel("Mean Lesion Pixel %")
        axes[1].set_title("Mean Lesion Coverage")
        for i, v in enumerate(mean_ratios):
            axes[1].text(i, v + 0.01, f"{v:.3f}%", ha="center", fontsize=8)

        # 3. Total lesion pixels (log scale)
        total_pixels = [stats[c]["total_pixels"] for c in classes]
        axes[2].bar(classes, total_pixels, color=colours, edgecolor="black")
        axes[2].set_ylabel("Total Lesion Pixels")
        axes[2].set_title("Total Lesion Pixels (All Images)")
        axes[2].set_yscale("log")

        plt.suptitle("IDRiD Dataset — Class Imbalance Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()

        save_path = self.output_dir / filename
        plt.savefig(save_path)
        plt.close()
        return str(save_path)

    def plot_sample_with_masks(
        self,
        image: np.ndarray,
        masks: Dict[str, np.ndarray],
        image_id: str = "",
        filename: str = "sample_overlay.png",
    ) -> str:
        
        n_masks = len(masks)
        fig, axes = plt.subplots(1, n_masks + 2, figsize=(4 * (n_masks + 2), 4))

        # Original image
        axes[0].imshow(image)
        axes[0].set_title(f"Original\n{image_id}")
        axes[0].axis("off")

        # Individual masks
        for i, (cls_name, mask) in enumerate(masks.items()):
            colour = CLASS_COLOURS.get(cls_name, (1, 1, 1, 0.7))
            overlay = image.copy()
            mask_bool = mask > 0
            overlay[mask_bool] = (
                np.array(colour[:3]) * 255 * 0.6 +
                overlay[mask_bool] * 0.4
            ).astype(np.uint8)

            axes[i + 1].imshow(overlay)
            pixel_pct = mask_bool.sum() / mask_bool.size * 100
            axes[i + 1].set_title(f"{cls_name}\n({pixel_pct:.4f}%)")
            axes[i + 1].axis("off")

        # Combined overlay
        combined = image.copy().astype(np.float32)
        for cls_name, mask in masks.items():
            colour = CLASS_COLOURS.get(cls_name, (1, 1, 1, 0.7))
            mask_bool = mask > 0
            for c_idx in range(3):
                combined[:, :, c_idx] = np.where(
                    mask_bool,
                    combined[:, :, c_idx] * 0.4 + colour[c_idx] * 255 * 0.6,
                    combined[:, :, c_idx],
                )

        axes[-1].imshow(combined.astype(np.uint8))
        axes[-1].set_title("All Lesions\nOverlay")
        axes[-1].axis("off")

        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path)
        plt.close()
        return str(save_path)

    
    def plot_preprocessing_stages(
        self,
        stages: Dict[str, np.ndarray],
        image_id: str = "",
        filename: str = "preprocessing_stages.png",
    ) -> str:
        
        n = len(stages)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

        if n == 1:
            axes = [axes]

        for i, (name, img) in enumerate(stages.items()):
            # Convert BGR → RGB for display
            if img.ndim == 3 and img.shape[2] == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img

            axes[i].imshow(img_rgb)
            axes[i].set_title(name.replace("_", " ").title(), fontsize=9)
            axes[i].axis("off")

        plt.suptitle(f"Preprocessing Pipeline — {image_id}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path)
        plt.close()
        return str(save_path)

    # ------------------------------------------------------------------
    # Augmentation Visualization
    # ------------------------------------------------------------------

    def plot_augmentation_grid(
        self,
        original_image: np.ndarray,
        original_mask: np.ndarray,
        augmented_pairs: List[Tuple[np.ndarray, np.ndarray]],
        filename: str = "augmentation_grid.png",
    ) -> str:
        """
        Grid showing original image+mask and multiple augmented versions.

        Args:
            original_image: RGB uint8 (H, W, 3).
            original_mask:  Binary mask (H, W) or (H, W, C).
            augmented_pairs: List of (augmented_image, augmented_mask) tuples.
            filename: Output filename.

        Returns:
            Path to saved figure.
        """
        n_aug = len(augmented_pairs)
        n_cols = min(n_aug + 1, 5)
        n_rows = 2  # Row 1: images, Row 2: masks

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 8))

        # Original
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")

        mask_display = original_mask if original_mask.ndim == 2 else original_mask.max(axis=-1)
        axes[1, 0].imshow(mask_display, cmap="hot")
        axes[1, 0].set_title("Original Mask")
        axes[1, 0].axis("off")

        # Augmented versions
        for i, (aug_img, aug_mask) in enumerate(augmented_pairs[:n_cols - 1]):
            axes[0, i + 1].imshow(aug_img)
            axes[0, i + 1].set_title(f"Aug #{i + 1}")
            axes[0, i + 1].axis("off")

            m_display = aug_mask if aug_mask.ndim == 2 else aug_mask.max(axis=-1)
            axes[1, i + 1].imshow(m_display, cmap="hot")
            axes[1, i + 1].set_title(f"Mask #{i + 1}")
            axes[1, i + 1].axis("off")

        plt.suptitle("Data Augmentation — Before/After", fontsize=13, fontweight="bold")
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path)
        plt.close()
        return str(save_path)

    # ------------------------------------------------------------------
    # Training Curves
    # ------------------------------------------------------------------

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        filename: str = "training_curves.png",
    ) -> str:
        """
        Publication-quality training and validation curves.

        Generates a 2×2 grid:
        - Loss curves (train + val)
        - Dice score curve
        - IoU curve
        - Per-class Dice curves
        - Learning rate schedule

        Args:
            history: Training history dict from Trainer.fit().
            filename: Output filename.

        Returns:
            Path to saved figure.
        """
        epochs = history["epoch"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Loss
        axes[0, 0].plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=1.5)
        axes[0, 0].plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=1.5)
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training & Validation Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Mean Dice + IoU
        axes[0, 1].plot(epochs, history["val_dice"], "g-", label="Val Dice", linewidth=1.5)
        axes[0, 1].plot(epochs, history["val_iou"], "m-", label="Val IoU", linewidth=1.5)
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Score")
        axes[0, 1].set_title("Validation Dice & IoU")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)

        # 3. Per-class Dice
        for cls in self.class_names:
            key = f"val_dice_{cls}"
            if key in history:
                colour = CLASS_COLOURS.get(cls, (0.5, 0.5, 0.5, 1.0))
                axes[1, 0].plot(
                    epochs, history[key],
                    label=cls, linewidth=1.5,
                    color=colour[:3],
                )
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Dice Score")
        axes[1, 0].set_title("Per-Class Validation Dice")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)

        # 4. Learning rate
        axes[1, 1].plot(epochs, history["lr"], "k-", linewidth=1.5)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].set_title("Learning Rate Schedule")
        axes[1, 1].set_yscale("log")
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle("RetinaAI — Training Progress", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path)
        plt.close()
        return str(save_path)

    # ------------------------------------------------------------------
    # Prediction Visualization
    # ------------------------------------------------------------------

    def plot_predictions(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        preds: torch.Tensor,
        image_ids: List[str] = None,
        num_samples: int = 4,
        filename: str = "predictions.png",
    ) -> str:
        """
        Multi-panel figure comparing predictions vs ground truth.

        Layout per sample row:
        [Input Image | GT Overlay | Pred Overlay | Per-class comparison]

        Args:
            images: (B, 3, H, W) — normalised input tensors.
            masks:  (B, C, H, W) — ground truth binary masks.
            preds:  (B, C, H, W) — predicted logits (before sigmoid).
            image_ids: Optional list of image identifiers.
            num_samples: Number of samples to show.
            filename: Output filename.

        Returns:
            Path to saved figure.
        """
        num_samples = min(num_samples, images.shape[0])
        n_classes = masks.shape[1]
        n_cols = 3 + n_classes  # image, GT, pred, per-class

        fig, axes = plt.subplots(num_samples, n_cols, figsize=(3.5 * n_cols, 3.5 * num_samples))
        if num_samples == 1:
            axes = axes[np.newaxis, :]

        pred_probs = torch.sigmoid(preds).cpu().numpy()
        masks_np = masks.cpu().numpy()

        for i in range(num_samples):
            # Denormalise image for display
            img = self._denormalize(images[i])

            # Column 0: Input image
            axes[i, 0].imshow(img)
            title = f"Input {image_ids[i]}" if image_ids else f"Sample {i}"
            axes[i, 0].set_title(title, fontsize=9)
            axes[i, 0].axis("off")

            # Column 1: Ground truth overlay
            gt_overlay = self._create_overlay(img, masks_np[i])
            axes[i, 1].imshow(gt_overlay)
            axes[i, 1].set_title("Ground Truth", fontsize=9)
            axes[i, 1].axis("off")

            # Column 2: Prediction overlay
            pred_binary = (pred_probs[i] > 0.5).astype(np.float32)
            pred_overlay = self._create_overlay(img, pred_binary)
            axes[i, 2].imshow(pred_overlay)
            axes[i, 2].set_title("Prediction", fontsize=9)
            axes[i, 2].axis("off")

            # Columns 3+: Per-class comparison (GT top, Pred bottom)
            for c in range(n_classes):
                cls_name = self.class_names[c] if c < len(self.class_names) else f"C{c}"
                colour = CLASS_COLOURS.get(cls_name, (1, 1, 1, 0.7))

                # Create side-by-side: GT (left) and Pred (right)
                combined = np.zeros((masks_np.shape[2], masks_np.shape[3] * 2), dtype=np.float32)
                combined[:, :masks_np.shape[3]] = masks_np[i, c]
                combined[:, masks_np.shape[3]:] = pred_probs[i, c]

                axes[i, 3 + c].imshow(combined, cmap="hot", vmin=0, vmax=1)
                dice = self._compute_single_dice(masks_np[i, c], pred_binary[c])
                axes[i, 3 + c].set_title(f"{cls_name}\nDice: {dice:.3f}", fontsize=8)
                axes[i, 3 + c].axis("off")

        plt.suptitle("Predictions vs Ground Truth", fontsize=13, fontweight="bold")
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path)
        plt.close()
        return str(save_path)

    def plot_best_worst_predictions(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray],
        preds: List[np.ndarray],
        dice_scores: List[float],
        image_ids: List[str],
        n: int = 3,
        filename: str = "best_worst.png",
    ) -> str:
        """
        Show the N best and N worst predictions by Dice score.

        Args:
            images:      List of RGB images.
            masks:       List of multi-class masks (C, H, W).
            preds:       List of prediction masks (C, H, W).
            dice_scores: List of mean Dice scores per sample.
            image_ids:   List of image identifiers.
            n:           Number of best/worst to show.
            filename:    Output filename.

        Returns:
            Path to saved figure.
        """
        indices = np.argsort(dice_scores)
        worst_idx = indices[:n]
        best_idx = indices[-n:][::-1]

        fig, axes = plt.subplots(2 * n, 3, figsize=(12, 4 * 2 * n))

        for row, idx in enumerate(list(best_idx) + list(worst_idx)):
            img = images[idx]
            gt_overlay = self._create_overlay(img, masks[idx])
            pred_overlay = self._create_overlay(img, preds[idx])

            category = "BEST" if row < n else "WORST"
            dice = dice_scores[idx]

            axes[row, 0].imshow(img)
            axes[row, 0].set_title(f"{category} — {image_ids[idx]}\nDice: {dice:.4f}", fontsize=9)
            axes[row, 0].axis("off")

            axes[row, 1].imshow(gt_overlay)
            axes[row, 1].set_title("Ground Truth")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(pred_overlay)
            axes[row, 2].set_title("Prediction")
            axes[row, 2].axis("off")

        plt.suptitle("Best & Worst Predictions", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path)
        plt.close()
        return str(save_path)

    # ------------------------------------------------------------------
    # Attention Map Visualization
    # ------------------------------------------------------------------

    def plot_attention_maps(
        self,
        image: np.ndarray,
        attention_maps: List[torch.Tensor],
        image_id: str = "",
        filename: str = "attention_maps.png",
    ) -> str:
        """
        Visualise attention maps from each decoder level.

        Args:
            image:          RGB image (H, W, 3).
            attention_maps: List of attention tensors from model.get_attention_maps().
            image_id:       Image identifier.
            filename:       Output filename.

        Returns:
            Path to saved figure.
        """
        valid_maps = [m for m in attention_maps if m is not None]
        n = len(valid_maps) + 1

        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

        axes[0].imshow(image)
        axes[0].set_title(f"Input — {image_id}")
        axes[0].axis("off")

        for i, att_map in enumerate(valid_maps):
            # att_map: (1, 1, H_i, W_i) — take first sample
            att = att_map[0, 0].cpu().numpy()

            # Resize to image size for overlay
            att_resized = cv2.resize(att, (image.shape[1], image.shape[0]))
            att_resized = (att_resized - att_resized.min()) / (att_resized.max() - att_resized.min() + 1e-8)

            axes[i + 1].imshow(image)
            axes[i + 1].imshow(att_resized, cmap="jet", alpha=0.5)
            axes[i + 1].set_title(f"Attention Level {i + 1}\n({att.shape[0]}×{att.shape[1]})")
            axes[i + 1].axis("off")

        plt.suptitle("Attention Gate Activations", fontsize=13, fontweight="bold")
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path)
        plt.close()
        return str(save_path)

    # ------------------------------------------------------------------
    # Metric Tables
    # ------------------------------------------------------------------

    def plot_metric_table(
        self,
        results: Dict,
        filename: str = "metric_table.png",
    ) -> str:
        """
        Render evaluation metrics as a publication-quality table image.

        Args:
            results: Output from SegmentationMetrics.compute().
            filename: Output filename.

        Returns:
            Path to saved figure.
        """
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.axis("off")

        headers = ["Class", "Dice", "IoU", "Precision", "Recall", "Specificity", "F1"]
        rows = []

        for cls_name in self.class_names:
            m = results["per_class"][cls_name]
            rows.append([
                cls_name,
                f"{m['dice']:.4f}",
                f"{m['iou']:.4f}",
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['specificity']:.4f}",
                f"{m['f1']:.4f}",
            ])

        m = results["mean"]
        rows.append([
            "MEAN",
            f"{m['dice']:.4f}",
            f"{m['iou']:.4f}",
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['specificity']:.4f}",
            f"{m['f1']:.4f}",
        ])

        table = ax.table(
            cellText=rows,
            colLabels=headers,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Style header
        for j in range(len(headers)):
            table[0, j].set_facecolor("#2c3e50")
            table[0, j].set_text_props(color="white", fontweight="bold")

        # Style mean row
        for j in range(len(headers)):
            table[len(rows), j].set_facecolor("#ecf0f1")
            table[len(rows), j].set_text_props(fontweight="bold")

        plt.title("Segmentation Evaluation Metrics", fontsize=14, fontweight="bold", pad=20)
        save_path = self.output_dir / filename
        plt.savefig(save_path)
        plt.close()
        return str(save_path)

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _denormalize(
        tensor: torch.Tensor,
        mean: Tuple = (0.485, 0.456, 0.406),
        std: Tuple = (0.229, 0.224, 0.225),
    ) -> np.ndarray:
        """
        Reverse ImageNet normalisation and convert to uint8 RGB.

        Args:
            tensor: (3, H, W) normalised float tensor.
            mean:   Channel means used during normalisation.
            std:    Channel stds used during normalisation.

        Returns:
            RGB uint8 numpy array (H, W, 3).
        """
        img = tensor.cpu().clone()
        for c in range(3):
            img[c] = img[c] * std[c] + mean[c]
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img

    def _create_overlay(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Create an overlay of multi-class masks on an image.

        Args:
            image: RGB uint8 (H, W, 3).
            masks: Binary masks (C, H, W) or (H, W) for single class.
            alpha: Overlay transparency.

        Returns:
            Overlaid image as uint8 RGB.
        """
        overlay = image.copy().astype(np.float32)

        if masks.ndim == 2:
            masks = masks[np.newaxis, :, :]

        for c in range(masks.shape[0]):
            cls_name = self.class_names[c] if c < len(self.class_names) else f"C{c}"
            colour = CLASS_COLOURS.get(cls_name, (1, 1, 1, 0.7))
            mask_bool = masks[c] > 0.5

            for ch in range(3):
                overlay[:, :, ch] = np.where(
                    mask_bool,
                    overlay[:, :, ch] * (1 - alpha) + colour[ch] * 255 * alpha,
                    overlay[:, :, ch],
                )

        return np.clip(overlay, 0, 255).astype(np.uint8)

    @staticmethod
    def _compute_single_dice(gt: np.ndarray, pred: np.ndarray, smooth: float = 1e-6) -> float:
        """Compute Dice coefficient between two binary masks."""
        gt_flat = gt.flatten()
        pred_flat = pred.flatten()
        intersection = (gt_flat * pred_flat).sum()
        return float((2 * intersection + smooth) / (gt_flat.sum() + pred_flat.sum() + smooth))
