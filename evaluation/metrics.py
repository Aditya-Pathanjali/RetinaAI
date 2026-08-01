import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from typing import Union
import cv2

import torch.nn.functional as F

def sliding_window_inference(model, image, patch_size=512, stride=256, num_classes=1):
    b, c, h, w = image.shape
    device = image.device
    probs = torch.zeros((b, num_classes, h, w), device=device)
    counts = torch.zeros((b, num_classes, h, w), device=device)
    
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    
    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h))
        probs = F.pad(probs, (0, pad_w, 0, pad_h))
        counts = F.pad(counts, (0, pad_w, 0, pad_h))
        
    H, W = image.shape[2:]
    
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = image[:, :, y:y+patch_size, x:x+patch_size]
            with torch.no_grad():
                logits = model(patch)
                patch_probs = torch.sigmoid(logits)
            probs[:, :, y:y+patch_size, x:x+patch_size] += patch_probs
            counts[:, :, y:y+patch_size, x:x+patch_size] += 1
            
    probs = probs / counts.clamp(min=1)
    
    if pad_h > 0 or pad_w > 0:
        probs = probs[:, :, :h, :w]
        
    # convert probabilities back to logits for loss calculation and metrics
    return torch.logit(probs.clamp(1e-7, 1 - 1e-7))

class SegmentationMetrics:
    
    def __init__(
        self,
        class_names: List[str],
        threshold: Union[float, Dict[str, float]] = 0.5,
        smooth: float = 1e-6,
        min_area: Optional[Dict[str, int]] = None,
    ):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.smooth = smooth
        
        # Handle per-class thresholds
        if isinstance(threshold, dict):
            self.thresholds = [threshold.get(cls, 0.5) for cls in class_names]
        else:
            self.thresholds = [threshold] * self.num_classes
            
        # Minimum area for post-processing
        self.min_area = [min_area.get(cls, 0) if min_area else 0 for cls in class_names]
        
        self.reset()

    def reset(self) -> None:
        
        self.tp = np.zeros(self.num_classes, dtype=np.float64)
        self.fp = np.zeros(self.num_classes, dtype=np.float64)
        self.fn = np.zeros(self.num_classes, dtype=np.float64)
        self.tn = np.zeros(self.num_classes, dtype=np.float64)
        self.num_samples = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor, valid_classes: Optional[torch.Tensor] = None) -> None:
       
        probs = torch.sigmoid(logits).cpu().numpy()
        targets_np = targets.cpu().numpy()
        valid_classes_np = valid_classes.cpu().numpy() if valid_classes is not None else None

        batch_size = probs.shape[0]
        self.num_samples += batch_size

        for c in range(self.num_classes):
            thresh = self.thresholds[c]
            pred_c = (probs[:, c] > thresh).astype(np.uint8)
            
            # Post-processing: remove small connected components & filter non-circular vessel noise for MA
            min_a = self.min_area[c]
            cls_name = self.class_names[c]
            if min_a > 0 or cls_name == "MA":
                for b in range(batch_size):
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_c[b], connectivity=8)
                    for i in range(1, num_labels): # skip background 0
                        area = stats[i, cv2.CC_STAT_AREA]
                        if min_a > 0 and area < min_a:
                            pred_c[b][labels == i] = 0
                            continue
                            
                        if cls_name == "MA" and area > 0:
                            # Calculate circularity: (4 * pi * area) / (perimeter^2)
                            component_mask = (labels == i).astype(np.uint8)
                            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                perimeter = cv2.arcLength(contours[0], True)
                                if perimeter > 0:
                                    circularity = (4.0 * np.pi * area) / (perimeter ** 2)
                                    # Discard linear/vessel artifacts (low circularity)
                                    if circularity < 0.25:
                                        pred_c[b][labels == i] = 0

            for b in range(batch_size):
                if valid_classes_np is not None and valid_classes_np[b, c] == 0:
                    continue

                pred_c_flat = pred_c[b].flatten()
                true_c_flat = targets_np[b, c].flatten()

                self.tp[c] += np.sum(pred_c_flat * true_c_flat)
                self.fp[c] += np.sum(pred_c_flat * (1 - true_c_flat))
                self.fn[c] += np.sum((1 - pred_c_flat) * true_c_flat)
                self.tn[c] += np.sum((1 - pred_c_flat) * (1 - true_c_flat))

    def compute(self) -> Dict[str, Dict[str, float]]:
        
        results = {"per_class": {}, "mean": {}}

        all_dice = []
        all_iou = []
        all_precision = []
        all_recall = []
        all_specificity = []
        all_f1 = []

        for c in range(self.num_classes):
            tp, fp, fn, tn = self.tp[c], self.fp[c], self.fn[c], self.tn[c]

            dice = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
            iou = (tp + self.smooth) / (tp + fp + fn + self.smooth)
            precision = (tp + self.smooth) / (tp + fp + self.smooth)
            recall = (tp + self.smooth) / (tp + fn + self.smooth)
            specificity = (tn + self.smooth) / (tn + fp + self.smooth)
            f1 = (2 * precision * recall + self.smooth) / (precision + recall + self.smooth)

            cls_name = self.class_names[c]
            results["per_class"][cls_name] = {
                "dice": float(dice),
                "iou": float(iou),
                "precision": float(precision),
                "recall": float(recall),
                "sensitivity": float(recall),  # Alias
                "specificity": float(specificity),
                "f1": float(f1),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
            }

            all_dice.append(dice)
            all_iou.append(iou)
            all_precision.append(precision)
            all_recall.append(recall)
            all_specificity.append(specificity)
            all_f1.append(f1)

        # Mean across classes
        results["mean"] = {
            "dice": float(np.mean(all_dice)),
            "iou": float(np.mean(all_iou)),
            "precision": float(np.mean(all_precision)),
            "recall": float(np.mean(all_recall)),
            "sensitivity": float(np.mean(all_recall)),
            "specificity": float(np.mean(all_specificity)),
            "f1": float(np.mean(all_f1)),
        }

        return results

    def compute_dice(self) -> Tuple[float, Dict[str, float]]:
        
        per_class = {}
        for c in range(self.num_classes):
            tp, fp, fn = self.tp[c], self.fp[c], self.fn[c]
            dice = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
            per_class[self.class_names[c]] = float(dice)

        mean_dice = float(np.mean(list(per_class.values())))
        return mean_dice, per_class

    def compute_iou(self) -> Tuple[float, Dict[str, float]]:
        
        per_class = {}
        for c in range(self.num_classes):
            tp, fp, fn = self.tp[c], self.fp[c], self.fn[c]
            iou = (tp + self.smooth) / (tp + fp + fn + self.smooth)
            per_class[self.class_names[c]] = float(iou)

        mean_iou = float(np.mean(list(per_class.values())))
        return mean_iou, per_class

    def format_results(self, results: Optional[Dict] = None) -> str:
       
        if results is None:
            results = self.compute()

        lines = []
        lines.append("=" * 75)
        lines.append(f"{'Class':<16} {'Dice':>8} {'IoU':>8} {'Prec':>8} "
                      f"{'Recall':>8} {'Spec':>8} {'F1':>8}")
        lines.append("-" * 75)

        for cls_name in self.class_names:
            m = results["per_class"][cls_name]
            lines.append(
                f"{cls_name:<16} {m['dice']:>8.4f} {m['iou']:>8.4f} "
                f"{m['precision']:>8.4f} {m['recall']:>8.4f} "
                f"{m['specificity']:>8.4f} {m['f1']:>8.4f}"
            )

        lines.append("-" * 75)
        m = results["mean"]
        lines.append(
            f"{'MEAN':<16} {m['dice']:>8.4f} {m['iou']:>8.4f} "
            f"{m['precision']:>8.4f} {m['recall']:>8.4f} "
            f"{m['specificity']:>8.4f} {m['f1']:>8.4f}"
        )
        lines.append("=" * 75)

        return "\n".join(lines)

@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: List[str],
    device: torch.device,
    threshold: Union[float, Dict[str, float]] = 0.5,
    min_area: Optional[Dict[str, int]] = None,
    use_tta: bool = False,
) -> Dict:
    
    model.eval()
    metrics = SegmentationMetrics(class_names=class_names, threshold=threshold, min_area=min_area)

    for images, masks, meta in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        if use_tta:
            # Horizontal flip Test-Time Augmentation (TTA)
            logits = sliding_window_inference(model, images, num_classes=len(class_names))
            images_flipped = torch.flip(images, dims=[3])
            logits_flipped = sliding_window_inference(model, images_flipped, num_classes=len(class_names))
            logits_flipped_unflipped = torch.flip(logits_flipped, dims=[3])
            logits = (logits + logits_flipped_unflipped) / 2.0
        else:
            logits = sliding_window_inference(model, images, num_classes=len(class_names))
            
        valid_classes = None
        if isinstance(meta, dict) and "valid_classes" in meta:
            valid_classes = meta["valid_classes"].to(device)

        metrics.update(logits, masks, valid_classes)

    return metrics.compute()
