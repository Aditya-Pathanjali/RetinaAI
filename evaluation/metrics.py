import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from typing import Union
import cv2

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
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
       
        probs = torch.sigmoid(logits).cpu().numpy()
        targets_np = targets.cpu().numpy()

        batch_size = probs.shape[0]
        self.num_samples += batch_size

        for c in range(self.num_classes):
            thresh = self.thresholds[c]
            pred_c = (probs[:, c] > thresh).astype(np.uint8)
            
            # Post-processing: remove small connected components
            min_a = self.min_area[c]
            if min_a > 0:
                for b in range(batch_size):
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_c[b], connectivity=8)
                    for i in range(1, num_labels): # skip background 0
                        if stats[i, cv2.CC_STAT_AREA] < min_a:
                            pred_c[b][labels == i] = 0

            pred_c = pred_c.flatten()
            true_c = targets_np[:, c].flatten()

            self.tp[c] += np.sum(pred_c * true_c)
            self.fp[c] += np.sum(pred_c * (1 - true_c))
            self.fn[c] += np.sum((1 - pred_c) * true_c)
            self.tn[c] += np.sum((1 - pred_c) * (1 - true_c))

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
) -> Dict:
    
    model.eval()
    metrics = SegmentationMetrics(class_names=class_names, threshold=threshold, min_area=min_area)

    for images, masks, meta in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
            
        metrics.update(logits, masks)

    return metrics.compute()
