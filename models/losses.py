
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List


#  Diceloss
class DiceLoss(nn.Module):

    def __init__(self, smooth: float = 1.0, class_weights: Optional[List[float]] = None):
        super().__init__()
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions: (B, C, H*W)
        probs_flat = probs.view(probs.size(0), probs.size(1), -1)
        target_flat = targets.view(targets.size(0), targets.size(1), -1)

        # Per-class Dice
        intersection = (probs_flat * target_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + target_flat.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        if self.class_weights is not None:
            weights = torch.tensor(self.class_weights, device=logits.device, dtype=logits.dtype)
            # Weighted average over classes, then mean over batch
            dice = (dice * weights.unsqueeze(0)).sum(dim=1) / weights.sum()
            return 1.0 - dice.mean()
        else:
            return 1.0 - dice.mean()


#BCELoss (wrapper)
class BCELoss(nn.Module):

    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.pos_weight is not None:
            pw = self.pos_weight.to(logits.device)
            return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw.view(1, -1, 1, 1))
        return F.binary_cross_entropy_with_logits(logits, targets)


#BCE_DICE Hybrid loss
class BCEDiceLoss(nn.Module):

    def __init__(
        self,
        bce_weight: float = 0.4,
        dice_weight: float = 0.6,
        smooth: float = 1.0,
        class_weights: Optional[List[float]] = None,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = BCELoss(pos_weight=pos_weight)
        self.dice = DiceLoss(smooth=smooth, class_weights=class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (self.bce_weight * self.bce(logits, targets) +
                self.dice_weight * self.dice(logits, targets))


#Focal Loss (Lin et al., 2017) adapted for segmentation.
class FocalLoss(nn.Module):

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        loss = focal_weight * bce_loss
        return loss.mean()


#Tversky Loss (Salehi et al., 2017) — generalisation of Dice loss.

class TverskyLoss(nn.Module):

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Flatten
        probs_flat = probs.view(probs.size(0), probs.size(1), -1)
        target_flat = targets.view(targets.size(0), targets.size(1), -1)

        # True positives, false negatives, false positives
        tp = (probs_flat * target_flat).sum(dim=2)
        fn = (target_flat * (1 - probs_flat)).sum(dim=2)
        fp = ((1 - target_flat) * probs_flat).sum(dim=2)

        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)

        return 1.0 - tversky.mean()


class DiceFocalLoss(nn.Module):
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        smooth: float = 1.0,
        class_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice = DiceLoss(smooth=smooth, class_weights=class_weights)
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (self.dice_weight * self.dice(logits, targets) +
                self.focal_weight * self.focal(logits, targets))


#Loss function factory to get specified loss function.
def get_loss_function(config: Dict[str, Any]) -> nn.Module:

    lc = config["loss"]
    name = lc["name"].lower()

    # Class weights for MA, HE, EX, SE (inversely proportional to lesion size)
    class_weights = lc.get("class_weights", None)

    # Positive weights for BCE (ratio of neg/pos pixels per class)
    pos_weight_vals = lc.get("pos_weight", None)
    pos_weight = torch.tensor(pos_weight_vals, dtype=torch.float32) if pos_weight_vals else None

    if name == "dice":
        return DiceLoss(smooth=1.0, class_weights=class_weights)

    elif name == "bce":
        return BCELoss(pos_weight=pos_weight)

    elif name == "bce_dice":
        return BCEDiceLoss(
            bce_weight=lc.get("bce_weight", 0.4),
            dice_weight=lc.get("dice_weight", 0.6),
            class_weights=class_weights,
            pos_weight=pos_weight,
        )

    elif name == "focal":
        return FocalLoss(
            alpha=lc.get("focal_alpha", 0.25),
            gamma=lc.get("focal_gamma", 2.0),
        )

    elif name == "tversky":
        return TverskyLoss(
            alpha=lc.get("tversky_alpha", 0.7),
            beta=lc.get("tversky_beta", 0.3),
        )

    elif name == "dice_focal":
        return DiceFocalLoss(
            dice_weight=lc.get("dice_weight", 0.5),
            focal_weight=lc.get("focal_weight", 0.5),
            focal_alpha=lc.get("focal_alpha", 0.75),
            focal_gamma=lc.get("focal_gamma", 2.0),
            class_weights=class_weights,
        )

    else:
        raise ValueError(
            f"Unknown loss function: '{name}'. "
            f"Choose from: dice, bce, bce_dice, focal, tversky, dice_focal."
        )
