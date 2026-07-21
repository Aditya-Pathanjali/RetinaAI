
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List


#  Diceloss
class DiceLoss(nn.Module):

    def __init__(self, smooth: float = 1e-5, class_weights: Optional[List[float]] = None):
        super().__init__()
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid_classes: Optional[torch.Tensor] = None) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions: (B, C, H*W)
        probs_flat = probs.view(probs.size(0), probs.size(1), -1)
        target_flat = targets.view(targets.size(0), targets.size(1), -1)

        # Per-class Dice
        intersection = (probs_flat * target_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + target_flat.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice # (B, C)

        if self.class_weights is not None:
            weights = torch.tensor(self.class_weights, device=logits.device, dtype=logits.dtype)
            if valid_classes is not None:
                effective_weights = weights.unsqueeze(0) * valid_classes # (B, C)
                loss_sum = (dice_loss * effective_weights).sum(dim=1)
                weight_sum = effective_weights.sum(dim=1).clamp(min=1e-6)
                return (loss_sum / weight_sum).mean()
            else:
                weighted_dice = (dice * weights.unsqueeze(0)).sum(dim=1) / weights.sum()
                return 1.0 - weighted_dice.mean()
        else:
            if valid_classes is not None:
                loss_sum = (dice_loss * valid_classes).sum(dim=1)
                valid_count = valid_classes.sum(dim=1).clamp(min=1.0)
                return (loss_sum / valid_count).mean()
            else:
                return dice_loss.mean()


#BCELoss (wrapper)
class BCELoss(nn.Module):

    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid_classes: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.pos_weight is not None:
            pw = self.pos_weight.to(logits.device)
            loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw.view(1, -1, 1, 1), reduction="none")
        else:
            loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            
        if valid_classes is not None:
            mask = valid_classes.view(valid_classes.size(0), valid_classes.size(1), 1, 1)
            masked_loss = loss * mask
            h, w = loss.shape[2:]
            valid_count = valid_classes.sum(dim=1) * (h * w)
            valid_count = valid_count.clamp(min=1.0)
            return (masked_loss.sum(dim=(1, 2, 3)) / valid_count).mean()
        else:
            return loss.mean()


#BCE_DICE Hybrid loss
class BCEDiceLoss(nn.Module):

    def __init__(
        self,
        bce_weight: float = 0.4,
        dice_weight: float = 0.6,
        smooth: float = 1e-5,
        class_weights: Optional[List[float]] = None,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = BCELoss(pos_weight=pos_weight)
        self.dice = DiceLoss(smooth=smooth, class_weights=class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid_classes: Optional[torch.Tensor] = None) -> torch.Tensor:
        return (self.bce_weight * self.bce(logits, targets, valid_classes) +
                self.dice_weight * self.dice(logits, targets, valid_classes))


#Focal Loss (Lin et al., 2017) adapted for segmentation.
class FocalLoss(nn.Module):

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid_classes: Optional[torch.Tensor] = None) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        loss = focal_weight * bce_loss
        
        if valid_classes is not None:
            mask = valid_classes.view(valid_classes.size(0), valid_classes.size(1), 1, 1)
            masked_loss = loss * mask
            h, w = loss.shape[2:]
            valid_count = valid_classes.sum(dim=1) * (h * w)
            valid_count = valid_count.clamp(min=1.0)
            return (masked_loss.sum(dim=(1, 2, 3)) / valid_count).mean()
        else:
            return loss.mean()


#Tversky Loss (Salehi et al., 2017) — generalisation of Dice loss.

class TverskyLoss(nn.Module):

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1.0, class_weights: Optional[List[float]] = None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid_classes: Optional[torch.Tensor] = None) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Flatten
        probs_flat = probs.view(probs.size(0), probs.size(1), -1)
        target_flat = targets.view(targets.size(0), targets.size(1), -1)

        # True positives, false negatives, false positives
        tp = (probs_flat * target_flat).sum(dim=2)
        fn = (target_flat * (1 - probs_flat)).sum(dim=2)
        fp = ((1 - target_flat) * probs_flat).sum(dim=2)

        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        tversky_loss = 1.0 - tversky # (B, C)

        if self.class_weights is not None:
            weights = torch.tensor(self.class_weights, device=logits.device, dtype=logits.dtype)
            if valid_classes is not None:
                effective_weights = weights.unsqueeze(0) * valid_classes  # (B, C)
                loss_sum = (tversky_loss * effective_weights).sum(dim=1)
                weight_sum = effective_weights.sum(dim=1).clamp(min=1e-6)
                return (loss_sum / weight_sum).mean()
            else:
                weighted_loss = (tversky_loss * weights.unsqueeze(0)).sum(dim=1) / weights.sum()
                return weighted_loss.mean()
        else:
            if valid_classes is not None:
                weighted_tversky_loss = tversky_loss * valid_classes
                loss_sum = weighted_tversky_loss.sum(dim=1)
                valid_count = valid_classes.sum(dim=1).clamp(min=1.0)
                return (loss_sum / valid_count).mean()
            else:
                return tversky_loss.mean()


class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.33,
        smooth: float = 1.0,
        class_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid_classes: Optional[torch.Tensor] = None) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), probs.size(1), -1)
        target_flat = targets.view(targets.size(0), targets.size(1), -1)

        tp = (probs_flat * target_flat).sum(dim=2)
        fn = (target_flat * (1 - probs_flat)).sum(dim=2)
        fp = ((1 - target_flat) * probs_flat).sum(dim=2)

        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        loss = (1.0 - tversky).pow(self.gamma)

        if self.class_weights is not None:
            weights = torch.tensor(self.class_weights, device=logits.device, dtype=logits.dtype)
            loss = loss * weights.unsqueeze(0)

        if valid_classes is not None:
            loss = loss * valid_classes
            valid_count = valid_classes.sum(dim=1).clamp(min=1.0)
            return (loss.sum(dim=1) / valid_count).mean()

        return loss.mean()


class DiceFocalLoss(nn.Module):
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        smooth: float = 1e-5,
        class_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice = DiceLoss(smooth=smooth, class_weights=class_weights)
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid_classes: Optional[torch.Tensor] = None) -> torch.Tensor:
        return (self.dice_weight * self.dice(logits, targets, valid_classes) +
                self.focal_weight * self.focal(logits, targets, valid_classes))


class HybridLesionLoss(nn.Module):
    def __init__(
        self,
        ft_indices: Optional[List[int]] = None,
        bce_dice_indices: Optional[List[int]] = None,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        focal_tversky_gamma: float = 1.33,
        bce_weight: float = 0.4,
        dice_weight: float = 0.6,
        class_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.ft_indices = ft_indices or [0, 1]
        self.bce_dice_indices = bce_dice_indices or [2, 3]
        self.class_weights = class_weights
        self.ft = FocalTverskyLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
            gamma=focal_tversky_gamma,
        )
        self.bce_dice = BCEDiceLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid_classes: Optional[torch.Tensor] = None) -> torch.Tensor:
        losses = []
        weights = []

        for idx in self.ft_indices:
            if idx >= logits.size(1):
                continue
            valid = valid_classes[:, idx:idx + 1] if valid_classes is not None else None
            losses.append(self.ft(logits[:, idx:idx + 1], targets[:, idx:idx + 1], valid))
            weights.append(self._class_weight(idx, logits))

        for idx in self.bce_dice_indices:
            if idx >= logits.size(1):
                continue
            valid = valid_classes[:, idx:idx + 1] if valid_classes is not None else None
            losses.append(self.bce_dice(logits[:, idx:idx + 1], targets[:, idx:idx + 1], valid))
            weights.append(self._class_weight(idx, logits))

        if not losses:
            return logits.sum() * 0.0

        weight_tensor = torch.stack(weights)
        loss_tensor = torch.stack(losses)
        return (loss_tensor * weight_tensor).sum() / weight_tensor.sum().clamp(min=1e-6)

    def _class_weight(self, idx: int, logits: torch.Tensor) -> torch.Tensor:
        if self.class_weights is None or idx >= len(self.class_weights):
            return torch.tensor(1.0, device=logits.device, dtype=logits.dtype)
        return torch.tensor(float(self.class_weights[idx]), device=logits.device, dtype=logits.dtype)


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
        return DiceLoss(smooth=1e-5, class_weights=class_weights)

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
            class_weights=class_weights,
        )

    elif name == "focal_tversky":
        return FocalTverskyLoss(
            alpha=lc.get("tversky_alpha", 0.3),
            beta=lc.get("tversky_beta", 0.7),
            gamma=lc.get("focal_tversky_gamma", 1.33),
            class_weights=class_weights,
        )

    elif name == "dice_focal":
        return DiceFocalLoss(
            dice_weight=lc.get("dice_weight", 0.5),
            focal_weight=lc.get("focal_weight", 0.5),
            focal_alpha=lc.get("focal_alpha", 0.75),
            focal_gamma=lc.get("focal_gamma", 2.0),
            class_weights=class_weights,
        )

    elif name == "hybrid_lesion":
        return HybridLesionLoss(
            ft_indices=lc.get("focal_tversky_indices", [0, 1]),
            bce_dice_indices=lc.get("bce_dice_indices", [2, 3]),
            tversky_alpha=lc.get("tversky_alpha", 0.3),
            tversky_beta=lc.get("tversky_beta", 0.7),
            focal_tversky_gamma=lc.get("focal_tversky_gamma", 1.33),
            bce_weight=lc.get("bce_weight", 0.4),
            dice_weight=lc.get("dice_weight", 0.6),
            class_weights=class_weights,
        )

    else:
        raise ValueError(
            f"Unknown loss function: '{name}'. "
            f"Choose from: dice, bce, bce_dice, focal, tversky, focal_tversky, dice_focal, hybrid_lesion."
        )
