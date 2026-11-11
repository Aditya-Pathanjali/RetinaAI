import pytest
import torch
from models.losses import FocalTverskyLoss, BCEDiceLoss


def test_focal_tversky_loss_bounds_and_gradients():
    """Verify Focal Tversky Loss produces scalar >= 0 and non-NaN gradients under extreme imbalance."""
    loss_fn = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.33)
    
    # Logits requiring grad
    logits = torch.randn(2, 4, 64, 64, requires_grad=True)
    
    # Highly sparse targets (99.9% background, 0.1% foreground)
    targets = torch.zeros(2, 4, 64, 64)
    targets[:, :, 10:12, 10:12] = 1.0
    
    loss = loss_fn(logits, targets)
    assert loss.item() >= 0.0
    assert not torch.isnan(loss)
    
    # Backpropagation check
    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
    assert not torch.isinf(logits.grad).any()


def test_dice_focal_combined_loss_zero_division():
    """Verify loss function does not divide by zero when targets and predictions are both completely empty."""
    loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5, smooth=1e-5)
    
    # Large negative logits -> sigmoid approaches 0.0
    logits = torch.full((2, 4, 32, 32), -100.0, requires_grad=True)
    targets = torch.zeros(2, 4, 32, 32)
    
    loss = loss_fn(logits, targets)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() >= 0.0
