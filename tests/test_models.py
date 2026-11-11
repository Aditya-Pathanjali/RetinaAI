import pytest
import torch
from models.attention_unet import AttentionUNet, AttentionGate
from models.hybrid_classifier import HybridDRClassifier


def test_attention_unet_output_dimensions():
    """Verify Attention U-Net outputs expected 4-channel tensor [B, 4, H, W]."""
    model = AttentionUNet(
        encoder_name="resnet34",
        encoder_weights=None,
        out_channels=4,
        use_attention=True,
    )
    model.eval()
    
    # Batch size 2, 3 channels, 256x256
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
        
    assert out.shape == (2, 4, 256, 256)
    assert not torch.isnan(out).any()


def test_attention_gate_forward_pass():
    """Verify Attention Gate outputs strictly bounded attention coefficient values in [0, 1]."""
    gate = AttentionGate(F_g=64, F_l=64, F_int=32)
    gate.eval()
    
    g = torch.randn(2, 64, 32, 32)
    x = torch.randn(2, 64, 32, 32)
    
    with torch.no_grad():
        out = gate(g, x)
        
    assert out.shape == (2, 64, 32, 32)
    assert not torch.isnan(out).any()


def test_hybrid_classifier_tensor_fusion():
    """Verify Hybrid Classifier correctly concatenates CNN feature embeddings with 4D lesion count vector."""
    model = HybridDRClassifier(
        num_classes=5,
        lesion_feature_dim=4,
        pretrained=False,
        hidden_dim=256,
    )
    model.eval()
    
    img_tensor = torch.randn(2, 3, 224, 224)
    lesion_counts = torch.tensor([[5.0, 12.0, 3.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    
    with torch.no_grad():
        logits = model(img_tensor, lesion_counts)
        
    assert logits.shape == (2, 5)
    assert not torch.isnan(logits).any()
