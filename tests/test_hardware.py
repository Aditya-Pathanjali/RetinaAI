import pytest
import torch
import numpy as np


def test_cuda_cpu_fallback_consistency():
    """Verify inference models can execute on CPU and maintain numerical stability."""
    # Test tensor operation on CPU
    x_cpu = torch.randn(1, 3, 64, 64, device="cpu")
    conv = torch.nn.Conv2d(3, 16, 3, padding=1).to("cpu")
    
    with torch.no_grad():
        out_cpu = conv(x_cpu)
        
    assert out_cpu.device.type == "cpu"
    assert not torch.isnan(out_cpu).any()
    
    # If CUDA is available, verify CPU vs CUDA output parity
    if torch.cuda.is_available():
        conv_cuda = conv.to("cuda")
        x_cuda = x_cpu.to("cuda")
        with torch.no_grad():
            out_cuda = conv_cuda(x_cuda)
        
        diff = (out_cpu - out_cuda.cpu()).abs().max().item()
        assert diff < 1e-4, f"CPU and CUDA discrepancy too large: {diff}"
