"""
RetinaAI — Evaluation Package
Dissertation-quality segmentation metrics.
"""
from .metrics import SegmentationMetrics, evaluate_model

__all__ = ["SegmentationMetrics", "evaluate_model"]
