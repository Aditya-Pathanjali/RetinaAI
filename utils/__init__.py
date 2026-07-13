"""
RetinaAI — Utils Package
Utility functions for logging, helpers, and visualization.
"""
from .logger import get_logger
from .helpers import (
    set_seed,
    load_config,
    save_json,
    load_json,
    ensure_dir,
    format_time,
    count_parameters,
)
try:
    from .visualization import Visualizer
except ImportError:
    Visualizer = None  # matplotlib excluded in .exe build
from .lesion_counter import (
    count_lesions_single_mask,
    count_lesions_multiclass,
    count_lesions_batch,
)

__all__ = [
    "get_logger",
    "set_seed",
    "load_config",
    "save_json",
    "load_json",
    "ensure_dir",
    "format_time",
    "count_parameters",
    "Visualizer",
    "count_lesions_single_mask",
    "count_lesions_multiclass",
    "count_lesions_batch",
]

