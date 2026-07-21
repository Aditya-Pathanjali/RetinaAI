
import os
import json
import random
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def save_json(data: Any, path: str, indent: int = 4) -> None:
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: str) -> Any:
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def ensure_dir(path: str) -> Path:
    
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def format_time(seconds: float) -> str:
    
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def get_device() -> torch.device:
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[Device] Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        return device
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[Device] Using Apple MPS")
        return torch.device("mps")
    else:
        print("[Device] No GPU detected — using CPU")
        return torch.device("cpu")


def save_checkpoint(
    state: Dict[str, Any],
    filepath: str,
    is_best: bool = False,
    best_path: Optional[str] = None,
) -> None:
   
    import shutil
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filepath)
    if is_best and best_path:
        shutil.copyfile(filepath, best_path)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    
    if device is None:
        device = get_device()

    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Move optimizer state tensors to the correct device (fixes CPU/GPU mismatch on resume)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
