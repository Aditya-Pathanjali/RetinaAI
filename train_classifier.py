
import argparse
import sys
import os
import time
import csv
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import (
    set_seed,
    load_config,
    get_device,
    ensure_dir,
    save_json,
    format_time,
)
from utils.logger import get_logger
from datasets.aptos_dataset import build_aptos_dataloaders
from models.attention_unet import build_model
from models.hybrid_classifier import build_classifier
from utils.lesion_counter import count_lesions_batch


def parse_args():
    parser = argparse.ArgumentParser(
        description="RetinaAI — Stage 2: DR Severity Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="hybrid",
        choices=["hybrid", "classifier_only", "mask_input"],
        help="Ablation study",
    )
    parser.add_argument(
        "--seg_checkpoint",
        type=str,
        default=None,
        help="Path to trained Stage 1 Attention U-Net checkpoint. Required for 'hybrid' and 'mask_input'.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Override experiment name from config.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force execution on device ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--limit_batches",
        type=int,
        default=None,
        help="Limit number of batches per epoch (useful for debugging/dry runs).",
    )
    return parser.parse_args()


def get_lesion_count_features(
    seg_probs: torch.Tensor,
    class_names: List[str],
    thresholds: Dict[str, float],
    min_areas: Dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    B, C, H, W = seg_probs.shape
    probs_np = seg_probs.detach().cpu().numpy()
    
    counts_tensor = torch.zeros((B, len(class_names)), dtype=torch.float32, device=device)
    
    for b in range(B):
        for c, cls_name in enumerate(class_names):
            thresh = thresholds.get(cls_name, 0.5)
            min_a = min_areas.get(cls_name, 0)
            
            mask = (probs_np[b, c] >= thresh).astype(np.uint8) * 255
            
            if mask.sum() == 0:
                continue
            
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )
            
            count = 0
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_a:
                    count += 1
            
            counts_tensor[b, c] = count
            
    return counts_tensor


def plot_curves(history: Dict[str, List[float]], output_path: Path):
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="tab:blue", lw=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="tab:orange", lw=2)
    axes[0].set_title("Cross-Entropy Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()
    
    # QWK plot
    axes[1].plot(epochs, history["val_qwk"], label="Val QWK", color="tab:green", lw=2)
    axes[1].plot(epochs, history["val_acc"], label="Val Accuracy", color="tab:purple", linestyle="--", lw=1.5)
    axes[1].set_title("Classification Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def train_epoch(
    classifier: nn.Module,
    seg_model: Optional[nn.Module],
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    variant: str,
    class_names: List[str],
    thresholds: Dict[str, float],
    min_areas: Dict[str, int],
    scaler: GradScaler,
    use_amp: bool,
    device: torch.device,
    grad_clip: float,
    limit_batches: Optional[int] = None,
) -> Tuple[float, List[int], List[int]]:
    classifier.train()
    if seg_model is not None:
        seg_model.eval()
    running_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []
    device_type = "cuda" if device.type == "cuda" else "cpu"
    for batch_idx, (images, labels, meta) in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type, enabled=use_amp):
            if variant == "hybrid":
                if isinstance(meta, dict) and "lesion_counts" in meta:
                    counts = meta["lesion_counts"].to(device)
                else:
                    with torch.no_grad():
                        seg_logits = seg_model(images)
                        seg_probs = torch.sigmoid(seg_logits)
                        counts = get_lesion_count_features(
                            seg_probs, class_names, thresholds, min_areas, device
                        )
                logits = classifier(images, counts)
            elif variant == "mask_input":
                with torch.no_grad():
                    seg_logits = seg_model(images)
                    seg_probs = torch.sigmoid(seg_logits)
                inputs = torch.cat([images, seg_probs], dim=1)
                logits = classifier(inputs, None)
            else:
                logits = classifier(images, None)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy().tolist())
    epoch_loss = running_loss / max(total_samples, 1)
    return epoch_loss, all_preds, all_labels


@torch.no_grad()
def validate_epoch(
    classifier: nn.Module,
    seg_model: Optional[nn.Module],
    loader: DataLoader,
    criterion: nn.Module,
    variant: str,
    class_names: List[str],
    thresholds: Dict[str, float],
    min_areas: Dict[str, int],
    use_amp: bool,
    device: torch.device,
    limit_batches: Optional[int] = None,
) -> Tuple[float, List[int], List[int]]:
    classifier.eval()
    if seg_model is not None:
        seg_model.eval()
    running_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []
    device_type = "cuda" if device.type == "cuda" else "cpu"
    for batch_idx, (images, labels, meta) in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(device_type, enabled=use_amp):
            if variant == "hybrid":
                if isinstance(meta, dict) and "lesion_counts" in meta:
                    counts = meta["lesion_counts"].to(device)
                else:
                    seg_logits = seg_model(images)
                    seg_probs = torch.sigmoid(seg_logits)
                    counts = get_lesion_count_features(
                        seg_probs, class_names, thresholds, min_areas, device
                    )
                logits = classifier(images, counts)
            elif variant == "mask_input":
                seg_logits = seg_model(images)
                seg_probs = torch.sigmoid(seg_logits)
                inputs = torch.cat([images, seg_probs], dim=1)
                logits = classifier(inputs, None)
            else:
                logits = classifier(images, None)
            loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy().tolist())
    epoch_loss = running_loss / max(total_samples, 1)
    return epoch_loss, all_preds, all_labels


def compute_metrics(all_preds: np.ndarray, all_labels: np.ndarray) -> Tuple[float, float, float]:
    if len(all_preds) == 0:
        return 0.0, 0.0, 0.0
    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    # Cohen's kappa can be NaN or negative if all predictions are identical or data is skewed
    if np.isnan(qwk):
        qwk = 0.0
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return float(qwk), float(acc), float(f1)


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # 1. Setup config overrides
    tc = config.get("classification_training", {})
    if args.epochs:
        tc["epochs"] = args.epochs
        
    # Map classifier parameters to training loader configs
    config["training"]["batch_size"] = tc.get("batch_size", 8)
    config["training"]["num_workers"] = tc.get("num_workers", 4)
    
    # Determine experiment name
    if args.experiment:
        exp_name = args.experiment
    else:
        exp_name = f"{config['experiment']['name']}_cls_{args.variant}"
        
    # Seed configuration
    seed = tc.get("random_seed", 42)
    set_seed(seed)
    
    # Output pathing
    output_root = Path(config["experiment"]["output_root"])
    exp_dir = ensure_dir(output_root / exp_name)
    ckpt_dir = ensure_dir(exp_dir / "checkpoints")
    log_dir = ensure_dir(exp_dir / "logs")
    plots_dir = ensure_dir(exp_dir / "plots")
    
    logger = get_logger(
        "ClassifierTrainer",
        level=config.get("logging", {}).get("level", "INFO"),
        log_file=str(log_dir / "classification.log"),
    )
    
    logger.info("=" * 60)
    logger.info("  RetinaAI — Stage 2 Severity Classification Training")
    logger.info(f"  Variant:     {args.variant.upper()}")
    logger.info(f"  Experiment:  {exp_name}")
    logger.info("=" * 60)
    
    # Device setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    logger.info(f"Using device: {device}")
    
    # Save config
    save_json(config, str(exp_dir / "classification_config.json"))
    
    # 2. Load Stage 1 segmentation model
    seg_model = None
    class_names = config["dataset"]["class_names"]
    thresholds = {}
    min_areas = {}
    
    if args.variant in ["hybrid", "mask_input"]:
        # Resolve thresholds
        eval_cfg = config.get("evaluation", {})
        config_thresholds = eval_cfg.get("threshold", 0.5)
        config_min_areas = eval_cfg.get("min_area", 0)
        
        if isinstance(config_thresholds, dict):
            thresholds = {cls: float(config_thresholds.get(cls, 0.5)) for cls in class_names}
        else:
            thresholds = {cls: float(config_thresholds) for cls in class_names}
            
        if isinstance(config_min_areas, dict):
            min_areas = {cls: int(config_min_areas.get(cls, 0)) for cls in class_names}
        else:
            min_areas = {cls: int(config_min_areas) for cls in class_names}
            
        # Determine checkpoint path
        if args.seg_checkpoint:
            seg_ckpt_path = Path(args.seg_checkpoint)
        else:
            seg_ckpt_path = output_root / config["experiment"]["name"] / "checkpoints" / "best.pth"
            if not seg_ckpt_path.exists():
                # Fallback to exp_03_optimized
                seg_ckpt_path = output_root / "exp_03_optimized" / "checkpoints" / "best.pth"
                
        if not seg_ckpt_path.exists():
            logger.error(f"Segmentation checkpoint not found at: {seg_ckpt_path}")
            logger.error("Please run Stage 1 or specify path via --seg_checkpoint.")
            sys.exit(1)
            
        logger.info(f"Loading Stage 1 Segmentation model from: {seg_ckpt_path}")
        seg_model = build_model(config)
        ckpt = torch.load(seg_ckpt_path, map_location=device)
        seg_model.load_state_dict(ckpt["model_state_dict"])
        seg_model = seg_model.to(device)
        seg_model.eval()
        for p in seg_model.parameters():
            p.requires_grad = False
            
    # 3. Instantiate Stage 2 classifier
    # Adjust config dynamically for the model construction
    config["classification_model"]["lesion_feature_dim"] = len(class_names) if args.variant == "hybrid" else 0
    in_channels = 3 + len(class_names) if args.variant == "mask_input" else 3
    
    logger.info(f"Building HybridDRClassifier ({args.variant}) | in_channels={in_channels} | feature_dim={config['classification_model']['lesion_feature_dim']}")
    classifier = build_classifier(config, in_channels=in_channels)
    classifier = classifier.to(device)
    
    # Print model params
    total_params = sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    logger.info(f"  Total classifier parameters:     {total_params:,}")
    logger.info(f"  Trainable classifier parameters: {trainable_params:,}")
    
    # 4. Build Dataloaders
    logger.info("Building APTOS 2019 dataloaders...")
    loaders = build_aptos_dataloaders(config)
    logger.info(f"  Train size: {len(loaders['train'].dataset)}")
    logger.info(f"  Val size:   {len(loaders['val'].dataset)}")
    logger.info(f"  Test size:  {len(loaders['test'].dataset)}")
    
    # Rebalance training dataset with WeightedRandomSampler for high recall
    train_dataset = loaders["train"].dataset
    train_labels = train_dataset.df["diagnosis"].values
    class_counts = np.bincount(train_labels)
    logger.info(f"  Train class counts: {class_counts.tolist()}")
    
    # Compute inverse frequency sample weights
    class_weights_inv = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights_inv[label] for label in train_labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Re-instantiate the training dataloader with the WeightedRandomSampler
    loaders["train"] = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        sampler=sampler,
        num_workers=config["training"]["num_workers"],
        pin_memory=config["training"]["pin_memory"],
        drop_last=True
    )
    logger.info("  Re-instantiated training dataloader with WeightedRandomSampler.")
    
    # 5. Optimizer, Loss, Scheduler setup
    opt_name = tc.get("optimizer", "adamw").lower()
    lr = tc.get("learning_rate", 1e-4)
    wd = tc.get("weight_decay", 1e-4)
    
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=wd)
        
    # Calculate class weights for Weighted Cross-Entropy Loss
    num_classes = len(class_counts)
    total_samples = len(train_labels)
    class_weights = total_samples / (num_classes * (class_counts + 1e-6))
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    logger.info(f"  Weighted Cross-Entropy Loss class weights: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    scheduler_name = tc.get("scheduler", "plateau").lower()
    if scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=tc.get("scheduler_patience", 5),
            factor=tc.get("scheduler_factor", 0.5),
            min_lr=tc.get("min_lr", 1e-7),
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=tc.get("epochs", 40),
            eta_min=tc.get("min_lr", 1e-7),
        )
        
    # Scaler
    use_amp = tc.get("use_amp", True) and device.type == "cuda"
    device_type = "cuda" if device.type == "cuda" else "cpu"
    scaler = GradScaler(device_type, enabled=use_amp)
    
    # CSV Log initialization
    csv_log_path = log_dir / "classification_metrics.csv"
    csv_headers = ["epoch", "train_loss", "train_acc", "train_qwk", "train_f1", 
                   "val_loss", "val_acc", "val_qwk", "val_f1", "lr"]
    with open(csv_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        
    # Metrics tracking
    history = {
        "epoch": [], "train_loss": [], "train_qwk": [], "train_acc": [],
        "val_loss": [], "val_qwk": [], "val_acc": [], "val_f1": [], "lr": []
    }
    
    best_qwk = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    patience = tc.get("early_stopping_patience", 10)
    
    total_start = time.time()
    
    # 6. Training Loop
    for epoch in range(1, tc["epochs"] + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_preds, train_labels = train_epoch(
            classifier=classifier,
            seg_model=seg_model,
            loader=loaders["train"],
            optimizer=optimizer,
            criterion=criterion,
            variant=args.variant,
            class_names=class_names,
            thresholds=thresholds,
            min_areas=min_areas,
            scaler=scaler,
            use_amp=use_amp,
            device=device,
            grad_clip=tc.get("grad_clip", 1.0),
            limit_batches=args.limit_batches,
        )
        
        # Validate
        val_loss, val_preds, val_labels = validate_epoch(
            classifier=classifier,
            seg_model=seg_model,
            loader=loaders["val"],
            criterion=criterion,
            variant=args.variant,
            class_names=class_names,
            thresholds=thresholds,
            min_areas=min_areas,
            use_amp=use_amp,
            device=device,
            limit_batches=args.limit_batches,
        )
        
        # Calculate Epoch Metrics
        train_qwk, train_acc, train_f1 = compute_metrics(np.array(train_preds), np.array(train_labels))
        val_qwk, val_acc, val_f1 = compute_metrics(np.array(val_preds), np.array(val_labels))
        
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Update Scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_qwk)
            else:
                scheduler.step()
                
        # Record history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_qwk"].append(train_qwk)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_qwk"].append(val_qwk)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)
        
        # Write to CSV log
        with open(csv_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, f"{train_loss:.6f}", f"{train_acc:.4f}", f"{train_qwk:.4f}", f"{train_f1:.4f}",
                f"{val_loss:.6f}", f"{val_acc:.4f}", f"{val_qwk:.4f}", f"{val_f1:.4f}", f"{current_lr:.8f}"
            ])
            
        # Logging output
        elapsed = time.time() - epoch_start
        is_best = val_qwk > best_qwk
        
        logger.info(
            f"Epoch {epoch:>2d}/{tc['epochs']} | "
            f"Loss: {train_loss:.4f} / {val_loss:.4f} | "
            f"Acc: {train_acc:.4f} / {val_acc:.4f} | "
            f"QWK: {train_qwk:.4f} / {val_qwk:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {format_time(elapsed)} | "
            f"{'* BEST' if is_best else ''}"
        )
        
        # Checkpointing
        if is_best:
            best_qwk = val_qwk
            best_epoch = epoch
            epochs_no_improve = 0
            
            # Save best checkpoint
            state = {
                "epoch": epoch,
                "model_state_dict": classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "best_qwk": best_qwk,
                "config": config,
            }
            torch.save(state, ckpt_dir / "best.pth")
        else:
            epochs_no_improve += 1
            
        # Always save latest checkpoint
        latest_state = {
            "epoch": epoch,
            "model_state_dict": classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "current_qwk": val_qwk,
            "config": config,
        }
        torch.save(latest_state, ckpt_dir / "latest.pth")
        
        # Early Stopping
        if epochs_no_improve >= patience:
            logger.warning(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best QWK: {best_qwk:.4f} at epoch {best_epoch}."
            )
            break
            
    total_time = time.time() - total_start
    logger.info("=" * 60)
    logger.info(f"  Training completed in {format_time(total_time)}")
    logger.info(f"  Best Val QWK: {best_qwk:.4f} (epoch {best_epoch})")
    logger.info("=" * 60)
    
    # Save training history
    save_json(history, str(exp_dir / "classification_history.json"))
    
    # 7. Generate curves
    plot_curves(history, plots_dir / "classification_curves.png")
    logger.info(f"Training curves saved to: {plots_dir / 'classification_curves.png'}")
    
    # 8. Evaluate on test set using the best model
    logger.info("\n" + "=" * 60)
    logger.info("  Evaluating best model on Test Set")
    logger.info("=" * 60)
    
    best_ckpt_path = ckpt_dir / "best.pth"
    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        classifier.load_state_dict(best_ckpt["model_state_dict"])
        logger.info(f"Loaded best classifier model from epoch {best_ckpt['epoch']}")
        
    test_loss, test_preds, test_labels = validate_epoch(
        classifier=classifier,
        seg_model=seg_model,
        loader=loaders["test"],
        criterion=criterion,
        variant=args.variant,
        class_names=class_names,
        thresholds=thresholds,
        min_areas=min_areas,
        use_amp=use_amp,
        device=device,
        limit_batches=args.limit_batches,
    )
    
    test_qwk, test_acc, test_f1 = compute_metrics(np.array(test_preds), np.array(test_labels))
    from sklearn.metrics import recall_score
    test_recall = recall_score(test_labels, test_preds, average="macro", zero_division=0)
    class_recalls = recall_score(test_labels, test_preds, average=None, zero_division=0)
    
    logger.info(f"Test Loss:     {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Test F1-Score: {test_f1:.4f}")
    logger.info(f"Test QWK:      {test_qwk:.4f}")
    logger.info(f"Test Recall:   {test_recall:.4f}")
    logger.info(f"Class Recalls: {[round(float(r), 4) for r in class_recalls]}")
    
    # Save test results
    test_results = {
        "loss": test_loss,
        "accuracy": test_acc,
        "f1_score": test_f1,
        "qwk": test_qwk,
        "recall_macro": float(test_recall),
        "class_recalls": [float(r) for r in class_recalls],
    }
    save_json(test_results, str(exp_dir / "test_results.json"))
    logger.info(f"Test results saved to: {exp_dir / 'test_results.json'}")


if __name__ == "__main__":
    main()
