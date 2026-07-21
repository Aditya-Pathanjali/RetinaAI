import os
import csv
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from evaluation.metrics import SegmentationMetrics, sliding_window_inference
from utils.helpers import (
    format_time,
    save_checkpoint,
    ensure_dir,
    save_json,
)
from utils.logger import get_logger


class Trainer:
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        config: Dict[str, Any],
        device: torch.device,
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device

        # Training config
        tc = config["training"]
        self.epochs = tc["epochs"]
        self.grad_clip = tc.get("grad_clip", 1.0)
        self.use_amp = tc.get("use_amp", True) and device.type == "cuda"
        self.save_every_n = tc.get("save_every_n", 10)
        self.save_best_only = tc.get("save_best_only", False)

        # Early stopping
        self.patience = tc.get("early_stopping_patience", 20)
        self.monitor_metric = tc.get("monitor_metric", "val_dice")

        # Class names for metrics
        self.class_names = config["dataset"]["class_names"]
        self.threshold = config["evaluation"].get("threshold", 0.5)
        self.min_area = config["evaluation"].get("min_area", None)

        # Experiment directories
        exp = config["experiment"]
        self.exp_dir = ensure_dir(
            Path(exp["output_root"]) / exp["name"]
        )
        self.ckpt_dir = ensure_dir(self.exp_dir / "checkpoints")
        self.log_dir = ensure_dir(self.exp_dir / "logs")
        self.pred_dir = ensure_dir(self.exp_dir / "predictions")

        # AMP scaler
        self.device_type = "cuda" if device.type == "cuda" else "cpu"
        self.scaler = GradScaler(self.device_type, enabled=self.use_amp)

        # Logger
        self.logger = get_logger(
            "Trainer",
            level=config.get("logging", {}).get("level", "INFO"),
            log_file=str(self.log_dir / "training.log"),
        )

        # Training history
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_dice": [],
            "val_iou": [],
            "lr": [],
        }
        # Per-class Dice history
        for cls in self.class_names:
            self.history[f"val_dice_{cls}"] = []

        # Best metric tracking
        self.best_metric = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0

        # CSV file
        self.csv_path = self.log_dir / "metrics.csv"
        self._init_csv()

        # Save config for reproducibility
        save_json(config, str(self.exp_dir / "config.json"))

    # Main training loop

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        start_epoch: int = 1,
    ) -> Dict[str, List[float]]:
        
        self.logger.info("=" * 60)
        self.logger.info("  RetinaAI — Training Started")
        self.logger.info("=" * 60)
        self.logger.info(f"  Device:       {self.device}")
        self.logger.info(f"  Epochs:       {self.epochs}")
        self.logger.info(f"  Batch size:   {self.config['training']['batch_size']}")
        self.logger.info(f"  Loss:         {self.config['loss']['name']}")
        self.logger.info(f"  Optimizer:    {self.config['training']['optimizer']}")
        self.logger.info(f"  LR:           {self.config['training']['learning_rate']}")
        self.logger.info(f"  AMP:          {self.use_amp}")
        self.logger.info(f"  Experiment:   {self.config['experiment']['name']}")
        self.logger.info("=" * 60)

        total_start = time.time()

        for epoch in range(start_epoch, self.epochs + 1):
            epoch_start = time.time()

            #Train
            train_loss = self._train_epoch(train_loader, epoch)

            #Validate
            val_loss, val_metrics = self._validate_epoch(val_loader, epoch)

            #Extract key metrics
            mean_dice = val_metrics["mean"]["dice"]
            mean_iou = val_metrics["mean"]["iou"]
            current_lr = self.optimizer.param_groups[0]["lr"]

            #Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(mean_dice)
                else:
                    self.scheduler.step()

            #Record history
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_dice"].append(mean_dice)
            self.history["val_iou"].append(mean_iou)
            self.history["lr"].append(current_lr)

            for cls in self.class_names:
                self.history[f"val_dice_{cls}"].append(
                    val_metrics["per_class"][cls]["dice"]
                )

            #csv logging
            self._log_csv(epoch, train_loss, val_loss, mean_dice, mean_iou,
                          current_lr, val_metrics)

            #Checkpointing
            is_best = mean_dice > self.best_metric
            if is_best:
                self.best_metric = mean_dice
                self.best_epoch = epoch
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            #Save checkpoints
            self._save_checkpoints(epoch, mean_dice, is_best)

            #Logging
            elapsed = time.time() - epoch_start
            self.logger.info(
                f"Epoch {epoch:>3d}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Dice: {mean_dice:.4f} | "
                f"Val IoU: {mean_iou:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {format_time(elapsed)} | "
                f"{'* BEST' if is_best else ''}"
            )

            # Per-class Dice
            per_class_str = " | ".join(
                f"{cls}: {val_metrics['per_class'][cls]['dice']:.4f}"
                for cls in self.class_names
            )
            self.logger.info(f"  Per-class Dice: {per_class_str}")

            #Early stopping
            if self.epochs_no_improve >= self.patience:
                self.logger.warning(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best Dice: {self.best_metric:.4f} at epoch {self.best_epoch}."
                )
                break

        total_time = time.time() - total_start
        self.logger.info("=" * 60)
        self.logger.info(f"  Training completed in {format_time(total_time)}")
        self.logger.info(f"  Best Val Dice: {self.best_metric:.4f} (epoch {self.best_epoch})")
        self.logger.info("=" * 60)

        # Save final history
        save_json(self.history, str(self.exp_dir / "training_history.json"))

        return self.history

    # Single Epoch: Train

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        for batch_idx, (images, masks, meta) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            valid_classes = None
            if isinstance(meta, dict) and "valid_classes" in meta:
                valid_classes = meta["valid_classes"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass with automatic mixed precision
            with autocast(self.device_type, enabled=self.use_amp):
                logits = self.model(images)
                if valid_classes is not None:
                    loss = self.criterion(logits, masks, valid_classes)
                else:
                    loss = self.criterion(logits, masks)

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Gradient clipping (unscale first for correct norm computation)
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            num_batches += 1

        return running_loss / max(num_batches, 1)

    
    #Single Epoch: Validate

    @torch.no_grad()
    def _validate_epoch(
        self, loader: DataLoader, epoch: int
    ) -> Tuple[float, Dict]:
        
        self.model.eval()
        running_loss = 0.0
        num_batches = 0

        metrics = SegmentationMetrics(
            class_names=self.class_names,
            threshold=self.threshold,
            min_area=self.min_area,
        )

        for images, masks, meta in loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            valid_classes = None
            if isinstance(meta, dict) and "valid_classes" in meta:
                valid_classes = meta["valid_classes"].to(self.device, non_blocking=True)

            with autocast(self.device_type, enabled=self.use_amp):
                # Use sliding window inference for validation to handle full res images
                logits = sliding_window_inference(
                    self.model, images, 
                    patch_size=self.config["preprocessing"].get("image_size", 512),
                    stride=self.config["preprocessing"].get("image_size", 512) // 2,
                    num_classes=len(self.class_names)
                )
                
                if valid_classes is not None:
                    loss = self.criterion(logits, masks, valid_classes)
                else:
                    loss = self.criterion(logits, masks)

            running_loss += loss.item()
            num_batches += 1

            metrics.update(logits, masks, valid_classes)

        avg_loss = running_loss / max(num_batches, 1)
        metric_results = metrics.compute()

        return avg_loss, metric_results

    # Checkpointing

    def _save_checkpoints(
        self, epoch: int, metric: float, is_best: bool
    ) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "best_metric": self.best_metric,
            "config": self.config,
        }

        # Always save latest
        if not self.save_best_only:
            save_checkpoint(
                state,
                filepath=str(self.ckpt_dir / "latest.pth"),
            )

        # Save best
        if is_best:
            save_checkpoint(
                state,
                filepath=str(self.ckpt_dir / "best.pth"),
            )

        # Periodic saves
        if self.save_every_n > 0 and epoch % self.save_every_n == 0:
            save_checkpoint(
                state,
                filepath=str(self.ckpt_dir / f"epoch_{epoch:03d}.pth"),
            )

    #csv logging
    
    def _init_csv(self) -> None:
        #csv headers
        headers = [
            "epoch", "train_loss", "val_loss", "val_dice_mean", "val_iou_mean", "lr",
        ]
        for cls in self.class_names:
            headers.extend([f"dice_{cls}", f"iou_{cls}", f"prec_{cls}", f"recall_{cls}"])

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def _log_csv(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        mean_dice: float,
        mean_iou: float,
        lr: float,
        val_metrics: Dict,
    ) -> None:
        """Append one row to the metrics CSV."""
        row = [epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
               f"{mean_dice:.6f}", f"{mean_iou:.6f}", f"{lr:.8f}"]

        for cls in self.class_names:
            m = val_metrics["per_class"][cls]
            row.extend([
                f"{m['dice']:.6f}",
                f"{m['iou']:.6f}",
                f"{m['precision']:.6f}",
                f"{m['recall']:.6f}",
            ])

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
