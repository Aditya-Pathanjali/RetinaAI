import os
import sys
import argparse
import time
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from utils.helpers import (
    set_seed,
    load_config,
    get_device,
    count_parameters,
    ensure_dir,
    save_json,
)
from utils.logger import get_logger
from datasets.idrid_dataset import build_dataloaders
from models.attention_unet import build_model
from models.losses import get_loss_function
from training.trainer import Trainer
from evaluation.metrics import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="RetinaAI — Attention U-Net Training Exclusively on DDR Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_ddr_attention_unet.yaml",
        help="Path to DDR Attention U-Net config file.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="exp_ddr_attention_unet",
        help="Experiment name.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint for resuming.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    if args.experiment:
        config["experiment"]["name"] = args.experiment

    exp_name = config["experiment"]["name"]
    output_root = config["experiment"].get("save_dir", "experiments")
    exp_dir = ensure_dir(Path(output_root) / exp_name)
    log_dir = ensure_dir(exp_dir / "logs")

    # System logger
    logger = get_logger("TrainDDR", log_file=str(log_dir / "training.log"))
    logger.info("=" * 60)
    logger.info("  RetinaAI — Attention U-Net DDR Training Pipeline")
    logger.info("=" * 60)

    # Set seed
    seed = config["training"].get("random_seed", 42)
    set_seed(seed)
    device = get_device()
    logger.info(f"Using Compute Device: {device}")

    # Build DataLoaders
    logger.info("Loading DDR Dataset...")
    loaders = build_dataloaders(config)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders.get("test", None)

    logger.info(f"DDR Train samples: {len(train_loader.dataset)}")
    logger.info(f"DDR Val   samples: {len(val_loader.dataset)}")
    if test_loader:
        logger.info(f"DDR Test  samples: {len(test_loader.dataset)}")

    # Build Model with Pretrained Weights
    logger.info("Building Attention U-Net with ImageNet Pre-trained ResNet34 Encoder...")
    model = build_model(config)
    params_summary = count_parameters(model)
    logger.info(f"Trainable Parameters: {params_summary['trainable']:,} / {params_summary['total']:,}")

    # Loss Function
    logger.info(f"Loss Function: {config['loss']['name']} (Alpha={config['loss'].get('alpha', 0.3)}, Beta={config['loss'].get('beta', 0.7)})")
    criterion = get_loss_function(config)

    # Optimizer
    tc = config["training"]
    opt_name = tc.get("optimizer", "adamw").lower()
    lr = tc.get("learning_rate", 3.0e-4)
    weight_decay = tc.get("weight_decay", 1.0e-4)

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning Rate Scheduler
    epochs = tc.get("epochs", 40)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=tc.get("min_lr", 1.0e-6)
    )

    # Trainer Initialization
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
    )

    # Training Execution
    logger.info("Starting DDR Training Loop...")
    start_time = time.time()
    history = trainer.fit(train_loader, val_loader)
    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed / 60:.1f} minutes. Best Validation Dice: {trainer.best_metric:.4f}")

    # Evaluation on Official Test Set
    if test_loader:
        logger.info("=" * 60)
        logger.info("  Evaluating Best Model on Official DDR Test Set")
        logger.info("=" * 60)
        best_ckpt = exp_dir / "checkpoints" / "best.pth"
        if best_ckpt.exists():
            checkpoint = torch.load(best_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Loaded best checkpoint weights successfully.")

        eval_cfg = config.get("evaluation", {})
        test_results = evaluate_model(
            model=model,
            dataloader=test_loader,
            class_names=config["dataset"]["class_names"],
            device=device,
            threshold=eval_cfg.get("threshold", 0.5),
            min_area=eval_cfg.get("min_area", None),
            use_tta=eval_cfg.get("use_tta", False),
        )

        test_json_path = exp_dir / "test_results.json"
        save_json(test_results, str(test_json_path))
        logger.info(f"Test results saved to {test_json_path}")
        logger.info(f"Mean Test Dice:   {test_results['mean']['dice']:.4f}")
        logger.info(f"Mean Test Recall: {test_results['mean']['recall']:.4f}")
        for cls in config["dataset"]["class_names"]:
            if cls in test_results["per_class"]:
                c_res = test_results["per_class"][cls]
                logger.info(f"  [{cls}] Dice: {c_res['dice']:.4f} | Recall: {c_res['recall']:.4f} | Precision: {c_res['precision']:.4f}")

    logger.info("All operations completed successfully.")


if __name__ == "__main__":
    main()
