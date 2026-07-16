
import argparse
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from utils.helpers import (
    set_seed,
    load_config,
    get_device,
    count_parameters,
    ensure_dir,
    save_json,
)
from utils.logger import get_logger
from utils.visualization import Visualizer
from datasets.idrid_dataset import IDRiDDataset, split_dataset, build_dataloaders
from preprocessing.enhancer import RetinalEnhancer
from preprocessing.transforms import get_train_transforms, get_val_transforms
from models.attention_unet import AttentionUNet, build_model
from models.losses import get_loss_function
from training.trainer import Trainer
from evaluation.metrics import SegmentationMetrics, evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="RetinaAI — Diabetic Retinopathy Lesion Segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "explore"],
        help="Pipeline mode: train, eval, or explore.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Override experiment name from config.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint for resuming training.",
    )
    return parser.parse_args()


def run_exploration(config: dict, logger) -> None:
    
    import cv2
    logger.info("=" * 60)
    logger.info("  Dataset Exploration Mode")
    logger.info("=" * 60)

    ds = config["dataset"]
    root = Path(ds["root"])
    image_dir = root / ds["train_images"]
    mask_root = root / ds["train_masks"]
    ext = ds.get("image_ext", ".jpg")

    output_dir = ensure_dir("outputs/exploration")
    viz = Visualizer(output_dir=str(output_dir), class_names=ds["class_names"])

    # Count images
    train_images = sorted(image_dir.glob(f"*{ext}"))
    test_dir = root / ds["test_images"]
    test_images = sorted(test_dir.glob(f"*{ext}"))
    logger.info(f"Training images:  {len(train_images)}")
    logger.info(f"Testing images:   {len(test_images)}")

    # Inspect resolutions
    resolutions = []
    file_sizes = []
    for img_path in train_images:
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            resolutions.append((w, h))
            file_sizes.append(img_path.stat().st_size / 1024)  # KB

    unique_res = set(resolutions)
    logger.info(f"Unique resolutions: {unique_res}")
    logger.info(f"File sizes: min={min(file_sizes):.0f}KB, max={max(file_sizes):.0f}KB, "
                f"mean={np.mean(file_sizes):.0f}KB")

    # Mask distribution analysis
    logger.info("\nAnalysing lesion mask distributions...")
    all_ids = sorted([f.stem for f in train_images])
    dataset = IDRiDDataset(
        image_ids=all_ids,
        config=config,
        transform=None,
        enhancer=None,
        is_training=True,
    )
    stats = dataset.compute_class_statistics()

    logger.info("\n" + "=" * 60)
    logger.info("  CLASS IMBALANCE ANALYSIS")
    logger.info("=" * 60)
    for cls_name, s in stats.items():
        logger.info(f"\n  {cls_name}:")
        logger.info(f"    Present in:     {s['present_count']}/{len(all_ids)} images")
        logger.info(f"    Absent in:      {s['absent_count']}/{len(all_ids)} images")
        logger.info(f"    Mean coverage:  {s['mean_ratio'] * 100:.4f}%")
        logger.info(f"    Max coverage:   {s['max_ratio'] * 100:.4f}%")
        logger.info(f"    Total pixels:   {s['total_pixels']:,}")

    # Save statistics
    save_json(stats, str(output_dir / "class_statistics.json"))

    # Generate plots
    logger.info("\nGenerating class distribution plot...")
    viz.plot_class_distribution(stats)

    # Sample visualizations ---
    logger.info("Generating sample visualizations...")
    sample_ids = all_ids[:5]  # First 5 images
    for img_id in sample_ids:
        img_path = image_dir / f"{img_id}{ext}"
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        masks = {}
        for cls_name in ds["class_names"]:
            mask = dataset._load_single_mask(img_id, cls_name, h, w)
            masks[cls_name] = mask

        viz.plot_sample_with_masks(
            image_rgb, masks, image_id=img_id,
            filename=f"sample_{img_id}.png",
        )

    # 6. Preprocessing comparison
    logger.info("Generating preprocessing comparisons...")
    enhancer = RetinalEnhancer(config["preprocessing"])
    for img_id in sample_ids[:3]:
        img_path = image_dir / f"{img_id}{ext}"
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        stages = enhancer.get_intermediate_results(image)
        viz.plot_preprocessing_stages(
            stages, image_id=img_id,
            filename=f"preprocessing_{img_id}.png",
        )

    logger.info(f"\nAll exploration outputs saved to: {output_dir}")
    logger.info("=" * 60)

    # 7. Print clinical context
    logger.info("\n" + "=" * 60)
    logger.info("  WHY RETINAL LESION SEGMENTATION IS DIFFICULT")
    logger.info("=" * 60)

#Training mode from here
def run_training(config: dict, logger, resume_path: str = None) -> None:
    #Reproducibility
    seed = config["training"]["random_seed"]
    set_seed(seed)
    logger.info(f"Random seed set to {seed}")

    #Device
    device = get_device()

    #Data
    logger.info("Creating dataset split...")
    split = split_dataset(config)
    logger.info(f"  Train: {len(split['train'])} images")
    logger.info(f"  Val:   {len(split['val'])} images")
    logger.info(f"  Test:  {len(split['test'])} images")

    logger.info("Building dataloaders...")
    loaders = build_dataloaders(config, split)

    # Sanity check: load one batch
    train_batch = next(iter(loaders["train"]))
    logger.info(f"  Batch image shape: {train_batch[0].shape}")
    logger.info(f"  Batch mask shape:  {train_batch[1].shape}")

    #Model
    logger.info("Building Attention U-Net...")
    model = build_model(config)
    params = count_parameters(model)
    logger.info(f"  Total parameters:     {params['total']:,}")
    logger.info(f"  Trainable parameters: {params['trainable']:,}")

    #Loss
    criterion = get_loss_function(config)
    logger.info(f"  Loss function: {config['loss']['name']}")

    #Optimizer — differential LR for pretrained encoder vs random decoder
    tc = config["training"]
    encoder_lr_factor = tc.get("encoder_lr_factor", 0.1)  # encoder gets 10x lower LR
    base_lr = tc["learning_rate"]

    # Separate encoder and decoder parameters
    if hasattr(model, 'encoder') and model.encoder_name is not None:
        encoder_params = list(model.encoder.parameters())
        encoder_param_ids = set(id(p) for p in encoder_params)
        decoder_params = [p for p in model.parameters() if id(p) not in encoder_param_ids]

        param_groups = [
            {"params": decoder_params, "lr": base_lr},
            {"params": encoder_params, "lr": base_lr * encoder_lr_factor},
        ]
        logger.info(f"  Differential LR: decoder={base_lr:.1e}, encoder={base_lr * encoder_lr_factor:.1e}")
    else:
        param_groups = [{"params": model.parameters(), "lr": base_lr}]

    if tc["optimizer"].lower() == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=tc["weight_decay"],
        )
    else:
        optimizer = torch.optim.Adam(
            param_groups,
            weight_decay=tc["weight_decay"],
        )
    logger.info(f"  Optimizer: {tc['optimizer']} (base_lr={base_lr})")

    # Encoder freezing: freeze encoder for first N epochs to let decoder warm up
    encoder_freeze_epochs = tc.get("encoder_freeze_epochs", 0)
    if encoder_freeze_epochs > 0 and hasattr(model, 'encoder') and model.encoder_name is not None:
        for param in model.encoder.parameters():
            param.requires_grad = False
        logger.info(f"  Encoder FROZEN for first {encoder_freeze_epochs} epochs")

    #Scheduler
    scheduler = _build_scheduler(optimizer, tc)
    logger.info(f"  Scheduler: {tc['scheduler']}")

    #Resume
    start_epoch = 1
    if resume_path:
        from utils.helpers import load_checkpoint
        ckpt = load_checkpoint(resume_path, model, optimizer, scheduler, device)
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"  Resumed from epoch {start_epoch - 1}")

    #Train
    trainer = Trainer(model, criterion, optimizer, scheduler, config, device)
    history = trainer.fit(loaders["train"], loaders["val"])

    #Evaluate on test set
    logger.info("\n" + "=" * 60)
    logger.info("  Final Evaluation on Test Set")
    logger.info("=" * 60)

    # Load best model
    best_ckpt_path = Path(config["experiment"]["output_root"]) / config["experiment"]["name"] / "checkpoints" / "best.pth"
    if best_ckpt_path.exists():
        from utils.helpers import load_checkpoint
        load_checkpoint(str(best_ckpt_path), model, device=device)
        logger.info(f"  Loaded best model from {best_ckpt_path}")

    eval_cfg = config.get("evaluation", {})
    test_results = evaluate_model(
        model, loaders["test"],
        class_names=config["dataset"]["class_names"],
        device=device,
        threshold=eval_cfg.get("threshold", 0.5),
        min_area=eval_cfg.get("min_area", None),
        use_tta=eval_cfg.get("use_tta", False),
    )

    metrics_obj = SegmentationMetrics(
        class_names=config["dataset"]["class_names"],
        threshold=eval_cfg.get("threshold", 0.5),
        min_area=eval_cfg.get("min_area", None),
    )
    # Manually set the results for formatting
    logger.info("\n" + "=" * 60)
    logger.info("  TEST SET RESULTS")
    logger.info("=" * 60)
    for cls_name in config["dataset"]["class_names"]:
        m = test_results["per_class"][cls_name]
        logger.info(f"  {cls_name}: Dice={m['dice']:.4f} | IoU={m['iou']:.4f} | "
                    f"Prec={m['precision']:.4f} | Recall={m['recall']:.4f}")
    m = test_results["mean"]
    logger.info(f"  MEAN: Dice={m['dice']:.4f} | IoU={m['iou']:.4f} | "
                f"Prec={m['precision']:.4f} | Recall={m['recall']:.4f}")

    # Save test results
    exp_dir = Path(config["experiment"]["output_root"]) / config["experiment"]["name"]
    save_json(test_results, str(exp_dir / "test_results.json"))

    #Visualizations
    logger.info("\nGenerating training plots...")
    viz = Visualizer(
        output_dir=str(exp_dir / "plots"),
        class_names=config["dataset"]["class_names"],
    )
    viz.plot_training_curves(history)
    viz.plot_metric_table(test_results)

    logger.info("\nTraining pipeline complete!")


### Evaluation mode from here

def run_evaluation(config: dict, logger) -> None:
    set_seed(config["training"]["random_seed"])
    device = get_device()

    #Load split
    split = split_dataset(config)
    loaders = build_dataloaders(config, split)

    # Build and load model
    model = build_model(config)
    exp_dir = Path(config["experiment"]["output_root"]) / config["experiment"]["name"]
    best_path = exp_dir / "checkpoints" / "best.pth"

    if not best_path.exists():
        logger.error(f"No checkpoint found at {best_path}. Train the model first.")
        return

    from utils.helpers import load_checkpoint
    load_checkpoint(str(best_path), model, device=device)
    model = model.to(device)
    logger.info(f"Loaded model from {best_path}")

    # Evaluate
    eval_cfg = config.get("evaluation", {})
    results = evaluate_model(
        model, loaders["test"],
        class_names=config["dataset"]["class_names"],
        device=device,
        threshold=eval_cfg.get("threshold", 0.5),
        min_area=eval_cfg.get("min_area", None),
        use_tta=eval_cfg.get("use_tta", False),
    )

    metrics = SegmentationMetrics(
        class_names=config["dataset"]["class_names"],
        threshold=eval_cfg.get("threshold", 0.5),
        min_area=eval_cfg.get("min_area", None),
    )
    logger.info("\n" + metrics.format_results(results))

    save_json(results, str(exp_dir / "test_results.json"))

    # Visualizations
    viz = Visualizer(
        output_dir=str(exp_dir / "plots"),
        class_names=config["dataset"]["class_names"],
    )
    viz.plot_metric_table(results)

    # Generate prediction visualizations
    logger.info("Generating prediction visualizations...")
    model.to(device).eval()
    test_loader = loaders["test"]

    for batch_idx, (images, masks, meta) in enumerate(test_loader):
        if batch_idx >= 1:  # Just first batch
            break
        images_gpu = images.to(device)
        with torch.no_grad():
            preds = model(images_gpu)

        viz.plot_predictions(
            images, masks, preds.cpu(),
            image_ids=[m["image_id"] for m in meta] if isinstance(meta, list) else [meta["image_id"][i] for i in range(images.shape[0])],
            filename=f"test_predictions_batch{batch_idx}.png",
        )

    logger.info("Evaluation complete!")



def _build_scheduler(optimizer, train_config: dict):
    """Build the learning rate scheduler from config."""
    name = train_config.get("scheduler", "cosine").lower()

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_config.get("scheduler_T_max", 50),
            eta_min=train_config.get("min_lr", 1e-7),
        )
    elif name == "cosine_warm":
        # CosineAnnealingWarmRestarts: resets LR every T_0 epochs
        # T_mult=2 means each restart doubles the period (30, 60, 120, ...)
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=train_config.get("scheduler_T_0", 30),
            T_mult=train_config.get("scheduler_T_mult", 2),
            eta_min=train_config.get("min_lr", 1e-6),
        )
        warmup_epochs = train_config.get("warmup_epochs", 0)
        if warmup_epochs > 0:
            return WarmupScheduler(optimizer, base_scheduler, warmup_epochs,
                                   train_config.get("min_lr", 1e-6))
        return base_scheduler
    elif name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=train_config.get("scheduler_patience", 10),
            factor=train_config.get("scheduler_factor", 0.5),
            min_lr=train_config.get("min_lr", 1e-7),
        )
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=train_config.get("scheduler_patience", 10),
            gamma=train_config.get("scheduler_factor", 0.5),
        )
    else:
        return None


class WarmupScheduler:
    """Linear warmup wrapper for any base scheduler.
    
    Ramps LR linearly from warmup_start_lr to the optimizer's initial LR
    over warmup_epochs, then delegates to the base scheduler.
    """
    def __init__(self, optimizer, base_scheduler, warmup_epochs, warmup_start_lr=1e-6):
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.target_lr = optimizer.param_groups[0]["lr"]
        self.current_epoch = 0

    def step(self, *args, **kwargs):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            alpha = self.current_epoch / self.warmup_epochs
            lr = self.warmup_start_lr + alpha * (self.target_lr - self.warmup_start_lr)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        else:
            self.base_scheduler.step(*args, **kwargs)

    def state_dict(self):
        return {
            "base": self.base_scheduler.state_dict(),
            "current_epoch": self.current_epoch,
        }

    def load_state_dict(self, state_dict):
        self.base_scheduler.load_state_dict(state_dict["base"])
        self.current_epoch = state_dict["current_epoch"]


# MAIN CODE
def main():
    args = parse_args()

    #Load config
    config = load_config(args.config)

    # Override experiment name if provided
    if args.experiment:
        config["experiment"]["name"] = args.experiment

    # Setup logger
    log_cfg = config.get("logging", {})
    logger = get_logger(
        "RetinaAI",
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("log_file", None),
    )

    logger.info("=" * 60)
    logger.info("  RetinaAI — Automated DR Screening System")
    logger.info("  Stage 1: Lesion Segmentation")
    logger.info("=" * 60)

    # Route to the appropriate mode
    if args.mode == "explore":
        run_exploration(config, logger)
    elif args.mode == "train":
        run_training(config, logger, resume_path=args.resume)
    elif args.mode == "eval":
        run_evaluation(config, logger)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
