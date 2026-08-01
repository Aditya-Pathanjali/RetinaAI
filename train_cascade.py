import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from datasets.idrid_dataset import build_dataloaders, split_dataset
from evaluation.metrics import SegmentationMetrics, evaluate_model
from main import _build_scheduler
from models.cascade_unet import build_cascade_model
from models.losses import get_loss_function
from training.trainer import Trainer
from utils.helpers import count_parameters, get_device, load_checkpoint, load_config, save_json, set_seed
from utils.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="RetinaAI — Train Dual-Stage Cascade Lesion Model.")
    parser.add_argument("--config", default="configs/config_multiclass_patch_hybrid.yaml", help="Path to config file.")
    parser.add_argument("--experiment", default="exp_11_cascade_dual_stage", help="Experiment name.")
    parser.add_argument("--resume", default=None, help="Checkpoint path.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.experiment:
        config["experiment"]["name"] = args.experiment

    # Enforce boundary focal tversky loss for cascade refiner
    config["loss"]["name"] = "boundary_focal_tversky"
    config["loss"]["boundary_weight"] = 0.35

    set_seed(config["training"].get("random_seed", 42))
    logger = get_logger("CascadeTrainer", level=config.get("logging", {}).get("level", "INFO"))
    device = get_device()

    logger.info("Initializing Dual-Stage Cascade Training Pipeline...")
    split = split_dataset(config)
    loaders = build_dataloaders(config, split)

    model = build_cascade_model(config).to(device)
    params = count_parameters(model)
    logger.info(f"Parameters: {params['trainable']:,} trainable / {params['total']:,} total")

    criterion = get_loss_function(config)
    tc = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc["learning_rate"],
        weight_decay=tc.get("weight_decay", 1e-4),
    )
    scheduler = _build_scheduler(optimizer, tc)

    start_epoch = 1
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch = ckpt.get("epoch", 0) + 1

    trainer = Trainer(model, criterion, optimizer, scheduler, config, device)
    trainer.fit(loaders["train"], loaders["val"], start_epoch=start_epoch)

    exp_dir = Path(config["experiment"]["output_root"]) / config["experiment"]["name"]
    best_path = exp_dir / "checkpoints" / "best.pth"
    if best_path.exists():
        load_checkpoint(str(best_path), model, device=device)

    eval_cfg = config.get("evaluation", {})
    results = evaluate_model(
        model,
        loaders["test"],
        class_names=config["dataset"]["class_names"],
        device=device,
        threshold=eval_cfg.get("threshold", 0.5),
        min_area=eval_cfg.get("min_area", None),
        use_tta=eval_cfg.get("use_tta", True),
    )
    save_json(results, str(exp_dir / "test_results.json"))

    metrics = SegmentationMetrics(
        class_names=config["dataset"]["class_names"],
        threshold=eval_cfg.get("threshold", 0.5),
        min_area=eval_cfg.get("min_area", None),
    )
    logger.info("\n" + metrics.format_results(results))


if __name__ == "__main__":
    main()
