import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from datasets.binary_lesion_dataset import build_binary_dataloaders
from evaluation.metrics import SegmentationMetrics, evaluate_model
from main import _build_scheduler
from models.attention_unet import build_model
from models.losses import get_loss_function
from training.trainer import Trainer
from utils.helpers import count_parameters, get_device, load_checkpoint, load_config, save_json, set_seed
from utils.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train one binary lesion segmentation model.")
    parser.add_argument("--config", required=True, help="Binary lesion config path.")
    parser.add_argument("--experiment", default=None, help="Optional experiment-name override.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint path.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.experiment:
        config["experiment"]["name"] = args.experiment

    target = config["binary"]["target_class"]
    config["dataset"]["class_names"] = [target]
    config["model"]["out_channels"] = 1

    set_seed(config["training"].get("random_seed", 42))
    logger = get_logger("BinaryTrainer", level=config.get("logging", {}).get("level", "INFO"))
    device = get_device()

    logger.info(f"Training binary {target} model")
    loaders = build_binary_dataloaders(config)
    logger.info(f"Train batches: {len(loaders['train'])}")
    logger.info(f"Val batches:   {len(loaders['val'])}")
    logger.info(f"Test batches:  {len(loaders['test'])}")

    model = build_model(config)
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
    final_model_path = exp_dir / f"model_{target.lower()}.pth"
    if best_path.exists():
        shutil.copyfile(best_path, final_model_path)
        load_checkpoint(str(best_path), model, device=device)
        logger.info(f"Saved binary model: {final_model_path}")

    eval_cfg = config.get("evaluation", {})
    results = evaluate_model(
        model,
        loaders["test"],
        class_names=[target],
        device=device,
        threshold=eval_cfg.get("threshold", 0.5),
        min_area=eval_cfg.get("min_area", None),
        use_tta=eval_cfg.get("use_tta", False),
    )
    save_json(results, str(exp_dir / "test_results.json"))

    metrics = SegmentationMetrics(
        class_names=[target],
        threshold=eval_cfg.get("threshold", 0.5),
        min_area=eval_cfg.get("min_area", None),
    )
    logger.info("\n" + metrics.format_results(results))


if __name__ == "__main__":
    main()
