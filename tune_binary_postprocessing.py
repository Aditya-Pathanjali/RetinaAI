import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from datasets.binary_lesion_dataset import build_binary_dataloaders
from models.attention_unet import build_model
from utils.helpers import get_device, load_checkpoint, load_config, save_json, set_seed


def values_from_range(bounds, steps, cast):
    start, end = bounds
    vals = np.linspace(start, end, steps)
    return [cast(v) for v in vals]


@torch.no_grad()
def collect_probs(model, loader, device, use_tta):
    model.eval()
    probs_all = []
    masks_all = []
    for images, masks, _ in loader:
        images = images.to(device)
        logits = model(images)
        if use_tta:
            flipped = torch.flip(images, dims=[3])
            flipped_logits = model(flipped)
            logits = (logits + torch.flip(flipped_logits, dims=[3])) / 2.0
        probs = torch.sigmoid(logits[:, 0]).cpu().numpy()
        masks = masks[:, 0].cpu().numpy().astype(np.uint8)
        probs_all.extend([(np.clip(p, 0.0, 1.0) * 255).astype(np.uint8) for p in probs])
        masks_all.extend(masks)
    return probs_all, masks_all


def score(probs_all, masks_all, threshold, min_area):
    threshold_value = int(round(threshold * 255))
    tp = fp = fn = tn = 0
    for probs, mask in zip(probs_all, masks_all):
        pred = (probs > threshold_value).astype(np.uint8)
        if min_area > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred, connectivity=8)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < min_area:
                    pred[labels == i] = 0
        mask = (mask > 0).astype(np.uint8)
        tp += int(np.sum(pred * mask))
        fp += int(np.sum(pred * (1 - mask)))
        fn += int(np.sum((1 - pred) * mask))
        tn += int(np.sum((1 - pred) * (1 - mask)))

    smooth = 1e-6
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return {
        "threshold": threshold,
        "min_area": min_area,
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main():
    parser = argparse.ArgumentParser(description="Tune binary model threshold and min-area.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--threshold-steps", type=int, default=10)
    parser.add_argument("--area-steps", type=int, default=8)
    args = parser.parse_args()

    config = load_config(args.config)
    target = config["binary"]["target_class"]
    config["dataset"]["class_names"] = [target]
    config["model"]["out_channels"] = 1
    set_seed(config["training"].get("random_seed", 42))

    exp_dir = Path(config["experiment"]["output_root"]) / config["experiment"]["name"]
    checkpoint = Path(args.checkpoint) if args.checkpoint else exp_dir / "checkpoints" / "best.pth"

    device = get_device()
    loaders = build_binary_dataloaders(config)
    model = build_model(config).to(device)
    load_checkpoint(str(checkpoint), model, device=device)

    eval_cfg = config["evaluation"]
    threshold_bounds = eval_cfg["threshold_search"][target]
    area_bounds = eval_cfg["min_area_search"][target]
    thresholds = values_from_range(threshold_bounds, args.threshold_steps, float)
    min_areas = sorted(set(values_from_range(area_bounds, args.area_steps, lambda v: int(round(v)))))

    probs_all, masks_all = collect_probs(model, loaders["val"], device, eval_cfg.get("use_tta", False))
    rows = [score(probs_all, masks_all, threshold, min_area) for threshold in thresholds for min_area in min_areas]
    rows.sort(key=lambda r: (r["dice"], r["precision"]), reverse=True)

    csv_path = exp_dir / f"postprocess_sweep_{target}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    save_json(best, str(exp_dir / f"best_postprocess_{target}.json"))
    print(
        f"{target}: threshold={best['threshold']:.4f}, min_area={best['min_area']}, "
        f"val_dice={best['dice']:.4f}, precision={best['precision']:.4f}, recall={best['recall']:.4f}"
    )
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
