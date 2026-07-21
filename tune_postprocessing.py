import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch

from datasets.idrid_dataset import build_dataloaders, split_dataset
from evaluation.metrics import evaluate_model
from models.attention_unet import build_model
from utils.helpers import get_device, load_checkpoint, load_config, save_json, set_seed


def parse_float_list(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_int_list(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def class_index(config: Dict, class_name: str) -> int:
    return config["dataset"]["class_names"].index(class_name)


@torch.no_grad()
def collect_class_probabilities(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    target_idx: int,
    use_tta: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    model.eval()
    probabilities: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    for images, masks, meta in loader:
        images = images.to(device)

        logits = model(images)
        if use_tta:
            flipped_images = torch.flip(images, dims=[3])
            flipped_logits = model(flipped_images)
            logits = (logits + torch.flip(flipped_logits, dims=[3])) / 2.0

        probs = torch.sigmoid(logits[:, target_idx]).detach().cpu().numpy()
        masks_np = masks[:, target_idx].cpu().numpy().astype(np.uint8)

        valid_classes = None
        if isinstance(meta, dict) and "valid_classes" in meta:
            valid_classes = meta["valid_classes"][:, target_idx].numpy()

        for i in range(probs.shape[0]):
            if valid_classes is not None and valid_classes[i] == 0:
                continue
            probabilities.append((np.clip(probs[i], 0.0, 1.0) * 255).astype(np.uint8))
            targets.append(masks_np[i])

    return probabilities, targets


def compute_binary_metrics(
    probabilities: Iterable[np.ndarray],
    targets: Iterable[np.ndarray],
    threshold: float,
    min_area: int,
) -> Dict[str, float]:
    threshold_value = int(round(threshold * 255))
    tp = fp = fn = tn = 0

    for prob, target in zip(probabilities, targets):
        pred = (prob > threshold_value).astype(np.uint8)

        if min_area > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred, connectivity=8)
            for label_idx in range(1, num_labels):
                if stats[label_idx, cv2.CC_STAT_AREA] < min_area:
                    pred[labels == label_idx] = 0

        target = (target > 0).astype(np.uint8)
        tp += int(np.sum(pred * target))
        fp += int(np.sum(pred * (1 - target)))
        fn += int(np.sum((1 - pred) * target))
        tn += int(np.sum((1 - pred) * (1 - target)))

    smooth = 1e-6
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    specificity = (tn + smooth) / (tn + fp + smooth)

    return {
        "threshold": threshold,
        "min_area": min_area,
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def sweep(
    probabilities: List[np.ndarray],
    targets: List[np.ndarray],
    thresholds: List[float],
    min_areas: List[int],
) -> List[Dict[str, float]]:
    rows = []
    for threshold in thresholds:
        for min_area in min_areas:
            rows.append(compute_binary_metrics(probabilities, targets, threshold, min_area))
    rows.sort(key=lambda row: (row["dice"], row["precision"]), reverse=True)
    return rows


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune per-class segmentation post-processing.")
    parser.add_argument("--config", default="experiments/exp_08/config.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--class-name", default="SE")
    parser.add_argument("--thresholds", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--min-areas", default="0,5,10,20,50,100,150,200,300")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["random_seed"])

    exp_dir = Path(config["experiment"]["output_root"]) / config["experiment"]["name"]
    checkpoint = Path(args.checkpoint) if args.checkpoint else exp_dir / "checkpoints" / "best.pth"

    config["training"]["shuffle"] = False
    device = get_device()
    split = split_dataset(config)
    loaders = build_dataloaders(config, split)

    model = build_model(config).to(device)
    load_checkpoint(str(checkpoint), model, device=device)

    eval_cfg = config.get("evaluation", {})
    use_tta = bool(eval_cfg.get("use_tta", False))
    target_idx = class_index(config, args.class_name)

    print(f"Loaded {checkpoint}")
    print(f"Tuning {args.class_name} on validation set...")
    val_probs, val_targets = collect_class_probabilities(
        model, loaders["val"], device, target_idx, use_tta
    )
    rows = sweep(
        val_probs,
        val_targets,
        parse_float_list(args.thresholds),
        parse_int_list(args.min_areas),
    )

    sweep_path = exp_dir / f"postprocess_sweep_{args.class_name}.csv"
    write_csv(sweep_path, rows)
    best = rows[0]

    tuned_thresholds = dict(eval_cfg.get("threshold", {}))
    tuned_min_areas = dict(eval_cfg.get("min_area", {}))
    tuned_thresholds[args.class_name] = best["threshold"]
    tuned_min_areas[args.class_name] = best["min_area"]

    print("Best validation setting:")
    print(
        f"  {args.class_name}: threshold={best['threshold']:.2f}, "
        f"min_area={best['min_area']}, dice={best['dice']:.4f}, "
        f"precision={best['precision']:.4f}, recall={best['recall']:.4f}"
    )

    print("Evaluating tuned setting on test set...")
    tuned_results = evaluate_model(
        model,
        loaders["test"],
        class_names=config["dataset"]["class_names"],
        device=device,
        threshold=tuned_thresholds,
        min_area=tuned_min_areas,
        use_tta=use_tta,
    )

    output = {
        "tuned_class": args.class_name,
        "selected_on": "validation",
        "best_validation": best,
        "threshold": tuned_thresholds,
        "min_area": tuned_min_areas,
        "test_results": tuned_results,
    }
    save_json(output, str(exp_dir / f"postprocess_tuned_{args.class_name}.json"))

    for cls in config["dataset"]["class_names"]:
        metrics = tuned_results["per_class"][cls]
        print(
            f"  {cls}: Dice={metrics['dice']:.4f} | IoU={metrics['iou']:.4f} | "
            f"Prec={metrics['precision']:.4f} | Recall={metrics['recall']:.4f}"
        )
    mean = tuned_results["mean"]
    print(
        f"  MEAN: Dice={mean['dice']:.4f} | IoU={mean['iou']:.4f} | "
        f"Prec={mean['precision']:.4f} | Recall={mean['recall']:.4f}"
    )
    print(f"Saved sweep: {sweep_path}")
    print(f"Saved tuned results: {exp_dir / f'postprocess_tuned_{args.class_name}.json'}")


if __name__ == "__main__":
    main()
