import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np
import torch

from datasets.idrid_dataset import build_dataloaders, split_dataset
from evaluation.metrics import evaluate_model
from models.attention_unet import build_model
from tune_postprocessing import collect_class_probabilities
from utils.helpers import get_device, load_checkpoint, load_config, save_json, set_seed


def values_from_range(bounds: List[float], steps: int, as_int: bool = False) -> List:
    if len(bounds) != 2:
        return bounds
    values = np.linspace(bounds[0], bounds[1], steps)
    if as_int:
        return sorted(set(int(round(v)) for v in values))
    return [round(float(v), 4) for v in values]


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
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return {
        "threshold": threshold,
        "min_area": min_area,
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune all segmentation post-processing settings.")
    parser.add_argument("--config", default="experiments/exp_03_recover_50/config.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--threshold-steps", type=int, default=11)
    parser.add_argument("--area-steps", type=int, default=9)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["random_seed"])

    exp_dir = Path(config["experiment"]["output_root"]) / config["experiment"]["name"]
    checkpoint = Path(args.checkpoint) if args.checkpoint else exp_dir / "checkpoints" / "best.pth"
    device = get_device()

    split = split_dataset(config)
    loaders = build_dataloaders(config, split)
    model = build_model(config).to(device)
    load_checkpoint(str(checkpoint), model, device=device)

    class_names = config["dataset"]["class_names"]
    eval_cfg = config.get("evaluation", {})
    tuned_thresholds = dict(eval_cfg.get("threshold", {}))
    tuned_min_areas = dict(eval_cfg.get("min_area", {}))
    best_by_class = {}

    for class_name in class_names:
        target_idx = class_names.index(class_name)
        threshold_bounds = eval_cfg.get("threshold_search", {}).get(class_name, [tuned_thresholds.get(class_name, 0.5)])
        area_bounds = eval_cfg.get("min_area_search", {}).get(class_name, [tuned_min_areas.get(class_name, 0)])
        thresholds = values_from_range(threshold_bounds, args.threshold_steps, as_int=False)
        min_areas = values_from_range(area_bounds, args.area_steps, as_int=True)

        probabilities, targets = collect_class_probabilities(
            model, loaders["val"], device, target_idx, bool(eval_cfg.get("use_tta", False))
        )
        rows = [
            compute_binary_metrics(probabilities, targets, threshold, min_area)
            for threshold in thresholds
            for min_area in min_areas
        ]
        rows.sort(key=lambda row: (row["dice"], row["precision"]), reverse=True)
        best = rows[0]
        tuned_thresholds[class_name] = best["threshold"]
        tuned_min_areas[class_name] = best["min_area"]
        best_by_class[class_name] = best
        write_csv(exp_dir / f"postprocess_sweep_{class_name}.csv", rows)

    test_results = evaluate_model(
        model,
        loaders["test"],
        class_names=class_names,
        device=device,
        threshold=tuned_thresholds,
        min_area=tuned_min_areas,
        use_tta=bool(eval_cfg.get("use_tta", False)),
    )

    output = {
        "selected_on": "validation",
        "best_by_class": best_by_class,
        "threshold": tuned_thresholds,
        "min_area": tuned_min_areas,
        "test_results": test_results,
    }
    save_json(output, str(exp_dir / "postprocess_tuned_all.json"))

    print("Best validation settings:")
    for class_name, best in best_by_class.items():
        print(
            f"  {class_name}: threshold={best['threshold']}, min_area={best['min_area']}, "
            f"dice={best['dice']:.4f}, precision={best['precision']:.4f}, recall={best['recall']:.4f}"
        )
    print("Tuned test Dice:")
    for class_name in class_names:
        print(f"  {class_name}: {test_results['per_class'][class_name]['dice']:.4f}")
    print(f"  MEAN: {test_results['mean']['dice']:.4f}")
    print(f"Saved: {exp_dir / 'postprocess_tuned_all.json'}")


if __name__ == "__main__":
    main()
