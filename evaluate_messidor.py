import os
import sys
import argparse
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score, recall_score, confusion_matrix

from utils.helpers import (
    set_seed,
    load_config,
    get_device,
    ensure_dir,
    save_json,
    format_time,
)
from utils.logger import get_logger
from datasets.messidor_dataset import build_messidor_dataloader
from models.attention_unet import build_model
from models.hybrid_classifier import build_classifier
from train_classifier import get_lesion_count_features


def parse_args():
    parser = argparse.ArgumentParser(
        description="RetinaAI — MESSIDOR-2 Cross-Dataset Zero-Shot Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_messidor.yaml",
        help="Path to MESSIDOR-2 evaluation config file.",
    )
    parser.add_argument(
        "--seg_checkpoint",
        type=str,
        default="experiments/exp_ddr_attention_unet/checkpoints/best.pth",
        help="Path to Stage 1 Attention U-Net checkpoint.",
    )
    parser.add_argument(
        "--cls_checkpoint",
        type=str,
        default="experiments/exp_12_cls_hybrid_high_recall/checkpoints/best.pth",
        help="Path to Stage 2 Hybrid Classifier checkpoint.",
    )
    parser.add_argument(
        "--limit_batches",
        type=int,
        default=None,
        help="Limit number of batches for debugging.",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate_messidor_pipeline(
    seg_model: nn.Module,
    cls_model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config: dict,
    device: torch.device,
    limit_batches: Optional[int] = None,
) -> dict:
    seg_model.eval()
    cls_model.eval()

    class_names = config["dataset"]["class_names"]
    thresholds = config["evaluation"]["threshold"]
    min_areas = config["evaluation"]["min_area"]

    all_preds = []
    all_probs = []
    all_labels = []
    all_lesion_counts = []
    image_ids = []

    for batch_idx, (images, labels, meta) in enumerate(dataloader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break

        images = images.to(device)

        # Stage 1: Segmentation
        seg_logits = seg_model(images)
        seg_probs = torch.sigmoid(seg_logits)

        # Extract 4-class Lesion Counts
        counts = get_lesion_count_features(
            seg_probs=seg_probs,
            class_names=class_names,
            thresholds=thresholds,
            min_areas=min_areas,
            device=device,
        )

        # Stage 2: Hybrid DR Classifier Inference
        cls_logits = cls_model(images, counts)
        probs = torch.softmax(cls_logits, dim=1)

        calib_w = config.get("evaluation", {}).get("class_calibration_weights", None)
        if calib_w is not None:
            w_tensor = torch.tensor(calib_w, device=device).unsqueeze(0)
            adj_probs = probs * w_tensor
            preds = torch.argmax(adj_probs, dim=1)
        else:
            preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels if isinstance(labels, list) else labels.cpu().numpy().tolist())
        all_lesion_counts.extend(counts.cpu().numpy().tolist())
        image_ids.extend(meta["image_id"])

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_lesion_counts = np.array(all_lesion_counts)

    # Compute predicted class distribution
    class_pred_counts = np.bincount(all_preds, minlength=5).tolist()

    results = {
        "num_evaluated_images": len(all_preds),
        "class_predictions_distribution": {
            "Grade_0_Healthy": class_pred_counts[0],
            "Grade_1_Mild": class_pred_counts[1],
            "Grade_2_Moderate": class_pred_counts[2],
            "Grade_3_Severe": class_pred_counts[3],
            "Grade_4_Proliferative": class_pred_counts[4],
        },
        "mean_detected_lesion_counts": {
            "MA": float(np.mean(all_lesion_counts[:, 0])),
            "HE": float(np.mean(all_lesion_counts[:, 1])),
            "EX": float(np.mean(all_lesion_counts[:, 2])),
            "SE": float(np.mean(all_lesion_counts[:, 3])),
        },
    }

    # If ground truth labels exist in CSV metadata
    if len(np.unique(all_labels)) > 1:
        acc = float(accuracy_score(all_labels, all_preds))
        qwk = float(cohen_kappa_score(all_labels, all_preds, weights="quadratic"))
        macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
        macro_rec = float(recall_score(all_labels, all_preds, average="macro", zero_division=0))
        per_class_rec = recall_score(all_labels, all_preds, average=None, zero_division=0).tolist()

        # Binary Referable DR Sensitivity (Grades 1, 2, 3, 4 vs Grade 0)
        binary_gt = (all_labels > 0).astype(int)
        binary_pred = (all_preds > 0).astype(int)
        referable_recall = float(recall_score(binary_gt, binary_pred, zero_division=0))

        results.update({
            "accuracy": acc,
            "qwk": qwk,
            "f1_macro": macro_f1,
            "recall_macro": macro_rec,
            "referable_dr_recall": referable_recall,
            "per_class_recalls": per_class_rec,
        })

    return results


def main():
    args = parse_args()

    config = load_config(args.config)
    exp_name = config["experiment"].get("name", "exp_messidor_eval")
    output_root = config["experiment"].get("output_root", "experiments")
    exp_dir = ensure_dir(Path(output_root) / exp_name)

    logger = get_logger("MessidorEval", log_file=str(exp_dir / "evaluation.log"))
    logger.info("=" * 60)
    logger.info("  RetinaAI — MESSIDOR-2 Cross-Dataset Zero-Shot Evaluation")
    logger.info("=" * 60)

    device = get_device()
    logger.info(f"Using Device: {device}")

    # Build Dataloader
    logger.info("Loading MESSIDOR-2 Dataset...")
    dataloader = build_messidor_dataloader(config, batch_size=8)
    logger.info(f"Total MESSIDOR-2 Images Loaded: {len(dataloader.dataset)}")

    # Load Stage 1 Segmentation Model
    logger.info(f"Loading Stage 1 Attention U-Net from: {args.seg_checkpoint}")
    seg_model = build_model(config).to(device)
    if Path(args.seg_checkpoint).exists():
        ckpt = torch.load(args.seg_checkpoint, map_location=device, weights_only=False)
        seg_model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded Stage 1 weights successfully.")

    # Load Stage 2 Hybrid DR Classifier
    cls_ckpt_path = config.get("checkpoints", {}).get("cls_checkpoint", args.cls_checkpoint)
    logger.info(f"Loading Stage 2 Hybrid DR Classifier from: {cls_ckpt_path}")
    cls_model = build_classifier(config, in_channels=3).to(device)
    if Path(cls_ckpt_path).exists():
        ckpt = torch.load(cls_ckpt_path, map_location=device, weights_only=False)
        cls_model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded Stage 2 weights successfully.")

    # Run Evaluation Loop
    start_time = time.time()
    logger.info("Executing Zero-Shot Evaluation Loop across MESSIDOR-2 images...")
    results = evaluate_messidor_pipeline(
        seg_model=seg_model,
        cls_model=cls_model,
        dataloader=dataloader,
        config=config,
        device=device,
        limit_batches=args.limit_batches,
    )
    elapsed = time.time() - start_time
    logger.info(f"Evaluation completed in {format_time(elapsed)}")

    # Save results
    save_path = exp_dir / "messidor_results.json"
    save_json(results, str(save_path))
    logger.info(f"MESSIDOR-2 evaluation results saved to: {save_path}")

    # Print Summary
    logger.info("=" * 60)
    logger.info("  MESSIDOR-2 Cross-Dataset Generalization Summary")
    logger.info("=" * 60)
    logger.info(f"Evaluated Images: {results['num_evaluated_images']}")
    logger.info("Predicted DR Severity Distribution:")
    for grade, count in results["class_predictions_distribution"].items():
        pct = (count / results['num_evaluated_images']) * 100
        logger.info(f"  {grade}: {count} images ({pct:.1f}%)")

    logger.info("Average Lesion Counts Detected per Image:")
    for lesion, count in results["mean_detected_lesion_counts"].items():
        logger.info(f"  {lesion}: {count:.2f} instances/image")

    if "accuracy" in results:
        logger.info("-" * 60)
        logger.info(f"Cross-Dataset Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Cross-Dataset QWK:      {results['qwk']:.4f}")
        logger.info(f"Cross-Dataset Recall:   {results['recall_macro']:.4f}")
        logger.info(f"Referable DR Recall:    {results['referable_dr_recall']:.4f}")

    logger.info("Evaluation finished successfully.")


if __name__ == "__main__":
    main()
