import argparse
import sys
import json
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import load_config, get_device, save_json
from utils.logger import get_logger
from datasets.aptos_dataset import build_aptos_dataloaders
from models.attention_unet import build_model
from models.hybrid_classifier import build_classifier
from train_classifier import validate_epoch, compute_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="RetinaAI — Evaluate Classification Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file.")
    parser.add_argument("--variant", type=str, default="hybrid", choices=["hybrid", "classifier_only", "mask_input"], help="Ablation study variant")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to classifier checkpoint. Default: best.pth in experiment folder")
    parser.add_argument("--seg_checkpoint", type=str, default=None, help="Path to segmentation checkpoint.")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    device = get_device()
    
    exp_name = f"{config['experiment']['name']}_cls_{args.variant}"
    exp_dir = Path(config["experiment"]["output_root"]) / exp_name
    
    # Resolve checkpoint path
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = exp_dir / "checkpoints" / "best.pth"
        
    if not ckpt_path.exists():
        print(f"Error: Classifier checkpoint not found at: {ckpt_path}")
        sys.exit(1)
        
    print(f"Evaluating variant: {args.variant.upper()}")
    print(f"Loading checkpoint: {ckpt_path}")
    
    # Load segmentation model if hybrid or mask_input
    seg_model = None
    class_names = config["dataset"]["class_names"]
    thresholds = {}
    min_areas = {}
    
    if args.variant in ["hybrid", "mask_input"]:
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
            
        if args.seg_checkpoint:
            seg_ckpt_path = Path(args.seg_checkpoint)
        else:
            seg_ckpt_path = Path(config["experiment"]["output_root"]) / config["experiment"]["name"] / "checkpoints" / "best.pth"
            if not seg_ckpt_path.exists():
                seg_ckpt_path = Path(config["experiment"]["output_root"]) / "exp_03_optimized" / "checkpoints" / "best.pth"
                
        if not seg_ckpt_path.exists():
            print(f"Error: Segmentation checkpoint not found at {seg_ckpt_path}")
            sys.exit(1)
            
        print(f"Loading segmentation model: {seg_ckpt_path}")
        seg_model = build_model(config)
        ckpt = torch.load(seg_ckpt_path, map_location=device)
        seg_model.load_state_dict(ckpt["model_state_dict"])
        seg_model = seg_model.to(device)
        seg_model.eval()
        
    # Build classifier
    config["classification_model"]["lesion_feature_dim"] = len(class_names) if args.variant == "hybrid" else 0
    in_channels = 3 + len(class_names) if args.variant == "mask_input" else 3
    
    classifier = build_classifier(config, in_channels=in_channels)
    ckpt = torch.load(ckpt_path, map_location=device)
    classifier.load_state_dict(ckpt["model_state_dict"])
    classifier = classifier.to(device)
    classifier.eval()
    
    # Dataloader
    print("Loading test data...")
    loaders = build_aptos_dataloaders(config)   
    test_loader = loaders["test"]
    
    # Running evaluation
    criterion = nn.CrossEntropyLoss()
    use_amp = config.get("classification_training", {}).get("use_amp", True) and device.type == "cuda"
    
    print("Running evaluation on test set...")
    test_loss, test_preds, test_labels = validate_epoch(
        classifier=classifier,
        seg_model=seg_model,
        loader=test_loader,
        criterion=criterion,
        variant=args.variant,
        class_names=class_names,
        thresholds=thresholds,
        min_areas=min_areas,
        use_amp=use_amp,
        device=device
    )
    
    test_qwk, test_acc, test_f1 = compute_metrics(np.array(test_preds), np.array(test_labels))
    from sklearn.metrics import recall_score
    test_recall = recall_score(test_labels, test_preds, average="macro", zero_division=0)
    class_recalls = recall_score(test_labels, test_preds, average=None, zero_division=0)
    
    print("\n" + "=" * 50)
    print("  TEST EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test F1-Score: {test_f1:.4f}")
    print(f"  Test QWK:      {test_qwk:.4f}")
    print(f"  Test Recall:   {test_recall:.4f}")
    print(f"  Class Recalls: {[round(float(r), 4) for r in class_recalls]}")
    print("=" * 50)
    
    test_results = {
        "loss": test_loss,
        "accuracy": test_acc,
        "f1_score": test_f1,
        "qwk": test_qwk,
        "recall_macro": float(test_recall),
        "class_recalls": [float(r) for r in class_recalls],
    }
    
    save_path = exp_dir / "test_results.json"
    save_json(test_results, str(save_path))
    print(f"Test results saved to: {save_path}\n")

if __name__ == "__main__":
    main()
