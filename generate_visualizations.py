import os
import sys
import json
import yaml
import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import albumentations as A
import torch.nn.functional as F

from models.attention_unet import AttentionUNet
from datasets.idrid_dataset import IDRiDDataset
from evaluation.metrics import SegmentationMetrics
from utils.visualization import CLASS_COLOURS

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_sliding_window_predictions(model, image_tensor, patch_size=512, overlap=0.25, use_tta=True):
    B, C, H, W = image_tensor.shape
    device = image_tensor.device
    
    stride = int(patch_size * (1 - overlap))
    
    out_logits = torch.zeros((B, 4, H, W), device=device)
    count = torch.zeros((B, 1, H, W), device=device)
    
    pad_h = max(0, patch_size - H)
    pad_w = max(0, patch_size - W)
    if pad_h > 0 or pad_w > 0:
        image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h))
        out_logits = F.pad(out_logits, (0, pad_w, 0, pad_h))
        count = F.pad(count, (0, pad_w, 0, pad_h))
        H_pad, W_pad = image_tensor.shape[2:]
    else:
        H_pad, W_pad = H, W

    y_steps = list(range(0, H_pad - patch_size + 1, stride))
    if not y_steps or y_steps[-1] != H_pad - patch_size: y_steps.append(H_pad - patch_size)
    
    x_steps = list(range(0, W_pad - patch_size + 1, stride))
    if not x_steps or x_steps[-1] != W_pad - patch_size: x_steps.append(W_pad - patch_size)
    
    for y in y_steps:
        for x in x_steps:
            patch = image_tensor[:, :, y:y+patch_size, x:x+patch_size]
            with torch.no_grad():
                if use_tta:
                    l1 = model(patch)
                    l2 = torch.flip(model(torch.flip(patch, dims=[3])), dims=[3])
                    logits = (l1 + l2) / 2.0
                else:
                    logits = model(patch)
            
            out_logits[:, :, y:y+patch_size, x:x+patch_size] += logits
            count[:, :, y:y+patch_size, x:x+patch_size] += 1
            
    out_logits = out_logits / count
    if pad_h > 0 or pad_w > 0:
        out_logits = out_logits[:, :, :H, :W]
        
    return out_logits

def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    img = tensor.clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img

def main():
    config_path = "configs/config.yaml"
    config = load_config(config_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    exp_dir = Path(config["experiment"]["output_root"]) / config["experiment"]["name"]
    viz_dir = exp_dir / "visualizations"
    
    # Create required subdirectories
    subdirs = ["heatmaps", "overlays", "error_maps", "best_samples", "worst_samples", 
               "probability_maps", "confusion_maps", "reports"]
    for subdir in subdirs:
        (viz_dir / subdir).mkdir(parents=True, exist_ok=True)
        
    # Load Model
    print("Loading model...")
    model = AttentionUNet(
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=config["model"]["encoder_weights"],
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
    ).to(device)
    
    ckpt_path = exp_dir / "checkpoints" / "best.pth"
    if not ckpt_path.exists():
        # Fallback to exp_03_optimized
        fallback_dir = Path(config["experiment"]["output_root"]) / "exp_03_optimized"
        ckpt_path = fallback_dir / "checkpoints" / "best.pth"
        if not ckpt_path.exists():
            print(f"Error: No checkpoint found at {ckpt_path} or original exp_dir.")
            return
        else:
            print(f"No checkpoint in {exp_dir}, falling back to {fallback_dir}")
            exp_dir = fallback_dir
            viz_dir = exp_dir / "visualizations"
            # Recreate dirs for fallback
            for subdir in subdirs:
                (viz_dir / subdir).mkdir(parents=True, exist_ok=True)
        
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    from albumentations.pytorch import ToTensorV2
    
    prep_cfg = config["preprocessing"]
    transform = A.Compose([
        A.Normalize(
            mean=prep_cfg.get("normalize_mean", [0.485, 0.456, 0.406]),
            std=prep_cfg.get("normalize_std", [0.229, 0.224, 0.225]),
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])
    
    split_file = Path(config["dataset"].get("split_file", "outputs/split_metadata.json"))
    with open(split_file, "r") as f:
        splits = json.load(f)
    test_ids = splits["ids"]["test"]

    dataset = IDRiDDataset(
        image_ids=test_ids,
        config=config,
        transform=transform,
        is_training=True,
    )
    
    class_names = config["dataset"]["class_names"]
    eval_cfg = config.get("evaluation", {})
    thresholds = eval_cfg.get("threshold", 0.5)
    if isinstance(thresholds, float):
        thresholds = {cls: thresholds for cls in class_names}
    
    use_tta = eval_cfg.get("use_tta", True)
    
    print(f"Starting visualization pipeline for {len(dataset)} test images...")
    
    all_dice_scores = []
    
    # Statistics for plots
    pred_area_dist = {cls: [] for cls in class_names}
    gt_area_dist = {cls: [] for cls in class_names}
    
    for i in tqdm(range(len(dataset))):
        img_tensor, mask_tensor, meta = dataset[i]
        img_id = meta["image_id"]
        
        # Expand dims for batch = 1
        img_input = img_tensor.unsqueeze(0).to(device)
        
        # Inference
        logits = get_sliding_window_predictions(model, img_input, patch_size=512, overlap=0.25, use_tta=use_tta)
        probs = torch.sigmoid(logits)[0].cpu().numpy()  # (4, H, W)
        gt_masks = mask_tensor.cpu().numpy()  # (4, H, W)
        
        # Thresholding
        pred_binary = np.zeros_like(probs)
        for c_idx, cls_name in enumerate(class_names):
            pred_binary[c_idx] = (probs[c_idx] >= thresholds.get(cls_name, 0.5)).astype(np.float32)
            
        # Denormalize image
        img_rgb = denormalize(img_tensor)
        
        # Calculate single image metrics
        img_dice_scores = []
        for c_idx in range(len(class_names)):
            intersection = (gt_masks[c_idx] * pred_binary[c_idx]).sum()
            dice = (2. * intersection + 1e-6) / (gt_masks[c_idx].sum() + pred_binary[c_idx].sum() + 1e-6)
            img_dice_scores.append(dice)
            
            # Save area distributions
            gt_area_dist[class_names[c_idx]].append(gt_masks[c_idx].sum())
            pred_area_dist[class_names[c_idx]].append(pred_binary[c_idx].sum())
            
        mean_dice = np.mean(img_dice_scores)
        all_dice_scores.append((img_id, mean_dice, img_rgb, gt_masks, pred_binary, probs, img_dice_scores))
        
        # --- 1. Per-Class Probability Heatmaps ---
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for c_idx, cls_name in enumerate(class_names):
            ax = axes[c_idx]
            im = ax.imshow(probs[c_idx], cmap='jet', vmin=0, vmax=1)
            ax.set_title(f"{cls_name} Heatmap")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(viz_dir / "heatmaps" / f"{img_id}_heatmaps.png", dpi=300)
        plt.close(fig)
        
        # --- 3. Segmentation Overlay Visualization ---
        overlay = img_rgb.copy().astype(np.float32)
        for c_idx, cls_name in enumerate(class_names):
            colour = CLASS_COLOURS.get(cls_name, (1,1,1,0.7))
            mask_bool = pred_binary[c_idx] > 0
            for ch in range(3):
                overlay[:, :, ch] = np.where(mask_bool, overlay[:, :, ch] * 0.4 + colour[ch] * 255 * 0.6, overlay[:, :, ch])
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(overlay.astype(np.uint8))
        ax.set_title(f"Multi-Class Overlay ({img_id})")
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(viz_dir / "overlays" / f"{img_id}_overlay.png", dpi=300)
        plt.close(fig)
        
        # --- 4. False Positive / False Negative Error Maps ---
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for c_idx, cls_name in enumerate(class_names):
            error_map = np.zeros((*probs.shape[1:], 3), dtype=np.uint8)
            tp = (gt_masks[c_idx] == 1) & (pred_binary[c_idx] == 1)
            fp = (gt_masks[c_idx] == 0) & (pred_binary[c_idx] == 1)
            fn = (gt_masks[c_idx] == 1) & (pred_binary[c_idx] == 0)
            
            error_map[tp] = [0, 255, 0]    # Green
            error_map[fp] = [255, 0, 0]    # Red
            error_map[fn] = [0, 0, 255]    # Blue
            
            axes[c_idx].imshow(error_map)
            axes[c_idx].set_title(f"{cls_name} Errors (G=TP, R=FP, B=FN)")
            axes[c_idx].axis('off')
        plt.tight_layout()
        plt.savefig(viz_dir / "error_maps" / f"{img_id}_errors.png", dpi=300)
        plt.close(fig)
        
        # --- 5. Binary Mask vs Probability Map Comparison ---
        fig, axes = plt.subplots(4, 2, figsize=(10, 20))
        for c_idx, cls_name in enumerate(class_names):
            axes[c_idx, 0].imshow(probs[c_idx], cmap='hot', vmin=0, vmax=1)
            axes[c_idx, 0].set_title(f"{cls_name} Probability")
            axes[c_idx, 0].axis('off')
            
            axes[c_idx, 1].imshow(pred_binary[c_idx], cmap='gray')
            axes[c_idx, 1].set_title(f"{cls_name} Binary (Thresh={thresholds.get(cls_name, 0.5)})")
            axes[c_idx, 1].axis('off')
        plt.tight_layout()
        plt.savefig(viz_dir / "probability_maps" / f"{img_id}_prob_vs_binary.png", dpi=300)
        plt.close(fig)

        # --- 8. Confidence Histogram ---
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for c_idx, cls_name in enumerate(class_names):
            p = probs[c_idx]
            gt = gt_masks[c_idx]
            tp_probs = p[(gt == 1) & (pred_binary[c_idx] == 1)]
            fp_probs = p[(gt == 0) & (pred_binary[c_idx] == 1)]
            
            if len(tp_probs) > 0:
                axes[c_idx].hist(tp_probs, bins=20, alpha=0.5, color='green', label='True Positive')
            if len(fp_probs) > 0:
                axes[c_idx].hist(fp_probs, bins=20, alpha=0.5, color='red', label='False Positive')
            axes[c_idx].set_title(f"{cls_name} Confidence")
            axes[c_idx].legend()
        plt.tight_layout()
        plt.savefig(viz_dir / "reports" / f"{img_id}_confidence_hist.png", dpi=300)
        plt.close(fig)
        
        # --- 9. Confusion Visualization ---
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for c_idx, cls_name in enumerate(class_names):
            conf_map = np.zeros((*probs.shape[1:], 3), dtype=np.uint8)
            tp = (gt_masks[c_idx] == 1) & (pred_binary[c_idx] == 1)
            fp = (gt_masks[c_idx] == 0) & (pred_binary[c_idx] == 1)
            fn = (gt_masks[c_idx] == 1) & (pred_binary[c_idx] == 0)
            tn = (gt_masks[c_idx] == 0) & (pred_binary[c_idx] == 0)
            
            conf_map[tp] = [0, 255, 0]      # Green
            conf_map[fp] = [255, 0, 0]      # Red
            conf_map[fn] = [255, 165, 0]    # Orange
            conf_map[tn] = [0, 0, 0]        # Black
            
            axes[c_idx].imshow(conf_map)
            axes[c_idx].set_title(f"{cls_name} Confusion")
            axes[c_idx].axis('off')
        plt.tight_layout()
        plt.savefig(viz_dir / "confusion_maps" / f"{img_id}_confusion.png", dpi=300)
        plt.close(fig)

    # --- 6. Best vs Worst Prediction Samples ---
    all_dice_scores.sort(key=lambda x: x[1])  # Sort by mean dice
    worst_samples = all_dice_scores[:5]
    best_samples = all_dice_scores[-5:][::-1]
    
    def save_samples(samples, folder_name):
        for rank, (img_id, mean_dice, img_rgb, gt_masks, pred_binary, probs, img_dice_scores) in enumerate(samples):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img_rgb)
            axes[0].set_title(f"Input: {img_id}")
            axes[0].axis('off')
            
            # GT Overlay
            gt_overlay = img_rgb.copy().astype(np.float32)
            for c_idx, cls_name in enumerate(class_names):
                colour = CLASS_COLOURS.get(cls_name, (1,1,1,0.7))
                for ch in range(3):
                    gt_overlay[:, :, ch] = np.where(gt_masks[c_idx] > 0, gt_overlay[:, :, ch] * 0.4 + colour[ch] * 255 * 0.6, gt_overlay[:, :, ch])
            axes[1].imshow(gt_overlay.astype(np.uint8))
            axes[1].set_title("Ground Truth")
            axes[1].axis('off')
            
            # Pred Overlay
            pred_overlay = img_rgb.copy().astype(np.float32)
            for c_idx, cls_name in enumerate(class_names):
                colour = CLASS_COLOURS.get(cls_name, (1,1,1,0.7))
                for ch in range(3):
                    pred_overlay[:, :, ch] = np.where(pred_binary[c_idx] > 0, pred_overlay[:, :, ch] * 0.4 + colour[ch] * 255 * 0.6, pred_overlay[:, :, ch])
            axes[2].imshow(pred_overlay.astype(np.uint8))
            
            title_parts = [f"Prediction (Mean Dice: {mean_dice:.4f})"]
            for c_idx, cls_name in enumerate(class_names):
                title_parts.append(f"{cls_name}: {img_dice_scores[c_idx]:.4f}")
            axes[2].set_title("\n".join(title_parts))
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(viz_dir / folder_name / f"rank{rank+1}_{img_id}.png", dpi=300)
            plt.close(fig)
            
    save_samples(worst_samples, "worst_samples")
    save_samples(best_samples, "best_samples")
    
    # Save standard prediction vs gt for all samples
    save_samples(all_dice_scores, "reports")
    
    # --- 7. Lesion Distribution Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(len(class_names))
    width = 0.35
    
    gt_means = [np.mean(gt_area_dist[cls]) for cls in class_names]
    pred_means = [np.mean(pred_area_dist[cls]) for cls in class_names]
    
    axes[0].bar(x - width/2, gt_means, width, label='Ground Truth')
    axes[0].bar(x + width/2, pred_means, width, label='Predicted')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names)
    axes[0].set_title("Mean Lesion Area per Image (Pixels)")
    axes[0].legend()
    axes[0].set_yscale('log')
    
    gt_counts = [np.sum(np.array(gt_area_dist[cls]) > 0) for cls in class_names]
    pred_counts = [np.sum(np.array(pred_area_dist[cls]) > 0) for cls in class_names]
    
    axes[1].bar(x - width/2, gt_counts, width, label='Ground Truth')
    axes[1].bar(x + width/2, pred_counts, width, label='Predicted')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names)
    axes[1].set_title("Number of Images Containing Lesion")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(viz_dir / "reports" / "lesion_distribution.png", dpi=300)
    plt.close(fig)
    
    # --- 10. Full Evaluation Report Figures ---
    # Copy training curves from history
    history_path = exp_dir / "training_history.json"
    if history_path.exists():
        from utils.visualization import Visualizer
        viz = Visualizer(output_dir=str(viz_dir / "reports"), class_names=class_names)
        with open(history_path, "r") as f:
            history = json.load(f)
        viz.plot_training_curves(history, filename="training_curves.png")
        
    # Also load test_results.json and plot metric table
    test_results_path = exp_dir / "test_results.json"
    if test_results_path.exists():
        with open(test_results_path, "r") as f:
            test_results = json.load(f)
        viz.plot_metric_table(test_results, filename="metric_table.png")
        
    print(f"Visualizations successfully generated and saved to {viz_dir}")

if __name__ == "__main__":
    main()
