import argparse
import sys
import os
from pathlib import Path
import cv2
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import load_config, get_device
from utils.logger import get_logger
from preprocessing.enhancer import RetinalEnhancer
from preprocessing.transforms import get_val_transforms
from models.attention_unet import build_model
from utils.visualization import Visualizer

def parse_args():
    parser = argparse.ArgumentParser(description="RetinaAI — Run Inference on a Single Image")
    parser.add_argument("--image", type=str, required=True, help="Path to the input retinal image.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file.")
    parser.add_argument("--output", type=str, default="outputs/inference", help="Directory to save inference results.")
    parser.add_argument("--device", type=str, default=None, help="Force device ('cpu' or 'cuda').")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Setup output dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = get_logger("Inference")
    logger.info("=" * 60)
    logger.info("  RetinaAI — Inference Pipeline")
    logger.info("=" * 60)
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    
    # Checkpoint path
    exp_dir = Path(config["experiment"]["output_root"]) / config["experiment"]["name"]
    best_path = exp_dir / "checkpoints" / "best.pth"
    if not best_path.exists():
        logger.error(f"No trained model found at {best_path}. Please train first.")
        sys.exit(1)
        
    # Build & load model
    logger.info("Building Attention U-Net model...")
    model = build_model(config)
    from utils.helpers import load_checkpoint
    load_checkpoint(str(best_path), model, device=device)
    model.to(device)
    model.eval()
    logger.info(f"Loaded weights from {best_path}")
    
    # Preprocessing
    enhancer = RetinalEnhancer(config["preprocessing"])
    transform = get_val_transforms(config)
    
    # Load Image
    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Input image not found: {image_path}")
        sys.exit(1)
        
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        sys.exit(1)
        
    orig_h, orig_w = image.shape[:2]
    logger.info(f"Loaded image: {image_path.name} (Resolution: {orig_w}x{orig_h})")
    
    # Process
    enhanced_image = enhancer.process(image)
    image_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
    transformed = transform(image=image_rgb)
    input_tensor = transformed["image"].unsqueeze(0).to(device)
    
    # Forward pass
    logger.info("Running inference...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)
    
    probs_np = probs.cpu().numpy()[0]  # (C, H, W)
    class_names = config["dataset"]["class_names"]
    
    config_thresholds = config["evaluation"].get("threshold", 0.5)
    if isinstance(config_thresholds, dict):
        thresholds = {cls: float(config_thresholds.get(cls, 0.5)) for cls in class_names}
    else:
        thresholds = {cls: float(config_thresholds) for cls in class_names}
        
    preds = np.zeros_like(probs_np)
    for c, cls_name in enumerate(class_names):
        thresh = thresholds.get(cls_name, 0.5)
        preds[c] = (probs_np[c] >= thresh).astype(np.float32)
    
    # Visualizer for overlays
    viz = Visualizer(output_dir=str(output_dir), class_names=class_names)
    
    # Save raw masks and generate individual overlays
    for c, cls_name in enumerate(class_names):
        mask_c = preds[c]
        if mask_c.sum() > 0:
            logger.info(f"  Detected {cls_name} lesions ({mask_c.sum()} pixels)")
        
        # Save binary mask
        mask_out = (mask_c * 255).astype(np.uint8)
        mask_out_resized = cv2.resize(mask_out, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(output_dir / f"{image_path.stem}_pred_{cls_name}.png"), mask_out_resized)
        
    # Generate combined overlay directly from viz using dummy batch tensor
    images_tensor = transformed["image"].unsqueeze(0)
    masks_tensor = torch.zeros((1, len(class_names), images_tensor.shape[2], images_tensor.shape[3])) # dummy
    viz.plot_predictions(
        images=images_tensor,
        masks=masks_tensor,
        preds=logits.cpu(),
        image_ids=[image_path.stem],
        num_samples=1,
        filename=f"{image_path.stem}_overlay.png"
    )
    
    logger.info(f"Inference complete! Results saved to {output_dir}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
