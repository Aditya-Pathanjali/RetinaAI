import os
import sys
import csv
import json
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import load_config, get_device
from preprocessing.enhancer import RetinalEnhancer
from models.attention_unet import build_model

def get_lesion_count(probs_np, class_names, thresholds, min_areas):
    counts = {}
    for c, cls_name in enumerate(class_names):
        thresh = thresholds.get(cls_name, 0.5)
        min_a = min_areas.get(cls_name, 0)
        mask = (probs_np[c] >= thresh).astype(np.uint8) * 255
        if mask.sum() == 0:
            counts[cls_name] = 0
            continue
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        count = 0
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_a:
                count += 1
        counts[cls_name] = count
    return counts

def main():
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    config = load_config(str(config_path))
    device = get_device()
    ds = config["aptos_dataset"]
    root = Path(ds["root"])
    output_images_dir = root / "enhanced_images"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    enhancer = RetinalEnhancer(config["preprocessing"])
    seg_model = build_model(config)
    seg_ckpt_path = Path(config["experiment"]["output_root"]) / "exp_03_optimized" / "checkpoints" / "best.pth"
    if not seg_ckpt_path.exists():
        seg_ckpt_path = Path(config["experiment"]["output_root"]) / "exp_04" / "checkpoints" / "best.pth"
    ckpt = torch.load(seg_ckpt_path, map_location=device)
    seg_model.load_state_dict(ckpt["model_state_dict"])
    seg_model = seg_model.to(device)
    seg_model.eval()
    class_names = config["dataset"]["class_names"]
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
    splits = [
        ("train", ds["train_csv"], ds["train_images"]),
        ("val", ds["val_csv"], ds["val_images"]),
        ("test", ds["test_csv"], ds["test_images"])
    ]
    counts_json_path = root / "lesion_counts.json"
    all_counts = {}
    if counts_json_path.exists():
        try:
            with open(counts_json_path, "r") as f:
                all_counts = json.load(f)
            print(f"Loaded {len(all_counts)} existing precomputed counts from {counts_json_path}")
        except Exception:
            all_counts = {}
    save_counter = 0
    for split_name, csv_file, img_dir in splits:
        csv_path = root / csv_file
        images_src = root / img_dir
        split_output_dir = output_images_dir / img_dir
        split_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processing split: {split_name}")
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        for row in tqdm(rows):
            img_id = row["id_code"]
            out_img_path = split_output_dir / f"{img_id}.jpg"
            out_img_path_png = split_output_dir / f"{img_id}.png"
            if img_id in all_counts and (out_img_path.exists() or out_img_path_png.exists()):
                continue
            if out_img_path.exists():
                enhanced_img = cv2.imread(str(out_img_path))
                if enhanced_img is None:
                    continue
            elif out_img_path_png.exists():
                enhanced_img = cv2.imread(str(out_img_path_png))
                if enhanced_img is None:
                    continue
            else:
                img_path = images_src / f"{img_id}{ds.get('image_ext', '.png')}"
                if not img_path.exists():
                    continue
                img = cv2.imread(str(img_path))
                enhanced_img = enhancer.process(img)
                cv2.imwrite(str(out_img_path), enhanced_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            img_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(device)
            with torch.no_grad():
                logits = seg_model(img_tensor)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            counts = get_lesion_count(probs, class_names, thresholds, min_areas)
            all_counts[img_id] = counts
            save_counter += 1
            if save_counter % 100 == 0:
                with open(counts_json_path, "w") as f:
                    json.dump(all_counts, f, indent=4)
    with open(counts_json_path, "w") as f:
        json.dump(all_counts, f, indent=4)
    print(f"Precomputation complete! Counts saved to {counts_json_path}")

if __name__ == "__main__":
    main()
