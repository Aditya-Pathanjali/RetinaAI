
import sys
import os
from pathlib import Path

# Ensure project root is in Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

from utils.helpers import load_config, ensure_dir, save_json
from utils.logger import get_logger
from utils.visualization import Visualizer, CLASS_COLOURS
from datasets.idrid_dataset import IDRiDDataset
from preprocessing.enhancer import RetinalEnhancer


def main():
    # --- Setup ---
    config = load_config(str(PROJECT_ROOT / "configs" / "config.yaml"))
    logger = get_logger("Exploration", level="INFO")
    output_dir = ensure_dir(PROJECT_ROOT / "outputs" / "exploration")
    viz = Visualizer(output_dir=str(output_dir), class_names=config["dataset"]["class_names"])

    ds = config["dataset"]
    root = Path(ds["root"])
    image_dir = root / ds["train_images"]
    test_image_dir = root / ds["test_images"]
    mask_root = root / ds["train_masks"]
    ext = ds.get("image_ext", ".jpg")
    mask_ext = ds.get("mask_ext", ".tif")
    class_names = ds["class_names"]

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("  RetinaAI — IDRiD Dataset Exploration Report")
    report_lines.append("=" * 70)

    #image counts
    logger.info("Step 1: Counting images...")
    train_images = sorted(image_dir.glob(f"*{ext}"))
    test_images = sorted(test_image_dir.glob(f"*{ext}"))
    all_train_ids = sorted([f.stem for f in train_images])

    report_lines.append(f"\n1. IMAGE COUNTS")
    report_lines.append(f"   Training images:  {len(train_images)}")
    report_lines.append(f"   Testing images:   {len(test_images)}")
    report_lines.append(f"   Total:            {len(train_images) + len(test_images)}")
    report_lines.append(f"   Training IDs:     {all_train_ids[0]} to {all_train_ids[-1]}")
    report_lines.append(f"   Testing IDs:      IDRiD_55 to IDRiD_81")

    logger.info(f"  Training: {len(train_images)}, Testing: {len(test_images)}")

    #resolution analysis
    logger.info("Step 2: Analysing resolutions...")
    resolutions = []
    file_sizes_kb = []
    brightness_values = []
    contrast_values = []

    for img_path in train_images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        resolutions.append((w, h))
        file_sizes_kb.append(img_path.stat().st_size / 1024)

        # Image quality metrics
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness_values.append(float(np.mean(grey)))
        contrast_values.append(float(np.std(grey)))

    unique_res = list(set(resolutions))
    res_counts = {str(r): resolutions.count(r) for r in unique_res}

    report_lines.append(f"\n2. RESOLUTION ANALYSIS")
    report_lines.append(f"   Unique resolutions: {len(unique_res)}")
    for r, c in res_counts.items():
        report_lines.append(f"     {r}: {c} images")
    report_lines.append(f"   File sizes: {min(file_sizes_kb):.0f}KB – {max(file_sizes_kb):.0f}KB "
                        f"(mean: {np.mean(file_sizes_kb):.0f}KB)")

    resolution_data = {
        "unique_resolutions": [{"width": r[0], "height": r[1], "count": resolutions.count(r)} for r in unique_res],
        "file_sizes_kb": {"min": min(file_sizes_kb), "max": max(file_sizes_kb), "mean": float(np.mean(file_sizes_kb))},
    }
    save_json(resolution_data, str(output_dir / "resolution_analysis.json"))

    #image quality variability
    logger.info("Step 3: Analysing image quality variability...")

    report_lines.append(f"\n3. IMAGE QUALITY VARIABILITY")
    report_lines.append(f"   Brightness (mean grey): {np.mean(brightness_values):.1f} "
                        f"± {np.std(brightness_values):.1f}")
    report_lines.append(f"   Contrast (std grey):    {np.mean(contrast_values):.1f} "
                        f"± {np.std(contrast_values):.1f}")
    report_lines.append(f"   Brightness range:       {min(brightness_values):.1f} – {max(brightness_values):.1f}")
    report_lines.append(f"   Contrast range:         {min(contrast_values):.1f} – {max(contrast_values):.1f}")

    # Plot image quality distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(brightness_values, bins=15, color="#3498db", edgecolor="black", alpha=0.8)
    axes[0].set_xlabel("Mean Brightness (0-255)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Image Brightness Distribution")
    axes[0].axvline(np.mean(brightness_values), color="red", linestyle="--", label=f"Mean: {np.mean(brightness_values):.1f}")
    axes[0].legend()

    axes[1].hist(contrast_values, bins=15, color="#e74c3c", edgecolor="black", alpha=0.8)
    axes[1].set_xlabel("Contrast (Std Dev)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Image Contrast Distribution")
    axes[1].axvline(np.mean(contrast_values), color="blue", linestyle="--", label=f"Mean: {np.mean(contrast_values):.1f}")
    axes[1].legend()

    plt.suptitle("IDRiD — Image Quality Variability", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "image_quality_analysis.png", dpi=300)
    plt.close()

   #mask distribution analysis
    logger.info("Step 4: Analysing lesion mask distributions...")

    dataset = IDRiDDataset(
        image_ids=all_train_ids,
        config=config,
        transform=None,
        enhancer=None,
        is_training=True,
    )
    stats = dataset.compute_class_statistics()
    save_json(stats, str(output_dir / "class_statistics.json"))

    report_lines.append(f"\n4. LESION MASK DISTRIBUTION")
    report_lines.append(f"   {'Class':<20} {'Present':<10} {'Absent':<10} {'Mean %':<12} {'Max %':<12}")
    report_lines.append(f"   {'-' * 64}")
    for cls_name, s in stats.items():
        report_lines.append(
            f"   {cls_name:<20} {s['present_count']:<10} {s['absent_count']:<10} "
            f"{s['mean_ratio']*100:<12.4f} {s['max_ratio']*100:<12.4f}"
        )

    # Mask count per lesion type
    mask_counts = {}
    for cls_name in class_names:
        subfolder = ds["lesion_classes"][cls_name]
        mask_dir = mask_root / subfolder
        if mask_dir.exists():
            mask_files = list(mask_dir.glob(f"*{mask_ext}"))
            mask_counts[cls_name] = len(mask_files)
        else:
            mask_counts[cls_name] = 0

    report_lines.append(f"\n   Mask file counts:")
    for cls_name, count in mask_counts.items():
        report_lines.append(f"     {cls_name}: {count} mask files")

    #per-image lesion pixel analysis
    logger.info("Step 5: Per-image lesion pixel analysis...")

    per_image_data = defaultdict(dict)
    all_ratios = {cls: [] for cls in class_names}

    for img_id in all_train_ids:
        img_path = image_dir / f"{img_id}{ext}"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        total_px = h * w

        for cls_name in class_names:
            mask = dataset._load_single_mask(img_id, cls_name, h, w)
            lesion_px = int((mask > 0).sum())
            ratio = lesion_px / total_px
            per_image_data[img_id][cls_name] = {
                "pixels": lesion_px,
                "ratio": ratio,
                "has_lesion": lesion_px > 0,
            }
            all_ratios[cls_name].append(ratio)

    # Histogram of lesion pixel ratios per class
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, cls_name in enumerate(class_names):
        ax = axes[i // 2, i % 2]
        ratios = [r * 100 for r in all_ratios[cls_name] if r > 0]
        colour = CLASS_COLOURS.get(cls_name, (0.5, 0.5, 0.5, 0.7))[:3]

        if ratios:
            ax.hist(ratios, bins=20, color=colour, edgecolor="black", alpha=0.8)
            ax.axvline(np.mean(ratios), color="red", linestyle="--",
                       label=f"Mean: {np.mean(ratios):.4f}%")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No positive masks", ha="center", va="center",
                    transform=ax.transAxes)

        ax.set_xlabel("Lesion Pixel Percentage (%)")
        ax.set_ylabel("Number of Images")
        ax.set_title(f"{cls_name} — Pixel Coverage Distribution")

    plt.suptitle("IDRiD — Per-Image Lesion Coverage", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "mask_pixel_histogram.png", dpi=300)
    plt.close()

    #class distribution plot
    logger.info("Step 6: Generating class distribution plot...")
    viz.plot_class_distribution(stats)

    #sample visualizations with masks
    logger.info("Step 7: Generating sample visualizations...")
    sample_ids = all_train_ids[:6]
    for img_id in sample_ids:
        img_path = image_dir / f"{img_id}{ext}"
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        masks = {}
        for cls_name in class_names:
            mask = dataset._load_single_mask(img_id, cls_name, h, w)
            masks[cls_name] = mask

        viz.plot_sample_with_masks(
            image_rgb, masks, image_id=img_id,
            filename=f"sample_{img_id}.png",
        )

    #preprocessing pipeline comparison
    logger.info("Step 8: Preprocessing pipeline comparison...")
    enhancer = RetinalEnhancer(config["preprocessing"])
    for img_id in sample_ids[:3]:
        img_path = image_dir / f"{img_id}{ext}"
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        stages = enhancer.get_intermediate_results(image)
        viz.plot_preprocessing_stages(
            stages, image_id=img_id,
            filename=f"preprocessing_{img_id}.png",
        )

    #lesion co-occurrence
    logger.info("Step 9: Lesion co-occurrence analysis...")
    report_lines.append(f"\n5. LESION CO-OCCURRENCE")

    co_occurrence = np.zeros((len(class_names), len(class_names)), dtype=int)
    for img_id in all_train_ids:
        img_path = image_dir / f"{img_id}{ext}"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        has_class = []
        for cls_name in class_names:
            mask = dataset._load_single_mask(img_id, cls_name, h, w)
            has_class.append((mask > 0).any())

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if has_class[i] and has_class[j]:
                    co_occurrence[i, j] += 1

    report_lines.append(f"   Co-occurrence matrix (count of images having both):")
    report_lines.append(f"   {'':>8}" + "".join(f"{cls:>8}" for cls in class_names))
    for i, cls in enumerate(class_names):
        row = f"   {cls:>8}" + "".join(f"{co_occurrence[i, j]:>8}" for j in range(len(class_names)))
        report_lines.append(row)

    # Plot co-occurrence heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(co_occurrence, cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(co_occurrence[i, j]), ha="center", va="center",
                    color="black" if co_occurrence[i, j] < co_occurrence.max() * 0.7 else "white",
                    fontweight="bold")
    plt.colorbar(im, ax=ax, label="Number of Images")
    ax.set_title("Lesion Co-occurrence Matrix", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "co_occurrence.png", dpi=300)
    plt.close()

    #clinical context
    report_lines.append(f"\n{'=' * 70}")
    report_lines.append(f"  OBSERVATIONS & CLINICAL CONTEXT")
    report_lines.append(f"{'=' * 70}")
    
    #saving report
    report_text = "\n".join(report_lines)
    report_path = output_dir / "exploration_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  Exploration complete!")
    logger.info(f"  All outputs saved to: {output_dir}")
    logger.info(f"  Report: {report_path}")
    logger.info(f"{'=' * 60}")

    # Print summary to console
    print(report_text)


if __name__ == "__main__":
    main()
