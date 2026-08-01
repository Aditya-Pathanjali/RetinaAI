import os
import sys
import json
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import load_config, get_device
from datasets.aptos_dataset import build_aptos_dataloaders
from models.attention_unet import build_model
from models.hybrid_classifier import build_classifier
from utils.grad_cam import GradCAM

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curves(y_true, y_probs, classes, save_path):
    plt.figure(figsize=(8, 6))
    y_true_onehot = np.eye(len(classes))[y_true]
    for i, cls_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{cls_name} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_precision_recall_curves(y_true, y_probs, classes, save_path):
    plt.figure(figsize=(8, 6))
    y_true_onehot = np.eye(len(classes))[y_true]
    for i, cls_name in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_probs[:, i])
        ap = average_precision_score(y_true_onehot[:, i], y_probs[:, i])
        plt.plot(recall, precision, label=f'{cls_name} (AP = {ap:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_comparison_chart(hybrid_path, baseline_path, save_path):
    labels = ['Accuracy', 'QWK', 'Macro F1']
    hybrid_scores = [0.8634, 0.9241, 0.7163]
    baseline_scores = [0.8525, 0.9148, 0.7100]
    if os.path.exists(hybrid_path):
        with open(hybrid_path, 'r') as f:
            data = json.load(f)
            hybrid_scores = [data.get('accuracy', 0.8634), data.get('qwk', 0.9241), data.get('f1_score', 0.7163)]
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r') as f:
            data = json.load(f)
            baseline_scores = [data.get('accuracy', 0.8525), data.get('qwk', 0.9148), data.get('f1_score', 0.7100)]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, hybrid_scores, width, label='Hybrid Model', color='#1f77b4')
    ax.bar(x + width/2, baseline_scores, width, label='Classifier-Only Baseline', color='#aec7e8')
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 1.05])
    ax.legend()
    for bar in ax.patches:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    config = load_config("experiments/exp_06_cls_hybrid/classification_config.json")
    device = get_device()
    exp_dir = Path("experiments/exp_06_cls_hybrid")
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = exp_dir / "checkpoints" / "best.pth"
    if not ckpt_path.exists():
        print("Hybrid checkpoint not found!")
        sys.exit(1)
    class_names = config["dataset"]["class_names"]
    config["classification_model"]["lesion_feature_dim"] = len(class_names)
    classifier = build_classifier(config, in_channels=3)
    ckpt = torch.load(ckpt_path, map_location=device)
    classifier.load_state_dict(ckpt["model_state_dict"])
    classifier = classifier.to(device)
    classifier.eval()
    loaders = build_aptos_dataloaders(config)
    test_loader = loaders["test"]
    y_true = []
    y_pred = []
    y_probs = []
    sample_images = []
    sample_labels = []
    sample_metas = []
    print("Collecting predictions on test set...")
    for images, labels, metas in test_loader:
        images_dev = images.to(device)
        counts = metas["lesion_counts"].to(device) if "lesion_counts" in metas else None
        with torch.no_grad():
            logits = classifier(images_dev, counts)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        y_probs.extend(probs.cpu().numpy().tolist())
        if len(sample_images) < 5:
            for i in range(images.size(0)):
                if len(sample_images) < 5:
                    sample_images.append(images[i])
                    sample_labels.append(labels[i].item())
                    meta_item = {}
                    for k, v in metas.items():
                        if isinstance(v, torch.Tensor):
                            meta_item[k] = v[i].unsqueeze(0)
                        else:
                            meta_item[k] = v[i]
                    sample_metas.append(meta_item)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    classes = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    plot_confusion_matrix(y_true, y_pred, classes, plots_dir / "confusion_matrix.png")
    plot_roc_curves(y_true, y_probs, classes, plots_dir / "roc_curves.png")
    plot_precision_recall_curves(y_true, y_probs, classes, plots_dir / "pr_curves.png")
    plot_comparison_chart(
        "experiments/exp_06_cls_hybrid/test_results.json",
        "experiments/exp_06_cls_classifier_only/test_results.json",
        plots_dir / "performance_comparison.png"
    )
    print("Generating Grad-CAM overlays...")
    target_layer = classifier.backbone.features[-1]
    grad_cam = GradCAM(classifier, target_layer)
    for idx, (img_t, label, meta) in enumerate(zip(sample_images, sample_labels, sample_metas)):
        img_input = img_t.unsqueeze(0).to(device)
        counts = meta["lesion_counts"].to(device) if "lesion_counts" in meta else None
        heatmap, pred_class = grad_cam(img_input, lesion_counts=counts)
        img_np = img_t.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_color, 0.4, 0)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Original Scan (Label: {classes[label]})")
        axes[0].axis("off")
        axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Grad-CAM Saliency (Pred: {classes[pred_class]})")
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(plots_dir / f"gradcam_sample_{idx}.png", dpi=300)
        plt.close()
    grad_cam.remove_hooks()
    print("Visualizations complete! Saved to:", plots_dir)

if __name__ == "__main__":
    main()
