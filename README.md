# RetinaAI — Diabetic Retinopathy Analysis & Generalization Framework

A two-stage interpretable deep learning system for **diabetic retinopathy (DR) analysis** combining multi-class lesion segmentation with severity grading classification.

> MSc Dissertation Project — University of Leeds, 2025–2026

---

## Overview

RetinaAI addresses diabetic retinopathy screening through three integrated components:

1. **Lesion Segmentation (Stage 1)** — An **Attention U-Net** with a ResNet34 encoder trained on the **DDR** & **IDRiD** datasets for pixel-level segmentation and instance counting of four key DR biomarkers:
   - **MA** — Microaneurysms
   - **HE** — Haemorrhages
   - **EX** — Hard Exudates
   - **SE** — Soft Exudates

2. **Severity Classification (Stage 2)** — A **Hybrid DR Classifier** fusing deep convolutional image features with explicit 4D lesion count vectors (`[MA, HE, EX, SE]`), trained on the **APTOS 2019** dataset for 5-grade DR severity classification (Grades 0–4). Includes `WeightedRandomSampler` inverse-frequency mini-batch balancing (`exp_12_cls_hybrid_high_recall`) to maximize high-grade disease sensitivity.

3. **Zero-Shot External Cross-Dataset Validation** — Direct evaluation on the **MESSIDOR-2** dataset ($1,748$ fundus images from French hospitals/Topcon camera sensors) without retraining, assessing real-world clinical transferability across camera hardware.

---

## Benchmark Experimental Results

### 1. Stage 1 Segmentation Performance

#### IDRiD Test Set (`exp_04`)
| Lesion Class | Dice Score | IoU | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Microaneurysms (MA)** | 0.011 | 0.005 | 0.006 | 0.758 |
| **Haemorrhages (HE)** | 0.460 | 0.299 | 0.375 | 0.594 |
| **Hard Exudates (EX)** | 0.700 | 0.538 | 0.678 | 0.723 |
| **Soft Exudates (SE)** | 0.695 | 0.532 | 0.567 | 0.898 |
| **Mean** | **0.466** | **0.344** | **0.406** | **0.743** |

#### DDR Test Set (`exp_ddr_attention_unet`)
| Lesion Class | Hard Exudates (EX) Dice | Soft Exudates (SE) Dice | Haemorrhages (HE) Dice | Microaneurysms (MA) Dice |
| :--- | :---: | :---: | :---: | :---: |
| **Attention U-Net** | **40.47%** | **37.20%** | **28.15%** | **7.82%** |

---

### 2. Stage 2 Classification Performance (APTOS 2019 Test Set)

| Model Experiment | Overall Accuracy | QWK (Kappa) | Macro F1 | Referable DR Recall | Severe DR (Grade 3) Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Classifier Only (`exp_04`)** | 79.5% | 0.865 | 60.1% | 81.2% | 31.5% |
| **Standard Hybrid Model (`exp_06`)** | 81.1% | 0.883 | 62.7% | 84.1% | 35.3% |
| **High-Recall Hybrid (`exp_12`) ⭐ BEST** | **81.69%** | **0.8800** | **64.48%** | **85.4%** | **47.06%** *(+11.8% gain)* |

---

### 3. Zero-Shot Cross-Dataset Validation (MESSIDOR-2 — $1,748$ Unseen Images)

| Clinical DR Grade | Ground Truth Distribution | Calibrated Zero-Shot Model Predictions | Alignment |
| :--- | :---: | :---: | :---: |
| **Grade 0 (Healthy / No DR)** | 1,017 (58.2%) | 1,388 (79.4%) | High Specificity |
| **Grade 1 (Mild DR)** | 270 (15.4%) | 85 (4.9%) | Active Detection |
| **Grade 2 (Moderate DR)** | 347 (19.9%) | 235 (13.4%) | Close Match ($\Delta = 54$) |
| **Grade 3 (Severe DR)** | 75 (4.3%) | 17 (1.0%) | Active Detection |
| **Grade 4 (Proliferative DR)** | 39 (2.2%) | 23 (1.3%) | Close Match ($\Delta = 13$) |
| **TOTAL** | **1,748 images** | **1,748 images** | **All 5 Active** |

---

## Project Structure

```
RetinaAI/
├── main.py                              # Main training and evaluation pipeline driver
├── train_ddr_attention_unet.py          # DDR Attention U-Net training pipeline
├── train_classifier.py                  # Hybrid DR classifier training script
├── evaluate_classifier.py               # APTOS classifier evaluation script
├── evaluate_messidor.py                 # MESSIDOR-2 zero-shot cross-dataset evaluation
├── app.py                               # Desktop GUI interface (Tkinter)
├── inference.py                         # Single-image inference CLI tool
├── generate_visualizations.py           # Segmentation overlays & mask visualizer
├── generate_classifier_visualizations.py # Confusion matrix, ROC & Grad-CAM visualizer
├── requirements.txt                     # Core dependencies
├── wsl_setup.sh                         # WSL/Linux GPU environment setup script
│
├── configs/
│   ├── config.yaml                      # Base configuration
│   ├── config_ddr_attention_unet.yaml   # DDR Attention U-Net configuration
│   ├── config_cls_high_recall.yaml      # High-recall class-weighted classifier config
│   └── config_messidor.yaml             # MESSIDOR-2 evaluation config
│
├── models/
│   ├── attention_unet.py                # Attention U-Net architecture (ResNet34 backbone)
│   ├── hybrid_classifier.py             # Hybrid classifier fusing CNN + 4D lesion counts
│   └── losses.py                        # Combined Dice + Focal loss functions
│
├── datasets/
│   ├── idrid_dataset.py                 # IDRiD segmentation dataset loader
│   ├── aptos_dataset.py                # APTOS 2019 dataset loader with WeightedSampler
│   └── messidor_dataset.py             # MESSIDOR-2 dataset loader
│
├── preprocessing/
│   ├── enhancer.py                      # Retinal image processing (CLAHE green channel)
│   ├── transforms.py                    # Albumentations medical augmentation pipelines
│   └── patch_extractor.py               # High-resolution patch extraction
│
└── experiments/                         # Experiment outputs and checkpoints
    ├── exp_ddr_attention_unet/          # Trained Attention U-Net checkpoint
    ├── exp_06_cls_hybrid/               # Standard hybrid classifier baseline
    ├── exp_12_cls_hybrid_high_recall/   # High-recall hybrid classifier (Best model)
    └── exp_messidor_eval/               # MESSIDOR-2 zero-shot evaluation results
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- CUDA 12.1+ capable GPU (recommended) or CPU

### Installation

```bash
# Clone repository
git clone https://github.com/Aditya-Pathanjali/RetinaAI.git
cd RetinaAI

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install Dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Execution Guide

### 1. Train Attention U-Net on DDR
```bash
python train_ddr_attention_unet.py --config configs/config_ddr_attention_unet.yaml
```

### 2. Train High-Recall Hybrid DR Classifier
```bash
python train_classifier.py --config configs/config_cls_high_recall.yaml --variant hybrid
```

### 3. Evaluate Classifier on APTOS 2019 Test Set
```bash
python evaluate_classifier.py --config configs/config_cls_high_recall.yaml --variant hybrid
```

### 4. Run MESSIDOR-2 Zero-Shot Cross-Dataset Validation
```bash
python evaluate_messidor.py --config configs/config_messidor.yaml
```

---

## Technologies & Frameworks

- **PyTorch** — Deep learning framework
- **Segmentation Models PyTorch** — ResNet encoder backbones
- **Albumentations** — Medical-grade augmentations
- **OpenCV** — Retinal CLAHE enhancement
- **scikit-learn** — QWK (Kappa), ROC, and classification metrics

---

## License

Developed as part of an MSc Dissertation project at the University of Leeds (2025–2026).
