# RetinaAI — Diabetic Retinopathy Analysis

A two-stage deep learning system for **diabetic retinopathy (DR) analysis** combining multi-class lesion segmentation with severity grading classification.

> MSc Dissertation Project — University of Leeds, 2025–2026

---

## Overview

RetinaAI addresses diabetic retinopathy screening through two complementary pipelines:

1. **Lesion Segmentation** — An Attention U-Net with a ResNet34 encoder trained on the [IDRiD dataset](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) for pixel-level segmentation of four lesion types:
   - **MA** — Microaneurysms
   - **HE** — Haemorrhages
   - **EX** — Hard Exudates
   - **SE** — Soft Exudates

2. **Severity Classification** — A hybrid classifier combining a pre-trained segmentation backbone with lesion-count features, trained on the [APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection) dataset for DR severity grading (0–4).

3. **Desktop Application** — A standalone Tkinter-based GUI for clinical inference, packaged with PyInstaller.

## Results

### Segmentation (IDRiD — Experiment `exp_04`)

| Lesion | Dice | IoU | Precision | Recall |
|--------|------|-----|-----------|--------|
| MA | 0.011 | 0.005 | 0.006 | 0.758 |
| HE | 0.460 | 0.299 | 0.375 | 0.594 |
| EX | 0.700 | 0.538 | 0.678 | 0.723 |
| SE | 0.695 | 0.532 | 0.567 | 0.898 |
| **Mean** | **0.466** | **0.344** | **0.406** | **0.743** |

### Classification (APTOS 2019 — Hybrid Model)

| Metric | Score |
|--------|-------|
| Accuracy | 84.2% |
| F1 Score | 0.645 |
| Quadratic Weighted Kappa | **0.906** |

## Project Structure

```
RetinaAI/
├── main.py                              # Main training/evaluation pipeline
├── app.py                               # Desktop GUI application (Tkinter)
├── train.py                             # Convenience wrapper → main.py --mode train
├── evaluate.py                          # Convenience wrapper → main.py --mode eval
├── inference.py                         # Single-image inference script
├── train_classifier.py                  # Classification model training
├── evaluate_classifier.py              # Classification model evaluation
├── generate_visualizations.py           # Segmentation visualization generator
├── generate_classifier_visualizations.py # Classification visualization generator
├── build_exe.py                         # PyInstaller packaging script
├── requirements.txt                     # Python dependencies
├── wsl_setup.sh                         # WSL/Linux GPU setup script
│
├── configs/
│   ├── config.yaml                      # Main configuration
│   └── config_focal.yaml               # Focal loss variant config
│
├── models/
│   ├── attention_unet.py               # Attention U-Net architecture
│   ├── hybrid_classifier.py            # Hybrid DR severity classifier
│   └── losses.py                       # Loss functions (Dice, Focal, combined)
│
├── datasets/
│   ├── idrid_dataset.py                # IDRiD segmentation dataset loader
│   └── aptos_dataset.py               # APTOS 2019 classification dataset loader
│
├── preprocessing/
│   ├── enhancer.py                     # Retinal image enhancement (CLAHE, etc.)
│   ├── transforms.py                   # Augmentation pipelines (Albumentations)
│   └── precompute_pipeline.py          # Offline preprocessing pipeline
│
├── training/
│   └── trainer.py                      # Training loop with AMP, early stopping
│
├── evaluation/
│   └── metrics.py                      # Segmentation metrics (Dice, IoU, etc.)
│
├── utils/
│   ├── helpers.py                      # Config loading, seeding, checkpoints
│   ├── logger.py                       # Logging utilities
│   ├── visualization.py               # Plotting and overlay generation
│   ├── grad_cam.py                     # Grad-CAM attention visualization
│   └── lesion_counter.py              # Lesion instance counting
│
├── notebooks/
│   └── 01_dataset_exploration.py       # Dataset analysis and statistics
│
├── scratch/                            # Ad-hoc test and debug scripts
│
├── experiments/                        # Training experiment outputs
│   ├── exp_01_baseline/               # Baseline experiment
│   ├── exp_02_fixed/                  # Fixed loss experiment
│   ├── exp_02_focal/                  # Focal loss experiment
│   ├── exp_03_optimized/              # Optimized augmentation experiment
│   ├── exp_04/                        # Final segmentation experiment
│   ├── exp_04_cls_classifier_only/    # Classification ablation: classifier only
│   ├── exp_04_cls_hybrid/             # Classification: hybrid model (best)
│   └── exp_04_cls_mask_input/         # Classification ablation: mask input
│
└── outputs/
    ├── inference/                      # Inference output images
    └── split_metadata.json            # Dataset split information
```

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU

### Installation

```bash
# Clone the repository
git clone https://github.com/Aditya-Pathanjali/RetinaAI.git
cd RetinaAI

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**WSL/Linux GPU users** can use the automated setup script:
```bash
bash wsl_setup.sh
```

### Dataset Setup

1. **IDRiD** — Download from [IEEE DataPort](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) and update `dataset.root` in `configs/config.yaml`
2. **APTOS 2019** — Download from [Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection) and update `aptos_dataset.root` in `configs/config.yaml`

## Usage

### Training

```bash
# Train segmentation model
python main.py --mode train --config configs/config.yaml

# Or use the convenience wrapper
python train.py

# Train classification model
python train_classifier.py --config configs/config.yaml --variant hybrid
```

### Evaluation

```bash
# Evaluate segmentation model
python main.py --mode eval --config configs/config.yaml

# Evaluate classification model
python evaluate_classifier.py --config configs/config.yaml --variant hybrid
```

### Single-Image Inference

```bash
python inference.py --image /path/to/retinal_image.jpg --config configs/config.yaml
```

### Visualization Generation

```bash
# Segmentation visualizations
python generate_visualizations.py

# Classification visualizations (confusion matrix, ROC, Grad-CAM)
python generate_classifier_visualizations.py
```

### Desktop Application

```bash
python app.py
```

To build a standalone executable:
```bash
python build_exe.py
```

## Architecture

### Attention U-Net (Segmentation)

- **Encoder**: ResNet34 (ImageNet pre-trained)
- **Decoder**: 4-level decoder with attention gates
- **Loss**: Dice-Focal combined loss (0.6 Dice + 0.4 Focal)
- **Training**: AdamW optimizer, cosine annealing with warm restarts, mixed-precision (AMP)

### Hybrid Classifier (Severity Grading)

- Frozen segmentation backbone extracts lesion probability maps
- Lesion instance counts provide complementary features
- Combined features fed to a classification head
- Evaluated with Quadratic Weighted Kappa (QWK)

## Technologies

- **PyTorch** — Deep learning framework
- **Segmentation Models PyTorch** — Encoder backbones
- **Albumentations** — Medical-imaging-safe augmentations
- **OpenCV** — Image processing and enhancement
- **Tkinter** — Desktop GUI framework

## License

This project was developed as part of an MSc dissertation. Please contact the author for usage permissions.
