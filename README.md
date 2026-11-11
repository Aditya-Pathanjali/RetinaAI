# RetinaAI — Automated and Explainable Diabetic Retinopathy Screening System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Azure%20Container%20Apps-0078D4?logo=microsoftazure&logoColor=white)](https://retinaai.proudcoast-af0aedc3.uksouth.azurecontainerapps.io/)
[![CI/CD](https://github.com/Aditya-Pathanjali/RetinaAI/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Aditya-Pathanjali/RetinaAI/actions/workflows/ci-cd.yml)

**Try it live: https://retinaai.proudcoast-af0aedc3.uksouth.azurecontainerapps.io/** · [API docs](https://retinaai.proudcoast-af0aedc3.uksouth.azurecontainerapps.io/docs)

> The demo scales to zero when idle, so the first load after a quiet period can take ~30 seconds. Research prototype — **not a medical device**; do not upload patient-identifiable images.

**Results (APTOS 2019 test set):** AUC 0.96, 92.4% sensitivity, 92.0% specificity for referable DR; controlled ablation improved referable-DR recall from 85.2% to 92.4%.

RetinaAI is an automated and explainable Diabetic Retinopathy (DR) screening system designed to identify retinal lesions, classify disease severity, and provide visual evidence for its predictions. 

The system uses an Attention U-Net with a ResNet-34 encoder for pixel-level segmentation of four key microvascular lesions: microaneurysms, haemorrhages, hard exudates, and soft exudates. This is followed by an EfficientNet-B4 classifier that combines global retinal image features with explicit, quantified lesion count biomarkers for 5-stage ICDR disease severity grading. Gradient-weighted Class Activation Mapping (Grad-CAM) is integrated to provide visual explanations of classification decisions. 

The system was developed and evaluated using the IDRiD, APTOS 2019, and MESSIDOR-2 datasets across experiments covering lesion segmentation, severity classification, lesion-count integration, and external zero-shot dataset evaluation. The complete workflow is packaged into an interactive desktop application integrating segmentation, classification, explainability heatmaps, and clinical PDF report generation.

---

## Datasets

- **[IDRiD (Indian Diabetic Retinopathy Image Dataset)](https://idrid.grand-challenge.org/):** Used for Stage 1 lesion segmentation training and testing ($81$ images with expert pixel-level binary annotations for MA, HE, EX, and SE).
- **[APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection):** Used for Stage 2 hybrid classification model development and validation ($3,662$ fundus images graded across 5 clinical severity levels).
- **[MESSIDOR-2](https://www.adcis.net/en/third-party/messidor2/):** Used for independent, external zero-shot cross-dataset generalization evaluation ($1,748$ images across $874$ patient examinations).

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA support (recommended) or CPU

### 1. Clone Repository & Create Environment
```bash
git clone https://github.com/Aditya-Pathanjali/RetinaAI.git
cd RetinaAI

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux / macOS
```

### 2. Install Dependencies
```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project requirements
pip install -r requirements.txt
```

---

## Execution Guide

### 1. Launch Desktop Application
```bash
python desktop_app.py
```

### 2. Stage 1: Train & Evaluate Lesion Segmentation
```bash
# Train Attention U-Net on IDRiD
python main.py --mode train --config configs/config.yaml

# Evaluate segmentation metrics (Dice, IoU, Recall) on IDRiD test set
python main.py --mode eval --config configs/config.yaml
```

### 3. Stage 2: Train & Evaluate Hybrid DR Classifier
```bash
# Train the high-recall hybrid classification model on APTOS 2019
python train_classifier.py --config configs/config_cls_high_recall.yaml --variant hybrid

# Evaluate classification performance (Accuracy, QWK, Macro-F1, Sensitivity)
python evaluate_classifier.py --config configs/config_cls_high_recall.yaml --variant hybrid
```

### 4. Zero-Shot Cross-Dataset Validation (MESSIDOR-2)
```bash
# Run external validation across all 1,748 unseen MESSIDOR-2 images
python evaluate_messidor.py --config configs/config_messidor.yaml
```

### 5. Generate Diagnostic Visualizations
```bash
# Generate ROC curves, confusion matrices, and Grad-CAM figure plots
python visualization_classifier.py
```

---

## Deployment

The inference service (FastAPI REST API + Gradio UI) runs as a CPU container on **Azure Container Apps** (Consumption plan, scale-to-zero, max 1 replica).

- **Endpoints:** `/` demo UI · `POST /api/v1/predict` · `GET /api/v1/health` · `/docs` (OpenAPI)
- **Weights:** hosted in a [Hugging Face model repo](https://huggingface.co/AdityaPathanjalii/retinaai-weights) and baked into the image at build time.
- **CI/CD:** GitHub Actions runs the unit tests, builds the image and pushes it to GHCR on every push to `master`.
- **Azure deploy:** `.\deploy\azure_deploy.ps1` (first deploy) or `.\deploy\azure_deploy.ps1 -Update` (roll out a new image).

### Run the service locally with Docker
```bash
docker run -p 7860:7860 ghcr.io/aditya-pathanjali/retinaai:latest
# open http://localhost:7860
```

### Call the API
```bash
curl -F file=@examples/IDRiD_028.jpg https://retinaai.proudcoast-af0aedc3.uksouth.azurecontainerapps.io/api/v1/predict
```

## Limitations

- Research prototype trained on public datasets (IDRiD, APTOS 2019); not clinically validated and not a medical device.
- Input must be a colour fundus photograph; the service rejects non-fundus images and low-resolution inputs (<256 px).
- Sample images in `examples/` are from IDRiD (CC BY 4.0).

---

## Technologies Used

- **Deep Learning Framework:** PyTorch, torchvision, segmentation-models-pytorch
- **Image Processing & Augmentation:** OpenCV, Albumentations
- **Graphical User Interface:** PyQt6 (desktop), Gradio (web)
- **Serving & DevOps:** FastAPI, Docker, GitHub Actions (CI/CD), GHCR, Azure Container Apps
- **Diagnostic Report Generation:** ReportLab
- **Evaluation & Visualisation:** Scikit-learn, NumPy, Pandas, Matplotlib, Seaborn

---

## Academic Context

Developed as part of an MSc Dissertation project at the University of Leeds (2025–2026).
