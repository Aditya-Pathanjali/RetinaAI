"""
RetinaAI — Hugging Face Space entrypoint.

Serves, from one process:
  /               Gradio demo UI
  /api/v1/predict REST inference endpoint (FastAPI)
  /api/v1/health  health check
  /docs           OpenAPI docs

Weights are downloaded from a Hugging Face model repo (override with env vars).
Run locally:  uvicorn app:app --port 7860
"""
import os
from pathlib import Path

import cv2
import gradio as gr
import torch
from huggingface_hub import hf_hub_download

import backend.app.main as api
from backend.app.inference import RetinaAIInferenceEngine

ROOT = Path(__file__).resolve().parent
WEIGHTS_REPO = os.getenv("WEIGHTS_REPO", "AdityaPathanjalii/retinaai-weights")
LOCAL_DIR = ROOT / "deploy"  # local fallback (used when the .pth files exist here)


def _weights(filename: str) -> str:
    local = LOCAL_DIR / filename
    if local.exists() and not os.getenv("FORCE_HUB"):
        return str(local)
    return hf_hub_download(WEIGHTS_REPO, filename)


engine = RetinaAIInferenceEngine(
    config_path=str(ROOT / "configs" / "config_deploy.yaml"),
    seg_ckpt=_weights("seg_attention_unet.pth"),
    cls_ckpt=_weights("cls_hybrid_effnetb4.pth"),
    device=torch.device("cpu"),
)
api.inference_engine = engine  # share the loaded models with the REST routes

DISCLAIMER = (
    "**Research prototype — not a medical device.** Built as an MSc dissertation "
    "(University of Leeds). Do not upload patient-identifiable images; uploads are "
    "processed in memory and not stored."
)


def run(img_rgb):
    if img_rgb is None:
        raise gr.Error("Please upload a retinal fundus image.")
    try:
        r = engine.process_image(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    except ValueError as e:  # input validation (non-fundus, low-res, etc.)
        raise gr.Error(str(e))
    overlay = cv2.cvtColor(r["overlay_bgr"], cv2.COLOR_BGR2RGB)
    lesions = "\n".join(f"- {k}: **{v}**" for k, v in r["lesion_counts"].items())
    summary = (
        f"### {r['grade_title']}\n"
        f"**{r['referable_status']}** · confidence {r['confidence_pct']:.1f}% · "
        f"image quality: {r['quality_status']} · {r['inference_time_sec']}s\n\n"
        f"{r['recommendation']}\n\n**Detected lesions**\n{lesions}"
    )
    return r["probabilities"], overlay, r["heatmap_rgb"], summary


with gr.Blocks(title="RetinaAI — Explainable DR Screening") as demo:
    gr.Markdown("# RetinaAI — Explainable Diabetic Retinopathy Screening")
    gr.Markdown(
        "Two-stage pipeline: Attention U-Net lesion segmentation (MA, HE, EX, SE) → "
        "EfficientNet-B4 hybrid severity classifier (ICDR grades 0–4) using lesion-count biomarkers.  \n"
        "[GitHub](https://github.com/Aditya-Pathanjali/RetinaAI) · "
        "[API docs](/docs)"
    )
    gr.Markdown(DISCLAIMER)
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="Fundus image")
            btn = gr.Button("Analyse", variant="primary")
        with gr.Column():
            summary = gr.Markdown()
            label = gr.Label(label="Grade probabilities", num_top_classes=5)
    with gr.Row():
        overlay = gr.Image(label="Lesion overlay")
        heat = gr.Image(label="Segmentation heatmap")
    gr.Examples(
        examples=[str(p) for p in sorted((ROOT / "examples").glob("*.jpg"))],
        inputs=inp,
        label="Sample images (IDRiD, CC BY 4.0)",
    )
    btn.click(run, inp, [label, overlay, heat, summary])

app = gr.mount_gradio_app(api.app, demo, path="/")
