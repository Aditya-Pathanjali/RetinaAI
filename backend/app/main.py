import os
import sys
import io
import time
import base64
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.inference import RetinaAIInferenceEngine

app = FastAPI(
    title="RetinaAI Diagnostic API",
    description="Production-grade REST API for two-stage Diabetic Retinopathy lesion segmentation and severity grading.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Inference Engine Singleton
inference_engine: Optional[RetinaAIInferenceEngine] = None


@app.on_event("startup")
def startup_event():
    global inference_engine
    try:
        inference_engine = RetinaAIInferenceEngine()
    except Exception as e:
        print(f"Warning: Inference Engine startup error: {e}")


@app.get("/api/v1/health", summary="Get API System Health & GPU Status")
def health_check():
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    return {
        "status": "healthy",
        "service": "RetinaAI Diagnostic API",
        "device": device_name,
        "cuda_available": cuda_avail,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models_loaded": inference_engine is not None,
    }


def encode_image_base64(image_rgb: np.ndarray) -> str:
    """Encodes numpy RGB image into base64 string for web UI rendering."""
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".png", image_bgr)
    return base64.b64encode(buffer).decode("utf-8")


@app.post("/api/v1/predict", summary="Run Two-Stage DR Diagnostic Inference on Uploaded Fundus Image")
async def predict_dr_severity(file: UploadFile = File(...)):
    global inference_engine
    if inference_engine is None:
        inference_engine = RetinaAIInferenceEngine()

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File uploaded is not a valid image format.",
        )

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image_bgr is None:
            raise ValueError("Unable to decode uploaded image data.")

        # Run inference
        result = inference_engine.process_image(image_bgr)

        # Convert images to Base64 for REST response
        raw_b64 = encode_image_base64(result["raw_rgb"])
        clahe_b64 = encode_image_base64(result["enhanced_rgb"])
        overlay_rgb = cv2.cvtColor(result["overlay_bgr"], cv2.COLOR_BGR2RGB)
        overlay_b64 = encode_image_base64(overlay_rgb)

        return JSONResponse(
            content={
                "filename": file.filename,
                "predicted_grade": result["predicted_grade"],
                "grade_title": result["grade_title"],
                "referable_status": result["referable_status"],
                "is_referable": result["is_referable"],
                "recommendation": result["recommendation"],
                "confidence_pct": round(result["confidence_pct"], 2),
                "probabilities": {k: round(v, 4) for k, v in result["probabilities"].items()},
                "lesion_counts": result["lesion_counts"],
                "images": {
                    "raw_base64": f"data:image/png;base64,{raw_b64}",
                    "clahe_base64": f"data:image/png;base64,{clahe_b64}",
                    "overlay_base64": f"data:image/png;base64,{overlay_b64}",
                },
            }
        )

    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
