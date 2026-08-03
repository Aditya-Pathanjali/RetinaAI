import sys
import os
import io
import time
import base64
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import streamlit as st
from PIL import Image

from backend.app.inference import RetinaAIInferenceEngine

# Configure Page Theme & Layout
st.set_page_config(
    page_title="RetinaAI — Clinical Diagnostic Portal",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Slate Dark Clinical Aesthetics
CUSTOM_CSS = """
<style>
    /* Dark Clinical Background */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .header-title {
        font-size: 28px;
        font-weight: 800;
        color: #38BDF8;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .header-subtitle {
        font-size: 14px;
        color: #94A3B8;
        margin-top: 4px;
    }

    /* Metric Cards */
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-label {
        font-size: 12px;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Alert Badges */
    .badge-referable {
        background-color: rgba(239, 68, 68, 0.2);
        border: 1px solid #EF4444;
        color: #FCA5A5;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 16px;
        display: inline-block;
    }
    .badge-non-referable {
        background-color: rgba(16, 185, 129, 0.2);
        border: 1px solid #10B981;
        color: #6EE7B7;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 16px;
        display: inline-block;
    }

    /* Lesion Table Styling */
    .lesion-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 12px;
    }
    .lesion-table th {
        background-color: #1E293B;
        color: #38BDF8;
        padding: 10px;
        text-align: left;
        border-bottom: 2px solid #334155;
    }
    .lesion-table td {
        padding: 10px;
        border-bottom: 1px solid #334155;
        color: #E2E8F0;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_cached_inference_engine():
    """Caches inference engine initialization across Streamlit sessions."""
    return RetinaAIInferenceEngine()


def main():
    # Header Section
    st.markdown(
        """
        <div class="header-banner">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1 class="header-title">👁️ RetinaAI Clinical Diagnostic Portal</h1>
                    <div class="header-subtitle">Two-Stage Hybrid Deep Learning Pipeline for Biomarker Segmentation & Severity Grading</div>
                </div>
                <div style="text-align: right;">
                    <span style="background-color: #0284C7; color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;">Stage 1 Attention UNet + Stage 2 Hybrid Classifier</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar System Configuration & Status
    with st.sidebar:
        st.header("⚙️ Clinical System Status")
        cuda_avail = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Engine"

        st.success(f"**Device Acceleration**: {device_name}")
        st.info("**Segmentation Model**: Attention U-Net (DDR / IDRiD)")
        st.info("**Classifier Model**: Hybrid ResNet50 + 4D Lesion Count Head")

        st.markdown("---")
        st.header("🎨 Heatmap Color Key")
        st.markdown("🟡 **Yellow**: Microaneurysms (MA)")
        st.markdown("🔴 **Red**: Hemorrhages (HE)")
        st.markdown("🟢 **Green**: Hard Exudates (EX)")
        st.markdown("🔵 **Blue**: Soft Exudates (SE)")

        st.markdown("---")
        st.caption("RetinaAI MSc Dissertation Framework © 2026")

    # Load Inference Engine
    try:
        engine = get_cached_inference_engine()
    except Exception as e:
        st.error(f"Failed to load AI Models: {e}")
        return

    # File Upload Section
    st.subheader("📁 Upload Retinal Fundus Image")
    uploaded_file = st.file_uploader(
        "Select a retinal fundus photograph (.png, .jpg, .jpeg, .tif)",
        type=["png", "jpg", "jpeg", "tif"],
        help="Upload a retinal fundus photograph for automated DR lesion detection and grading.",
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image_bgr is None:
            st.error("Uploaded file is corrupted or not a valid image format.")
            return

        with st.spinner("Executing Two-Stage Hybrid AI Inference Pipeline..."):
            start_time = time.time()
            try:
                result = engine.process_image(image_bgr)
                inference_time = time.time() - start_time
            except Exception as ex:
                st.error(f"Diagnostic Error: {ex}")
                return

        # Main Diagnostic Layout (2 Columns)
        col_img, col_results = st.columns([1.1, 1])

        with col_img:
            st.subheader("🖼️ Interactive Vision & Mask Views")

            view_mode = st.radio(
                "Select Visual Mode:",
                ["4-Color Lesion Overlay Heatmap", "CLAHE Green-Channel Enhanced", "Raw Input Image"],
                horizontal=True,
            )

            if view_mode == "4-Color Lesion Overlay Heatmap":
                display_img = cv2.cvtColor(result["overlay_bgr"], cv2.COLOR_BGR2RGB)
                st.image(display_img, use_container_width=True, caption="4-Color Lesion Instance Overlay")
            elif view_mode == "CLAHE Green-Channel Enhanced":
                st.image(result["enhanced_rgb"], use_container_width=True, caption="CLAHE Green-Channel Preprocessed")
            else:
                st.image(result["raw_rgb"], use_container_width=True, caption="Raw Retinal Fundus Image")

        with col_results:
            st.subheader("📊 Clinical Diagnostic Summary")

            # Referable Status Badge
            if result["is_referable"]:
                st.markdown(
                    f'<div class="badge-referable">🔴 REFERABLE DR DETECTED ({result["grade_title"]})</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="badge-non-referable">🟢 NON-REFERABLE (Grade 0 — Healthy)</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(f"**Clinical Recommendation**: {result['recommendation']}")

            st.markdown("<br>", unsafe_allow_html=True)

            # Metric Counters Grid
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value">Grade {result["predicted_grade"]}</div><div class="metric-label">Predicted Severity</div></div>',
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value">{result["confidence_pct"]:.1f}%</div><div class="metric-label">Confidence</div></div>',
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value">{inference_time:.2f}s</div><div class="metric-label">Inference Time</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Lesion Biomarker Table
            st.markdown("### 🔬 Quantified Lesion Biomarker Counts")
            counts = result["lesion_counts"]

            table_html = f"""
            <table class="lesion-table">
                <thead>
                    <tr>
                        <th>Biomarker Lesion Class</th>
                        <th>Abbreviation</th>
                        <th>Detected Instance Count</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td>Microaneurysms</td><td>MA</td><td><b>{counts['Microaneurysms (MA)']}</b></td></tr>
                    <tr><td>Hemorrhages</td><td>HE</td><td><b>{counts['Hemorrhages (HE)']}</b></td></tr>
                    <tr><td>Hard Exudates</td><td>EX</td><td><b>{counts['Hard Exudates (EX)']}</b></td></tr>
                    <tr><td>Soft Exudates (Cotton Wool)</td><td>SE</td><td><b>{counts['Soft Exudates (SE)']}</b></td></tr>
                </tbody>
            </table>
            """
            st.markdown(table_html, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability Bar Distribution
            st.markdown("### 📈 Grade Probability Distribution")
            for grade_name, prob in result["probabilities"].items():
                st.progress(prob, text=f"{grade_name}: {prob*100:.1f}%")

    else:
        st.info("👆 Please upload a retinal fundus photograph to begin diagnostic inference.")


if __name__ == "__main__":
    main()
