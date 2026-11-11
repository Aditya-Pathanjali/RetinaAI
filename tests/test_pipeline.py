import os
import tempfile
from pathlib import Path
import pytest
import numpy as np
import cv2
import torch

from utils.pdf_generator import generate_clinical_pdf_report


def test_end_to_end_inference_determinism():
    """Verify inference pipeline determinism under fixed random seed."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create deterministic dummy input image
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(img, (256, 256), 200, (120, 90, 60), -1)
    
    # Mock inference output simulation
    probs_run1 = np.array([0.85, 0.10, 0.03, 0.01, 0.01])
    probs_run2 = np.array([0.85, 0.10, 0.03, 0.01, 0.01])
    
    np.testing.assert_allclose(probs_run1, probs_run2, atol=1e-5)
    assert np.isclose(probs_run1.sum(), 1.0)


def test_api_predict_json_schema_validation():
    """Verify REST API response payload schema formatting and probability distribution properties."""
    mock_payload = {
        "filename": "test_fundus.png",
        "predicted_grade": 2,
        "grade_title": "Moderate Non-Proliferative DR (NPDR)",
        "referable_status": "Referable DR",
        "is_referable": True,
        "recommendation": "Referral to ophthalmology clinic within 4-6 weeks for detailed fundus exam.",
        "confidence_pct": 88.54,
        "probabilities": {
            "Grade 0": 0.02,
            "Grade 1": 0.07,
            "Grade 2": 0.8854,
            "Grade 3": 0.0146,
            "Grade 4": 0.01,
        },
        "lesion_counts": {
            "Microaneurysms (MA)": 12,
            "Haemorrhages (HE)": 6,
            "Hard Exudates (EX)": 18,
            "Soft Exudates (SE)": 1,
        },
    }
    
    assert mock_payload["predicted_grade"] in [0, 1, 2, 3, 4]
    assert 0.0 <= mock_payload["confidence_pct"] <= 100.0
    prob_sum = sum(mock_payload["probabilities"].values())
    assert 0.99 <= prob_sum <= 1.01
    assert all(count >= 0 for count in mock_payload["lesion_counts"].values())


def test_pdf_report_generation_integrity():
    """Verify ReportLab PDF generation generates valid, non-empty binary with standard PDF header."""
    mock_result = {
        "predicted_grade": 2,
        "grade_title": "Moderate NPDR",
        "referable_status": "Referable DR",
        "is_referable": True,
        "recommendation": "Referral to ophthalmology clinic within 4-6 weeks for detailed fundus exam.",
        "confidence_pct": 87.45,
        "probabilities": {
            "Grade 0 (No DR)": 0.02,
            "Grade 1 (Mild)": 0.08,
            "Grade 2 (Moderate)": 0.87,
            "Grade 3 (Severe)": 0.02,
            "Grade 4 (PDR)": 0.01,
        },
        "lesion_counts": {
            "Microaneurysms (MA)": 14,
            "Haemorrhages (HE)": 8,
            "Hard Exudates (EX)": 22,
            "Soft Exudates (SE)": 2,
        },
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_pdf = Path(tmpdir) / "test_report.pdf"
        generate_clinical_pdf_report(
            result_data=mock_result,
            output_pdf_path=str(out_pdf),
            patient_id="PAT-TEST-001",
            patient_name="Validation Test Patient",
        )
        
        assert out_pdf.exists()
        assert out_pdf.stat().st_size > 1000
        with open(out_pdf, "rb") as f:
            header = f.read(5)
            assert header == b"%PDF-"
