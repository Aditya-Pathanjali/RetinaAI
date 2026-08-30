import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


def generate_clinical_pdf_report(
    result_data: Dict[str, Any],
    output_pdf_path: str,
    patient_id: str = "PAT-2026-8891",
    patient_name: str = "Anonymous Patient",
    eye_laterality: str = "OD (Right Eye)",
    clinician_name: str = "Dr. Anonymous",
) -> str:
    output_path = Path(output_pdf_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    COLOR_NAVY = colors.HexColor("#1E3A8A")       # Primary Header Navy
    COLOR_PRIMARY = colors.HexColor("#2563EB")    # Muted Blue
    COLOR_TEXT_DARK = colors.HexColor("#111827")  # Charcoal Text
    COLOR_TEXT_MUTED = colors.HexColor("#4B5563") # Muted Gray
    COLOR_BORDER = colors.HexColor("#E5E7EB")     # Subtle Border
    COLOR_BG_LIGHT = colors.HexColor("#F8FAFC")   # Light Background
    COLOR_ALERT_BG = colors.HexColor("#EFF6FF")   # Alert Box Background
    COLOR_ALERT_BORDER = colors.HexColor("#BFDBFE")

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=COLOR_NAVY,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=COLOR_PRIMARY,
    )
    h2_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=COLOR_NAVY,
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=COLOR_TEXT_DARK,
    )
    bold_body_style = ParagraphStyle(
        "BoldBodyCustom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13,
        textColor=COLOR_TEXT_DARK,
    )
    small_muted_style = ParagraphStyle(
        "SmallMuted",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=COLOR_TEXT_MUTED,
    )

    story = []

    report_id = f"REP-{int(time.time())}"
    exam_date = time.strftime("%Y-%m-%d %H:%M")

    header_data = [
        [
            Paragraph("<b>RETINAAI CLINICAL REPORT</b>", title_style),
            Paragraph(f"<b>Report ID:</b> {report_id}<br/><b>Exam Date:</b> {exam_date}", ParagraphStyle("RightH1", parent=small_muted_style, alignment=2)),
        ],
        [
            Paragraph("AI-Assisted Retinal Screening Assessment", subtitle_style),
            Paragraph(f"<b>Clinician:</b> {clinician_name}", ParagraphStyle("RightH2", parent=small_muted_style, alignment=2)),
        ],
    ]
    header_table = Table(header_data, colWidths=[4.8 * inch, 2.7 * inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=COLOR_NAVY, spaceBefore=0, spaceAfter=8))

    # Patient & Examination Summary Table
    patient_summary_data = [
        [
            Paragraph(f"<b>Patient ID:</b> {patient_id}", body_style),
            Paragraph(f"<b>Patient Name:</b> {patient_name}", body_style),
            Paragraph(f"<b>Eye Examined:</b> <b>{eye_laterality}</b>", bold_body_style),
        ]
    ]
    patient_table = Table(patient_summary_data, colWidths=[2.5 * inch, 2.7 * inch, 2.3 * inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), COLOR_BG_LIGHT),
        ('BOX', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 10))

    is_ref = result_data.get("is_referable", False)
    pred_grade = result_data.get("predicted_grade", 0)
    grade_title = result_data.get("grade_title", "Grade 0 — Healthy / No DR")
    confidence_pct = result_data.get("confidence_pct", 0.0)

    # Clean clinical finding title
    finding_clean = grade_title.split("—")[-1].strip() if "—" in grade_title else grade_title
    severity_text = f"Grade {pred_grade} — {finding_clean}"
    
    referral_rec = "Ophthalmology evaluation recommended" if is_ref else "Routine annual screening protocol"
    status_fg = colors.HexColor("#DC2626") if is_ref else colors.HexColor("#16A34A")

    assessment_box_data = [
        [
            Paragraph("<b>1. AI-ASSISTED RETINAL ASSESSMENT</b>", ParagraphStyle("AssHead", parent=h2_style, textColor=COLOR_NAVY, fontSize=11)),
            Paragraph(f"<font color='{status_fg.hexval()}'><b>{'REFERABLE' if is_ref else 'NON-REFERABLE'}</b></font>", ParagraphStyle("AssStat", parent=bold_body_style, alignment=2)),
        ],
        [
            Paragraph(f"<b>Primary Finding:</b> {finding_clean}", bold_body_style),
            Paragraph(f"<b>Assessment Confidence:</b> {confidence_pct:.1f}%", body_style),
        ],
        [
            Paragraph(f"<b>Severity:</b> {severity_text}", body_style),
            Paragraph(f"<b>Referral Recommendation:</b> {referral_rec}", bold_body_style),
        ]
    ]
    assessment_table = Table(assessment_box_data, colWidths=[4.2 * inch, 3.3 * inch])
    assessment_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), COLOR_ALERT_BG),
        ('BOX', (0, 0), (-1, -1), 1, COLOR_ALERT_BORDER),
        ('SPAN', (0, 0), (0, 0)),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(assessment_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("2. KEY FINDINGS", h2_style))
    counts = result_data.get("lesion_counts", {})
    ma = counts.get("Microaneurysms (MA)", 0)
    he = counts.get("Hemorrhages (HE)", 0)
    ex = counts.get("Hard Exudates (EX)", 0)
    se = counts.get("Soft Exudates (SE)", 0)

    summary_bullets = []
    if pred_grade == 0:
        summary_bullets.append("The retinal image shows normal retinal architecture with no microvascular lesions detected.")
    elif pred_grade == 1:
        summary_bullets.append("The retinal image shows focal microvascular changes consistent with mild non-proliferative diabetic retinopathy.")
    elif pred_grade == 2:
        summary_bullets.append("The retinal image shows findings consistent with moderate non-proliferative diabetic retinopathy.")
    elif pred_grade == 3:
        summary_bullets.append("The retinal image shows severe intraretinal microvascular abnormalities across multiple quadrants.")
    else:
        summary_bullets.append("The retinal image shows high-risk visual abnormalities consistent with proliferative diabetic retinopathy.")

    detected_list = []
    if ma > 0:
        detected_list.append("• Microaneurysms")
    if he > 0:
        detected_list.append("• Intraretinal hemorrhages")
    if ex > 0:
        detected_list.append("• Hard lipid exudates")
    if se > 0:
        detected_list.append("• Soft exudates / cotton-wool spots")

    if not detected_list:
        detected_list.append("• No overt microvascular lesion clusters detected.")

    findings_text = "<br/>".join([f"• {b}" for b in summary_bullets] + ["<br/><b>Detected Retinal Abnormalities:</b>"] + detected_list)
    story.append(Paragraph(findings_text, body_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("3. QUANTIFIED RETINAL FINDINGS", h2_style))
    quant_rows = [
        [Paragraph("<b>Retinal Finding</b>", bold_body_style), Paragraph("<b>Detected Count</b>", bold_body_style), Paragraph("<b>Clinical Significance</b>", bold_body_style)],
        [Paragraph("Microaneurysms", body_style), Paragraph(str(ma), body_style), Paragraph("Low" if ma < 5 else "Moderate", body_style)],
        [Paragraph("Intraretinal Hemorrhages", body_style), Paragraph(str(he), body_style), Paragraph("Low" if he == 0 else "High", body_style)],
        [Paragraph("Hard Exudates", body_style), Paragraph(str(ex), body_style), Paragraph("Low" if ex == 0 else ("Moderate" if ex < 15 else "High"), body_style)],
        [Paragraph("Soft Exudates (Cotton Wool Spots)", body_style), Paragraph(str(se), body_style), Paragraph("Low" if se == 0 else "High", body_style)],
    ]
    quant_table = Table(quant_rows, colWidths=[3.2 * inch, 1.8 * inch, 2.5 * inch])
    quant_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_BG_LIGHT),
        ('GRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(quant_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("4. WHAT THIS MEANS", h2_style))
    if pred_grade == 0:
        meaning_text = "The analyzed retinal image contains no visible microvascular signs of diabetic retinal disease. Routine monitoring is recommended."
    else:
        meaning_text = f"The analyzed retinal image contains structural abnormalities associated with diabetic retinal disease. The findings are classified as <b>{finding_clean}</b> and warrant review by an eye-care professional."
    story.append(Paragraph(meaning_text, body_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("5. RECOMMENDED NEXT STEP", h2_style))
    action_plan = result_data.get("recommendation", "Schedule routine 12-month re-screening.")
    next_step_data = [[Paragraph(f"<b>Recommended Action:</b> {action_plan}", body_style)]]
    next_step_table = Table(next_step_data, colWidths=[7.5 * inch])
    next_step_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FEF3C7")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#FCD34D")),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(next_step_table)
    story.append(Spacer(1, 10))

    qual_status = result_data.get("quality_status", "Acceptable")
    if qual_status.lower() in ["good", "acceptable"]:
        qual_note = "Image quality is sufficient for diagnostic assessment."
    else:
        qual_note = "Image quality may limit the reliability of this assessment. Repeat imaging or clinical examination may be recommended."

    qual_conf_data = [
        [
            Paragraph("<b>6. IMAGE QUALITY</b>", h2_style),
            Paragraph("<b>7. EXAMINED EYE</b>", h2_style),
            Paragraph("<b>8. AI ASSESSMENT CONFIDENCE</b>", h2_style),
        ],
        [
            Paragraph(f"<b>Status:</b> {qual_status}<br/><font color='#6B7280'>{qual_note}</font>", body_style),
            Paragraph(f"<b>Eye:</b> {eye_laterality}<br/><font color='#6B7280'>Standard Retinal Field</font>", body_style),
            Paragraph(f"<b>Confidence:</b> {confidence_pct:.1f}%<br/><font color='#6B7280'>Classification certainty score.</font>", body_style),
        ]
    ]
    qual_conf_table = Table(qual_conf_data, colWidths=[2.5 * inch, 2.4 * inch, 2.6 * inch])
    qual_conf_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('PADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(qual_conf_table)
    story.append(Spacer(1, 10))

    story.append(HRFlowable(width="100%", thickness=0.5, color=COLOR_BORDER, spaceBefore=4, spaceAfter=6))
    disclaimer_text = (
        "<b>CLINICAL REVIEW DISCLAIMER:</b> This report provides an AI-assisted assessment of the submitted retinal photograph. "
        "It is intended to support clinical decision-making and does not replace examination, diagnosis, or treatment planning "
        "by a qualified ophthalmologist or optometrist. Final diagnosis must be confirmed by a licensed clinician."
    )
    story.append(Paragraph(disclaimer_text, small_muted_style))
    story.append(Spacer(1, 8))

    admin_data = [
        [
            Paragraph(f"<b>Report ID:</b> {report_id}", small_muted_style),
            Paragraph(f"<b>Patient ID:</b> {patient_id}", small_muted_style),
            Paragraph(f"<b>Exam Date:</b> {exam_date}", small_muted_style),
            Paragraph(f"<b>Examined Eye:</b> {eye_laterality}", small_muted_style),
        ]
    ]
    admin_table = Table(admin_data, colWidths=[1.9 * inch, 1.8 * inch, 2.0 * inch, 1.8 * inch])
    admin_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), COLOR_BG_LIGHT),
        ('PADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(admin_table)

    doc.build(story)
    return str(output_path)
