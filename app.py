import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import os

st.set_page_config(page_title="PHANTOM", layout="wide")

PHANTOM_FULL = "Predictive Health Analytics of Nephron Toxicity from Oral Microbiomes"

# ----------------------------
# Styling
# ----------------------------
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: Georgia, "Times New Roman", serif !important;
        color: #f0f1f2 !important;
    }

    .stApp {
        background-color: #1f2937 !important;
        color: #1f2937 !important;
    }

    p, div, label, span, li {
        color: #f0f1f2 !important;
    }

    h1, h2, h3 {
        color: #a95eff !important;
        font-family: Georgia, "Times New Roman", serif !important;
    }

    div.stButton > button {
        background-color: #a95eff !important;
        color: white !important;
        border-radius: 10px !important;
        border: 1px solid #4a009e !important;
        font-weight: 600 !important;
    }

    div.stButton > button:hover {
        background-color: #4a009e !important;
        color: white !important;
    }

    .landing-wrap {
        text-align: center;
        padding-top: 40px;
        padding-bottom: 30px;
    }

    .landing-title {
        font-size: 64px;
        font-weight: 800;
        letter-spacing: 2px;
        color: #7a0019;
        margin-top: 16px;
        margin-bottom: 10px;
    }

    .landing-sub {
        font-size: 20px;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 10px;
    }

    .landing-desc {
        font-size: 20px;
        line-height: 1.6;
        color: #374151;
        max-width: 900px;
        margin: 0 auto 24px auto;
    }

    .dashboard-card {
        background: black;
        border: 2px solid #9046e3;
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        margin-bottom: 16px;
    }

    .section-panel {
        background: black;
        border: 1px solid #9046e3;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        margin-bottom: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Session state
# ----------------------------
if "started" not in st.session_state:
    st.session_state.started = False
if "latest_report_pdf" not in st.session_state:
    st.session_state.latest_report_pdf = None

# ----------------------------
# Helper functions
# ----------------------------
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def draw_unicode_text(image_np, text_items):
    pil_img = Image.fromarray(image_np)
    draw = ImageDraw.Draw(pil_img)

    font = None
    possible_fonts = ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"]
    for f in possible_fonts:
        try:
            font = ImageFont.truetype(f, 24)
            break
        except:
            continue

    if font is None:
        font = ImageFont.load_default()

    for text, x, y in text_items:
        draw.text((x + 1, y + 1), text, fill=(0, 0, 0), font=font)
        draw.text((x, y), text, fill=(255, 255, 0), font=font)

    return np.array(pil_img)


def detect_cytokine_bands(image_np, lane_order):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    h, w = gray_blur.shape

    x_start = int(w * 0.10)
    x_end = int(w * 0.90)
    y_start = int(h * 0.15)
    y_end = int(h * 0.85)

    roi = gray_blur[y_start:y_end, x_start:x_end]
    roi_h, roi_w = roi.shape
    lane_width = roi_w // 3

    annotated = image_np.copy()
    results = {}
    text_items = []

    for i, marker in enumerate(lane_order):
        lx1 = x_start + i * lane_width
        lx2 = x_start + (i + 1) * lane_width if i < 2 else x_end

        lane = gray_blur[y_start:y_end, lx1:lx2]
        row_means = lane.mean(axis=1)
        darkest_row = int(np.argmin(row_means))

        band_half_height = max(8, roi_h // 40)
        by1 = max(y_start + darkest_row - band_half_height, y_start)
        by2 = min(y_start + darkest_row + band_half_height, y_end)

        band_region = gray_blur[by1:by2, lx1:lx2]
        band_darkness = 255 - float(np.mean(band_region))

        results[marker] = {
            "lane_x1": lx1,
            "lane_x2": lx2,
            "band_y1": by1,
            "band_y2": by2,
            "raw_darkness": band_darkness,
        }

    raw_vals = [results[m]["raw_darkness"] for m in lane_order]
    min_v = min(raw_vals)
    max_v = max(raw_vals)

    for marker in lane_order:
        raw = results[marker]["raw_darkness"]

        if max_v - min_v < 1e-6:
            rel = 0.5
        else:
            rel = (raw - min_v) / (max_v - min_v)

        if marker == "IL-1β":
            pgml = 5 + rel * 95
        elif marker == "IL-6":
            pgml = 5 + rel * 145
        else:
            pgml = 5 + rel * 115

        results[marker]["relative_intensity"] = rel
        results[marker]["pgml"] = pgml

        cv2.rectangle(
            annotated,
            (results[marker]["lane_x1"], results[marker]["band_y1"]),
            (results[marker]["lane_x2"], results[marker]["band_y2"]),
            (122, 0, 25),
            2,
        )

        label_x = results[marker]["lane_x1"] + 5
        label_y = max(results[marker]["band_y1"] - 28, 20)
        text_items.append((f"{marker}: {pgml:.1f} pg/mL", label_x, label_y))

    annotated = draw_unicode_text(annotated, text_items)
    return annotated, results


def resolve_other(selection, other_text):
    if selection == "Other":
        return other_text.strip() if other_text.strip() else "Other (not specified)"
    return selection


def calculate_risk(patient, cytokines):
    contributions = {}

    contributions["IL-1β"] = cytokines["IL-1β"] * 0.18
    contributions["IL-6"] = cytokines["IL-6"] * 0.22
    contributions["TNF-α"] = cytokines["TNF-α"] * 0.20

    age = patient["Age"]
    if age >= 65:
        contributions["Age"] = 14
    elif age >= 50:
        contributions["Age"] = 9
    elif age >= 35:
        contributions["Age"] = 5
    else:
        contributions["Age"] = 1

    perio_map = {
        "Healthy": 0,
        "Gingivitis": 6,
        "Mild Periodontitis": 12,
        "Moderate Periodontitis": 22,
        "Severe Periodontitis": 34,
        "Advanced / Refractory Periodontitis": 40
    }
    contributions["Periodontal Status"] = perio_map.get(patient["Periodontal Status Resolved"], 10)

    bleeding_map = {
        "None": 0,
        "Occasional": 4,
        "Frequent": 9,
        "Severe / spontaneous": 14
    }
    contributions["Bleeding Gums"] = bleeding_map.get(patient["Bleeding Gums Resolved"], 6)

    smoking_map = {
        "Never": 0,
        "Former (>5 years)": 3,
        "Former (<5 years)": 6,
        "Occasional": 8,
        "Daily": 14,
        "Heavy": 18
    }
    contributions["Smoking"] = smoking_map.get(patient["Smoking Resolved"], 6)

    diabetes_map = {
        "No": 0,
        "Prediabetes": 6,
        "Type 1 Diabetes": 14,
        "Type 2 Diabetes": 16,
        "Gestational history": 4
    }
    contributions["Diabetes"] = diabetes_map.get(patient["Diabetes Resolved"], 6)

    bp_map = {
        "No history": 0,
        "Borderline": 4,
        "Controlled": 8,
        "Uncontrolled": 14
    }
    contributions["Blood Pressure"] = bp_map.get(patient["Blood Pressure Resolved"], 6)

    family_map = {
        "None known": 0,
        "Extended family": 3,
        "One first-degree relative": 8,
        "Multiple first-degree relatives": 12
    }
    contributions["Family Kidney History"] = family_map.get(patient["Family Kidney History Resolved"], 4)

    bmi = patient["BMI"]
    if bmi >= 35:
        contributions["BMI"] = 10
    elif bmi >= 30:
        contributions["BMI"] = 7
    elif bmi >= 25:
        contributions["BMI"] = 4
    else:
        contributions["BMI"] = 1

    hygiene_map = {
        "Excellent": 0,
        "Good": 2,
        "Average": 5,
        "Poor": 9,
        "Very poor": 13
    }
    contributions["Oral Hygiene"] = hygiene_map.get(patient["Oral Hygiene Resolved"], 5)

    egfr = patient["eGFR"]
    uacr = patient["UACR"]

    if egfr > 0:
        if egfr < 60:
            contributions["eGFR"] = 22
        elif egfr < 90:
            contributions["eGFR"] = 8
        else:
            contributions["eGFR"] = 0
    else:
        contributions["eGFR"] = 0

    if uacr > 0:
        if uacr >= 300:
            contributions["UACR"] = 24
        elif uacr >= 30:
            contributions["UACR"] = 14
        else:
            contributions["UACR"] = 0
    else:
        contributions["UACR"] = 0

    creatinine = patient["Serum Creatinine"]
    if creatinine > 0:
        if creatinine >= 1.5:
            contributions["Serum Creatinine"] = 10
        elif creatinine >= 1.2:
            contributions["Serum Creatinine"] = 5
        else:
            contributions["Serum Creatinine"] = 0
    else:
        contributions["Serum Creatinine"] = 0

    albumin = patient["Serum Albumin"]
    if albumin > 0:
        if albumin < 3.5:
            contributions["Serum Albumin"] = 8
        else:
            contributions["Serum Albumin"] = 0
    else:
        contributions["Serum Albumin"] = 0

    crp = patient["CRP"]
    if crp > 0:
        if crp >= 10:
            contributions["CRP"] = 10
        elif crp >= 3:
            contributions["CRP"] = 5
        else:
            contributions["CRP"] = 0
    else:
        contributions["CRP"] = 0

    score = sum(contributions.values())

    if score < 55:
        category = "Low"
        color = "green"
    elif score < 105:
        category = "Moderate"
        color = "orange"
    else:
        category = "High"
        color = "red"

    return score, category, color


def project_risk_over_time(current_score):
    days = [0, 7, 14, 21, 28]
    projected = [
        current_score,
        current_score * 1.04,
        current_score * 1.08,
        current_score * 1.13,
        current_score * 1.19,
    ]
    return days, projected


def risk_progress_html(score, category):
    score = max(0, min(score, 160))
    percent = (score / 160) * 100
    category_colors = {"Low": "#16a34a", "Moderate": "#f59e0b", "High": "#dc2626"}
    color = category_colors.get(category, "#7a0019")

    html = f"""
    <div style="margin-top:10px;">
        <div style="width:100%; background:linear-gradient(90deg,#16a34a 0%,#f59e0b 55%,#dc2626 100%);
                    height:22px; border-radius:999px; position:relative;">
            <div style="position:absolute; left:{percent}%; top:-6px; transform:translateX(-50%);
                        width:6px; height:34px; background:black; border-radius:4px;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:12px; margin-top:4px;">
            <span>Low</span><span>Moderate</span><span>High</span>
        </div>
        <div style="margin-top:8px; font-size:14px;">
            Current status: <span style="color:{color}; font-weight:700;">{category}</span>
        </div>
    </div>
    """
    return html


def build_pdf_report(patient_data, cytokine_values, score, category, logo_path=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontName="Times-Bold",
        fontSize=22,
        textColor=colors.HexColor("#7a0019"),
        alignment=TA_CENTER,
        spaceAfter=10,
    )
    sub_style = ParagraphStyle(
        "SubStyle",
        parent=styles["BodyText"],
        fontName="Times-Roman",
        fontSize=10.5,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#333333"),
        spaceAfter=16,
    )
    section_style = ParagraphStyle(
        "SectionStyle",
        parent=styles["Heading2"],
        fontName="Times-Bold",
        fontSize=13,
        textColor=colors.HexColor("#7a0019"),
        alignment=TA_LEFT,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["BodyText"],
        fontName="Times-Roman",
        fontSize=10.5,
        leading=14,
        textColor=colors.black,
    )

    story = []

    if logo_path and os.path.exists(logo_path):
        story.append(RLImage(logo_path, width=75, height=75))
        story.append(Spacer(1, 8))

    story.append(Paragraph("PHANTOM REPORT", title_style))
    story.append(Paragraph(PHANTOM_FULL, sub_style))

    story.append(Paragraph("Patient Summary", section_style))
    patient_table = Table([
        ["Patient ID", str(patient_data["Patient ID"])],
        ["Age", str(patient_data["Age"])],
        ["Sex", str(patient_data["Sex"])],
        ["Ethnicity", str(patient_data["Ethnicity"])],
        ["Periodontal Status", str(patient_data["Periodontal Status Resolved"])],
        ["Smoking", str(patient_data["Smoking Resolved"])],
        ["Diabetes", str(patient_data["Diabetes Resolved"])],
        ["Blood Pressure", str(patient_data["Blood Pressure Resolved"])],
    ], colWidths=[180, 300])
    patient_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#833F3F")),
        ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"),
        ("FONTNAME", (0, 0), (0, -1), "Times-Bold"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Cytokine Estimates", section_style))
    cytokine_table = Table([
        ["Biomarker", "Estimated Value (pg/mL)"],
        ["IL-1β", f"{cytokine_values['IL-1β']:.2f}"],
        ["IL-6", f"{cytokine_values['IL-6']:.2f}"],
        ["TNF-α", f"{cytokine_values['TNF-α']:.2f}"],
    ], colWidths=[220, 180])
    cytokine_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7a0019")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(cytokine_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("PHANTOM Result", section_style))
    story.append(Paragraph(f"<b>Risk Score:</b> {score:.2f}", body_style))
    story.append(Paragraph(f"<b>Risk Category:</b> {category}", body_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Interpretation", section_style))
    interp = (
        "This profile suggests a relatively low oral–renal inflammatory burden."
        if category == "Low" else
        "This profile suggests a moderate inflammatory burden that may warrant closer monitoring."
        if category == "Moderate" else
        "This profile suggests a high inflammatory burden associated with elevated oral–renal risk concern."
    )
    story.append(Paragraph(interp, body_style))
    story.append(Spacer(1, 14))

    story.append(Paragraph("Prototype Note", section_style))
    story.append(Paragraph(
        "This PHANTOM report is generated from a proof-of-concept system using prototype Western blot image analysis and simulated biomarker calibration. It is intended for demonstration purposes only and is not for clinical diagnosis.",
        body_style
    ))

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf


# ----------------------------
# Landing page
# ----------------------------
if not st.session_state.started:
    logo_path = "phantom_logo.png"

    if os.path.exists(logo_path):
        c1, c2, c3 = st.columns([3, 1, 3])
        with c2:
            st.image(logo_path, width=220)

    st.markdown("<div class='landing-title'>PHANTOM</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='landing-sub'>{PHANTOM_FULL}</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='landing-desc'>
        A proof-of-concept clinical analytics platform for estimating oral–renal inflammatory risk
        through Western blot cytokine analysis, patient biodata, and predictive modeling.
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns([1.2, 1, 1.2])
    with c2:
        if st.button("Start now", use_container_width=True, type="primary"):
            st.session_state.started = True
            st.rerun()

    st.stop()

# ----------------------------
# Main App
# ----------------------------
st.title("PHANTOM")
st.subheader(PHANTOM_FULL)

top_left, top_right = st.columns([4, 1])
with top_right:
    if st.button("Return to Start Page", use_container_width=True):
        st.session_state.started = False
        st.rerun()

st.info(
    "PHANTOM is a proof-of-concept prototype. It estimates cytokine abundance from a Western blot image "
    "and combines those results with patient factors to generate an oral–renal inflammatory risk profile."
)

# ----------------------------
# Patient Data
# ----------------------------
with st.container(border=True):
    st.header("Patient Data")

    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)
    r3c1, r3c2 = st.columns(2)

    with r1c1:
        patient_id = st.text_input("Patient / Study ID", value="PH-001")
        age = st.slider("Age", 10, 100, 35)
        sex = st.selectbox("Sex", ["Female", "Male", "Intersex", "Prefer not to say"])
        ethnicity = st.selectbox(
            "Ethnicity",
            [
                "South Asian", "East Asian", "Black", "White", "Middle Eastern",
                "Latino/Hispanic", "Indigenous", "Mixed", "Other", "Prefer not to say"
            ]
        )
        ethnicity_other = st.text_area("Specify ethnicity", height=70) if ethnicity == "Other" else ""

    with r1c2:
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=23.5, step=0.1)
        periodontal_status = st.selectbox(
            "Periodontal Status",
            [
                "Healthy", "Gingivitis", "Mild Periodontitis", "Moderate Periodontitis",
                "Severe Periodontitis", "Advanced / Refractory Periodontitis", "Other"
            ],
        )
        periodontal_other = st.text_area("Describe periodontal findings", height=80) if periodontal_status == "Other" else ""

        bleeding_gums = st.selectbox(
            "Bleeding Gums",
            ["None", "Occasional", "Frequent", "Severe / spontaneous", "Other"]
        )
        bleeding_other = st.text_area("Describe bleeding gum findings", height=80) if bleeding_gums == "Other" else ""

    with r2c1:
        oral_hygiene = st.selectbox(
            "Oral Hygiene",
            ["Excellent", "Good", "Average", "Poor", "Very poor", "Other"]
        )
        oral_hygiene_other = st.text_area("Describe oral hygiene observations", height=70) if oral_hygiene == "Other" else ""

        tooth_loss = st.selectbox(
            "Tooth Loss from Gum Disease",
            ["None", "1–2 teeth", "3–5 teeth", "6+ teeth", "Unknown", "Other"]
        )
        tooth_loss_other = st.text_area("Describe tooth loss findings", height=70) if tooth_loss == "Other" else ""

    with r2c2:
        smoking = st.selectbox(
            "Smoking",
            ["Never", "Former (>5 years)", "Former (<5 years)", "Occasional", "Daily", "Heavy", "Other"]
        )
        smoking_other = st.text_area("Specify smoking status", height=70) if smoking == "Other" else ""

        diabetes = st.selectbox(
            "Diabetes",
            ["No", "Prediabetes", "Type 1 Diabetes", "Type 2 Diabetes", "Gestational history", "Other"]
        )
        diabetes_other = st.text_area("Specify diabetes-related details", height=70) if diabetes == "Other" else ""

    with r3c1:
        blood_pressure = st.selectbox(
            "Blood Pressure",
            ["No history", "Borderline", "Controlled", "Uncontrolled", "Other"]
        )
        bp_other = st.text_area("Specify blood pressure details", height=70) if blood_pressure == "Other" else ""

        family_kidney = st.selectbox(
            "Family Kidney History",
            ["None known", "Extended family", "One first-degree relative", "Multiple first-degree relatives", "Other"]
        )
        family_other = st.text_area("Specify kidney family history", height=70) if family_kidney == "Other" else ""

    with r3c2:
        notes = st.text_area("Clinical Notes / Observed Symptoms", height=140)

    st.markdown("### Advanced Clinical Inputs")
    adv1, adv2, adv3 = st.columns(3)
    with adv1:
        egfr = st.number_input("eGFR (mL/min/1.73m²)", min_value=0.0, value=0.0, step=0.1)
        uacr = st.number_input("UACR (mg/g)", min_value=0.0, value=0.0, step=0.1)

    with adv2:
        serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, value=0.0, step=0.1)
        serum_albumin = st.number_input("Serum Albumin (g/dL)", min_value=0.0, value=0.0, step=0.1)

    with adv3:
        crp = st.number_input("C-Reactive Protein / CRP (mg/L)", min_value=0.0, value=0.0, step=0.1)
        st.caption("Optional values may be left at 0 if unavailable.")

patient_data = {
    "Patient ID": patient_id,
    "Age": age,
    "Sex": sex,
    "Ethnicity": resolve_other(ethnicity, ethnicity_other),
    "BMI": bmi,
    "Periodontal Status Resolved": resolve_other(periodontal_status, periodontal_other),
    "Bleeding Gums Resolved": resolve_other(bleeding_gums, bleeding_other),
    "Oral Hygiene Resolved": resolve_other(oral_hygiene, oral_hygiene_other),
    "Tooth Loss": resolve_other(tooth_loss, tooth_loss_other),
    "Smoking Resolved": resolve_other(smoking, smoking_other),
    "Diabetes Resolved": resolve_other(diabetes, diabetes_other),
    "Blood Pressure Resolved": resolve_other(blood_pressure, bp_other),
    "Family Kidney History Resolved": resolve_other(family_kidney, family_other),
    "eGFR": egfr,
    "UACR": uacr,
    "Serum Creatinine": serum_creatinine,
    "Serum Albumin": serum_albumin,
    "CRP": crp,
    "Clinical Notes": notes,
}

# ----------------------------
# Blot Upload
# ----------------------------
with st.container(border=True):
    st.header("Blot Upload and Analysis")

    lane_order_option = st.selectbox(
        "Lane Order",
        [
            "IL-1β | IL-6 | TNF-α",
            "IL-6 | TNF-α | IL-1β",
            "TNF-α | IL-1β | IL-6",
            "Custom fixed order (left to right)"
        ]
    )

    if lane_order_option == "IL-1β | IL-6 | TNF-α":
        lane_order = ["IL-1β", "IL-6", "TNF-α"]
    elif lane_order_option == "IL-6 | TNF-α | IL-1β":
        lane_order = ["IL-6", "TNF-α", "IL-1β"]
    elif lane_order_option == "TNF-α | IL-1β | IL-6":
        lane_order = ["TNF-α", "IL-1β", "IL-6"]
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            custom1 = st.selectbox("Lane 1", ["IL-1β", "IL-6", "TNF-α"], key="lane1")
        with c2:
            custom2 = st.selectbox("Lane 2", ["IL-1β", "IL-6", "TNF-α"], key="lane2")
        with c3:
            custom3 = st.selectbox("Lane 3", ["IL-1β", "IL-6", "TNF-α"], key="lane3")
        lane_order = [custom1, custom2, custom3]

    uploaded_file = st.file_uploader(
        "Upload blot image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
    )

    analyze = st.button("Run PHANTOM Analysis", type="primary")

# ----------------------------
# Results
# ----------------------------
if analyze:
    if uploaded_file is None:
        st.error("Please upload a Western blot image first.")
    else:
        image_np = preprocess_image(uploaded_file)
        annotated_image, band_results = detect_cytokine_bands(image_np, lane_order)

        cytokine_values = {
            "IL-1β": band_results["IL-1β"]["pgml"],
            "IL-6": band_results["IL-6"]["pgml"],
            "TNF-α": band_results["TNF-α"]["pgml"],
        }

        score, category, color = calculate_risk(patient_data, cytokine_values)
        logo_path = "phantom_logo.png" if os.path.exists("phantom_logo.png") else None
        st.session_state.latest_report_pdf = build_pdf_report(patient_data, cytokine_values, score, category, logo_path)

        st.header("Results")

        left, right = st.columns([1.2, 1])

        with left:
            st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
            st.subheader("Blot Image")
            st.image(annotated_image, caption="Uploaded blot with detected cytokine bands", use_container_width=True)

            st.markdown("#### Projected Risk Severity Over Time")
            days, projected_scores = project_risk_over_time(score)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(days, projected_scores, marker="o", linewidth=2)
            ax.set_xlabel("Days")
            ax.set_ylabel("Risk Score")
            ax.set_title("Projected Oral–Renal Risk Trajectory")
            ax.set_xticks(days)

            ymin = max(0, min(projected_scores) - 5)
            ymax = max(projected_scores) + 8
            ax.set_ylim(ymin, ymax)

            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            st.caption(
                "Projection shown for demonstration only. This represents a prototype untreated-risk trajectory, "
                "not a clinical prediction."
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
            st.subheader("Calculated Results")
            st.markdown(f"### Risk Profile: :{color}[{category}]")
            st.metric("PHANTOM Risk Score", f"{score:.1f}")
            st.markdown(risk_progress_html(score, category), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
            st.subheader("Cytokine Estimates")
            result_df = pd.DataFrame({
                "Biomarker": ["IL-1β", "IL-6", "TNF-α"],
                "Estimated Value (pg/mL)": [
                    round(cytokine_values["IL-1β"], 2),
                    round(cytokine_values["IL-6"], 2),
                    round(cytokine_values["TNF-α"], 2),
                ],
                "Relative Intensity": [
                    round(band_results["IL-1β"]["relative_intensity"], 3),
                    round(band_results["IL-6"]["relative_intensity"], 3),
                    round(band_results["TNF-α"]["relative_intensity"], 3),
                ]
            })
            st.dataframe(result_df, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
            st.subheader("Patient Summary")
            st.write(f"**ID:** {patient_data['Patient ID']}")
            st.write(f"**Age:** {patient_data['Age']}")
            st.write(f"**Sex:** {patient_data['Sex']}")
            st.write(f"**Ethnicity:** {patient_data['Ethnicity']}")
            st.write(f"**Periodontal status:** {patient_data['Periodontal Status Resolved']}")
            st.write(f"**Smoking:** {patient_data['Smoking Resolved']}")
            st.write(f"**Diabetes:** {patient_data['Diabetes Resolved']}")
            st.write(f"**Blood pressure:** {patient_data['Blood Pressure Resolved']}")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
            st.subheader("Interpretation")
            if category == "Low":
                st.success("This profile suggests a relatively low oral–renal inflammatory burden.")
            elif category == "Moderate":
                st.warning("This profile suggests a moderate inflammatory burden that may warrant closer monitoring.")
            else:
                st.error("This profile suggests a high inflammatory burden associated with elevated oral–renal risk concern.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.caption(
            "Prototype note: cytokine values are estimated from relative band darkness and are intended for demonstration purposes only, not clinical diagnosis."
        )

# ----------------------------
# PDF Report
# ----------------------------
with st.container(border=True):
    st.header("Format Results")
    if st.session_state.latest_report_pdf is None:
        st.write("Run an analysis first to generate a formatted report.")
    else:
        st.success("A styled PHANTOM PDF report is ready.")
        st.download_button(
            label="Download PHANTOM PDF Report",
            data=st.session_state.latest_report_pdf,
            file_name="PHANTOM_report.pdf",
            mime="application/pdf"
        )
        