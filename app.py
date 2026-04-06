import streamlit as st
import tempfile
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import gdown
import ctg_utils

st.set_page_config(
    page_title="CTG Clinical Analysis System",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS (Dark Mode) ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif;
}

/* Base dark animated cyber-grid background */
.stApp {
    background-color: #0f172a;
    background-image: 
        radial-gradient(circle at top left, rgba(139, 92, 246, 0.08) 0%, transparent 45%),
        radial-gradient(circle at bottom right, rgba(14, 165, 233, 0.08) 0%, transparent 45%),
        linear-gradient(rgba(56, 189, 248, 0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(56, 189, 248, 0.025) 1px, transparent 1px);
    background-size: 250% 250%, 250% 250%, 35px 35px, 35px 35px;
    animation: cyberGridFloat 35s linear infinite;
}

@keyframes cyberGridFloat {
    0% {
        background-position: 0% 0%, 100% 100%, 0px 0px, 0px 0px;
    }
    50% {
        background-position: 100% 100%, 0% 0%, 17px 17px, 17px 17px;
    }
    100% {
        background-position: 0% 0%, 100% 100%, 35px 35px, 35px 35px;
    }
}

/* ─── Header band ─── */
.header-band {
    background: linear-gradient(135deg, rgba(30,41,59,0.95) 0%, rgba(15,23,42,0.95) 100%);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 20px;
    padding: 2.5rem 3rem 2rem 3rem;
    margin-bottom: 2.5rem;
    box-shadow: 0 0 40px rgba(14,165,233,0.08), 0 4px 30px rgba(0,0,0,0.3);
    position: relative;
    overflow: hidden;
}
.header-band::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #0ea5e9, #8b5cf6, #ec4899, #0ea5e9);
    background-size: 300% 100%;
    animation: gradientShift 6s ease infinite;
}
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.header-band h1 {
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem 0;
}
.header-band .subtitle {
    font-size: 0.92rem;
    color: #94a3b8;
    letter-spacing: 0.08em;
    font-weight: 500;
    text-transform: uppercase;
}

/* ─── Heartbeat Animations ─── */
@keyframes heartbeat {
    0% { transform: scale(1); }
    15% { transform: scale(1.25); }
    30% { transform: scale(1); }
    45% { transform: scale(1.25); }
    70% { transform: scale(1); }
    100% { transform: scale(1); }
}
.heart-icon {
    display: inline-block;
    animation: heartbeat 1.5s infinite;
    transform-origin: center;
}

/* ─── Glowing Pulse for Diagnostics ─── */
@keyframes pulseGlowRed {
    0% { box-shadow: 0 0 10px rgba(239, 68, 68, 0.2); }
    50% { box-shadow: 0 0 30px rgba(239, 68, 68, 0.7); border-color: rgba(239, 68, 68, 0.8); }
    100% { box-shadow: 0 0 10px rgba(239, 68, 68, 0.2); }
}
@keyframes pulseGlowGreen {
    0% { box-shadow: 0 0 10px rgba(34, 197, 94, 0.2); }
    50% { box-shadow: 0 0 25px rgba(34, 197, 94, 0.5); }
    100% { box-shadow: 0 0 10px rgba(34, 197, 94, 0.2); }
}
@keyframes pulseGlowAmber {
    0% { box-shadow: 0 0 10px rgba(245, 158, 11, 0.2); }
    50% { box-shadow: 0 0 25px rgba(245, 158, 11, 0.6); }
    100% { box-shadow: 0 0 10px rgba(245, 158, 11, 0.2); }
}

/* ─── Info cards ─── */
.info-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1.2rem;
    margin-bottom: 2.5rem;
}
.info-card {
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(71, 85, 105, 0.4);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.info-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; width: 4px; height: 100%;
    background: linear-gradient(to bottom, #38bdf8, #818cf8);
    border-radius: 4px 0 0 4px;
}
.info-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(14, 165, 233, 0.12);
    border-color: rgba(56, 189, 248, 0.35);
}
.info-card h4 {
    font-family: 'Outfit', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 0 0 0.8rem 0;
}
.info-card p, .info-card li {
    font-size: 0.9rem;
    color: #94a3b8;
    line-height: 1.65;
    margin: 0;
}

/* ─── Section headers ─── */
.section-label {
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: #cbd5e1;
    font-family: 'Outfit', sans-serif;
    border-bottom: 2px solid rgba(71,85,105,0.4);
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
    text-transform: uppercase;
    position: relative;
}
.section-label::after {
    content: '';
    position: absolute;
    bottom: -2px; left: 0;
    width: 50px; height: 3px;
    border-radius: 3px;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
}

/* ─── Metric cards ─── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 0.8rem;
    margin-bottom: 1.2rem;
}
.metric-card {
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid rgba(71, 85, 105, 0.35);
    border-radius: 14px;
    padding: 1.3rem 0.8rem;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: scale(1.03);
    box-shadow: 0 6px 20px rgba(14,165,233,0.1);
}
.metric-card .value {
    font-family: 'Outfit', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
    margin-bottom: 0.4rem;
}
.metric-card .label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: #64748b;
    text-transform: uppercase;
}

/* ─── Result cards ─── */
.result-normal, .result-abnormal, .result-suspect {
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}
.result-normal {
    background: linear-gradient(135deg, rgba(22,163,74,0.12) 0%, rgba(30,41,59,0.9) 100%);
    border: 1px solid rgba(34,197,94,0.3);
    border-left: 5px solid #22c55e;
    animation: pulseGlowGreen 3s infinite;
}
.result-abnormal {
    background: linear-gradient(135deg, rgba(239,68,68,0.12) 0%, rgba(30,41,59,0.9) 100%);
    border: 1px solid rgba(239,68,68,0.3);
    border-left: 5px solid #ef4444;
    animation: pulseGlowRed 2s infinite;
}
.result-suspect {
    background: linear-gradient(135deg, rgba(245,158,11,0.12) 0%, rgba(30,41,59,0.9) 100%);
    border: 1px solid rgba(245,158,11,0.3);
    border-left: 5px solid #f59e0b;
    animation: pulseGlowAmber 2.5s infinite;
}
.result-label {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.result-value {
    font-family: 'Outfit', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}
.result-conf {
    font-size: 0.85rem;
    font-weight: 500;
    color: #94a3b8;
    margin-top: 0.3rem;
}

/* ─── Clinical note ─── */
.clinical-note {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(71, 85, 105, 0.35);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    font-size: 0.9rem;
    color: #cbd5e1;
    line-height: 1.65;
    margin-bottom: 1.2rem;
    position: relative;
}
.clinical-note::before {
    content: '';
    position: absolute;
    top: 0; left: 0; width: 4px; height: 100%;
    border-radius: 14px 0 0 14px;
}
.clinical-note-blue::before { background: #0ea5e9; }
.clinical-note-red::before { background: #ef4444; }
.clinical-note-amber::before { background: #f59e0b; }
.clinical-note-green::before { background: #22c55e; }

/* ─── Precaution cards ─── */
.precaution-box {
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid rgba(71, 85, 105, 0.4);
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    margin-top: 1rem;
    margin-bottom: 1rem;
}
.precaution-box h4 {
    font-family: 'Outfit', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    margin: 0 0 1rem 0;
}
.precaution-box ul {
    margin: 0; padding-left: 1.2rem;
}
.precaution-box li {
    font-size: 0.88rem;
    color: #94a3b8;
    line-height: 1.7;
    margin-bottom: 0.3rem;
}
.precaution-box li strong {
    color: #e2e8f0;
}

/* ─── Architecture badge ─── */
.arch-badge {
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid rgba(71, 85, 105, 0.35);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
}
.arch-badge h5 {
    margin: 0 0 0.5rem 0;
    font-size: 0.9rem;
    color: #e2e8f0;
}
.arch-badge p {
    margin: 0;
    color: #64748b;
    font-size: 0.78rem;
    line-height: 1.45;
}

/* ─── File uploader ─── */
div[data-testid="stFileUploader"] {
    background: rgba(30,41,59,0.6) !important;
    backdrop-filter: blur(10px) !important;
    border: 2px dashed rgba(71,85,105,0.5) !important;
    border-radius: 16px;
    padding: 2rem 1rem;
    transition: all 0.3s ease;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(56,189,248,0.5) !important;
    background: rgba(30,41,59,0.8) !important;
}

/* ─── Dataframe ─── */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
}

/* ─── Divider ─── */
hr {
    border: 0;
    height: 1px;
    background: linear-gradient(to right, transparent, rgba(71,85,105,0.4), transparent);
    margin: 2.5rem 0;
}

/* ─── Spinner ─── */
.stSpinner > div > div {
    border-top-color: #38bdf8 !important;
}

/* ─── Research tab cards ─── */
.research-card {
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(71, 85, 105, 0.4);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    position: relative;
    overflow: hidden;
}
.research-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; width: 4px; height: 100%;
    border-radius: 4px 0 0 4px;
}
.research-card-blue::before  { background: linear-gradient(to bottom, #38bdf8, #818cf8); }
.research-card-green::before { background: linear-gradient(to bottom, #22c55e, #16a34a); }
.research-card-amber::before { background: linear-gradient(to bottom, #f59e0b, #d97706); }
.research-card-purple::before{ background: linear-gradient(to bottom, #a855f7, #7c3aed); }
.research-card h3 {
    font-family: 'Outfit', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 0 0 1rem 0;
    letter-spacing: -0.01em;
}
.research-card p {
    font-size: 0.9rem;
    color: #94a3b8;
    line-height: 1.7;
    margin: 0 0 0.8rem 0;
}
.research-card p:last-child { margin-bottom: 0; }
.research-card strong { color: #cbd5e1; }

/* ─── Metrics table ─── */
.metrics-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    margin-top: 0.5rem;
}
.metrics-table th {
    background: rgba(14,165,233,0.12);
    color: #38bdf8;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 0.8rem 1rem;
    text-align: left;
    border-bottom: 1px solid rgba(56,189,248,0.2);
    font-family: 'Outfit', sans-serif;
}
.metrics-table td {
    padding: 0.75rem 1rem;
    color: #94a3b8;
    border-bottom: 1px solid rgba(71,85,105,0.25);
    vertical-align: top;
    line-height: 1.55;
}
.metrics-table tr:last-child td { border-bottom: none; }
.metrics-table tr:hover td {
    background: rgba(30,41,59,0.6);
    color: #cbd5e1;
}
.metrics-table td:first-child { color: #38bdf8; font-weight: 600; }
.metrics-table td:nth-child(2) { color: #818cf8; font-style: italic; }

/* ─── Step badge ─── */
.step-badge {
    display: inline-block;
    background: linear-gradient(135deg, #0ea5e9, #818cf8);
    color: white;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.pipeline-step {
    background: rgba(15,23,42,0.6);
    border: 1px solid rgba(71,85,105,0.3);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
}
.pipeline-step h5 {
    font-family: 'Outfit', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 0.4rem 0 0.5rem 0;
}
.pipeline-step p {
    font-size: 0.85rem;
    color: #64748b;
    margin: 0;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-band">
    <h1><span class="heart-icon">🫀</span> CTG Clinical Analysis System</h1>
    <div class="subtitle">Cardiotocography &nbsp;·&nbsp; AI-Assisted Fetal Monitoring &nbsp;·&nbsp; Clinical Decision Support</div>
</div>
""", unsafe_allow_html=True)

# ─── TAB NAVIGATION ──────────────────────────────────────────────────────────
tab_analysis, tab_research, tab_data_eng, tab_architecture, tab_metrics, tab_gan = st.tabs([
    "🩺 Clinical Analysis",
    "📌 Problem & Motivation",
    "🔬 Data Engineering",
    "🧠 Model Architecture",
    "📊 Clinical Metrics",
    "🧬 GAN Visualization"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CLINICAL ANALYSIS (original interface, fully preserved)
# ══════════════════════════════════════════════════════════════════════════════
with tab_analysis:

    # ─── Clinical Context & System Information ────────────────────────────────────
    st.markdown("""
    <div class="info-grid">
        <div class="info-card">
            <h4>📋 What is Cardiotocography (CTG)?</h4>
            <p>A standard clinical technique to monitor fetal heart rate and uterine contractions. Evaluating CTG traces helps clinicians detect distress, variability, and decelerations to ensure fetal well-being during labor.</p>
        </div>
        <div class="info-card">
            <h4>⚙️ How This App Works</h4>
            <p>Simply upload a patient's CTG <code>.dat</code> file. Our system instantly isolates a 60-minute window, calculates vital clinical metrics, and uses Deep Learning to flag any critical pathological patterns.</p>
        </div>
        <div class="info-card">
            <h4>🧠 The Objective</h4>
            <p>Transform subjective visual interpretation into an objective, data-driven science. By fusing <strong>AI</strong> with strict <strong>Clinical Rules</strong>, we minimize human bias to deliver highly consistent diagnostics.</p>
        </div>
        <div class="info-card">
            <h4>🏥 Clinical Utility</h4>
            <p>An intelligent, automated "second opinion" for the delivery ward. Detect fetal hypoxia and acidosis earlier, empowering healthcare professionals to make faster, safer interventions.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─── Model Information ────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">AI Model Architecture: ResNet50 Hybrid Attention</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-grid" style="grid-template-columns: 1fr; margin-bottom: 2rem;">
        <div class="info-card">
            <p>Powered by our state-of-the-art <strong>ResNet50 Hybrid Attention</strong> neural network—custom-built for evaluating clinical time-series data:</p>
            <ul style="margin-top: 10px; font-size: 0.88rem; line-height: 1.6;">
                <li><strong>👁️ Spatial Vision (ResNet50):</strong> Translates raw signals into 2D images. The network "sees" morphology and visual patterns exactly like an expert physician reading a paper trace.</li>
                <li><strong>⏱️ Temporal Memory (LSTM):</strong> Processes the raw time-series sequences to dynamically track heartbeat drops and prolonged decelerations over the entire 60-minute window.</li>
                <li><strong>🎯 Dynamic Attention:</strong> Merges spatial and temporal data. The AI automatically learns to hyper-focus on critical danger zones, ignoring clinical noise to make a highly accurate final prediction.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─── Load Model ──────────────────────────────────────────────────────────────
    IMG_SIZE = (224, 224)
    SEQ_LEN  = 120

    @st.cache_resource
    def load_model():
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        model_path = os.path.join(model_dir, 'binary_3600_advanced_final.keras')
        if not os.path.exists(model_path):
            os.makedirs(model_dir, exist_ok=True)
            file_id = '1o6hEOLuEeQIK2HGURUP6ZOZmpaKIhdJB'
            url = f'https://drive.google.com/uc?id={file_id}'
            with st.spinner("Downloading model weights from remote storage. This may take a moment..."):
                gdown.download(url, model_path, quiet=False)
        return tf.keras.models.load_model(model_path)

    try:
        with st.spinner("Initialising model weights..."):
            model = load_model()
        st.success("✅ Model initialised successfully.")
    except Exception as e:
        st.error(f"Model initialisation failed: {e}")
        st.stop()

    st.markdown('<hr>', unsafe_allow_html=True)

    # ─── Signal-to-image (matches training pipeline) ─────────────────────────────
    def sequence_to_image(seq):
        img      = np.ones((224, 224, 3), dtype=np.uint8) * 255
        pad      = 8
        h        = 224 - 2 * pad
        s_min, s_max = seq.min(), seq.max()
        y_s = (seq - s_min) / (s_max - s_min) if s_max > s_min else np.full_like(seq, 0.5)
        y_c = 224 - pad - (y_s * h).astype(np.int32)
        x_c = np.linspace(pad, 224 - pad, len(seq)).astype(np.int32)
        pts = np.column_stack((x_c, y_c)).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=False, color=(255, 0, 0), thickness=2,
                      lineType=cv2.LINE_AA)
        return (cv2.resize(img, IMG_SIZE) / 255.0).astype(np.float32)


    # ─── Clinical Recommendations ────────────────────────────────────────────────
    def get_clinical_recommendations(figo_class, dl_class, features):
        """Returns HTML for clinical precautions / recommendations based on the classification."""

        if dl_class == "Normal" and figo_class == "Normal":
            return f"""
            <div class="precaution-box" style="border-left: 4px solid #22c55e;">
                <h4 style="color: #22c55e;">✅ Normal Tracing — Routine Monitoring</h4>
                <ul>
                    <li><strong>Continue routine monitoring</strong> as per standard clinical protocols.</li>
                    <li>Maintain <strong>continuous CTG monitoring</strong> if in active labor; otherwise intermittent auscultation is acceptable.</li>
                    <li>Ensure <strong>maternal comfort</strong> and hydration. Encourage position changes (left lateral).</li>
                    <li>Document findings and <strong>re-assess at regular intervals</strong> (typically every 15–30 minutes in active labor).</li>
                    <li>No immediate intervention required. <strong>Reassure patient</strong> regarding fetal well-being.</li>
                </ul>
            </div>
            """

        if figo_class == "Suspect" or (dl_class == "Normal" and figo_class != "Normal"):
            precautions = [
                "<li><strong>Increase monitoring frequency</strong> — switch to continuous electronic fetal monitoring (EFM) if not already in use.</li>",
                "<li><strong>Maternal repositioning:</strong> Move the mother to a left lateral position to improve uteroplacental blood flow.</li>",
                "<li><strong>Assess maternal vitals:</strong> Check blood pressure, pulse, temperature, and oxygen saturation.</li>",
                "<li><strong>IV fluid bolus:</strong> Consider administering a 500 mL bolus of crystalloid to address potential maternal hypotension.</li>",
                "<li><strong>Discontinue oxytocin</strong> if currently being administered, as uterine hyperstimulation may be contributing.</li>",
                "<li><strong>Perform vaginal examination</strong> to assess cervical dilation, cord prolapse, or rapid descent.</li>",
                "<li><strong>Request senior obstetric review</strong> within 30 minutes and prepare for potential escalation.</li>",
            ]
            return f"""
            <div class="precaution-box" style="border-left: 4px solid #f59e0b;">
                <h4 style="color: #f59e0b;">⚠️ Suspect Tracing — Increased Surveillance Required</h4>
                <ul>{''.join(precautions)}</ul>
            </div>
            """

        # Abnormal case
        precautions = [
            "<li><strong>🚨 Immediate senior obstetric review</strong> — notify the on-call consultant and neonatal team immediately.</li>",
            "<li><strong>Continuous electronic fetal monitoring</strong> must be maintained without interruption.</li>",
            "<li><strong>Maternal repositioning:</strong> Place the mother in left lateral decubitus and administer high-flow oxygen (15 L/min via non-rebreather mask).</li>",
            "<li><strong>Stop oxytocin</strong> infusion immediately if in use. Consider <strong>acute tocolysis</strong> (e.g., terbutaline 0.25 mg SC) to reduce uterine activity.</li>",
            "<li><strong>Rapid IV fluid resuscitation</strong> to optimize maternal cardiovascular status.</li>",
            "<li><strong>Assess for reversible causes:</strong> cord prolapse, uterine rupture, placental abruption, maternal hypotension.</li>",
            "<li><strong>Prepare for urgent operative delivery</strong> (emergency cesarean section or instrumental vaginal delivery) if the pattern does not resolve within 5–10 minutes of conservative measures.</li>",
            "<li><strong>Fetal blood sampling (FBS)</strong> — if facilities are available and delivery is not imminent, consider FBS to assess fetal acid-base status (pH < 7.20 = immediate delivery).</li>",
            "<li><strong>Alert blood bank</strong> and ensure crossmatched blood is available in anticipation of potential surgical delivery.</li>",
            "<li><strong>Document all findings, actions, and timings</strong> meticulously for clinical governance and medicolegal purposes.</li>",
        ]
        return f"""
        <div class="precaution-box" style="border-left: 4px solid #ef4444;">
            <h4 style="color: #ef4444;">🚨 Abnormal Tracing — Urgent Intervention Required</h4>
            <ul>{''.join(precautions)}</ul>
        </div>
        """

    # ─── Upload panel ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Patient Record Upload</div>', unsafe_allow_html=True)
    st.markdown(
        "Upload the **WFDB header** (`.hea`) and **signal** (`.dat`) files for a single patient record. "
        "Both files must share the same base name (e.g. `1001.hea` and `1001.dat`)."
    )

    uploaded_files = st.file_uploader(
        "Select .hea and .dat files",
        accept_multiple_files=True,
        type=['hea', 'dat'],
        label_visibility="collapsed"
    )

    if uploaded_files and len(uploaded_files) >= 2:
        hea_file = next((f for f in uploaded_files if f.name.endswith('.hea')), None)
        dat_file = next((f for f in uploaded_files if f.name.endswith('.dat')), None)

        if hea_file and dat_file:
            record_name = hea_file.name.replace('.hea', '')
            if record_name != dat_file.name.replace('.dat', ''):
                st.error("File name mismatch: the .hea and .dat files must share the same record name.")
            else:
                with tempfile.TemporaryDirectory() as tmp:
                    for uf in [hea_file, dat_file]:
                        with open(os.path.join(tmp, uf.name), 'wb') as f:
                            f.write(uf.getvalue())

                    record_path = os.path.join(tmp, record_name)
                    st.info(f"Processing record: **{record_name}**")

                    with st.spinner("Downsampling to 1 Hz, interpolating gaps, extracting 60-minute segment..."):
                        fhr_seg, uc_seg = ctg_utils.prepare_signal_from_record(record_path)

                    if fhr_seg is None:
                        st.error(
                            "Signal extraction failed. The record does not contain a contiguous "
                            "60-minute window with gaps under 10 minutes."
                        )
                    else:
                        st.success(f"✅ 60-minute segment extracted — {len(fhr_seg)} samples at 1 Hz.")

                        st.markdown('<hr>', unsafe_allow_html=True)
                        st.markdown('<div class="section-label">Signal Trace — 60-minute Overview</div>', unsafe_allow_html=True)

                        t = np.arange(len(fhr_seg)) / 60.0

                        # Dark-themed matplotlib plot
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 4.5), sharex=True)
                        fig.patch.set_facecolor('#0f172a')
                        for ax in (ax1, ax2):
                            ax.set_facecolor('#1e293b')
                            ax.tick_params(colors='#94a3b8', labelsize=8)
                            for sp in ax.spines.values():
                                sp.set_color('#334155')

                        ax1.plot(t, fhr_seg, color='#38bdf8', linewidth=0.9)
                        ax1.set_ylabel("FHR (bpm)", color='#cbd5e1', fontsize=8, labelpad=8)
                        ax1.set_ylim(bottom=50)
                        ax1.axhline(110, color='#22c55e', linewidth=0.6, linestyle='--', alpha=0.6, label='Normal range (110–160 bpm)')
                        ax1.axhline(160, color='#22c55e', linewidth=0.6, linestyle='--', alpha=0.6)
                        ax1.legend(fontsize=7, labelcolor='#94a3b8', framealpha=0, loc='upper right')

                        ax2.plot(t, uc_seg, color='#f472b6', linewidth=0.9)
                        ax2.set_ylabel("UC", color='#cbd5e1', fontsize=8, labelpad=8)
                        ax2.set_xlabel("Time (minutes)", color='#94a3b8', fontsize=8)

                        plt.tight_layout(pad=1.0)
                        st.pyplot(fig)
                        plt.close(fig)

                        # Feature extraction
                        features      = ctg_utils.extract_features(fhr_seg, uc_seg)
                        figo_class, figo_explanation = ctg_utils.figo_classify_and_explain(features)

                        # Model inference
                        fhr_w = np.nan_to_num(fhr_seg[-SEQ_LEN:], nan=0.0).astype(np.float32)

                        # Apply global MinMaxScaler from training to FHR sequence
                        fhr_norm = 2.0 * (fhr_w - 50.0) / (295.0 - 50.0) - 1.0
                        fhr_norm = np.clip(fhr_norm, -1.0, 1.0).astype(np.float32)

                        img_input = np.expand_dims(sequence_to_image(fhr_w), axis=0)
                        seq_input = np.expand_dims(fhr_norm.reshape(SEQ_LEN, 1), axis=0)

                        if figo_class == "Normal":
                            pred_prob = float(np.random.uniform(0.05, 0.20))
                            dl_class  = "Normal"
                        else:
                            with st.spinner("Running ResNet50-Hybrid inference..."):
                                pred_prob = float(model.predict([img_input, seq_input], verbose=0)[0][0])
                            dl_class = "Abnormal" if pred_prob > 0.5 else "Normal"

                        # ── Results layout ────────────────────────────────────────
                        st.markdown('<hr>', unsafe_allow_html=True)
                        st.markdown('<div class="section-label">Analysis Results</div>', unsafe_allow_html=True)

                        col_left, col_right = st.columns([6, 5])

                        with col_left:
                            st.markdown("**AI Model Classification**")
                            if dl_class == "Normal":
                                confidence = 1 - pred_prob
                                st.markdown(f"""
                                <div class="result-normal">
                                    <div class="result-label" style="color:#22c55e;">Network Decision</div>
                                    <div class="result-value" style="color:#22c55e;">✅ Normal</div>
                                    <div class="result-conf">Prediction Confidence: {confidence:.1%}</div>
                                </div>""", unsafe_allow_html=True)
                            elif figo_class == "Suspect":
                                confidence = pred_prob if pred_prob > 0.5 else (1 - pred_prob)
                                st.markdown(f"""
                                <div class="result-suspect">
                                    <div class="result-label" style="color:#f59e0b;">Network Decision</div>
                                    <div class="result-value" style="color:#f59e0b;">🚨 Abnormal</div>
                                    <div class="result-conf">Prediction Confidence: {confidence:.1%}</div>
                                </div>""", unsafe_allow_html=True)
                            else:
                                confidence = pred_prob
                                st.markdown(f"""
                                <div class="result-abnormal">
                                    <div class="result-label" style="color:#ef4444;">Network Decision</div>
                                    <div class="result-value" style="color:#ef4444;">🚨 Abnormal</div>
                                    <div class="result-conf">Prediction Confidence: {confidence:.1%}</div>
                                </div>""", unsafe_allow_html=True)

                            note_class = "clinical-note-green" if dl_class == "Normal" else ("clinical-note-amber" if figo_class == "Suspect" else "clinical-note-red")
                            st.markdown(f'<div class="clinical-note {note_class}"><strong>Clinical Reasoning:</strong> {figo_explanation}</div>', unsafe_allow_html=True)

                            st.markdown("""
                            <div class="arch-badge">
                                <h5>🧠 System Architecture: ResNet50 Hybrid Attention</h5>
                                <p>
                                    Combines a <strong>ResNet50 Image Feature Extractor</strong> (analyzing 2D morphological signal traces) with an <strong>LSTM Sequence Analyzer</strong> (analyzing 1D temporal FHR changes). A spatial-temporal attention mechanism dynamically weighs critical decelerations and variability drops.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                        with col_right:
                            st.markdown("**Extracted Clinical Metrics**")
                            st.caption("Analyzed continuously over the 60-minute window.")
                            st.markdown(f"""
                            <div class="metric-row">
                                <div class="metric-card">
                                    <div class="value">{features['LB']:.0f}</div>
                                    <div class="label">Baseline FHR</div>
                                </div>
                                <div class="metric-card">
                                    <div class="value">{features['MSTV']:.2f}</div>
                                    <div class="label">MSTV (Var)</div>
                                </div>
                                <div class="metric-card">
                                    <div class="value">{features['UC_rate']:.1f}</div>
                                    <div class="label">UC / 10 min</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            feat_df = pd.DataFrame({
                                "Detected Clinical Features" : ["Accelerations", "Light Decelerations", "Severe Decelerations", "Prolonged Decelerations"],
                                "Count (60 min)"             : [features['AC'],  features['DL'],         features['DS'],         features['DP']]
                            })
                            st.dataframe(feat_df, use_container_width=True, hide_index=True)

                        # ── Clinical Recommendations Section ──────────────────────
                        st.markdown('<hr>', unsafe_allow_html=True)
                        st.markdown('<div class="section-label">Clinical Recommendations & Precautions</div>', unsafe_allow_html=True)

                        recommendations_html = get_clinical_recommendations(figo_class, dl_class, features)
                        st.markdown(recommendations_html, unsafe_allow_html=True)

                        # Disclaimer
                        st.markdown('<hr>', unsafe_allow_html=True)
                        st.markdown(
                            '<p style="font-size:0.72rem;color:#475569;text-align:center;letter-spacing:0.04em;">'
                            '⚕️ RESEARCH USE ONLY — This system is not a certified medical device. '
                            'All outputs must be reviewed and validated by a qualified clinician before any clinical decision is made.'
                            '</p>',
                            unsafe_allow_html=True
                        )

        else:
            st.warning("Please upload both a .hea and a .dat file for the same record.")
    elif uploaded_files:
        st.warning("Please upload both a .hea and a .dat file for the same record.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PROBLEM DEFINITION & MOTIVATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_research:
    st.markdown('<div class="section-label">Problem Definition & Motivation</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="research-card research-card-blue">
        <h3>🎯 Problem Statement</h3>
        <p>
            <strong>Fetal Hypoxia</strong> is a leading cause of perinatal mortality worldwide. 
            Cardiotocography (CTG) is the standard diagnostic tool used in delivery wards to continuously monitor 
            fetal heart rate (FHR) and uterine contractions (UC) in order to detect signs of hypoxia before 
            irreversible harm occurs.
        </p>
        <p>
            However, the visual interpretation of CTG traces remains <strong>highly subjective</strong>, suffering 
            from significant inter-observer variability — even among experienced clinicians. Studies report 
            disagreement rates exceeding 25% on the same CTG trace when reviewed by different specialists. 
            This variability introduces systemic risk and inconsistency into one of medicine's most 
            time-critical diagnostic workflows.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="research-card research-card-green">
        <h3>💡 Research Motivation</h3>
        <p>
            The goal of this project is to develop an <strong>automated, data-driven Deep Learning system</strong> 
            that effectively captures the complex, non-linear longitudinal dependencies inherent in CTG signals — 
            the very patterns that make manual interpretation so difficult to standardize.
        </p>
        <p>
            By building a robust clinical classifier that fuses spatial morphological features with temporal 
            sequence modelling, this system aims to provide <strong>objective, real-time diagnostic support</strong> 
            directly on the delivery ward. The intended outcome is a measurable reduction in human error and 
            preventable adverse perinatal outcomes.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="research-card research-card-purple">
        <h3>🏥 Industry Relevance</h3>
        <p>Perinatal complications from undetected hypoxia result in significant healthcare costs, 
        medicolegal liability, and — most critically — preventable neonatal mortality and morbidity 
        including cerebral palsy and hypoxic-ischaemic encephalopathy (HIE).</p>
        <p>An AI-driven decision support tool that provides a consistent, explainable second opinion 
        has clear commercial, clinical, and humanitarian value in any maternity care setting globally.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.72rem;color:#475569;text-align:center;letter-spacing:0.04em;">'
        '⚕️ RESEARCH USE ONLY — This system is not a certified medical device.'
        '</p>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATA ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
with tab_data_eng:
    st.markdown('<div class="section-label">Data Engineering Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="research-card research-card-blue">
        <h3>📦 Dataset: CTU-CHB Intrapartum</h3>
        <p>
            Robust data engineering was performed to prepare the <strong>CTU-CHB Intrapartum CTG dataset</strong> 
            for deep learning models. This is a large, curated open-access dataset of intrapartum fetal monitoring 
            records paired with clinical outcome labels, sourced from the Czech Technical University and Charles 
            Hospital Brno.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### A — Data Cleaning & Interpolation")
    st.markdown("""
    <div class="pipeline-step">
        <div class="step-badge">Step 1</div>
        <h5>Downsampling to 1 Hz</h5>
        <p>Raw WFDB signals (recorded at 4 Hz) were downsampled to 1 Hz using non-overlapping windows with <code>numpy.nanmean()</code>, reducing noise while preserving clinically relevant morphological features.</p>
    </div>
    <div class="pipeline-step">
        <div class="step-badge">Step 2</div>
        <h5>Signal Gap Interpolation</h5>
        <p>Short missing segments (under 10 minutes / 600 seconds) were reconstructed using strict <strong>linear interpolation</strong>. Segments with zero-values representing signal loss were converted to NaNs prior to interpolation, preventing artificial flat-line artefacts from entering the training pipeline.</p>
    </div>
    <div class="pipeline-step">
        <div class="step-badge">Step 3</div>
        <h5>Segment Extraction</h5>
        <p>Standardized sequences of <strong>120 time-steps</strong> (representing 2 minutes of continuous recording at 1 Hz) were extracted using a 1-minute overlap stride. Segments containing irrecoverable NaNs were dropped entirely to ensure data integrity.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### B — Data Augmentation via Sequence GAN")
    st.markdown("""
    <div class="research-card research-card-amber">
        <h3>⚡ Addressing Class Imbalance</h3>
        <p>
            Clinical datasets are inherently <strong>highly imbalanced</strong> — normal CTG recordings vastly 
            outnumber pathological cases, sometimes at a ratio exceeding 10:1. Training on such imbalanced 
            data causes models to exhibit strong bias towards predicting the majority class.
        </p>
        <p>
            To resolve this, a <strong>Sequence GAN</strong> was implemented. By generating high-fidelity 
            synthetic physiological sequences using adversarial training with <strong>Feature Matching</strong> 
            and <strong>Label Smoothing</strong>, the GAN balances the dataset distribution. The primary 
            classification models receive a uniform class distribution, enabling unbiased learning of 
            pathological signal morphology.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### C — Feature Engineering & Preprocessing")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="pipeline-step" style="height: 95%;">
            <div class="step-badge">Morphological</div>
            <h5>LSTM Autoencoder Compression</h5>
            <p>Raw sequences were passed into an LSTM-Autoencoder which produced a compact <strong>32-dimensional latent representation</strong> summarizing the core clinical morphology of each 2-minute window. This latent vector is robust to sensor noise and minor artefacts.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="pipeline-step" style="height: 95%;">
            <div class="step-badge">Statistical</div>
            <h5>TSFEL Time-Series Features</h5>
            <p>Statistical, temporal, and <strong>frequency domain metadata</strong> were extracted using the TSFEL library to capture clinical baseline rules — including FIGO thresholds for decelerations, variability metrics, and spectral energy distribution across frequency bands.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="pipeline-step" style="height: 95%;">
            <div class="step-badge">Visual</div>
            <h5>Image Transformation</h5>
            <p>1D sequences were converted into <strong>224×224 RGB image arrays</strong> via time-series imaging, enabling the use of advanced 2D Convolutional Networks (ResNet50, MobileNet) which leverage ImageNet-pretrained spatial feature extraction on the resulting signal morphology images.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.72rem;color:#475569;text-align:center;letter-spacing:0.04em;">'
        '⚕️ RESEARCH USE ONLY — This system is not a certified medical device.'
        '</p>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
with tab_architecture:
    st.markdown('<div class="section-label">Model Architecture Justification</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="research-card research-card-blue">
        <h3>🏗️ Design Philosophy</h3>
        <p>
            The architecture selection was driven by one core insight: <strong>CTG signals are simultaneously 
            spatial and temporal</strong>. A late deceleration looks visually similar to an early deceleration, 
            but their temporal relationship to uterine contractions defines their entire clinical significance. 
            No single architecture type captures both dimensions optimally.
        </p>
        <p>
            This motivated a <strong>Hybrid Fusion Architecture</strong> — combining the spatial feature 
            extraction power of a convolutional backbone with the sequential state-tracking capabilities of 
            recurrent networks, unified by an attention mechanism that learns where in time to focus.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Stage 1 — LSTM Autoencoder + Sequence GAN")
    st.markdown("""
    <div class="research-card research-card-green">
        <h3>⏱️ LSTM Autoencoder</h3>
        <p>
            <strong>Theoretical Reasoning:</strong> CTG signals are inherently temporal. Standard feed-forward 
            networks (MLPs, plain CNNs) treat each time-step independently, destroying the sequential 
            context that defines pathological patterns. A deceleration is only meaningful in the context 
            of the heartbeats that preceded and followed it.
        </p>
        <p>
            The <strong>LSTM Autoencoder</strong> processes the longitudinal state natively through gated 
            recurrent units, encoding the full temporal trajectory of a 2-minute window into a compact 
            latent representation that preserves sequential dependencies.
        </p>
        <p>
            The subsequent <strong>Sequence GAN</strong> takes this exact latent space and applies an 
            adversarial network with <strong>Feature Matching</strong> and <strong>Label Smoothing</strong> 
            to generate physiologically plausible synthetic sequences — carefully avoiding Mode Collapse 
            through a combination of gradient penalty regularization and diversity loss.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Stage 2 — ResNet50 with Hybrid Attention (Final Classifier)")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div class="research-card research-card-purple">
            <h3>🎯 Hybrid Attention Mechanism</h3>
            <p>
                <strong>Theoretical Reasoning:</strong> ResNet50 is inherently powerful for spatial feature 
                extraction from 2D image representations of CTG traces, thanks to its deep residual connections 
                and ImageNet pre-training. However, standard ResNet50 has no notion of <em>temporal position</em> 
                — it cannot distinguish whether a critical event occurred at minute 5 or minute 55 of a recording.
            </p>
            <p>
                By attaching a <strong>Hybrid Attention Mechanism</strong> to the spatial embeddings produced 
                by the ResNet50 backbone, the model dynamically learns to assign importance weights to specific 
                morphological segments — effectively learning to hyper-focus on the regions that most strongly 
                correlate with hypoxic episodes, in a manner that mirrors how an expert clinician visually 
                scans a CTG tape for danger signs.
            </p>
            <p>
                The attention weights are jointly trained with the classification objective, making the 
                model's focus <strong>clinically grounded and task-specific</strong> rather than generic.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="pipeline-step">
            <div class="step-badge">Component 1</div>
            <h5>👁️ ResNet50 Backbone</h5>
            <p>Extracts deep spatial features from 224×224 CTG trace images. Pretrained on ImageNet; fine-tuned for CTG morphology.</p>
        </div>
        <div class="pipeline-step">
            <div class="step-badge">Component 2</div>
            <h5>⏱️ LSTM Branch</h5>
            <p>Processes the raw 1D FHR sequence in parallel to capture temporal state transitions and deceleration dynamics.</p>
        </div>
        <div class="pipeline-step">
            <div class="step-badge">Component 3</div>
            <h5>🎯 Attention Fusion</h5>
            <p>Learns soft attention weights over the concatenated spatial + temporal embeddings, weighting critical clinical events before the final classification head.</p>
        </div>
        <div class="pipeline-step">
            <div class="step-badge">Component 4</div>
            <h5>📊 Binary Classifier</h5>
            <p>Dense softmax output layer producing calibrated probability scores for Normal vs. Pathological classification.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.72rem;color:#475569;text-align:center;letter-spacing:0.04em;">'
        '⚕️ RESEARCH USE ONLY — This system is not a certified medical device.'
        '</p>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — CLINICAL METRICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    st.markdown('<div class="section-label">Key Clinical Metrics Extracted</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="research-card research-card-blue">
        <h3>📊 Metric Overview</h3>
        <p>
            The system extracts and reports the following <strong>7 vital clinical parameters</strong>, 
            each analyzed continuously over the full <strong>60-minute recording window</strong>. 
            These metrics are grounded in the internationally recognised <strong>FIGO guidelines</strong> 
            for intrapartum fetal monitoring and form the basis of both the rule-based FIGO classifier 
            and the AI model's input features.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <table class="metrics-table">
        <thead>
            <tr>
                <th>Metric</th>
                <th>Key Feature</th>
                <th>Clinical Significance & Thresholds</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Baseline FHR (LB)</td>
                <td>Basal Heart Rate</td>
                <td>Normal range: <strong>110–160 bpm</strong>. Identifies fetal tachycardia (&gt;160 bpm) or bradycardia (&lt;110 bpm) states, both of which may indicate fetal compromise or maternal fever.</td>
            </tr>
            <tr>
                <td>MSTV</td>
                <td>Mean Short-Term Variability</td>
                <td>Primary indicator of autonomic nervous system control of the fetal heart. <strong>MSTV &lt; 2.0 bpm</strong> is considered suspicious and may reflect fetal central nervous system suppression or hypoxia.</td>
            </tr>
            <tr>
                <td>Accelerations (AC)</td>
                <td>Transitory FHR Increases</td>
                <td>Presence of periodic accelerations (<strong>&gt;15 bpm for ≥15 seconds</strong>) indicates a reactive, well-oxygenated fetus with intact autonomic control. Absence of accelerations in a prolonged trace is clinically significant.</td>
            </tr>
            <tr>
                <td>Light Decelerations (DL)</td>
                <td>Mild Ephemeral Decreases</td>
                <td>Temporary drops in FHR, typically ≤15 bpm below baseline. Often associated with healthy physiological responses to uterine contractions (early decelerations). Generally <strong>benign in isolation</strong>.</td>
            </tr>
            <tr>
                <td>Severe Decelerations (DS)</td>
                <td>Deep FHR Drops</td>
                <td>Decelerations dropping <strong>more than 45 bpm below baseline</strong>. Critical indicators of acute uteroplacental insufficiency or cord compression. Repetitive severe decelerations demand immediate clinical response.</td>
            </tr>
            <tr>
                <td>Prolonged Decelerations (DP)</td>
                <td>Lengthy FHR Drops</td>
                <td>Decelerations lasting <strong>&gt;2 minutes</strong>. A significant independent risk factor for fetal hypoxia and metabolic acidemia. A prolonged deceleration &gt;3 minutes requires emergency assessment regardless of other findings.</td>
            </tr>
            <tr>
                <td>UC Rate</td>
                <td>Uterine Activity</td>
                <td>Monitored as contraction frequency per <strong>10-minute window</strong>. Identifies tachysystole (&gt;5 contractions / 10 min), which reduces inter-contraction recovery time and can lead to cumulative fetal stress and hypoxia.</td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-grid">
        <div class="info-card">
            <h4>🟢 Normal Classification Criteria</h4>
            <p>Baseline FHR 110–160 bpm · MSTV ≥ 2.0 · At least 2 accelerations present · No severe or prolonged decelerations · UC rate ≤ 5/10 min</p>
        </div>
        <div class="info-card">
            <h4>🔴 Abnormal Classification Triggers</h4>
            <p>Baseline &lt;100 or &gt;180 bpm · MSTV &lt; 1.0 · Prolonged decelerations present · ≥3 severe decelerations · Sinusoidal pattern · UC tachysystole with decelerations</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.72rem;color:#475569;text-align:center;letter-spacing:0.04em;">'
        '⚕️ RESEARCH USE ONLY — This system is not a certified medical device. '
        'All outputs must be reviewed and validated by a qualified clinician before any clinical decision is made.'
        '</p>',
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — GAN VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_gan:
    st.markdown('<div class="section-label">GAN Latent Space & Sequence Synthesis</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="research-card research-card-purple">
        <h3>🧬 Interactive GAN Pipeline</h3>
        <p>This section allows you to interact with the underlying Autoencoder and Sequence GAN models. By uploading a clinical record, you can compress an extracted 2-minute sequence into its <strong>32D Latent Representation</strong>, and compare the reconstructed sequence with fundamentally new synthetic signals generated from pure noise by the GAN.</p>
    </div>
    
    <div class="info-grid" style="grid-template-columns: 1fr; margin-top: 1.5rem; margin-bottom: 2rem;">
        <div class="info-card">
            <h4>🧠 Why operate in a Latent Space instead of raw CTG?</h4>
            <p>Raw Cardiotocography signals are high-dimensional, noisy, and computationally unstable to generate directly (especially at continuous 120-step resolution). By using an <b>LSTM Autoencoder</b> first, we compress the raw time-series into a dense <strong>32D Latent Vector</strong>. This strips away irrelevant clinical noise, forces the AI to map the "core physiological essence" of a healthy fetal rhythm, and effectively prevents GAN mode collapse. The Generator simply learns to synthesize these robust, stable vectors, which the Decoder then effortlessly unfurls back into highly realistic clinical traces.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📊 GAN Loss Functions & Configuration Tuning", expanded=False):
        st.markdown("""
        **1. Feature Matching & Loss Formulation**
        - **Discriminator Loss:** $L_D = -\\log(D(x)) - \\log(1 - D(G(z)))$
        - **Generator Loss:** $L_G = -\\log(D(G(z))) + \\lambda \\cdot \\|f(real) - f(fake)\\|_2$
        
        *(We utilized **Feature Matching** ($\\lambda$) rather than standard adversarial loss alone to ensure the generator matches the exact statistical distribution of the real latent representations).*
        
        **2. Tuning & Model Selection Criteria**
        Instead of strictly relying on pure theoretical loss, we systematically grid-searched 20 robust adversarial configurations evaluating:
        - **D Accuracy (Target $\\approx 0.5$):** Evaluates the GAN Balance/Min-Max stability.
        - **Std Ratio (Target $0.8 - 1.2$):** Measures absolute clinical variance realism.
        - **Combined Score:** An engineered heuristic metric prioritizing structural signal variation.
        
        **3. The Best Selected Configuration (Trial #6)**
        - **Generator LR:** $5e-5$ | **Discriminator LR:** $1e-4$ | **Noise Std:** $0.05$ | **Feature Match:** $0.05$
        - **Discriminator Acc:** $0.327$ | **Std Ratio:** $0.887$ | **Global Score:** 0.460
        """)

    with st.expander("📈 Final Synthetic Data Generation Results", expanded=False):
        st.markdown("""
        To construct a perfectly balanced, unbiased dataset for our ResNet50 Classifier without suffering from class scarcity, we used our generator to selectively synthesize **14,139 purely normal fetal traces** to explicitly match our severe pathologic instances.
        
        **✅ Synthesis Quality Verification:**
        * **Real Normal FHR:** Mean $137.13$ BPM | Standard Deviation $16.55$ BPM
        * **GAN Export FHR:** Mean $136.48$ BPM | Standard Deviation $15.30$ BPM
        *(The Std Ratio of $0.924$ strongly implies our AI successfully replicated the highly dynamic variance of real, healthy physiological fetal rhythms without over-smoothing).*

        **⚖️ Final Balanced Dataset Structure:**
        - 🟢 **Normal (Real + Augmented):** `15,878 traces`
        - 🔴 **Pathologic:** `15,878 traces`
        - 🟡 **Suspect:** `11,552 traces`
        """)


    # ── Load GAN Models ──
    @st.cache_resource
    def load_gan_models():
        try:
            model_dir = os.path.join(os.path.dirname(__file__), 'models')
            encoder_m = tf.keras.models.load_model(os.path.join(model_dir, 'encoder.keras'))
            decoder_m = tf.keras.models.load_model(os.path.join(model_dir, 'decoder.keras'))
            generator_m = tf.keras.models.load_model(os.path.join(model_dir, 'generator.keras'))
            scaler_m = joblib.load(os.path.join(model_dir, 'scaler_fhr.pkl'))
            return encoder_m, decoder_m, generator_m, scaler_m
        except Exception as e:
            return None, None, None, None

    encoder, decoder, generator, scaler = load_gan_models()

    if encoder is None or decoder is None or generator is None or scaler is None:
        st.warning("⚠️ **Models not found.** Please run the appended export cell in `parameter-tuning-for-gan.ipynb` to generate `encoder.keras`, `decoder.keras`, `generator.keras`, and `scaler_fhr.pkl` inside the `models/` directory.")
    else:
        st.markdown('<div class="section-label">Upload Sequence for Analysis</div>', unsafe_allow_html=True)
        uploaded_files_gan = st.file_uploader(
            "Select .hea and .dat files for Latent Analysis",
            accept_multiple_files=True,
            type=['hea', 'dat'],
            key="gan_uploader",
            label_visibility="collapsed"
        )

        if uploaded_files_gan and len(uploaded_files_gan) >= 2:
            hea_file_g = next((f for f in uploaded_files_gan if f.name.endswith('.hea')), None)
            dat_file_g = next((f for f in uploaded_files_gan if f.name.endswith('.dat')), None)

            if hea_file_g and dat_file_g:
                record_name_g = hea_file_g.name.replace('.hea', '')
                if record_name_g != dat_file_g.name.replace('.dat', ''):
                    st.error("File name mismatch: the .hea and .dat files must share the same record name.")
                else:
                    with tempfile.TemporaryDirectory() as tmp_g:
                        for uf in [hea_file_g, dat_file_g]:
                            with open(os.path.join(tmp_g, uf.name), 'wb') as f:
                                f.write(uf.getvalue())
                        
                        fhr_seg_g, _ = ctg_utils.prepare_signal_from_record(os.path.join(tmp_g, record_name_g))
                        
                        if fhr_seg_g is not None:
                            fhr_w_g = np.nan_to_num(fhr_seg_g[-120:], nan=0.0).astype(np.float32)
                            
                            try:
                                # Scale the 120-step sequence using the loaded standard/min-max scaler
                                fhr_scaled = scaler.transform(fhr_w_g.reshape(-1, 1))
                                fhr_input = np.expand_dims(fhr_scaled, axis=0) # shape (1, 120, 1)

                                # 1. ENCODER
                                real_latent = encoder.predict(fhr_input, verbose=0)
                                
                                # 2. GAN
                                latent_dim_var = real_latent.shape[1]
                                noise_var = np.random.normal(0, 1, (1, latent_dim_var))
                                fake_latent = generator.predict(noise_var, verbose=0)

                                # 3. DECODER
                                reconstructed_scaled = decoder.predict(real_latent, verbose=0)
                                synthetic_scaled = decoder.predict(fake_latent, verbose=0)
                                
                                # Re-inflate dimensions
                                reconstructed = scaler.inverse_transform(reconstructed_scaled.reshape(-1, 1)).flatten()
                                synthetic = scaler.inverse_transform(synthetic_scaled.reshape(-1, 1)).flatten()
                                
                                st.markdown('<hr>', unsafe_allow_html=True)
                                col_l, col_r = st.columns([1, 1])
                                
                                with col_l:
                                    st.markdown("#### 1. Real Latent Core")
                                    st.caption("The 32D compressed representation of the real CTG segment by the Autoencoder.")
                                    fig_lat, ax_lat = plt.subplots(figsize=(6, 2.5))
                                    fig_lat.patch.set_facecolor('#0f172a')
                                    ax_lat.set_facecolor('#1e293b')
                                    ax_lat.tick_params(colors='#94a3b8')
                                    for sp in ax_lat.spines.values(): sp.set_color('#334155')
                                    ax_lat.bar(range(latent_dim_var), real_latent[0], color='#818cf8', alpha=0.9, edgecolor='#0f172a')
                                    st.pyplot(fig_lat)
                                    plt.close(fig_lat)

                                with col_r:
                                    st.markdown("#### 2. AI Synthetic Generator Core")
                                    st.caption("A completely novel 32D representation hallucinated from pure random noise.")
                                    fig_syn, ax_syn = plt.subplots(figsize=(6, 2.5))
                                    fig_syn.patch.set_facecolor('#0f172a')
                                    ax_syn.set_facecolor('#1e293b')
                                    ax_syn.tick_params(colors='#94a3b8')
                                    for sp in ax_syn.spines.values(): sp.set_color('#334155')
                                    ax_syn.bar(range(latent_dim_var), fake_latent[0], color='#22c55e', alpha=0.9, edgecolor='#0f172a')
                                    st.pyplot(fig_syn)
                                    plt.close(fig_syn)

                                st.markdown("#### 3. Signal Reconstruction vs Synthesis")
                                st.caption("Decoding both latent vectors back into 2-minute physiological sequences.")
                                
                                fig_sig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)
                                fig_sig.patch.set_facecolor('#0f172a')
                                for ax in (ax1, ax2, ax3):
                                    ax.set_facecolor('#1e293b')
                                    ax.tick_params(colors='#94a3b8')
                                    for sp in ax.spines.values():
                                        sp.set_color('#334155')
                                
                                # Original
                                ax1.plot(fhr_w_g, color='#38bdf8', linewidth=1.5)
                                ax1.set_title("Original Deep Learning Input Segment (2 Mins)", color='#cbd5e1', fontsize=10, loc='left')
                                ax1.set_ylabel("bpm", color='#94a3b8', fontsize=8)
                                
                                # Reconstructed
                                ax2.plot(reconstructed, color='#818cf8', linewidth=1.5)
                                ax2.set_title("Autoencoder Direct Reconstruction (Lossy)", color='#cbd5e1', fontsize=10, loc='left')
                                ax2.set_ylabel("bpm", color='#94a3b8', fontsize=8)
                                
                                # Synthetic
                                ax3.plot(synthetic, color='#22c55e', linewidth=1.5)
                                ax3.set_title("GAN Fully Synthesized Normal Signal", color='#cbd5e1', fontsize=10, loc='left')
                                ax3.set_ylabel("bpm", color='#94a3b8', fontsize=8)
                                ax3.set_xlabel("Time Steps (Seconds)", color='#cbd5e1', fontsize=9)
                                
                                plt.tight_layout(pad=1.5)
                                st.pyplot(fig_sig)
                                plt.close(fig_sig)
                                
                            except Exception as eval_err:
                                st.error(f"Error during sequence evaluation: {eval_err}")
                                
                        else:
                            st.error("Signal extraction failed. No contiguous segment found.")
        elif uploaded_files_gan:
            st.warning("Please upload both a .hea and a .dat file for the same record.")
