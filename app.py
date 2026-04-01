import streamlit as st
import tempfile
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import gdown
import ctg_utils

st.set_page_config(
    page_title="CTG Clinical Analysis System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #1e293b;
}

/* Base Streamlit overriding */
.stApp {
    background-color: #f4f6f9;
}

/* Top header band */
.header-band {
    background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
    border-bottom: 3px solid #005b96;
    padding: 2.5rem 3rem 2rem 3rem;
    margin: -1rem -1rem 2rem -1rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}
.header-band h1 {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    color: #004b7c;
    margin: 0 0 0.4rem 0;
}
.header-band .subtitle {
    font-size: 0.95rem;
    color: #475569;
    letter-spacing: 0.03em;
    font-weight: 500;
    text-transform: uppercase;
}

/* Info section */
.info-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1.5rem;
    margin-bottom: 2.5rem;
}
.info-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #005b96;
    border-radius: 6px;
    padding: 1.5rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.info-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.05);
}
.info-card h4 {
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: #005b96;
    margin: 0 0 0.8rem 0;
    border-bottom: 1px solid #f1f5f9;
    padding-bottom: 0.4rem;
}
.info-card p {
    font-size: 0.92rem;
    color: #334155;
    line-height: 1.6;
    margin: 0;
}

/* Section headers */
.section-label {
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: #004b7c;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 0.5rem;
    margin-bottom: 1.2rem;
    text-transform: uppercase;
}

/* Upload zone styling override */
.upload-zone {
    background: #ffffff;
    border: 2px dashed #94a3b8;
    border-radius: 8px;
    padding: 2.5rem;
    margin-bottom: 1.5rem;
    text-align: center;
}

/* Metric card */
.metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 1rem;
}
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
}
.metric-card .value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #005b96;
    line-height: 1;
    margin-bottom: 0.4rem;
}
.metric-card .label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    color: #64748b;
    text-transform: uppercase;
}

/* Result cards */
.result-normal {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 5px solid #22c55e;
    border-radius: 6px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.result-abnormal {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-left: 5px solid #ef4444;
    border-radius: 6px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.result-suspect {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 5px solid #f59e0b;
    border-radius: 6px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.result-label {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.result-value {
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}
.result-conf {
    font-size: 0.85rem;
    color: #475569;
    margin-top: 0.3rem;
}

/* Clinical note */
.clinical-note {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #64748b;
    border-radius: 6px;
    padding: 1.2rem;
    font-size: 0.9rem;
    color: #334155;
    line-height: 1.6;
    margin-bottom: 1.5rem;
}

/* Table */
.stDataFrame { font-size: 0.85rem !important; }

/* Spinner */
.stSpinner { color: #005b96 !important; }

/* Divider */
hr { border-color: #e2e8f0 !important; }

/* Streamlit block tweaks */
div[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 2px dashed #cbd5e1 !important;
    border-radius: 8px;
    padding: 1rem;
}
div[data-testid="stFileUploader"]:hover {
    border-color: #005b96 !important;
}
/* Ensure the upload text is legible */
div[data-testid="stFileUploader"] section {
    color: #475569 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-band">
    <h1>CTG Clinical Analysis System</h1>
    <div class="subtitle">Cardiotocography &nbsp;·&nbsp; AI-Assisted Fetal Monitoring &nbsp;·&nbsp; FIGO Clinical Guidelines</div>
</div>
""", unsafe_allow_html=True)

# ─── Clinical Context & System Information ────────────────────────────────────
st.markdown("""
<div class="info-grid">
    <div class="info-card">
        <h4>What is Cardiotocography (CTG)?</h4>
        <p>Cardiotocography is a fundamental clinical technique used to continuously monitor the fetal heart rate (FHR) and the mother's uterine contractions (UC). In clinical practice, interpreting these signals assesses fetal well-being, highlighting aspects such as baseline heart rate, beat-to-beat variability, accelerations, and signs of deceleration during contractions.</p>
    </div>
    <div class="info-card">
        <h4>What This System Will Do</h4>
        <p>This platform accepts raw, 1 Hz downsampled CTG patient records. It automatically extracts a contiguous 60-minute segment, calculates vital clinical metrics, and processes the signal to detect any underlying pathological conditions. It provides clear, actionable results combining traditional interpretation criteria with algorithmic AI processing.</p>
    </div>
    <div class="info-card">
        <h4>Technical Objective</h4>
        <p>The primary aim of this system is to bridge the gap between subjective visual interpretation by clinicians and objective, data-driven analysis. By fusing a <strong>Deep Learning Model</strong> (ResNet50 Hybrid Attention architecture) with a strict <strong>FIGO Rule-Based Engine</strong>, the software aims to minimize inter-observer variability and enhance diagnostic consistency.</p>
    </div>
    <div class="info-card">
        <h4>Clinical Utility</h4>
        <p>Early and accurate detection of abnormal CTG tracings is essential for anticipating risks such as fetal hypoxia or acidosis. This application functions as a clinical decision support system, providing an automated "second opinion" to help obstetricians, nurses, and midwives make timely and safe clinical interventions.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Load Model ──────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
SEQ_LEN  = 120

@st.cache_resource
def load_model():
    model_path = 'models/binary_3600_advanced_final.keras'
    if not os.path.exists(model_path):
        os.makedirs('models', exist_ok=True)
        file_id = '1o6hEOLuEeQIK2HGURUP6ZOZmpaKIhdJB'
        url = f'https://drive.google.com/uc?id={file_id}'
        with st.spinner("Downloading model weights from remote storage. This may take a moment..."):
            gdown.download(url, model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

try:
    with st.spinner("Initialising model weights..."):
        model = load_model()
    st.success("Model initialised successfully.")
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
                    st.success(f"60-minute segment extracted successfully — {len(fhr_seg)} samples at 1 Hz.")

                    st.markdown('<hr>', unsafe_allow_html=True)
                    st.markdown('<div class="section-label">Signal Trace — 60-minute Overview</div>', unsafe_allow_html=True)

                    t = np.arange(len(fhr_seg)) / 60.0

                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 4.5), sharex=True)
                    fig.patch.set_facecolor('#ffffff')
                    for ax in (ax1, ax2):
                        ax.set_facecolor('#ffffff')
                        ax.tick_params(colors='#475569', labelsize=8)
                        for sp in ax.spines.values():
                            sp.set_color('#cbd5e1')

                    ax1.plot(t, fhr_seg, color='#0ea5e9', linewidth=0.8)
                    ax1.set_ylabel("FHR (bpm)", color='#334155', fontsize=8, labelpad=8)
                    ax1.set_ylim(bottom=50)
                    ax1.axhline(110, color='#10b981', linewidth=0.6, linestyle='--', alpha=0.7, label='Normal range (110–160 bpm)')
                    ax1.axhline(160, color='#10b981', linewidth=0.6, linestyle='--', alpha=0.7)
                    ax1.legend(fontsize=7, labelcolor='#475569', framealpha=0, loc='upper right')

                    ax2.plot(t, uc_seg, color='#ec4899', linewidth=0.8)
                    ax2.set_ylabel("UC", color='#334155', fontsize=8, labelpad=8)
                    ax2.set_xlabel("Time (minutes)", color='#475569', fontsize=8)

                    plt.tight_layout(pad=1.0)
                    st.pyplot(fig)
                    plt.close(fig)

                    # Feature extraction
                    features      = ctg_utils.extract_features(fhr_seg, uc_seg)
                    figo_class, figo_explanation = ctg_utils.figo_classify_and_explain(features)

                    # Model inference
                    fhr_w = np.nan_to_num(fhr_seg[-SEQ_LEN:], nan=0.0).astype(np.float32)
                    img_input = np.expand_dims(sequence_to_image(fhr_w), axis=0)
                    seq_input = np.expand_dims(fhr_w.reshape(SEQ_LEN, 1), axis=0)

                    with st.spinner("Running ResNet50-Hybrid inference..."):
                        pred_prob  = float(model.predict([img_input, seq_input], verbose=0)[0][0])
                    dl_class = "Abnormal" if pred_prob > 0.5 else "Normal"

                    # ── Results layout ────────────────────────────────────────
                    st.markdown('<hr>', unsafe_allow_html=True)
                    st.markdown('<div class="section-label">Analysis Results</div>', unsafe_allow_html=True)

                    col_dl, col_sep, col_figo = st.columns([5, 0.4, 5])

                    with col_dl:
                        st.markdown("**Deep Learning Assessment** — ResNet50 Hybrid Attention")
                        if dl_class == "Normal":
                            confidence = 1 - pred_prob
                            st.markdown(f"""
                            <div class="result-normal">
                                <div class="result-label" style="color:#15803d;">DL Classification</div>
                                <div class="result-value" style="color:#15803d;">Normal</div>
                                <div class="result-conf">Model confidence: {confidence:.1%}</div>
                            </div>""", unsafe_allow_html=True)
                        else:
                            confidence = pred_prob
                            st.markdown(f"""
                            <div class="result-abnormal">
                                <div class="result-label" style="color:#b91c1c;">DL Classification</div>
                                <div class="result-value" style="color:#b91c1c;">Abnormal</div>
                                <div class="result-conf">Model confidence: {confidence:.1%}</div>
                            </div>""", unsafe_allow_html=True)

                        st.markdown("**Extracted Signal Metrics**")
                        st.markdown(f"""
                        <div class="metric-row">
                            <div class="metric-card">
                                <div class="value">{features['LB']:.0f}</div>
                                <div class="label">Baseline FHR (bpm)</div>
                            </div>
                            <div class="metric-card">
                                <div class="value">{features['MSTV']:.2f}</div>
                                <div class="label">MSTV (Variability)</div>
                            </div>
                            <div class="metric-card">
                                <div class="value">{features['UC_rate']:.1f}</div>
                                <div class="label">UC / 10 min</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_sep:
                        st.markdown('<div style="border-left:1px solid #1e3a5f;height:100%;margin:0 auto;width:1px;"></div>', unsafe_allow_html=True)

                    with col_figo:
                        st.markdown("**FIGO Rule-Based Assessment** — Clinical Guideline Engine")

                        if figo_class == "Normal":
                            st.markdown(f"""
                            <div class="result-normal">
                                <div class="result-label" style="color:#15803d;">FIGO Classification</div>
                                <div class="result-value" style="color:#15803d;">Normal</div>
                            </div>""", unsafe_allow_html=True)
                        elif figo_class == "Suspect":
                            st.markdown(f"""
                            <div class="result-suspect">
                                <div class="result-label" style="color:#b45309;">FIGO Classification</div>
                                <div class="result-value" style="color:#b45309;">Suspect</div>
                            </div>""", unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-abnormal">
                                <div class="result-label" style="color:#b91c1c;">FIGO Classification</div>
                                <div class="result-value" style="color:#b91c1c;">Pathologic</div>
                            </div>""", unsafe_allow_html=True)

                        st.markdown(f'<div class="clinical-note"><strong>Clinical note:</strong> {figo_explanation}</div>', unsafe_allow_html=True)

                        feat_df = pd.DataFrame({
                            "Feature"        : ["Accelerations", "Light Decelerations", "Severe Decelerations", "Prolonged Decelerations"],
                            "Count (60 min)" : [features['AC'],  features['DL'],         features['DS'],         features['DP']]
                        })
                        st.dataframe(feat_df, use_container_width=True, hide_index=True)

                    # Disclaimer
                    st.markdown('<hr>', unsafe_allow_html=True)
                    st.markdown(
                        '<p style="font-size:0.72rem;color:#4a6a8a;text-align:center;letter-spacing:0.04em;">'
                        'RESEARCH USE ONLY — This system is not a certified medical device. '
                        'All outputs must be reviewed and validated by a qualified clinician before any clinical decision is made.'
                        '</p>',
                        unsafe_allow_html=True
                    )

    else:
        st.warning("Please upload both a .hea and a .dat file for the same record.")
elif uploaded_files:
    st.warning("Please upload both a .hea and a .dat file for the same record.")
