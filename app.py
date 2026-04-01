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
}

/* Page background */
.stApp {
    background-color: #0a0f1e;
    color: #e2e8f0;
}

/* Top header band */
.header-band {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2e4a 50%, #0d2137 100%);
    border-bottom: 1px solid #1e3a5f;
    padding: 2.5rem 3rem 2rem 3rem;
    margin: -1rem -1rem 2rem -1rem;
}
.header-band h1 {
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: #e2f0ff;
    margin: 0 0 0.4rem 0;
    text-transform: uppercase;
}
.header-band .subtitle {
    font-size: 0.85rem;
    color: #7fafd4;
    letter-spacing: 0.06em;
    font-weight: 400;
    text-transform: uppercase;
}

/* Info section */
.info-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.25rem;
    margin-bottom: 2rem;
}
.info-card {
    background: #0d1b2a;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #2d7dd2;
    border-radius: 4px;
    padding: 1.2rem 1.4rem;
}
.info-card h4 {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    color: #5b9bd5;
    text-transform: uppercase;
    margin: 0 0 0.5rem 0;
}
.info-card p {
    font-size: 0.82rem;
    color: #94a3b8;
    line-height: 1.6;
    margin: 0;
}

/* Section headers */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    color: #5b9bd5;
    text-transform: uppercase;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* Upload zone styling override */
.upload-zone {
    background: #0d1b2a;
    border: 1px dashed #2d4a6b;
    border-radius: 6px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}

/* Metric card */
.metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 1rem;
}
.metric-card {
    background: #0d1b2a;
    border: 1px solid #1e3a5f;
    border-radius: 4px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-card .value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #63b3ed;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.metric-card .label {
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    color: #4a7fab;
    text-transform: uppercase;
}

/* Result cards */
.result-normal {
    background: #051a14;
    border: 1px solid #1a4731;
    border-left: 4px solid #22c55e;
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.result-abnormal {
    background: #1a0d0d;
    border: 1px solid #4a1919;
    border-left: 4px solid #ef4444;
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.result-suspect {
    background: #1a1508;
    border: 1px solid #4a3c10;
    border-left: 4px solid #f59e0b;
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.result-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.result-value {
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 0.03em;
}
.result-conf {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 0.2rem;
}

/* Clinical note */
.clinical-note {
    background: #0d1b2a;
    border: 1px solid #1e3a5f;
    border-radius: 4px;
    padding: 1rem 1.2rem;
    font-size: 0.82rem;
    color: #94a3b8;
    line-height: 1.6;
    margin-bottom: 1rem;
}

/* Table */
.stDataFrame { font-size: 0.8rem !important; }

/* Spinner */
.stSpinner { color: #2d7dd2 !important; }

/* Divider */
hr { border-color: #1e3a5f !important; }

/* Streamlit block tweaks */
div[data-testid="stFileUploader"] {
    background: #0d1b2a !important;
    border: 1px dashed #2d4a6b !important;
    border-radius: 6px;
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

# ─── About CTG section ────────────────────────────────────────────────────────
st.markdown("""
<div class="info-grid">
    <div class="info-card">
        <h4>What is CTG?</h4>
        <p>Cardiotocography (CTG) is the simultaneous electronic recording of the fetal heart rate (FHR) and uterine contractions (UC). It is the primary clinical tool for assessing fetal well-being during labour and high-risk pregnancies.</p>
    </div>
    <div class="info-card">
        <h4>Clinical Significance</h4>
        <p>Abnormal CTG patterns are associated with fetal hypoxia and acidosis. Early detection of pathological tracings allows timely clinical intervention — reducing perinatal morbidity and mortality. Misinterpretation remains a leading cause of adverse birth outcomes.</p>
    </div>
    <div class="info-card">
        <h4>How This System Works</h4>
        <p>The platform fuses a <strong>deep learning model</strong> (ResNet50 + Bidirectional LSTM + Attention, trained on the CTU-CHB intrapartum database) with a <strong>FIGO rule-based engine</strong> to produce dual, interpretable clinical outputs.</p>
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
                    fig.patch.set_facecolor('#0d1b2a')
                    for ax in (ax1, ax2):
                        ax.set_facecolor('#0d1b2a')
                        ax.tick_params(colors='#94a3b8', labelsize=8)
                        for sp in ax.spines.values():
                            sp.set_color('#1e3a5f')

                    ax1.plot(t, fhr_seg, color='#63b3ed', linewidth=0.7)
                    ax1.set_ylabel("FHR (bpm)", color='#7fafd4', fontsize=8, labelpad=8)
                    ax1.set_ylim(bottom=50)
                    ax1.axhline(110, color='#22c55e', linewidth=0.4, linestyle='--', alpha=0.5, label='Normal range (110–160 bpm)')
                    ax1.axhline(160, color='#22c55e', linewidth=0.4, linestyle='--', alpha=0.5)
                    ax1.legend(fontsize=7, labelcolor='#94a3b8', framealpha=0, loc='upper right')

                    ax2.plot(t, uc_seg, color='#f97316', linewidth=0.7)
                    ax2.set_ylabel("UC", color='#7fafd4', fontsize=8, labelpad=8)
                    ax2.set_xlabel("Time (minutes)", color='#94a3b8', fontsize=8)

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
                                <div class="result-label" style="color:#22c55e;">DL Classification</div>
                                <div class="result-value" style="color:#22c55e;">Normal</div>
                                <div class="result-conf">Model confidence: {confidence:.1%}</div>
                            </div>""", unsafe_allow_html=True)
                        else:
                            confidence = pred_prob
                            st.markdown(f"""
                            <div class="result-abnormal">
                                <div class="result-label" style="color:#ef4444;">DL Classification</div>
                                <div class="result-value" style="color:#ef4444;">Abnormal</div>
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
                                <div class="result-label" style="color:#22c55e;">FIGO Classification</div>
                                <div class="result-value" style="color:#22c55e;">Normal</div>
                            </div>""", unsafe_allow_html=True)
                        elif figo_class == "Suspect":
                            st.markdown(f"""
                            <div class="result-suspect">
                                <div class="result-label" style="color:#f59e0b;">FIGO Classification</div>
                                <div class="result-value" style="color:#f59e0b;">Suspect</div>
                            </div>""", unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-abnormal">
                                <div class="result-label" style="color:#ef4444;">FIGO Classification</div>
                                <div class="result-value" style="color:#ef4444;">Pathologic</div>
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
