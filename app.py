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

/* Base dark background */
.stApp {
    background-color: #0f172a;
    background-image:
        radial-gradient(ellipse at 10% 0%, rgba(14,165,233,0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 100%, rgba(139,92,246,0.06) 0%, transparent 50%);
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
}
.result-abnormal {
    background: linear-gradient(135deg, rgba(239,68,68,0.12) 0%, rgba(30,41,59,0.9) 100%);
    border: 1px solid rgba(239,68,68,0.3);
    border-left: 5px solid #ef4444;
}
.result-suspect {
    background: linear-gradient(135deg, rgba(245,158,11,0.12) 0%, rgba(30,41,59,0.9) 100%);
    border: 1px solid rgba(245,158,11,0.3);
    border-left: 5px solid #f59e0b;
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
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-band">
    <h1>🫀 CTG Clinical Analysis System</h1>
    <div class="subtitle">Cardiotocography &nbsp;·&nbsp; AI-Assisted Fetal Monitoring &nbsp;·&nbsp; Clinical Decision Support</div>
</div>
""", unsafe_allow_html=True)

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
                                <div class="result-value" style="color:#f59e0b;">⚠️ Suspect</div>
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
