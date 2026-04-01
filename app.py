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

st.set_page_config(page_title="CTG Classification Engine", layout="wide", page_icon="🫀")

# ─── Header ──────────────────────────────────────────────────────────────────
st.title("🫀 Intrapartum CTG Analysis Dashboard")
st.markdown(
    "Upload your Fetal Heart Rate `.dat` and `.hea` WFDB files below. "
    "Our model will preprocess the signal, run a **ResNet50 Hybrid** deep-learning "
    "classification, and provide **FIGO rule-based** clinical explanations."
)
st.markdown("---")

# ─── Load Model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = 'models/binary_3600_advanced_final.keras'
    if not os.path.exists(model_path):
        st.info("⬇️ Downloading deep learning model weights from Google Drive…")
        os.makedirs('models', exist_ok=True)
        file_id = '1o6hEOLuEeQIK2HGURUP6ZOZmpaKIhdJB'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

try:
    with st.spinner("Loading Deep Learning Model weights…"):
        model = load_model()
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.markdown("---")

# ─── Signal-to-image helper (matches training pipeline exactly) ──────────────
IMG_SIZE = (224, 224)
SEQ_LEN  = 120          # ← model was trained on 120-sample FHR windows at 1 Hz (2 min)

def sequence_to_image(seq):
    """Converts a 1-D FHR sequence (length = SEQ_LEN) to a 224×224 RGB image
    using the same cv2 polyline approach used during training."""
    img = np.ones((224, 224, 3), dtype=np.uint8) * 255
    pad = 8
    w = 224 - 2 * pad
    h = 224 - 2 * pad

    seq_min, seq_max = seq.min(), seq.max()
    if seq_max > seq_min:
        y_scaled = (seq - seq_min) / (seq_max - seq_min)
    else:
        y_scaled = np.zeros_like(seq) + 0.5

    y_coords = 224 - pad - (y_scaled * h).astype(np.int32)
    x_coords = np.linspace(pad, 224 - pad, len(seq)).astype(np.int32)
    pts = np.column_stack((x_coords, y_coords)).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=False, color=(255, 0, 0), thickness=2,
                  lineType=cv2.LINE_AA)
    img_resized = cv2.resize(img, IMG_SIZE)
    return (img_resized / 255.0).astype(np.float32)

# ─── File Upload ─────────────────────────────────────────────────────────────
st.markdown("### 📂 Upload Patient Record")
st.markdown("Select **both** the `.hea` header file and the `.dat` signal file for the same record.")

uploaded_files = st.file_uploader(
    "Drop WFDB files here",
    accept_multiple_files=True,
    type=['hea', 'dat']
)

if uploaded_files and len(uploaded_files) >= 2:
    hea_file = next((f for f in uploaded_files if f.name.endswith('.hea')), None)
    dat_file = next((f for f in uploaded_files if f.name.endswith('.dat')), None)

    if hea_file and dat_file:
        record_name = hea_file.name.replace('.hea', '')
        if record_name != dat_file.name.replace('.dat', ''):
            st.error("❌ The `.hea` and `.dat` files must have the **exact same name**!")
        else:
            with tempfile.TemporaryDirectory() as tmpdirname:
                # Write uploaded bytes to disk so wfdb can read them
                for uf in [hea_file, dat_file]:
                    with open(os.path.join(tmpdirname, uf.name), 'wb') as f:
                        f.write(uf.getvalue())

                record_path = os.path.join(tmpdirname, record_name)
                st.info(f"🔬 Processing record: **{record_name}**")

                with st.spinner("Extracting & Preprocessing 60-min signal at 1 Hz…"):
                    fhr_seg, uc_seg = ctg_utils.prepare_signal_from_record(record_path)

                if fhr_seg is None:
                    st.error("❌ Could not extract a valid 60-minute continuous segment (large gaps > 10 min found).")
                else:
                    st.success("✅ 60-minute continuous window extracted and interpolated (3 600 timesteps).")

                    # ── Waveform Chart ────────────────────────────────────────
                    st.markdown("### 📈 Signal Overview (1 Hz)")
                    t = np.arange(len(fhr_seg)) / 60.0  # minutes axis

                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
                    fig.patch.set_facecolor('#0E1117')
                    for ax in (ax1, ax2):
                        ax.set_facecolor('#0E1117')
                        ax.tick_params(colors='#FAFAFA')
                        ax.spines[:].set_color('#333333')

                    ax1.plot(t, fhr_seg, color='#3D9DF3', linewidth=0.6, label='FHR (bpm)')
                    ax1.set_ylabel("FHR (bpm)", color='#FAFAFA')
                    ax1.legend(loc='upper right', labelcolor='#FAFAFA', framealpha=0)

                    ax2.plot(t, uc_seg, color='#F25757', linewidth=0.6, label='UC')
                    ax2.set_ylabel("UC", color='#FAFAFA')
                    ax2.set_xlabel("Time (min)", color='#FAFAFA')
                    ax2.legend(loc='upper right', labelcolor='#FAFAFA', framealpha=0)

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    # ── Feature Extraction (Rule-Based) ───────────────────────
                    features = ctg_utils.extract_features(fhr_seg, uc_seg)
                    figo_class, figo_explanation = ctg_utils.figo_classify_and_explain(features)

                    # ── Model Inference ───────────────────────────────────────
                    # The model was trained on 120-sample (2-minute) FHR windows.
                    # We take the last 120 timesteps of the 3 600-step FHR segment,
                    # apply the same cv2 polyline image renderer used at training time,
                    # and reshape the sequence to (120, 1) as expected by the LSTM branch.
                    fhr_window = np.nan_to_num(fhr_seg[-SEQ_LEN:], nan=0.0).astype(np.float32)

                    # Image branch  (1, 224, 224, 3)
                    img_arr   = sequence_to_image(fhr_window)
                    img_input = np.expand_dims(img_arr, axis=0)

                    # Sequence branch  (1, 120, 1)
                    seq_arr   = fhr_window.reshape(SEQ_LEN, 1)
                    seq_input = np.expand_dims(seq_arr, axis=0)

                    with st.spinner("Running ResNet50 inference…"):
                        pred_prob = float(model.predict([img_input, seq_input], verbose=0)[0][0])
                    pred_class = "Abnormal / Pathologic" if pred_prob > 0.5 else "Normal"

                    # ── Results Panel ─────────────────────────────────────────
                    st.markdown("---")
                    st.markdown("### 🧬 Analysis Results")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### 🤖 Deep Learning (ResNet50 Hybrid)")
                        if pred_class == "Normal":
                            st.success(f"**Prediction:** {pred_class}  \n**Confidence:** {(1 - pred_prob):.1%}")
                        else:
                            st.error(f"**Prediction:** {pred_class}  \n**Confidence:** {pred_prob:.1%}")

                        st.markdown("##### Key Signal Metrics")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Baseline FHR", f"{features['LB']:.0f} bpm")
                        m2.metric("MSTV", f"{features['MSTV']:.2f}")
                        m3.metric("UC / 10 min", f"{features['UC_rate']:.1f}")

                    with col2:
                        st.markdown("#### 📋 Clinical Rule-Based (FIGO)")
                        if figo_class == 'Normal':
                            st.success(f"**Classification:** {figo_class}")
                        elif figo_class == 'Suspect':
                            st.warning(f"**Classification:** {figo_class}")
                        else:
                            st.error(f"**Classification:** {figo_class}")

                        st.info(f"💬 {figo_explanation}")

                        feat_df = pd.DataFrame({
                            "Feature": ["Accelerations", "Light Decels", "Severe Decels", "Prolonged Decels"],
                            "Count":   [features['AC'], features['DL'], features['DS'],    features['DP']]
                        })
                        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    else:
        st.warning("⚠️ Please upload **both** a `.hea` and a `.dat` file together.")
elif uploaded_files:
    st.warning("⚠️ Please upload **both** a `.hea` and a `.dat` file together.")
