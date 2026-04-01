import streamlit as st
import tempfile
import os
import tensorflow as tf
import numpy as np
import ctg_utils

st.set_page_config(page_title="CTG Classification Engine", layout="wide")
st.title("🫀 Intrapartum CTG Analysis Dashboard")
st.markdown("Upload your Fetal Heart Rate `.dat` and `.hea` WFDB files below. Our model will preprocess the files, run a ResNet50-based classification, and provide clinical Rule-Based explanations.")

import gdown

@st.cache_resource
def load_model():
    model_path = 'models/binary_3600_advanced_final.keras'
    
    if not os.path.exists(model_path):
        st.info("Downloading deep learning model weights from Google Drive...")
        os.makedirs('models', exist_ok=True)
        # You successfully replaced the ID!
        file_id = '1o6hEOLuEeQIK2HGURUP6ZOZmpaKIhdJB'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_path, quiet=False)
        
    return tf.keras.models.load_model(model_path)

try:
    with st.spinner("Loading Deep Learning Model weights..."):
        model = load_model()
except Exception as e:
    st.error(f"Error loading model: ensure models/binary_3600_advanced_final.keras exists. Error: {e}")
    st.stop()

st.markdown("---")
st.markdown("### 📂 Upload Patient File")
uploaded_files = st.file_uploader("Select BOTH the `.hea` and `.dat` files for a given record", accept_multiple_files=True, type=['hea', 'dat'])

if uploaded_files and len(uploaded_files) >= 2:
    hea_file = next((f for f in uploaded_files if f.name.endswith('.hea')), None)
    dat_file = next((f for f in uploaded_files if f.name.endswith('.dat')), None)
    
    if hea_file and dat_file:
        record_name = hea_file.name.replace('.hea', '')
        if record_name != dat_file.name.replace('.dat', ''):
            st.error("The `.hea` and `.dat` files must match and have the exact same name!")
        else:
            with tempfile.TemporaryDirectory() as tmpdirname:
                # 1. Write files to tempdir so wfdb can read them from disk
                with open(os.path.join(tmpdirname, hea_file.name), 'wb') as f:
                    f.write(hea_file.getvalue())
                with open(os.path.join(tmpdirname, dat_file.name), 'wb') as f:
                    f.write(dat_file.getvalue())
                    
                record_path = os.path.join(tmpdirname, record_name)
                
                st.info(f"Processing Record: **{record_name}**")
                
                with st.spinner("Extracting & Preprocessing 60-min Signal..."):
                    # 2. Extract and Preprocess (1Hz, Interp, 3600 steps)
                    fhr_seg, uc_seg = ctg_utils.prepare_signal_from_record(record_path)
                    
                    if fhr_seg is None:
                        st.error("Failed to extract a valid 60-minute continuous segment (gaps > 10 mins).")
                    else:
                        st.success("Successfully extracted and interpolated 60-minute continuous window (3600 steps).")
                        
                        # 3. Extract Features & Explain (Rule-Based)
                        features = ctg_utils.extract_features(fhr_seg, uc_seg)
                        figo_class, figo_explanation = ctg_utils.figo_classify_and_explain(features)
                        
                        # 4. Model Inference (ResNet50 Hybrid)
                        # Extract the last 600 timesteps (10 minutes) for the model
                        fhr_600 = fhr_seg[-600:]
                        uc_600 = uc_seg[-600:]
                        
                        # Generate Image branch input
                        img_input = ctg_utils.window_to_image(fhr_600, uc_600)
                        img_input = np.expand_dims(img_input, axis=0) # (1, 224, 224, 3)
                        
                        # Generate Sequence branch input (600, 2)
                        seq_input = np.stack((fhr_600, uc_600), axis=-1)
                        seq_input = np.nan_to_num(seq_input, nan=0.0) 
                        seq_input = np.expand_dims(seq_input, axis=0) # (1, 600, 2)
                        
                        pred_prob = model.predict([img_input, seq_input])[0][0]
                        pred_class = "Abnormal / Pathologic" if pred_prob > 0.5 else "Normal"
                        
                        # 5. Show Beautiful Output
                        st.markdown("### 🧬 Analysis Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 1. Deep Learning Prediction (ResNet50)")
                            if pred_class == "Abnormal / Pathologic":
                                st.error(f"**Diagnostic Status:** {pred_class} \n\n**Confidence:** {pred_prob:.2%}")
                            else:
                                st.success(f"**Diagnostic Status:** {pred_class} \n\n**Confidence:** {1-pred_prob:.2%}")
                            
                            sub1, sub2 = st.columns(2)
                            sub1.metric("Base FHR", f"{features['LB']:.1f} bpm")
                            sub2.metric("Contractions / 10min", f"{features['UC_rate']:.1f}")
                            
                        with col2:
                            st.markdown("#### 2. Clinical Rule-Based Explanation")
                            if figo_class == 'Normal':
                                st.success(f"**Rule Classification:** {figo_class}")
                            elif figo_class == 'Suspect':
                                st.warning(f"**Rule Classification:** {figo_class}")
                            else:
                                st.error(f"**Rule Classification:** {figo_class}")
                            
                            st.info(f"**Clinical Note:** {figo_explanation}")
                            st.write(f"- **MSTV (Variability)**: {features['MSTV']:.2f}")
                            st.write(f"- **Severe Decelerations**: {features['DS']}")
                            st.write(f"- **Prolonged Decelerations**: {features['DP']}")
                            st.write(f"- **Light Decelerations**: {features['DL']}")
                            st.write(f"- **Accelerations**: {features['AC']}")

    else:
        st.warning("Please make sure you have uploaded BOTH a .hea and .dat file.")
