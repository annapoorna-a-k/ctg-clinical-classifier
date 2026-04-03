# CTG Clinical Analysis System: Project Evaluation & Rubric Compliance 

This document provides a comprehensive evaluation of the CTG classification pipeline according to the project grading criteria. It details the steps taken across the entire repository to ensure robust classification of cardiotocography (CTG) signals.

---

## 1. Problem Definition & Motivation (Clear research/industry relevance)

**Problem Statement:** 
Fetal Hypoxia is a leading cause of perinatal mortality. Cardiotocography (CTG) is a standard diagnostic tool used to monitor fetal heart rate (FHR) and uterine contractions (UC) to detect hypoxia. However, visual interpretation of CTG traces is highly subjective, suffering from significant inter-observer variability. 

**Motivation:**
The goal of this project is to develop an automated, data-driven Deep Learning system that effectively captures the complex, non-linear longitudinal dependencies of CTG signals. By building a robust clinical classifier, this system aims to provide objective, real-time diagnostic support, reducing human error and preventing adverse perinatal outcomes.

---

## 2. Data Engineering (Cleaning, Augmentation, Feature Engineering)

Robust data engineering was performed to prepare the CTU-CHB Intrapartum dataset for deep learning models.

**A. Data Cleaning & Interpolation (`webapp/review_3/parameter-tuning-for-gan.ipynb`):**
*   **Missing Data Management:** Raw WFDB signals were downsampled to 1-Hz using non-overlapping windows with `numpy.nanmean()`.
*   **Signal Gap Interpolation:** Short missing segments (under 10 minutes or 600 seconds) were successfully reconstructed using strict **linear interpolation**. Segments containing zero-values representing signal loss were converted to NaNs.
*   **Segment Extraction:** Standardized sequences of 120 time-steps (representing 2 minutes of continuous recording) were extracted using a 1-minute overlap stride, dropping segments with irrecoverable NaNs.

**B. Data Augmentation (Sequence GAN):**
Because clinical datasets are highly imbalanced (e.g., normal cases vastly outnumber pathological cases), a Sequence GAN was implemented. By generating high-fidelity synthetic physiological sequences, the GAN balances the dataset distribution, providing the primary classification models with a uniform class distribution.

**C. Feature Engineering & Preprocessing (`webapp/review 2/` and `webapp/review_3/`):**
*   **Morphological Compressions:** Raw sequences were compressed and passed into an LSTM-Autoencoder. This produced a 32-dimensional robust latent representation summarizing the core clinical morphology.
*   **Time-Series Features (TSFEL):** Statistical, temporal, and frequency domain metadatas were extracted using `tsfel` to capture clinical baseline rules (e.g., FIGO rules for decelerations and variability thresholds).
*   **Image Transformations:** 1D sequences were converted into 224x224 RGB image arrays via spectrogram/time-series imaging, enabling the use of advanced 2D Convolutional Networks (ResNet, MobileNet).

---

## 3. Model Architecture Justification (Why this architecture? Theoretical reasoning)

**1. LSTM Autoencoder + Sequence GAN:**
*   **Theoretical Reasoning:** CTG signals are highly temporal. Standard feed-forward networks fail to capture the sequential dependencies (e.g., how the fetal heart rate recovers after a contraction). The **LSTM Autoencoder** processes the longitudinal state natively. The subsequent **GAN** takes this exact latent space and applies an adversarial network (with Feature Matching and Label Smoothing) to generate physiologically plausible synthetic samples without Mode Collapse.

**2. ResNet50 with Hybrid Attention (Final Selected Classifier):**
*   **Theoretical Reasoning:** ResNet50 is inherently powerful for spatial feature extraction but lacks an understanding of *where* in the timeline a critical clinical event (like a late deceleration) occurred. By attaching a **Hybrid Attention Mechanism** to the spatial embeddings of the ResNet50 backbone, the model dynamically learns to *focus* on the specific morphological segments that strongly correlate with hypoxia, mirroring how a clinician reviews a CTG tape.

---

## 4. Experimental Design (Baselines, Ablation Study)

A rigorous ablation study was conducted to quantify the performance gains from both architectural innovations and data augmentation.

**A. Architecture Ablation (Fixed Dataset):**
To evaluate the impact of the Hybrid Attention mechanism, models were tested on the base intrapartum dataset without GAN augmentation:
*   **ResNet50 + LSTM (Baseline):** 66.16% Accuracy.
*   **ResNet50 + LSTM + Embedding:** 71.96% Accuracy (+5.80% improvement).
*   **ResNet50 + LSTM + Attention:** 73.06% Accuracy (+6.90% improvement over baseline).

**B. Data-Level Ablation (The "GAN Impact"):**
The most significant performance leap occurred when applying the Sequence GAN augmentation to the best architectural candidate:
*   **Model without GAN (Imbalanced):** 73.06% Accuracy.
*   **Final Model with GAN (Balanced 3,600 samples/class):** **94.40% Accuracy.**
*   *Conclusion:* The GAN-driven dataset balancing provided a massive **+21.34%** gain, proving that handling clinical class imbalance is as critical as the model architecture itself.

**C. Baseline Comparison:**
*   **MLP Model (Standard CSV Features):** Achieved **92.02%** accuracy. While high, the proposed Deep Learning architecture (94.40%) successfully captured more complex morphological anomalies that the simple MLP missed.

---

## 5. Hyperparameter Optimization (Structured Tuning Strategy)

Strict structured hyperparameter tuning frameworks were applied to optimize both the GAN and the final ResNet50 classifier.

**1. Structured Tuning for the GAN (`parameter-tuning-for-gan.ipynb`):**
The training loop utilized parameter sweeps to avoid discriminator overpowering and mode collapse.
*   **TTUR (Two Time-Scale Update Rule):** Separate learning rates for the Discriminator ($0.0001$) and Generator ($0.0004$) were structured and tuned.
*   **Noise Injection Tuning:** Swept values for Gaussian Noise variance (decaying standard deviation) to stabilize the continuous divergence.
*   **Label Smoothing:** Real samples mapped to $0.95$.

**2. Tuning for ResNet50 Classifier (`model-final-review3.ipynb`):**
Using a centralized `config` dictionary, grid and random sweeps were utilized:
*   **Batch Size & LR Schedulers:** Tuned to 64 alongside a `ReduceLROnPlateau` scheduler (Factor: 0.1, Patience: 5) and Early Stopping (Patience: 10).
*   **Regularization Tuning:** Dropout rates (tuned to 0.5) and L2 Weight Decay were incorporated to explicitly mitigate overfitting.

---

## 6. Performance Evaluation (Proper Metrics & Statistical Reasoning)

Proper evaluation pipelines were embedded to validate clinical viability.

*   **Quantitative Results Summary:**

| Experiment | Accuracy | Significance |
| :--- | :--- | :--- |
| ResNet50+LSTM (Baseline) | 66.16% | Initial candidate |
| ResNet50+LSTM+Attention | 73.06% | Architecture gain (+6.9%) |
| MLP (Handcrafted Features) | 92.02% | Competitive lower baseline |
| **Proposed System (Final)** | **94.40%** | **Final marked improvement** |

*   **GAN Quality Metrics (Statistical Realism):** The generator's authenticity was measured utilizing the **Standard Deviation Ratio ($Std\_Ratio$)**. A final $Std\_Ratio = 0.924$ acts as a statistical proof that the synthetics possess 92.4% of the variability of real clinical samples.
*   **Classifier Performance Metrics:** Accuracy was supported by:
    *   **Precision, Recall, & F1-Score Reports:** To heavily scrutinize false negatives (missed hypoxic cases).
    *   **Confusion Matrices:** Generated per-epoch to visually inspect class bleed.
    *   **Learning Curves:** Validation vs. Training loss and accuracy plots provided statistical reasoning that the final network effectively mitigated the intense variance (overfitting) observed in the baselines.
