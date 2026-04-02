import numpy as np
import pandas as pd
import wfdb
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')   # headless backend for Streamlit Cloud
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import warnings
warnings.filterwarnings('ignore')

IMG_SIZE = 224

def window_to_image(fhr_w, uc_w):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(2.24, 2.24), dpi=100)
    fig.patch.set_facecolor('white')
    ax1.plot(fhr_w, color='blue', linewidth=0.8)
    ax1.axis('off')
    ax2.plot(uc_w,  color='red',  linewidth=0.8)
    ax2.axis('off')
    plt.tight_layout(pad=0)
    fig.canvas.draw()

    buf = np.asarray(fig.canvas.buffer_rgba())  # RGBA
    img = buf[:, :, :3]                         # RGB
    plt.close(fig)

    img = PILImage.fromarray(img).resize((IMG_SIZE, IMG_SIZE))
    return np.array(img, dtype=np.float32) / 255.0

def prepare_signal_from_record(record_path):
    """
    Reads a WFDB record directly from a path.
    Prepares the FHR and UC signals according to the DeepCTG pipeline:
    1. 1Hz Downsampling.
    2. Linear interpolation for missing gaps < 10 mins.
    3. Finds the latest valid continuous segment >= 10 mins.
    4. Extracts exactly 60 minutes (3600s).
    """
    record = wfdb.rdrecord(record_path)
    fhr = record.p_signal[:, 0].astype(float)
    uc  = record.p_signal[:, 1].astype(float)
    fs  = int(record.fs)
    
    fhr[fhr == 0] = np.nan
    uc[uc   == 0] = np.nan
    
    n = (len(fhr) // fs) * fs
    if n == 0:
        return None, None
        
    fhr_1hz = np.nanmean(fhr[:n].reshape(-1, fs), axis=1)
    uc_1hz  = np.nanmean(uc[:n].reshape(-1,  fs), axis=1)
    
    def interpolate_small_gaps(sig, max_gap=600):
        s = pd.Series(sig)
        mask = s.isna()
        groups = mask.ne(mask.shift()).cumsum()
        gap_sizes = mask.groupby(groups).transform('size')
        large_gaps_mask = mask & (gap_sizes > max_gap)
        
        s_interp = s.interpolate(method='linear', limit_direction='both')
        s_interp[large_gaps_mask] = np.nan
        return s_interp.values

    fhr_1hz = interpolate_small_gaps(fhr_1hz)
    uc_1hz  = interpolate_small_gaps(uc_1hz)
    
    is_valid = ~np.isnan(fhr_1hz)
    edges = np.diff(np.concatenate(([0], is_valid.view(np.int8), [0])))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    
    valid_segments = [(s, e) for s, e in zip(starts, ends) if (e - s) >= 600]
    
    if not valid_segments:
        return None, None
        
    latest_s, latest_e = valid_segments[-1]
    
    if len(fhr_1hz) - latest_s > 5400:
        return None, None
        
    fhr_seg = fhr_1hz[latest_s:latest_e]
    uc_seg  = uc_1hz[latest_s:latest_e]
    
    if len(fhr_seg) > 3600:
        fhr_seg = fhr_seg[-3600:]
        uc_seg  = uc_seg[-3600:]
    elif len(fhr_seg) < 3600:
        pad_len = 3600 - len(fhr_seg)
        fhr_seg = np.pad(fhr_seg, (pad_len, 0), 'constant', constant_values=np.nan)
        uc_seg  = np.pad(uc_seg, (pad_len, 0), 'constant', constant_values=np.nan)
        
    return fhr_seg, uc_seg


def estimate_baseline(fhr):
    baseline = np.full_like(fhr, np.nan)
    for i in range(len(fhr)):
        s = max(0, i - 300)
        e = min(len(fhr), i + 300)
        seg = fhr[s:e]
        seg = seg[~np.isnan(seg)]
        if len(seg) > 0:
            baseline[i] = np.median(seg)
    return baseline

def count_events(mask, min_dur):
    events, dur = 0, 0
    for v in mask:
        if v:
            dur += 1
        else:
            if dur >= min_dur:
                events += 1
            dur = 0
    if dur >= min_dur:
        events += 1
    return events


def extract_features(fhr, uc):
    """
    Extracts FIGO features from 1Hz FHR and UC arrays.
    """
    baseline = estimate_baseline(fhr)
    diff = fhr - baseline
    LB = float(np.nanmean(baseline))
    
    variability = np.nanmean(np.abs(diff))
    MSTV = float(variability)

    AC = count_events(diff >= 15, 15)

    DL, DS, DP = 0, 0, 0
    mask = diff <= -15
    in_event = False
    start = 0

    for i, val in enumerate(mask):
        if val and not in_event:
            in_event = True
            start = i
        elif not val and in_event:
            end = i
            dur = end - start
            if dur >= 15:
                depth = np.nanmin(diff[start:end])
                if dur >= 120:
                    DP += 1
                elif depth <= -45:
                    DS += 1
                else:
                    DL += 1
            in_event = False
            
    if in_event:
        end = len(mask)
        dur = end - start
        if dur >= 15:
            depth = np.nanmin(diff[start:end])
            if dur >= 120:
                DP += 1
            elif depth <= -45:
                DS += 1
            else:
                DL += 1

    uc_clean = np.where(np.isnan(uc), 0, uc)
    peaks, _ = find_peaks(uc_clean, distance=60)
    UC = len(peaks)
    
    duration_min = len(uc_clean) / 60
    UC_rate = float((UC / duration_min) * 10)

    return dict(
        LB=LB,
        MSTV=MSTV,
        AC=AC,
        DL=DL,
        DS=DS,
        DP=DP,
        UC=UC,
        UC_rate=UC_rate
    )

def figo_classify_and_explain(row):
    """
    Applies clinical rule extraction to generate detailed explanations.
    Returns 'Normal', 'Suspect', or 'Abnormal' and the explanation string.
    """
    LB   = row['LB']
    MSTV = row['MSTV']
    AC   = row['AC']
    DL   = row['DL']
    DS   = row['DS']
    DP   = row['DP']
    UC_rate = row['UC_rate']

    explanations = []

    # BASELINE
    if 110 <= LB <= 160:
        base = 'normal'
    elif 100 <= LB < 110 or 160 < LB <= 180:
        base = 'suspicious'
        explanations.append(f"Baseline FHR is deviating from normal ({LB:.1f} bpm), suggesting early signs of physiological stress.")
    else:
        base = 'abnormal'
        explanations.append(f"Baseline FHR is highly abnormal ({LB:.1f} bpm), a strong indicator of potential hypoxia or severe distress.")

    # VARIABILITY
    if MSTV < 1:
        var = 'abnormal'
        explanations.append(f"Heart rate variability is critically low (MSTV = {MSTV:.2f} bpm). This lack of fluctuation points to a potential depression in the central nervous system or significant acidemia.")
    elif MSTV < 2:
        var = 'suspicious'
        explanations.append(f"Heart rate variability is reduced (MSTV = {MSTV:.2f} bpm), warranting closer observation for potential fetal sleep state or emerging physiological stress.")
    else:
        var = 'normal'

    # DECELERATIONS
    if DP > 0 or DS >= 5:
        dec = 'abnormal'
        explanations.append(f"Critical episodic changes detected: the sequence contains significant drops in heart rate ({DP} prolonged, {DS} severe decelerations), indicating acute hypoxic events during contractions.")
    elif DS >= 2 or DL >= 2:
        dec = 'suspicious'
        explanations.append(f"Notable episodic changes detected: presence of recurring decelerations ({DS} severe, {DL} light) suggests increased fetal vulnerability to uterine contractions.")
    else:
        dec = 'normal'

    # UC EFFECT
    if UC_rate > 7 and dec == 'suspicious':
        dec = 'abnormal'
        explanations.append("The presence of uterine tachysystole (> 7 contractions per 10 minutes) combined with decelerating patterns elevates the risk profile significantly.")

    # CLASSIFICATION LOGIC
    explanation_text = " ".join(explanations) if explanations else "All extracted clinical parameters (Baseline, Variability, and Decelerations) fall well within standard physiological norms, indicating a healthy fetal state over the 60-minute observation window."

    if dec == 'abnormal' or (var == 'abnormal' and dec != 'normal'):
        return 'Abnormal', explanation_text
    
    if base == 'normal' and var != 'abnormal' and dec == 'normal':
        return 'Normal', explanation_text

    return 'Suspect', explanation_text
