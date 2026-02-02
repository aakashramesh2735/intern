# ===============================
# EMG FILTERING PIPELINE (Spyder)
# ===============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# -------------------------------
# STEP 1: Load EMG data
# -------------------------------

file_path = r"C:/Users/abish/Downloads/aakash main project/raw_emg_6muscles_8phases_5subjects.xlsx"
df = pd.read_excel(file_path)

# Select one example (change later for loops)
emg_raw = df[(df["Subject"] == 1) &
             (df["Gait_Phase"] == 1)]["RF"].values

# -------------------------------
# STEP 2: Sampling frequency
# -------------------------------

Fs = 1000  # Hz

# -------------------------------
# STEP 3: Remove DC offset
# -------------------------------

emg_dc = emg_raw - np.mean(emg_raw)

# -------------------------------
# STEP 4: Band-pass filter
# -------------------------------

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

emg_band = bandpass_filter(emg_dc, 20, 450, Fs)

# -------------------------------
# STEP 5: Rectification
# -------------------------------

emg_rect = np.abs(emg_band)

# -------------------------------
# STEP 6: Low-pass filter (Envelope)
# -------------------------------

def lowpass_filter(signal, cutoff, fs, order=4):
    nyq = 0.5 * fs
    norm = cutoff / nyq
    b, a = butter(order, norm, btype='low')
    return filtfilt(b, a, signal)

emg_env = lowpass_filter(emg_rect, 6, Fs)

# -------------------------------
# STEP 7: Normalization
# -------------------------------

emg_norm = emg_env / np.max(emg_env)

# -------------------------------
# STEP 8: Plot results
# -------------------------------

plt.figure(figsize=(12,6))
plt.plot(emg_raw, label="Raw EMG", alpha=0.4)
plt.plot(emg_band, label="Band-passed EMG", alpha=0.7)
plt.plot(emg_env, label="EMG Envelope", linewidth=2)
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.title("EMG Filtering in Spyder")
plt.legend()
plt.grid(True)
plt.show()

