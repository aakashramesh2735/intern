

# ======================================
# EMG + IMU FILTRATION PIPELINE
# (Windows-safe, thesis-ready)
# ======================================

import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt


# ======================================
# USER SETTINGS (ONLY CHANGE THIS)
# ======================================
BASE_PATH = r"C:\Users\abish\Downloads\aakash main project\raw data"

EMG_FILE = r"C:\Users\abish\Downloads\aakash main project\raw data\raw_EMG_data.xlsx"
IMU_FILE = r"C:\Users\abish\Downloads\aakash main project\raw data\raw_IMU_data.xlsx"


# ======================================
# LOAD DATA
# ======================================
emg_df = pd.read_excel(os.path.join(BASE_PATH, EMG_FILE))
imu_df = pd.read_excel(os.path.join(BASE_PATH, IMU_FILE))


# ======================================
# FILTER FUNCTIONS
# ======================================
def bandpass_filter(signal, fs, lowcut=20, highcut=450, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def lowpass_filter(signal, fs, cutoff=6, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low")
    return filtfilt(b, a, signal)


def rms_envelope(signal, window_size):
    return np.sqrt(
        np.convolve(
            signal ** 2,
            np.ones(window_size) / window_size,
            mode="same"
        )
    )


# ======================================
# EMG FILTERING
# ======================================
fs_emg = 1000  # Hz
emg_muscles = ["RF", "BF", "SEM", "VL", "GAS", "TA"]

emg_filtered = pd.DataFrame()
emg_filtered["Time_s"] = emg_df["Time_s"]

for muscle in emg_muscles:
    raw = emg_df[muscle].values
    bp = bandpass_filter(raw, fs_emg)
    emg_filtered[muscle] = np.abs(bp)


# ======================================
# EMG RMS ENVELOPE
# ======================================
window_ms = 50
window_samples = int(window_ms * fs_emg / 1000)

emg_rms = pd.DataFrame()
emg_rms["Time_s"] = emg_filtered["Time_s"]

for muscle in emg_muscles:
    emg_rms[muscle] = rms_envelope(
        emg_filtered[muscle].values,
        window_samples
    )


# ======================================
# IMU FILTERING
# ======================================
fs_imu = 100  # Hz
imu_joints = [
    "Hip_FlexExt_deg",
    "Knee_FlexExt_deg",
    "Ankle_DorsiPlantar_deg"
]

imu_filtered = pd.DataFrame()
imu_filtered["Time_s"] = imu_df["Time_s"]

for joint in imu_joints:
    imu_filtered[joint] = lowpass_filter(
        imu_df[joint].values,
        fs_imu
    )


# ======================================
# SAVE OUTPUT FILES
# ======================================
os.makedirs(BASE_PATH, exist_ok=True)

emg_filtered.to_excel(os.path.join(BASE_PATH, r"C:\Users\abish\Downloads\aakash main project\Filtered data.xlsx"),
    index=False
)

emg_rms.to_excel(
    os.path.join(BASE_PATH, r"C:\Users\abish\Downloads\aakash main project\Filtered data\filtered data emg.xlsx"),
    index=False
)

imu_filtered.to_excel(
    os.path.join(BASE_PATH, r"C:\Users\abish\Downloads\aakash main project\Filtered data\filtered data imu.xlsx"),
    index=False
)

print("✅ EMG & IMU filtration completed successfully.")
