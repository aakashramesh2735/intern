import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# -------------------------------
# 1. Create dummy time vector
# -------------------------------
fs = 1000            # EMG sampling rate (Hz)
duration = 10        # seconds
t = np.linspace(0, duration, fs * duration)

# -------------------------------
# 2. Create dummy EMG signals
# -------------------------------
np.random.seed(0)

# Tibialis Anterior (active during swing)
emg_ta = 0.5 * np.sin(2 * np.pi * 1 * t) * (t % 1 < 0.4)
emg_ta += 0.1 * np.random.randn(len(t))

# Gastrocnemius (active during stance)
emg_gas = 0.6 * np.sin(2 * np.pi * 1 * t) * (t % 1 > 0.4)
emg_gas += 0.1 * np.random.randn(len(t))

# Soleus (similar to gastrocnemius)
emg_sol = 0.5 * np.sin(2 * np.pi * 1 * t) * (t % 1 > 0.4)
emg_sol += 0.1 * np.random.randn(len(t))

# -------------------------------
# 3. EMG processing functions
# -------------------------------
def bandpass(signal, lowcut, highcut, fs, order=4):
    b, a = butter(order, [lowcut/(fs/2), highcut/(fs/2)], btype='band')
    return filtfilt(b, a, signal)

def lowpass(signal, cutoff, fs, order=4):
    b, a = butter(order, cutoff/(fs/2), btype='low')
    return filtfilt(b, a, signal)

def process_emg(emg):
    emg_bp = bandpass(emg, 20, 450, fs)
    emg_rect = np.abs(emg_bp)
    emg_env = lowpass(emg_rect, 6, fs)
    emg_norm = emg_env / np.max(emg_env)
    return emg_norm

# -------------------------------
# 4. Process EMG
# -------------------------------
a_ta = process_emg(emg_ta)
a_gas = process_emg(emg_gas)
a_sol = process_emg(emg_sol)

# -------------------------------
# 5. Muscle force model (simple)
# -------------------------------
Fmax_TA = 800     # N
Fmax_GAS = 1500   # N
Fmax_SOL = 1200   # N

F_ta = a_ta * Fmax_TA
F_pf = (a_gas * Fmax_GAS) + (a_sol * Fmax_SOL)

# -------------------------------
# 6. Ankle torque calculation
# -------------------------------
r_ta = 0.03   # m (dorsiflexor moment arm)
r_pf = 0.05   # m (plantarflexor moment arm)

ankle_torque = (F_pf * r_pf) - (F_ta * r_ta)

# -------------------------------
# 7. Plot results
# -------------------------------
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, a_ta, label="TA")
plt.plot(t, a_gas, label="GAS")
plt.plot(t, a_sol, label="SOL")
plt.ylabel("Activation")
plt.legend()
plt.title("Processed EMG (Muscle Activation)")

plt.subplot(3, 1, 2)
plt.plot(t, F_ta, label="TA Force")
plt.plot(t, F_pf, label="Plantarflexor Force")
plt.ylabel("Force (N)")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, ankle_torque)
plt.ylabel("Torque (Nm)")
plt.xlabel("Time (s)")
plt.title("Estimated Ankle Torque")

plt.tight_layout()
plt.show()