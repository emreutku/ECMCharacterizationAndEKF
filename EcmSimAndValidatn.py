import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import PchipInterpolator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error
from scipy.spatial import cKDTree

# ---- CONFIG ----
CAPACITY_Ah = 2.5
USE_OCV = True
K_NEIGHBORS = 5

# Specify which data you want to validate
INPUT_DATA = 'WLTP'
# INPUT_DATA = 'HPPC'
# INPUT_DATA = 'UDDS'

# Path declaration and SOC initialization
if INPUT_DATA == 'WLTP':
    MAT_FILE = "C:\\Users\\EMRE\\Documents\\MeaTechCaseStudy\\10-25-19_11.29 960_WLTP206b.mat"
    SOC0 = 50
    COMPRESS_VOLTAGE = False
elif INPUT_DATA == 'HPPC':
    MAT_FILE = "C:\\Users\\EMRE\\Documents\\MeaTechCaseStudy\\11-17-19_17.23 1030_HPPC.mat"
    SOC0 = 100
    COMPRESS_VOLTAGE = False
elif INPUT_DATA == 'UDDS':
    MAT_FILE = "C:\\Users\\EMRE\\Documents\\MeaTechCaseStudy\\UDDS_25degC.mat"
    SOC0 = 100
    # Set COMPRESS_VOLTAGE = False if you want to validate the UDDS data
    COMPRESS_VOLTAGE = True

# ---- Load WLTP Data ----
data = loadmat(MAT_FILE)
meas = data["meas"][0, 0]
time = meas["Time"].flatten()
current = meas["Current"].flatten()
voltage = meas["Voltage"].flatten()

mask = np.isfinite(time) & np.isfinite(current) & np.isfinite(voltage)
time = time[mask]
current = current[mask]
voltage = voltage[mask]


if COMPRESS_VOLTAGE :
    # ---- NEW: Compress only the voltage to start at 3.6 V and end at 2.9 V ----
    # Map [v_min_old, v_max_old] -> [2.9, 3.6] linearly and clip to ensure bounds.
    v_min_old = float(np.nanmin(voltage))
    v_max_old = float(np.nanmax(voltage))
    if not np.isclose(v_max_old, v_min_old):
        target_min, target_max = 2.9, 3.6
        scale = (target_max - target_min) / (v_max_old - v_min_old)
        voltage = target_min + (voltage - v_min_old) * scale
        voltage = np.clip(voltage, target_min, target_max)

# ---- Load Final Merged LUT ----

# df_lut = pd.read_csv("C:\\Users\\EMRE\\Documents\\MeaTechCaseStudy\\ecm_lookup_table_refined_with_wltp.csv")
# df_lut = pd.read_csv("C:\\Users\\EMRE\\Documents\\MeaTechCaseStudy\\ecm_lookup_table_r0_corrected.csv")
df_lut = pd.read_csv("C:\\Users\\EMRE\\Documents\\MeaTechCaseStudy\\ecm_lookup_table_rc_corrected.csv")

# ---- Build KD-tree-based lookup ----
class RCkNNLUT:
    def __init__(self, df, k=3):
        self.k = k
        self.X = df[["SOC", "C_rate"]].values
        self.Ys = {
            col: df[col].values for col in ["R0", "R1", "C1", "R2", "C2"]
        }
        self.tree = cKDTree(self.X)

    def query_one(self, soc, crate):
        dist, idxs = self.tree.query([soc, crate], k=self.k)
        weights = 1 / (dist + 1e-6)
        weights /= weights.sum()
        return tuple(np.dot(weights, self.Ys[key][idxs]) for key in ["R0", "R1", "C1", "R2", "C2"])

lut = RCkNNLUT(df_lut, k=K_NEIGHBORS)

# ---- OCV Table(taken from 1-A123 Aging Tests_Data Summary.xlsx) ----
ocv_soc = np.array([100, 95, 90, 80, 70, 60, 50, 40, 30, 20, 15, 10, 5, 2.5])
ocv_volt = np.array([3.4469, 3.3316, 3.3292, 3.3274, 3.3039, 3.2892, 3.286, 3.2838,
                     3.2636, 3.2338, 3.2132, 3.1967, 3.1749, 3.077])
OCV_FN = PchipInterpolator(ocv_soc[::-1], ocv_volt[::-1], extrapolate=True)

# ---- SOC & C-rate ----
def coulomb_soc(t, i, capacity, soc0=100.0):
    dt = np.diff(t, prepend=t[0])
    Ah_used = np.cumsum(i * dt) / 3600.0
    soc = soc0 + (Ah_used / capacity) * 100
    return np.clip(soc, 0, 100)

soc = coulomb_soc(time, current, CAPACITY_Ah, SOC0)
crate = np.abs(current) / CAPACITY_Ah

# ---- ECM Simulation ----
def exact_rc_step(v_prev, I, dt, R, C):
    alpha = np.exp(-dt / (R * C))
    return alpha * v_prev + R * (1 - alpha) * I

def simulate(time, current, soc, crate, lut):
    dt = np.diff(time, prepend=time[0])
    v_rc1 = np.zeros_like(current, dtype=float)
    v_rc2 = np.zeros_like(current, dtype=float)
    V     = np.zeros_like(current, dtype=float)
    ocv   = OCV_FN(soc) 

    for k in range(1, len(time)):
        R0, R1, C1, R2, C2 = lut.query_one(soc[k], crate[k])

        v_rc1[k] = exact_rc_step(v_rc1[k-1], current[k], dt[k], R1, C1)
        v_rc2[k] = exact_rc_step(v_rc2[k-1], current[k], dt[k], R2, C2)
        V[k] = ocv[k] - current[k]*R0 - v_rc1[k] - v_rc2[k]

    R0_0, _, _, _, _ = lut.query_one(soc[0], crate[0])
    V[0] = ocv[0] - current[0]*(R0_0) 
    return V



# ---- Simulation  ----
V_sim = simulate(time, -current, soc, crate, lut)

# ---- Metrics ----
mask_valid = np.isfinite(V_sim) & np.isfinite(voltage)
rmse = mean_squared_error(voltage[mask_valid], V_sim[mask_valid], squared=False)
mae  = mean_absolute_error(voltage[mask_valid], V_sim[mask_valid])
me   = np.mean(V_sim[mask_valid] - voltage[mask_valid])
r2   = r2_score(voltage[mask_valid], V_sim[mask_valid])
maxe = max_error(voltage[mask_valid], V_sim[mask_valid])

print(f"---- WLTP Simulation metrics (KD-Tree LUT) ----")
print(f"N={mask_valid.sum()}")
print(f"RMSE={rmse:.4f} V")
print(f"MAE ={mae:.4f} V")
print(f"ME  ={me:.4f} V")
print(f"R²  ={r2:.4f}")
print(f"MaxErr={maxe:.4f} V")


# ---- Plots ----
fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

axs[0].plot(time, voltage, label="Measured Voltage", linewidth=0.8)
axs[0].set_title("Measured Voltage")
axs[0].set_ylabel("Voltage [V]")
axs[0].grid(True)

axs[1].plot(time, V_sim, label="Simulated Voltage", linewidth=0.8, color='orange')
axs[1].set_title("Simulated Voltage [2RC ECM via KNN KD-Tree LUT]")
axs[1].set_ylabel("Voltage [V]")
axs[1].grid(True)

axs[2].plot(time, V_sim - voltage, label="Residual (Sim - Meas)", linewidth=0.8)
axs[2].axhline(0, color="k", linestyle="--", linewidth=0.5)
axs[2].set_title("Simulation Residuals (Scaled)")
axs[2].set_ylabel("Residual [V]")
axs[2].set_xlabel("Time [s]")
axs[2].grid(True)

plt.tight_layout()
plt.show()
