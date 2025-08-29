import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, interp1d
from scipy.io import loadmat  # match EcmSimAndValidatn loading style

# Path declaration
LUT_FILE = "C:\\Users\\EMRE\\Documents\\MeaTechCaseStudy\\ecm_lookup_table_rc_corrected.csv"
MAT_FILE = "C:\\Users\\EMRE\\Documents\\MeaTechCaseStudy\\UDDS_25degC.mat"

# Load UDDS drivecycle
data = loadmat(MAT_FILE)
meas = data['meas'][0, 0]
time = meas['Time'].flatten()
current = meas['Current'].flatten()
voltage = meas['Voltage'].flatten()
Ah = meas['Ah'].flatten()  # cumulative discharge (positive value)

# Finite-data mask (align with EcmSimAndValidatn)
mask = np.isfinite(time) & np.isfinite(current) & np.isfinite(voltage)
# Apply the mask
time = time[mask]
current = current[mask]
voltage = voltage[mask]
# If Ah length matches, mask it too (keeps other logic intact)
if Ah.shape[0] == mask.shape[0]:
    Ah = Ah[mask]

COMPRESS_VOLTAGE = False
if COMPRESS_VOLTAGE :
    # === NEW: Compress only the voltage to start at 3.6 V and end at 2.9 V ===
    # Map [v_min_old, v_max_old] -> [2.9, 3.6] linearly and clip to ensure bounds.
    v_min_old = float(np.nanmin(voltage))
    v_max_old = float(np.nanmax(voltage))
    if not np.isclose(v_max_old, v_min_old):
        target_min, target_max = 2.9, 3.6
        scale = (target_max - target_min) / (v_max_old - v_min_old)
        voltage = target_min + (voltage - v_min_old) * scale
        voltage = np.clip(voltage, target_min, target_max)

# OCV-SOC table
ocv_soc = np.array([100, 95, 90, 80, 70, 60, 50, 40, 30, 20, 15, 10, 5, 2.5])
ocv_volt = np.array([3.4469, 3.3316, 3.3292, 3.3274, 3.3039, 3.2892, 3.286, 3.2838,
                     3.2636, 3.2338, 3.2132, 3.1967, 3.1749, 3.077])

# Create OCV and dOCV/dSOC interpolators
ecm_df = pd.read_csv(LUT_FILE)
f_ocv = interp1d(ocv_soc, ocv_volt, kind='linear', bounds_error=False, fill_value=(3.077, 3.4469))
d_ocv = np.gradient(ocv_volt, ocv_soc)
f_docv = interp1d(ocv_soc, d_ocv, kind='linear', bounds_error=False, fill_value=(d_ocv[0], d_ocv[-1]))

# Create ECM interpolators
points = ecm_df[['SOC', 'C_rate']].values
interpolators = {}
for param in ['R0', 'R1', 'C1', 'R2', 'C2']:
    lin_interp = LinearNDInterpolator(points, ecm_df[param].values)
    near_interp = NearestNDInterpolator(points, ecm_df[param].values)
    interpolators[param] = lambda soc, cr, lin=lin_interp, near=near_interp: (
        lin(soc, cr) if not np.isnan(lin(soc, cr)) else near(soc, cr)
    )

# Set nominal capacity (2.5 Ah)
capacity = 2.5  # Ah

# Calculate actual discharge from current data (negative = discharge)
total_discharge = np.trapz(current, time) / 3600  # Ah (integral of current in A-s converted to Ah)
print(f"Total discharge from current: {total_discharge:.4f} Ah")
print(f"Total discharge from Ah field: {Ah[-1] - Ah[0]:.4f} Ah")

# Initialize EKF
soc_ekf = np.zeros(len(time))
v1_ekf = np.zeros(len(time))
v2_ekf = np.zeros(len(time))

# Set initial SOC using OCV
soc0 = np.interp(voltage[0], ocv_volt[::-1], ocv_soc[::-1])
soc_ekf[0] = np.clip(soc0, 0, 100)
print(f"Initial voltage: {voltage[0]:.4f} V -> Initial SOC: {soc_ekf[0]:.2f}%")

# CORRECT Coulomb counting (using actual current integration)
soc_cc = np.zeros(len(time))
soc_cc[0] = 100
for i in range(1, len(time)):
    dt = time[i] - time[i-1]
    soc_cc[i] = soc_cc[i-1] + (current[i-1] * dt) / (capacity * 3600) * 100
soc_cc = np.clip(soc_cc, 0, 100)

# EKF setup
P = np.diag([0.01, 0.001, 0.001])  # Initial covariance
Q = np.diag([1e-7, 1e-5, 1e-5])    # Process noise
# Measurement noise piece-wise adjustment
MEAS_NOISE_VAR = True
R = 1e-5

# Run EKF
for k in range(len(time)-1):
    # Do not trust ECM model at high voltage region
    if (MEAS_NOISE_VAR == True) and voltage[k] > 3.9:
        R = 1
    else
        R = 1e-5
        

    dt = time[k+1] - time[k]
    if dt <= 0: 
        continue

    # Get C-rate (always positive)
    cr = abs(current[k]) / capacity

    # Get ECM parameters - clamp SOC to valid range for interpolation
    interp_soc = np.clip(soc_ekf[k], 0.1, 99.9)
    R0 = interpolators['R0'](interp_soc, cr)
    R1 = interpolators['R1'](interp_soc, cr)
    C1 = interpolators['C1'](interp_soc, cr)
    R2 = interpolators['R2'](interp_soc, cr)
    C2 = interpolators['C2'](interp_soc, cr)

    # STATE PREDICTION
    # SOC update: (current[k] * dt) / (capacity * 3600) * 100, Coulomb count based
    soc_pred = soc_ekf[k] + (current[k] * dt) / (capacity * 3600) * 100

    # Clamp predicted SOC
    soc_pred = np.clip(soc_pred, 0, 100)
    
    # RC voltage updates
    tau1 = R1 * C1
    a1 = np.exp(-dt/tau1) if tau1 > 0 else 0
    v1_pred = a1 * v1_ekf[k] + R1 * (1 - a1) * current[k]

    tau2 = R2 * C2
    a2 = np.exp(-dt/tau2) if tau2 > 0 else 0
    v2_pred = a2 * v2_ekf[k] + R2 * (1 - a2) * current[k]

    # COVARIANCE PREDICTION
    F = np.diag([1, a1, a2])
    P_pred = F @ P @ F.T + Q

    # MEASUREMENT UPDATE
    ocv_pred = f_ocv(soc_pred)
    v_pred = ocv_pred - v1_pred - v2_pred - R0 * current[k]

    # Jacobian
    H = np.array([f_docv(soc_pred), -1, -1]).reshape(1, 3)

    # Kalman gain
    S = H @ P_pred @ H.T + R
    K = (P_pred @ H.T) / S

    # State correction
    innovation = voltage[k] - v_pred
    state_corr = np.array([soc_pred, v1_pred, v2_pred]) + K.flatten() * innovation

    # Clamp and store
    soc_ekf[k+1] = np.clip(state_corr[0], 0.1, 99.9)
    v1_ekf[k+1] = state_corr[1]
    v2_ekf[k+1] = state_corr[2]

    # Covariance update
    P = (np.eye(3) - K @ H) @ P_pred

# Plot results
plt.figure(figsize=(12, 8))
plt.plot(time/3600, soc_ekf, 'b-', linewidth=2, label='EKF SOC')
plt.plot(time/3600, soc_cc, 'r--', linewidth=1.5, label='Coulomb Counting SOC')
plt.xlabel('Time (hours)', fontsize=12)
plt.ylabel('SOC (%)', fontsize=12)
plt.title('SOC Estimation Comparison', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.ylim([0, 105])
plt.tight_layout()
plt.show()

# Debugging plots
fig, ax = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
ax[0].plot(time/3600, current)
ax[0].set_ylabel('Current (A)')
ax[0].grid(True)

ax[1].plot(time/3600, voltage)
ax[1].set_ylabel('Voltage (V)')
ax[1].grid(True)

ax[2].plot(time/3600, soc_ekf, 'b-', label='EKF')
ax[2].plot(time/3600, soc_cc, 'r--', label='Coulomb')
ax[2].set_ylabel('SOC (%)')
ax[2].legend()
ax[2].grid(True)
ax[2].set_ylim([0, 105])

ax[3].plot(time/3600, soc_ekf-soc_cc)
ax[3].set_ylabel('SOC Estimation Error (%)')
ax[3].legend()
ax[3].grid(True)
ax[3].set_ylim([0,20])

plt.tight_layout()
plt.show()

