# ============================================================
# R0 CORRECTION (iterative, weighted) on ECM LUT
#
# PURPOSE
#   Refine ONLY the ohmic resistance R0 in a 2nd-order RC ECM lookup table
#   (SOC, C-rate) -> (R0, R1, C1, R2, C2), while keeping (R1, C1, R2, C2)
#   fixed. We use WLTP and HPPC datasets, simulate the ECM, measure the
#   residual voltage error, fit a low-order correction model deltaR0(SOC, C-rate),
#   apply a bounded update to the LUT, and iterate.
#
# INPUT
#   - LUT_IN  : CSV with columns [SOC, C_rate, R0, R1, C1, R2, C2]
#   - WLTP_MAT, HPPC_MAT : .mat files containing 'meas' struct with
#         Time [s], Current [A], Voltage [V]
#
# OUTPUT
#   - LUT_OUT : CSV with corrected R0; other parameters unchanged
#
# STABILITY / REGULARIZATION
#   - Ridge regression for robustness (RIDGE_LAMBDA)
#   - Per-iteration relative cap for R0 updates (R0_MAX_REL_CHANGE)
#   - Early stop if improvement is small (IMPROVE_TOL)
# ============================================================

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, PchipInterpolator, interp1d

# -------------------- FILES / CONSTANTS --------------------
CAPACITY_AH = 2.5  # cell capacity (Ah), used for SOC update and C-rate

# Path declaration
LUT_IN   = "C:\\Users\\EMRE\\Documents\\MeaTechCaseStudy\\ecm_lookup_table_refined_with_wltp.csv"
LUT_OUT  = "C:\\Users\\EMRE\\Documents\\MeaTechCaseStudy\\ecm_lookup_table_r0_corrected.csv"

WLTP_MAT = "C:\\Users\\EMRE\\Documents\\MeaTechCaseStudy\\10-24-19_16.28 960_WLTP206a.mat"
HPPC_MAT = "C:\\Users\\EMRE\\Documents\\MeaTechCaseStudy\\10-16-19_20.16 948_HPPC.mat"

# -------------------- HYPERPARAMS TO TUNE --------------------
# Stronger correction = higher cap, lower ridge, higher gain, more iters, larger weight power
R0_MAX_REL_CHANGE = 0.60     # Max relative |deltaR0| per iteration w.r.t. baseline R0 (e.g., 0.30–0.60)
RIDGE_LAMBDA      = 5e1      # Ridge regularization (smaller -> stronger update), e.g. 3e1–1e2
CORRECTION_GAIN   = 1.35     # Scales the fitted deltaR0 before capping, typical 1.0–1.6
WEIGHT_POWER_P    = 1.0      # Weight samples by |I|^p; p>1 emphasizes high-current

N_ITER            = 3        # Number of refine-apply iterations (1–4 typical)
IMPROVE_TOL       = 1e-4     # Early-stop if Avg(MAE) improvement < tol

# Sample selection for "steady" ohmic estimate (helps isolate R0 effect)
DI_THRESH_A   = 0.10         # |deltaI| < threshold (A) marks steady region; increase to include more samples
IMIN_THRESH_A = 0.20         # Discard near-zero current (|I| ≤ this)


# -------------- OCV(SOC) TABLE (taken from 1-A123 Aging Tests_Data Summary.xlsx) ---------------
_OCV_SOC = np.array([100.0, 95.0, 90.0, 80.0, 70.0, 60.0, 50.0,
                     40.0, 30.0, 20.0, 15.0, 10.0,  5.0,  2.5], dtype=float)
_OCV_V   = np.array([3.4469, 3.3316, 3.3292, 3.3274, 3.3039, 3.2892, 3.2860,
                     3.2838, 3.2636, 3.2338, 3.2132, 3.1967, 3.1749, 3.0770], dtype=float)
order = np.argsort(_OCV_SOC)
x_u, idx_u = np.unique(_OCV_SOC[order], return_index=True)
y_u = _OCV_V[order][idx_u]
try:
    OCV_FN = PchipInterpolator(x_u, y_u, extrapolate=True)  # smooth, monotone-preserving
except Exception:
    OCV_FN = interp1d(x_u, y_u, kind="linear", fill_value="extrapolate", assume_sorted=True)  # safe fallback

# -------------------- HELPERS --------------------
def ensure_positive(x, eps=1e-12):
    # Clamp values to be strictly positive (for R/C parameters).
    return np.maximum(x, eps)

def make_interpolators(df):
    # Build per-parameter 2D interpolators over (SOC, C_rate).
    # Returns dict: name -> (LinearNDInterpolator, NearestNDInterpolator)
    # The linear interpolator is tried first; nearest used as a safe fallback.
    pts = df[["SOC", "C_rate"]].values  # shape [N,2]
    interps = {}
    for col in ["R0","R1","C1","R2","C2"]:
        vals = df[col].values
        interps[col] = (
            LinearNDInterpolator(pts, vals, fill_value=np.nan),
            NearestNDInterpolator(pts, vals)
        )
    return interps

def interp_param(s, c, interps, name):
    # Interpolate a single parameter at (SOC=s, C_rate=c) with nearest fallback.
    lin, near = interps[name]
    val = lin(s, c)
    if np.isnan(val):
        val = near(s, c)
    return float(val)

def estimate_soc_from_current(time_s, current_a, soc0_pct):
    # Coulomb count SOC using the sign convention:
    #   i_pos := -current  (discharge is positive)
    # SOC[k] = soc0_pct - (∫ i_pos dt / CAPACITY_AH) * 100
    # Returns (soc[%], i_pos[A]).
    t = np.asarray(time_s, float).reshape(-1)
    i = np.asarray(current_a, float).reshape(-1)
    dt = np.diff(t, prepend=t[0])
    if len(dt) > 1:
        dt[0] = dt[1]  # avoid dt[0]=0 in some files
    i_pos = -i  # discharge positive
    discharge_ah = np.cumsum(i_pos * dt) / 3600.0
    soc = soc0_pct - (discharge_ah / CAPACITY_AH) * 100.0
    return np.clip(soc, 0.0, 100.0), i_pos

def simulate_ecm_2rc(time_s, current_a, soc0_pct, interps, ocv_fn):
    # Forward simulate the 2nd-order RC ECM with parameters interpolated from LUT.
    t = np.asarray(time_s, float).reshape(-1)
    i_meas = np.asarray(current_a, float).reshape(-1)
    soc, i_pos = estimate_soc_from_current(t, i_meas, soc0_pct)
    dt = np.diff(t, prepend=t[0])
    if len(dt) > 1:
        dt[0] = dt[1]
    n = len(t)
    v1 = np.zeros(n); v2 = np.zeros(n); v_sim = np.zeros(n)
    r0 = np.zeros(n); r1 = np.zeros(n); r2 = np.zeros(n)
    c1 = np.zeros(n); c2 = np.zeros(n)
    crate = np.abs(i_pos)/CAPACITY_AH  # |I| normalized by capacity

    v1k = 0.0; v2k = 0.0
    for k in range(n):
        s = soc[k]; c = crate[k]
        # Interpolate ECM parameters at (s, c)
        R0 = ensure_positive(interp_param(s, c, interps, "R0"))
        R1 = ensure_positive(interp_param(s, c, interps, "R1"))
        C1 = ensure_positive(interp_param(s, c, interps, "C1"))
        R2 = ensure_positive(interp_param(s, c, interps, "R2"))
        C2 = ensure_positive(interp_param(s, c, interps, "C2"))

        r0[k], r1[k], r2[k], c1[k], c2[k] = R0, R1, R2, C1, C2

        # RC branch updates (exact discrete-time update for 1st-order RC)
        a1 = np.exp(-dt[k]/(R1*C1))
        a2 = np.exp(-dt[k]/(R2*C2))
        v1k = a1*v1k + (1-a1)*R1*i_pos[k]
        v2k = a2*v2k + (1-a2)*R2*i_pos[k]
        v1[k] = v1k; v2[k] = v2k

        # Terminal voltage
        v_sim[k] = float(ocv_fn(s)) - i_pos[k]*R0 - v1k - v2k

    return {"time":t,"i_meas":i_meas,"i_pos":i_pos,"soc":soc,"crate":crate,
            "v_sim":v_sim,"r0":r0,"r1":r1,"r2":r2,"c1":c1,"c2":c2}

def features_sc(soc, crate):
    # Feature map phi(SOC, C_rate) used to represent deltaR0 = phi @ beta.
    # We center SOC at around 55% and scale by 10 for numerical conditioning.
    # Basis = [1, s, c, s*c, s^2, c^2] where s = (SOC - 55)/10, c = C_rate
    s0 = (soc - 55.0)/10.0
    c0 = crate
    return np.column_stack([np.ones_like(s0), s0, c0, s0*c0, s0**2, c0**2])

def weighted_ridge_delta_r0(residual_v, i_pos, X, ridge_lambda, weights=None, gain=1.0):
    # Solve for beta in: residual_v = -i_pos * (X @ beta)
    #   => y = residual_v, Z = (-i_pos)[:,None] * X
    # Weighted ridge regression:
    #   argmin_beta ||W^{1/2}(y - Z beta)||^2 + lambda||beta||^2
    #   => (Z^T W Z + lambda*I) beta = Z^T W y
    # RETURNS gain * beta.
    #
    # NOTE: This implementation explicitly forms W = diag(weights).
    # If you hit MemoryError for very large N, replace with a row-scaling
    # approach (scale Z and y by sqrt(w)) to avoid materializing W.
    y = residual_v.reshape(-1,1)               # [N,1]
    Z = (-i_pos.reshape(-1,1)) * X             # [N,F]
    if weights is None:
        W = np.eye(len(y))                     # [N,N]
    else:
        w = np.asarray(weights, float).reshape(-1)
        W = np.diag(np.maximum(w, 1e-12))      # [N,N]
    ZTWZ = Z.T @ W @ Z                         # [F,F]
    ZTWy = Z.T @ W @ y                         # [F,1]
    lamI = ridge_lambda * np.eye(ZTWZ.shape[0])
    beta = np.linalg.solve(ZTWZ + lamI, ZTWy).ravel()
    return gain * beta

def apply_delta_r0_to_lut(df, beta, per_iter_cap=0.45):
    # Apply deltaR0 = phi(SOC,C_rate) @ beta to the whole LUT, with a
    # per-entry relative cap: |deltaR0| ≤ per_iter_cap * R0_base.
    # Optional SOC window restricts the update to a given band.
    df2 = df.copy()
    Xg  = features_sc(df2["SOC"].values, df2["C_rate"].values)  # [G,F]
    dR0 = Xg @ beta                                             # [G,]

    mask = np.ones(len(df2), dtype=bool)
    R0b = df2["R0"].values
    cap = per_iter_cap * R0b
    dR0 = np.clip(dR0, -cap, cap)
    df2.loc[mask, "R0"] = ensure_positive(R0b[mask] + dR0[mask])
    return df2

def mae(x):
    # Mean absolute error.
    return float(np.mean(np.abs(x)))

def plot_dataset(prefix, t, v_meas, v_sim_base, v_sim_corr, err_base, err_corr, i, soc):
    # Quick visualization helper: voltage overlay, error overlay, current, SOC.
    plt.figure(); plt.plot(t, v_meas, linewidth=1.0, label="Measured")
    plt.plot(t, v_sim_base, linewidth=1.0, label="Sim (Base)")
    plt.plot(t, v_sim_corr, linewidth=1.0, label="Sim (R0-corrected)")
    plt.title(f"{prefix} Voltage"); plt.xlabel("Time [s]"); plt.ylabel("Voltage [V]")
    plt.grid(True); plt.legend(); plt.show()

    plt.figure(); plt.plot(t, err_base, linewidth=1.0, label="Error (Base)")
    plt.plot(t, err_corr, linewidth=1.0, label="Error (Corrected)")
    plt.title(f"{prefix} Voltage Error (Measured - Sim)")
    plt.xlabel("Time [s]"); plt.ylabel("Error [V]"); plt.grid(True); plt.legend(); plt.show()

    plt.figure(); plt.plot(t, i, linewidth=1.0)
    plt.title(f"{prefix} Current"); plt.xlabel("Time [s]"); plt.ylabel("Current [A]"); plt.grid(True); plt.show()

    plt.figure(); plt.plot(t, soc, linewidth=1.0)
    plt.title(f"{prefix} SOC"); plt.xlabel("Time [s]"); plt.ylabel("SOC [%]"); plt.grid(True); plt.show()

# -------------------- LOAD LUT + DATA --------------------
# Load baseline LUT and make sure parameters are strictly positive
lut = pd.read_csv(LUT_IN)
for col in ["R0","R1","C1","R2","C2"]:
    lut[col] = ensure_positive(lut[col])

# Load WLTP dataset (starts near 50% SOC) and HPPC (near 100% SOC)
meas_w = sio.loadmat(WLTP_MAT)["meas"][0,0]
t_w = meas_w["Time"][:,0].astype(float)
i_w = meas_w["Current"][:,0].astype(float)
v_w = meas_w["Voltage"][:,0].astype(float)

meas_h = sio.loadmat(HPPC_MAT)["meas"][0,0]
t_h = meas_h["Time"][:,0].astype(float)
i_h = meas_h["Current"][:,0].astype(float)
v_h = meas_h["Voltage"][:,0].astype(float)

# -------------------- ITERATIVE R0 CORRECTION --------------------
interps = make_interpolators(lut)

last_mae = np.inf  # track average MAE across WLTP & HPPC
for it in range(1, N_ITER+1):
    # 1) Simulate with the current LUT
    sim_w = simulate_ecm_2rc(t_w, i_w, soc0_pct=50.0,  interps=interps, ocv_fn=OCV_FN)
    sim_h = simulate_ecm_2rc(t_h, i_h, soc0_pct=100.0, interps=interps, ocv_fn=OCV_FN)

    # Residuals (measured - simulated); positive residual means simulated voltage too low
    res_w = v_w - sim_w["v_sim"]
    res_h = v_h - sim_h["v_sim"]

    # 2) Select "steady" samples: small |deltaI| and |I| above threshold
    dI_w = np.diff(i_w, prepend=i_w[0])
    dI_h = np.diff(i_h, prepend=i_h[0])
    steady_w = (np.abs(dI_w) < DI_THRESH_A) & (np.abs(i_w) > IMIN_THRESH_A)
    steady_h = (np.abs(dI_h) < DI_THRESH_A) & (np.abs(i_h) > IMIN_THRESH_A)

    # 3) Build features delta(SOC, C-rate) and weights w = |I|^p
    X_w = features_sc(sim_w["soc"][steady_w],  sim_w["crate"][steady_w])
    X_h = features_sc(sim_h["soc"][steady_h],  sim_h["crate"][steady_h])
    y_w = res_w[steady_w]; y_h = res_h[steady_h]
    i_wpos = sim_w["i_pos"][steady_w]; i_hpos = sim_h["i_pos"][steady_h]
    w_w = np.abs(i_wpos)**WEIGHT_POWER_P
    w_h = np.abs(i_hpos)**WEIGHT_POWER_P

    # Stack both datasets together for a joint fit
    X_all = np.vstack([X_w, X_h])           # [N,F]
    y_all = np.concatenate([y_w, y_h])      # [N,]
    i_all = np.concatenate([i_wpos, i_hpos])# [N,]
    w_all = np.concatenate([w_w, w_h])      # [N,]

    # 4) Fit weighted ridge for beta in deltaR0 = phi beta (with the -i_pos mapping)
    beta = weighted_ridge_delta_r0(
        residual_v=y_all,
        i_pos=i_all,
        X=X_all,
        ridge_lambda=RIDGE_LAMBDA,
        weights=w_all,
        gain=CORRECTION_GAIN
    )

    # 5) Apply CAPped ΔR0 to LUT
    lut = apply_delta_r0_to_lut(lut, beta, per_iter_cap=R0_MAX_REL_CHANGE)
    interps = make_interpolators(lut)  # refresh interpolators after update

    # 6) Re-simulate to evaluate improvement and check early stop
    sim_w_new = simulate_ecm_2rc(t_w, i_w, soc0_pct=50.0,  interps=interps, ocv_fn=OCV_FN)
    sim_h_new = simulate_ecm_2rc(t_h, i_h, soc0_pct=100.0, interps=interps, ocv_fn=OCV_FN)
    mae_w = mae(v_w - sim_w_new["v_sim"])
    mae_h = mae(v_h - sim_h_new["v_sim"])
    cur_mae = 0.5*(mae_w + mae_h)

    print(f"[Iter {it}]  WLTP MAE: {mae_w:.4f} V | HPPC MAE: {mae_h:.4f} V | Avg: {cur_mae:.4f} V")
    if last_mae - cur_mae < IMPROVE_TOL:
        print("Early stop: improvement below tolerance.")
        break
    last_mae = cur_mae

# -------------------- SAVE UPDATED LUT --------------------
lut.to_csv(LUT_OUT, index=False)
print("Saved corrected LUT to:", LUT_OUT)

# -------------------- FINAL COMPARISON PLOTS --------------------
# Simulate base (pre-correction) once more for A/B comparison
lut_base = pd.read_csv(LUT_IN)
interps_base = make_interpolators(lut_base)
sim_w_base = simulate_ecm_2rc(t_w, i_w, soc0_pct=50.0,  interps=interps_base, ocv_fn=OCV_FN)
sim_h_base = simulate_ecm_2rc(t_h, i_h, soc0_pct=100.0, interps=interps_base, ocv_fn=OCV_FN)

# Simulate corrected
interps_corr = make_interpolators(lut)
sim_w_corr = simulate_ecm_2rc(t_w, i_w, soc0_pct=50.0,  interps=interps_corr, ocv_fn=OCV_FN)
sim_h_corr = simulate_ecm_2rc(t_h, i_h, soc0_pct=100.0, interps=interps_corr, ocv_fn=OCV_FN)

# Errors (Measured - Sim)
res_w_base = v_w - sim_w_base["v_sim"]; res_w_corr = v_w - sim_w_corr["v_sim"]
res_h_base = v_h - sim_h_base["v_sim"]; res_h_corr = v_h - sim_h_corr["v_sim"]

print(f"FINAL  WLTP  MAE  Base: {mae(res_w_base):.4f} V  |  Corrected: {mae(res_w_corr):.4f} V")
print(f"FINAL  HPPC  MAE  Base: {mae(res_h_base):.4f} V  |  Corrected: {mae(res_h_corr):.4f} V")

# ---- WLTP plots ----
plot_dataset(
    prefix="WLTP",
    t=sim_w_base["time"],
    v_meas=v_w,
    v_sim_base=sim_w_base["v_sim"],
    v_sim_corr=sim_w_corr["v_sim"],
    err_base=res_w_base,
    err_corr=res_w_corr,
    i=sim_w_base["i_meas"],
    soc=sim_w_base["soc"]
)

# ---- HPPC plots ----
plot_dataset(
    prefix="HPPC",
    t=sim_h_base["time"],
    v_meas=v_h,
    v_sim_base=sim_h_base["v_sim"],
    v_sim_corr=sim_h_corr["v_sim"],
    err_base=res_h_base,
    err_corr=res_h_corr,
    i=sim_h_base["i_meas"],
    soc=sim_h_base["soc"]
)