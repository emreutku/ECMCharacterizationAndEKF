# ============================================================
# FAST RC (R1,R2,C1,C2) CORRECTION via ANALYTIC SENSITIVITIES
# - Input LUT  : ecm_lookup_table_r0_corrected.csv  (R0 already fixed)
# - Output LUT : ecm_lookup_table_rc_corrected_fast.csv
# - OCV(SOC)   : fixed table (below)
# - Speedups   :
#     * Analytic forward sensitivities (no per-feature re-sims)
#     * Vectorized feature basis phi(SOC,C)
#     * Optional subsampling & edge-weighting
# - R0 is NOT modified here.
# - Plots: measured vs sim (base & corrected), errors, current, SOC
# ============================================================

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, PchipInterpolator, interp1d
from pathlib import Path

# -------------------- FILES / CONSTANTS --------------------
CAPACITY_AH = 2.5

# Path declaration
LUT_IN   = "C:\\Users\\EMRE\\Documents\\MeaTechCaseStudy\\ecm_lookup_table_r0_corrected.csv"
LUT_OUT  = "C:\\Users\\EMRE\\Documents\\MeaTechCaseStudy\\ecm_lookup_table_rc_corrected.csv"

WLTP_MAT = "C:\\Users\\EMRE\\Documents\\MeaTechCaseStudy\\10-24-19_16.28 960_WLTP206a.mat"  # 50% SOC start
HPPC_MAT = "C:\\Users\\EMRE\\Documents\\MeaTechCaseStudy\\10-16-19_20.16 948_HPPC.mat"      # 100% SOC start

# -------------------- HYPERPARAMS (speed/strength knobs) --------------------
# Iterations / stopping
N_ITER       = 3
IMPROVE_TOL  = 1e-3           # avg MAE improvement threshold

# Ridge (smaller => stronger correction)
RIDGE_LAMBDA = 3e1

# Per-iteration relative caps (keep close to base)
CAP_R_REL    = 0.60           # for R1,R2
CAP_C_REL    = 0.70           # for C1,C2

# Subsampling (speed): use every K-th sample (1=use all)
SUBSAMPLE_W  = 2              # WLTP stride (2 or 3 speeds things up a lot)
SUBSAMPLE_H  = 1              # HPPC stride

# Edge-weighting (emphasize dynamics RC explains)
EDGE_POWER_Q   = 1.0          # weight ~ |deltaI|^q
I_MIN_FOR_USE  = 0.10         # ignore tiny-current samples
USE_ALL_SAMPLES = True        # False -> only edge-ish samples (|deltaI| >= DI_EDGE_THRESH)
DI_EDGE_THRESH  = 0.08        # edge threshold if USE_ALL_SAMPLES=False

# Optional SOC focus (None => whole range)
SOC_WINDOW = None             # e.g., (45.0, 65.0)

# -------------------- OCV(SOC) TABLE (taken from 1-A123 Aging Tests_Data Summary.xlsx) --------------------
_OCV_SOC = np.array([100.0, 95.0, 90.0, 80.0, 70.0, 60.0, 50.0,
                     40.0, 30.0, 20.0, 15.0, 10.0,  5.0,  2.5], dtype=float)
_OCV_V   = np.array([3.4469, 3.3316, 3.3292, 3.3274, 3.3039, 3.2892, 3.2860,
                     3.2838, 3.2636, 3.2338, 3.2132, 3.1967, 3.1749, 3.0770], dtype=float)
order = np.argsort(_OCV_SOC)
x_u, idx_u = np.unique(_OCV_SOC[order], return_index=True)
y_u = _OCV_V[order][idx_u]
try:
    OCV_FN = PchipInterpolator(x_u, y_u, extrapolate=True)
except Exception:
    OCV_FN = interp1d(x_u, y_u, kind="linear", fill_value="extrapolate", assume_sorted=True)

# -------------------- HELPERS --------------------
def ensure_positive(x, eps=1e-12):
    return np.maximum(x, eps)

def make_interpolators(df):
    pts = df[["SOC","C_rate"]].values
    interps = {}
    for col in ["R0","R1","C1","R2","C2"]:
        vals = df[col].values
        interps[col] = (
            LinearNDInterpolator(pts, vals, fill_value=np.nan),
            NearestNDInterpolator(pts, vals)
        )
    return interps

def interp_param_vec(s_arr, c_arr, interps, name):
    lin, near = interps[name]
    v = lin(s_arr, c_arr)
    m = np.isnan(v)
    if m.any():
        v[m] = near(s_arr[m], c_arr[m])
    return np.asarray(v, float)

def estimate_soc_from_current(time_s, current_a, soc0_pct):
    t = np.asarray(time_s, float).reshape(-1)
    i = np.asarray(current_a, float).reshape(-1)
    dt = np.diff(t, prepend=t[0])
    if len(dt) > 1: dt[0] = dt[1]
    i_pos = -i
    discharge_ah = np.cumsum(i_pos * dt) / 3600.0
    soc = soc0_pct - (discharge_ah / CAPACITY_AH) * 100.0
    return np.clip(soc, 0.0, 100.0), i_pos, dt

def features_sc(soc, crate):
    s0 = (soc - 55.0)/10.0
    c0 = crate
    # 6 basis features (matches earlier R0 code)
    return np.column_stack([
        np.ones_like(s0),   # 0 intercept
        s0,                 # 1
        c0,                 # 2
        s0*c0,              # 3
        s0**2,              # 4
        c0**2               # 5
    ])

def subsample(*arrays, stride=1):
    if stride <= 1:
        return arrays if len(arrays) > 1 else arrays[0]
    idx = slice(None, None, int(stride))
    outs = [a[idx] for a in arrays]
    return outs if len(outs) > 1 else outs[0]

# -------------------- BASELINE + SENSITIVITY SIM --------------------
def simulate_and_sensitivities(t, i, soc0, interps):
    """
    One forward pass that returns:
      - baseline sim (v_sim, v1, v2, soc, crate, dt)
      - analytic sensitivities of v_sim wrt parameter GROUPS:
        dV_dR1 (N,), dV_dC1 (N,), dV_dR2 (N,), dV_dC2 (N,)
    These are derivatives wrt the *local parameter scalar* at time k.
    We then project to feature-space Φ by elementwise multiply at each k.
    """
    t = np.asarray(t, float).reshape(-1)
    i_meas = np.asarray(i, float).reshape(-1)
    soc, i_pos, dt = estimate_soc_from_current(t, i_meas, soc0)
    crate = np.abs(i_pos)/CAPACITY_AH
    N = len(t)

    # Interpolate parameter arrays once (fast)
    R0 = ensure_positive(interp_param_vec(soc, crate, interps, "R0"))
    R1 = ensure_positive(interp_param_vec(soc, crate, interps, "R1"))
    C1 = ensure_positive(interp_param_vec(soc, crate, interps, "C1"))
    R2 = ensure_positive(interp_param_vec(soc, crate, interps, "R2"))
    C2 = ensure_positive(interp_param_vec(soc, crate, interps, "C2"))

    # Baseline states & output
    v1 = np.zeros(N); v2 = np.zeros(N); v_sim = np.zeros(N)

    # Sensitivities wrt local scalars at each k:
    # We propagate dv1/dp and dv2/dp (forward sensitivities),
    # but here p is a *time-varying* parameter (R1_k, C1_k, R2_k, C2_k).
    # For projection onto phi, we only need dv/dp at the same k (see phi below).
    dVdR1 = np.zeros(N); dVdC1 = np.zeros(N)
    dVdR2 = np.zeros(N); dVdC2 = np.zeros(N)

    # Auxiliary running sensitivities of v1 and v2 w.r.t. previous parameters are not needed
    # because we parameterize R1_k = R1_base_k + phi_k.beta  (only the "local" p_k couples to beta via phi_k).
    # Thus, dv_k/dbeta = dv_k/dp_k * phi_k + higher-order cross-terms we neglect in this linearization.
    # This is equivalent to "instantaneous" sensitivity at k, which works well for modest updates.

    for k in range(N):
        a1 = np.exp(-dt[k]/(R1[k]*C1[k]))
        a2 = np.exp(-dt[k]/(R2[k]*C2[k]))

        # Update states
        v1 = np.asarray(v1)  # ensure array (for k=0 case)
        v2 = np.asarray(v2)
        v1_prev = v1[k-1] if k > 0 else 0.0
        v2_prev = v2[k-1] if k > 0 else 0.0

        v1_k = a1*v1_prev + (1 - a1)*R1[k]*i_pos[k]
        v2_k = a2*v2_prev + (1 - a2)*R2[k]*i_pos[k]
        v1[k] = v1_k; v2[k] = v2_k

        v_sim[k] = float(OCV_FN(soc[k])) - i_pos[k]*R0[k] - v1_k - v2_k

        # Local partials of a1,a2
        da1_dR1 = a1 * (dt[k]/(R1[k]**2 * C1[k]))
        da1_dC1 = a1 * (dt[k]/(R1[k] * C1[k]**2))
        da2_dR2 = a2 * (dt[k]/(R2[k]**2 * C2[k]))
        da2_dC2 = a2 * (dt[k]/(R2[k] * C2[k]**2))

        # Local (instantaneous) dv1_k/dR1_k and dv1_k/dC1_k
        # v1_k = a1 v1_{k-1} + (1-a1) R1 i
        # d/dR1_k: = da1_dR1 * v1_{k-1} + [ -da1_dR1 * R1 i + (1-a1) i ]
        dv1_dR1_local = da1_dR1 * (v1_prev - R1[k]*i_pos[k]) + (1 - a1)*i_pos[k]
        # d/dC1_k: = da1_dC1 * (v1_{k-1} - R1 i)
        dv1_dC1_local = da1_dC1 * (v1_prev - R1[k]*i_pos[k])

        # Similarly for v2
        dv2_dR2_local = da2_dR2 * (v2_prev - R2[k]*i_pos[k]) + (1 - a2)*i_pos[k]
        dv2_dC2_local = da2_dC2 * (v2_prev - R2[k]*i_pos[k])

        # v_sim = ... - v1 - v2, so:
        dVdR1[k] = -dv1_dR1_local
        dVdC1[k] = -dv1_dC1_local
        dVdR2[k] = -dv2_dR2_local
        dVdC2[k] = -dv2_dC2_local

    return {
        "v_sim": v_sim, "v1": v1, "v2": v2, "soc": soc, "crate": crate, "dt": dt, "i_pos": i_pos,
        "R0": R0, "R1": R1, "C1": C1, "R2": R2, "C2": C2,
        "dVdR1": dVdR1, "dVdC1": dVdC1, "dVdR2": dVdR2, "dVdC2": dVdC2
    }

# -------------------- WEIGHTED RIDGE SOLVER --------------------
def weighted_ridge(J, y, w_diag, lam):
    """
    Solve min ||W^{1/2}(y - J beta)||^2 + lam ||beta||^2
    without forming the NxN diagonal W. Works for very large N.
    J: [N x p], y: [N x 1], w_diag: length-N nonnegative weights
    """
    if w_diag is None:
        JTJ = J.T @ J
        JTy = J.T @ y
    else:
        w = np.asarray(w_diag, dtype=np.float64).reshape(-1)
        sw = np.sqrt(np.maximum(w, 1e-12))           # N
        Jt = J * sw[:, None]                         # scale rows of J
        yt = y * sw[:, None]                         # scale y
        JTJ = Jt.T @ Jt
        JTy = Jt.T @ yt
    lamI = lam * np.eye(J.shape[1], dtype=JTJ.dtype)
    beta = np.linalg.solve(JTJ + lamI, JTy)
    return beta


def mae(x): return float(np.mean(np.abs(x)))

# -------------------- MAIN --------------------
# Load LUT and build interpolators
lut = pd.read_csv(LUT_IN)
for col in ["R0","R1","C1","R2","C2"]:
    lut[col] = ensure_positive(lut[col])
interps = make_interpolators(lut)

# Load datasets
meas_w = sio.loadmat(WLTP_MAT)["meas"][0,0]
t_w = meas_w["Time"][:,0].astype(float)
i_w = meas_w["Current"][:,0].astype(float)
v_w = meas_w["Voltage"][:,0].astype(float)

meas_h = sio.loadmat(HPPC_MAT)["meas"][0,0]
t_h = meas_h["Time"][:,0].astype(float)
i_h = meas_h["Current"][:,0].astype(float)
v_h = meas_h["Voltage"][:,0].astype(float)

# Optional subsampling for speed
t_w, i_w, v_w = subsample(t_w, i_w, v_w, stride=SUBSAMPLE_W)
t_h, i_h, v_h = subsample(t_h, i_h, v_h, stride=SUBSAMPLE_H)

last_avg_mae = np.inf
F = features_sc(np.array([50.0]), np.array([0.5])).shape[1]  # #features

for it in range(1, N_ITER+1):
    print(f"\n=== Iteration {it} ===")

    # Baseline + sensitivities (fast) for both datasets
    simW = simulate_and_sensitivities(t_w, i_w, soc0=50.0,  interps=interps)
    simH = simulate_and_sensitivities(t_h, i_h, soc0=100.0, interps=interps)

    # Residuals
    e_w = v_w - simW["v_sim"]
    e_h = v_h - simH["v_sim"]

    # Build masks & weights
    dI_w = np.diff(i_w, prepend=i_w[0]); dI_h = np.diff(i_h, prepend=i_h[0])
    m_w = (np.abs(i_w) >= I_MIN_FOR_USE)
    m_h = (np.abs(i_h) >= I_MIN_FOR_USE)
    if not USE_ALL_SAMPLES:
        m_w &= (np.abs(dI_w) >= DI_EDGE_THRESH)
        m_h &= (np.abs(dI_h) >= DI_EDGE_THRESH)
    if SOC_WINDOW is not None:
        lo, hi = SOC_WINDOW
        m_w &= (simW["soc"] >= lo) & (simW["soc"] <= hi)
        m_h &= (simH["soc"] >= lo) & (simH["soc"] <= hi)

    # Features phi(SOC,C) at all time-steps
    Phi_w = features_sc(simW["soc"], simW["crate"])
    Phi_h = features_sc(simH["soc"], simH["crate"])

    # Project local sensitivities onto feature space:
    # J_param = diag(dV/dParam_k) @ Phi  => elementwise multiply each column by dV/dParam
    def proj(dV_local, Phi):
        # returns [N x F] matrix
        return dV_local.reshape(-1,1) * Phi

    JR1_w = proj(simW["dVdR1"], Phi_w)[m_w]
    JC1_w = proj(simW["dVdC1"], Phi_w)[m_w]
    JR2_w = proj(simW["dVdR2"], Phi_w)[m_w]
    JC2_w = proj(simW["dVdC2"], Phi_w)[m_w]

    JR1_h = proj(simH["dVdR1"], Phi_h)[m_h]
    JC1_h = proj(simH["dVdC1"], Phi_h)[m_h]
    JR2_h = proj(simH["dVdR2"], Phi_h)[m_h]
    JC2_h = proj(simH["dVdC2"], Phi_h)[m_h]

    # Stack Jacobian columns in PARAM order: [R1,F | C1,F | R2,F | C2,F]
    J_w = np.hstack([JR1_w, JC1_w, JR2_w, JC2_w])
    J_h = np.hstack([JR1_h, JC1_h, JR2_h, JC2_h])
    J = np.vstack([J_w, J_h])

    # Targets and weights
    y = np.concatenate([e_w[m_w], e_h[m_h]]).reshape(-1,1)
    w = np.concatenate([np.abs(dI_w[m_w])**EDGE_POWER_Q,
                        np.abs(dI_h[m_h])**EDGE_POWER_Q])

    # Solve weighted ridge for all 4 param groups at once
    beta_all = weighted_ridge(J, y, w, RIDGE_LAMBDA).ravel()

    # Split into groups (each length F)
    beta_R1 = beta_all[0*F:1*F]
    beta_C1 = beta_all[1*F:2*F]
    beta_R2 = beta_all[2*F:3*F]
    beta_C2 = beta_all[3*F:4*F]

    # ---- Apply capped updates to LUT (R0 untouched) ----
    lut2 = lut.copy()
    Sg = lut2["SOC"].values
    Cg = lut2["C_rate"].values
    Phi_g = features_sc(Sg, Cg)

    # Helper to apply one group
    def apply_group(col, beta, cap_rel):
        base = lut2[col].values
        d = Phi_g @ beta
        if SOC_WINDOW is not None:
            lo, hi = SOC_WINDOW
            mask = (Sg >= lo) & (Sg <= hi)
        else:
            mask = np.ones_like(Sg, dtype=bool)
        cap = cap_rel * base
        d = np.clip(d, -cap, cap)
        newv = base.copy()
        newv[mask] = ensure_positive(base[mask] + d[mask])
        lut2[col] = newv

    apply_group("R1", beta_R1, CAP_R_REL)
    apply_group("C1", beta_C1, CAP_C_REL)
    apply_group("R2", beta_R2, CAP_R_REL)
    apply_group("C2", beta_C2, CAP_C_REL)

    for col in ["R0","R1","C1","R2","C2"]:
        lut2[col] = ensure_positive(lut2[col])

    # Measure improvement
    interps2 = make_interpolators(lut2)
    v_w_new = simulate_and_sensitivities(t_w, i_w, 50.0,  interps2)["v_sim"]
    v_h_new = simulate_and_sensitivities(t_h, i_h, 100.0, interps2)["v_sim"]
    mae_w = mae(v_w - v_w_new)
    mae_h = mae(v_h - v_h_new)
    avg_mae = 0.5*(mae_w + mae_h)
    print(f"WLTP MAE: {mae_w:.4f} V | HPPC MAE: {mae_h:.4f} V | Avg: {avg_mae:.4f} V")

    # Accept & continue
    lut = lut2
    interps = interps2
    if last_avg_mae - avg_mae < IMPROVE_TOL:
        print("Early stop: improvement below tolerance.")
        break
    last_avg_mae = avg_mae

# -------------------- SAVE UPDATED LUT --------------------
lut.to_csv(LUT_OUT, index=False)
print("Saved RC-corrected LUT to:", LUT_OUT)

# -------------------- FINAL COMPARISON PLOTS --------------------
# Re-simulate base (pre-RC) and corrected (post-RC)
lut_base = pd.read_csv(LUT_IN)
for c in ["R0","R1","C1","R2","C2"]:
    lut_base[c] = ensure_positive(lut_base[c])
interps_base = make_interpolators(lut_base)
interps_corr = make_interpolators(lut)

sim_w_base = simulate_and_sensitivities(t_w, i_w, 50.0,  interps_base)
sim_h_base = simulate_and_sensitivities(t_h, i_h, 100.0, interps_base)
sim_w_corr = simulate_and_sensitivities(t_w, i_w, 50.0,  interps_corr)
sim_h_corr = simulate_and_sensitivities(t_h, i_h, 100.0, interps_corr)

res_w_base = v_w - sim_w_base["v_sim"]; res_w_corr = v_w - sim_w_corr["v_sim"]
res_h_base = v_h - sim_h_base["v_sim"]; res_h_corr = v_h - sim_h_corr["v_sim"]

def plot_dataset(prefix, t, v_meas, v_sim_base, v_sim_corr, err_base, err_corr, i, soc):
    plt.figure(); plt.plot(t, v_meas, linewidth=1.0, label="Measured")
    plt.plot(t, v_sim_base, linewidth=1.0, label="Sim (Base)")
    plt.plot(t, v_sim_corr, linewidth=1.0, label="Sim (RC-corrected)")
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

print(f"FINAL WLTP  MAE  Base: {mae(res_w_base):.4f} V  |  RC-corrected: {mae(res_w_corr):.4f} V")
print(f"FINAL HPPC  MAE  Base: {mae(res_h_base):.4f} V  |  RC-corrected: {mae(res_h_corr):.4f} V")

# WLTP plots
plot_dataset(
    prefix="WLTP",
    t=t_w,
    v_meas=v_w,
    v_sim_base=sim_w_base["v_sim"],
    v_sim_corr=sim_w_corr["v_sim"],
    err_base=res_w_base,
    err_corr=res_w_corr,
    i=i_w,
    soc=sim_w_base["soc"]
)

# HPPC plots
plot_dataset(
    prefix="HPPC",
    t=t_h,
    v_meas=v_h,
    v_sim_base=sim_h_base["v_sim"],
    v_sim_corr=sim_h_corr["v_sim"],
    err_base=res_h_base,
    err_corr=res_h_corr,
    i=i_h,
    soc=sim_h_base["soc"]
)

