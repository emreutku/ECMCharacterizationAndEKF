# ============================================================
# Build HPPC LUT from HPPC data, then refine (augment) with WLTP
# - Step 1 (HPPC): detect pulses, fit 2RC relaxation, save HPPC LUT
# - Step 2 (WLTP refine): add WLTP (SOC,C) points only within 45–65%SOC
#            using baseline interpolated params, save refined LUT and
#            run sims/plots (WLTP + HPPC)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.optimize import curve_fit
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, PchipInterpolator, interp1d

# -------------------- CONFIG --------------------
CAPACITY_AH = 2.5

# Path declaration
# BASE_LOOKUP_CSV = "C:\\Users\\EMRE\\Desktop\\Raisim_ws\\Exoskeleton_RL_project\\ecm_lookup_table_hppc.csv"
# WLTP_MAT = "C:\\Users\\EMRE\\Downloads\\Data Folder-20250805T053943Z-1-002\\Data Folder\\3-206k km case\\206k km case\\Cycle 0\\10-24-19_16.28 960_WLTP206a.mat"
# HPPC_MAT = "C:\\Users\\EMRE\\Downloads\\Data Folder-20250805T053943Z-1-002\\Data Folder\\3-206k km case\\206k km case\\Cycle 0\\10-16-19_20.16 948_HPPC.mat"
# FINAL_LOOKUP_CSV = "C:\\Users\\EMRE\\Desktop\\Raisim_ws\\Exoskeleton_RL_project\\ecm_lookup_table_refined_with_wltp.csv"

BASE_LOOKUP_CSV = "C:\\Users\\EMRE\\Documents\\MeaTechCaseStudy\\ecm_lookup_table_hppc.csv"
WLTP_MAT = "C:\\Users\\EMRE\\Documents\\MeaTechCaseStudy\\10-24-19_16.28 960_WLTP206a.mat"
HPPC_MAT = "C:\\Users\\EMRE\\Documents\\MeaTechCaseStudy\\10-16-19_20.16 948_HPPC.mat"
FINAL_LOOKUP_CSV = "C:\\Users\\EMRE\\Documents\\MeaTechCaseStudy\\ecm_lookup_table_refined_with_wltp.csv"

# HPPC extraction knobs
PULSE_DURATION = 10.0          # s of constant current before relaxation
RELAXATION_DURATION = 60.0     # s of relaxation window
CURRENT_THRESHOLD = 0.5        # A; treat smaller magnitudes as zero
CC_TOLERANCE = 1e-3            # A; CC segment flatness check


# WLTP augment window
SOC_REFINE_MIN = 45.0
SOC_REFINE_MAX = 65.0

# --------------- OCV(SOC) TABLE (taken from 1-A123 Aging Tests_Data Summary.xlsx) ---------------
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

# ============================================================
#                 PART A — HPPC → LUT (2RC)
# ============================================================

def coulomb_soc(time, current, cap_Ah, soc0=100.0):
    # SOC via Coulomb counting (positive current -> increases SOC)
    t = np.asarray(time, float)
    i = np.asarray(current, float)
    dt = np.diff(t, prepend=t[0])
    Ah = np.cumsum(i * dt) / 3600.0
    soc = soc0 + (Ah / cap_Ah) * 100.0
    return np.clip(soc, 0.0, 100.0)

def v_response_2rc(t, V0, R1, C1, R2, C2, I):
    # Two-RC relaxation response after a current step ends
    return V0 - I * R1 * (1 - np.exp(-t / (R1 * C1))) - I * R2 * (1 - np.exp(-t / (R2 * C2)))

def fit_2rc(t, v, I):
    # Fit V(t) with the two-RC relaxation model for a given step current I
    V0_guess = v[-1]
    dv = abs(v[0] - v[-1])
    R_guess = dv / max(abs(I), 1e-6) if dv > 1e-3 else 0.01
    bounds_lower = [np.min(v) - 0.1, 1e-5, 1, 1e-5, 1]
    bounds_upper = [np.max(v) + 0.1, max(1.0, 5*R_guess), 1e5, max(1.0, 5*R_guess), 1e5]
    guess = [V0_guess, R_guess*0.7, 100, R_guess*0.3, 1000]
    try:
        popt, _ = curve_fit(
            lambda tt, V0, R1, C1, R2, C2: v_response_2rc(tt, V0, R1, C1, R2, C2, I),
            t, v, p0=guess, bounds=(bounds_lower, bounds_upper), maxfev=10000
        )
        return popt  # V0, R1, C1, R2, C2
    except Exception:
        return [np.nan]*5

def detect_pulses(mask_nonzero):
    # Return start/end indices of contiguous True segments in 'mask_nonzero' (boolean)
    nz = np.where(mask_nonzero)[0]
    if len(nz) == 0:
        return [], []
    gaps = np.where(np.diff(nz) > 1)[0]
    starts = [nz[0]] + [nz[g+1] for g in gaps]
    ends   = [nz[g] for g in gaps] + [nz[-1]]
    return starts, ends

def process_pulses_for_lookup(hppc_mat_filename, out_csv_path):
    # Extract 2RC parameters from HPPC steps and save LUT
    data = sio.loadmat(hppc_mat_filename)
    meas = data["meas"][0, 0]
    time = meas["Time"].flatten().astype(float)
    voltage = meas["Voltage"].flatten().astype(float)
    current = meas["Current"].flatten().astype(float)

    # SOC via Coulomb counting (HPPC assumed near 100% start)
    dt = np.diff(time, prepend=time[0])
    Ah_used = np.cumsum(current * dt) / 3600.0
    soc = np.clip(100 + (Ah_used / CAPACITY_AH) * 100, 0, 100)

    # Zero-out tiny currents to segment pulses
    current_clean = current.copy()
    current_clean[np.abs(current_clean) < CURRENT_THRESHOLD] = 0.0

    # Find charge/discharge pulses (boolean masks)
    starts_ch, ends_ch   = detect_pulses(current_clean > 0)
    starts_dis, ends_dis = detect_pulses(current_clean < 0)
    # Back up one index to include the step edge into the segment
    starts_ch  = [s-1 for s in starts_ch  if s > 0]
    starts_dis = [s-1 for s in starts_dis if s > 0]

    def process(starts, ends, pulse_type):
        rows = []
        for idx_pulse, (start, end) in enumerate(zip(starts, ends), start=1):
            # Extend segment with a relaxation tail
            end_ext = end
            t_end = time[end]
            tmax = t_end + RELAXATION_DURATION
            k = end + 1
            while k < len(time):
                if time[k] > tmax: break
                if current_clean[k] != 0: break
                end_ext = k
                k += 1

            t_seg = time[start:end_ext+1] - time[start]
            v_seg = voltage[start:end_ext+1]
            i_seg = current[start:end_ext+1]
            soc0  = soc[start]

            # Require PULSE_DURATION seconds of constant current
            idx_end = np.where(t_seg <= PULSE_DURATION)[0]
            if len(idx_end) == 0:
                continue
            idx_end = idx_end[-1]
            if t_seg[idx_end] < 0.95 * PULSE_DURATION:
                continue

            # Check constant-current flatness in first 3 s
            i1 = np.searchsorted(t_seg, 1.0, side="left")
            i3 = np.searchsorted(t_seg, 3.0, side="left")
            init_mean  = np.mean(i_seg[i1:i3])
            final_mean = np.mean(i_seg[i3:idx_end+1])
            if abs(init_mean - final_mean) > CC_TOLERANCE:
                continue

            I_mag = abs(init_mean)
            if I_mag < 0.01:
                continue

            crate = I_mag / CAPACITY_AH

            # Instantaneous R0 from step edge (robustify denom, div by 0 protn)
            denom = max(abs(i_seg[1] - i_seg[0]), 1e-9)
            R0 = abs(v_seg[1] - v_seg[0]) / denom

            # Relaxation portion after the CC window
            if idx_end + 1 >= len(t_seg):
                continue
            t_relax = t_seg[idx_end+1:] - t_seg[idx_end+1]
            v_relax = v_seg[idx_end+1:]
            if len(t_relax) < 10:
                continue

            V0, R1, C1, R2, C2 = fit_2rc(t_relax, v_relax, init_mean)
            if np.isnan(R1):
                continue

            rows.append({
                "PulseType": pulse_type, "PulseIndex": idx_pulse,
                "SOC": soc0, "C_rate": crate,
                "R0": R0, "R1": R1, "C1": C1, "R2": R2, "C2": C2
            })
        return rows

    results = process(starts_ch, ends_ch, "charge") + process(starts_dis, ends_dis, "discharge")
    if not results:
        raise RuntimeError("No valid HPPC pulses were fitted to build the LUT.")

    df = pd.DataFrame(results).sort_values(["SOC", "C_rate"], ascending=[False, True]).reset_index(drop=True)
    df.to_csv(out_csv_path, index=False)
    print(f" Saved HPPC LUT to '{out_csv_path}'  (rows={len(df)})")
    return df

# ============================================================
#                 PART B — WLTP augment (NO R0 change)
# ============================================================

def ensure_positive(x, eps=1e-12):
    return np.maximum(x, eps)

def make_interpolators(df):
    # Build 2D interpolators (SOC, C_rate) -> param
    points = df[["SOC", "C_rate"]].values
    interps = {}
    for col in ["R0","R1","C1","R2","C2"]:
        vals = df[col].values
        lin = LinearNDInterpolator(points, vals, fill_value=np.nan)
        near = NearestNDInterpolator(points, vals)
        interps[col] = (lin, near)
    return interps

def interp_param(s, c, interps, name):
    lin, near = interps[name]
    val = lin(s, c)
    if np.isnan(val):
        val = near(s, c)
    return float(val)

def estimate_soc_from_current(time, current, soc0_pct):
    # Coulomb count; discharge positive decreases SOC for ECM sim
    t = np.asarray(time, dtype=float).reshape(-1)
    i = np.asarray(current, dtype=float).reshape(-1)
    dt = np.diff(t, prepend=t[0])
    if len(dt) > 1:
        dt[0] = dt[1]
    i_pos = -i  # discharge positive
    discharge_ah = np.cumsum(i_pos * dt) / 3600.0
    soc = soc0_pct - (discharge_ah / CAPACITY_AH) * 100.0
    return np.clip(soc, 0.0, 100.0), i_pos

# ---- 2RC ECM simulation (uses FIXED OCV_FN) ----
def simulate_ecm(time, current, soc0_pct, interps, ocv_func):
    t = np.asarray(time, dtype=float).reshape(-1)
    i_meas = np.asarray(current, dtype=float).reshape(-1)
    soc, i_pos = estimate_soc_from_current(t, i_meas, soc0_pct)
    dt = np.diff(t, prepend=t[0])
    if len(dt) > 1:
        dt[0] = dt[1]
    n = len(t)
    v1 = np.zeros(n); v2 = np.zeros(n); v_sim = np.zeros(n)
    r0_arr = np.zeros(n); r1_arr = np.zeros(n); r2_arr = np.zeros(n)
    c1_arr = np.zeros(n); c2_arr = np.zeros(n)
    crate = np.abs(i_pos) / CAPACITY_AH

    v1_k = 0.0; v2_k = 0.0
    for k in range(n):
        s = soc[k]; c = crate[k]
        R0 = ensure_positive(interp_param(s, c, interps, "R0"))
        R1 = ensure_positive(interp_param(s, c, interps, "R1"))
        C1 = ensure_positive(interp_param(s, c, interps, "C1"))
        R2 = ensure_positive(interp_param(s, c, interps, "R2"))
        C2 = ensure_positive(interp_param(s, c, interps, "C2"))
        r0_arr[k] = R0; r1_arr[k] = R1; r2_arr[k] = R2; c1_arr[k] = C1; c2_arr[k] = C2

        a1 = np.exp(-dt[k] / (R1*C1))
        a2 = np.exp(-dt[k] / (R2*C2))
        v1_k = a1 * v1_k + (1 - a1) * R1 * i_pos[k]
        v2_k = a2 * v2_k + (1 - a2) * R2 * i_pos[k]
        v1[k] = v1_k
        v2[k] = v2_k
        v_sim[k] = float(ocv_func(s)) - i_pos[k] * R0 - v1_k - v2_k

    return {
        "time": t, "i_meas": i_meas, "i_pos": i_pos, "soc": soc,
        "v_sim": v_sim, "v1": v1, "v2": v2,
        "r0": r0_arr, "r1": r1_arr, "r2": r2_arr, "c1": c1_arr, "c2": c2_arr,
        "crate": crate
    }

# ---- Add WLTP (SOC,C_rate) points WITHOUT changing any parameter values ----
def add_wltp_points(df, soc_series, crate_series, interps, soc_min, soc_max):

    # Adds new rows at WLTP-observed (SOC, C_rate) inside [soc_min, soc_max],
    # using baseline interpolated parameters (R0, R1, C1, R2, C2) WITHOUT any change.

    s = np.asarray(soc_series)
    c = np.asarray(crate_series)
    mask = (s >= soc_min) & (s <= soc_max)
    s_sel = s[mask]
    c_sel = c[mask]

    # Bin to reduce duplication and grid explosion
    s_bins = np.round(s_sel * 2) / 2.0   # 0.5% SOC bins
    c_bins = np.round(c_sel, 2)          # 0.01C bins

    seen = set()
    rows = []
    for S, C in zip(s_bins, c_bins):
        key = (float(S), float(C))
        if key in seen:
            continue
        seen.add(key)
        R0 = interp_param(S, C, interps, "R0")
        R1 = interp_param(S, C, interps, "R1")
        C1 = interp_param(S, C, interps, "C1")
        R2 = interp_param(S, C, interps, "R2")
        C2 = interp_param(S, C, interps, "C2")
        rows.append({
            "PulseType":"WLTP_augmented","PulseIndex":-1,"SOC":float(S),"C_rate":float(C),
            "R0":R0,"R1":R1,"C1":C1,"R2":R2,"C2":C2
        })
    if not rows:
        return df.copy()

    df_new = pd.DataFrame(rows)
    df_combined = pd.concat([df, df_new], ignore_index=True)
    # Prefer the WLTP_augmented row for duplicates
    df_combined = df_combined.sort_values(["SOC","C_rate","PulseType"]).drop_duplicates(
        subset=["SOC","C_rate"], keep="last"
    )
    return df_combined.reset_index(drop=True)

def plot_series(label_prefix, time, series, ylabel):
    plt.figure()
    plt.plot(time, series, linewidth=1.0)
    plt.xlabel("Time [s]")
    plt.ylabel(ylabel)
    plt.title(f"{label_prefix} {ylabel}")
    plt.grid(True)
    plt.show()

# ============================================================
#                           MAIN
# ============================================================
if __name__ == "__main__":
    # --------- A) Build HPPC LUT from HPPC data and save ---------
    hppc_lut_df = process_pulses_for_lookup(HPPC_MAT, BASE_LOOKUP_CSV)

    # --------- B) WLTP refinement (augmentation only; NO R0 change) ---------
    # Use the just-saved HPPC LUT as baseline
    base_df = pd.read_csv(BASE_LOOKUP_CSV)
    for col in ["R0","R1","C1","R2","C2"]:
        base_df[col] = ensure_positive(base_df[col])
    interps_base = make_interpolators(base_df)

    # --- WLTP data ---
    wltp = sio.loadmat(WLTP_MAT)
    meas_w = wltp["meas"][0,0]
    t_w = meas_w["Time"][:,0].astype(float)
    i_w = meas_w["Current"][:,0].astype(float)
    v_w = meas_w["Voltage"][:,0].astype(float)

    # SOC starts at 50% for WLTP
    soc_w, i_w_pos = estimate_soc_from_current(t_w, i_w, soc0_pct=50.0)
    crate_w = np.abs(i_w_pos) / CAPACITY_AH

    # Baseline simulation on WLTP 
    sim_w_base = simulate_ecm(t_w, i_w, soc0_pct=50.0, interps=interps_base, ocv_func=OCV_FN)

    # --- Augment LUT with WLTP points INSIDE [45,65]% SOC (NO parameter changes) ---
    refined_df = base_df.copy()  # keep all original values
    interps_for_aug = make_interpolators(refined_df)
    refined_aug_df = add_wltp_points(refined_df, soc_w, crate_w, interps_for_aug, SOC_REFINE_MIN, SOC_REFINE_MAX)

    # Build final interpolators from the refined/augmented table
    interps_final = make_interpolators(refined_aug_df)

    # Re-simulate WLTP using final table (grid denser; params unchanged)
    sim_w_final = simulate_ecm(t_w, i_w, soc0_pct=50.0, interps=interps_final, ocv_func=OCV_FN)
    err_w_final = v_w - sim_w_final["v_sim"]

    # Save refined (augmented) LUT
    refined_aug_df.to_csv(FINAL_LOOKUP_CSV, index=False)
    print("Saved refined (no-R0-change) lookup to:", FINAL_LOOKUP_CSV)

    # ---- Plot WLTP results ----
    plot_series("WLTP", t_w, v_w,                    "Measured Voltage [V]")
    plot_series("WLTP", t_w, sim_w_final["v_sim"],   "Simulated Voltage [V]")
    plot_series("WLTP", t_w, err_w_final,            "Voltage Error [V]")
    plot_series("WLTP", t_w, sim_w_final["soc"],     "SOC [%]")
    plot_series("WLTP", t_w, i_w,                    "Current [A]")

    # --- HPPC simulation using final (augmented) table  ---
    hppc = sio.loadmat(HPPC_MAT)
    meas_h = hppc["meas"][0,0]
    t_h = meas_h["Time"][:,0].astype(float)
    i_h = meas_h["Current"][:,0].astype(float)
    v_h = meas_h["Voltage"][:,0].astype(float)

    # For HPPC, assume start at 100% SOC
    sim_h_final = simulate_ecm(t_h, i_h, soc0_pct=100.0, interps=interps_final, ocv_func=OCV_FN)
    err_h_final = v_h - sim_h_final["v_sim"]

    # ---- Plot HPPC results ----
    plot_series("HPPC", t_h, v_h,                    "Measured Voltage [V]")
    plot_series("HPPC", t_h, sim_h_final["v_sim"],   "Simulated Voltage [V]")
    plot_series("HPPC", t_h, err_h_final,            "Voltage Error [V]")
    plot_series("HPPC", t_h, sim_h_final["soc"],     "SOC [%]")
    plot_series("HPPC", t_h, i_h,                    "Current [A]")
