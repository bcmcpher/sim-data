"""Strictly blind MRIQC + fMRIPrep aligned functional (4D BOLD) IQMs.

All metrics—including framewise motion estimates—are quantified solely from the
4D BOLD volume with zero access to simulation metadata.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from sim_data.io.orientation import compute_anatomical_priors
from sim_data.iqm.structural import compute_efc, compute_fwhm, compute_gsr

_EPS = 1e-6


def compute_dvars_series(
    data_4d: np.ndarray, brain_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute raw DVARS (dvars_nstd), standardized DVARS (dvars_std / fMRIPrep std_dvars), and voxel-std DVARS."""
    ts = data_4d[brain_mask].astype(np.float64)  # (V, T)
    if ts.shape[1] < 2 or ts.shape[0] < 8:
        z = np.zeros(max(1, ts.shape[1]), dtype=np.float32)
        return z, z, z

    diff = np.diff(ts, axis=1)
    dvars_nstd = np.sqrt(np.mean(diff**2, axis=0))

    vox_sd = np.std(ts, axis=1)
    v0 = ts[:, :-1] - np.mean(ts[:, :-1], axis=1, keepdims=True)
    v1 = ts[:, 1:] - np.mean(ts[:, 1:], axis=1, keepdims=True)
    ar1 = np.sum(v0 * v1, axis=1) / (
        np.sqrt(np.sum(v0**2, axis=1) * np.sum(v1**2, axis=1)) + _EPS
    )
    ar1 = np.clip(ar1, -0.95, 0.95)
    expected_sd = np.sqrt(np.maximum(np.mean(2.0 * (1.0 - ar1) * (vox_sd**2)), _EPS))
    dvars_std = dvars_nstd / (expected_sd + _EPS)

    vox_norm = diff / (vox_sd[:, None] + _EPS)
    dvars_vstd = np.sqrt(np.mean(vox_norm**2, axis=0))

    pad = np.array([0.0], dtype=np.float64)
    return (
        np.concatenate([pad, dvars_nstd]).astype(np.float32),
        np.concatenate([pad, dvars_std]).astype(np.float32),
        np.concatenate([pad, dvars_vstd]).astype(np.float32),
    )


def estimate_fd_proxy(
    data_4d: np.ndarray,
    brain_mask: np.ndarray,
    zooms: tuple[float, float, float],
) -> np.ndarray:
    """Blind image-only framewise motion proxy (mm) sensitive to both translation and rotation.

    Computes frame-to-frame L1 center-of-mass shifts across 6 directional brain half-spaces
    (left/right, anterior/posterior, inferior/superior) within a dilated foreground envelope,
    capturing both bulk translations and rotations around the head center.
    """
    n_t = data_4d.shape[3]
    fd = np.zeros(n_t, dtype=np.float32)
    if n_t < 2:
        return fd

    # Use full-volume positive coordinates inside a mild dilated envelope so shifted edges stay in view
    coords = np.argwhere(brain_mask).astype(np.float64)
    if coords.shape[0] < 32:
        return fd

    zoom_arr = np.array(zooms, dtype=np.float64)[:, None]
    centroid = np.mean(coords, axis=0)
    weights = np.maximum(data_4d[brain_mask], 0.0).astype(np.float64)  # (V, T)

    half_displacements: list[np.ndarray] = []
    for axis in range(3):
        for side_mask in (coords[:, axis] >= centroid[axis], coords[:, axis] < centroid[axis]):
            if int(np.sum(side_mask)) < 16:
                continue
            c_sub = coords[side_mask]  # (V_sub, 3)
            w_sub = weights[side_mask]  # (V_sub, T)
            w_sum = np.sum(w_sub, axis=0) + _EPS
            com_mm = ((c_sub.T @ w_sub) / w_sum[None, :]) * zoom_arr  # (3, T)
            half_displacements.append(np.sum(np.abs(np.diff(com_mm, axis=1)), axis=0))

    if half_displacements:
        # Mean L1 displacement across half-brain regions (in mm)
        fd[1:] = np.mean(np.vstack(half_displacements), axis=0).astype(np.float32)
    return fd


def extract_functional_iqms(
    data_4d: np.ndarray,
    zooms: tuple[float, float, float],
) -> dict[str, float]:
    """Extract MRIQC + fMRIPrep aligned IQMs strictly from a 4D BOLD volume."""
    arr = np.nan_to_num(np.asarray(data_4d, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 3:
        arr = arr[..., None]
    n_t = arr.shape[3]

    mean_vol = np.mean(arr, axis=3)
    priors = compute_anatomical_priors(mean_vol)
    brain = priors.brain_mask
    bg = priors.bg_mask

    fg_mean = float(np.mean(mean_vol[brain]))
    fg_std = float(np.std(mean_vol[brain]))
    snr = float(fg_mean / (fg_std + _EPS))
    efc = compute_efc(mean_vol)
    fg_e = float(np.mean(mean_vol[brain].astype(np.float64) ** 2))
    bg_e = float(np.mean(mean_vol[bg].astype(np.float64) ** 2)) if int(bg.sum()) > 0 else 0.0
    fber = float(min(fg_e / (bg_e + 1e-4), 1e5))

    fwhm_x, fwhm_y, fwhm_z, fwhm_avg = compute_fwhm(mean_vol, brain, zooms)
    gsr_x = compute_gsr(mean_vol, brain, axis=0)
    gsr_y = compute_gsr(mean_vol, brain, axis=1)

    ts = arr[brain].astype(np.float64)  # (V, T)
    if n_t > 1:
        t_mean = np.mean(ts, axis=1)
        t_std = np.std(ts, axis=1) + _EPS
        tsnr_vox = t_mean / t_std
        tsnr = float(np.median(tsnr_vox))
        tsnr_iqr = float(np.percentile(tsnr_vox, 75) - np.percentile(tsnr_vox, 25))
    else:
        tsnr, tsnr_iqr = 0.0, 0.0

    dvars_nstd_s, dvars_std_s, dvars_vstd_s = compute_dvars_series(arr, brain)
    dvars_nstd = float(np.mean(dvars_nstd_s[1:])) if n_t > 1 else 0.0
    dvars_std = float(np.mean(dvars_std_s[1:])) if n_t > 1 else 0.0
    dvars_vstd = float(np.mean(dvars_vstd_s[1:])) if n_t > 1 else 0.0

    # Strictly image-derived FD proxy (never reads simulation metadata)
    fd_series = estimate_fd_proxy(arr, brain, zooms)
    fd_mean = float(np.mean(fd_series))
    fd_max = float(np.max(fd_series))
    fd_num = int(np.sum(fd_series > 0.2))
    fd_perc = float(100.0 * fd_num / max(1, n_t))

    med_ts = np.median(ts, axis=1)
    aqi_vals: list[float] = []
    for t in range(n_t):
        r, _ = stats.spearmanr(ts[:, t], med_ts)
        if np.isfinite(r):
            aqi_vals.append(float(1.0 - r))
    aqi = float(np.mean(aqi_vals)) if aqi_vals else 0.0

    mad = np.median(np.abs(ts - med_ts[:, None]), axis=1) + _EPS
    z_mad = np.abs(ts - med_ts[:, None]) / (1.4826 * mad[:, None])
    aor = float(np.mean(z_mad > 3.5))

    global_sig = np.mean(ts, axis=0)
    gs_mean = float(np.mean(global_sig)) + _EPS
    global_signal_cv = float(np.std(global_sig) / gs_mean)
    if n_t > 2:
        t_norm = np.linspace(0.0, 1.0, n_t)
        slope = float(np.polyfit(t_norm, global_sig / gs_mean, 1)[0])
    else:
        slope = 0.0

    gs_med = float(np.median(global_sig))
    gs_mad = float(np.median(np.abs(global_sig - gs_med))) + _EPS
    dummy_trs = 0
    for t in range(min(5, n_t)):
        if abs(global_sig[t] - gs_med) > 3.5 * 1.4826 * gs_mad:
            dummy_trs += 1
        else:
            break

    return {
        "snr": round(snr, 5),
        "efc": round(efc, 5),
        "fber": round(fber, 5),
        "fwhm_x": round(fwhm_x, 5),
        "fwhm_y": round(fwhm_y, 5),
        "fwhm_z": round(fwhm_z, 5),
        "fwhm_avg": round(fwhm_avg, 5),
        "gsr_x": round(gsr_x, 5),
        "gsr_y": round(gsr_y, 5),
        "tsnr": round(tsnr, 5),
        "tsnr_iqr": round(tsnr_iqr, 5),
        "dvars_nstd": round(dvars_nstd, 5),
        "dvars_std": round(dvars_std, 5),
        "dvars_vstd": round(dvars_vstd, 5),
        "fd_mean": round(fd_mean, 5),
        "fd_max": round(fd_max, 5),
        "fd_num": float(fd_num),
        "fd_perc": round(fd_perc, 4),
        "aqi": round(aqi, 6),
        "aor": round(aor, 6),
        "dummy_trs": float(dummy_trs),
        "global_signal_cv": round(global_signal_cv, 6),
        "temporal_drift_slope": round(slope, 6),
    }
