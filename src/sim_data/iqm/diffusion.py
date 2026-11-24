"""Strictly blind QSIPrep ImageQC + MRIQC-dMRI aligned diffusion (4D DWI) IQMs.

All metrics are quantified solely from the modified 4D DWI NIfTI and .bval/.bvec files
with zero access to simulation metadata.
"""

from __future__ import annotations

import numpy as np

from sim_data.io.orientation import compute_anatomical_priors
from sim_data.iqm.functional import estimate_fd_proxy
from sim_data.iqm.structural import compute_efc, compute_fwhm, compute_gsr

_EPS = 1e-6


def compute_neighbor_dwi_corr(
    data_4d: np.ndarray,
    brain_mask: np.ndarray,
    bvals: np.ndarray | None = None,
    bvecs: np.ndarray | None = None,
) -> float:
    """Compute QSIPrep / DSI Studio raw_neighbor_corr (Neighboring DWI Correlation / NDC)."""
    n_vols = data_4d.shape[3]
    if n_vols < 2:
        return 1.0
    if bvals is not None and len(bvals) == n_vols:
        dw_idx = np.where(bvals > 150.0)[0]
    else:
        dw_idx = np.arange(n_vols)
    if dw_idx.size < 2:
        dw_idx = np.arange(n_vols)

    vecs = data_4d[brain_mask][:, dw_idx].astype(np.float64)  # (V, M)
    m = vecs.shape[1]
    centered = vecs - np.mean(vecs, axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0, keepdims=True) + _EPS
    unit = centered / norms

    corrs: list[float] = []
    if bvecs is not None and bvecs.shape == (3, n_vols):
        g = bvecs[:, dw_idx].astype(np.float64)
        g_norm = g / (np.linalg.norm(g, axis=0, keepdims=True) + _EPS)
        ang_sim = np.abs(g_norm.T @ g_norm)
        np.fill_diagonal(ang_sim, -1.0)
        neighbors = np.argmax(ang_sim, axis=1)
        for i in range(m):
            j = int(neighbors[i])
            corrs.append(float(np.dot(unit[:, i], unit[:, j])))
    else:
        for i in range(m - 1):
            corrs.append(float(np.dot(unit[:, i], unit[:, i + 1])))

    return float(np.clip(np.mean(corrs) if corrs else 0.0, -1.0, 1.0))


def compute_bad_slices_qsiprep(
    data_4d: np.ndarray,
    brain_mask: np.ndarray,
    slice_axis: int = 2,
    bvals: np.ndarray | None = None,
) -> tuple[int, np.ndarray]:
    """Count corrupted/dropped slices using QSIPrep/DSI Studio neighbor-slice difference heuristic."""
    n_vols = data_4d.shape[3]
    n_slices = data_4d.shape[slice_axis]
    moved = np.moveaxis(data_4d, slice_axis, 0)
    mask_m = np.moveaxis(brain_mask, slice_axis, 0)

    slice_means = np.zeros((n_vols, n_slices), dtype=np.float64)
    valid_z = np.zeros(n_slices, dtype=bool)
    for z in range(n_slices):
        m_z = mask_m[z]
        if int(m_z.sum()) >= 12:
            valid_z[z] = True
            slice_means[:, z] = np.mean(moved[z][m_z], axis=0)

    if bvals is not None and len(bvals) == n_vols:
        dw_mask = bvals > 150.0
        if not np.any(dw_mask):
            dw_mask = np.ones(n_vols, dtype=bool)
    else:
        dw_mask = np.ones(n_vols, dtype=bool)

    ratio_map = np.ones((n_vols, n_slices), dtype=np.float32)
    bad_count = 0
    valid_indices = np.where(valid_z)[0]

    for v in range(n_vols):
        if not dw_mask[v]:
            continue
        for idx_pos in range(1, len(valid_indices) - 1):
            z_prev = valid_indices[idx_pos - 1]
            z_curr = valid_indices[idx_pos]
            z_next = valid_indices[idx_pos + 1]
            s_prev = slice_means[v, z_prev]
            s_curr = slice_means[v, z_curr]
            s_next = slice_means[v, z_next]
            neighbor_avg = 0.5 * (s_prev + s_next) + _EPS
            ratio = s_curr / neighbor_avg
            ratio_map[v, z_curr] = float(ratio)

            diff_curr = abs(s_curr - s_prev) + abs(s_curr - s_next)
            diff_neigh = abs(s_next - s_prev)
            if diff_curr > 2.2 * diff_neigh and ratio < 0.78:
                bad_count += 1

    return bad_count, ratio_map


def extract_diffusion_iqms(
    data_4d: np.ndarray,
    zooms: tuple[float, float, float],
    bvals: np.ndarray | None = None,
    bvecs: np.ndarray | None = None,
    slice_axis: int = 2,
) -> dict[str, float]:
    """Extract QSIPrep desc-ImageQC_dwi + MRIQC-dMRI aligned IQMs strictly from the 4D DWI volume."""
    arr = np.nan_to_num(np.asarray(data_4d, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 3:
        arr = arr[..., None]
    n_vols = arr.shape[3]

    if bvals is not None and len(bvals) == n_vols:
        b0_idx = np.where(bvals <= 50.0)[0]
        high_b_idx = np.where(bvals > 150.0)[0]
        raw_max_b = float(np.max(bvals))
    else:
        b0_idx = np.array([0], dtype=int)
        high_b_idx = np.arange(1, n_vols) if n_vols > 1 else np.array([0], dtype=int)
        raw_max_b = 1000.0

    b0_vol = np.mean(arr[..., b0_idx], axis=3) if b0_idx.size > 0 else np.mean(arr, axis=3)
    priors = compute_anatomical_priors(b0_vol)
    brain = priors.brain_mask
    bg = priors.bg_mask

    bg_vals_b0 = b0_vol[bg] if int(bg.sum()) > 0 else np.array([0.0], dtype=np.float32)
    piesno_sigma = float(max(np.std(bg_vals_b0) * 1.5264, np.mean(bg_vals_b0) * 0.7979, _EPS))
    raw_b0_snr = float(np.mean(b0_vol[brain]) / piesno_sigma)

    efc_b0 = compute_efc(b0_vol)
    gsr_x_b0 = compute_gsr(b0_vol, brain, axis=0)
    gsr_y_b0 = compute_gsr(b0_vol, brain, axis=1)
    _, _, _, fwhm_avg_b0 = compute_fwhm(b0_vol, brain, zooms)

    if high_b_idx.size > 0:
        high_b_mean_vol = np.mean(arr[..., high_b_idx], axis=3)
        bg_high = high_b_mean_vol[bg] if int(bg.sum()) > 0 else bg_vals_b0
        sigma_high = float(max(np.std(bg_high) * 1.5264, np.mean(bg_high) * 0.7979, _EPS))
        snr_shell_high = float(np.mean(high_b_mean_vol[brain]) / sigma_high)
    else:
        snr_shell_high = raw_b0_snr

    raw_neighbor_corr = compute_neighbor_dwi_corr(arr, brain, bvals=bvals, bvecs=bvecs)
    raw_num_bad_slices, _ = compute_bad_slices_qsiprep(
        arr, brain, slice_axis=slice_axis, bvals=bvals
    )

    if high_b_idx.size >= 3:
        shell_ts = arr[brain][:, high_b_idx].astype(np.float64)
        mean_dw = np.mean(shell_ts, axis=1, keepdims=True) + _EPS
        cv_dw = float(np.median(np.std(shell_ts, axis=1) / mean_dw[:, 0]))
        raw_coherence_index = float(np.clip(1.0 / (1.0 + cv_dw), 0.0, 1.0))
    else:
        raw_coherence_index = 1.0

    if high_b_idx.size >= 3:
        dw_ts = arr[brain][:, high_b_idx].astype(np.float64)
        med = np.median(dw_ts, axis=1, keepdims=True)
        mad = np.median(np.abs(dw_ts - med), axis=1, keepdims=True) + _EPS
        spike_mask = np.abs(dw_ts - med) > 4.5 * 1.4826 * mad
        spikes_ppm = float(np.mean(spike_mask) * 1e6)
    else:
        spikes_ppm = 0.0

    # Strictly image-derived motion proxy (never reads simulation metadata)
    fd_series = estimate_fd_proxy(arr, brain, zooms)
    mean_fd = float(np.mean(fd_series))
    max_fd = float(np.max(fd_series))

    return {
        "raw_neighbor_corr": round(raw_neighbor_corr, 5),
        "raw_num_bad_slices": float(raw_num_bad_slices),
        "raw_coherence_index": round(raw_coherence_index, 5),
        "raw_b0_snr": round(raw_b0_snr, 5),
        "snr_shell_high": round(snr_shell_high, 5),
        "piesno_sigma": round(piesno_sigma, 5),
        "spikes_ppm": round(spikes_ppm, 2),
        "efc_b0": round(efc_b0, 5),
        "gsr_x_b0": round(gsr_x_b0, 5),
        "gsr_y_b0": round(gsr_y_b0, 5),
        "fwhm_avg_b0": round(fwhm_avg_b0, 5),
        "mean_fd": round(mean_fd, 5),
        "max_fd": round(max_fd, 5),
        "raw_dimension_x": float(arr.shape[0]),
        "raw_dimension_y": float(arr.shape[1]),
        "raw_dimension_z": float(arr.shape[2]),
        "raw_num_directions": float(high_b_idx.size),
        "raw_max_b": round(raw_max_b, 1),
    }
