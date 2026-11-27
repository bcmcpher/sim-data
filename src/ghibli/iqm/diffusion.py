"""Strictly blind QSIPrep ImageQC + MRIQC-dMRI aligned diffusion (4D DWI) IQMs.

All metrics are quantified solely from the modified 4D DWI NIfTI and .bval/.bvec files
with zero access to simulation metadata.
"""

from __future__ import annotations

import numpy as np
from dipy.core.gradients import gradient_table
from dipy.denoise.noise_estimate import estimate_sigma, piesno
from scipy import ndimage
from skimage.registration import phase_cross_correlation

from ghibli.io.orientation import compute_anatomical_priors
from ghibli.iqm.structural import compute_efc, compute_fwhm, compute_gsr

_EPS = 1e-6


def estimate_dwi_fd_proxy(
    data_4d: np.ndarray,
    brain_mask: np.ndarray,
    zooms: tuple[float, float, float],
    bvals: np.ndarray | None = None,
) -> np.ndarray:
    """Shell-aware blind framewise motion proxy (mm) for multi-shell 4D DWI.

    Prevents cross-shell contrast jumps (b=0 -> b=1000 -> b=2000) and anisotropic WM
    fiber-direction attenuation from masquerading as head motion by:
    1. Converting each volume into a direction-invariant outer brain silhouette (clamped
       smoothly between background air p99 and low foreground p15 so GM and both parallel
       and perpendicular WM saturate cleanly to 1.0 regardless of gradient orientation).
    2. Registering consecutive volumes within the same b-shell (250 s/mm^2 bins) via
       3D sub-voxel phase cross-correlation across left/right hemispheres (capturing both
       rigid translation and rotation while reporting ~0.0 mm on stationary multi-shell DWI).
    """
    n_vols = data_4d.shape[3]
    fd = np.zeros(n_vols, dtype=np.float32)
    if n_vols < 2 or int(brain_mask.sum()) < 32:
        return fd

    bg = ~ndimage.binary_dilation(brain_mask, iterations=5)
    raw_all = np.maximum(data_4d, 0.0).astype(np.float32)
    sils = np.zeros_like(raw_all, dtype=np.float32)

    for v in range(n_vols):
        vol = raw_all[..., v]
        bg_v = vol[bg] if int(bg.sum()) >= 32 else np.array([0.0], dtype=np.float32)
        bg_hi = float(np.percentile(bg_v, 99.0))
        fg_lo = max(bg_hi * 1.5 + 1e-4, float(np.percentile(vol[brain_mask], 15.0)))
        sils[..., v] = np.clip((vol - bg_hi) / max(fg_lo - bg_hi, _EPS), 0.0, 1.0)

    if bvals is not None and len(bvals) == n_vols:
        shells = np.where(bvals <= 50.0, 0, np.rint(bvals / 250.0).astype(int) * 250)
    else:
        shells = np.zeros(n_vols, dtype=int)

    zoom_arr = np.array(zooms, dtype=np.float64)
    coords = np.argwhere(brain_mask)
    cx = int(round(float(np.mean(coords[:, 0]))))
    use_hemis = cx >= 8 and (sils.shape[0] - cx) >= 8

    last_idx_by_shell: dict[int, int] = {}
    for v in range(n_vols):
        sh = int(shells[v])
        if sh in last_idx_by_shell:
            prev_v = last_idx_by_shell[sh]
            if use_hemis:
                shift_l, _, _ = phase_cross_correlation(
                    sils[:cx, :, :, prev_v],
                    sils[:cx, :, :, v],
                    upsample_factor=10,
                    normalization=None,
                )
                shift_r, _, _ = phase_cross_correlation(
                    sils[cx:, :, :, prev_v],
                    sils[cx:, :, :, v],
                    upsample_factor=10,
                    normalization=None,
                )
                disp_l = float(np.sum(np.abs(shift_l) * zoom_arr))
                disp_r = float(np.sum(np.abs(shift_r) * zoom_arr))
                fd[v] = float(0.5 * (disp_l + disp_r))
            else:
                shift, _, _ = phase_cross_correlation(
                    sils[..., prev_v],
                    sils[..., v],
                    upsample_factor=10,
                    normalization=None,
                )
                fd[v] = float(np.sum(np.abs(shift) * zoom_arr))
        last_idx_by_shell[sh] = v

    return fd


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
        # Match multi-shell QSIPrep NDC: restrict nearest neighbor search to the same b-shell
        dw_bvals = bvals[dw_idx] if (bvals is not None and len(bvals) == n_vols) else None
        for i in range(m):
            sims = ang_sim[i].copy()
            if dw_bvals is not None:
                same_shell = np.abs(dw_bvals - dw_bvals[i]) < 150.0
                same_shell[i] = False
                if np.any(same_shell):
                    sims[~same_shell] = -2.0
            j = int(np.argmax(sims))
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
        bvecs_for_gtab = (
            bvecs.T if (bvecs is not None and bvecs.shape == (3, n_vols)) else None
        )
        gtab = gradient_table(bvals, bvecs=bvecs_for_gtab, b0_threshold=50.0)
        b0_idx = np.where(gtab.b0s_mask)[0]
        high_b_idx = np.where(~gtab.b0s_mask)[0]
        raw_max_b = float(np.max(bvals))
    else:
        b0_idx = np.array([0], dtype=int)
        high_b_idx = np.arange(1, n_vols) if n_vols > 1 else np.array([0], dtype=int)
        raw_max_b = 1000.0

    b0_vol = np.mean(arr[..., b0_idx], axis=3) if b0_idx.size > 0 else np.mean(arr, axis=3)
    priors = compute_anatomical_priors(b0_vol)
    brain = priors.brain_mask
    bg = priors.bg_mask

    # Genuine Koay et al. (2009) PIESNO noise estimation via DIPY (with estimate_sigma & Rayleigh fallback)
    piesno_sigma = 0.0
    try:
        p_sig = np.asarray(piesno(arr, N=1, return_mask=False), dtype=np.float64)
        pos_sig = p_sig[np.isfinite(p_sig) & (p_sig > 1e-4)]
        if pos_sig.size > 0:
            piesno_sigma = float(np.median(pos_sig))
    except Exception:
        piesno_sigma = 0.0

    if piesno_sigma <= 1e-4:
        try:
            est = np.asarray(estimate_sigma(arr, N=1), dtype=np.float64)
            pos_est = est[np.isfinite(est) & (est > 1e-4)]
            if pos_est.size > 0:
                piesno_sigma = float(np.median(pos_est))
        except Exception:
            piesno_sigma = 0.0

    bg_vals_b0 = b0_vol[bg] if int(bg.sum()) > 0 else np.array([0.0], dtype=np.float32)
    if piesno_sigma <= 1e-4:
        piesno_sigma = float(max(np.std(bg_vals_b0) * 1.5264, np.mean(bg_vals_b0) * 0.7979, _EPS))
    raw_b0_snr = float(np.mean(b0_vol[brain]) / max(piesno_sigma, _EPS))

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

    # Strictly image-derived shell-aware motion proxy (never reads simulation metadata)
    fd_series = estimate_dwi_fd_proxy(arr, brain, zooms, bvals=bvals)
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
