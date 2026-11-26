"""Strictly blind MRIQC-aligned structural IQMs (T1w, T2w, FLAIR) + supplemental metrics.

Every metric is quantified solely from the modified 3D NIfTI volume with zero access
to clean baselines or simulation metadata.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from scipy import ndimage, stats

from ghibli.io.orientation import compute_anatomical_priors

_EPS = 1e-6


def compute_efc(vol: np.ndarray) -> float:
    """Entropy-Focus Criterion (Atkinson et al., 1997 / MRIQC): lower is sharper."""
    v = np.asarray(vol, dtype=np.float64)
    energy = float(np.sum(v * v))
    if energy <= _EPS:
        return 0.0
    bmax = np.sqrt(energy)
    normed = np.abs(v) / (bmax + _EPS)
    nonzero = normed[normed > 1e-12]
    if nonzero.size == 0:
        return 0.0
    n_vox = float(v.size)
    entropy = -float(np.sum(nonzero * np.log(nonzero)))
    max_entropy = float(np.sqrt(n_vox) * np.log(np.sqrt(n_vox))) if n_vox > 1 else 1.0
    return float(entropy / max(max_entropy, _EPS))


def compute_fwhm(
    vol: np.ndarray,
    mask: np.ndarray,
    zooms: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Forman/AFNI-style spatial smoothness FWHM (in mm) along x, y, z and average."""
    v = np.asarray(vol, dtype=np.float64)
    fg = v[mask]
    var_fg = float(np.var(fg))
    if var_fg <= _EPS:
        return (0.0, 0.0, 0.0, 0.0)

    fwhms: list[float] = []
    for axis in range(3):
        if v.shape[axis] < 3:
            fwhms.append(0.0)
            continue
        diff = np.diff(v, axis=axis)
        sl_a: list[slice] = [slice(None), slice(None), slice(None)]
        sl_b: list[slice] = [slice(None), slice(None), slice(None)]
        sl_a[axis] = slice(0, -1)
        sl_b[axis] = slice(1, None)
        pair_mask = mask[tuple(sl_a)] & mask[tuple(sl_b)]
        if int(pair_mask.sum()) < 16:
            fwhms.append(0.0)
            continue
        var_diff = float(np.var(diff[pair_mask]))
        ratio = np.clip(1.0 - var_diff / (2.0 * var_fg + _EPS), 1e-4, 0.9999)
        fwhm_vox = float(np.sqrt(-2.0 * np.log(2.0) / np.log(ratio)))
        fwhms.append(float(np.clip(fwhm_vox * zooms[axis], 0.0, 40.0)))

    avg = float(np.mean(fwhms))
    return (fwhms[0], fwhms[1], fwhms[2], avg)


def compute_qi1_qi2(bg_vals: np.ndarray) -> tuple[float, float]:
    """Mortamet QI1 (artifactual background fraction) and QI2 (chi-fit residual proxy)."""
    if bg_vals.size < 32:
        return (0.0, 0.0)
    bg = np.abs(bg_vals.astype(np.float64))
    pos = bg[bg > 0]
    if pos.size < 16:
        return (0.0, 0.0)

    p95 = float(np.percentile(pos, 95.0))
    if p95 <= _EPS:
        return (0.0, 0.0)
    hist, edges = np.histogram(pos[pos <= p95], bins=64)
    mode_idx = int(np.argmax(hist))
    mode_val = 0.5 * (edges[mode_idx] + edges[mode_idx + 1])

    mad = float(np.median(np.abs(pos - mode_val))) + _EPS
    thresh = mode_val + 4.5 * 1.4826 * mad
    qi1 = float(np.mean(bg > thresh))

    clean = pos[pos <= thresh]
    if clean.size < 32 or float(np.std(clean)) <= _EPS:
        return (qi1, 0.0)
    emp_hist, bin_edges = np.histogram(clean, bins=32, density=True)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    scale = max(float(np.sqrt(np.mean(clean**2) / 2.0)), _EPS)
    rayleigh_pdf = stats.rayleigh.pdf(centers, loc=0.0, scale=scale)
    bin_w = float(bin_edges[1] - bin_edges[0])
    qi2 = float(np.clip(np.sum(np.abs(emp_hist - rayleigh_pdf)) * bin_w * 0.5, 0.0, 1.0))
    return (qi1, qi2)


def compute_inu_n4(vol: np.ndarray, brain_mask: np.ndarray) -> tuple[float, float]:
    """MRIQC-aligned N4ITK bias field estimate (inu_med, inu_range) via SimpleITK."""
    fg = vol[brain_mask]
    med = float(np.median(fg))
    if med <= _EPS or int(brain_mask.sum()) < 64:
        return (1.0, 0.0)

    try:
        # SimpleITK expects strictly positive float32 values for log-space N4ITK estimation
        v_pos = np.maximum(vol.astype(np.float32), 1e-3)
        sitk_img = sitk.GetImageFromArray(v_pos)
        sitk_mask = sitk.GetImageFromArray(brain_mask.astype(np.uint8))

        shrink_factor = max(1, min(3, min(vol.shape) // 24))
        if shrink_factor > 1:
            img_small = sitk.Shrink(sitk_img, [shrink_factor] * 3)
            mask_small = sitk.Shrink(sitk_mask, [shrink_factor] * 3)
        else:
            img_small = sitk_img
            mask_small = sitk_mask

        n4 = sitk.N4BiasFieldCorrectionImageFilter()
        n4.SetMaximumNumberOfIterations([40, 30, 20])
        n4.SetBiasFieldFullWidthAtHalfMaximum(0.30)
        n4.SetConvergenceThreshold(1e-6)
        n4.Execute(img_small, mask_small)

        log_bias = n4.GetLogBiasFieldAsImage(sitk_img)
        bias_arr = np.exp(sitk.GetArrayFromImage(log_bias).astype(np.float64))
        vals = bias_arr[brain_mask]
        b_med = max(float(np.median(vals)), _EPS)
        rel = vals / b_med
        inu_med = float(np.median(vals))
        inu_range = float(np.percentile(rel, 95.0) - np.percentile(rel, 5.0))
        if np.isfinite(inu_med) and np.isfinite(inu_range):
            return (inu_med, inu_range)
    except Exception:
        pass

    # Fallback smooth convolution proxy on degenerate arrays
    sigma = tuple(max(3.0, s / 8.0) for s in vol.shape)
    num = ndimage.gaussian_filter(vol * brain_mask.astype(np.float32), sigma=sigma, mode="constant")
    den = ndimage.gaussian_filter(brain_mask.astype(np.float32), sigma=sigma, mode="constant")
    smooth = np.where(den > 0.15, num / np.maximum(den, _EPS), med)
    rel_fb = smooth[brain_mask] / med
    return (float(np.median(rel_fb)), float(np.percentile(rel_fb, 95.0) - np.percentile(rel_fb, 5.0)))


def compute_gsr(vol: np.ndarray, brain_mask: np.ndarray, axis: int) -> float:
    """MRIQC Ghost-to-Signal Ratio along a given axis: (mu_G - mu_NG) / mu_S."""
    n = vol.shape[axis]
    shifted = np.roll(brain_mask, shift=n // 2, axis=axis)
    ghost_mask = shifted & (~brain_mask)
    non_ghost_mask = (~shifted) & (~brain_mask)
    mu_s = float(np.mean(vol[brain_mask])) if int(brain_mask.sum()) > 0 else 0.0
    if mu_s <= _EPS or int(ghost_mask.sum()) < 8 or int(non_ghost_mask.sum()) < 8:
        return 0.0
    mu_g = float(np.mean(vol[ghost_mask]))
    mu_ng = float(np.mean(vol[non_ghost_mask]))
    return float((mu_g - mu_ng) / (mu_s + _EPS))


def extract_structural_iqms(
    vol: np.ndarray,
    zooms: tuple[float, float, float],
) -> dict[str, float]:
    """Extract MRIQC-aligned structural IQMs strictly from a single 3D volume."""
    v = np.nan_to_num(np.asarray(vol, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    priors = compute_anatomical_priors(v)

    brain = priors.brain_mask
    bg = priors.bg_mask
    wm = priors.wm_mask
    gm = priors.gm_mask

    fg_vals = v[brain]
    bg_vals = v[bg] if int(bg.sum()) > 0 else np.array([0.0], dtype=np.float32)
    wm_vals = v[wm] if int(wm.sum()) > 0 else fg_vals
    gm_vals = v[gm] if int(gm.sum()) > 0 else fg_vals

    fg_mean = float(np.mean(fg_vals))
    fg_std = float(np.std(fg_vals))
    fg_p95 = float(np.percentile(fg_vals, 95.0))
    bg_mean = float(np.mean(bg_vals))
    bg_std = float(np.std(bg_vals))
    bg_p95 = float(np.percentile(bg_vals, 95.0))

    wm_mean = float(np.mean(wm_vals))
    wm_std = float(np.std(wm_vals))
    wm_med = float(np.median(wm_vals))
    gm_mean = float(np.mean(gm_vals))
    gm_std = float(np.std(gm_vals))

    snr_total = float(fg_mean / (fg_std + _EPS))
    snr_wm = float(wm_mean / (wm_std + _EPS))
    snr_gm = float(gm_mean / (gm_std + _EPS))
    bg_sigma_dietrich = max(bg_std * 1.5264, bg_mean * 0.7979, _EPS)
    snrd_total = float(fg_mean / bg_sigma_dietrich)

    tissue_diff = abs(wm_mean - gm_mean)
    cnr = float(tissue_diff / np.sqrt(wm_std**2 + gm_std**2 + bg_std**2 + _EPS))
    cjv = float((wm_std + gm_std) / (tissue_diff + _EPS))

    p995 = float(np.percentile(v[v > 0], 99.5)) if np.any(v > 0) else 1.0
    wm2max = float(wm_med / (p995 + _EPS))

    efc = compute_efc(v)
    fg_energy = float(np.mean(fg_vals.astype(np.float64) ** 2))
    bg_energy = float(np.mean(bg_vals.astype(np.float64) ** 2))
    fber = float(min(fg_energy / (bg_energy + 1e-4), 1e5))

    qi1, qi2 = compute_qi1_qi2(bg_vals)
    inu_med, inu_range = compute_inu_n4(v, brain)
    fwhm_x, fwhm_y, fwhm_z, fwhm_avg = compute_fwhm(v, brain, zooms)

    # Blind supplemental metrics computed purely from the modified image:
    # 1. Edge clipping fraction (brain touching FOV faces)
    border = np.zeros_like(brain, dtype=bool)
    border[0, :, :] = border[-1, :, :] = True
    border[:, 0, :] = border[:, -1, :] = True
    border[:, :, 0] = border[:, :, -1] = True
    edge_touch = float(np.sum(brain & border) / max(1, int(np.sum(brain))))

    # 2. Exact-zero voxel fraction in anterior-inferior quadrant of the image (post-noise defacing detector)
    # Normal magnitude MRI has nonzero Rician background noise; post-noise defacing/slab-clipping leaves exact 0.0
    y_start = int(v.shape[1] * 0.55)
    z_end = int(v.shape[2] * 0.60)
    ant_inf_slab = v[:, y_start:, :z_end]
    ant_inf_zero = float(np.mean(ant_inf_slab <= 1e-6)) if ant_inf_slab.size > 0 else 0.0

    # 3. High-frequency energy ratio & k-space spike ratio
    k_mag = np.abs(np.fft.fftshift(np.fft.fftn(v.astype(np.float32))))
    kx, ky, kz = np.meshgrid(
        np.linspace(-1.0, 1.0, v.shape[0], dtype=np.float32),
        np.linspace(-1.0, 1.0, v.shape[1], dtype=np.float32),
        np.linspace(-1.0, 1.0, v.shape[2], dtype=np.float32),
        indexing="ij",
    )
    kr = np.sqrt(kx**2 + ky**2 + kz**2)
    outer = k_mag[kr > 0.55]
    total_k = float(np.sum(k_mag**2)) + _EPS
    hf_ratio = float(np.sum(outer**2) / total_k) if outer.size > 0 else 0.0

    mid_band = k_mag[(kr >= 0.12) & (kr <= 0.75)]
    if mid_band.size > 32:
        med_k = float(np.median(mid_band)) + _EPS
        kspace_spike_ratio = float(np.max(mid_band) / med_k)
    else:
        kspace_spike_ratio = 1.0

    return {
        "snr_total": round(snr_total, 5),
        "snr_wm": round(snr_wm, 5),
        "snr_gm": round(snr_gm, 5),
        "snrd_total": round(snrd_total, 5),
        "cnr": round(cnr, 5),
        "cjv": round(min(cjv, 50.0), 5),
        "wm2max": round(wm2max, 5),
        "efc": round(efc, 5),
        "fber": round(fber, 5),
        "qi1": round(qi1, 5),
        "qi2": round(qi2, 5),
        "inu_med": round(inu_med, 5),
        "inu_range": round(inu_range, 5),
        "fwhm_x": round(fwhm_x, 5),
        "fwhm_y": round(fwhm_y, 5),
        "fwhm_z": round(fwhm_z, 5),
        "fwhm_avg": round(fwhm_avg, 5),
        "summary_fg_mean": round(fg_mean, 5),
        "summary_fg_stdv": round(fg_std, 5),
        "summary_fg_p95": round(fg_p95, 5),
        "summary_bg_mean": round(bg_mean, 5),
        "summary_bg_stdv": round(bg_std, 5),
        "summary_bg_p95": round(bg_p95, 5),
        "summary_wm_mean": round(wm_mean, 5),
        "summary_gm_mean": round(gm_mean, 5),
        "edge_clip_fraction": round(edge_touch, 5),
        "anterior_inferior_zero_ratio": round(ant_inf_zero, 5),
        "high_freq_energy_ratio": round(hf_ratio, 6),
        "kspace_spike_ratio": round(min(kspace_spike_ratio, 1e4), 4),
    }
