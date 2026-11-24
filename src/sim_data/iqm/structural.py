"""MRIQC-aligned structural IQMs (T1w, T2w, FLAIR) + supplemental blind-spot metrics."""

from __future__ import annotations

import numpy as np
from scipy import ndimage, stats

from sim_data.io.orientation import AnatomicalPriors, compute_anatomical_priors

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
    # Normalize to [0, 1] scale matching MRIQC's normalized EFC convention
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

    # Estimate noise mode via histogram on lower 95% of positive background
    p95 = float(np.percentile(pos, 95.0))
    if p95 <= _EPS:
        return (0.0, 0.0)
    hist, edges = np.histogram(pos[pos <= p95], bins=64)
    mode_idx = int(np.argmax(hist))
    mode_val = 0.5 * (edges[mode_idx] + edges[mode_idx + 1])

    # QI1: fraction of background voxels far above symmetric noise tail (artifactual background)
    mad = float(np.median(np.abs(pos - mode_val))) + _EPS
    thresh = mode_val + 4.5 * 1.4826 * mad
    qi1 = float(np.mean(bg > thresh))

    # QI2: goodness-of-fit L1 residual between empirical clean background PDF and fitted Rayleigh/chi
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


def compute_inu_proxy(vol: np.ndarray, brain_mask: np.ndarray) -> tuple[float, float]:
    """Smooth low-frequency bias field proxy approximating N4ITK INU (inu_med, inu_range)."""
    fg = vol[brain_mask]
    med = float(np.median(fg))
    if med <= _EPS or int(brain_mask.sum()) < 64:
        return (1.0, 0.0)
    # Estimate smooth multiplicative envelope via normalized convolution
    sigma = tuple(max(3.0, s / 8.0) for s in vol.shape)
    num = ndimage.gaussian_filter(vol * brain_mask.astype(np.float32), sigma=sigma, mode="constant")
    den = ndimage.gaussian_filter(brain_mask.astype(np.float32), sigma=sigma, mode="constant")
    smooth = np.where(den > 0.15, num / np.maximum(den, _EPS), med)
    rel = smooth[brain_mask] / med
    inu_med = float(np.median(rel))
    inu_range = float(np.percentile(rel, 95.0) - np.percentile(rel, 5.0))
    return (inu_med, inu_range)


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
    clean_priors: AnatomicalPriors | None = None,
) -> dict[str, float]:
    """Extract MRIQC-aligned structural IQMs + supplemental blind-spot metrics from a 3D volume."""
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

    # MRIQC SNR variants
    snr_total = float(fg_mean / (fg_std + _EPS))
    snr_wm = float(wm_mean / (wm_std + _EPS))
    snr_gm = float(gm_mean / (gm_std + _EPS))
    # Dietrich SNR (background Rayleigh correction factor sqrt(2 / (4 - pi)) ~ 1.526)
    bg_sigma_dietrich = max(bg_std * 1.5264, bg_mean * 0.7979, _EPS)
    snrd_total = float(fg_mean / bg_sigma_dietrich)

    # CNR & CJV (Ganzetti et al.)
    tissue_diff = abs(wm_mean - gm_mean)
    cnr = float(tissue_diff / np.sqrt(wm_std**2 + gm_std**2 + bg_std**2 + _EPS))
    cjv = float((wm_std + gm_std) / (tissue_diff + _EPS))

    # WM2Max
    p995 = float(np.percentile(v[v > 0], 99.5)) if np.any(v > 0) else 1.0
    wm2max = float(wm_med / (p995 + _EPS))

    # EFC & FBER
    efc = compute_efc(v)
    fg_energy = float(np.mean(fg_vals.astype(np.float64) ** 2))
    bg_energy = float(np.mean(bg_vals.astype(np.float64) ** 2))
    fber = float(min(fg_energy / (bg_energy + 1e-4), 1e5))

    # Mortamet QI1 & QI2
    qi1, qi2 = compute_qi1_qi2(bg_vals)

    # INU proxy & FWHM
    inu_med, inu_range = compute_inu_proxy(v, brain)
    fwhm_x, fwhm_y, fwhm_z, fwhm_avg = compute_fwhm(v, brain, zooms)

    # Supplemental blind-spot metrics:
    # 1. Edge clipping fraction (fraction of foreground voxels touching any of the 6 FOV boundary planes)
    border = np.zeros_like(brain, dtype=bool)
    border[0, :, :] = border[-1, :, :] = True
    border[:, 0, :] = border[:, -1, :] = True
    border[:, :, 0] = border[:, :, -1] = True
    # Also detect sharp interior slab zero-cuts by checking sharp zero-plane transitions near clean bbox
    edge_touch = float(np.sum(brain & border) / max(1, int(np.sum(brain))))

    # 2. Anterior-inferior zero ratio (detects post-noise defacing cuts!)
    bmin = clean_priors.bbox_min if (clean_priors is not None and clean_priors.brain_mask.shape == v.shape) else priors.bbox_min
    bmax = clean_priors.bbox_max if (clean_priors is not None and clean_priors.brain_mask.shape == v.shape) else priors.bbox_max
    y_mid = int(bmin[1] + 0.65 * (bmax[1] - bmin[1]))
    z_mid = int(bmin[2] + 0.55 * (bmax[2] - bmin[2]))
    ant_inf_ref = (
        clean_priors.brain_mask
        if (clean_priors is not None and clean_priors.brain_mask.shape == v.shape)
        else brain
    )
    ant_inf_roi = np.zeros_like(brain, dtype=bool)
    ant_inf_roi[:, y_mid : bmax[1] + 1, bmin[2] : z_mid + 1] = True
    ant_inf_brain = ant_inf_ref & ant_inf_roi
    if int(ant_inf_brain.sum()) > 16:
        ant_inf_zero = float(np.mean(v[ant_inf_brain] <= 1e-5))
    else:
        ant_inf_zero = 0.0

    # 3. High-frequency energy ratio & k-space spike ratio (catches RF spikes, Gibbs, noise)
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
