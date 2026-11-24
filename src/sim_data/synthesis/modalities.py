"""Forward MRI physics synthesizers: Bloch structural (T1w/T2w/FLAIR), 4D BOLD, and 4D DWI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from sim_data.artifacts.base import resolve_device
from sim_data.artifacts.noise import apply_noncentral_chi_noise
from sim_data.config import (
    BoldSynthesisConfig,
    DwiSynthesisConfig,
    Modality,
    StructuralSynthesisConfig,
)
from sim_data.synthesis.tissue import TissueMRIParams, TissuePVEMaps, resample_nifti_to_grid
from sim_data.synthesis.warp import DiffeomorphicField, warp_vector_field_ppd, warp_volume


def _fibonacci_hemisphere(n_dirs: int, rng: np.random.Generator) -> np.ndarray:
    """Generate evenly distributed unit gradient vectors on the unit hemisphere (3, n_dirs)."""
    indices = np.arange(n_dirs, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - indices / max(n_dirs, 1))
    theta = np.pi * (1.0 + 5.0**0.5) * indices + float(rng.uniform(0.0, 2.0 * np.pi))
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    vecs = np.vstack([x, y, z])
    norms = np.linalg.norm(vecs, axis=0, keepdims=True) + 1e-8
    return np.asarray(vecs / norms, dtype=np.float32)


def _resample_pve_to_spacing(
    pve: TissuePVEMaps,
    src_affine: np.ndarray,
    src_zooms: tuple[float, float, float],
    target_zooms: tuple[float, float, float],
) -> tuple[TissuePVEMaps, np.ndarray]:
    """Resample 3D tissue PVE maps to a coarser/finer voxel grid and update the NIfTI affine."""
    src_shape = np.array(pve.wm.shape, dtype=np.float64)
    scale_factors = np.array(src_zooms, dtype=np.float64) / np.maximum(
        np.array(target_zooms, dtype=np.float64), 0.2
    )
    new_shape = tuple(max(12, int(round(s * f))) for s, f in zip(src_shape, scale_factors, strict=True))
    actual_zoom = tuple(
        float(new_shape[i]) / float(pve.wm.shape[i]) for i in range(3)
    )

    csf = np.clip(ndimage.zoom(pve.csf, actual_zoom, order=1), 0.0, 1.0).astype(np.float32)
    gm = np.clip(ndimage.zoom(pve.gm, actual_zoom, order=1), 0.0, 1.0).astype(np.float32)
    wm = np.clip(ndimage.zoom(pve.wm, actual_zoom, order=1), 0.0, 1.0).astype(np.float32)
    scalp = np.clip(ndimage.zoom(pve.scalp, actual_zoom, order=1), 0.0, 1.0).astype(np.float32)

    total = csf + gm + wm + scalp
    head_mask = total > 0.15
    denom = np.where(head_mask, np.maximum(total, 1e-5), 1.0)
    csf = np.where(head_mask, csf / denom, 0.0).astype(np.float32)
    gm = np.where(head_mask, gm / denom, 0.0).astype(np.float32)
    wm = np.where(head_mask, wm / denom, 0.0).astype(np.float32)
    scalp = np.where(head_mask, scalp / denom, 0.0).astype(np.float32)
    brain_mask = (csf + gm + wm) > 0.35

    new_affine = src_affine.copy()
    for i in range(3):
        ratio = float(pve.wm.shape[i]) / float(new_shape[i])
        new_affine[:3, i] = src_affine[:3, i] * ratio

    return (
        TissuePVEMaps(
            csf=csf,
            gm=gm,
            wm=wm,
            scalp=scalp,
            brain_mask=np.asarray(brain_mask, dtype=bool),
            head_mask=np.asarray(head_mask, dtype=bool),
        ),
        new_affine,
    )


def synthesize_structural_modality(
    pve: TissuePVEMaps,
    warped_t1: np.ndarray,
    warped_t2: np.ndarray | None,
    modality: Modality,
    params: TissueMRIParams,
    cfg: StructuralSynthesisConfig,
    rng: np.random.Generator,
    device: str = "auto",
) -> tuple[np.ndarray, dict[str, float]]:
    """Synthesize clean T1w, T2w, or FLAIR volume using Bloch equations + real seed micro-texture."""
    comp_map = {"wm": pve.wm, "gm": pve.gm, "csf": pve.csf, "scalp": pve.scalp}
    signal = np.zeros(pve.wm.shape, dtype=np.float32)

    if modality == "T1w":
        tr_ms, te_ms, ti_ms = 650.0, 10.0, 0.0
        for name, frac in comp_map.items():
            w = (
                params.rho[name]
                * (1.0 - np.exp(-tr_ms / params.t1_ms[name]))
                * np.exp(-te_ms / params.t2_ms[name])
            )
            signal += frac * float(w)
        sidecar_timing = {"RepetitionTime": 2.3, "EchoTime": 0.0029}
    elif modality == "T2w":
        tr_ms, te_ms, ti_ms = 4500.0, 100.0, 0.0
        for name, frac in comp_map.items():
            w = (
                params.rho[name]
                * (1.0 - np.exp(-tr_ms / params.t1_ms[name]))
                * np.exp(-te_ms / params.t2_ms[name])
            )
            signal += frac * float(w)
        sidecar_timing = {"RepetitionTime": 3.2, "EchoTime": 0.408}
    else:  # FLAIR (CSF-nulled inversion recovery)
        tr_ms, te_ms = 9000.0, 110.0
        t1_csf = params.t1_ms["csf"]
        # Null point for CSF longitudinal magnetization
        ti_ms = float(t1_csf * np.log(2.0 / (1.0 + np.exp(-tr_ms / t1_csf))))
        for name, frac in comp_map.items():
            ir_factor = abs(
                1.0
                - 2.0 * np.exp(-ti_ms / params.t1_ms[name])
                + np.exp(-tr_ms / params.t1_ms[name])
            )
            w = params.rho[name] * ir_factor * np.exp(-te_ms / params.t2_ms[name])
            signal += frac * float(w)
        sidecar_timing = {
            "RepetitionTime": 5.0,
            "EchoTime": 0.388,
            "InversionTime": round(ti_ms / 1000.0, 3),
        }

    # Preserve realistic fine anatomical micro-texture from the warped real seed scan
    tex_src = warped_t2 if (modality in ("T2w", "FLAIR") and warped_t2 is not None) else warped_t1
    w_tex = cfg.preserve_seed_texture_weight
    if w_tex > 0.0 and np.any(pve.head_mask):
        smooth_seed = ndimage.gaussian_filter(tex_src, sigma=2.2)
        p10 = max(float(np.percentile(tex_src[pve.head_mask], 10.0)), 1e-4)
        detail_ratio = np.clip(tex_src / np.maximum(smooth_seed, p10), 0.78, 1.24)
        tex_mod = np.where(pve.head_mask, 1.0 + w_tex * (detail_ratio - 1.0), 1.0).astype(
            np.float32
        )
        signal = signal * tex_mod

    # Normalize foreground scale to standard ~1000 MRI units
    if np.any(pve.brain_mask):
        p95 = max(float(np.percentile(signal[pve.brain_mask], 95.0)), 1e-5)
        signal = signal * (1000.0 / p95)

    # Mild clean baseline magnitude Rician noise
    fg_med = float(np.median(signal[pve.brain_mask])) if np.any(pve.brain_mask) else 500.0
    sigma = fg_med / max(cfg.baseline_snr, 1.0)
    noisy = apply_noncentral_chi_noise(
        signal,
        sigma=sigma,
        coil_count=1,
        seed=int(rng.integers(0, 2**31 - 1)),
        device=resolve_device(device),
    )
    return noisy, sidecar_timing


def _bandpass_timecourses(
    n_series: int,
    n_trs: int,
    tr_sec: float,
    lo_hz: float,
    hi_hz: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate unit-variance resting-state bandpass-filtered time courses (n_series, n_trs)."""
    white = rng.normal(0.0, 1.0, size=(n_series, n_trs)).astype(np.float64)
    freqs = np.fft.rfftfreq(n_trs, d=tr_sec)
    spec = np.fft.rfft(white, axis=1)
    band = (freqs >= lo_hz) & (freqs <= hi_hz)
    if not np.any(band):
        band[min(1, len(band) - 1)] = True
    spec[:, ~band] = 0.0
    filtered = np.fft.irfft(spec, n=n_trs, axis=1)
    sd = np.std(filtered, axis=1, keepdims=True) + 1e-6
    return np.asarray((filtered - np.mean(filtered, axis=1, keepdims=True)) / sd, dtype=np.float32)


def synthesize_bold_4d(
    pve: TissuePVEMaps,
    src_affine: np.ndarray,
    src_zooms: tuple[float, float, float],
    params: TissueMRIParams,
    cfg: BoldSynthesisConfig,
    warp_field: DiffeomorphicField,
    rsn_template_path: Path | None,
    rng: np.random.Generator,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize a clean 4D BOLD EPI timeseries with resting-state GM RSNs, physio, and AR(1) noise."""
    epi_pve, epi_affine = _resample_pve_to_spacing(pve, src_affine, src_zooms, cfg.voxel_size_mm)
    shape = epi_pve.wm.shape
    n_t = cfg.n_trs
    te_ms = cfg.te_sec * 1000.0

    # T2*-weighted EPI baseline volume S0
    s0 = np.zeros(shape, dtype=np.float32)
    for name, frac in [
        ("wm", epi_pve.wm),
        ("gm", epi_pve.gm),
        ("csf", epi_pve.csf),
        ("scalp", epi_pve.scalp * 0.45),
    ]:
        s0 += frac * float(params.rho[name] * np.exp(-te_ms / params.t2s_ms[name]))

    if np.any(epi_pve.brain_mask):
        p95 = max(float(np.percentile(s0[epi_pve.brain_mask], 95.0)), 1e-5)
        s0 = s0 * (950.0 / p95)

    # 1. Spatially structured GM resting-state network (RSN) fluctuations (0.01 - 0.10 Hz)
    bold_mod = np.zeros((*shape, n_t), dtype=np.float32)
    if cfg.gm_bold_pct > 0.0 and np.any(epi_pve.gm > 0.1):
        spatial_maps: list[np.ndarray] = []
        if rsn_template_path is not None:
            # Load template in seed structural space, warp coherently, then resample to EPI grid
            src_shape = (int(pve.wm.shape[0]), int(pve.wm.shape[1]), int(pve.wm.shape[2]))
            tpl = resample_nifti_to_grid(rsn_template_path, src_shape, src_affine, order=0)
            zoom_epi = tuple(float(shape[i]) / float(src_shape[i]) for i in range(3))
            if tpl.ndim == 3:
                warped_lbl = warp_volume(np.rint(tpl).astype(np.int32), warp_field, order=0)
                epi_lbl = np.rint(ndimage.zoom(warped_lbl.astype(np.float32), zoom_epi, order=0)).astype(
                    np.int32
                )
                unique_ids = [int(u) for u in np.unique(epi_lbl) if u > 0][:16]
                for uid in unique_ids:
                    m = ndimage.gaussian_filter((epi_lbl == uid).astype(np.float32), sigma=0.8)
                    spatial_maps.append(m * epi_pve.gm)
            else:
                n_comp = min(16, tpl.shape[3])
                for c in range(n_comp):
                    wc = warp_volume(tpl[..., c], warp_field, order=1)
                    ec = ndimage.zoom(wc, zoom_epi, order=1).astype(np.float32)
                    max_abs = max(float(np.max(np.abs(ec))), 1e-5)
                    spatial_maps.append((ec / max_abs) * epi_pve.gm)
        else:
            # Template-free: 8 smooth resting-state spatial clusters masked to cortical GM
            for _ in range(8):
                raw_field = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
                smooth_f = ndimage.gaussian_filter(raw_field, sigma=3.0)
                sd_f = float(np.std(smooth_f[epi_pve.brain_mask])) + 1e-5
                spatial_maps.append((smooth_f / sd_f) * epi_pve.gm)

        if spatial_maps:
            n_comp = len(spatial_maps)
            tcs = _bandpass_timecourses(n_comp, n_t, cfg.tr_sec, 0.01, 0.10, rng)
            maps_stack = np.stack(spatial_maps, axis=0)  # (K, X, Y, Z)
            # Vectorized tensor contraction: (K, X, Y, Z) x (K, T) -> (X, Y, Z, T)
            rsn_4d = np.tensordot(maps_stack, tcs, axes=([0], [0])).astype(np.float32)
            rsn_4d /= np.sqrt(float(n_comp))
            bold_mod += cfg.gm_bold_pct * rsn_4d

    # 2. Physiological fluctuations (respiratory ~0.25 Hz + cardiac ~1.05 Hz) weighted to CSF & edge
    if cfg.physio_pct > 0.0:
        t_sec = np.arange(n_t, dtype=np.float32) * cfg.tr_sec
        resp_hz = float(rng.uniform(0.20, 0.30))
        card_hz = float(rng.uniform(0.95, 1.20))
        resp_wave = np.sin(2.0 * np.pi * resp_hz * t_sec + float(rng.uniform(0, 6.28)))
        card_wave = np.sin(2.0 * np.pi * card_hz * t_sec + float(rng.uniform(0, 6.28)))
        edge = epi_pve.brain_mask & (~ndimage.binary_erosion(epi_pve.brain_mask, iterations=1))
        physio_mask = np.clip(epi_pve.csf + 0.5 * edge.astype(np.float32), 0.0, 1.0)
        physio_sig = 0.6 * resp_wave + 0.4 * card_wave
        bold_mod += (cfg.physio_pct * physio_mask[..., None] * physio_sig[None, None, None, :]).astype(
            np.float32
        )

    # 3. Spatially smooth AR(1) temporal noise inside head
    if cfg.ar1_phi > 0.0:
        ar_scale = 0.008
        ar_noise = np.zeros((*shape, n_t), dtype=np.float32)
        prev = ndimage.gaussian_filter(rng.normal(0.0, 1.0, size=shape).astype(np.float32), sigma=1.0)
        ar_noise[..., 0] = prev
        innov_sd = float(np.sqrt(max(1.0 - cfg.ar1_phi**2, 0.05)))
        for t in range(1, n_t):
            innov = ndimage.gaussian_filter(
                rng.normal(0.0, innov_sd, size=shape).astype(np.float32), sigma=1.0
            )
            prev = cfg.ar1_phi * prev + innov
            ar_noise[..., t] = prev
        bold_mod += ar_scale * ar_noise * epi_pve.head_mask[..., None]

    clean_4d = np.maximum(s0[..., None] * (1.0 + bold_mod), 0.0).astype(np.float32)

    # 4. Clean baseline magnitude Rician thermal noise
    fg_med = float(np.median(s0[epi_pve.brain_mask])) if np.any(epi_pve.brain_mask) else 500.0
    sigma = fg_med / max(cfg.baseline_snr, 1.0)
    bold_4d = apply_noncentral_chi_noise(
        clean_4d,
        sigma=sigma,
        coil_count=1,
        seed=int(rng.integers(0, 2**31 - 1)),
        device=resolve_device(device),
    )
    return bold_4d, epi_affine


def _build_anatomical_wm_fiber_field(
    pve: TissuePVEMaps,
    rng: np.random.Generator,
) -> np.ndarray:
    """Construct a smooth template-free 3D white-matter principal fiber field (X, Y, Z, 3) in RAS+."""
    shape = pve.wm.shape
    coords = np.argwhere(pve.brain_mask)
    if coords.size == 0:
        out = np.zeros((*shape, 3), dtype=np.float32)
        out[..., 0] = 1.0
        return out

    bmin = coords.min(axis=0).astype(np.float32)
    bmax = np.maximum(coords.max(axis=0).astype(np.float32), bmin + 1.0)
    gx, gy, gz = np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        np.arange(shape[2], dtype=np.float32),
        indexing="ij",
    )
    nx = 2.0 * (gx - bmin[0]) / (bmax[0] - bmin[0]) - 1.0  # [-1, 1] L->R
    ny = 2.0 * (gy - bmin[1]) / (bmax[1] - bmin[1]) - 1.0  # [-1, 1] P->A
    nz = 2.0 * (gz - bmin[2]) / (bmax[2] - bmin[2]) - 1.0  # [-1, 1] I->S

    # Smooth anatomical zone blending:
    # - Corpus callosum (central midline, mid-superior): strong R-L (x)
    cc_w = np.exp(-((nx / 0.35) ** 2 + (ny / 0.55) ** 2 + ((nz - 0.15) / 0.30) ** 2))
    # - Brainstem & corticospinal tract (inferior-central): strong I-S (z)
    cst_w = np.exp(-((nx / 0.32) ** 2 + ((ny + 0.12) / 0.35) ** 2 + ((nz + 0.55) / 0.45) ** 2))
    # - Bilateral association tracts (SLF/IFOF): strong A-P (y)
    assoc_w = np.exp(-(((np.abs(nx) - 0.48) / 0.32) ** 2 + (ny / 0.70) ** 2 + (nz / 0.55) ** 2))

    # Mild smooth random curvature perturbation for inter-subject microstructural diversity
    px = ndimage.gaussian_filter(rng.normal(0.0, 0.35, size=shape).astype(np.float32), sigma=4.0)
    py = ndimage.gaussian_filter(rng.normal(0.0, 0.35, size=shape).astype(np.float32), sigma=4.0)
    pz = ndimage.gaussian_filter(rng.normal(0.0, 0.35, size=shape).astype(np.float32), sigma=4.0)

    vx = cc_w + 0.25 * (1.0 - assoc_w) + px
    vy = assoc_w + 0.35 + py
    vz = cst_w + 0.20 * (nz < 0.0) + pz

    vec = np.stack([vx, vy, vz], axis=-1).astype(np.float32)
    norms = np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-6
    return np.asarray(vec / norms, dtype=np.float32)


def _extract_primary_eigenvector_from_template(
    dti_path: Path,
    src_shape: tuple[int, int, int],
    src_affine: np.ndarray,
) -> np.ndarray:
    """Load a 4D (X,Y,Z,3) primary eigenvector or (X,Y,Z,6) symmetric DTI tensor template."""
    raw = resample_nifti_to_grid(dti_path, src_shape, src_affine, order=1)
    if raw.ndim != 4 or raw.shape[3] not in (3, 6):
        raise ValueError(
            f"dti_eigenvector_nifti must be 4D with 3 (vector) or 6 (tensor) channels, got {raw.shape}"
        )
    if raw.shape[3] == 3:
        norms = np.linalg.norm(raw, axis=-1, keepdims=True) + 1e-6
        return np.asarray(raw / norms, dtype=np.float32)

    # Symmetric 6-component tensor [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz] -> principal eigenvector
    n_vox = int(np.prod(src_shape))
    flat = raw.reshape(n_vox, 6)
    mats = np.zeros((n_vox, 3, 3), dtype=np.float32)
    mats[:, 0, 0] = flat[:, 0]
    mats[:, 0, 1] = mats[:, 1, 0] = flat[:, 1]
    mats[:, 0, 2] = mats[:, 2, 0] = flat[:, 2]
    mats[:, 1, 1] = flat[:, 3]
    mats[:, 1, 2] = mats[:, 2, 1] = flat[:, 4]
    mats[:, 2, 2] = flat[:, 5]
    _evals, evecs = np.linalg.eigh(mats)
    # eigh sorts ascending; index 2 is principal eigenvector
    v1 = evecs[:, :, 2].reshape(*src_shape, 3).astype(np.float32)
    norms = np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-6
    return np.asarray(v1 / norms, dtype=np.float32)


def synthesize_dwi_4d(
    pve: TissuePVEMaps,
    src_affine: np.ndarray,
    src_zooms: tuple[float, float, float],
    params: TissueMRIParams,
    cfg: DwiSynthesisConfig,
    warp_field: DiffeomorphicField,
    dti_template_path: Path | None,
    rng: np.random.Generator,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize clean multi-shell 4D DWI + affine + bvals + bvecs via vectorized multi-tensor model."""
    src_shape = (int(pve.wm.shape[0]), int(pve.wm.shape[1]), int(pve.wm.shape[2]))

    # Build or load fiber orientation field in seed structural space, warp & PPD-reorient
    if dti_template_path is not None:
        v1_seed = _extract_primary_eigenvector_from_template(
            dti_template_path, src_shape, src_affine
        )
    else:
        v1_seed = _build_anatomical_wm_fiber_field(pve, rng)

    v1_warped = warp_vector_field_ppd(v1_seed, warp_field)

    # Resample warped PVE and primary fiber field to DWI resolution
    dwi_pve, dwi_affine = _resample_pve_to_spacing(pve, src_affine, src_zooms, cfg.voxel_size_mm)
    dwi_shape = dwi_pve.wm.shape
    zoom_dwi = tuple(float(dwi_shape[i]) / float(src_shape[i]) for i in range(3))

    v1 = np.empty((*dwi_shape, 3), dtype=np.float32)
    for k in range(3):
        v1[..., k] = ndimage.zoom(v1_warped[..., k], zoom_dwi, order=1)
    v1_norm = np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-6
    v1 = (v1 / v1_norm).astype(np.float32)

    # Orthogonal crossing-fiber secondary direction v2
    ref_ax = np.zeros_like(v1)
    ref_ax[..., 2] = 1.0
    nearly_z = np.abs(v1[..., 2]) > 0.9
    ref_ax[nearly_z, 2] = 0.0
    ref_ax[nearly_z, 0] = 1.0
    v2 = np.cross(v1, ref_ax)
    v2 = (v2 / (np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-6)).astype(np.float32)

    # Spatially smooth crossing-fiber mask inside white matter
    cross_field = ndimage.gaussian_filter(rng.normal(0.0, 1.0, size=dwi_shape).astype(np.float32), sigma=2.5)
    if cfg.crossing_fiber_fraction > 0.0 and np.any(dwi_pve.wm > 0.2):
        thresh = float(np.percentile(cross_field[dwi_pve.wm > 0.2], 100.0 * (1.0 - cfg.crossing_fiber_fraction)))
        w_cross = np.where(cross_field >= thresh, 0.42, 0.0).astype(np.float32)
    else:
        w_cross = np.zeros(dwi_shape, dtype=np.float32)

    # Build multi-shell gradient table (bvals, bvecs)
    bval_parts: list[np.ndarray] = [np.zeros(cfg.n_b0, dtype=np.float32)]
    bvec_parts: list[np.ndarray] = [np.zeros((3, cfg.n_b0), dtype=np.float32)]
    for b_val, n_dirs in cfg.shells:
        bval_parts.append(np.full(int(n_dirs), float(b_val), dtype=np.float32))
        bvec_parts.append(_fibonacci_hemisphere(int(n_dirs), rng))
    bvals = np.concatenate(bval_parts, axis=0)
    bvecs = np.hstack(bvec_parts).astype(np.float32)  # (3, N_vols)

    # T2-weighted b=0 baseline S0
    te_ms = cfg.te_sec * 1000.0
    s0 = np.zeros(dwi_shape, dtype=np.float32)
    for name, frac in [
        ("wm", dwi_pve.wm),
        ("gm", dwi_pve.gm),
        ("csf", dwi_pve.csf),
        ("scalp", dwi_pve.scalp * 0.35),
    ]:
        s0 += frac * float(params.rho[name] * np.exp(-te_ms / params.t2_ms[name]))
    if np.any(dwi_pve.brain_mask):
        p95 = max(float(np.percentile(s0[dwi_pve.brain_mask], 95.0)), 1e-5)
        s0 = s0 * (900.0 / p95)

    # Vectorized multi-compartment Stejskal-Tanner signal evaluation on PyTorch device (CPU or RTX 5080)
    dev = resolve_device(device)
    b_t = torch.from_numpy(bvals).to(dev)  # (T,)
    g_t = torch.from_numpy(bvecs).to(dev)  # (3, T)
    v1_t = torch.from_numpy(v1).to(dev)  # (X, Y, Z, 3)
    v2_t = torch.from_numpy(v2).to(dev)  # (X, Y, Z, 3)
    w_cross_t = torch.from_numpy(w_cross).to(dev)[..., None]  # (X, Y, Z, 1)

    f_wm = torch.from_numpy(dwi_pve.wm).to(dev)[..., None]
    f_gm = torch.from_numpy(dwi_pve.gm).to(dev)[..., None]
    f_csf = torch.from_numpy(dwi_pve.csf).to(dev)[..., None]
    f_sc = torch.from_numpy(dwi_pve.scalp).to(dev)[..., None]
    s0_t = torch.from_numpy(s0).to(dev)[..., None]

    # Realistic 3T diffusivities in mm^2/s (b in s/mm^2)
    d_par_wm, d_perp_wm = 1.7e-3, 0.25e-3
    d_gm, d_csf, d_sc = 0.80e-3, 3.0e-3, 1.1e-3

    # Projection (g . v)^2 for fiber populations 1 and 2 -> shape (X, Y, Z, T)
    cos2_1 = torch.einsum("xyzi,it->xyzt", v1_t, g_t) ** 2
    cos2_2 = torch.einsum("xyzi,it->xyzt", v2_t, g_t) ** 2

    adc_1 = d_perp_wm + (d_par_wm - d_perp_wm) * cos2_1
    adc_2 = d_perp_wm + (d_par_wm - d_perp_wm) * cos2_2
    b_bc = b_t.view(1, 1, 1, -1)

    att_wm = (1.0 - w_cross_t) * torch.exp(-b_bc * adc_1) + w_cross_t * torch.exp(-b_bc * adc_2)
    att_gm = torch.exp(-b_bc * d_gm)
    att_csf = torch.exp(-b_bc * d_csf)
    att_sc = torch.exp(-b_bc * d_sc)

    denom_f = torch.clamp(f_wm + f_gm + f_csf + f_sc, min=1e-5)
    att_total = (f_wm * att_wm + f_gm * att_gm + f_csf * att_csf + f_sc * att_sc) / denom_f
    clean_dwi = (s0_t * att_total).detach().cpu().numpy().astype(np.float32)

    # Clean baseline magnitude Rician noise
    fg_med = float(np.median(s0[dwi_pve.brain_mask])) if np.any(dwi_pve.brain_mask) else 450.0
    sigma = fg_med / max(cfg.baseline_snr, 1.0)
    dwi_4d = apply_noncentral_chi_noise(
        clean_dwi,
        sigma=sigma,
        coil_count=1,
        seed=int(rng.integers(0, 2**31 - 1)),
        device=dev,
    )
    return dwi_4d, dwi_affine, bvals, bvecs
