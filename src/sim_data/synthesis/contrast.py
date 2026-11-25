"""Tissue-aware continuous T1w -> T2w / EPI contrast conversion (Options 1 + 2, Option 3 protocol).

Converts a clean T1w volume into a T2w, T2*-weighted BOLD S0, or SE-EPI DWI b=0 baseline while:
1. Correcting smooth B1 bias via SimpleITK N4ITK so tissue intensities are spatially stationary.
2. Separating intracranial volume (ICV) from cortical skull bone/air and outer scalp/fat
   (using an optional external brain mask when provided, or morphological ICV fallback).
3. Mapping intracranial intensities continuously through a strictly monotone decreasing
   PchipInterpolator anchored to GMM tissue modes (CSF, GM, WM) and Bloch target contrasts,
   preserving continuous sulcal/gyral micro-texture with proper inverted polarity.
4. Retaining air and skull bone dark while controlling scalp fat independently (retained for
   clinical FSE T2w; fat-suppressed for EPI BOLD and DWI).

Extension note (Option 3): Implement `ContrastConverter` to plug in a pretrained neural
T1w -> T2w backend (e.g., SynthSR / MONAI) once formal clinical validation begins.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.interpolate import PchipInterpolator
from sklearn.mixture import GaussianMixture

from sim_data.config import ContrastConversionConfig
from sim_data.io.orientation import compute_anatomical_priors
from sim_data.synthesis.tissue import (
    TissueMRIParams,
    resample_nifti_to_grid,
    sample_tissue_mri_params,
)

ContrastTarget = Literal["T2w", "bold_s0", "dwi_b0"]


@dataclass
class ContrastConversionResult:
    """Output of T1w -> T2w-like contrast conversion with compartment masks and diagnostics."""

    data: np.ndarray  # float32 (X, Y, Z)
    icv_mask: np.ndarray  # bool (X, Y, Z) intracranial volume
    bone_air_mask: np.ndarray  # bool (X, Y, Z) air + cortical bone kept dark
    scalp_mask: np.ndarray  # bool (X, Y, Z) extracranial scalp/fat
    t1_n4: np.ndarray  # float32 (X, Y, Z) N4-corrected T1w
    tissue_anchors_t1: dict[str, float]  # normalized T1w GMM means for csf/gm/wm
    tissue_targets_out: dict[str, float]  # Bloch target signals for csf/gm/wm
    used_external_mask: bool
    backend: str


class ContrastConverter(Protocol):
    """Protocol for T1w -> T2w-like contrast converters (pchip_bloch now; neural in Option 3)."""

    def convert(
        self,
        t1_data: np.ndarray,
        affine: np.ndarray,
        target: ContrastTarget = "T2w",
        params: TissueMRIParams | None = None,
        cfg: ContrastConversionConfig | None = None,
        brain_mask_path: str | Path | None = None,
        brain_mask_array: np.ndarray | None = None,
        te_ms: float | None = None,
        seed: int = 42,
    ) -> ContrastConversionResult:
        """Convert a 3D RAS+ T1w volume into the requested T2-like contrast."""


def n4_bias_correct_volume(
    vol: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply fast shrunk-grid SimpleITK N4ITK bias correction while preserving foreground scale."""
    v = np.nan_to_num(np.asarray(vol, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if mask is None:
        mask = compute_anatomical_priors(v).brain_mask
    if int(mask.sum()) < 64:
        return v.copy()

    orig_med = float(np.median(v[mask]))
    if orig_med <= 1e-6:
        return v.copy()

    try:
        v_pos = np.maximum(v, 1e-3).astype(np.float32)
        sitk_img = sitk.GetImageFromArray(v_pos)
        sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))

        shrink_factor = max(1, min(3, min(v.shape) // 24))
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
        bias_arr = np.exp(sitk.GetArrayFromImage(log_bias).astype(np.float32))
        bias_arr = np.clip(bias_arr, 0.25, 4.0)
        corrected = np.where(v > 0.0, v / bias_arr, 0.0).astype(np.float32)
        corr_med = float(np.median(corrected[mask]))
        if np.isfinite(corr_med) and corr_med > 1e-6:
            corrected = corrected * (orig_med / corr_med)
            return np.maximum(corrected, 0.0).astype(np.float32)
    except Exception:
        pass
    return v.copy()


def resolve_icv_and_compartments(
    t1_corr: np.ndarray,
    affine: np.ndarray,
    brain_mask_path: str | Path | None = None,
    brain_mask_array: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Separate intracranial volume (ICV), dark skull/air, and outer scalp/fat.

    Uses an explicit brain/ICV mask when supplied (Option 2), otherwise falls back to
    morphological + intensity-guided cranial compartment separation.
    """
    shape = (int(t1_corr.shape[0]), int(t1_corr.shape[1]), int(t1_corr.shape[2]))
    priors = compute_anatomical_priors(t1_corr)
    head_mask = priors.brain_mask

    if brain_mask_array is not None and brain_mask_array.shape == shape:
        icv_mask = np.asarray(brain_mask_array > 0.5, dtype=bool)
        used_external = True
    elif brain_mask_path is not None:
        raw_mask = resample_nifti_to_grid(brain_mask_path, shape, affine, order=0)
        if raw_mask.ndim == 4:
            raw_mask = raw_mask[..., 0]
        icv_mask = np.asarray(raw_mask > 0.5, dtype=bool)
        used_external = True
    else:
        used_external = False
        eroded = ndimage.binary_erosion(head_mask, iterations=2)
        outer_rim = head_mask & (~eroded)
        if int(eroded.sum()) < 64:
            icv_mask = head_mask.copy()
            scalp_mask = np.zeros(shape, dtype=bool)
            bone_air_mask = ~icv_mask
            return icv_mask, bone_air_mask, scalp_mask, False

        inner_vals = t1_corr[eroded]
        p15 = float(np.percentile(inner_vals, 15.0))
        p50 = float(np.percentile(inner_vals, 50.0))
        p97 = float(np.percentile(inner_vals, 97.0))

        # Outer cranial shell: moderate-to-low T1 scalp/marrow OR hyperintense subcutaneous fat
        scalp_cand = outer_rim & ((t1_corr >= p15 * 0.45) & (t1_corr < p50 * 0.92) | (t1_corr > p97))
        dark_bone = outer_rim & (t1_corr < p15 * 0.45)
        icv_mask = head_mask & (~scalp_cand) & (~dark_bone)
        if int(icv_mask.sum()) < 64:
            icv_mask = eroded
        scalp_mask = scalp_cand
        bone_air_mask = (~icv_mask) & (~scalp_mask)
        return icv_mask, bone_air_mask, scalp_mask, False

    if int(icv_mask.sum()) < 32:
        icv_mask = head_mask.copy()

    outer_shell = head_mask & (~icv_mask)
    if np.any(outer_shell):
        icv_p20 = float(np.percentile(t1_corr[icv_mask], 20.0))
        scalp_mask = outer_shell & (t1_corr >= icv_p20 * 0.40)
    else:
        scalp_mask = np.zeros(shape, dtype=bool)
    bone_air_mask = (~icv_mask) & (~scalp_mask)
    return icv_mask, bone_air_mask, scalp_mask, used_external


def _bloch_target_intensities(
    target: ContrastTarget,
    params: TissueMRIParams,
    te_ms: float | None = None,
) -> dict[str, float]:
    """Compute physical Bloch target signals for CSF, GM, WM for T2w / BOLD S0 / DWI b=0."""
    out: dict[str, float] = {}
    if target == "T2w":
        tr = 4500.0
        te = 100.0 if te_ms is None else float(te_ms)
        for name in ("csf", "gm", "wm"):
            out[name] = float(
                params.rho[name]
                * (1.0 - np.exp(-tr / params.t1_ms[name]))
                * np.exp(-te / params.t2_ms[name])
            )
    elif target == "bold_s0":
        te = 30.0 if te_ms is None else float(te_ms)
        for name in ("csf", "gm", "wm"):
            out[name] = float(params.rho[name] * np.exp(-te / params.t2s_ms[name]))
    else:  # dwi_b0 (spin-echo EPI T2-weighted)
        te = 89.0 if te_ms is None else float(te_ms)
        for name in ("csf", "gm", "wm"):
            out[name] = float(params.rho[name] * np.exp(-te / params.t2_ms[name]))

    # Enforce physically consistent T2 polarity CSF > GM > WM even under parameter jitter
    s_wm = max(out["wm"], 1e-4)
    s_gm = max(out["gm"], s_wm * 1.15)
    s_csf = max(out["csf"], s_gm * 1.25)
    return {"csf": s_csf, "gm": s_gm, "wm": s_wm}


def invert_t1w_to_t2w(
    t1_data: np.ndarray,
    affine: np.ndarray,
    target: ContrastTarget = "T2w",
    params: TissueMRIParams | None = None,
    cfg: ContrastConversionConfig | None = None,
    brain_mask_path: str | Path | None = None,
    brain_mask_array: np.ndarray | None = None,
    te_ms: float | None = None,
    seed: int = 42,
) -> ContrastConversionResult:
    """Convert a 3D RAS+ T1w image into continuous T2w / BOLD S0 / DWI b=0 contrast.

    Uses N4ITK bias correction + ICV/bone/scalp compartment separation + strictly monotone
    decreasing PCHIP spline anchored to GMM tissue modes (CSF, GM, WM) and Bloch target
    intensities. Preserves fine sulcal/gyral micro-texture with inverted polarity while
    keeping background air and cortical bone dark.
    """
    cfg = cfg or ContrastConversionConfig()
    rng = np.random.default_rng(seed)
    params = params or sample_tissue_mri_params(rng, jitter=0.0)

    t1 = np.nan_to_num(np.asarray(t1_data, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if t1.ndim == 4:
        t1 = t1[..., 0]

    t1_corr = n4_bias_correct_volume(t1) if cfg.n4_bias_correct else t1.copy()
    icv_mask, bone_air_mask, scalp_mask, used_ext = resolve_icv_and_compartments(
        t1_corr,
        affine=affine,
        brain_mask_path=brain_mask_path,
        brain_mask_array=brain_mask_array,
    )

    icv_vals = t1_corr[icv_mask]
    if icv_vals.size < 32 or float(np.max(icv_vals)) <= 1e-6:
        return ContrastConversionResult(
            data=np.zeros_like(t1, dtype=np.float32),
            icv_mask=icv_mask,
            bone_air_mask=bone_air_mask,
            scalp_mask=scalp_mask,
            t1_n4=t1_corr,
            tissue_anchors_t1={"csf": 0.2, "gm": 0.6, "wm": 0.9},
            tissue_targets_out={"csf": 1.0, "gm": 0.7, "wm": 0.45},
            used_external_mask=used_ext,
            backend=cfg.backend,
        )

    p98 = max(float(np.percentile(icv_vals, 98.0)), 1e-5)
    t1_norm = np.clip(t1_corr / p98, 0.0, 1.5).astype(np.float32)

    # Fit 3-class GMM on intracranial T1 intensities to locate CSF < GM < WM anchors
    fit_vals = t1_norm[icv_mask]
    if fit_vals.size > 25_000:
        sub_idx = rng.choice(fit_vals.size, size=25_000, replace=False)
        fit_sample = fit_vals[sub_idx][:, None]
    else:
        fit_sample = fit_vals[:, None]

    try:
        gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=seed, max_iter=100)
        gmm.fit(fit_sample)
        means = np.sort(np.asarray(gmm.means_, dtype=np.float64).reshape(-1))
        mu_csf, mu_gm, mu_wm = float(means[0]), float(means[1]), float(means[2])
    except Exception:
        mu_csf = float(np.percentile(fit_vals, 15.0))
        mu_gm = float(np.percentile(fit_vals, 52.0))
        mu_wm = float(np.percentile(fit_vals, 85.0))

    # Guarantee strictly increasing anchor knots for shape-preserving PCHIP spline
    mu_csf = max(0.05, mu_csf)
    mu_gm = max(mu_csf + 0.08, mu_gm)
    mu_wm = max(mu_gm + 0.08, mu_wm)
    x_hi = max(mu_wm + 0.20, float(np.percentile(fit_vals, 99.5)) + 0.05, 1.25)

    targets = _bloch_target_intensities(target, params, te_ms=te_ms)
    s_csf, s_gm, s_wm = targets["csf"], targets["gm"], targets["wm"]

    x_knots = np.array([0.0, 0.45 * mu_csf, mu_csf, mu_gm, mu_wm, x_hi], dtype=np.float64)
    y_knots = np.array(
        [s_csf * 1.05, s_csf * 1.02, s_csf, s_gm, s_wm, s_wm * 0.78],
        dtype=np.float64,
    )
    pchip = PchipInterpolator(x_knots, y_knots, extrapolate=True)

    # Evaluate continuous inverted contrast inside ICV
    brain_out = np.zeros_like(t1_norm, dtype=np.float32)
    x_eval = np.clip(t1_norm[icv_mask].astype(np.float64), 0.0, x_hi)
    brain_out[icv_mask] = np.maximum(pchip(x_eval), 0.0).astype(np.float32)

    if not cfg.preserve_micro_texture:
        # Optional smoother variant without fine detail
        brain_out = ndimage.gaussian_filter(brain_out, sigma=0.8) * icv_mask

    # Extracranial scalp/fat compartment: retained on FSE T2w, fat-suppressed on EPI BOLD & DWI
    fat_ret = cfg.fat_retention_t2w if target == "T2w" else cfg.fat_retention_epi
    scalp_out = np.zeros_like(t1_norm, dtype=np.float32)
    if np.any(scalp_mask):
        scalp_rel = np.clip(t1_norm[scalp_mask] / max(mu_gm, 0.1), 0.25, 1.35)
        scalp_out[scalp_mask] = (fat_ret * s_gm * scalp_rel).astype(np.float32)

    # Softly feather ICV vs. extracranial boundary to prevent artificial sharp 1-voxel rings
    head_mask = icv_mask | scalp_mask
    w_icv = ndimage.gaussian_filter(icv_mask.astype(np.float32), sigma=0.65)
    w_icv = np.where(head_mask, np.clip(w_icv, 0.0, 1.0), 0.0)
    # Keep bone/air strictly dark (only brain + scalp carry signal)
    out = np.where(
        icv_mask,
        brain_out,
        np.where(scalp_mask, (1.0 - w_icv) * scalp_out + w_icv * s_wm * 0.35, 0.0),
    ).astype(np.float32)
    out[bone_air_mask] = 0.0

    # Normalize intracranial foreground to standard modality scale (~1000 / 950 / 900)
    scale_map: dict[ContrastTarget, float] = {
        "T2w": 1000.0,
        "bold_s0": 950.0,
        "dwi_b0": 900.0,
    }
    target_p95 = scale_map[target]
    out_p95 = max(float(np.percentile(out[icv_mask], 95.0)), 1e-5)
    out = np.maximum(out * (target_p95 / out_p95), 0.0).astype(np.float32)

    return ContrastConversionResult(
        data=out,
        icv_mask=icv_mask,
        bone_air_mask=bone_air_mask,
        scalp_mask=scalp_mask,
        t1_n4=t1_corr,
        tissue_anchors_t1={"csf": round(mu_csf, 4), "gm": round(mu_gm, 4), "wm": round(mu_wm, 4)},
        tissue_targets_out={"csf": round(s_csf, 4), "gm": round(s_gm, 4), "wm": round(s_wm, 4)},
        used_external_mask=used_ext,
        backend=cfg.backend,
    )


class PchipBlochContrastConverter:
    """Default tissue-aware continuous PCHIP/Bloch T1w -> T2w-like contrast converter."""

    def convert(
        self,
        t1_data: np.ndarray,
        affine: np.ndarray,
        target: ContrastTarget = "T2w",
        params: TissueMRIParams | None = None,
        cfg: ContrastConversionConfig | None = None,
        brain_mask_path: str | Path | None = None,
        brain_mask_array: np.ndarray | None = None,
        te_ms: float | None = None,
        seed: int = 42,
    ) -> ContrastConversionResult:
        return invert_t1w_to_t2w(
            t1_data=t1_data,
            affine=affine,
            target=target,
            params=params,
            cfg=cfg,
            brain_mask_path=brain_mask_path,
            brain_mask_array=brain_mask_array,
            te_ms=te_ms,
            seed=seed,
        )


def get_contrast_converter(cfg: ContrastConversionConfig | None = None) -> ContrastConverter:
    """Resolve the configured contrast converter backend (extensible to Option 3 neural models)."""
    effective = cfg or ContrastConversionConfig()
    if effective.backend == "pchip_bloch":
        return PchipBlochContrastConverter()
    raise ValueError(f"Unsupported contrast conversion backend: {effective.backend}")
