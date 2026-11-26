"""Tissue-aware continuous T1w -> T2w / FLAIR / EPI contrast conversion.

Converts a clean T1w volume into T2w, CSF-nulled FLAIR, T2*-weighted BOLD S0, or
SE-EPI DWI b=0 contrast while:
1. Correcting smooth B1 bias via SimpleITK N4ITK so tissue intensities are spatially stationary.
2. Separating intracranial brain volume (ICV) from cortical skull bone/air and outer scalp/neck
   using an external brain mask (Option 2), DIPY's PyTorch EVAC+ extractor (`brain_extraction="auto"`),
   or an offline morphological ICV fallback.
3. Mapping intracranial intensities continuously through a shape-preserving PchipInterpolator
   anchored to GMM tissue modes (CSF, GM, WM) and Bloch target contrasts:
   - `T2w`, `bold_s0`, `dwi_b0`: strictly monotone decreasing (CSF > GM > WM).
   - `FLAIR`: unimodal CSF-nulled curve (GM > WM >> CSF ~ 0).
4. Retaining air and skull bone dark, retaining scalp fat on clinical structural scans (`T2w`,
   `FLAIR`), and cleanly skull-stripping / fat-suppressing EPI baselines (`bold_s0`, `dwi_b0`).
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

from sim_data.config import BrainExtractionMode, ContrastConversionConfig
from sim_data.io.orientation import compute_anatomical_priors
from sim_data.synthesis.brain_mask import extract_brain_mask
from sim_data.synthesis.tissue import TissueMRIParams, sample_tissue_mri_params

ContrastTarget = Literal["T2w", "FLAIR", "bold_s0", "dwi_b0"]


@dataclass
class ContrastConversionResult:
    """Output of T1w -> T2w/FLAIR/EPI contrast conversion with compartment masks and diagnostics."""

    data: np.ndarray  # float32 (X, Y, Z)
    icv_mask: np.ndarray  # bool (X, Y, Z) intracranial brain volume
    bone_air_mask: np.ndarray  # bool (X, Y, Z) air + cortical bone kept dark
    scalp_mask: np.ndarray  # bool (X, Y, Z) extracranial scalp/fat/neck
    t1_n4: np.ndarray  # float32 (X, Y, Z) N4-corrected T1w
    tissue_anchors_t1: dict[str, float]  # normalized T1w GMM means for csf/gm/wm
    tissue_targets_out: dict[str, float]  # Bloch target signals for csf/gm/wm
    used_external_mask: bool
    backend: str
    mask_method: str = "morphological"


class ContrastConverter(Protocol):
    """Protocol for T1w -> T2w/FLAIR/EPI contrast converters (pchip_bloch now; neural in Option 3)."""

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
        """Convert a 3D RAS+ T1w volume into the requested target contrast."""


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
    method: BrainExtractionMode = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Separate intracranial volume (ICV), dark skull/air, and outer scalp/fat."""
    res = extract_brain_mask(
        t1_corr=t1_corr,
        affine=affine,
        method=method,
        brain_mask_path=brain_mask_path,
        brain_mask_array=brain_mask_array,
    )
    return res.icv_mask, res.bone_air_mask, res.scalp_mask, res.used_external_mask


def _bloch_target_intensities(
    target: ContrastTarget,
    params: TissueMRIParams,
    te_ms: float | None = None,
) -> dict[str, float]:
    """Compute physical Bloch target signals for CSF, GM, WM for T2w / FLAIR / BOLD S0 / DWI b=0."""
    out: dict[str, float] = {}
    if target == "FLAIR":
        tr = 9000.0
        te = 110.0 if te_ms is None else float(te_ms)
        t1_csf = params.t1_ms["csf"]
        ti = float(t1_csf * np.log(2.0 / (1.0 + np.exp(-tr / t1_csf))))
        for name in ("csf", "gm", "wm"):
            ir = abs(1.0 - 2.0 * np.exp(-ti / params.t1_ms[name]) + np.exp(-tr / params.t1_ms[name]))
            out[name] = float(params.rho[name] * ir * np.exp(-te / params.t2_ms[name]))
        # Enforce clinical FLAIR contrast: bright GM > mid-gray WM >> dark nulled CSF
        s_gm = max(out["gm"], 0.25)
        s_wm = min(max(out["wm"], s_gm * 0.58), s_gm * 0.72)
        s_csf = min(max(out["csf"], 0.01), s_gm * 0.08)
        return {"csf": s_csf, "gm": s_gm, "wm": s_wm}

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
    """Convert a 3D RAS+ T1w image into continuous T2w / FLAIR / BOLD S0 / DWI b=0 contrast."""
    cfg = cfg or ContrastConversionConfig()
    rng = np.random.default_rng(seed)
    params = params or sample_tissue_mri_params(rng, jitter=0.0)

    t1 = np.nan_to_num(np.asarray(t1_data, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if t1.ndim == 4:
        t1 = t1[..., 0]

    t1_corr = n4_bias_correct_volume(t1) if cfg.n4_bias_correct else t1.copy()
    mask_res = extract_brain_mask(
        t1_corr=t1_corr,
        affine=affine,
        method=cfg.brain_extraction,
        brain_mask_path=brain_mask_path,
        brain_mask_array=brain_mask_array,
    )
    icv_mask = mask_res.icv_mask
    bone_air_mask = mask_res.bone_air_mask
    scalp_mask = mask_res.scalp_mask
    used_ext = mask_res.used_external_mask

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
            mask_method=mask_res.method,
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

    if target == "FLAIR":
        # Unimodal CSF-nulled curve: air/CSF dark -> bright cortical GM -> mid-gray WM
        x_knots = np.array(
            [0.0, 0.50 * mu_csf, mu_csf, 0.5 * (mu_csf + mu_gm), mu_gm, mu_wm, x_hi],
            dtype=np.float64,
        )
        y_knots = np.array(
            [0.0, s_csf * 0.4, s_csf, 0.45 * s_gm, s_gm, s_wm, s_wm * 0.72],
            dtype=np.float64,
        )
    else:
        # Monotone decreasing T2 / T2* / b=0 curve: bright CSF -> mid GM -> dark WM
        x_knots = np.array([0.0, 0.45 * mu_csf, mu_csf, mu_gm, mu_wm, x_hi], dtype=np.float64)
        y_knots = np.array(
            [s_csf * 1.05, s_csf * 1.02, s_csf, s_gm, s_wm, s_wm * 0.78],
            dtype=np.float64,
        )
    pchip = PchipInterpolator(x_knots, y_knots, extrapolate=True)

    # Evaluate continuous target contrast inside ICV
    brain_out = np.zeros_like(t1_norm, dtype=np.float32)
    x_eval = np.clip(t1_norm[icv_mask].astype(np.float64), 0.0, x_hi)
    brain_out[icv_mask] = np.maximum(pchip(x_eval), 0.0).astype(np.float32)

    if not cfg.preserve_micro_texture:
        brain_out = ndimage.gaussian_filter(brain_out, sigma=0.8) * icv_mask

    is_epi = target in ("bold_s0", "dwi_b0")
    if is_epi and cfg.epi_skull_strip:
        # Real EPI acquisitions are fat-suppressed & brain-focused: strip skull/scalp/neck cleanly
        # with a subtle sub-voxel boundary feather to avoid hard stair-step aliasing on downsampling
        soft_icv = ndimage.gaussian_filter(icv_mask.astype(np.float32), sigma=0.55)
        soft_icv = np.where(icv_mask, np.maximum(soft_icv, 0.85), soft_icv * 0.25)
        out = (brain_out * np.clip(soft_icv, 0.0, 1.0)).astype(np.float32)
        out[~icv_mask] = 0.0
    else:
        if target == "T2w":
            fat_ret = cfg.fat_retention_t2w
        elif target == "FLAIR":
            fat_ret = cfg.fat_retention_flair
        else:
            fat_ret = cfg.fat_retention_epi

        scalp_out = np.zeros_like(t1_norm, dtype=np.float32)
        if np.any(scalp_mask) and fat_ret > 0.0:
            scalp_rel = np.clip(t1_norm[scalp_mask] / max(mu_gm, 0.1), 0.20, 1.30)
            scalp_out[scalp_mask] = (fat_ret * s_gm * scalp_rel).astype(np.float32)

        head_mask = icv_mask | scalp_mask
        w_icv = ndimage.gaussian_filter(icv_mask.astype(np.float32), sigma=0.65)
        w_icv = np.where(head_mask, np.clip(w_icv, 0.0, 1.0), 0.0)
        out = np.where(
            icv_mask,
            brain_out,
            np.where(scalp_mask, (1.0 - w_icv) * scalp_out + w_icv * s_wm * 0.25, 0.0),
        ).astype(np.float32)
        out[bone_air_mask] = 0.0

    # Normalize intracranial foreground to standard modality scale
    scale_map: dict[ContrastTarget, float] = {
        "T2w": 1000.0,
        "FLAIR": 1000.0,
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
        mask_method=mask_res.method,
    )


class PchipBlochContrastConverter:
    """Default tissue-aware continuous PCHIP/Bloch T1w -> T2w/FLAIR/EPI contrast converter."""

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
