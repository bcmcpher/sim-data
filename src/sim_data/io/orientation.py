"""RAS+ canonicalization, encoding axis & bvec remapping, noise estimation, and anatomical priors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import nibabel as nib
import numpy as np
from dipy.denoise.noise_estimate import estimate_sigma, piesno
from dipy.segment.mask import median_otsu
from scipy import ndimage
from skimage.filters import threshold_otsu


@dataclass
class AnatomicalPriors:
    """Clean anatomical masks and tissue intensity estimates computed pre-corruption."""

    brain_mask: np.ndarray  # 3D bool mask of head/brain foreground
    bg_mask: np.ndarray  # 3D bool mask of background air outside dilated foreground
    wm_mask: np.ndarray  # 3D bool pseudo-white-matter mask (higher tissue cluster)
    gm_mask: np.ndarray  # 3D bool pseudo-gray-matter mask (lower tissue cluster)
    bbox_min: tuple[int, int, int]  # (x_min, y_min, z_min) in RAS+ voxels
    bbox_max: tuple[int, int, int]  # (x_max, y_max, z_max) inclusive in RAS+ voxels
    fg_median: float
    fg_p95: float
    bg_std: float


def canonicalize_to_ras_with_transform(
    img: nib.Nifti1Image,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float], np.ndarray]:
    """Reorient a NIfTI image to closest RAS+ orientation and return its orientation transform.

    Returns `(data, affine, zooms, ornt_transform)` where `ornt_transform` is the `(3, 2)`
    array from `nib.orientations.io_orientation(img.affine)` mapping each original voxel axis
    to `(ras_axis, polarity_sign)`.
    """
    ornt_transform = np.asarray(nib.orientations.io_orientation(img.affine), dtype=np.float64)
    ras_img = nib.as_closest_canonical(img)
    data = np.asarray(ras_img.get_fdata(dtype=np.float32), dtype=np.float32)
    affine = np.asarray(ras_img.affine, dtype=np.float64)
    raw_zooms = ras_img.header.get_zooms()
    zooms = (float(raw_zooms[0]), float(raw_zooms[1]), float(raw_zooms[2]))
    return data, affine, zooms, ornt_transform


def canonicalize_to_ras(
    img: nib.Nifti1Image,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Reorient a NIfTI image to closest RAS+ canonical orientation and return float32 data."""
    data, affine, zooms, _ = canonicalize_to_ras_with_transform(img)
    return data, affine, zooms


def reorient_bvecs_to_ras(
    bvecs: np.ndarray,
    ornt_transform: np.ndarray,
) -> np.ndarray:
    """Remap voxel-space FSL/BIDS `(3, N)` gradient vectors into canonical RAS+ voxel axes."""
    arr = np.asarray(bvecs, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != 3:
        return arr.copy()
    out = np.empty_like(arr, dtype=np.float32)
    for in_ax in range(3):
        out_ax = int(ornt_transform[in_ax, 0])
        flip = float(ornt_transform[in_ax, 1])
        out[out_ax, :] = flip * arr[in_ax, :]
    return np.ascontiguousarray(out, dtype=np.float32)


def resolve_encoding_axes(
    sidecar: dict[str, Any] | None,
    modality: str,
    ornt_transform: np.ndarray | None = None,
) -> tuple[int, int, int]:
    """Resolve (phase_encode_axis, readout_axis, slice_axis) in RAS+ coordinates.

    When `ornt_transform` (from `canonicalize_to_ras_with_transform`) is supplied,
    sidecar `PhaseEncodingDirection` and `SliceEncodingDirection` (which refer to
    original voxel axes before RAS+ canonicalization) are remapped into RAS+ axes.
    Returns integer indices in {0 (R-L), 1 (A-P), 2 (S-I)}.
    """
    sidecar = sidecar or {}
    axis_map = {"i": 0, "i-": 0, "j": 1, "j-": 1, "k": 2, "k-": 2}

    def _map_axis(orig_ax: int) -> int:
        if ornt_transform is not None and ornt_transform.shape == (3, 2):
            return int(ornt_transform[orig_ax, 0])
        return orig_ax

    pe_raw = str(sidecar.get("PhaseEncodingDirection", "")).strip().lower()
    sl_raw = str(sidecar.get("SliceEncodingDirection", "")).strip().lower()

    if pe_raw in axis_map:
        pe_axis = _map_axis(axis_map[pe_raw])
    elif modality in ("bold", "dwi"):
        pe_axis = 1  # Anterior-Posterior standard for EPI
    else:
        pe_axis = 1  # AP default for structural

    if sl_raw in axis_map:
        mapped_sl = _map_axis(axis_map[sl_raw])
        slice_axis = mapped_sl if mapped_sl != pe_axis else (2 if pe_axis != 2 else 0)
    else:
        slice_axis = 2 if pe_axis != 2 else 0

    readout_candidates = [ax for ax in (0, 1, 2) if ax not in (pe_axis, slice_axis)]
    readout_axis = readout_candidates[0]
    return pe_axis, readout_axis, slice_axis


def estimate_background_noise_sigma(
    data: np.ndarray,
    bg_mask: np.ndarray,
    modality: str,
    bvals: np.ndarray | None = None,
) -> float:
    """Estimate native single-volume background noise sigma_0 on a clean input scan.

    Used by `excess_variance` noise calibration so real QC-passing scans (which already
    contain scanner thermal/coil noise) degrade consistently alongside simulated scans.
    """
    arr = np.nan_to_num(np.asarray(data, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    if modality == "dwi" and arr.ndim == 4:
        try:
            p_sig = np.asarray(piesno(arr, N=1, return_mask=False), dtype=np.float64)
            pos_sig = p_sig[np.isfinite(p_sig) & (p_sig > 1e-4)]
            if pos_sig.size > 0:
                return float(np.median(pos_sig))
        except Exception:
            pass
        try:
            est = np.asarray(estimate_sigma(arr, N=1), dtype=np.float64)
            pos_est = est[np.isfinite(est) & (est > 1e-4)]
            if pos_est.size > 0:
                return float(np.median(pos_est))
        except Exception:
            pass

    if int(bg_mask.sum()) < 16:
        return 0.0

    if arr.ndim == 4:
        n_t = arr.shape[3]
        if modality == "dwi" and bvals is not None and len(bvals) == n_t:
            b0_idx = np.where(bvals <= 50.0)[0]
            vol_indices = b0_idx[:8] if b0_idx.size > 0 else np.arange(min(8, n_t))
        else:
            step = max(1, n_t // 8)
            vol_indices = np.arange(0, n_t, step)[:8]
        sigmas: list[float] = []
        for idx in vol_indices:
            bg_v = arr[..., int(idx)][bg_mask]
            # Rayleigh / Dietrich estimator on single-volume background air
            s_std = float(np.std(bg_v)) * 1.5264
            s_mean = float(np.mean(np.abs(bg_v))) * 0.7979
            sigmas.append(min(s_std, s_mean) if s_std > 1e-6 else s_mean)
        return float(np.median(sigmas)) if sigmas else 0.0

    bg_vals = arr[bg_mask]
    s_std = float(np.std(bg_vals)) * 1.5264
    s_mean = float(np.mean(np.abs(bg_vals))) * 0.7979
    return float(min(s_std, s_mean) if s_std > 1e-6 else s_mean)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, num = ndimage.label(mask)
    if num <= 1:
        return mask
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest = int(np.argmax(counts))
    return np.asarray(labeled == largest, dtype=bool)


def compute_anatomical_priors(
    data: np.ndarray,
    bvals: np.ndarray | None = None,
) -> AnatomicalPriors:
    """Compute lightweight foreground/background/tissue priors from a 3D or 4D MRI volume."""
    if data.ndim == 4:
        if bvals is not None and len(bvals) == data.shape[3]:
            b0_idx = np.where(bvals <= 50.0)[0]
            ref = np.mean(data[..., b0_idx], axis=3) if len(b0_idx) > 0 else np.mean(data, axis=3)
        else:
            ref = np.mean(data, axis=3)
    else:
        ref = data

    ref = np.nan_to_num(ref.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    pos = ref[ref > 0]
    if pos.size < 32:
        shape = ref.shape
        brain_mask = np.ones(shape, dtype=bool)
        bg_mask = np.zeros(shape, dtype=bool)
        return AnatomicalPriors(
            brain_mask=brain_mask,
            bg_mask=bg_mask,
            wm_mask=brain_mask.copy(),
            gm_mask=np.zeros(shape, dtype=bool),
            bbox_min=(0, 0, 0),
            bbox_max=(shape[0] - 1, shape[1] - 1, shape[2] - 1),
            fg_median=1.0,
            fg_p95=1.0,
            bg_std=0.01,
        )

    p99 = float(np.percentile(pos, 99.5))
    clipped = np.clip(ref, 0.0, p99)
    try:
        # Light DIPY median_otsu denoising combined with conservative low-signal threshold
        # preserves cortical GM and darker CSF instead of shrinking to inner WM core
        _masked, otsu_mask = median_otsu(clipped, median_radius=1, numpass=1, dilate=1)
        thresh_low = float(threshold_otsu(clipped[clipped > 0])) * 0.45
        raw_fg = np.asarray(otsu_mask, dtype=bool) | (
            ref > max(thresh_low, float(np.percentile(pos, 10.0)))
        )
    except Exception:
        raw_fg = ref > float(np.percentile(pos, 25.0))

    raw_fg = ndimage.binary_closing(raw_fg, iterations=2)
    raw_fg = ndimage.binary_fill_holes(raw_fg)
    brain_mask = _largest_component(raw_fg)
    if int(brain_mask.sum()) < 64:
        brain_mask = ref > float(np.percentile(pos, 30.0))

    # Background air mask: outside a dilated head envelope
    dilated = ndimage.binary_dilation(brain_mask, iterations=4)
    bg_mask = ~dilated
    if int(bg_mask.sum()) < 32:
        bg_mask = ~brain_mask

    # Pseudo-tissue split inside brain mask (higher cluster vs. lower cluster)
    fg_vals = ref[brain_mask]
    fg_median = float(np.median(fg_vals))
    fg_p95 = float(np.percentile(fg_vals, 95.0))
    fg_p25 = float(np.percentile(fg_vals, 25.0))
    fg_p65 = float(np.percentile(fg_vals, 65.0))

    # Use Otsu inside core brain values (excluding extreme CSF/dark & bright outliers)
    core_vals = fg_vals[(fg_vals >= np.percentile(fg_vals, 15.0)) & (fg_vals <= fg_p95)]
    if core_vals.size > 64 and float(np.std(core_vals)) > 1e-6:
        try:
            tissue_split = float(threshold_otsu(core_vals))
        except Exception:
            tissue_split = fg_median
    else:
        tissue_split = fg_median

    wm_mask = brain_mask & (ref >= tissue_split) & (ref <= fg_p95)
    gm_mask = brain_mask & (ref >= fg_p25) & (ref < tissue_split)
    if int(wm_mask.sum()) < 16 or int(gm_mask.sum()) < 16:
        wm_mask = brain_mask & (ref >= fg_p65)
        gm_mask = brain_mask & (ref >= fg_p25) & (ref < fg_p65)

    coords = np.argwhere(brain_mask)
    c_min = coords.min(axis=0)
    c_max = coords.max(axis=0)
    bbox_min = (int(c_min[0]), int(c_min[1]), int(c_min[2]))
    bbox_max = (int(c_max[0]), int(c_max[1]), int(c_max[2]))

    bg_vals = ref[bg_mask] if int(bg_mask.sum()) > 0 else np.array([0.0], dtype=np.float32)
    bg_std = float(np.std(bg_vals)) if bg_vals.size > 1 else 0.0

    return AnatomicalPriors(
        brain_mask=brain_mask,
        bg_mask=bg_mask,
        wm_mask=wm_mask,
        gm_mask=gm_mask,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        fg_median=max(fg_median, 1e-6),
        fg_p95=max(fg_p95, 1e-6),
        bg_std=bg_std,
    )
