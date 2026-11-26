"""Tiered intracranial brain extraction (external mask -> DIPY EVAC+ -> morphological fallback)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from sim_data.config import BrainExtractionMode
from sim_data.io.orientation import compute_anatomical_priors
from sim_data.synthesis.tissue import resample_nifti_to_grid


@dataclass
class BrainMaskResult:
    """Separated cranial tissue compartments in RAS+ voxel coordinates."""

    icv_mask: np.ndarray  # bool (X, Y, Z) intracranial brain volume
    head_mask: np.ndarray  # bool (X, Y, Z) full head foreground envelope
    scalp_mask: np.ndarray  # bool (X, Y, Z) extracranial scalp/fat/neck
    bone_air_mask: np.ndarray  # bool (X, Y, Z) air + dark cortical skull bone
    method: str  # "external" | "evac" | "morphological" | "head_only"
    used_external_mask: bool


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, num = ndimage.label(mask)
    if num <= 1:
        return np.asarray(mask, dtype=bool)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    return np.asarray(labeled == int(np.argmax(counts)), dtype=bool)


def _is_human_scale_volume(shape: tuple[int, int, int], affine: np.ndarray) -> bool:
    """Detect human-scale 3D head MRI vs. tiny synthetic unit-test phantoms."""
    zooms = np.linalg.norm(affine[:3, :3], axis=0)
    fov_mm = np.array(shape, dtype=np.float64) * zooms
    return bool(min(shape) >= 64 and float(np.max(fov_mm)) >= 140.0)


def _run_dipy_evac(
    t1_vol: np.ndarray,
    affine: np.ndarray,
    device: str = "auto",
) -> np.ndarray | None:
    """Run DIPY's built-in PyTorch EVAC+ brain extractor on a 3D RAS+ T1w volume."""
    try:
        from dipy.nn.torch.evac import EVACPlus

        use_cuda = device == "cuda" or (device == "auto" and torch.cuda.is_available())
        model = EVACPlus(verbose=False, use_cuda=use_cuda)
        raw_mask = model.predict(
            np.asarray(t1_vol, dtype=np.float32),
            np.asarray(affine, dtype=np.float64),
            finalize_mask=True,
        )
        mask = np.asarray(raw_mask > 0.5, dtype=bool)
        if mask.shape == t1_vol.shape and int(mask.sum()) >= 128:
            return mask
    except Exception:
        return None
    return None


def _morphological_icv_fallback(
    t1_corr: np.ndarray,
    head_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Offline zero-download morphological ICV vs. scalp/bone separation.

    On full-head volumes, applies stronger opening to sever thin scalp/neck connections
    and keep the dominant superior intracranial cavity; on compact phantoms, preserves
    the gentle 2-voxel shell separation.
    """
    shape = (int(t1_corr.shape[0]), int(t1_corr.shape[1]), int(t1_corr.shape[2]))
    min_dim = min(shape)

    if min_dim >= 96:
        # Full-head scan: erode deeper (~5 voxels) to detach scalp/skull & narrow neck stalk
        erode_iters = max(4, min(7, min_dim // 32))
        core = ndimage.binary_erosion(head_mask, iterations=erode_iters)
        if int(core.sum()) >= 512:
            # Keep the largest interior blob and mildly re-dilate inside head_mask
            core = _largest_component(core)
            icv_cand = ndimage.binary_dilation(core, iterations=max(2, erode_iters - 2)) & head_mask
            icv_cand = ndimage.binary_fill_holes(icv_cand)
            outer_shell = head_mask & (~icv_cand)
            icv_p20 = float(np.percentile(t1_corr[icv_cand], 20.0)) if np.any(icv_cand) else 0.0
            scalp_mask = outer_shell & (t1_corr >= icv_p20 * 0.40)
            bone_air_mask = (~icv_cand) & (~scalp_mask)
            return np.asarray(icv_cand, dtype=bool), np.asarray(scalp_mask, dtype=bool), np.asarray(bone_air_mask, dtype=bool)

    # Compact volume / synthetic phantom fallback
    eroded = ndimage.binary_erosion(head_mask, iterations=2)
    outer_rim = head_mask & (~eroded)
    if int(eroded.sum()) < 64:
        icv_mask = head_mask.copy()
        scalp_mask = np.zeros(shape, dtype=bool)
        bone_air_mask = ~icv_mask
        return icv_mask, scalp_mask, bone_air_mask

    inner_vals = t1_corr[eroded]
    p15 = float(np.percentile(inner_vals, 15.0))
    p50 = float(np.percentile(inner_vals, 50.0))
    p97 = float(np.percentile(inner_vals, 97.0))

    scalp_cand = outer_rim & (
        ((t1_corr >= p15 * 0.45) & (t1_corr < p50 * 0.92)) | (t1_corr > p97)
    )
    dark_bone = outer_rim & (t1_corr < p15 * 0.45)
    icv_mask = head_mask & (~scalp_cand) & (~dark_bone)
    if int(icv_mask.sum()) < 64:
        icv_mask = eroded
    scalp_mask = scalp_cand
    bone_air_mask = (~icv_mask) & (~scalp_mask)
    return (
        np.asarray(icv_mask, dtype=bool),
        np.asarray(scalp_mask, dtype=bool),
        np.asarray(bone_air_mask, dtype=bool),
    )


def extract_brain_mask(
    t1_corr: np.ndarray,
    affine: np.ndarray,
    method: BrainExtractionMode = "auto",
    brain_mask_path: str | Path | None = None,
    brain_mask_array: np.ndarray | None = None,
    device: str = "auto",
) -> BrainMaskResult:
    """Separate intracranial brain (ICV) from skull bone/air and outer scalp/neck.

    Priority order:
    1. External `brain_mask_array` or `brain_mask_path` (always honored when supplied).
    2. `dipy.nn.torch.evac.EVACPlus` when `method="evac"` or (`method="auto"` on human-scale MRI).
    3. Offline morphological ICV heuristic (`method="morphological"` or fallback).
    4. Full foreground head mask (`method="none"`).
    """
    shape = (int(t1_corr.shape[0]), int(t1_corr.shape[1]), int(t1_corr.shape[2]))
    priors = compute_anatomical_priors(t1_corr)
    head_mask = priors.brain_mask

    icv_mask: np.ndarray | None = None
    used_external = False
    method_used = "morphological"

    if brain_mask_array is not None and brain_mask_array.shape == shape:
        icv_mask = np.asarray(brain_mask_array > 0.5, dtype=bool)
        used_external = True
        method_used = "external"
    elif brain_mask_path is not None:
        raw_mask = resample_nifti_to_grid(brain_mask_path, shape, affine, order=0)
        if raw_mask.ndim == 4:
            raw_mask = raw_mask[..., 0]
        icv_mask = np.asarray(raw_mask > 0.5, dtype=bool)
        used_external = True
        method_used = "external"
    elif method == "none":
        icv_mask = head_mask.copy()
        method_used = "head_only"
    elif method == "evac" or (method == "auto" and _is_human_scale_volume(shape, affine)):
        evac_mask = _run_dipy_evac(t1_corr, affine, device=device)
        if evac_mask is not None:
            icv_mask = evac_mask
            head_mask = head_mask | icv_mask
            method_used = "evac"

    if icv_mask is None:
        icv_m, scalp_m, bone_m = _morphological_icv_fallback(t1_corr, head_mask)
        return BrainMaskResult(
            icv_mask=icv_m,
            head_mask=head_mask,
            scalp_mask=scalp_m,
            bone_air_mask=bone_m,
            method="morphological",
            used_external_mask=False,
        )

    if int(icv_mask.sum()) < 32:
        icv_mask = head_mask.copy()

    outer_shell = head_mask & (~icv_mask)
    if np.any(outer_shell):
        icv_p20 = float(np.percentile(t1_corr[icv_mask], 20.0))
        scalp_mask = outer_shell & (t1_corr >= icv_p20 * 0.40)
    else:
        scalp_mask = np.zeros(shape, dtype=bool)
    bone_air_mask = (~icv_mask) & (~scalp_mask)

    return BrainMaskResult(
        icv_mask=np.asarray(icv_mask, dtype=bool),
        head_mask=np.asarray(head_mask, dtype=bool),
        scalp_mask=np.asarray(scalp_mask, dtype=bool),
        bone_air_mask=np.asarray(bone_air_mask, dtype=bool),
        method=method_used,
        used_external_mask=used_external,
    )
