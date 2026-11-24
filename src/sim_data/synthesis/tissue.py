"""Tissue Partial Volume Estimation (PVE) & 3T MRI tissue parameter priors.

Supports both:
- Template-free GMM + MRF spatial regularization from a seed T1w (+ optional co-registered T2w).
- Optional external 4D PVE NIfTI resampling into the seed RAS+ grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from sklearn.mixture import GaussianMixture

from sim_data.io.orientation import canonicalize_to_ras, compute_anatomical_priors


@dataclass
class TissuePVEMaps:
    """Soft tissue partial-volume probability maps in RAS+ voxel coordinates."""

    csf: np.ndarray  # float32 [0, 1]
    gm: np.ndarray  # float32 [0, 1]
    wm: np.ndarray  # float32 [0, 1]
    scalp: np.ndarray  # float32 [0, 1]
    brain_mask: np.ndarray  # bool
    head_mask: np.ndarray  # bool


@dataclass
class TissueMRIParams:
    """Subject-specific 3T MRI tissue physical parameters (proton density, T1, T2, T2*)."""

    rho: dict[str, float]
    t1_ms: dict[str, float]
    t2_ms: dict[str, float]
    t2s_ms: dict[str, float]


def resample_nifti_to_grid(
    nifti_path: str | Path,
    target_shape: tuple[int, int, int],
    target_affine: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    """Load a 3D or 4D NIfTI template and resample it onto (target_shape, target_affine) in RAS+."""
    img = nib.load(str(nifti_path))
    if not isinstance(img, nib.Nifti1Image):
        if isinstance(img, nib.spatialimages.SpatialImage):
            img = nib.Nifti1Image(np.asarray(img.get_fdata(), dtype=np.float32), img.affine)
        else:
            raise TypeError(f"Unsupported template image type: {type(img).__name__}")
    src_data, src_affine, _ = canonicalize_to_ras(img)

    # Map target voxel index -> world -> source voxel index
    rel = np.linalg.inv(src_affine) @ target_affine
    mat = rel[:3, :3]
    offset = rel[:3, 3]

    if src_data.ndim == 3:
        out = ndimage.affine_transform(
            src_data, matrix=mat, offset=offset, output_shape=target_shape, order=order, mode="nearest"
        )
        return np.asarray(out, dtype=np.float32)

    n_ch = src_data.shape[3]
    out_4d = np.empty((*target_shape, n_ch), dtype=np.float32)
    for c in range(n_ch):
        out_4d[..., c] = ndimage.affine_transform(
            src_data[..., c],
            matrix=mat,
            offset=offset,
            output_shape=target_shape,
            order=order,
            mode="nearest",
        )
    return out_4d


def sample_tissue_mri_params(
    rng: np.random.Generator,
    jitter: float = 0.06,
) -> TissueMRIParams:
    """Sample realistic 3T tissue relaxation constants with mild per-subject biological variation."""
    base_rho = {"wm": 0.77, "gm": 0.86, "csf": 1.00, "scalp": 0.60}
    base_t1 = {"wm": 830.0, "gm": 1330.0, "csf": 4160.0, "scalp": 550.0}
    base_t2 = {"wm": 80.0, "gm": 110.0, "csf": 1500.0, "scalp": 60.0}
    base_t2s = {"wm": 45.0, "gm": 52.0, "csf": 180.0, "scalp": 22.0}

    def _jitter_dict(base: dict[str, float]) -> dict[str, float]:
        if jitter <= 0.0:
            return dict(base)
        return {
            k: float(v * np.clip(1.0 + rng.normal(0.0, jitter), 0.75, 1.25))
            for k, v in base.items()
        }

    return TissueMRIParams(
        rho=_jitter_dict(base_rho),
        t1_ms=_jitter_dict(base_t1),
        t2_ms=_jitter_dict(base_t2),
        t2s_ms=_jitter_dict(base_t2s),
    )


def estimate_tissue_pve(
    t1_data: np.ndarray,
    target_affine: np.ndarray,
    t2_data: np.ndarray | None = None,
    template_pve_path: str | Path | None = None,
    seed: int = 42,
) -> TissuePVEMaps:
    """Estimate smooth soft tissue PVE maps (CSF, GM, WM, Scalp) from T1w (+ optional T2w or template)."""
    t1 = np.nan_to_num(np.asarray(t1_data, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    shape = (int(t1.shape[0]), int(t1.shape[1]), int(t1.shape[2]))

    if template_pve_path is not None:
        pve_raw = resample_nifti_to_grid(template_pve_path, shape, target_affine, order=1)
        if pve_raw.ndim != 4 or pve_raw.shape[3] < 3:
            raise ValueError(
                f"Expected 4D PVE NIfTI with >=3 channels (CSF, GM, WM[, Scalp]), got shape {pve_raw.shape}"
            )
        csf = np.clip(pve_raw[..., 0], 0.0, 1.0)
        gm = np.clip(pve_raw[..., 1], 0.0, 1.0)
        wm = np.clip(pve_raw[..., 2], 0.0, 1.0)
        scalp = (
            np.clip(pve_raw[..., 3], 0.0, 1.0)
            if pve_raw.shape[3] >= 4
            else np.zeros(shape, dtype=np.float32)
        )
        total = csf + gm + wm + scalp
        head_mask = total > 0.10
        brain_mask = (csf + gm + wm) > 0.15
        denom = np.where(head_mask, np.maximum(total, 1e-5), 1.0)
        return TissuePVEMaps(
            csf=np.asarray(np.where(head_mask, csf / denom, 0.0), dtype=np.float32),
            gm=np.asarray(np.where(head_mask, gm / denom, 0.0), dtype=np.float32),
            wm=np.asarray(np.where(head_mask, wm / denom, 0.0), dtype=np.float32),
            scalp=np.asarray(np.where(head_mask, scalp / denom, 0.0), dtype=np.float32),
            brain_mask=np.asarray(brain_mask, dtype=bool),
            head_mask=np.asarray(head_mask, dtype=bool),
        )

    priors = compute_anatomical_priors(t1)
    head_mask = priors.brain_mask

    # Identify outer boundary rim as scalp candidate while protecting interior cortex/ventricles
    eroded = ndimage.binary_erosion(head_mask, iterations=2)
    outer_rim = head_mask & (~eroded)
    if int(eroded.sum()) < 64:
        eroded = head_mask
        outer_rim = np.zeros(shape, dtype=bool)

    fg_vals = t1[head_mask]
    t1_p98 = max(float(np.percentile(fg_vals, 98.0)), 1e-5)
    t1_norm = np.clip(t1 / t1_p98, 0.0, 1.5)

    has_t2 = t2_data is not None and t2_data.shape == t1.shape
    if has_t2 and t2_data is not None:
        t2 = np.nan_to_num(np.asarray(t2_data, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        t2_p98 = max(float(np.percentile(t2[head_mask], 98.0)), 1e-5)
        t2_norm = np.clip(t2 / t2_p98, 0.0, 1.5)
        feats_fit = np.column_stack([t1_norm[eroded], t2_norm[eroded]])
        feats_all = np.column_stack([t1_norm[head_mask], t2_norm[head_mask]])
    else:
        t2_norm = None
        feats_fit = t1_norm[eroded][:, None]
        feats_all = t1_norm[head_mask][:, None]

    # Fit 3-class GaussianMixture (CSF, GM, WM) on subsampled interior voxels for speed
    rng = np.random.default_rng(seed)
    n_fit = feats_fit.shape[0]
    if n_fit > 25_000:
        sub_idx = rng.choice(n_fit, size=25_000, replace=False)
        fit_sample = feats_fit[sub_idx]
    else:
        fit_sample = feats_fit

    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=seed, max_iter=100)
    gmm.fit(fit_sample)
    probs = gmm.predict_proba(feats_all)  # (N_head, 3)

    # Order GMM components physically: CSF (low T1, high T2) < GM (mid) < WM (high T1, low T2)
    means = np.asarray(gmm.means_, dtype=np.float64)
    if has_t2:
        scores = means[:, 0] - 0.55 * means[:, 1]
    else:
        scores = means[:, 0]
    order = np.argsort(scores)
    idx_csf, idx_gm, idx_wm = int(order[0]), int(order[1]), int(order[2])

    csf = np.zeros(shape, dtype=np.float32)
    gm = np.zeros(shape, dtype=np.float32)
    wm = np.zeros(shape, dtype=np.float32)
    scalp = np.zeros(shape, dtype=np.float32)

    csf[head_mask] = probs[:, idx_csf].astype(np.float32)
    gm[head_mask] = probs[:, idx_gm].astype(np.float32)
    wm[head_mask] = probs[:, idx_wm].astype(np.float32)

    # Separate outer rim scalp/skull tissue: intermediate-to-low T1 (and low T2 if available) on head exterior
    if np.any(outer_rim):
        if t2_norm is not None:
            scalp_cand = outer_rim & (t2_norm < float(np.percentile(t2_norm[eroded], 45.0)))
        else:
            # Outer shell voxels not clearly bright CSF-dark or WM-bright
            gm_mean_t1 = float(means[idx_gm, 0])
            scalp_cand = outer_rim & (t1_norm < gm_mean_t1 * 0.92)
        scalp[scalp_cand] = 0.80
        csf[scalp_cand] *= 0.20
        gm[scalp_cand] *= 0.20
        wm[scalp_cand] *= 0.20

    # Mild spatial MRF smoothing of posteriors + exact simplex normalization inside head_mask
    csf = ndimage.gaussian_filter(csf, sigma=0.6) * head_mask
    gm = ndimage.gaussian_filter(gm, sigma=0.6) * head_mask
    wm = ndimage.gaussian_filter(wm, sigma=0.6) * head_mask
    scalp = ndimage.gaussian_filter(scalp, sigma=0.6) * head_mask

    total = csf + gm + wm + scalp
    valid = total > 1e-4
    denom = np.where(valid, total, 1.0)
    csf = np.where(valid, csf / denom, 0.0).astype(np.float32)
    gm = np.where(valid, gm / denom, 0.0).astype(np.float32)
    wm = np.where(valid, wm / denom, 0.0).astype(np.float32)
    scalp = np.where(valid, scalp / denom, 0.0).astype(np.float32)

    brain_mask = (csf + gm + wm) > 0.35
    return TissuePVEMaps(
        csf=csf,
        gm=gm,
        wm=wm,
        scalp=scalp,
        brain_mask=np.asarray(brain_mask, dtype=bool),
        head_mask=np.asarray(head_mask, dtype=bool),
    )
