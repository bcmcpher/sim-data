"""Image-domain scanner/site physics profiles inducing location, scale, and covariance effects.

Split cleanly into two physical stages:
1. Pre-acquisition scanner profile (Stage SITE_PRE):
   - Tissue contrast transfer function (MONAI AdjustContrast gamma + gain + offset)
   - Anisotropic scanner reconstruction PSF (Gaussian smoothing kernel)
   - Static receive-coil B1 sensitivity profile (deterministic low-order polynomial + small jitter)
   - Subtle gradient nonlinearity / isotropic-anisotropic voxel scaling
2. Reconstruction noise floor (Stage NOISE_DRIFT):
   - Multi-coil Non-Central Chi (L coils) + spatial parallel-imaging g-factor noise floor

Supports `site.passthrough=True` for untouched native real-scanner baselines (bit-for-bit identity at s=0).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from monai.transforms import AdjustContrast
from scipy import ndimage

from sim_data.artifacts.base import ScanContext
from sim_data.artifacts.noise import apply_noncentral_chi_noise
from sim_data.config import SiteProfileConfig


def _build_polynomial_bias(
    spatial_shape: tuple[int, int, int],
    coeffs: tuple[float, float, float],
    order: int,
    jitter: np.ndarray,
) -> np.ndarray:
    """Construct a reproducible low-order 3D coil sensitivity field centered at 1.0."""
    xs = np.linspace(-1.0, 1.0, spatial_shape[0], dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, spatial_shape[1], dtype=np.float32)
    zs = np.linspace(-1.0, 1.0, spatial_shape[2], dtype=np.float32)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")

    cx = float(coeffs[0] + jitter[0])
    cy = float(coeffs[1] + jitter[1])
    cz = float(coeffs[2] + jitter[2])

    log_field = cx * gx + cy * gy + cz * gz
    if order >= 2:
        log_field = log_field + 0.45 * (
            cx * (gx**2 - 0.33) + cy * (gy**2 - 0.33) + cz * (gz**2 - 0.33)
        )
    if order >= 3:
        log_field = log_field + 0.25 * (cx * gx * gy + cy * gy * gz + cz * gx * gz)

    field = np.exp(np.clip(log_field, -0.75, 0.75)).astype(np.float32)
    return field


def apply_site_pre_profile(
    ctx: ScanContext,
    site: SiteProfileConfig,
) -> dict[str, Any]:
    """Apply Stage SITE_PRE intrinsic scanner physics (or exact no-op if `site.passthrough`)."""
    if site.passthrough:
        return {
            "site_id": site.site_id,
            "passthrough": True,
            "realized_gamma": 1.0,
            "realized_gain": 1.0,
            "realized_offset": 0.0,
            "realized_psf_sigma": [0.0, 0.0, 0.0],
            "realized_voxel_scale": [1.0, 1.0, 1.0],
            "coil_count": site.coil_count,
        }

    # In coupled trajectory mode, draw site jitter from dedicated site_seed so site physics stay fixed
    site_rng = np.random.default_rng(ctx.site_seed) if ctx.site_seed is not None else ctx.rng

    j = site.jitter_scale
    gamma = float(np.clip(site.contrast_gamma * (1.0 + site_rng.normal(0.0, 0.25 * j)), 0.45, 2.2))
    gain = float(np.clip(site.intensity_gain * (1.0 + site_rng.normal(0.0, 0.35 * j)), 0.4, 3.0))
    offset = float(site.intensity_offset + site_rng.normal(0.0, 0.05 * j))
    psf = tuple(
        max(0.0, float(s * (1.0 + site_rng.normal(0.0, 0.25 * j)))) for s in site.psf_sigma_vox
    )
    vox_scale = tuple(
        float(np.clip(vs * (1.0 + site_rng.normal(0.0, 0.03 * j)), 0.96, 1.04))
        for vs in site.voxel_scale
    )

    arr = ctx.data.astype(np.float32, copy=True)
    ref_scale = max(ctx.priors.fg_p95, 1e-6)

    if abs(gamma - 1.0) > 1e-3:
        normed = np.clip(arr / ref_scale, 0.0, 2.5)
        t = torch.from_numpy(normed).to(ctx.torch_device)
        adj = AdjustContrast(gamma=gamma)
        out_t = adj(t)
        if isinstance(out_t, torch.Tensor):
            arr = out_t.detach().cpu().numpy().astype(np.float32) * ref_scale
        else:
            arr = np.asarray(out_t, dtype=np.float32) * ref_scale

    arr = np.maximum(arr * gain + offset * ref_scale, 0.0)

    if any(s > 0.05 for s in psf):
        if arr.ndim == 4:
            for t_idx in range(arr.shape[3]):
                arr[..., t_idx] = ndimage.gaussian_filter(
                    arr[..., t_idx], sigma=psf, mode="nearest"
                )
        else:
            arr = ndimage.gaussian_filter(arr, sigma=psf, mode="nearest")

    if any(abs(c) > 1e-4 for c in site.bias_coeffs):
        bias_jitter = site_rng.normal(0.0, 0.15 * j, size=3)
        b1_field = _build_polynomial_bias(
            ctx.spatial_shape, site.bias_coeffs, site.bias_order, bias_jitter
        )
        brain_med = float(np.median(b1_field[ctx.priors.brain_mask]))
        if brain_med > 1e-4:
            b1_field = b1_field / brain_med
        if arr.ndim == 4:
            arr = arr * b1_field[..., None]
        else:
            arr = arr * b1_field

    if any(abs(vs - 1.0) > 2e-3 for vs in vox_scale):
        center = np.array(ctx.spatial_shape, dtype=np.float64) / 2.0
        inv_mat = np.diag([1.0 / vox_scale[0], 1.0 / vox_scale[1], 1.0 / vox_scale[2]])
        shift = center - inv_mat @ center
        if arr.ndim == 4:
            for t_idx in range(arr.shape[3]):
                arr[..., t_idx] = ndimage.affine_transform(
                    arr[..., t_idx], matrix=inv_mat, offset=shift, order=1, mode="nearest"
                )
        else:
            arr = ndimage.affine_transform(
                arr, matrix=inv_mat, offset=shift, order=1, mode="nearest"
            )

    ctx.data = np.maximum(arr.astype(np.float32), 0.0)

    return {
        "site_id": site.site_id,
        "passthrough": False,
        "realized_gamma": round(gamma, 4),
        "realized_gain": round(gain, 4),
        "realized_offset": round(offset, 4),
        "realized_psf_sigma": [round(s, 4) for s in psf],
        "realized_voxel_scale": [round(v, 4) for v in vox_scale],
        "coil_count": site.coil_count,
    }


def apply_site_noise_floor(
    ctx: ScanContext,
    site: SiteProfileConfig,
) -> dict[str, Any]:
    """Apply Stage NOISE_DRIFT site-specific coil noise floor (or exact no-op if `site.passthrough`)."""
    if site.passthrough or site.noise_sigma_rel <= 0.0:
        return {
            "site_noise_sigma_rel": 0.0,
            "site_noise_sigma_abs": 0.0,
            "site_g_factor": 0.0,
            "coil_count": site.coil_count,
            "passthrough": site.passthrough,
        }

    site_rng = (
        np.random.default_rng((ctx.site_seed + 17) % (2**31 - 1))
        if ctx.site_seed is not None
        else ctx.rng
    )

    j = site.jitter_scale
    sigma_rel = max(0.001, float(site.noise_sigma_rel * (1.0 + site_rng.normal(0.0, 0.20 * j))))
    g_strength = max(
        0.0, float(site.g_factor_strength * (1.0 + site_rng.normal(0.0, 0.15 * j)))
    )
    sigma_abs = sigma_rel * ctx.priors.fg_median * site.intensity_gain
    noise_seed = int(site_rng.integers(0, 2**31 - 1))

    ctx.data = apply_noncentral_chi_noise(
        ctx.data,
        sigma=sigma_abs,
        coil_count=site.coil_count,
        g_factor_strength=g_strength,
        seed=noise_seed,
        device=ctx.torch_device,
    )
    return {
        "site_noise_sigma_rel": round(sigma_rel, 5),
        "site_noise_sigma_abs": round(sigma_abs, 5),
        "site_g_factor": round(g_strength, 4),
        "coil_count": site.coil_count,
        "passthrough": False,
    }
