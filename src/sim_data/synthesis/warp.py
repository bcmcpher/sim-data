"""Coherent 3D diffeomorphic spatial warping (scaling-and-squaring) & Jacobian PPD vector reorientation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

from sim_data.config import DiffeomorphicWarpConfig


@dataclass
class DiffeomorphicField:
    """Coherent 3D diffeomorphic coordinate mapping shared across all modalities of a subject."""

    coords: np.ndarray  # float32 (3, X, Y, Z) sampling coordinates in source voxel space
    displacement_vox: np.ndarray  # float32 (3, X, Y, Z) nonlinear displacement
    affine_3x3: np.ndarray  # float64 (3, 3) linear part of inverse affine + flip
    flipped_lr: bool
    max_disp_mm: float
    mean_jac_det: float
    min_jac_det: float


def _compose_displacement(
    disp_a: np.ndarray,
    disp_b: np.ndarray,
    base_grid: np.ndarray,
) -> np.ndarray:
    """Compose two 3D displacement fields: (Id + disp_a) o (Id + disp_b) - Id."""
    sample_coords = base_grid + disp_b
    warped_a = np.empty_like(disp_a)
    for axis in range(3):
        warped_a[axis] = ndimage.map_coordinates(
            disp_a[axis], sample_coords, order=1, mode="nearest"
        )
    return (disp_b + warped_a).astype(np.float32)


def _jacobian_determinant(coords: np.ndarray) -> np.ndarray:
    """Compute pointwise 3x3 Jacobian determinant of a (3, X, Y, Z) coordinate mapping."""
    gx = np.gradient(coords[0])
    gy = np.gradient(coords[1])
    gz = np.gradient(coords[2])
    det = (
        gx[0] * (gy[1] * gz[2] - gy[2] * gz[1])
        - gx[1] * (gy[0] * gz[2] - gy[2] * gz[0])
        + gx[2] * (gy[0] * gz[1] - gy[1] * gz[0])
    )
    return np.asarray(det, dtype=np.float32)


def sample_diffeomorphic_field(
    shape: tuple[int, int, int],
    zooms: tuple[float, float, float],
    cfg: DiffeomorphicWarpConfig,
    rng: np.random.Generator,
) -> DiffeomorphicField:
    """Sample a topology-preserving 3D diffeomorphic warp + mild affine + optional L-R flip."""
    sx, sy, sz = shape
    gx, gy, gz = np.meshgrid(
        np.arange(sx, dtype=np.float32),
        np.arange(sy, dtype=np.float32),
        np.arange(sz, dtype=np.float32),
        indexing="ij",
    )
    base_grid = np.stack([gx, gy, gz], axis=0)  # (3, X, Y, Z)
    zoom_arr = np.maximum(np.array(zooms, dtype=np.float32), 0.2)

    # 1. Smooth stationary velocity field integrated via 4-step scaling-and-squaring
    disp = np.zeros_like(base_grid)
    if cfg.strength > 0.0:
        sigma_vox = np.maximum(cfg.smooth_sigma_mm / zoom_arr, 2.0)
        vel = np.empty_like(base_grid)
        for ax in range(3):
            noise = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
            vel[ax] = ndimage.gaussian_filter(noise, sigma=tuple(float(s) for s in sigma_vox))

        # Scale smoothly so peak displacement is proportional to strength (~0..7 mm)
        # while staying strictly bounded below folding threshold
        vel_mm = np.sqrt(
            (vel[0] * zoom_arr[0]) ** 2 + (vel[1] * zoom_arr[1]) ** 2 + (vel[2] * zoom_arr[2]) ** 2
        )
        p99 = max(float(np.percentile(vel_mm, 99.0)), 1e-5)
        target_mm = float(min(7.5 * cfg.strength, 0.38 * cfg.smooth_sigma_mm))
        vel = vel * (target_mm / p99)

        # Scaling and squaring: exp(v) = (Id + v / 2^4)^(2^4) guarantees det(J) > 0
        disp = (vel / 16.0).astype(np.float32)
        for _ in range(4):
            disp = _compose_displacement(disp, disp, base_grid)

    # 2. Mild random global affine (anisotropic scale + small rotation + shear)
    lo_s, hi_s = cfg.affine_scale_range
    scales = rng.uniform(lo_s, hi_s, size=3) if cfg.strength > 0.0 else np.ones(3)
    rot_deg = rng.uniform(-cfg.affine_rot_deg, cfg.affine_rot_deg, size=3) * cfg.strength
    rot_mat = Rotation.from_euler("xyz", np.deg2rad(rot_deg)).as_matrix()

    shear_mat = np.eye(3, dtype=np.float64)
    if cfg.affine_shear > 0.0 and cfg.strength > 0.0:
        s_lim = cfg.affine_shear * cfg.strength
        shear_mat[0, 1] = float(rng.uniform(-s_lim, s_lim))
        shear_mat[0, 2] = float(rng.uniform(-s_lim, s_lim))
        shear_mat[1, 2] = float(rng.uniform(-s_lim, s_lim))

    flipped_lr = bool(rng.random() < cfg.flip_lr_prob)
    flip_mat = np.diag([-1.0 if flipped_lr else 1.0, 1.0, 1.0])

    # Inverse mapping from output voxel coordinates to source voxel coordinates
    lin_inv = flip_mat @ np.linalg.inv(rot_mat @ shear_mat @ np.diag(scales))

    center = np.array([(sx - 1) / 2.0, (sy - 1) / 2.0, (sz - 1) / 2.0], dtype=np.float64)
    shifted = (base_grid + disp).astype(np.float64) - center[:, None, None, None]
    coords_flat = lin_inv @ shifted.reshape(3, -1) + center[:, None]
    coords = coords_flat.reshape(3, sx, sy, sz).astype(np.float32)

    # Evaluate Jacobian determinant of the nonlinear diffeomorphism (sign-independent of L-R flip)
    nl_coords = base_grid + disp
    jac_det = _jacobian_determinant(nl_coords)
    # Measure over interior volume excluding 1-voxel boundary gradient edge
    inner = jac_det[1:-1, 1:-1, 1:-1] if min(shape) > 4 else jac_det
    disp_mm = np.sqrt(
        (disp[0] * zoom_arr[0]) ** 2 + (disp[1] * zoom_arr[1]) ** 2 + (disp[2] * zoom_arr[2]) ** 2
    )

    return DiffeomorphicField(
        coords=coords,
        displacement_vox=disp,
        affine_3x3=lin_inv,
        flipped_lr=flipped_lr,
        max_disp_mm=round(float(np.max(disp_mm)), 3),
        mean_jac_det=round(float(np.mean(inner)), 4),
        min_jac_det=round(float(np.min(inner)), 4),
    )


def warp_volume(
    vol: np.ndarray,
    field: DiffeomorphicField,
    order: int = 1,
) -> np.ndarray:
    """Apply a DiffeomorphicField to a 3D scalar or integer label volume."""
    out = ndimage.map_coordinates(vol, field.coords, order=order, mode="nearest")
    return np.asarray(out, dtype=vol.dtype if order == 0 else np.float32)


def warp_vector_field_ppd(
    vec_field: np.ndarray,
    field: DiffeomorphicField,
) -> np.ndarray:
    """Warp a 3D unit vector field (X, Y, Z, 3) with Jacobian Preservation of Principal Directions.

    Resamples vector components smoothly and applies the local forward Jacobian (inverse of the
    pullback coordinate gradient) so fiber orientations rotate consistently with anatomical warps.
    """
    warped = np.empty_like(vec_field, dtype=np.float32)
    for k in range(3):
        warped[..., k] = ndimage.map_coordinates(
            vec_field[..., k], field.coords, order=1, mode="nearest"
        )

    # Apply linear forward transform (inverse of lin_inv) + local nonlinear Jacobian perturbation
    lin_fwd = np.linalg.inv(field.affine_3x3)
    # Approximate local forward Jacobian: J_fwd ~ lin_fwd @ (I - grad(disp))
    gx = np.gradient(field.displacement_vox[0])
    gy = np.gradient(field.displacement_vox[1])
    gz = np.gradient(field.displacement_vox[2])

    # Local (I - grad_disp) applied to warped vectors, then linear affine rotation/flip
    vx, vy, vz = warped[..., 0], warped[..., 1], warped[..., 2]
    rx = (1.0 - gx[0]) * vx - gx[1] * vy - gx[2] * vz
    ry = -gy[0] * vx + (1.0 - gy[1]) * vy - gy[2] * vz
    rz = -gz[0] * vx - gz[1] * vy + (1.0 - gz[2]) * vz

    stacked = np.stack([rx, ry, rz], axis=0).reshape(3, -1).astype(np.float64)
    rot_vec = (lin_fwd @ stacked).reshape(3, *vec_field.shape[:3])
    out = np.moveaxis(rot_vec, 0, -1).astype(np.float32)

    norms = np.linalg.norm(out, axis=-1, keepdims=True)
    valid = norms[..., 0] > 1e-5
    out[valid] = out[valid] / norms[valid]
    return out
