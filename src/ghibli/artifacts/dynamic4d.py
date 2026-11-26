"""4D dynamic MRI artifacts: inter-volume rigid Motion4D (FD + spin-history) and DWI Slice Dropout."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

from ghibli.artifacts.base import BaseArtifactTransform, PipelineStage, ScanContext


def compute_power_fd(
    translations_mm: np.ndarray,
    rotations_rad: np.ndarray,
    radius_mm: float = 50.0,
) -> np.ndarray:
    """Compute Power (2012) Framewise Displacement in mm across timepoints (T,)."""
    n_t = translations_mm.shape[0]
    fd = np.zeros(n_t, dtype=np.float32)
    if n_t <= 1:
        return fd
    d_trans = np.abs(np.diff(translations_mm, axis=0)).sum(axis=1)
    d_rot = np.abs(np.diff(rotations_rad, axis=0)).sum(axis=1) * radius_mm
    fd[1:] = (d_trans + d_rot).astype(np.float32)
    return fd


class Motion4D(BaseArtifactTransform):
    """Inter-volume 6-DOF head motion trajectory for 4D BOLD and DWI.

    Uses Common-Random-Numbers (CRN) pre-sampled latent drift & jerk arrays so increasing
    `severity` along a coupled trajectory preserves exact jerk timing/direction while
    monotonically scaling motion amplitude and Power FD.
    """

    name = "motion_4d"
    stage = PipelineStage.SPATIAL_FIELD_4D
    supported_modalities = ("bold", "dwi")
    qc_weight = 1.2

    def _apply_impl(self, ctx: ScanContext) -> dict[str, Any]:
        if not ctx.is_4d:
            return {"skipped": True, "reason": "not_4d"}

        n_t = ctx.data.shape[3]
        zooms = np.array(ctx.zooms, dtype=np.float64)
        center = np.array(ctx.spatial_shape, dtype=np.float64) / 2.0

        drift_trans_std = float(self.params.get("drift_trans_mm", 0.08 + 0.55 * self.severity))
        drift_rot_deg = float(self.params.get("drift_rot_deg", 0.08 + 0.55 * self.severity))
        spike_prob = float(self.params.get("spike_prob", 0.04 + 0.20 * self.severity))
        spike_trans_mm = float(self.params.get("spike_trans_mm", 0.4 + 2.8 * self.severity))
        spike_rot_deg = float(self.params.get("spike_rot_deg", 0.4 + 2.5 * self.severity))
        rotate_bvecs = bool(self.params.get("rotate_bvecs", ctx.rotate_dwi_bvecs))

        # Pre-sample fixed-shape latent arrays once so CRN trajectories are strictly nested & monotone
        u_drift_trans = ctx.rng.normal(0.0, 1.0, size=(n_t, 3))
        u_drift_rot = ctx.rng.normal(0.0, 1.0, size=(n_t, 3))
        u_spike_trigger = ctx.rng.random(size=n_t)
        u_spike_trans = ctx.rng.normal(0.0, 1.0, size=(n_t, 3))
        u_spike_rot = ctx.rng.normal(0.0, 1.0, size=(n_t, 3))
        u_spin_sign = ctx.rng.choice([-1.0, 1.0], size=n_t)

        trans_mm = np.zeros((n_t, 3), dtype=np.float64)
        rot_deg = np.zeros((n_t, 3), dtype=np.float64)
        spike_frames: list[int] = []

        for t in range(1, n_t):
            trans_mm[t] = 0.85 * trans_mm[t - 1] + drift_trans_std * u_drift_trans[t]
            rot_deg[t] = 0.85 * rot_deg[t - 1] + drift_rot_deg * u_drift_rot[t]
            if float(u_spike_trigger[t]) < spike_prob:
                trans_mm[t] += spike_trans_mm * u_spike_trans[t]
                rot_deg[t] += spike_rot_deg * u_spike_rot[t]
                spike_frames.append(t)

        rot_rad = np.deg2rad(rot_deg)
        fd = compute_power_fd(trans_mm, rot_rad, radius_mm=50.0)

        out = ctx.data.copy()
        new_bvecs = ctx.bvecs.copy() if (ctx.bvecs is not None and rotate_bvecs) else None

        for t in range(n_t):
            rot_mat = Rotation.from_euler("xyz", rot_rad[t]).as_matrix()
            trans_vox = trans_mm[t] / np.maximum(zooms, 1e-3)
            inv_rot = rot_mat.T
            offset = center - inv_rot @ center - inv_rot @ trans_vox

            vol = ndimage.affine_transform(
                ctx.data[..., t], matrix=inv_rot, offset=offset, order=1, mode="nearest"
            )

            # Spin-history intensity transient on high-FD volumes (BOLD)
            if t > 0 and fd[t] > 0.25 and ctx.modality == "bold":
                spin_amp = float(np.clip(0.04 * fd[t], 0.0, 0.22))
                sign = float(u_spin_sign[t])
                vol = vol.copy()
                vol[:, :, ::2] *= 1.0 + sign * spin_amp
                vol[:, :, 1::2] *= 1.0 - 0.5 * sign * spin_amp

            out[..., t] = vol

            if new_bvecs is not None and new_bvecs.shape == (3, n_t):
                bvec = new_bvecs[:, t]
                norm = float(np.linalg.norm(bvec))
                if norm > 1e-4:
                    rotated = rot_mat @ bvec
                    new_bvecs[:, t] = rotated / (float(np.linalg.norm(rotated)) + 1e-8) * norm

        ctx.data = np.maximum(out.astype(np.float32), 0.0)
        if new_bvecs is not None:
            ctx.bvecs = new_bvecs.astype(np.float32)

        ctx.metadata["gt_fd_timeseries"] = [round(float(x), 4) for x in fd]
        ctx.metadata["gt_spike_frames"] = spike_frames
        ctx.metadata["gt_max_rel_translation"] = float(
            np.max(np.linalg.norm(np.diff(trans_mm, axis=0), axis=1)) if n_t > 1 else 0.0
        )
        ctx.metadata["gt_max_rel_rotation"] = float(
            np.max(np.linalg.norm(np.diff(rot_rad, axis=0), axis=1)) if n_t > 1 else 0.0
        )
        ctx.metadata["rotate_bvecs"] = rotate_bvecs

        fd_gt_02 = int(np.sum(fd > 0.2))
        return {
            "engine": "custom.Motion4D",
            "fd_mean_mm": round(float(np.mean(fd)), 4),
            "fd_max_mm": round(float(np.max(fd)), 4),
            "fd_num_gt_02": fd_gt_02,
            "fd_perc_gt_02": round(100.0 * fd_gt_02 / max(1, n_t), 2),
            "n_spike_frames": len(spike_frames),
            "rotate_bvecs": rotate_bvecs,
        }


class DWISliceDropout(BaseArtifactTransform):
    """Intra-TR diffusion signal dropout ('venetian blind' / dark slices) on high-b DWI volumes.

    Pre-samples volume/slice priority permutations and unit attenuation uniforms once per
    trajectory so increasing `severity` produces strictly nested dropped-slice sets and
    monotonically darker signal attenuation.
    """

    name = "dwi_slice_dropout"
    stage = PipelineStage.SPATIAL_FIELD_4D
    supported_modalities = ("dwi",)
    qc_weight = 1.2

    def _apply_impl(self, ctx: ScanContext) -> dict[str, Any]:
        if not ctx.is_4d:
            return {"skipped": True, "reason": "not_4d"}

        n_vols = ctx.data.shape[3]
        sl_axis = ctx.slice_axis
        bmin, bmax = ctx.priors.bbox_min[sl_axis], ctx.priors.bbox_max[sl_axis]
        valid_slices = np.arange(max(1, bmin + 1), min(ctx.spatial_shape[sl_axis] - 1, bmax))
        if valid_slices.size == 0:
            valid_slices = np.arange(ctx.spatial_shape[sl_axis])

        bvals = (
            ctx.bvals
            if (ctx.bvals is not None and len(ctx.bvals) == n_vols)
            else np.full(n_vols, 1000.0)
        )
        high_b_idx = np.where(bvals > 150.0)[0]
        if high_b_idx.size == 0:
            high_b_idx = np.arange(n_vols)

        # Pre-sample nested volume order, per-volume slice orders, and unit attenuation uniforms
        vol_perm = ctx.rng.permutation(high_b_idx)
        slice_perms: dict[int, np.ndarray] = {}
        u_attens: dict[int, np.ndarray] = {}
        for v_raw in high_b_idx:
            v_int = int(v_raw)
            slice_perms[v_int] = ctx.rng.permutation(valid_slices)
            u_attens[v_int] = ctx.rng.random(size=valid_slices.size)

        vol_frac = float(self.params.get("volume_fraction", 0.12 + 0.50 * self.severity))
        n_aff_vols = max(1, min(high_b_idx.size, int(round(high_b_idx.size * vol_frac))))
        aff_vols = sorted(int(v) for v in vol_perm[:n_aff_vols])

        min_atten = float(self.params.get("min_attenuation", max(0.03, 0.40 - 0.36 * self.severity)))
        max_atten = float(self.params.get("max_attenuation", max(0.10, 0.55 - 0.35 * self.severity)))
        n_slices_drop = max(1, min(valid_slices.size, int(round(1 + 3 * self.severity))))

        out = ctx.data.copy()
        dropped_pairs: list[tuple[int, int]] = []
        for v in aff_vols:
            chosen_slices = slice_perms[v][:n_slices_drop]
            for s_pos, z in enumerate(chosen_slices):
                atten = float(min_atten + u_attens[v][s_pos] * max(0.0, max_atten - min_atten))
                sel: list[Any] = [slice(None), slice(None), slice(None), v]
                sel[sl_axis] = int(z)
                out[tuple(sel)] *= atten
                dropped_pairs.append((int(v), int(z)))

        ctx.data = out
        total_brain_slices = max(1, int(valid_slices.size * high_b_idx.size))
        ctx.metadata["gt_dropped_slices_count"] = len(dropped_pairs)
        ctx.metadata["gt_dropped_pairs"] = dropped_pairs
        return {
            "engine": "custom.DWISliceDropout",
            "slice_axis": sl_axis,
            "affected_volumes": len(aff_vols),
            "n_dropped_slices": len(dropped_pairs),
            "dropped_slice_fraction": round(len(dropped_pairs) / total_brain_slices, 4),
        }
