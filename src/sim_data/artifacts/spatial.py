"""Anatomy-aware post-reconstruction spatial defects: FOV Cropping and Defacing Damage."""

from __future__ import annotations

from typing import Any

import numpy as np

from sim_data.artifacts.base import BaseArtifactTransform, PipelineStage, ScanContext


class FOVCropping(BaseArtifactTransform):
    """Clips superior (vertex), inferior (cerebellum), anterior, or posterior brain slabs.

    Supports:
    - mode="zero_slab": zeroes out the clipped boundary slab (preserves matrix dimensions).
    - mode="truncate_matrix": physically slices the NIfTI volume and updates the affine origin.
    """

    name = "fov_cropping"
    stage = PipelineStage.POST_RECON
    supported_modalities = ("T1w", "T2w", "FLAIR", "bold", "dwi")
    qc_weight = 1.1

    def _apply_impl(self, ctx: ScanContext) -> dict[str, Any]:
        side = str(
            self.params.get(
                "side",
                str(ctx.rng.choice(["inferior", "superior", "anterior", "posterior"])),
            )
        )
        mode = str(self.params.get("mode", "zero_slab"))
        crop_frac = float(self.params.get("crop_fraction", 0.06 + 0.22 * self.severity))

        axis_side_map = {
            "inferior": (2, "low"),
            "superior": (2, "high"),
            "posterior": (1, "low"),
            "anterior": (1, "high"),
            "left": (0, "low"),
            "right": (0, "high"),
        }
        axis, end = axis_side_map.get(side, (2, "low"))
        bmin, bmax = ctx.priors.bbox_min[axis], ctx.priors.bbox_max[axis]
        brain_span = max(6, bmax - bmin + 1)
        cut_depth = max(2, int(round(brain_span * crop_frac)))

        brain_total = max(1, int(ctx.priors.brain_mask.sum()))
        shape = ctx.spatial_shape

        if end == "low":
            cut_idx = min(shape[axis] - 4, bmin + cut_depth)
            keep_slice = slice(cut_idx, shape[axis])
            drop_slice = slice(0, cut_idx)
        else:
            cut_idx = max(4, bmax - cut_depth + 1)
            keep_slice = slice(0, cut_idx)
            drop_slice = slice(cut_idx, shape[axis])

        drop_selector: list[Any] = [slice(None), slice(None), slice(None)]
        drop_selector[axis] = drop_slice
        clipped_voxels = int(ctx.priors.brain_mask[tuple(drop_selector)].sum())
        clipped_pct = 100.0 * clipped_voxels / brain_total

        if mode == "truncate_matrix":
            keep_selector: list[Any] = [slice(None), slice(None), slice(None)]
            keep_selector[axis] = keep_slice
            if ctx.is_4d:
                keep_selector.append(slice(None))
            ctx.data = np.ascontiguousarray(ctx.data[tuple(keep_selector)])
            if end == "low":
                shift_vox = np.zeros(3, dtype=np.float64)
                shift_vox[axis] = float(cut_idx)
                new_affine = ctx.affine.copy()
                new_affine[:3, 3] = new_affine[:3, 3] + new_affine[:3, :3] @ shift_vox
                ctx.affine = new_affine
        else:
            arr = ctx.data.copy()
            arr[tuple(drop_selector)] = 0.0
            ctx.data = arr

        ctx.metadata["fov_cropped_pct"] = round(clipped_pct, 2)
        return {
            "engine": "custom.FOVCropping",
            "side": side,
            "mode": mode,
            "crop_fraction": round(crop_frac, 4),
            "cut_idx": int(cut_idx),
            "brain_voxels_clipped_pct": round(clipped_pct, 2),
        }


class DefacingDamage(BaseArtifactTransform):
    """Post-noise over-aggressive defacing damage in RAS+ coordinates.

    Applied strictly after noise (PipelineStage.POST_RECON) so damaged regions are
    authentically zero-valued with sharp borders, matching PyDeface/MRI_deface/Quickshear bugs.
    """

    name = "defacing_damage"
    stage = PipelineStage.POST_RECON
    supported_modalities = ("T1w", "T2w", "FLAIR")
    qc_weight = 1.25

    def _apply_impl(self, ctx: ScanContext) -> dict[str, Any]:
        shape = ctx.spatial_shape
        bmin = np.array(ctx.priors.bbox_min, dtype=np.float32)
        bmax = np.array(ctx.priors.bbox_max, dtype=np.float32)
        bspan = np.maximum(bmax - bmin, 6.0)

        pattern = str(
            self.params.get(
                "pattern",
                str(ctx.rng.choice(["anterior_wedge", "orbitofrontal_slab", "ear_temporal"])),
            )
        )

        gx, gy, gz = np.meshgrid(
            np.arange(shape[0], dtype=np.float32),
            np.arange(shape[1], dtype=np.float32),
            np.arange(shape[2], dtype=np.float32),
            indexing="ij",
        )
        # Normalized brain coordinates in [0, 1] along R-L (x), P-A (y), I-S (z)
        nx = (gx - bmin[0]) / bspan[0]
        ny = (gy - bmin[1]) / bspan[1]
        nz = (gz - bmin[2]) / bspan[2]

        if pattern == "anterior_wedge":
            # Oblique plane slicing from anterior-superior down through orbitofrontal cortex
            depth = float(self.params.get("depth", 0.08 + 0.30 * self.severity))
            # Plane equation: ny - 0.35 * nz > (1 - depth) cuts anterior-inferior aggressively
            cut_mask = (ny - 0.35 * nz) > (0.96 - depth)
        elif pattern == "orbitofrontal_slab":
            # Rectangular/ellipsoidal facial box extending too far posterior & superior
            y_cut = float(0.92 - (0.08 + 0.28 * self.severity))
            z_cut = float(0.38 + 0.32 * self.severity)
            cut_mask = (ny >= y_cut) & (nz <= z_cut) & (nx >= 0.12) & (nx <= 0.88)
        else:
            # Bilateral/unilateral ear-defacing overshoot clipping lateral temporal cortex
            side = str(ctx.rng.choice(["left", "right", "both"]))
            inset = float(0.06 + 0.20 * self.severity)
            lat_mask = np.zeros(shape, dtype=bool)
            if side in ("left", "both"):
                lat_mask |= nx <= inset
            if side in ("right", "both"):
                lat_mask |= nx >= (1.0 - inset)
            cut_mask = lat_mask & (ny >= 0.20) & (ny <= 0.80) & (nz >= 0.10) & (nz <= 0.65)

        brain_total = max(1, int(ctx.priors.brain_mask.sum()))
        damaged_voxels = int((ctx.priors.brain_mask & cut_mask).sum())
        damaged_pct = 100.0 * damaged_voxels / brain_total

        arr = ctx.data.copy()
        if ctx.is_4d:
            arr[cut_mask, :] = 0.0
        else:
            arr[cut_mask] = 0.0
        ctx.data = arr
        ctx.metadata["defacing_damaged_pct"] = round(damaged_pct, 2)

        return {
            "engine": "custom.DefacingDamage",
            "pattern": pattern,
            "brain_voxels_damaged_pct": round(damaged_pct, 2),
        }
