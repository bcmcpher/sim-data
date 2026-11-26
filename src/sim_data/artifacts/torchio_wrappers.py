"""TorchIO-backed severity-calibrated MRI artifact transforms."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torchio as tio

from sim_data.artifacts.base import BaseArtifactTransform, PipelineStage, ScanContext


def _run_torchio_transform(
    transform: tio.Transform,
    data: np.ndarray,
    affine: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Run a TorchIO transform deterministically on 3D (X,Y,Z) or 4D (X,Y,Z,T) data."""
    was_4d = data.ndim == 4
    if was_4d:
        tensor = torch.from_numpy(np.ascontiguousarray(np.moveaxis(data, -1, 0), dtype=np.float32))
    else:
        tensor = torch.from_numpy(np.ascontiguousarray(data[None, ...], dtype=np.float32))

    image = tio.ScalarImage(tensor=tensor, affine=affine)
    cpu_state = torch.random.get_rng_state()
    np_state = np.random.get_state()
    try:
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32 - 1))
        out_img = transform(image)
    finally:
        torch.random.set_rng_state(cpu_state)
        np.random.set_state(np_state)

    out_arr = out_img.data.detach().cpu().numpy().astype(np.float32, copy=False)
    if was_4d:
        return np.ascontiguousarray(np.moveaxis(out_arr, 0, -1))
    return np.ascontiguousarray(out_arr[0])


class Motion3D(BaseArtifactTransform):
    """3D k-space intra-scan rigid head motion via deterministic CRN-scaled TorchIO Motion.

    Pre-samples unit rotation/translation directions and sorted k-space segment times from
    `ctx.rng`, then scales pose amplitudes strictly monotonically with `severity` so coupled
    dose trajectories progress smoothly from mild blur/ghosting to severe clinical ringing.
    """

    name = "motion_3d"
    stage = PipelineStage.KSPACE
    supported_modalities = ("T1w", "T2w", "FLAIR")
    qc_weight = 1.15

    def _apply_impl(self, ctx: ScanContext) -> dict[str, Any]:
        max_deg = float(self.params.get("degrees", 1.8 + 14.2 * (self.severity**1.15)))
        max_trans = float(self.params.get("translation", 1.2 + 10.8 * (self.severity**1.15)))
        default_n = 4 if ctx.coupled_trajectory else max(2, int(round(2 + 4 * self.severity)))
        num_transforms = int(self.params.get("num_transforms", default_n))

        # Pre-sample CRN unit pose directions & sorted transition times spanning central k-space
        signs_r = ctx.rng.choice([-1.0, 1.0], size=(num_transforms, 3))
        mags_r = ctx.rng.uniform(0.45, 1.0, size=(num_transforms, 3))
        signs_t = ctx.rng.choice([-1.0, 1.0], size=(num_transforms, 3))
        mags_t = ctx.rng.uniform(0.45, 1.0, size=(num_transforms, 3))
        raw_times = np.sort(ctx.rng.uniform(0.18, 0.84, size=num_transforms))
        # Guarantee at least one movement hits the central k-space energy band (0.38..0.62)
        mid_slot = num_transforms // 2
        raw_times[mid_slot] = float(ctx.rng.uniform(0.40, 0.60))
        times = np.sort(np.clip(raw_times, 0.12, 0.88))

        degrees_arr = signs_r * mags_r * max_deg
        trans_arr = signs_t * mags_t * max_trans

        tfm = tio.Motion(
            degrees=degrees_arr,
            translation=trans_arr,
            times=times,
            image_interpolation="linear",
        )
        ctx.data = np.maximum(
            _run_torchio_transform(tfm, ctx.data, ctx.affine, seed=ctx.spawn_seed()),
            0.0,
        )
        ctx.metadata["gt_motion3d_degrees"] = np.round(degrees_arr, 3).tolist()
        ctx.metadata["gt_motion3d_translation_mm"] = np.round(trans_arr, 3).tolist()
        ctx.metadata["gt_motion3d_times"] = [round(float(t), 4) for t in times]
        return {
            "engine": "torchio.Motion",
            "degrees": round(max_deg, 3),
            "translation_mm": round(max_trans, 3),
            "num_transforms": num_transforms,
            "times": [round(float(t), 3) for t in times],
        }


class Ghosting3D(BaseArtifactTransform):
    """Periodic k-space ghosting along phase-encoding axis via TorchIO RandomGhosting."""

    name = "ghosting_3d"
    stage = PipelineStage.KSPACE
    supported_modalities = ("T1w", "T2w", "FLAIR")
    qc_weight = 1.05

    def _apply_impl(self, ctx: ScanContext) -> dict[str, Any]:
        axis = int(self.params.get("axis", ctx.pe_axis))
        default_g = 4 if ctx.coupled_trajectory else max(2, int(round(2 + 4 * self.severity)))
        num_ghosts = int(self.params.get("num_ghosts", default_g))
        intensity = float(self.params.get("intensity", 0.20 + 1.05 * self.severity))

        tfm = tio.RandomGhosting(
            num_ghosts=(num_ghosts, num_ghosts),
            axes=(axis,),
            intensity=(intensity * 0.9, intensity),
            p=1.0,
        )
        ctx.data = np.maximum(
            _run_torchio_transform(tfm, ctx.data, ctx.affine, seed=ctx.spawn_seed()),
            0.0,
        )
        return {
            "engine": "torchio.RandomGhosting",
            "axis": axis,
            "num_ghosts": num_ghosts,
            "intensity": round(intensity, 3),
        }


class Anisotropy(BaseArtifactTransform):
    """Thick-slice / anisotropic resolution degradation via TorchIO RandomAnisotropy."""

    name = "anisotropy"
    stage = PipelineStage.KSPACE
    supported_modalities = ("T1w", "T2w", "FLAIR", "bold", "dwi")
    qc_weight = 0.85

    def _apply_impl(self, ctx: ScanContext) -> dict[str, Any]:
        axis = int(self.params.get("axis", ctx.slice_axis))
        downsampling = float(self.params.get("downsampling", 1.5 + 3.8 * self.severity))
        max_safe = max(1.5, ctx.spatial_shape[axis] / 8.0)
        downsampling = float(min(downsampling, max_safe))

        tfm = tio.RandomAnisotropy(
            axes=(axis,),
            downsampling=(downsampling, downsampling),
            image_interpolation="linear",
            p=1.0,
        )
        ctx.data = np.maximum(
            _run_torchio_transform(tfm, ctx.data, ctx.affine, seed=ctx.spawn_seed()),
            0.0,
        )
        return {
            "engine": "torchio.RandomAnisotropy",
            "axis": axis,
            "downsampling": round(downsampling, 3),
        }
