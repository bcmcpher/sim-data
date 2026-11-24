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
    # Scope torch/numpy RNG state deterministically without disturbing caller RNG
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
    """3D k-space intra-scan rigid head motion via TorchIO RandomMotion."""

    name = "motion_3d"
    stage = PipelineStage.KSPACE
    supported_modalities = ("T1w", "T2w", "FLAIR")
    qc_weight = 1.15

    def _apply_impl(self, ctx: ScanContext) -> dict[str, Any]:
        degrees = float(self.params.get("degrees", 1.5 + 11.5 * self.severity))
        translation = float(self.params.get("translation", 1.5 + 10.5 * self.severity))
        num_transforms = int(self.params.get("num_transforms", max(2, int(round(2 + 5 * self.severity)))))

        tfm = tio.RandomMotion(
            degrees=degrees,
            translation=translation,
            num_transforms=num_transforms,
            image_interpolation="linear",
            p=1.0,
        )
        ctx.data = np.maximum(
            _run_torchio_transform(tfm, ctx.data, ctx.affine, seed=ctx.spawn_seed()),
            0.0,
        )
        return {
            "engine": "torchio.RandomMotion",
            "degrees": round(degrees, 3),
            "translation_mm": round(translation, 3),
            "num_transforms": num_transforms,
        }


class Ghosting3D(BaseArtifactTransform):
    """Periodic k-space ghosting along phase-encoding axis via TorchIO RandomGhosting."""

    name = "ghosting_3d"
    stage = PipelineStage.KSPACE
    supported_modalities = ("T1w", "T2w", "FLAIR")
    qc_weight = 1.05

    def _apply_impl(self, ctx: ScanContext) -> dict[str, Any]:
        axis = int(self.params.get("axis", ctx.pe_axis))
        num_ghosts = int(self.params.get("num_ghosts", max(2, int(round(2 + 4 * self.severity)))))
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
        # Clamp to avoid collapsing tiny volumes to <4 slices
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
