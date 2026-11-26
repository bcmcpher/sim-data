"""MRI magnetic field physics artifacts: Susceptibility/Metal dipoles, EPI B0 distortion, DWI Eddy."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from ghibli.artifacts.base import BaseArtifactTransform, PipelineStage, ScanContext


class SusceptibilityMetal3D(BaseArtifactTransform):
    """3D magnetic dipole susceptibility / metal artifact (dental, sinus, surgical clip).

    Models localized off-resonance field dB0 ~ (3 cos^2(theta) - 1) / (r^3 + r0^3),
    producing central intravoxel dephasing signal void + bright pile-up rim + local readout warp.
    """

    name = "susceptibility_metal"
    stage = PipelineStage.SPATIAL_FIELD_4D
    supported_modalities = ("T1w", "T2w", "FLAIR")
    qc_weight = 1.1

    def _apply_impl(self, ctx: ScanContext) -> dict[str, Any]:
        shape = ctx.spatial_shape
        bmin = np.array(ctx.priors.bbox_min, dtype=np.float32)
        bmax = np.array(ctx.priors.bbox_max, dtype=np.float32)
        bspan = np.maximum(bmax - bmin, 4.0)

        # Place dipole near realistic anatomical landmarks (anterior-inferior dental/sinus or ear)
        landmark = str(
            self.params.get("landmark", str(ctx.rng.choice(["sinus_frontal", "dental_inferior", "ear_lateral"])))
        )
        if landmark == "sinus_frontal":
            center = np.array(
                [
                    bmin[0] + bspan[0] * float(ctx.rng.uniform(0.42, 0.58)),
                    bmin[1] + bspan[1] * float(ctx.rng.uniform(0.74, 0.90)),
                    bmin[2] + bspan[2] * float(ctx.rng.uniform(0.22, 0.38)),
                ],
                dtype=np.float32,
            )
        elif landmark == "dental_inferior":
            center = np.array(
                [
                    bmin[0] + bspan[0] * float(ctx.rng.uniform(0.35, 0.65)),
                    bmin[1] + bspan[1] * float(ctx.rng.uniform(0.70, 0.92)),
                    bmin[2] + bspan[2] * float(ctx.rng.uniform(0.08, 0.24)),
                ],
                dtype=np.float32,
            )
        else:
            side = float(ctx.rng.choice([0.14, 0.86]))
            center = np.array(
                [
                    bmin[0] + bspan[0] * side,
                    bmin[1] + bspan[1] * float(ctx.rng.uniform(0.38, 0.58)),
                    bmin[2] + bspan[2] * float(ctx.rng.uniform(0.20, 0.38)),
                ],
                dtype=np.float32,
            )

        radius = float(self.params.get("radius_vox", (0.06 + 0.14 * self.severity) * float(np.mean(bspan))))
        radius = max(2.5, radius)

        gx, gy, gz = np.meshgrid(
            np.arange(shape[0], dtype=np.float32),
            np.arange(shape[1], dtype=np.float32),
            np.arange(shape[2], dtype=np.float32),
            indexing="ij",
        )
        dx = gx - center[0]
        dy = gy - center[1]
        dz = gz - center[2]
        r2 = dx**2 + dy**2 + dz**2
        r = np.sqrt(r2 + 1e-4)
        cos_theta = dz / r
        dipole = (3.0 * (cos_theta**2) - 1.0) / (((r / radius) ** 3) + 0.25)

        # Readout-axis geometric warp proportional to dipole field
        warp_strength = float(self.params.get("warp_vox", 1.2 + 5.0 * self.severity))
        shift = np.clip(dipole * warp_strength * 0.45, -8.0, 8.0)

        coords = [gx, gy, gz]
        coords[ctx.readout_axis] = coords[ctx.readout_axis] + shift
        warped = ndimage.map_coordinates(ctx.data, coords, order=1, mode="nearest")

        # Central intravoxel dephasing void + characteristic bright pile-up rim
        void_mask = np.exp(-0.5 * (r / (0.62 * radius)) ** 2)
        attenuation = np.clip(1.0 - (0.55 + 0.45 * self.severity) * void_mask, 0.02, 1.0)
        rim_ring = np.exp(-0.5 * ((r - 0.92 * radius) / (0.28 * radius)) ** 2) * np.clip(dipole, 0.0, 2.5)
        pileup = 1.0 + (0.25 + 0.65 * self.severity) * np.clip(rim_ring, 0.0, 1.5)

        ctx.data = np.maximum((warped * attenuation * pileup).astype(np.float32), 0.0)
        return {
            "engine": "custom.SusceptibilityMetal3D",
            "landmark": landmark,
            "center_vox": [round(float(c), 2) for c in center],
            "radius_vox": round(radius, 2),
            "warp_vox": round(warp_strength, 2),
        }


class EPISusceptibilityDistortion(BaseArtifactTransform):
    """EPI B0 off-resonance geometric PE distortion + Jacobian pile-up + T2* signal dropout.

    Models air-tissue susceptibility near orbitofrontal cortex and inferior temporal ear canals.
    """

    name = "epi_susceptibility"
    stage = PipelineStage.SPATIAL_FIELD_4D
    supported_modalities = ("bold", "dwi")
    qc_weight = 1.05

    def _apply_impl(self, ctx: ScanContext) -> dict[str, Any]:
        shape = ctx.spatial_shape
        bmin = np.array(ctx.priors.bbox_min, dtype=np.float32)
        bmax = np.array(ctx.priors.bbox_max, dtype=np.float32)
        bspan = np.maximum(bmax - bmin, 4.0)

        gx, gy, gz = np.meshgrid(
            np.arange(shape[0], dtype=np.float32),
            np.arange(shape[1], dtype=np.float32),
            np.arange(shape[2], dtype=np.float32),
            indexing="ij",
        )

        # Air-tissue susceptibility hotspots: orbitofrontal (anterior-inferior) + bilateral temporal
        hotspots = [
            (
                bmin + bspan * np.array([0.50, 0.78, 0.25], dtype=np.float32),
                bspan * np.array([0.22, 0.18, 0.16], dtype=np.float32),
                1.0,
            ),
            (
                bmin + bspan * np.array([0.20, 0.48, 0.22], dtype=np.float32),
                bspan * np.array([0.14, 0.16, 0.14], dtype=np.float32),
                0.7,
            ),
            (
                bmin + bspan * np.array([0.80, 0.48, 0.22], dtype=np.float32),
                bspan * np.array([0.14, 0.16, 0.14], dtype=np.float32),
                0.7,
            ),
        ]

        b0_field = np.zeros(shape, dtype=np.float32)
        for center, sig, weight in hotspots:
            sig = np.maximum(sig, 1.5)
            gauss = np.exp(
                -0.5
                * (
                    ((gx - center[0]) / sig[0]) ** 2
                    + ((gy - center[1]) / sig[1]) ** 2
                    + ((gz - center[2]) / sig[2]) ** 2
                )
            )
            b0_field += weight * gauss

        max_shift = float(self.params.get("max_shift_vox", 1.0 + 4.8 * self.severity))
        pe_sign = float(self.params.get("pe_sign", float(ctx.rng.choice([-1.0, 1.0]))))
        shift_field = pe_sign * max_shift * b0_field

        # Jacobian modulation along phase-encoding axis: compression -> pile-up, stretch -> attenuation
        grad_pe = np.gradient(shift_field, axis=ctx.pe_axis)
        jacobian = np.clip(1.0 / np.maximum(1.0 + grad_pe, 0.35), 0.45, 2.1).astype(np.float32)

        # Through-plane T2* dephasing signal dropout proportional to |dB0/dz| and B0 field
        grad_z = np.abs(np.gradient(b0_field, axis=ctx.slice_axis))
        dropout_strength = float(self.params.get("dropout_strength", 0.25 + 0.68 * self.severity))
        dropout_mask = np.clip(
            1.0 - dropout_strength * np.clip(b0_field + 1.5 * grad_z, 0.0, 1.0),
            0.08,
            1.0,
        ).astype(np.float32)

        coords = [gx, gy, gz]
        coords[ctx.pe_axis] = coords[ctx.pe_axis] + shift_field

        if ctx.is_4d:
            out = np.empty_like(ctx.data)
            for t in range(ctx.data.shape[3]):
                warped = ndimage.map_coordinates(ctx.data[..., t], coords, order=1, mode="nearest")
                out[..., t] = warped * jacobian * dropout_mask
            ctx.data = np.maximum(out, 0.0)
        else:
            warped = ndimage.map_coordinates(ctx.data, coords, order=1, mode="nearest")
            ctx.data = np.maximum((warped * jacobian * dropout_mask).astype(np.float32), 0.0)

        return {
            "engine": "custom.EPISusceptibilityDistortion",
            "pe_axis": ctx.pe_axis,
            "pe_sign": pe_sign,
            "max_shift_vox": round(max_shift, 3),
            "dropout_strength": round(dropout_strength, 3),
        }


class EddyCurrentsDWI(BaseArtifactTransform):
    """Gradient-direction-correlated eddy current shear, scale, and translation for DWI."""

    name = "eddy_currents_dwi"
    stage = PipelineStage.SPATIAL_FIELD_4D
    supported_modalities = ("dwi",)
    qc_weight = 0.95

    def _apply_impl(self, ctx: ScanContext) -> dict[str, Any]:
        if not ctx.is_4d:
            return {"skipped": True, "reason": "not_4d"}

        n_vols = ctx.data.shape[3]
        bvals = ctx.bvals if (ctx.bvals is not None and len(ctx.bvals) == n_vols) else np.full(n_vols, 1000.0)
        bmax = max(float(np.max(bvals)), 1.0)
        if ctx.bvecs is not None and ctx.bvecs.shape == (3, n_vols):
            bvecs = ctx.bvecs
        else:
            bvecs = ctx.rng.normal(size=(3, n_vols)).astype(np.float32)
            norms = np.linalg.norm(bvecs, axis=0, keepdims=True) + 1e-6
            bvecs = bvecs / norms

        max_shear = float(self.params.get("max_shear", 0.015 + 0.075 * self.severity))
        max_scale = float(self.params.get("max_scale", 0.015 + 0.065 * self.severity))
        max_shift = float(self.params.get("max_shift_vox", 0.5 + 3.0 * self.severity))

        center = np.array(ctx.spatial_shape, dtype=np.float64) / 2.0
        pe = ctx.pe_axis
        ro = ctx.readout_axis
        sl = ctx.slice_axis

        out = ctx.data.copy()
        applied_vols = 0
        for v in range(n_vols):
            b_weight = float(np.sqrt(max(float(bvals[v]), 0.0) / bmax))
            if b_weight < 0.15:
                continue
            g = bvecs[:, v]
            mat = np.eye(3, dtype=np.float64)
            # Eddy currents distort primarily along PE axis as linear function of (x, y, z)
            mat[pe, ro] += max_shear * b_weight * float(g[0])
            mat[pe, pe] += max_scale * b_weight * float(g[1])
            mat[pe, sl] += max_shear * b_weight * float(g[2])

            shift = np.zeros(3, dtype=np.float64)
            shift[pe] = max_shift * b_weight * float(g[1])
            offset = center - mat @ center + shift
            out[..., v] = ndimage.affine_transform(
                ctx.data[..., v], matrix=mat, offset=offset, order=1, mode="nearest"
            )
            applied_vols += 1

        ctx.data = np.maximum(out.astype(np.float32), 0.0)
        return {
            "engine": "custom.EddyCurrentsDWI",
            "pe_axis": pe,
            "max_shear": round(max_shear, 4),
            "max_scale": round(max_scale, 4),
            "max_shift_vox": round(max_shift, 3),
            "affected_volumes": applied_vols,
        }
