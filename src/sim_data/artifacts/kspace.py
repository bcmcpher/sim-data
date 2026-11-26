"""Custom FFT / k-space MRI artifacts: Wrap-Around aliasing and EPI Nyquist N/2 ghosting."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from sim_data.artifacts.base import BaseArtifactTransform, PipelineStage, ScanContext


class WrapAround(BaseArtifactTransform):
    """Phase-encoding FOV fold-over (aliasing / wrap-around) artifact.

    Models an undersized phase-encoding FOV of width `fov_len < brain_span` by
    aliasing peripheral anatomy circularly onto the opposite side of the reduced FOV
    with smooth raised-cosine edge falloff (no artificial interior step seam).
    """

    name = "wrap_around"
    stage = PipelineStage.SPATIAL_FIELD_4D
    supported_modalities = ("T1w", "T2w", "FLAIR", "bold", "dwi")
    qc_weight = 1.05

    def _apply_impl(self, ctx: ScanContext) -> dict[str, Any]:
        axis = int(self.params.get("axis", ctx.pe_axis))
        dim_len = ctx.spatial_shape[axis]

        bmin, bmax = ctx.priors.bbox_min[axis], ctx.priors.bbox_max[axis]
        brain_span = max(8, bmax - bmin + 1)
        wrap_frac = float(self.params.get("wrap_fraction", 0.08 + 0.24 * self.severity))
        shift_vox = max(2, min(brain_span // 2, int(round(brain_span * wrap_frac))))
        direction = int(self.params.get("direction", int(ctx.rng.choice([-1, 1]))))
        blend_gain = float(self.params.get("blend_gain", 0.50 + 0.45 * self.severity))

        # Reduced PE FOV length: peripheral overhang of width `shift_vox` aliases modulo `fov_len`
        fov_len = max(8, brain_span - shift_vox)
        arr = ctx.data
        moved = np.moveaxis(arr, axis, 0)
        out = moved.copy()

        # Smooth cosine taper across the wrapped overhang (avoids sharp interior step seam)
        taper = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, shift_vox, dtype=np.float32)))
        taper = (0.35 + 0.65 * taper).reshape((shift_vox,) + (1,) * (moved.ndim - 1))

        if direction > 0:
            # High-index peripheral anatomy folds circularly onto low-index side of reduced FOV
            src_start = max(0, bmax - shift_vox + 1)
            src_end = min(dim_len, src_start + shift_vox)
            slab = moved[src_start:src_end].copy()
            copy_len = slab.shape[0]
            dst_start = max(0, src_start - fov_len)
            dst_end = min(dim_len, dst_start + copy_len)
            eff_len = dst_end - dst_start
            if eff_len > 0:
                t_eff = taper[:eff_len]
                out[dst_start:dst_end] = out[dst_start:dst_end] + blend_gain * slab[:eff_len] * t_eff
                # Smoothly crop/zero the out-of-FOV peripheral overhang beyond the reduced FOV edge
                edge_keep = (1.0 - t_eff * float(np.clip(0.35 + 0.65 * self.severity, 0.0, 1.0)))
                out[src_start : src_start + eff_len] = out[src_start : src_start + eff_len] * edge_keep
        else:
            # Low-index peripheral anatomy folds circularly onto high-index side of reduced FOV
            src_start = max(0, bmin)
            src_end = min(dim_len, src_start + shift_vox)
            slab = moved[src_start:src_end].copy()
            copy_len = slab.shape[0]
            dst_start = min(dim_len - 1, src_start + fov_len)
            dst_end = min(dim_len, dst_start + copy_len)
            eff_len = dst_end - dst_start
            if eff_len > 0:
                t_eff = taper[-eff_len:][::-1]
                out[dst_start:dst_end] = out[dst_start:dst_end] + blend_gain * slab[:eff_len] * t_eff
                edge_keep = (1.0 - t_eff * float(np.clip(0.35 + 0.65 * self.severity, 0.0, 1.0)))
                out[src_start : src_start + eff_len] = out[src_start : src_start + eff_len] * edge_keep

        ctx.data = np.ascontiguousarray(np.moveaxis(out, 0, axis))
        return {
            "engine": "custom.WrapAround",
            "axis": axis,
            "direction": direction,
            "wrap_fraction": round(wrap_frac, 4),
            "shift_vox": shift_vox,
            "fov_len": fov_len,
            "blend_gain": round(blend_gain, 3),
        }


class NyquistN2Ghost(BaseArtifactTransform):
    """EPI Nyquist N/2 ghosting via odd/even k-space PE-line phase and amplitude mismatch."""

    name = "nyquist_n2_ghost"
    stage = PipelineStage.KSPACE
    supported_modalities = ("bold", "dwi")
    qc_weight = 1.1

    def _apply_impl(self, ctx: ScanContext) -> dict[str, Any]:
        pe_axis = int(self.params.get("pe_axis", ctx.pe_axis))
        ro_axis = int(self.params.get("readout_axis", ctx.readout_axis))

        # Phase offset between even/odd k_pe lines produces an exact N/2-shifted ghost
        phase_err = float(self.params.get("phase_error_rad", 0.18 + 1.15 * self.severity))
        amp_err = float(self.params.get("amp_error", 0.04 + 0.22 * self.severity))

        device = ctx.torch_device
        t = torch.from_numpy(np.ascontiguousarray(ctx.data, dtype=np.float32)).to(device)
        kspace = torch.fft.fft2(t, dim=(ro_axis, pe_axis))

        # Build phase/amplitude modulation vector along PE axis (alternating odd/even lines)
        n_pe = ctx.spatial_shape[pe_axis]
        mod = torch.ones(n_pe, dtype=torch.complex64, device=device)
        odd_factor = (1.0 - amp_err) * np.exp(1j * phase_err)
        mod[1::2] = torch.complex(
            torch.tensor(float(odd_factor.real), dtype=torch.float32, device=device),
            torch.tensor(float(odd_factor.imag), dtype=torch.float32, device=device),
        )

        shape_bc = [1] * t.ndim
        shape_bc[pe_axis] = n_pe
        kspace_mod = kspace * mod.view(*shape_bc)
        recon = torch.abs(torch.fft.ifft2(kspace_mod, dim=(ro_axis, pe_axis)))

        ctx.data = np.ascontiguousarray(recon.detach().cpu().numpy().astype(np.float32))
        return {
            "engine": "custom.NyquistN2Ghost",
            "pe_axis": pe_axis,
            "readout_axis": ro_axis,
            "phase_error_rad": round(phase_err, 4),
            "amp_error": round(amp_err, 4),
        }
