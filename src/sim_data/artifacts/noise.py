"""Multi-coil Non-Central Chi noise and 4D BOLD temporal scanner drift."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from sim_data.artifacts.base import BaseArtifactTransform, PipelineStage, ScanContext


def build_g_factor_map(
    spatial_shape: tuple[int, int, int],
    strength: float,
) -> np.ndarray:
    """Create a smooth spatially varying parallel-imaging g-factor noise amplification map."""
    if strength <= 0.0:
        return np.ones(spatial_shape, dtype=np.float32)
    coords = [np.linspace(-1.0, 1.0, n, dtype=np.float32) for n in spatial_shape]
    gx, gy, gz = np.meshgrid(*coords, indexing="ij")
    # Parallel imaging (GRAPPA/SENSE) typically elevates noise near center/phase-encode overlap
    r2 = 0.55 * gx**2 + 0.85 * gy**2 + 0.35 * gz**2
    g_map = 1.0 + strength * np.exp(-1.8 * r2)
    return g_map.astype(np.float32)


def apply_noncentral_chi_noise(
    data: np.ndarray,
    sigma: float,
    coil_count: int = 1,
    g_factor_strength: float = 0.0,
    seed: int = 0,
    device: torch.device | None = None,
) -> np.ndarray:
    """Apply magnitude Rician (L=1) or multi-coil Sum-of-Squares Non-Central Chi (L>1) noise.

    For L effective coils with complex Gaussian noise N(0, sigma_c^2) per coil,
    we normalize sigma_c = sigma / sqrt(L) so total signal energy matches while
    producing the realistic positive non-central chi background floor.
    """
    if sigma <= 0.0:
        return data.astype(np.float32, copy=True)

    dev = device or torch.device("cpu")
    coil_count = max(1, int(coil_count))
    # Cap explicit coil loop at 16 effective degrees (for L>16, chi distribution converges rapidly)
    eff_coils = min(coil_count, 16)
    sigma_per_coil = float(sigma / np.sqrt(eff_coils))

    g_map_np = build_g_factor_map(data.shape[:3], g_factor_strength)
    g_map = torch.from_numpy(g_map_np).to(dev)
    if data.ndim == 4:
        g_map = g_map.unsqueeze(-1)

    sig = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32)).to(dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))

    # Signal split across coils in magnitude sum-of-squares reconstruction
    sig_per_coil = sig / np.sqrt(float(eff_coils))
    sq_sum = torch.zeros_like(sig)
    for _ in range(eff_coils):
        n_re = torch.randn(sig.shape, generator=gen, device=dev, dtype=torch.float32) * (
            sigma_per_coil * g_map
        )
        n_im = torch.randn(sig.shape, generator=gen, device=dev, dtype=torch.float32) * (
            sigma_per_coil * g_map
        )
        sq_sum = sq_sum + (sig_per_coil + n_re) ** 2 + n_im**2

    noisy = torch.sqrt(torch.clamp(sq_sum, min=0.0))
    return np.ascontiguousarray(noisy.detach().cpu().numpy().astype(np.float32))


class TemporalDrift(BaseArtifactTransform):
    """Low-frequency scanner baseline drift and physiological oscillation for 4D BOLD."""

    name = "temporal_drift"
    stage = PipelineStage.NOISE_DRIFT
    supported_modalities = ("bold",)
    qc_weight = 0.75

    def _apply_impl(self, ctx: ScanContext) -> dict[str, Any]:
        if not ctx.is_4d:
            return {"skipped": True, "reason": "not_4d"}
        n_t = ctx.data.shape[3]
        t = np.linspace(-1.0, 1.0, n_t, dtype=np.float32)

        linear_amp = float(self.params.get("linear_amp", (0.02 + 0.14 * self.severity) * float(ctx.rng.choice([-1.0, 1.0]))))
        quad_amp = float(self.params.get("quad_amp", (0.01 + 0.06 * self.severity) * float(ctx.rng.choice([-1.0, 1.0]))))
        sin_amp = float(self.params.get("sin_amp", 0.01 + 0.05 * self.severity))
        sin_cycles = float(self.params.get("sin_cycles", float(ctx.rng.uniform(0.7, 2.5))))
        phase = float(ctx.rng.uniform(0.0, 2.0 * np.pi))

        curve = 1.0 + linear_amp * t + quad_amp * (t**2 - 0.33) + sin_amp * np.sin(
            2.0 * np.pi * sin_cycles * (t + 1.0) / 2.0 + phase
        )
        curve = np.clip(curve, 0.5, 1.8).astype(np.float32)
        ctx.data = ctx.data * curve[None, None, None, :]

        return {
            "engine": "custom.TemporalDrift",
            "linear_amp": round(linear_amp, 4),
            "quad_amp": round(quad_amp, 4),
            "sin_amp": round(sin_amp, 4),
            "sin_cycles": round(sin_cycles, 3),
        }
