"""MONAI-backed severity-calibrated MRI artifact transforms."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from monai.transforms import GibbsNoise, KSpaceSpikeNoise, RandBiasField, RandRicianNoise

from sim_data.artifacts.base import BaseArtifactTransform, PipelineStage, ScanContext


def _to_channel_first_tensor(data: np.ndarray, device: torch.device) -> tuple[torch.Tensor, bool]:
    """Convert (X,Y,Z) or (X,Y,Z,T) numpy array to (C,X,Y,Z) torch tensor on device."""
    if data.ndim == 3:
        t = torch.from_numpy(np.ascontiguousarray(data[None, ...], dtype=np.float32)).to(device)
        return t, False
    t = torch.from_numpy(
        np.ascontiguousarray(np.moveaxis(data, -1, 0), dtype=np.float32)
    ).to(device)
    return t, True


def _from_channel_first_tensor(tensor: torch.Tensor | np.ndarray, was_4d: bool) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        arr = np.asarray(tensor, dtype=np.float32)
    if was_4d:
        return np.ascontiguousarray(np.moveaxis(arr, 0, -1))
    return np.ascontiguousarray(arr[0])


class GibbsRinging(BaseArtifactTransform):
    """K-space truncation Gibbs ringing via MONAI GibbsNoise."""

    name = "gibbs_ringing"
    stage = PipelineStage.KSPACE
    supported_modalities = ("T1w", "T2w", "FLAIR", "bold", "dwi")
    qc_weight = 0.85

    def _apply_impl(self, ctx: ScanContext) -> dict[str, Any]:
        # MONAI GibbsNoise(alpha): alpha in [0, 1] is fraction of outer k-space removed
        alpha = float(self.params.get("alpha", 0.25 + 0.45 * self.severity))
        alpha = float(np.clip(alpha, 0.05, 0.80))

        tensor, was_4d = _to_channel_first_tensor(ctx.data, ctx.torch_device)
        transform = GibbsNoise(alpha=alpha)
        out = transform(tensor)
        ctx.data = np.maximum(_from_channel_first_tensor(out, was_4d), 0.0)
        return {"engine": "monai.GibbsNoise", "alpha": round(alpha, 4)}


class RFSpike(BaseArtifactTransform):
    """K-space RF spike (herringbone striping) via MONAI KSpaceSpikeNoise.

    For 3D structural scans, injects 1-3 controlled off-center spikes.
    For 4D scans (BOLD/DWI), injects transient spikes into a random subset of volumes.
    """

    name = "rf_spike"
    stage = PipelineStage.KSPACE
    supported_modalities = ("T1w", "T2w", "FLAIR", "bold", "dwi")
    qc_weight = 1.05

    def _sample_spike_loc(self, shape: tuple[int, int, int], rng: np.random.Generator) -> tuple[int, int, int]:
        center = np.array(shape, dtype=np.float64) / 2.0
        # Off-center radial band: 8% to 32% of half-extent produces clear herringbone fringes
        frac = rng.uniform(0.08, 0.32, size=3) * rng.choice([-1.0, 1.0], size=3)
        # Ensure at least two axes have meaningful offset so striping is visible in all planes
        loc = np.round(center + frac * center).astype(int)
        loc = np.clip(loc, 2, np.array(shape) - 3)
        if np.all(np.abs(loc - center) < 2):
            loc[0] = int(np.clip(center[0] + max(3, int(shape[0] * 0.15)), 2, shape[0] - 3))
        return (int(loc[0]), int(loc[1]), int(loc[2]))

    def _apply_impl(self, ctx: ScanContext) -> dict[str, Any]:
        n_spikes = int(self.params.get("num_spikes", max(1, int(round(1 + 2 * self.severity)))))
        # MONAI KSpaceSpikeNoise uses natural log (ln) of k-space magnitude
        shape = ctx.spatial_shape
        ref_vol = ctx.data[..., 0] if ctx.is_4d else ctx.data
        fft_ref = np.abs(np.fft.fftn(ref_vol))
        log_max = float(np.log(max(float(fft_ref.max()), 1.0)))
        k_intensity = float(
            self.params.get("k_intensity", log_max - 2.6 + 1.8 * self.severity)
        )

        locs: list[tuple[int, int, int]] = []
        affected_volumes: list[int] = []

        if not ctx.is_4d:
            tensor, _ = _to_channel_first_tensor(ctx.data, ctx.torch_device)
            locs = [self._sample_spike_loc(shape, ctx.rng) for _ in range(n_spikes)]
            loc_tuples = [(0, loc[0], loc[1], loc[2]) for loc in locs]
            tfm = KSpaceSpikeNoise(loc=loc_tuples, k_intensity=[k_intensity] * len(loc_tuples))
            out_spike = tfm(tensor)
            ctx.data = np.maximum(_from_channel_first_tensor(out_spike, False), 0.0)
        else:
            n_vols = ctx.data.shape[3]
            frac_vols = float(self.params.get("volume_fraction", 0.12 + 0.45 * self.severity))
            n_aff = max(1, int(round(n_vols * frac_vols)))
            vol_indices = sorted(
                int(i) for i in ctx.rng.choice(n_vols, size=min(n_aff, n_vols), replace=False)
            )
            affected_volumes = vol_indices
            out = ctx.data.copy()
            for v in vol_indices:
                vol_t = torch.from_numpy(np.ascontiguousarray(out[None, ..., v])).to(ctx.torch_device)
                v_locs = [self._sample_spike_loc(shape, ctx.rng) for _ in range(n_spikes)]
                if len(locs) < 6:
                    locs.extend(v_locs)
                loc_tuples = [(0, loc[0], loc[1], loc[2]) for loc in v_locs]
                tfm = KSpaceSpikeNoise(loc=loc_tuples, k_intensity=[k_intensity] * len(loc_tuples))
                vol_out = tfm(vol_t)
                out[..., v] = np.maximum(_from_channel_first_tensor(vol_out, False), 0.0)
            ctx.data = out

        return {
            "engine": "monai.KSpaceSpikeNoise",
            "num_spikes": n_spikes,
            "k_intensity_ln": round(k_intensity, 3),
            "sample_locs": [list(x) for x in locs[:4]],
            "affected_volumes_count": len(affected_volumes) if ctx.is_4d else 1,
        }


class BiasField(BaseArtifactTransform):
    """Smooth 3D multiplicative B1 bias field via MONAI RandBiasField."""

    name = "bias_field"
    stage = PipelineStage.BIAS_FIELD
    supported_modalities = ("T1w", "T2w", "FLAIR", "bold", "dwi")
    qc_weight = 0.85

    def _apply_impl(self, ctx: ScanContext) -> dict[str, Any]:
        degree = int(self.params.get("degree", 3))
        # Symmetric polynomial coefficients avoid all-positive exponential blowup in MONAI RandBiasField
        coeff = float(self.params.get("coeff_max", 0.14 + 0.34 * self.severity))
        seed = ctx.spawn_seed()

        # Generate multiplicative field on a unit 3D volume so 4D shares an identical static B1 field
        ones = torch.ones((1, *ctx.spatial_shape), dtype=torch.float32, device=ctx.torch_device)
        tfm = RandBiasField(degree=degree, coeff_range=(-coeff, coeff), prob=1.0)
        tfm.set_random_state(seed=seed)
        field_3d = tfm(ones)[0].detach().cpu().numpy().astype(np.float32)

        # Ensure meaningful shading magnitude (rescaling log-field if random draw was near zero)
        brain_log = np.log(np.clip(field_3d[ctx.priors.brain_mask], 1e-4, 1e4))
        cur_span = float(np.percentile(brain_log, 95) - np.percentile(brain_log, 5))
        target_min_span = 0.20 + 0.85 * self.severity
        if cur_span > 1e-4 and cur_span < target_min_span:
            scale = min(4.0, target_min_span / cur_span)
            log_full = np.log(np.clip(field_3d, 1e-4, 1e4))
            log_full = (log_full - float(np.median(brain_log))) * scale
            field_3d = np.exp(np.clip(log_full, -1.6, 1.6)).astype(np.float32)
        else:
            field_3d = np.clip(field_3d, 0.15, 6.0).astype(np.float32)

        # Normalize median inside brain to ~1.0 so global scale isn't trivially collapsed
        brain_med = float(np.median(field_3d[ctx.priors.brain_mask]))
        if brain_med > 1e-4:
            field_3d = field_3d / brain_med

        if ctx.is_4d:
            ctx.data = ctx.data * field_3d[..., None]
        else:
            ctx.data = ctx.data * field_3d

        brain_field = field_3d[ctx.priors.brain_mask]
        field_range = float(np.percentile(brain_field, 98) - np.percentile(brain_field, 2))
        return {
            "engine": "monai.RandBiasField",
            "degree": degree,
            "coeff_max": round(coeff, 4),
            "realized_brain_field_range": round(field_range, 4),
        }


class RicianNoiseArtifact(BaseArtifactTransform):
    """Magnitude MRI Rician (or multi-coil Non-Central Chi) noise degrading SNR."""

    name = "rician_noise"
    stage = PipelineStage.NOISE_DRIFT
    supported_modalities = ("T1w", "T2w", "FLAIR", "bold", "dwi")
    qc_weight = 1.0

    def _apply_impl(self, ctx: ScanContext) -> dict[str, Any]:
        from sim_data.artifacts.noise import apply_noncentral_chi_noise

        coil_count = int(self.params.get("coil_count", 1))
        g_factor_strength = float(self.params.get("g_factor_strength", 0.0))
        # Relative sigma calibrated to clean foreground median
        sigma_rel = float(self.params.get("sigma_rel", 0.018 + 0.22 * (self.severity**1.2)))
        sigma_abs = sigma_rel * ctx.priors.fg_median

        if coil_count == 1 and g_factor_strength <= 0.0:
            tensor, was_4d = _to_channel_first_tensor(ctx.data, ctx.torch_device)
            tfm = RandRicianNoise(prob=1.0, mean=0.0, std=sigma_abs, sample_std=False)
            tfm.set_random_state(seed=ctx.spawn_seed())
            out = tfm(tensor)
            ctx.data = np.maximum(_from_channel_first_tensor(out, was_4d), 0.0)
            engine = "monai.RandRicianNoise"
        else:
            ctx.data = apply_noncentral_chi_noise(
                ctx.data,
                sigma=sigma_abs,
                coil_count=coil_count,
                g_factor_strength=g_factor_strength,
                seed=ctx.spawn_seed(),
                device=ctx.torch_device,
            )
            engine = "custom.NonCentralChiNoise"

        return {
            "engine": engine,
            "sigma_rel": round(sigma_rel, 5),
            "sigma_abs": round(sigma_abs, 5),
            "coil_count": coil_count,
            "g_factor_strength": round(g_factor_strength, 3),
        }
