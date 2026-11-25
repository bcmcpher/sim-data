"""MONAI-backed severity-calibrated MRI artifact transforms with CRN coupling & real-scan noise calibration."""

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
        alpha = float(self.params.get("alpha", 0.25 + 0.45 * self.severity))
        alpha = float(np.clip(alpha, 0.05, 0.80))

        tensor, was_4d = _to_channel_first_tensor(ctx.data, ctx.torch_device)
        transform = GibbsNoise(alpha=alpha)
        out = transform(tensor)
        ctx.data = np.maximum(_from_channel_first_tensor(out, was_4d), 0.0)
        return {"engine": "monai.GibbsNoise", "alpha": round(alpha, 4)}


class RFSpike(BaseArtifactTransform):
    """K-space RF spike (herringbone striping) via MONAI KSpaceSpikeNoise.

    Pre-samples spike-location banks and 4D volume permutations once so coupled
    trajectories produce strictly nested spikes/volumes with monotonically increasing amplitude.
    """

    name = "rf_spike"
    stage = PipelineStage.KSPACE
    supported_modalities = ("T1w", "T2w", "FLAIR", "bold", "dwi")
    qc_weight = 1.05

    def _sample_spike_loc(
        self, shape: tuple[int, int, int], rng: np.random.Generator
    ) -> tuple[int, int, int]:
        center = np.array(shape, dtype=np.float64) / 2.0
        frac = rng.uniform(0.08, 0.32, size=3) * rng.choice([-1.0, 1.0], size=3)
        loc = np.round(center + frac * center).astype(int)
        loc = np.clip(loc, 2, np.array(shape) - 3)
        if np.all(np.abs(loc - center) < 2):
            loc[0] = int(np.clip(center[0] + max(3, int(shape[0] * 0.15)), 2, shape[0] - 3))
        return (int(loc[0]), int(loc[1]), int(loc[2]))

    def _apply_impl(self, ctx: ScanContext) -> dict[str, Any]:
        n_spikes = int(self.params.get("num_spikes", max(1, int(round(1 + 2 * self.severity)))))
        max_bank = max(4, n_spikes)
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
            bank = [self._sample_spike_loc(shape, ctx.rng) for _ in range(max_bank)]
            locs = bank[:n_spikes]
            tensor, _ = _to_channel_first_tensor(ctx.data, ctx.torch_device)
            loc_tuples = [(0, loc[0], loc[1], loc[2]) for loc in locs]
            tfm = KSpaceSpikeNoise(loc=loc_tuples, k_intensity=[k_intensity] * len(loc_tuples))
            out_spike = tfm(tensor)
            ctx.data = np.maximum(_from_channel_first_tensor(out_spike, False), 0.0)
        else:
            n_vols = ctx.data.shape[3]
            vol_perm = ctx.rng.permutation(n_vols)
            vol_banks = [
                [self._sample_spike_loc(shape, ctx.rng) for _ in range(max_bank)]
                for _ in range(n_vols)
            ]
            frac_vols = float(self.params.get("volume_fraction", 0.12 + 0.45 * self.severity))
            n_aff = max(1, min(n_vols, int(round(n_vols * frac_vols))))
            vol_indices = sorted(int(i) for i in vol_perm[:n_aff])
            affected_volumes = vol_indices
            out = ctx.data.copy()
            for v in vol_indices:
                vol_t = torch.from_numpy(np.ascontiguousarray(out[None, ..., v])).to(
                    ctx.torch_device
                )
                v_locs = vol_banks[v][:n_spikes]
                if len(locs) < 6:
                    locs.extend(v_locs)
                loc_tuples = [(0, loc[0], loc[1], loc[2]) for loc in v_locs]
                tfm = KSpaceSpikeNoise(loc=loc_tuples, k_intensity=[k_intensity] * len(loc_tuples))
                vol_out = tfm(vol_t)
                out[..., v] = np.maximum(_from_channel_first_tensor(vol_out, False), 0.0)
            ctx.data = out

        ctx.metadata["gt_rf_spike_locs"] = locs
        ctx.metadata["gt_rf_spike_volumes"] = affected_volumes
        return {
            "engine": "monai.KSpaceSpikeNoise",
            "num_spikes": n_spikes,
            "k_intensity_ln": round(k_intensity, 3),
            "sample_locs": [list(x) for x in locs[:4]],
            "affected_volumes_count": len(affected_volumes) if ctx.is_4d else 1,
        }


class BiasField(BaseArtifactTransform):
    """Smooth 3D multiplicative B1 bias field via MONAI RandBiasField.

    Draws a unit-normalized 3D log-polynomial basis from the trajectory seed and scales its
    amplitude monotonically with `severity` so spatial shading orientation stays locked across
    coupled dose steps.
    """

    name = "bias_field"
    stage = PipelineStage.BIAS_FIELD
    supported_modalities = ("T1w", "T2w", "FLAIR", "bold", "dwi")
    qc_weight = 0.85

    def _apply_impl(self, ctx: ScanContext) -> dict[str, Any]:
        degree = int(self.params.get("degree", 3))
        seed = ctx.spawn_seed()

        # Draw fixed-range polynomial basis independent of severity, then scale log-span monotonically
        ones = torch.ones((1, *ctx.spatial_shape), dtype=torch.float32, device=ctx.torch_device)
        tfm = RandBiasField(degree=degree, coeff_range=(-0.5, 0.5), prob=1.0)
        tfm.set_random_state(seed=seed)
        raw_field = tfm(ones)[0].detach().cpu().numpy().astype(np.float32)

        log_full = np.log(np.clip(raw_field, 1e-4, 1e4)).astype(np.float64)
        brain_log = log_full[ctx.priors.brain_mask]
        med_log = float(np.median(brain_log))
        cur_span = float(np.percentile(brain_log, 95.0) - np.percentile(brain_log, 5.0))

        target_span = float(self.params.get("target_log_span", 0.20 + 0.90 * self.severity))
        if cur_span > 1e-4:
            unit_log = (log_full - med_log) / cur_span
            field_3d = np.exp(np.clip(unit_log * target_span, -1.6, 1.6)).astype(np.float32)
        else:
            field_3d = np.clip(raw_field, 0.15, 6.0).astype(np.float32)

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
            "target_log_span": round(target_span, 4),
            "realized_brain_field_range": round(field_range, 4),
        }


class RicianNoiseArtifact(BaseArtifactTransform):
    """Magnitude MRI Rician (or multi-coil Non-Central Chi) noise degrading SNR.

    Supports `noise_calibration="excess_variance"` (default) to subtract pre-existing
    native scanner noise sigma_0 in quadrature on real QC-passing scans, or `"additive"`.
    """

    name = "rician_noise"
    stage = PipelineStage.NOISE_DRIFT
    supported_modalities = ("T1w", "T2w", "FLAIR", "bold", "dwi")
    qc_weight = 1.0

    def _apply_impl(self, ctx: ScanContext) -> dict[str, Any]:
        from sim_data.artifacts.noise import apply_noncentral_chi_noise

        coil_count = int(self.params.get("coil_count", 1))
        g_factor_strength = float(self.params.get("g_factor_strength", 0.0))
        sigma_rel = float(self.params.get("sigma_rel", 0.018 + 0.22 * (self.severity**1.2)))
        sigma_target = sigma_rel * ctx.priors.fg_median

        if ctx.noise_calibration == "excess_variance" and ctx.baseline_noise_sigma > 0.0:
            excess_sq = sigma_target**2 - ctx.baseline_noise_sigma**2
            if excess_sq > 0.0:
                sigma_applied = float(np.sqrt(excess_sq))
            else:
                sigma_applied = float(0.15 * sigma_target)
        else:
            sigma_applied = float(sigma_target)

        if coil_count == 1 and g_factor_strength <= 0.0:
            tensor, was_4d = _to_channel_first_tensor(ctx.data, ctx.torch_device)
            tfm = RandRicianNoise(prob=1.0, mean=0.0, std=sigma_applied, sample_std=False)
            tfm.set_random_state(seed=ctx.spawn_seed())
            out = tfm(tensor)
            ctx.data = np.maximum(_from_channel_first_tensor(out, was_4d), 0.0)
            engine = "monai.RandRicianNoise"
        else:
            ctx.data = apply_noncentral_chi_noise(
                ctx.data,
                sigma=sigma_applied,
                coil_count=coil_count,
                g_factor_strength=g_factor_strength,
                seed=ctx.spawn_seed(),
                device=ctx.torch_device,
            )
            engine = "custom.NonCentralChiNoise"

        return {
            "engine": engine,
            "sigma_rel": round(sigma_rel, 5),
            "sigma_target_abs": round(sigma_target, 5),
            "sigma_abs": round(sigma_applied, 5),
            "baseline_noise_sigma": round(ctx.baseline_noise_sigma, 5),
            "noise_calibration": ctx.noise_calibration,
            "coil_count": coil_count,
            "g_factor_strength": round(g_factor_strength, 3),
        }
