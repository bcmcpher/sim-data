"""Curated site catalog loading and parametric multi-site profile generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from ghibli.config import SiteProfileConfig


def load_sites_catalog(path: str | Path) -> list[SiteProfileConfig]:
    """Load site profiles from a YAML catalog file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    items = raw.get("sites", raw if isinstance(raw, list) else [])
    return [SiteProfileConfig.model_validate(item) for item in items]


def generate_parametric_sites(
    n_sites: int = 4,
    effect_strength: float = 1.0,
    seed: int = 42,
) -> list[SiteProfileConfig]:
    """Generate S synthetic scanner/site profiles spanning distinct location, scale, and covariance regimes."""
    rng = np.random.default_rng(seed)
    coil_options = [4, 8, 16, 32, 64]
    profiles: list[SiteProfileConfig] = []

    for idx in range(n_sites):
        gamma = float(np.clip(1.0 + rng.uniform(-0.22, 0.24) * effect_strength, 0.65, 1.45))
        gain = float(np.clip(1.0 + rng.uniform(-0.22, 0.30) * effect_strength, 0.65, 1.55))
        offset = float(rng.uniform(-0.015, 0.025) * effect_strength)
        psf_xy = float(np.clip(rng.uniform(0.15, 0.80) * effect_strength, 0.0, 1.2))
        psf_z = float(np.clip(psf_xy * rng.uniform(1.0, 1.6), 0.0, 1.5))
        bias_c = tuple(float(v) for v in rng.uniform(-0.18, 0.18, size=3) * effect_strength)
        coils = int(coil_options[idx % len(coil_options)])
        noise_rel = float(np.clip(rng.uniform(0.008, 0.038) * effect_strength, 0.004, 0.06))
        g_fac = float(np.clip(rng.uniform(0.08, 0.45) * effect_strength, 0.0, 0.7))
        vscale = tuple(
            float(np.clip(1.0 + v * 0.012 * effect_strength, 0.98, 1.02))
            for v in rng.uniform(-1.0, 1.0, size=3)
        )
        profiles.append(
            SiteProfileConfig(
                site_id=f"Site_{idx + 1:02d}",
                description=f"Parametric simulated scanner site {idx + 1}",
                contrast_gamma=round(gamma, 3),
                intensity_gain=round(gain, 3),
                intensity_offset=round(offset, 4),
                psf_sigma_vox=(round(psf_xy, 3), round(psf_xy, 3), round(psf_z, 3)),
                bias_order=int(rng.choice([2, 3])),
                bias_coeffs=(round(bias_c[0], 3), round(bias_c[1], 3), round(bias_c[2], 3)),
                coil_count=coils,
                noise_sigma_rel=round(noise_rel, 4),
                g_factor_strength=round(g_fac, 3),
                voxel_scale=(round(vscale[0], 4), round(vscale[1], 4), round(vscale[2], 4)),
                jitter_scale=0.08,
            )
        )
    return profiles
