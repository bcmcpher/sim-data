"""Experimental recipe samplers: dose-response, combinatorial mixtures, and multi-site factorial."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sim_data.artifacts.registry import ARTIFACT_REGISTRY, compatible_artifacts
from sim_data.config import ArtifactSpec, SimulationConfig, SiteProfileConfig
from sim_data.io.bids import BIDSScan


@dataclass
class PlannedArtifact:
    name: str
    severity: float
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanRecipe:
    """Complete deterministic specification for simulating one scan."""

    recipe_id: str
    scan: BIDSScan
    site: SiteProfileConfig
    artifacts: list[PlannedArtifact]
    seed: int
    rotate_dwi_bvecs: bool = False


def derive_seed(global_seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{global_seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:4], "little") % (2**31 - 1)


def _eligible_specs(config: SimulationConfig, scan: BIDSScan) -> list[ArtifactSpec]:
    valid_names = set(compatible_artifacts(scan.modality))
    if config.sampling.artifacts:
        specs = [spec for spec in config.sampling.artifacts if spec.name in valid_names]
        if specs:
            return specs
    return [ArtifactSpec(name=name) for name in sorted(valid_names)]


def _default_site() -> SiteProfileConfig:
    return SiteProfileConfig(site_id="DefaultSite")


def _sample_combination(
    specs: list[ArtifactSpec],
    rng: np.random.Generator,
    min_k: int,
    max_k: int,
    default_range: tuple[float, float],
) -> list[PlannedArtifact]:
    if not specs or max_k <= 0:
        return []
    lo_k = max(0, min(min_k, len(specs)))
    hi_k = max(lo_k, min(max_k, len(specs)))
    k = int(rng.integers(lo_k, hi_k + 1))
    if k == 0:
        return []

    weights = np.array([s.weight for s in specs], dtype=np.float64)
    probs = weights / weights.sum()
    chosen_idx = rng.choice(len(specs), size=k, replace=False, p=probs)

    planned: list[PlannedArtifact] = []
    for idx in chosen_idx:
        spec = specs[int(idx)]
        lo, hi = spec.severity_range or default_range
        sev = float(rng.uniform(lo, hi))
        planned.append(PlannedArtifact(name=spec.name, severity=round(sev, 4), params=dict(spec.params)))
    return planned


def build_recipes(
    scans: list[BIDSScan],
    config: SimulationConfig,
) -> list[ScanRecipe]:
    """Expand BIDS scans + SimulationConfig into deterministic ScanRecipes."""
    sites = config.sites if config.sites else [_default_site()]
    sampling = config.sampling
    recipes: list[ScanRecipe] = []

    for scan in scans:
        specs = _eligible_specs(config, scan)
        if sampling.mode == "dose_response":
            site = sites[0]
            if sampling.include_clean_baseline:
                rid = "clean000"
                recipes.append(
                    ScanRecipe(
                        recipe_id=rid,
                        scan=scan,
                        site=site,
                        artifacts=[],
                        seed=derive_seed(config.global_seed, f"{scan.base_id}:{site.site_id}:{rid}"),
                        rotate_dwi_bvecs=sampling.rotate_dwi_bvecs,
                    )
                )
            for spec in specs:
                if spec.name not in ARTIFACT_REGISTRY:
                    continue
                for sev_idx, sev in enumerate(sampling.dose_severities):
                    if sev <= 0.0 and sampling.include_clean_baseline:
                        continue
                    rid = f"dose{spec.name[:6]}{sev_idx:02d}"
                    arts = (
                        [PlannedArtifact(name=spec.name, severity=float(sev), params=dict(spec.params))]
                        if sev > 0.0
                        else []
                    )
                    recipes.append(
                        ScanRecipe(
                            recipe_id=rid,
                            scan=scan,
                            site=site,
                            artifacts=arts,
                            seed=derive_seed(config.global_seed, f"{scan.base_id}:{site.site_id}:{rid}"),
                            rotate_dwi_bvecs=sampling.rotate_dwi_bvecs,
                        )
                    )

        elif sampling.mode == "combinatorial":
            for sample_idx in range(sampling.samples_per_scan):
                site = sites[sample_idx % len(sites)]
                rid = f"comb{sample_idx:03d}"
                seed = derive_seed(config.global_seed, f"{scan.base_id}:{site.site_id}:{rid}")
                rng = np.random.default_rng(seed)
                if sample_idx == 0 and sampling.include_clean_baseline:
                    comb_arts: list[PlannedArtifact] = []
                else:
                    comb_arts = _sample_combination(
                        specs,
                        rng=rng,
                        min_k=sampling.min_artifacts,
                        max_k=sampling.max_artifacts,
                        default_range=sampling.severity_range,
                    )
                recipes.append(
                    ScanRecipe(
                        recipe_id=rid,
                        scan=scan,
                        site=site,
                        artifacts=comb_arts,
                        seed=seed,
                        rotate_dwi_bvecs=sampling.rotate_dwi_bvecs,
                    )
                )

        elif sampling.mode == "multisite_factorial":
            for site in sites:
                if sampling.include_clean_baseline:
                    rid = "clean000"
                    seed = derive_seed(config.global_seed, f"{scan.base_id}:{site.site_id}:{rid}")
                    recipes.append(
                        ScanRecipe(
                            recipe_id=rid,
                            scan=scan,
                            site=site,
                            artifacts=[],
                            seed=seed,
                            rotate_dwi_bvecs=sampling.rotate_dwi_bvecs,
                        )
                    )
                for s_idx in range(sampling.samples_per_scan_per_site):
                    rid = f"sim{s_idx + 1:03d}"
                    seed = derive_seed(config.global_seed, f"{scan.base_id}:{site.site_id}:{rid}")
                    rng = np.random.default_rng(seed)
                    arts = _sample_combination(
                        specs,
                        rng=rng,
                        min_k=sampling.min_artifacts,
                        max_k=sampling.max_artifacts,
                        default_range=sampling.severity_range,
                    )
                    recipes.append(
                        ScanRecipe(
                            recipe_id=rid,
                            scan=scan,
                            site=site,
                            artifacts=arts,
                            seed=seed,
                            rotate_dwi_bvecs=sampling.rotate_dwi_bvecs,
                        )
                    )

    return recipes
