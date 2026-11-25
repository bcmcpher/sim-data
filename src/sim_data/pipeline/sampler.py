"""Experimental recipe samplers with coupled CRN dose trajectories and zero-leakage naming."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sim_data.artifacts.registry import ARTIFACT_REGISTRY, compatible_artifacts
from sim_data.config import ArtifactSpec, SimulationConfig, SiteProfileConfig
from sim_data.io.bids import BIDSScan


def _alnum(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", value)


@dataclass
class PlannedArtifact:
    name: str
    severity: float
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanRecipe:
    """Complete deterministic specification for simulating one BIDS scan variation."""

    recipe_id: str
    scan: BIDSScan
    site: SiteProfileConfig
    artifacts: list[PlannedArtifact]
    seed: int
    run_index: int = 1
    rotate_dwi_bvecs: bool = False
    trajectory_id: str = ""
    target_artifact: str = "none"
    replicate: int = 1
    dose_step: int = 0
    dose_level: float = 0.0
    acq_label: str | None = None
    baseline_recipe_key: str = ""
    coupled_trajectory: bool = False
    site_seed: int | None = None


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


def _default_site(passthrough: bool = False) -> SiteProfileConfig:
    if passthrough:
        return SiteProfileConfig(site_id="NativeSite", passthrough=True)
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
        planned.append(
            PlannedArtifact(name=spec.name, severity=round(sev, 4), params=dict(spec.params))
        )
    return planned


def _assign_runs_and_acq(
    recipes: list[ScanRecipe],
    config: SimulationConfig,
) -> list[ScanRecipe]:
    """Assign BIDS `run-` and/or `acq-` entities according to `config.output.naming_scheme`.

    - `blinded`: Shuffled `run-01..run-N` per BIDS stem group (+ optional `desc-<Site>`).
    - `acq_blinded`: Shuffled opaque `acq-v01..acq-vNN` per `(sub, ses, task, mod)` with
      zero artifact, severity, or run-order leakage (raw-BIDS compliant).
    - `unblinded_validation` (explicit opt-in ONLY): Readable `acq-<Art>` and monotonic
      `run-01..run-K` sorted ascending by `dose_level`.
    """
    scheme = config.output.naming_scheme

    if scheme == "blinded":
        groups: dict[tuple[str, str, str, str, str], list[ScanRecipe]] = {}
        for rec in recipes:
            site_key = rec.site.site_id if config.output.site_naming == "desc" else ""
            task_key = rec.scan.entities.get("task", "")
            group_key = (
                rec.scan.subject,
                rec.scan.session or "",
                task_key,
                str(rec.scan.modality),
                site_key,
            )
            groups.setdefault(group_key, []).append(rec)

        ordered: list[ScanRecipe] = []
        for group_key in sorted(groups):
            bucket = list(groups[group_key])
            if config.sampling.shuffle_run_order and len(bucket) > 1:
                shuf_seed = derive_seed(config.global_seed, f"run_shuffle:{':'.join(group_key)}")
                rng = np.random.default_rng(shuf_seed)
                perm = rng.permutation(len(bucket))
                bucket = [bucket[int(i)] for i in perm]
            for idx, rec in enumerate(bucket, start=1):
                rec.run_index = idx
                rec.acq_label = None
                ordered.append(rec)
        return ordered

    if scheme == "acq_blinded":
        acq_groups: dict[tuple[str, str, str, str], list[ScanRecipe]] = {}
        for rec in recipes:
            task_key = rec.scan.entities.get("task", "")
            gkey = (
                rec.scan.subject,
                rec.scan.session or "",
                task_key,
                str(rec.scan.modality),
            )
            acq_groups.setdefault(gkey, []).append(rec)

        ordered = []
        for gkey in sorted(acq_groups):
            bucket = list(acq_groups[gkey])
            if config.sampling.shuffle_run_order and len(bucket) > 1:
                shuf_seed = derive_seed(config.global_seed, f"acq_shuffle:{':'.join(gkey)}")
                rng = np.random.default_rng(shuf_seed)
                perm = rng.permutation(len(bucket))
                bucket = [bucket[int(i)] for i in perm]
            width = 3 if len(bucket) >= 100 else 2
            for idx, rec in enumerate(bucket, start=1):
                rec.acq_label = f"v{idx:0{width}d}"
                rec.run_index = 0
                ordered.append(rec)
        return ordered

    # scheme == "unblinded_validation" (explicit opt-in only)
    multi_site = len({r.site.site_id for r in recipes}) > 1
    multi_rep = config.sampling.replicates_per_artifact > 1

    val_groups: dict[tuple[str, str, str, str, str], list[ScanRecipe]] = {}
    for rec in recipes:
        if rec.target_artifact == "baseline":
            acq = "baseline"
        else:
            acq = _alnum(rec.target_artifact) or "sim"
            if multi_rep:
                acq = f"{acq}Rep{rec.replicate:02d}"
            if config.output.include_severity_in_acq:
                acq = f"{acq}S{int(round(rec.dose_level * 100.0)):03d}"
        if multi_site:
            acq = f"{acq}Site{_alnum(rec.site.site_id)}"
        rec.acq_label = acq

        task_key = rec.scan.entities.get("task", "")
        vkey = (
            rec.scan.subject,
            rec.scan.session or "",
            task_key,
            str(rec.scan.modality),
            acq,
        )
        val_groups.setdefault(vkey, []).append(rec)

    ordered = []
    for vkey in sorted(val_groups):
        bucket = sorted(
            val_groups[vkey],
            key=lambda r: (r.dose_level, r.dose_step, r.replicate, r.recipe_id),
        )
        for idx, rec in enumerate(bucket, start=1):
            rec.run_index = idx
            ordered.append(rec)
    return ordered


def build_recipes(
    scans: list[BIDSScan],
    config: SimulationConfig,
) -> list[ScanRecipe]:
    """Expand BIDS scans + SimulationConfig into deterministic ScanRecipes."""
    sampling = config.sampling
    if config.sites:
        sites = config.sites
    else:
        sites = [_default_site(passthrough=(sampling.mode == "dose_response"))]

    recipes: list[ScanRecipe] = []

    for scan in scans:
        specs = _eligible_specs(config, scan)
        valid_names = set(compatible_artifacts(scan.modality))

        if sampling.mode == "dose_response":
            dose_sites = sites if sampling.sweep_all_sites else [sites[0]]
            sorted_sevs = sorted(float(s) for s in sampling.dose_severities)
            pos_sevs = [s for s in sorted_sevs if s > 0.0]
            include_base = sampling.include_clean_baseline or any(s <= 0.0 for s in sorted_sevs)

            for site in dose_sites:
                base_key = f"{scan.base_id}:{site.site_id}:baseline"
                site_seed = (
                    derive_seed(config.global_seed, f"site:{scan.base_id}:{site.site_id}")
                    if sampling.coupled_trajectories
                    else None
                )

                # Emit ONE shared s=0.0 baseline per (scan, site)
                if include_base:
                    rid = "baseline000"
                    recipes.append(
                        ScanRecipe(
                            recipe_id=rid,
                            scan=scan,
                            site=site,
                            artifacts=[],
                            seed=derive_seed(
                                config.global_seed, f"{scan.base_id}:{site.site_id}:{rid}"
                            ),
                            rotate_dwi_bvecs=sampling.rotate_dwi_bvecs,
                            trajectory_id=f"{scan.base_id}__{site.site_id}__baseline",
                            target_artifact="baseline",
                            replicate=1,
                            dose_step=0,
                            dose_level=0.0,
                            baseline_recipe_key=base_key,
                            coupled_trajectory=sampling.coupled_trajectories,
                            site_seed=site_seed,
                        )
                    )

                # Single-artifact coupled trajectories
                for spec in specs:
                    if spec.name not in ARTIFACT_REGISTRY:
                        continue
                    for rep in range(1, sampling.replicates_per_artifact + 1):
                        traj_id = f"{scan.base_id}__{site.site_id}__{spec.name}__rep{rep:02d}"
                        traj_seed = derive_seed(config.global_seed, f"traj:{traj_id}")
                        for step_idx, sev in enumerate(pos_sevs, start=1):
                            step_seed = (
                                traj_seed
                                if sampling.coupled_trajectories
                                else derive_seed(
                                    config.global_seed, f"{traj_id}:step{step_idx:02d}"
                                )
                            )
                            rid = f"dose{spec.name[:6]}r{rep:02d}s{step_idx:02d}"
                            recipes.append(
                                ScanRecipe(
                                    recipe_id=rid,
                                    scan=scan,
                                    site=site,
                                    artifacts=[
                                        PlannedArtifact(
                                            name=spec.name,
                                            severity=sev,
                                            params=dict(spec.params),
                                        )
                                    ],
                                    seed=step_seed,
                                    rotate_dwi_bvecs=sampling.rotate_dwi_bvecs,
                                    trajectory_id=traj_id,
                                    target_artifact=spec.name,
                                    replicate=rep,
                                    dose_step=step_idx,
                                    dose_level=sev,
                                    baseline_recipe_key=base_key,
                                    coupled_trajectory=sampling.coupled_trajectories,
                                    site_seed=site_seed,
                                )
                            )

                # Optional compound / multi-artifact coupled trajectories
                for comp in sampling.compound_trajectories:
                    comp_members = [
                        m
                        for m in comp.artifacts
                        if m.name in valid_names and m.name in ARTIFACT_REGISTRY
                    ]
                    if not comp_members:
                        continue
                    for rep in range(1, sampling.replicates_per_artifact + 1):
                        traj_id = f"{scan.base_id}__{site.site_id}__{comp.name}__rep{rep:02d}"
                        traj_seed = derive_seed(config.global_seed, f"traj:{traj_id}")
                        for step_idx, sev in enumerate(pos_sevs, start=1):
                            step_seed = (
                                traj_seed
                                if sampling.coupled_trajectories
                                else derive_seed(
                                    config.global_seed, f"{traj_id}:step{step_idx:02d}"
                                )
                            )
                            planned_comp: list[PlannedArtifact] = []
                            for member in comp_members:
                                m_sev = (
                                    float(member.fixed_severity)
                                    if member.fixed_severity is not None
                                    else float(np.clip(sev * member.weight, 0.0, 1.0))
                                )
                                planned_comp.append(
                                    PlannedArtifact(
                                        name=member.name,
                                        severity=round(m_sev, 4),
                                        params=dict(member.params),
                                    )
                                )
                            rid = f"comp{comp.name[:6]}r{rep:02d}s{step_idx:02d}"
                            recipes.append(
                                ScanRecipe(
                                    recipe_id=rid,
                                    scan=scan,
                                    site=site,
                                    artifacts=planned_comp,
                                    seed=step_seed,
                                    rotate_dwi_bvecs=sampling.rotate_dwi_bvecs,
                                    trajectory_id=traj_id,
                                    target_artifact=comp.name,
                                    replicate=rep,
                                    dose_step=step_idx,
                                    dose_level=sev,
                                    baseline_recipe_key=base_key,
                                    coupled_trajectory=sampling.coupled_trajectories,
                                    site_seed=site_seed,
                                )
                            )

        elif sampling.mode == "combinatorial":
            for sample_idx in range(sampling.samples_per_scan):
                site = sites[sample_idx % len(sites)]
                rid = f"comb{sample_idx:03d}"
                seed = derive_seed(config.global_seed, f"{scan.base_id}:{site.site_id}:{rid}")
                rng = np.random.default_rng(seed)
                is_clean = sample_idx == 0 and sampling.include_clean_baseline
                if is_clean:
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
                        trajectory_id=f"{scan.base_id}__{site.site_id}__{rid}",
                        target_artifact="baseline" if is_clean else "combinatorial",
                        baseline_recipe_key=f"{scan.base_id}:{site.site_id}:baseline",
                    )
                )

        elif sampling.mode == "multisite_factorial":
            for site in sites:
                base_key = f"{scan.base_id}:{site.site_id}:baseline"
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
                            trajectory_id=f"{scan.base_id}__{site.site_id}__baseline",
                            target_artifact="baseline",
                            baseline_recipe_key=base_key,
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
                            trajectory_id=f"{scan.base_id}__{site.site_id}__{rid}",
                            target_artifact="multisite_factorial",
                            baseline_recipe_key=base_key,
                        )
                    )

    return _assign_runs_and_acq(recipes, config)
