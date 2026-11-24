"""Physics-ordered 7-stage MRI simulation execution engine."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from sim_data.artifacts.base import ArtifactRecord, PipelineStage, ScanContext
from sim_data.artifacts.registry import ARTIFACT_REGISTRY, create_artifact
from sim_data.config import SimulationConfig
from sim_data.io.bids import (
    BIDSScan,
    build_derivative_stem,
    discover_bids_scans,
    load_bids_scan,
    write_bids_derivative,
)
from sim_data.io.orientation import (
    canonicalize_to_ras,
    compute_anatomical_priors,
    resolve_encoding_axes,
)
from sim_data.iqm.extractor import extract_iqms
from sim_data.pipeline.labels import compute_fidelity_metrics, compute_qc_targets, write_manifest
from sim_data.pipeline.sampler import ScanRecipe, build_recipes
from sim_data.sites.profiles import apply_site_noise_floor, apply_site_pre_profile
from sim_data.views.mosaic import render_scan_views
from sim_data.views.packet import assemble_qc_packet, write_qc_packet


@dataclass
class SimulationResult:
    """Summary returned after running a complete simulation plan."""

    out_dir: Path
    n_scans_generated: int
    manifest_paths: dict[str, Path]
    rows: list[dict[str, Any]]


def simulate_single_recipe(
    recipe: ScanRecipe,
    config: SimulationConfig,
    out_dir: str | Path,
    device: str = "auto",
) -> dict[str, Any]:
    """Execute the strict 7-stage physical causal pipeline for one ScanRecipe."""
    out_root = Path(out_dir)
    loaded = load_bids_scan(recipe.scan)
    clean_data, affine, zooms = canonicalize_to_ras(loaded.img)
    pe_axis, ro_axis, sl_axis = resolve_encoding_axes(loaded.sidecar, recipe.scan.modality)
    clean_priors = compute_anatomical_priors(clean_data, bvals=loaded.bvals)

    rng = np.random.default_rng(recipe.seed)
    ctx = ScanContext(
        data=clean_data.copy(),
        affine=affine.copy(),
        zooms=zooms,
        modality=recipe.scan.modality,
        pe_axis=pe_axis,
        readout_axis=ro_axis,
        slice_axis=sl_axis,
        priors=clean_priors,
        rng=rng,
        seed=recipe.seed,
        device=device,
        bvals=loaded.bvals.copy() if loaded.bvals is not None else None,
        bvecs=loaded.bvecs.copy() if loaded.bvecs is not None else None,
        rotate_dwi_bvecs=recipe.rotate_dwi_bvecs,
    )

    # Instantiate requested artifacts and sort strictly by PipelineStage
    transforms = [create_artifact(a.name, a.severity, a.params) for a in recipe.artifacts]
    transforms.sort(key=lambda t: int(t.stage))

    pre_noise_transforms = [t for t in transforms if t.stage < PipelineStage.NOISE_DRIFT]
    noise_transforms = [t for t in transforms if t.stage == PipelineStage.NOISE_DRIFT]
    post_recon_transforms = [t for t in transforms if t.stage >= PipelineStage.POST_RECON]

    records: list[ArtifactRecord] = []

    # Stage 2: Intrinsic site physics (contrast gamma, gain, PSF, static coil B1, voxel scaling)
    site_pre_meta = apply_site_pre_profile(ctx, recipe.site)

    # Stages 3, 4, 5: Spatial/Field/4D -> K-Space -> Artifact Bias Field
    for tfm in pre_noise_transforms:
        records.append(tfm.apply(ctx))

    # Stage 6: Site coil noise floor + artifact Rician/Non-Central Chi noise & drift
    site_noise_meta = apply_site_noise_floor(ctx, recipe.site)
    for tfm in noise_transforms:
        records.append(tfm.apply(ctx))

    # Stage 7: Post-reconstruction anonymization & export defects (Cropping, Defacing Damage)
    for tfm in post_recon_transforms:
        records.append(tfm.apply(ctx))

    # Compute labels, fidelity vs clean source, and NiPreps-aligned IQMs
    qc_targets = compute_qc_targets(records, thresholds=config.output.qc_thresholds)
    fidelity = compute_fidelity_metrics(clean_data, ctx.data, clean_priors=clean_priors)

    iqms: dict[str, float] = {}
    if config.output.extract_iqms:
        iqms = extract_iqms(
            ctx.data,
            modality=recipe.scan.modality,
            zooms=ctx.zooms,
            bvals=ctx.bvals,
            bvecs=ctx.bvecs,
            slice_axis=ctx.slice_axis,
            clean_priors=clean_priors,
            metadata=ctx.metadata,
        )

    _rel_dir, stem = build_derivative_stem(recipe.scan, recipe.recipe_id, recipe.site.site_id)
    scan_id = stem

    provenance = {
        "scan_id": scan_id,
        "source_nifti": str(recipe.scan.nifti_path),
        "recipe_id": recipe.recipe_id,
        "seed": recipe.seed,
        "site_pre": site_pre_meta,
        "site_noise": site_noise_meta,
        "artifacts": [
            {"name": r.name, "severity": r.severity, "stage": r.stage, "params": r.params}
            for r in records
        ],
        "qc_targets": qc_targets,
        "fidelity": fidelity,
    }

    nifti_rel_path: str | None = None
    if config.output.save_nifti:
        written = write_bids_derivative(
            out_dir=out_root,
            scan=recipe.scan,
            recipe_id=recipe.recipe_id,
            site_id=recipe.site.site_id,
            data=ctx.data,
            affine=ctx.affine,
            sidecar=loaded.sidecar,
            provenance=provenance,
            bvals=ctx.bvals,
            bvecs=ctx.bvecs,
        )
        nifti_rel_path = str(written["nifti"].relative_to(out_root))

    views: dict[str, dict[str, str]] = {"midline": {}, "enhanced": {}}
    if config.output.render_views:
        views = render_scan_views(
            data=ctx.data,
            modality=recipe.scan.modality,
            zooms=ctx.zooms,
            out_dir=out_root,
            scan_id=scan_id,
            image_format=config.output.image_format,
            render_midline=config.output.render_midline,
            render_enhanced=config.output.render_enhanced,
            bvals=ctx.bvals,
            slice_axis=ctx.slice_axis,
            metadata=ctx.metadata,
        )

    packet_rel_path: str | None = None
    if config.output.write_packets:
        gt_block = {
            **qc_targets,
            "fidelity": fidelity,
            "site_id": recipe.site.site_id,
            "rotate_dwi_bvecs": recipe.rotate_dwi_bvecs,
            "artifacts": [
                {"name": r.name, "severity": r.severity, "params": r.params}
                for r in records
                if r.severity > 0.0
            ],
        }
        packet = assemble_qc_packet(
            scan_id=scan_id,
            modality=recipe.scan.modality,
            site_id=recipe.site.site_id,
            iqms=iqms,
            views=views,
            ground_truth=gt_block,
            nifti_rel_path=nifti_rel_path,
        )
        pkt_path = write_qc_packet(packet, out_root)
        packet_rel_path = str(pkt_path.relative_to(out_root))

    # Build flat manifest row
    row: dict[str, Any] = {
        "scan_id": scan_id,
        "subject": recipe.scan.subject,
        "session": recipe.scan.session or "",
        "modality": recipe.scan.modality,
        "site_id": recipe.site.site_id,
        "recipe_id": recipe.recipe_id,
        "seed": recipe.seed,
        "source_nifti": str(recipe.scan.nifti_path),
        "nifti_path": nifti_rel_path or "",
        "packet_path": packet_rel_path or "",
        "qc_label": qc_targets["qc_label"],
        "qc_rating_int": qc_targets["qc_rating_int"],
        "qc_score": qc_targets["qc_score"],
        "defect_burden": qc_targets["defect_burden"],
        "max_artifact_severity": qc_targets["max_artifact_severity"],
        "num_artifacts": qc_targets["num_artifacts"],
        "primary_artifact": qc_targets["primary_artifact"],
        "artifacts_json": json.dumps(
            [{"name": r.name, "severity": r.severity, "params": r.params} for r in records]
        ),
    }

    # Explicit per-artifact severity columns for every registered artifact
    sev_by_name = {r.name: r.severity for r in records}
    for art_name in sorted(ARTIFACT_REGISTRY):
        row[f"sev_{art_name}"] = float(sev_by_name.get(art_name, 0.0))

    for k, v in fidelity.items():
        row[k] = v

    for k, v in iqms.items():
        row[f"iqm_{k}"] = v

    return row


def run_simulation_plan(
    bids_dir: str | Path,
    config: SimulationConfig,
    out_dir: str | Path,
    workers: int = 1,
    device: str = "auto",
    scans_override: list[BIDSScan] | None = None,
) -> SimulationResult:
    """Discover BIDS scans, expand recipes, run simulation in parallel, and write manifest."""
    scans = scans_override if scans_override is not None else discover_bids_scans(
        bids_dir, modalities=config.modalities
    )
    if not scans:
        raise ValueError(f"No matching BIDS scans found under {bids_dir} for {config.modalities}")

    recipes = build_recipes(scans, config)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Save resolved config for full reproducibility
    config.to_yaml(out_root / "simulation_config_resolved.yaml")

    if workers <= 1 or len(recipes) <= 1:
        rows = [
            simulate_single_recipe(r, config=config, out_dir=out_root, device=device)
            for r in recipes
        ]
    else:
        rows = Parallel(n_jobs=workers, prefer="processes")(
            delayed(simulate_single_recipe)(r, config, out_root, device) for r in recipes
        )

    manifest_paths = write_manifest(rows, out_dir=out_root, formats=config.output.manifest_formats)
    return SimulationResult(
        out_dir=out_root,
        n_scans_generated=len(rows),
        manifest_paths=manifest_paths,
        rows=rows,
    )
