"""Physics-ordered 7-stage MRI simulation execution engine with zero-leakage BIDS export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import nibabel.processing as nib_proc
import numpy as np
from joblib import Parallel, delayed

from ghibli.artifacts.base import ArtifactRecord, PipelineStage, ScanContext
from ghibli.artifacts.registry import ARTIFACT_REGISTRY, create_artifact
from ghibli.config import SimulationConfig
from ghibli.io.bids import (
    BIDSScan,
    build_output_bids_stem,
    discover_bids_scans,
    discover_subject_brain_mask,
    load_bids_scan,
    write_bids_root_files,
    write_bids_scan,
)
from ghibli.io.orientation import (
    canonicalize_to_ras,
    canonicalize_to_ras_with_transform,
    compute_anatomical_priors,
    estimate_background_noise_sigma,
    reorient_bvecs_to_ras,
    resolve_encoding_axes,
)
from ghibli.iqm.extractor import extract_iqms
from ghibli.pipeline.labels import (
    compute_defect_summary,
    compute_fidelity_metrics,
    write_manifest_tsv,
)
from ghibli.pipeline.sampler import ScanRecipe, build_recipes
from ghibli.sites.profiles import apply_site_noise_floor, apply_site_pre_profile
from ghibli.views.mosaic import render_scan_views
from ghibli.views.packet import assemble_qc_packet, write_qc_packet


@dataclass
class SimulationResult:
    """Summary returned after running a complete simulation plan."""

    out_dir: Path
    n_scans_generated: int
    manifest_path: Path
    bids_root_files: dict[str, Path]
    rows: list[dict[str, Any]]


def _load_aligned_icv_mask(
    mask_path: Path | None,
    target_shape: tuple[int, int, int],
    target_affine: np.ndarray,
) -> np.ndarray | None:
    """Load and RAS-align an optional external/cached intracranial brain mask."""
    if mask_path is None or not mask_path.exists():
        return None
    try:
        m_img = nib.load(str(mask_path))
        if not isinstance(m_img, nib.Nifti1Image):
            if isinstance(m_img, nib.spatialimages.SpatialImage):
                m_img = nib.Nifti1Image(np.asarray(m_img.get_fdata(), dtype=np.float32), m_img.affine)
            else:
                return None
        m_data, m_aff, _ = canonicalize_to_ras(m_img)
        if m_data.ndim == 4:
            m_data = m_data[..., 0]
        if m_data.shape != target_shape or not np.allclose(m_aff, target_affine, atol=1e-2):
            res = nib_proc.resample_from_to(
                nib.Nifti1Image(m_data.astype(np.float32), m_aff),
                (target_shape, target_affine),
                order=0,
            )
            m_data = np.asarray(res.get_fdata(dtype=np.float32), dtype=np.float32)
        mask = np.asarray(m_data > 0.5, dtype=bool)
        return mask if int(mask.sum()) >= 64 else None
    except Exception:
        return None


def simulate_single_recipe(
    recipe: ScanRecipe,
    config: SimulationConfig,
    out_dir: str | Path,
    device: str = "auto",
    brain_mask_path: Path | None = None,
) -> dict[str, Any]:
    """Execute the strict 7-stage physical causal pipeline for one ScanRecipe."""
    out_root = Path(out_dir)
    loaded = load_bids_scan(recipe.scan)
    clean_data, affine, zooms, ornt_transform = canonicalize_to_ras_with_transform(loaded.img)
    pe_axis, ro_axis, sl_axis = resolve_encoding_axes(
        loaded.sidecar, recipe.scan.modality, ornt_transform=ornt_transform
    )
    bvecs_ras = (
        reorient_bvecs_to_ras(loaded.bvecs, ornt_transform) if loaded.bvecs is not None else None
    )
    icv_mask: np.ndarray | None = None
    if brain_mask_path is not None and recipe.scan.modality in ("T1w", "T2w", "FLAIR"):
        spatial_shape = (int(clean_data.shape[0]), int(clean_data.shape[1]), int(clean_data.shape[2]))
        icv_mask = _load_aligned_icv_mask(brain_mask_path, spatial_shape, affine)
    clean_priors = compute_anatomical_priors(clean_data, bvals=loaded.bvals, icv_mask=icv_mask)
    baseline_sigma = estimate_background_noise_sigma(
        clean_data, clean_priors.bg_mask, modality=recipe.scan.modality, bvals=loaded.bvals
    )

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
        bvecs=bvecs_ras.copy() if bvecs_ras is not None else None,
        rotate_dwi_bvecs=recipe.rotate_dwi_bvecs,
        coupled_trajectory=recipe.coupled_trajectory,
        baseline_noise_sigma=baseline_sigma,
        noise_calibration=config.sampling.noise_calibration,
        site_seed=recipe.site_seed,
    )

    transforms = [create_artifact(a.name, a.severity, a.params) for a in recipe.artifacts]
    transforms.sort(key=lambda t: int(t.stage))

    pre_noise_transforms = [t for t in transforms if t.stage < PipelineStage.NOISE_DRIFT]
    noise_transforms = [t for t in transforms if t.stage == PipelineStage.NOISE_DRIFT]
    post_recon_transforms = [t for t in transforms if t.stage >= PipelineStage.POST_RECON]

    records: list[ArtifactRecord] = []

    # Stage 2: Intrinsic site physics (no-op if site.passthrough=True)
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

    defect_summary = compute_defect_summary(
        records,
        thresholds=config.output.qc_thresholds,
        include_proxy_labels=config.output.include_proxy_qc_labels,
    )
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
        )

    site_for_name = recipe.site.site_id if config.sites else None
    _rel_dir, stem = build_output_bids_stem(
        recipe.scan,
        run_index=recipe.run_index,
        site_id=site_for_name,
        site_naming=config.output.site_naming,
        naming_scheme=config.output.naming_scheme,
        acq_label=recipe.acq_label,
    )
    scan_id = stem

    nifti_rel_path: str | None = None
    if config.output.save_nifti:
        written = write_bids_scan(
            out_dir=out_root,
            scan=recipe.scan,
            run_index=recipe.run_index,
            site_id=site_for_name,
            site_naming=config.output.site_naming,
            data=ctx.data,
            affine=ctx.affine,
            sidecar=loaded.sidecar,
            bvals=ctx.bvals,
            bvecs=ctx.bvecs,
            naming_scheme=config.output.naming_scheme,
            acq_label=recipe.acq_label,
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
        )

    packet_rel_path: str | None = None
    if config.output.write_packets:
        gt_block = {
            **defect_summary,
            "fidelity": fidelity,
            "site_id": recipe.site.site_id,
            "trajectory_id": recipe.trajectory_id,
            "dose_level": recipe.dose_level,
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

    psf_real = site_pre_meta.get("realized_psf_sigma", [0.0, 0.0, 0.0])
    row: dict[str, Any] = {
        "scan_id": scan_id,
        "subject": recipe.scan.subject,
        "session": recipe.scan.session or "",
        "acq_label": recipe.acq_label or "",
        "run": f"{recipe.run_index:02d}" if recipe.run_index > 0 else "",
        "modality": recipe.scan.modality,
        "site_id": recipe.site.site_id,
        "site_passthrough": bool(recipe.site.passthrough),
        "trajectory_id": recipe.trajectory_id,
        "target_artifact": recipe.target_artifact,
        "replicate": recipe.replicate,
        "dose_step": recipe.dose_step,
        "dose_level": round(float(recipe.dose_level), 4),
        "_baseline_recipe_key": recipe.baseline_recipe_key,
        "baseline_scan_id": "",
        "baseline_noise_sigma": round(float(baseline_sigma), 5),
        "seed": recipe.seed,
        "source_nifti": str(recipe.scan.nifti_path),
        "nifti_path": nifti_rel_path or "",
        "packet_path": packet_rel_path or "",
        "defect_burden": defect_summary["defect_burden"],
        "max_artifact_severity": defect_summary["max_artifact_severity"],
        "num_artifacts": defect_summary["num_artifacts"],
        "primary_artifact": defect_summary["primary_artifact"],
        "site_gamma": site_pre_meta.get("realized_gamma", 1.0),
        "site_gain": site_pre_meta.get("realized_gain", 1.0),
        "site_offset": site_pre_meta.get("realized_offset", 0.0),
        "site_psf_x": psf_real[0],
        "site_psf_y": psf_real[1],
        "site_psf_z": psf_real[2],
        "site_coil_count": site_pre_meta.get("coil_count", recipe.site.coil_count),
        "site_noise_sigma_rel": site_noise_meta.get("site_noise_sigma_rel", 0.0),
        "site_g_factor": site_noise_meta.get("site_g_factor", 0.0),
        "artifacts_json": json.dumps(
            [{"name": r.name, "severity": r.severity, "params": r.params} for r in records]
        ),
    }

    if config.output.include_proxy_qc_labels:
        row["proxy_qc_label"] = defect_summary["proxy_qc_label"]
        row["proxy_qc_rating_int"] = defect_summary["proxy_qc_rating_int"]

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
    manifest_out: str | Path | None = None,
    scans_override: list[BIDSScan] | None = None,
) -> SimulationResult:
    """Discover BIDS scans, expand recipes, simulate in parallel, and write manifest.tsv."""
    scans = (
        scans_override
        if scans_override is not None
        else discover_bids_scans(bids_dir, modalities=config.modalities, acq=config.acq)
    )
    if not scans:
        raise ValueError(f"No matching BIDS scans found under {bids_dir} for {config.modalities}")

    # Auto-discover optional per-subject brain/ICV masks once so structural priors & landmarks
    # anchor to the cranium instead of full head+neck without re-running extraction per recipe.
    mask_by_sub_ses: dict[tuple[str, str | None], Path | None] = {}
    for s in scans:
        key = (s.subject, s.session)
        if key not in mask_by_sub_ses:
            mask_by_sub_ses[key] = discover_subject_brain_mask(
                bids_dir, subject=s.subject, session=s.session
            )

    recipes = build_recipes(scans, config)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    config.to_yaml(out_root / "simulation_config_resolved.yaml")

    if workers <= 1 or len(recipes) <= 1:
        rows = [
            simulate_single_recipe(
                r,
                config=config,
                out_dir=out_root,
                device=device,
                brain_mask_path=mask_by_sub_ses.get((r.scan.subject, r.scan.session)),
            )
            for r in recipes
        ]
    else:
        rows = Parallel(n_jobs=workers, prefer="processes")(
            delayed(simulate_single_recipe)(
                r,
                config,
                out_root,
                device,
                mask_by_sub_ses.get((r.scan.subject, r.scan.session)),
            )
            for r in recipes
        )

    # Link every trajectory step to its shared s=0 baseline scan_id
    baseline_map: dict[str, str] = {}
    for row in rows:
        bkey = str(row.get("_baseline_recipe_key", ""))
        if bkey and row.get("target_artifact") == "baseline":
            baseline_map[bkey] = str(row["scan_id"])
    for row in rows:
        bkey = str(row.pop("_baseline_recipe_key", ""))
        row["baseline_scan_id"] = baseline_map.get(bkey, "")

    bids_root_files: dict[str, Path] = {}
    if config.output.write_bids_metadata:
        bids_root_files = write_bids_root_files(
            out_root,
            rows=rows,
            site_naming=config.output.site_naming,
            naming_scheme=config.output.naming_scheme,
        )

    manifest_target = Path(manifest_out) if manifest_out is not None else out_root / "manifest.tsv"
    manifest_path = write_manifest_tsv(rows, manifest_path=manifest_target)

    return SimulationResult(
        out_dir=out_root,
        n_scans_generated=len(rows),
        manifest_path=manifest_path,
        bids_root_files=bids_root_files,
        rows=rows,
    )
