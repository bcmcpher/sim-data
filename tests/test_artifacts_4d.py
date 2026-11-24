"""Unit & NiPreps IQM tests for 4D BOLD (fMRIPrep/MRIQC) and 4D DWI (QSIPrep/MRIQC) artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sim_data.artifacts.base import ScanContext
from sim_data.artifacts.registry import create_artifact
from sim_data.io.bids import discover_bids_scans, load_bids_scan
from sim_data.io.orientation import (
    canonicalize_to_ras,
    compute_anatomical_priors,
    resolve_encoding_axes,
)
from sim_data.iqm.diffusion import extract_diffusion_iqms
from sim_data.iqm.functional import extract_functional_iqms


def _make_4d_context(
    bids_phantom_dir: Path,
    modality: str,
    seed: int = 111,
    rotate_bvecs: bool = False,
) -> ScanContext:
    scans = discover_bids_scans(bids_phantom_dir, modalities=[modality])  # type: ignore[list-item]
    loaded = load_bids_scan(scans[0])
    data, affine, zooms = canonicalize_to_ras(loaded.img)
    pe, ro, sl = resolve_encoding_axes(loaded.sidecar, modality)
    priors = compute_anatomical_priors(data, bvals=loaded.bvals)
    return ScanContext(
        data=data.copy(),
        affine=affine.copy(),
        zooms=zooms,
        modality=modality,  # type: ignore[arg-type]
        pe_axis=pe,
        readout_axis=ro,
        slice_axis=sl,
        priors=priors,
        rng=np.random.default_rng(seed),
        seed=seed,
        device="cpu",
        bvals=loaded.bvals.copy() if loaded.bvals is not None else None,
        bvecs=loaded.bvecs.copy() if loaded.bvecs is not None else None,
        rotate_dwi_bvecs=rotate_bvecs,
    )


def test_bold_motion_4d_increases_fd_dvars_and_aqi(bids_phantom_dir: Path) -> None:
    ctx_clean = _make_4d_context(bids_phantom_dir, "bold", seed=10)
    iqm_clean = extract_functional_iqms(ctx_clean.data, ctx_clean.zooms)

    ctx_mod = _make_4d_context(bids_phantom_dir, "bold", seed=10)
    create_artifact("motion_4d", severity=0.35).apply(ctx_mod)
    iqm_mod = extract_functional_iqms(ctx_mod.data, ctx_mod.zooms, metadata=ctx_mod.metadata)

    ctx_sev = _make_4d_context(bids_phantom_dir, "bold", seed=10)
    create_artifact("motion_4d", severity=0.85).apply(ctx_sev)
    iqm_sev = extract_functional_iqms(ctx_sev.data, ctx_sev.zooms, metadata=ctx_sev.metadata)

    assert iqm_clean["fd_mean"] < iqm_mod["fd_mean"] < iqm_sev["fd_mean"]
    assert iqm_sev["dvars_nstd"] > iqm_clean["dvars_nstd"]
    assert iqm_sev["aqi"] > iqm_clean["aqi"]
    assert iqm_sev["tsnr"] < iqm_clean["tsnr"]


def test_bold_nyquist_n2_ghost_elevates_gsr_y(bids_phantom_dir: Path) -> None:
    ctx_clean = _make_4d_context(bids_phantom_dir, "bold", seed=20)
    iqm_clean = extract_functional_iqms(ctx_clean.data, ctx_clean.zooms)

    ctx_ghost = _make_4d_context(bids_phantom_dir, "bold", seed=20)
    create_artifact("nyquist_n2_ghost", severity=0.80).apply(ctx_ghost)
    iqm_ghost = extract_functional_iqms(ctx_ghost.data, ctx_ghost.zooms)

    assert iqm_ghost["gsr_y"] > iqm_clean["gsr_y"] + 0.03


def test_bold_epi_susceptibility_and_drift(bids_phantom_dir: Path) -> None:
    ctx = _make_4d_context(bids_phantom_dir, "bold", seed=30)
    clean_shape = ctx.data.shape
    create_artifact("epi_susceptibility", severity=0.75).apply(ctx)
    create_artifact("temporal_drift", severity=0.80).apply(ctx)
    assert ctx.data.shape == clean_shape
    assert np.all(np.isfinite(ctx.data))
    iqm = extract_functional_iqms(ctx.data, ctx.zooms)
    assert abs(iqm["temporal_drift_slope"]) > 0.02


def test_dwi_slice_dropout_drives_qsiprep_bad_slices_and_ndc(bids_phantom_dir: Path) -> None:
    ctx_clean = _make_4d_context(bids_phantom_dir, "dwi", seed=40)
    iqm_clean = extract_diffusion_iqms(
        ctx_clean.data, ctx_clean.zooms, bvals=ctx_clean.bvals, bvecs=ctx_clean.bvecs
    )

    ctx_drop = _make_4d_context(bids_phantom_dir, "dwi", seed=40)
    rec = create_artifact("dwi_slice_dropout", severity=0.85).apply(ctx_drop)
    iqm_drop = extract_diffusion_iqms(
        ctx_drop.data, ctx_drop.zooms, bvals=ctx_drop.bvals, bvecs=ctx_drop.bvecs
    )

    assert rec.params["n_dropped_slices"] > 0
    assert iqm_drop["raw_num_bad_slices"] > iqm_clean["raw_num_bad_slices"]
    assert iqm_drop["raw_neighbor_corr"] < iqm_clean["raw_neighbor_corr"]


def test_dwi_motion_preserves_unrotated_bvecs_by_default_and_rotates_on_flag(
    bids_phantom_dir: Path,
) -> None:
    ctx_raw = _make_4d_context(bids_phantom_dir, "dwi", seed=50, rotate_bvecs=False)
    orig_bvecs = ctx_raw.bvecs.copy()
    create_artifact("motion_4d", severity=0.80).apply(ctx_raw)
    assert np.allclose(ctx_raw.bvecs, orig_bvecs, atol=1e-6)

    ctx_prep = _make_4d_context(bids_phantom_dir, "dwi", seed=50, rotate_bvecs=True)
    create_artifact("motion_4d", severity=0.80).apply(ctx_prep)
    assert not np.allclose(ctx_prep.bvecs, orig_bvecs, atol=1e-4)
    # Unit norms preserved on high-b vectors
    high_b = ctx_prep.bvals > 150.0
    norms = np.linalg.norm(ctx_prep.bvecs[:, high_b], axis=0)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_dwi_eddy_currents_runs_cleanly(bids_phantom_dir: Path) -> None:
    ctx = _make_4d_context(bids_phantom_dir, "dwi", seed=60)
    orig = ctx.data.copy()
    rec = create_artifact("eddy_currents_dwi", severity=0.75).apply(ctx)
    assert rec.params["affected_volumes"] > 0
    assert ctx.data.shape == orig.shape
    assert float(np.mean(np.abs(ctx.data - orig))) > 0.5
