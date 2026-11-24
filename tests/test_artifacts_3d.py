"""Unit & physical-monotonicity tests for all 11 3D structural MRI artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sim_data.artifacts.base import ScanContext
from sim_data.artifacts.registry import create_artifact
from sim_data.io.bids import discover_bids_scans, load_bids_scan
from sim_data.io.orientation import (
    canonicalize_to_ras,
    compute_anatomical_priors,
    resolve_encoding_axes,
)
from sim_data.iqm.structural import extract_structural_iqms
from sim_data.pipeline.labels import compute_fidelity_metrics

STRUCTURAL_ARTIFACTS = [
    "motion_3d",
    "ghosting_3d",
    "wrap_around",
    "bias_field",
    "rf_spike",
    "susceptibility_metal",
    "rician_noise",
    "fov_cropping",
    "gibbs_ringing",
    "anisotropy",
    "defacing_damage",
]


def _make_t1_context(bids_phantom_dir: Path, seed: int = 100) -> tuple[ScanContext, np.ndarray]:
    scans = discover_bids_scans(bids_phantom_dir, modalities=["T1w"])
    loaded = load_bids_scan(scans[0])
    data, affine, zooms = canonicalize_to_ras(loaded.img)
    pe, ro, sl = resolve_encoding_axes(loaded.sidecar, "T1w")
    priors = compute_anatomical_priors(data)
    ctx = ScanContext(
        data=data.copy(),
        affine=affine.copy(),
        zooms=zooms,
        modality="T1w",
        pe_axis=pe,
        readout_axis=ro,
        slice_axis=sl,
        priors=priors,
        rng=np.random.default_rng(seed),
        seed=seed,
        device="cpu",
    )
    return ctx, data


@pytest.mark.parametrize("artifact_name", STRUCTURAL_ARTIFACTS)
def test_structural_artifact_runs_and_is_deterministic(
    bids_phantom_dir: Path, artifact_name: str
) -> None:
    ctx_a, clean = _make_t1_context(bids_phantom_dir, seed=202)
    ctx_b, _ = _make_t1_context(bids_phantom_dir, seed=202)

    rec_a = create_artifact(artifact_name, severity=0.65).apply(ctx_a)
    rec_b = create_artifact(artifact_name, severity=0.65).apply(ctx_b)

    assert rec_a.name == artifact_name
    assert rec_a.severity == pytest.approx(0.65)
    assert np.all(np.isfinite(ctx_a.data))
    assert np.allclose(ctx_a.data, ctx_b.data, rtol=1e-4, atol=1e-3)
    assert float(np.mean(np.abs(ctx_a.data - clean))) > 1e-3
    assert rec_a.params == rec_b.params


def test_rician_noise_degrades_snrd_monotonically(bids_phantom_dir: Path) -> None:
    snrds = []
    bg_means = []
    for sev in (0.0, 0.35, 0.85):
        ctx, _ = _make_t1_context(bids_phantom_dir, seed=303)
        create_artifact("rician_noise", severity=sev).apply(ctx)
        iqm = extract_structural_iqms(ctx.data, zooms=ctx.zooms)
        snrds.append(iqm["snrd_total"])
        bg_means.append(iqm["summary_bg_mean"])

    assert snrds[0] > snrds[1] > snrds[2]
    assert bg_means[0] < bg_means[1] < bg_means[2]


def test_bias_field_increases_inu_and_cjv(bids_phantom_dir: Path) -> None:
    ctx_clean, _ = _make_t1_context(bids_phantom_dir, seed=404)
    iqm_clean = extract_structural_iqms(ctx_clean.data, zooms=ctx_clean.zooms)

    ctx_sev, _ = _make_t1_context(bids_phantom_dir, seed=404)
    create_artifact("bias_field", severity=0.85).apply(ctx_sev)
    iqm_sev = extract_structural_iqms(ctx_sev.data, zooms=ctx_sev.zooms)

    assert iqm_sev["inu_range"] > iqm_clean["inu_range"] * 1.4
    assert iqm_sev["cjv"] > iqm_clean["cjv"]


def test_anisotropy_increases_slice_fwhm(bids_phantom_dir: Path) -> None:
    ctx_clean, _ = _make_t1_context(bids_phantom_dir, seed=505)
    iqm_clean = extract_structural_iqms(ctx_clean.data, zooms=ctx_clean.zooms)

    ctx_aniso, _ = _make_t1_context(bids_phantom_dir, seed=505)
    create_artifact("anisotropy", severity=0.85).apply(ctx_aniso)
    iqm_aniso = extract_structural_iqms(ctx_aniso.data, zooms=ctx_aniso.zooms)

    assert iqm_aniso["fwhm_z"] > iqm_clean["fwhm_z"]


def test_defacing_damage_is_post_noise_zero_and_reduces_retention(bids_phantom_dir: Path) -> None:
    ctx, clean = _make_t1_context(bids_phantom_dir, seed=606)
    create_artifact("rician_noise", severity=0.30).apply(ctx)
    iqm_before = extract_structural_iqms(ctx.data, zooms=ctx.zooms)

    rec = create_artifact(
        "defacing_damage", severity=0.85, params={"pattern": "anterior_wedge"}
    ).apply(ctx)

    assert rec.params["brain_voxels_damaged_pct"] > 5.0
    fid = compute_fidelity_metrics(clean, ctx.data, ctx.priors)
    assert fid["brain_mask_retention"] < 0.95

    # Blind image-only anterior_inferior_zero_ratio jumps strictly from ~0 to >0.10
    iqm_after = extract_structural_iqms(ctx.data, zooms=ctx.zooms)
    assert iqm_before["anterior_inferior_zero_ratio"] < 0.01
    assert iqm_after["anterior_inferior_zero_ratio"] > 0.10


def test_fov_cropping_truncate_matrix_updates_affine(bids_phantom_dir: Path) -> None:
    ctx, clean = _make_t1_context(bids_phantom_dir, seed=707)
    orig_shape = ctx.data.shape
    orig_origin = ctx.affine[:3, 3].copy()

    rec = create_artifact(
        "fov_cropping",
        severity=0.75,
        params={"side": "inferior", "mode": "truncate_matrix"},
    ).apply(ctx)

    assert ctx.data.shape[2] < orig_shape[2]
    assert ctx.affine[2, 3] > orig_origin[2]
    assert rec.params["brain_voxels_clipped_pct"] > 5.0
    fid = compute_fidelity_metrics(clean, ctx.data, ctx.priors)
    assert fid["brain_mask_retention"] < 0.96
