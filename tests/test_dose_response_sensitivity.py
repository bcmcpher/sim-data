"""Tests for zero-leakage acq_blinded naming, CRN coupled trajectories, and IQM sensitivity analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from ghibli.artifacts.base import ScanContext
from ghibli.artifacts.registry import create_artifact
from ghibli.config import (
    ArtifactSpec,
    CompoundTrajectorySpec,
    OutputConfig,
    SamplingPlanConfig,
    SimulationConfig,
)
from ghibli.eval.sensitivity import run_sensitivity_analysis
from ghibli.io.bids import discover_bids_scans, load_bids_scan
from ghibli.io.orientation import canonicalize_to_ras, compute_anatomical_priors
from ghibli.iqm.extractor import extract_iqms_for_bids_dir
from ghibli.pipeline.engine import run_simulation_plan


def test_blinding_guardrail_blocks_severity_in_acq_unless_unblinded_validation() -> None:
    with pytest.raises(ValidationError, match="include_severity_in_acq"):
        OutputConfig(naming_scheme="blinded", include_severity_in_acq=True)
    with pytest.raises(ValidationError, match="include_severity_in_acq"):
        OutputConfig(naming_scheme="acq_blinded", include_severity_in_acq=True)

    ok = OutputConfig(naming_scheme="unblinded_validation", include_severity_in_acq=True)
    assert ok.include_severity_in_acq is True


def test_crn_coupled_trajectories_are_strictly_nested_and_locked(bids_phantom_dir: Path) -> None:
    """Shared trajectory seed must lock geometry and produce strictly nested events across dose."""
    t1_scan = discover_bids_scans(bids_phantom_dir, modalities=["T1w"])[0]
    t1_loaded = load_bids_scan(t1_scan)
    t1_data, t1_aff, t1_zooms = canonicalize_to_ras(t1_loaded.img)
    t1_priors = compute_anatomical_priors(t1_data)

    def _t1_ctx() -> ScanContext:
        return ScanContext(
            data=t1_data.copy(),
            affine=t1_aff.copy(),
            zooms=t1_zooms,
            modality="T1w",
            pe_axis=1,
            readout_axis=0,
            slice_axis=2,
            priors=t1_priors,
            rng=np.random.default_rng(777),
            seed=777,
            device="cpu",
            coupled_trajectory=True,
        )

    # Susceptibility dipole landmark/center locked
    r_lo = create_artifact("susceptibility_metal", severity=0.25).apply(_t1_ctx())
    r_hi = create_artifact("susceptibility_metal", severity=0.80).apply(_t1_ctx())
    assert r_lo.params["landmark"] == r_hi.params["landmark"]
    assert r_lo.params["center_vox"] == r_hi.params["center_vox"]
    assert r_hi.params["radius_vox"] > r_lo.params["radius_vox"]

    # FOV cropping side locked & clipped fraction strictly increases
    c_lo = create_artifact("fov_cropping", severity=0.20).apply(_t1_ctx())
    c_hi = create_artifact("fov_cropping", severity=0.80).apply(_t1_ctx())
    assert c_lo.params["side"] == c_hi.params["side"]
    assert c_hi.params["brain_voxels_clipped_pct"] > c_lo.params["brain_voxels_clipped_pct"]

    # Defacing pattern locked & damaged fraction strictly increases
    d_lo = create_artifact("defacing_damage", severity=0.20).apply(_t1_ctx())
    d_hi = create_artifact("defacing_damage", severity=0.85).apply(_t1_ctx())
    assert d_lo.params["pattern"] == d_hi.params["pattern"]
    assert d_hi.params["brain_voxels_damaged_pct"] > d_lo.params["brain_voxels_damaged_pct"]

    # BiasField shading depth strictly increases on identical basis
    b_lo = create_artifact("bias_field", severity=0.20).apply(_t1_ctx())
    b_hi = create_artifact("bias_field", severity=0.85).apply(_t1_ctx())
    assert b_hi.params["realized_brain_field_range"] > b_lo.params["realized_brain_field_range"]

    # RFSpike locations strictly nested
    ctx_sp_lo, ctx_sp_hi = _t1_ctx(), _t1_ctx()
    create_artifact("rf_spike", severity=0.20).apply(ctx_sp_lo)
    create_artifact("rf_spike", severity=0.85).apply(ctx_sp_hi)
    locs_lo = set(ctx_sp_lo.metadata["gt_rf_spike_locs"])
    locs_hi = set(ctx_sp_hi.metadata["gt_rf_spike_locs"])
    assert locs_lo.issubset(locs_hi)

    # 4D DWI slice dropout pairs strictly nested
    dwi_scan = discover_bids_scans(bids_phantom_dir, modalities=["dwi"])[0]
    dwi_loaded = load_bids_scan(dwi_scan)
    dwi_data, dwi_aff, dwi_zooms = canonicalize_to_ras(dwi_loaded.img)
    dwi_priors = compute_anatomical_priors(dwi_data, bvals=dwi_loaded.bvals)

    def _dwi_ctx() -> ScanContext:
        return ScanContext(
            data=dwi_data.copy(),
            affine=dwi_aff.copy(),
            zooms=dwi_zooms,
            modality="dwi",
            pe_axis=1,
            readout_axis=0,
            slice_axis=2,
            priors=dwi_priors,
            rng=np.random.default_rng(888),
            seed=888,
            device="cpu",
            bvals=dwi_loaded.bvals.copy() if dwi_loaded.bvals is not None else None,
            bvecs=dwi_loaded.bvecs.copy() if dwi_loaded.bvecs is not None else None,
            coupled_trajectory=True,
        )

    ctx_dr_lo, ctx_dr_hi = _dwi_ctx(), _dwi_ctx()
    create_artifact("dwi_slice_dropout", severity=0.20).apply(ctx_dr_lo)
    create_artifact("dwi_slice_dropout", severity=0.85).apply(ctx_dr_hi)
    pairs_lo = set(ctx_dr_lo.metadata["gt_dropped_pairs"])
    pairs_hi = set(ctx_dr_hi.metadata["gt_dropped_pairs"])
    assert pairs_lo.issubset(pairs_hi)
    assert len(pairs_hi) > len(pairs_lo)

    # 4D BOLD motion jerk frames strictly nested & FD monotonically rises
    bold_scan = discover_bids_scans(bids_phantom_dir, modalities=["bold"])[0]
    bold_loaded = load_bids_scan(bold_scan)
    bold_data, bold_aff, bold_zooms = canonicalize_to_ras(bold_loaded.img)
    bold_priors = compute_anatomical_priors(bold_data)

    def _bold_ctx() -> ScanContext:
        return ScanContext(
            data=bold_data.copy(),
            affine=bold_aff.copy(),
            zooms=bold_zooms,
            modality="bold",
            pe_axis=1,
            readout_axis=0,
            slice_axis=2,
            priors=bold_priors,
            rng=np.random.default_rng(999),
            seed=999,
            device="cpu",
            coupled_trajectory=True,
        )

    ctx_m_lo, ctx_m_hi = _bold_ctx(), _bold_ctx()
    rec_m_lo = create_artifact("motion_4d", severity=0.20).apply(ctx_m_lo)
    rec_m_hi = create_artifact("motion_4d", severity=0.85).apply(ctx_m_hi)
    assert set(ctx_m_lo.metadata["gt_spike_frames"]).issubset(
        set(ctx_m_hi.metadata["gt_spike_frames"])
    )
    assert rec_m_hi.params["fd_mean_mm"] > rec_m_lo.params["fd_mean_mm"]


def test_acq_blinded_dose_response_zero_leakage_and_sensitivity_analysis(
    bids_phantom_dir: Path, tmp_path: Path
) -> None:
    """Default raw-BIDS acq_blinded dose-response leaks zero severity & yields strong IQM sensitivity."""
    cfg = SimulationConfig(
        global_seed=2026,
        modalities=["T1w", "bold", "dwi"],
        sampling=SamplingPlanConfig(
            mode="dose_response",
            include_clean_baseline=True,
            shuffle_run_order=True,
            coupled_trajectories=True,
            replicates_per_artifact=1,
            dose_severities=[0.0, 0.25, 0.55, 0.85],
            artifacts=[
                ArtifactSpec(name="rician_noise"),
                ArtifactSpec(name="bias_field"),
                ArtifactSpec(name="anisotropy"),
                ArtifactSpec(name="nyquist_n2_ghost"),
                ArtifactSpec(name="dwi_slice_dropout"),
            ],
            compound_trajectories=[
                CompoundTrajectorySpec(
                    name="noisePlusBias",
                    artifacts=[
                        ArtifactSpec(name="rician_noise", weight=1.0),
                        ArtifactSpec(name="bias_field", fixed_severity=0.35),
                    ],
                )
            ],
        ),
        sites=[],
        output=OutputConfig(
            naming_scheme="acq_blinded",
            extract_iqms=False,
            render_views=False,
            write_packets=False,
        ),
    )

    out_dir = tmp_path / "dose_blinded_bids"
    manifest_out = tmp_path / "dose_manifest.tsv"
    run_simulation_plan(
        bids_dir=bids_phantom_dir,
        config=cfg,
        out_dir=out_dir,
        workers=2,
        device="cpu",
        manifest_out=manifest_out,
    )

    desc = json.loads((out_dir / "dataset_description.json").read_text(encoding="utf-8"))
    assert desc["DatasetType"] == "raw"

    df = pd.read_csv(manifest_out, sep="\t")
    # Shared s=0 baseline emitted ONCE per (subject, modality) = 2 * 3 = 6
    base_df = df[df["target_artifact"] == "baseline"]
    assert len(base_df) == 6
    assert (df[df["target_artifact"] != "baseline"]["baseline_scan_id"].astype(str).str.len() > 0).all()

    # Zero leakage in filenames, JSON sidecars, and *_scans.tsv
    forbidden = [
        "baseline",
        "rician",
        "noise",
        "bias",
        "aniso",
        "ghost",
        "dropout",
        "s025",
        "s055",
        "s085",
        "_desc-",
        "_run-",
    ]
    for _, row in df.iterrows():
        nifti_p = out_dir / str(row["nifti_path"])
        assert nifti_p.exists()
        fname_l = nifti_p.name.lower()
        assert "_acq-v" in fname_l
        for tok in forbidden:
            assert tok not in fname_l, f"Leaked '{tok}' in {nifti_p.name}"
        sidecar = json.loads(Path(str(nifti_p).replace(".nii.gz", ".json")).read_text())
        assert "SimulationProvenance" not in sidecar

    scans_tsv = pd.read_csv(out_dir / "sub-01" / "sub-01_scans.tsv", sep="\t")
    assert set(scans_tsv.columns) == {"filename", "site_id"}

    # Blind IQM extraction -> Sensitivity analysis
    iqm_df = extract_iqms_for_bids_dir(out_dir, out_tsv=tmp_path / "dose_iqms.tsv", workers=2)
    assert len(iqm_df) == len(df)

    sens_t1 = run_sensitivity_analysis(
        df, out_dir=tmp_path / "sens_t1", iqms_df=iqm_df, modality="T1w"
    )
    assert sens_t1.summary_path.exists()
    assert sens_t1.matrix_path.exists()
    assert sens_t1.report_path.exists()
    assert sens_t1.heatmap_path is not None and sens_t1.heatmap_path.exists()
    assert sens_t1.curves_path is not None and sens_t1.curves_path.exists()

    mat_t1 = sens_t1.matrix_df
    assert float(mat_t1.loc["rician_noise", "iqm_snrd_total"]) < -0.85
    assert float(mat_t1.loc["bias_field", "iqm_inu_range"]) > 0.85
    assert float(mat_t1.loc["anisotropy", "iqm_fwhm_z"]) > 0.85
    assert "noisePlusBias" in mat_t1.index

    sens_bold = run_sensitivity_analysis(
        df, out_dir=tmp_path / "sens_bold", iqms_df=iqm_df, modality="bold"
    )
    assert float(sens_bold.matrix_df.loc["nyquist_n2_ghost", "iqm_gsr_y"]) > 0.85

    sens_dwi = run_sensitivity_analysis(
        df, out_dir=tmp_path / "sens_dwi", iqms_df=iqm_df, modality="dwi"
    )
    assert float(sens_dwi.matrix_df.loc["dwi_slice_dropout", "iqm_raw_num_bad_slices"]) > 0.80


def test_unblinded_validation_naming_orders_runs_monotonically(
    bids_phantom_dir: Path, tmp_path: Path
) -> None:
    cfg = SimulationConfig(
        global_seed=55,
        modalities=["T1w"],
        sampling=SamplingPlanConfig(
            mode="dose_response",
            include_clean_baseline=True,
            coupled_trajectories=True,
            replicates_per_artifact=2,
            dose_severities=[0.0, 0.20, 0.60, 0.90],
            artifacts=[ArtifactSpec(name="rician_noise")],
        ),
        sites=[],
        output=OutputConfig(
            naming_scheme="unblinded_validation",
            include_severity_in_acq=False,
        ),
    )
    out_dir = tmp_path / "unblinded_out"
    res = run_simulation_plan(bids_phantom_dir, cfg, out_dir=out_dir, workers=1, device="cpu")
    df = pd.read_csv(res.manifest_path, sep="\t", dtype={"subject": str, "run": str})

    sub1 = df[df["subject"] == "01"]
    base = sub1[sub1["target_artifact"] == "baseline"]
    assert len(base) == 1
    assert "_acq-baseline_run-01_T1w.nii.gz" in str(base.iloc[0]["nifti_path"])

    rep1 = sub1[(sub1["target_artifact"] == "rician_noise") & (sub1["replicate"] == 1)].sort_values(
        "run"
    )
    assert rep1["dose_level"].tolist() == [0.20, 0.60, 0.90]
    assert all("_acq-riciannoiseRep01_run-" in str(p) for p in rep1["nifti_path"])

    scans_tsv = pd.read_csv(out_dir / "sub-01" / "sub-01_scans.tsv", sep="\t")
    assert {"filename", "site_id", "trajectory_id", "dose_step", "dose_level"}.issubset(
        set(scans_tsv.columns)
    )
