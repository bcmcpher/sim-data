"""Multi-site ComBat vs. CovBat image-domain harmonization verification tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ghibli.config import (
    ArtifactSpec,
    OutputConfig,
    SamplingPlanConfig,
    SimulationConfig,
    SiteProfileConfig,
)
from ghibli.harmonization.audit import run_harmonization_audit
from ghibli.iqm.extractor import extract_iqms_for_bids_dir
from ghibli.phantoms.synthetic import generate_synthetic_bids_cohort
from ghibli.pipeline.engine import run_simulation_plan


def test_multisite_image_profiles_harmonized_by_combat_and_covbat(tmp_path: Path) -> None:
    bids_dir = generate_synthetic_bids_cohort(
        out_dir=tmp_path / "bids_ms",
        n_subjects=6,
        anat_shape=(36, 40, 36),
        epi_shape=(32, 32, 24),
        n_bold_trs=8,
        n_dwi_dirs=6,
        seed=555,
    )

    sites = [
        SiteProfileConfig(
            site_id="Siemens_3T_32ch",
            contrast_gamma=0.90,
            intensity_gain=1.10,
            psf_sigma_vox=(0.15, 0.15, 0.20),
            bias_coeffs=(0.08, -0.05, 0.06),
            coil_count=32,
            noise_sigma_rel=0.010,
            g_factor_strength=0.10,
            jitter_scale=0.08,
        ),
        SiteProfileConfig(
            site_id="GE_15T_8ch",
            contrast_gamma=1.22,
            intensity_gain=0.84,
            intensity_offset=0.02,
            psf_sigma_vox=(0.75, 0.75, 1.00),
            bias_coeffs=(-0.18, 0.16, -0.12),
            coil_count=8,
            noise_sigma_rel=0.036,
            g_factor_strength=0.25,
            jitter_scale=0.10,
        ),
        SiteProfileConfig(
            site_id="Philips_3T_16ch",
            contrast_gamma=1.05,
            intensity_gain=1.30,
            psf_sigma_vox=(0.45, 0.45, 0.80),
            bias_coeffs=(0.16, 0.14, -0.14),
            coil_count=16,
            noise_sigma_rel=0.022,
            g_factor_strength=0.35,
            jitter_scale=0.09,
        ),
    ]

    cfg = SimulationConfig(
        global_seed=808,
        modalities=["T1w"],
        sampling=SamplingPlanConfig(
            mode="multisite_factorial",
            include_clean_baseline=True,
            shuffle_run_order=True,
            samples_per_scan_per_site=2,
            min_artifacts=1,
            max_artifacts=2,
            severity_range=(0.30, 0.85),
            artifacts=[
                ArtifactSpec(name="rician_noise"),
                ArtifactSpec(name="bias_field"),
                ArtifactSpec(name="gibbs_ringing"),
                ArtifactSpec(name="anisotropy"),
            ],
        ),
        sites=sites,
        output=OutputConfig(
            save_nifti=True,
            write_bids_metadata=True,
            site_naming="desc",
            extract_iqms=False,
            render_views=False,
            write_packets=False,
            include_proxy_qc_labels=False,
        ),
    )

    out_dir = tmp_path / "sim_ms_out"
    res = run_simulation_plan(bids_dir=bids_dir, config=cfg, out_dir=out_dir, workers=2, device="cpu")
    df = pd.read_csv(res.manifest_path, sep="\t")
    assert len(df) == 6 * 3 * 3  # 54 blinded BIDS scans across 3 sites

    # Extract blind IQMs strictly from the generated NIfTI files on disk
    iqm_df = extract_iqms_for_bids_dir(out_dir, out_tsv=tmp_path / "ms_iqms.tsv", workers=2)
    audit = run_harmonization_audit(df, out_dir=out_dir / "audit", iqms_df=iqm_df, modality="T1w")
    rep = audit.report

    assert audit.report_path.exists()
    assert (out_dir / "audit" / "iqms_combat.tsv").exists()
    assert (out_dir / "audit" / "iqms_covbat.tsv").exists()
    assert audit.plot_path is not None and audit.plot_path.exists()

    # 1. Raw image-domain site profiles induce strong detectable site effects
    assert rep["raw"]["site_linear_auc"] > 0.78
    assert rep["raw"]["site_rf_auc"] > 0.80
    assert rep["raw"]["mean_anova_f"] > 2.0

    # 2. ComBat sharply collapses marginal site location differences (ANOVA F) and linear site AUC
    assert rep["combat"]["mean_anova_f"] < rep["raw"]["mean_anova_f"] * 0.25
    assert rep["combat"]["site_linear_auc"] < rep["raw"]["site_linear_auc"] - 0.15

    # 3. Both ComBat & CovBat reduce within-site residual covariance divergence, and
    # CovBat specifically reduces off-diagonal inter-site correlation divergence beyond ComBat
    assert rep["combat"]["covariance_frobenius_dist"] < rep["raw"]["covariance_frobenius_dist"]
    assert rep["covbat"]["covariance_frobenius_dist"] < rep["raw"]["covariance_frobenius_dist"]
    assert rep["covbat"]["correlation_frobenius_dist"] < rep["combat"]["correlation_frobenius_dist"]

    # 4. Linear site detectability collapses to near chance after CovBat, and RF detectability drops
    assert rep["covbat"]["site_linear_auc"] < 0.62
    assert rep["covbat"]["site_rf_auc"] < rep["raw"]["site_rf_auc"]

    # 5. Continuous defect_burden signal remains well-preserved after CovBat
    assert rep["covbat"]["defect_burden_r2"] >= 0.45
