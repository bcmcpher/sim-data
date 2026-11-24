"""End-to-end zero-leakage blinded BIDS pipeline, TSV manifest, and decoupled IQM extraction tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sim_data.config import (
    ArtifactSpec,
    OutputConfig,
    SamplingPlanConfig,
    SimulationConfig,
    SiteProfileConfig,
)
from sim_data.eval.deterministic_qc import evaluate_deterministic_qc
from sim_data.iqm.extractor import extract_iqms_for_bids_dir
from sim_data.pipeline.engine import run_simulation_plan


def test_blinded_bids_zero_leakage_and_decoupled_iqms(
    bids_phantom_dir: Path, tmp_path: Path
) -> None:
    """Default generate produces strictly blinded BIDS files, clean JSON sidecars, and TSV manifest only."""
    cfg = SimulationConfig(
        global_seed=99,
        modalities=["T1w", "bold", "dwi"],
        sampling=SamplingPlanConfig(
            mode="combinatorial",
            include_clean_baseline=True,
            shuffle_run_order=True,
            samples_per_scan=4,
            min_artifacts=1,
            max_artifacts=2,
            severity_range=(0.25, 0.85),
            artifacts=[
                ArtifactSpec(name="motion_3d"),
                ArtifactSpec(name="rician_noise"),
                ArtifactSpec(name="bias_field"),
                ArtifactSpec(name="defacing_damage"),
                ArtifactSpec(name="motion_4d"),
                ArtifactSpec(name="nyquist_n2_ghost"),
                ArtifactSpec(name="dwi_slice_dropout"),
            ],
        ),
        sites=[
            SiteProfileConfig(site_id="SiteA", contrast_gamma=0.95, noise_sigma_rel=0.01),
            SiteProfileConfig(site_id="SiteB", contrast_gamma=1.15, noise_sigma_rel=0.025),
        ],
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

    out_dir = tmp_path / "blinded_bids"
    external_manifest = tmp_path / "ground_truth_manifest.tsv"
    res = run_simulation_plan(
        bids_dir=bids_phantom_dir,
        config=cfg,
        out_dir=out_dir,
        workers=2,
        device="cpu",
        manifest_out=external_manifest,
    )

    assert res.n_scans_generated == 24
    assert external_manifest.exists()
    assert not list(out_dir.rglob("*.parquet"))
    # Root BIDS files exist
    assert (out_dir / "dataset_description.json").exists()
    assert (out_dir / "participants.tsv").exists()
    assert (out_dir / ".bidsignore").exists()
    assert (out_dir / "sub-01" / "sub-01_scans.tsv").exists()

    df = pd.read_csv(external_manifest, sep="\t")
    assert set(df["modality"]) == {"T1w", "bold", "dwi"}
    assert "defect_burden" in df.columns
    assert "sev_motion_3d" in df.columns
    assert "site_gamma" in df.columns
    # No fabricated categorical QC labels or inline IQMs by default
    assert "qc_label" not in df.columns
    assert "proxy_qc_label" not in df.columns
    assert not any(c.startswith("iqm_") for c in df.columns)

    # Zero-leakage check on ALL filenames and JSON sidecars
    forbidden_tokens = [
        "clean",
        "sim",
        "comb",
        "dose",
        "motion",
        "noise",
        "bias",
        "deface",
        "ghost",
        "dropout",
    ]
    for _, row in df.iterrows():
        nifti_rel = str(row["nifti_path"])
        nifti_p = out_dir / nifti_rel
        assert nifti_p.exists()
        fname_lower = nifti_p.name.lower()
        assert "_run-" in fname_lower
        assert "_desc-site" in fname_lower
        for tok in forbidden_tokens:
            assert tok not in fname_lower, f"Leaked token '{tok}' in filename {nifti_p.name}"

        json_p = Path(str(nifti_p).replace(".nii.gz", ".json"))
        assert json_p.exists()
        sidecar = json.loads(json_p.read_text(encoding="utf-8"))
        assert "SimulationProvenance" not in sidecar
        sidecar_str = json.dumps(sidecar).lower()
        assert "defect_burden" not in sidecar_str
        assert "severity" not in sidecar_str

        if row["modality"] == "dwi":
            assert Path(str(nifti_p).replace(".nii.gz", ".bval")).exists()
            assert Path(str(nifti_p).replace(".nii.gz", ".bvec")).exists()

    # Now run decoupled, strictly blind post-hoc IQM extraction from the BIDS directory alone
    iqms_tsv = tmp_path / "blind_iqms.tsv"
    iqm_df = extract_iqms_for_bids_dir(out_dir, out_tsv=iqms_tsv, workers=2)
    assert iqms_tsv.exists()
    assert len(iqm_df) == 24
    assert set(iqm_df["scan_id"]) == set(df["scan_id"])

    # Deterministic benchmark joins blind IQMs onto ground-truth manifest cleanly
    eval_res = evaluate_deterministic_qc(
        df, out_dir=tmp_path / "qc_eval", iqms_df=iqm_df, modality="T1w"
    )
    assert eval_res.report_path.exists()
    assert eval_res.report["target_source"] == "defect_burden_binned"


def test_site_naming_none_and_opt_in_views_packets(
    bids_phantom_dir: Path, tmp_path: Path
) -> None:
    """Verify raw-BIDS `site_naming='none'` plus opt-in visual-QC views & packets."""
    cfg = SimulationConfig(
        global_seed=77,
        modalities=["T1w", "bold", "dwi"],
        sampling=SamplingPlanConfig(
            mode="combinatorial",
            include_clean_baseline=True,
            samples_per_scan=2,
            min_artifacts=1,
            max_artifacts=1,
            severity_range=(0.40, 0.80),
            artifacts=[
                ArtifactSpec(name="rician_noise"),
                ArtifactSpec(name="nyquist_n2_ghost"),
                ArtifactSpec(name="dwi_slice_dropout"),
            ],
        ),
        sites=[SiteProfileConfig(site_id="SiteA", contrast_gamma=0.95, noise_sigma_rel=0.01)],
        output=OutputConfig(
            save_nifti=True,
            write_bids_metadata=True,
            site_naming="none",
            extract_iqms=True,
            render_views=True,
            render_midline=True,
            render_enhanced=True,
            write_packets=True,
            include_proxy_qc_labels=True,
        ),
    )

    out_dir = tmp_path / "opt_in_out"
    res = run_simulation_plan(bids_dir=bids_phantom_dir, config=cfg, out_dir=out_dir, workers=2)
    df = pd.read_csv(res.manifest_path, sep="\t")
    assert len(df) == 12
    assert "proxy_qc_label" in df.columns

    desc = json.loads((out_dir / "dataset_description.json").read_text(encoding="utf-8"))
    assert desc["DatasetType"] == "raw"

    for mod in ("T1w", "bold", "dwi"):
        row = df[df["modality"] == mod].iloc[0]
        assert "_desc-" not in str(row["nifti_path"])
        assert "_run-" in str(row["nifti_path"])

        pkt = json.loads((out_dir / row["packet_path"]).read_text(encoding="utf-8"))
        assert set(pkt["views"]["midline"].keys()) == {"sagittal", "axial", "coronal"}
        assert "ortho_mosaic" in pkt["views"]["enhanced"]
        assert "bg_boost_mosaic" in pkt["views"]["enhanced"]
        if mod == "bold":
            assert (out_dir / pkt["views"]["enhanced"]["fmri_carpet_summary"]).exists()
        if mod == "dwi":
            assert (out_dir / pkt["views"]["enhanced"]["dwi_shell_dropout_summary"]).exists()
