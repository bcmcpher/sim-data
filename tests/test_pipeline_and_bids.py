"""End-to-end BIDS pipeline, manifest, visual-QC packet, and deterministic QC baseline tests."""

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
from sim_data.pipeline.engine import run_simulation_plan


def test_end_to_end_multimodal_pipeline_packets_and_deterministic_eval(
    bids_phantom_dir: Path, tmp_path: Path
) -> None:
    cfg = SimulationConfig(
        global_seed=99,
        modalities=["T1w", "bold", "dwi"],
        sampling=SamplingPlanConfig(
            mode="combinatorial",
            include_clean_baseline=True,
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
            extract_iqms=True,
            render_views=True,
            render_midline=True,
            render_enhanced=True,
            image_format="png",
            write_packets=True,
            manifest_formats=["parquet", "tsv"],
        ),
    )

    out_dir = tmp_path / "sim_out"
    res = run_simulation_plan(
        bids_dir=bids_phantom_dir,
        config=cfg,
        out_dir=out_dir,
        workers=2,
        device="cpu",
    )

    # 2 subjects x 3 modalities x 4 samples = 24 simulated scans
    assert res.n_scans_generated == 24
    assert (out_dir / "manifest.parquet").exists()
    assert (out_dir / "manifest.tsv").exists()

    df = pd.read_parquet(out_dir / "manifest.parquet")
    assert set(df["modality"]) == {"T1w", "bold", "dwi"}
    assert set(df["qc_label"]).issubset({"pass", "borderline", "fail"})

    # Verify NIfTI, JSON provenance, .bval/.bvec, midline & enhanced views, and packet.json
    for mod in ("T1w", "bold", "dwi"):
        row = df[df["modality"] == mod].iloc[0]
        nifti_p = out_dir / row["nifti_path"]
        assert nifti_p.exists()
        json_p = Path(str(nifti_p).replace(".nii.gz", ".json"))
        assert json_p.exists()
        sidecar = json.loads(json_p.read_text(encoding="utf-8"))
        assert "SimulationProvenance" in sidecar

        if mod == "dwi":
            assert Path(str(nifti_p).replace(".nii.gz", ".bval")).exists()
            assert Path(str(nifti_p).replace(".nii.gz", ".bvec")).exists()

        pkt_p = out_dir / row["packet_path"]
        assert pkt_p.exists()
        pkt = json.loads(pkt_p.read_text(encoding="utf-8"))
        assert pkt["packet_version"] == "1.1"
        assert set(pkt["views"]["midline"].keys()) == {"sagittal", "axial", "coronal"}
        for rel_img in pkt["views"]["midline"].values():
            assert (out_dir / rel_img).exists()
            assert (out_dir / rel_img).stat().st_size > 200
        assert "ortho_mosaic" in pkt["views"]["enhanced"]
        assert "bg_boost_mosaic" in pkt["views"]["enhanced"]
        if mod == "bold":
            assert "fmri_carpet_summary" in pkt["views"]["enhanced"]
            assert (out_dir / pkt["views"]["enhanced"]["fmri_carpet_summary"]).exists()
        if mod == "dwi":
            assert "dwi_shell_dropout_summary" in pkt["views"]["enhanced"]
            assert (out_dir / pkt["views"]["enhanced"]["dwi_shell_dropout_summary"]).exists()

    # Run deterministic CV Logistic Regression benchmark & enrich packet.json
    eval_res = evaluate_deterministic_qc(
        df, out_dir=out_dir / "qc_eval", modality="T1w", update_packets_root=out_dir
    )
    assert eval_res.report_path.exists()
    assert eval_res.report["updated_packets"] == len(df[df["modality"] == "T1w"])

    t1_pkt = json.loads((out_dir / df[df["modality"] == "T1w"].iloc[0]["packet_path"]).read_text())
    assert "deterministic_priors" in t1_pkt
    assert set(t1_pkt["deterministic_priors"]["probabilities"].keys()) == {
        "pass",
        "borderline",
        "fail",
    }
