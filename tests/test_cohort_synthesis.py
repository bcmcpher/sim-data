"""Tests for multimodal BIDS cohort synthesis from structural seeds (T1w + optional T2w)."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from sim_data.config import (
    ArtifactSpec,
    BoldSynthesisConfig,
    CohortSynthesisConfig,
    DiffeomorphicWarpConfig,
    DwiSynthesisConfig,
    OptionalTemplateConfig,
    OutputConfig,
    SamplingPlanConfig,
    SimulationConfig,
    SiteProfileConfig,
    StructuralSynthesisConfig,
)
from sim_data.io.bids import discover_bids_scans
from sim_data.io.orientation import compute_anatomical_priors
from sim_data.iqm.diffusion import extract_diffusion_iqms
from sim_data.iqm.extractor import extract_iqms_for_bids_dir
from sim_data.iqm.functional import extract_functional_iqms
from sim_data.phantoms.synthetic import generate_synthetic_bids_cohort
from sim_data.pipeline.engine import run_simulation_plan
from sim_data.synthesis.cohort import synthesize_cohort


@pytest.fixture(scope="module")
def three_seed_bids(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("three_seed_bids")
    return generate_synthetic_bids_cohort(
        out_dir=root,
        n_subjects=3,
        anat_shape=(44, 52, 44),
        epi_shape=(32, 36, 24),
        n_bold_trs=10,
        n_dwi_dirs=6,
        seed=909,
    )


def test_min_seed_subject_guardrail(bids_phantom_dir: Path, tmp_path: Path) -> None:
    """bids_phantom_dir has 2 subjects; default min_seed_subjects=3 must fail unless allow_few_seeds=True."""
    cfg = CohortSynthesisConfig(
        n_subjects=2,
        min_seed_subjects=3,
        allow_few_seeds=False,
        modalities=["T1w"],
    )
    with pytest.raises(ValueError, match="min_seed_subjects=3"):
        synthesize_cohort(bids_phantom_dir, cfg, out_dir=tmp_path / "fail_out")

    cfg.allow_few_seeds = True
    res = synthesize_cohort(bids_phantom_dir, cfg, out_dir=tmp_path / "ok_out")
    assert res.n_subjects == 2


def test_template_free_multimodal_cohort_physics_and_diversity(
    three_seed_bids: Path, tmp_path: Path
) -> None:
    cfg = CohortSynthesisConfig(
        global_seed=2026,
        n_subjects=3,
        min_seed_subjects=3,
        modalities=["T1w", "T2w", "FLAIR", "bold", "dwi"],
        warp=DiffeomorphicWarpConfig(strength=0.6, smooth_sigma_mm=14.0, flip_lr_prob=0.5),
        structural=StructuralSynthesisConfig(
            preserve_seed_texture_weight=0.55, tissue_param_jitter=0.06, baseline_snr=85.0
        ),
        bold=BoldSynthesisConfig(
            voxel_size_mm=(3.0, 3.0, 3.0), n_trs=24, tr_sec=2.0, baseline_snr=70.0
        ),
        dwi=DwiSynthesisConfig(
            voxel_size_mm=(3.0, 3.0, 3.0),
            n_b0=2,
            shells=[(1000.0, 8), (2000.0, 8)],
            crossing_fiber_fraction=0.30,
            baseline_snr=60.0,
        ),
    )

    out_dir = tmp_path / "synth_cohort"
    res = synthesize_cohort(three_seed_bids, cfg, out_dir=out_dir, workers=2, device="cpu")

    assert res.n_subjects == 3
    assert res.n_seed_subjects == 3
    assert (out_dir / "dataset_description.json").exists()
    assert (out_dir / "participants.tsv").exists()
    assert (out_dir / ".bidsignore").exists()
    assert res.manifest_path.exists()

    df = pd.read_csv(res.manifest_path, sep="\t")
    assert len(df) == 3
    # Diffeomorphisms must remain topology-preserving (positive Jacobian determinant)
    assert (df["warp_min_jac_det"] > 0.1).all()
    assert (df["warp_max_disp_mm"] > 0.5).all()

    # Check morphological diversity across synthetic subjects
    t1_1 = np.asarray(nib.load(str(out_dir / "sub-001/anat/sub-001_T1w.nii.gz")).get_fdata())
    t1_2 = np.asarray(nib.load(str(out_dir / "sub-002/anat/sub-002_T1w.nii.gz")).get_fdata())
    assert float(np.mean(np.abs(t1_1 - t1_2))) > 15.0

    # Check realistic structural contrast ordering (T1w vs T2w vs FLAIR) inside head foreground
    t2_1 = np.asarray(nib.load(str(out_dir / "sub-001/anat/sub-001_T2w.nii.gz")).get_fdata())
    fl_1 = np.asarray(nib.load(str(out_dir / "sub-001/anat/sub-001_FLAIR.nii.gz")).get_fdata())
    fg = compute_anatomical_priors(t1_1).brain_mask
    # Inside brain foreground: brightest T2w voxels (CSF) are strongly suppressed on FLAIR vs GM
    csf_like = fg & (t2_1 > np.percentile(t2_1[fg], 93.0))
    gm_like = fg & (t2_1 > np.percentile(t2_1[fg], 45.0)) & (t2_1 < np.percentile(t2_1[fg], 75.0))
    assert float(np.median(fl_1[csf_like])) < float(np.median(fl_1[gm_like]))

    # Check 4D BOLD realism via blind NiPreps IQMs
    bold_img = nib.load(str(out_dir / "sub-001/func/sub-001_task-rest_bold.nii.gz"))
    bold_4d = np.asarray(bold_img.get_fdata(), dtype=np.float32)
    assert bold_4d.shape[3] == 24
    bold_zooms = tuple(float(z) for z in bold_img.header.get_zooms()[:3])
    bold_iqm = extract_functional_iqms(bold_4d, (bold_zooms[0], bold_zooms[1], bold_zooms[2]))
    assert bold_iqm["tsnr"] > 18.0
    assert bold_iqm["fd_mean"] < 0.20

    # Check 4D DWI multi-shell attenuation & QSIPrep IQMs
    dwi_img = nib.load(str(out_dir / "sub-001/dwi/sub-001_dwi.nii.gz"))
    dwi_4d = np.asarray(dwi_img.get_fdata(), dtype=np.float32)
    bvals = np.loadtxt(out_dir / "sub-001/dwi/sub-001_dwi.bval")
    bvecs = np.loadtxt(out_dir / "sub-001/dwi/sub-001_dwi.bvec")
    assert dwi_4d.shape[3] == 2 + 8 + 8
    assert bvecs.shape == (3, 18)

    b0_mean = float(np.mean(dwi_4d[..., bvals == 0]))
    b1000_mean = float(np.mean(dwi_4d[..., bvals == 1000]))
    b2000_mean = float(np.mean(dwi_4d[..., bvals == 2000]))
    assert b0_mean > b1000_mean > b2000_mean

    dwi_zooms = tuple(float(z) for z in dwi_img.header.get_zooms()[:3])
    dwi_iqm = extract_diffusion_iqms(
        dwi_4d, (dwi_zooms[0], dwi_zooms[1], dwi_zooms[2]), bvals=bvals, bvecs=bvecs
    )
    assert dwi_iqm["raw_neighbor_corr"] > 0.70
    assert dwi_iqm["raw_num_bad_slices"] == 0.0

    # Verify synthesized cohort plugs directly into blinded `sim-data generate` -> `extract-iqms`
    sim_cfg = SimulationConfig(
        global_seed=42,
        modalities=["T1w", "bold", "dwi"],
        sampling=SamplingPlanConfig(
            mode="combinatorial",
            include_clean_baseline=True,
            samples_per_scan=2,
            min_artifacts=1,
            max_artifacts=1,
            artifacts=[
                ArtifactSpec(name="rician_noise"),
                ArtifactSpec(name="nyquist_n2_ghost"),
                ArtifactSpec(name="dwi_slice_dropout"),
            ],
        ),
        sites=[SiteProfileConfig(site_id="SiteA", contrast_gamma=0.95, noise_sigma_rel=0.01)],
        output=OutputConfig(site_naming="desc"),
    )
    gen_out = tmp_path / "generated_from_synth"
    gen_res = run_simulation_plan(out_dir, sim_cfg, gen_out, workers=2)
    assert gen_res.n_scans_generated == 3 * 3 * 2
    iqms_df = extract_iqms_for_bids_dir(gen_out, out_tsv=tmp_path / "gen_iqms.tsv", workers=2)
    assert len(iqms_df) == gen_res.n_scans_generated


def test_optional_templates_pve_rsn_dti(three_seed_bids: Path, tmp_path: Path) -> None:
    """Verify optional external PVE, RSN parcellation, and DTI eigenvector templates load & warp cleanly."""
    t1_scan = discover_bids_scans(three_seed_bids, modalities=["T1w"])[0]
    t1_img = nib.load(str(t1_scan.nifti_path))
    shape = t1_img.shape[:3]
    aff = t1_img.affine

    # Build simple synthetic template NIfTIs in seed space
    pve_4d = np.zeros((*shape, 4), dtype=np.float32)
    pve_4d[8:-8, 8:-8, 8:-8, 2] = 0.7  # WM
    pve_4d[8:-8, 8:-8, 8:-8, 1] = 0.25  # GM
    pve_4d[8:-8, 8:-8, 8:-8, 0] = 0.05  # CSF
    pve_path = tmp_path / "tpl_pve.nii.gz"
    nib.save(nib.Nifti1Image(pve_4d, aff), str(pve_path))

    rsn_3d = np.zeros(shape, dtype=np.int16)
    rsn_3d[: shape[0] // 2, :, :] = 1
    rsn_3d[shape[0] // 2 :, :, :] = 2
    rsn_path = tmp_path / "tpl_rsn.nii.gz"
    nib.save(nib.Nifti1Image(rsn_3d, aff), str(rsn_path))

    dti_vec = np.zeros((*shape, 3), dtype=np.float32)
    dti_vec[..., 0] = 1.0  # pure R-L fiber template
    dti_path = tmp_path / "tpl_dti_vec.nii.gz"
    nib.save(nib.Nifti1Image(dti_vec, aff), str(dti_path))

    cfg = CohortSynthesisConfig(
        global_seed=11,
        n_subjects=2,
        min_seed_subjects=3,
        modalities=["T1w", "bold", "dwi"],
        warp=DiffeomorphicWarpConfig(strength=0.35),
        templates=OptionalTemplateConfig(
            tissue_pve_nifti=pve_path,
            rsn_parcellation_nifti=rsn_path,
            dti_eigenvector_nifti=dti_path,
        ),
        bold=BoldSynthesisConfig(voxel_size_mm=(3.5, 3.5, 3.5), n_trs=16),
        dwi=DwiSynthesisConfig(voxel_size_mm=(3.5, 3.5, 3.5), n_b0=1, shells=[(1000.0, 6)]),
    )

    out_dir = tmp_path / "synth_with_templates"
    res = synthesize_cohort(three_seed_bids, cfg, out_dir=out_dir, workers=1, device="cpu")
    assert res.n_subjects == 2
    assert (out_dir / "sub-001/anat/sub-001_T1w.nii.gz").exists()
    assert (out_dir / "sub-001/func/sub-001_task-rest_bold.nii.gz").exists()
    assert (out_dir / "sub-001/dwi/sub-001_dwi.nii.gz").exists()


def test_preserve_real_modalities_completes_missing_without_warp(
    bids_phantom_dir: Path, tmp_path: Path
) -> None:
    """Real subjects keep existing QC-passing modalities untouched while missing modalities are completed."""
    import json
    import shutil

    # Build a 1-subject partial real BIDS dataset with real T1w + real bold only (missing T2w, FLAIR, dwi)
    partial_dir = tmp_path / "partial_real_bids"
    (partial_dir / "sub-01" / "anat").mkdir(parents=True)
    (partial_dir / "sub-01" / "func").mkdir(parents=True)
    shutil.copy2(
        bids_phantom_dir / "sub-01/anat/sub-01_T1w.nii.gz",
        partial_dir / "sub-01/anat/sub-01_T1w.nii.gz",
    )
    shutil.copy2(
        bids_phantom_dir / "sub-01/anat/sub-01_T1w.json",
        partial_dir / "sub-01/anat/sub-01_T1w.json",
    )
    shutil.copy2(
        bids_phantom_dir / "sub-01/func/sub-01_task-rest_bold.nii.gz",
        partial_dir / "sub-01/func/sub-01_task-rest_bold.nii.gz",
    )
    shutil.copy2(
        bids_phantom_dir / "sub-01/func/sub-01_task-rest_bold.json",
        partial_dir / "sub-01/func/sub-01_task-rest_bold.json",
    )

    cfg = CohortSynthesisConfig(
        global_seed=77,
        preserve_real_modalities=True,
        modalities=["T1w", "T2w", "FLAIR", "bold", "dwi"],
        bold=BoldSynthesisConfig(voxel_size_mm=(3.5, 3.5, 3.5), n_trs=12),
        dwi=DwiSynthesisConfig(voxel_size_mm=(3.5, 3.5, 3.5), n_b0=1, shells=[(1000.0, 6)]),
    )
    out_dir = tmp_path / "completed_real_bids"
    res = synthesize_cohort(partial_dir, cfg, out_dir=out_dir, workers=1, device="cpu")
    assert res.n_subjects == 1
    assert res.rows[0]["participant_id"] == "sub-01"
    assert res.rows[0]["warp_max_disp_mm"] == 0.0

    prov = json.loads(res.rows[0]["modality_provenance_json"])
    assert prov["T1w"] == "real"
    assert prov["bold"] == "real"
    assert prov["T2w"] == "contrast_converted_from_t1w"
    assert prov["dwi"] == "synthesized_from_t1w_contrast"

    # Real T1w and real BOLD must be preserved bit-for-bit
    orig_t1 = np.asarray(nib.load(str(partial_dir / "sub-01/anat/sub-01_T1w.nii.gz")).get_fdata())
    out_t1 = np.asarray(nib.load(str(out_dir / "sub-01/anat/sub-01_T1w.nii.gz")).get_fdata())
    assert np.array_equal(orig_t1, out_t1)

    orig_bold = np.asarray(
        nib.load(str(partial_dir / "sub-01/func/sub-01_task-rest_bold.nii.gz")).get_fdata()
    )
    out_bold = np.asarray(
        nib.load(str(out_dir / "sub-01/func/sub-01_task-rest_bold.nii.gz")).get_fdata()
    )
    assert np.array_equal(orig_bold, out_bold)

    # Synthesized missing modalities exist and have valid shapes
    assert (out_dir / "sub-01/anat/sub-01_T2w.nii.gz").exists()
    assert (out_dir / "sub-01/anat/sub-01_FLAIR.nii.gz").exists()
    assert (out_dir / "sub-01/dwi/sub-01_dwi.nii.gz").exists()

