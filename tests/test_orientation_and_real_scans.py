"""Tests for real-scan non-RAS orientation remapping, passthrough baseline, and noise calibration."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from ghibli.artifacts.base import ScanContext
from ghibli.artifacts.registry import create_artifact
from ghibli.config import (
    ArtifactSpec,
    OutputConfig,
    SamplingPlanConfig,
    SimulationConfig,
)
from ghibli.io.bids import discover_bids_scans, load_bids_scan
from ghibli.io.orientation import (
    canonicalize_to_ras,
    canonicalize_to_ras_with_transform,
    compute_anatomical_priors,
    estimate_background_noise_sigma,
    reorient_bvecs_to_ras,
    resolve_encoding_axes,
)
from ghibli.pipeline.engine import run_simulation_plan


def test_non_ras_orientation_remaps_encoding_axes_and_bvecs() -> None:
    """PIR-oriented real NIfTI must remap sidecar PE/SL axes and voxel-space bvecs into RAS+."""
    # PIR affine: axis 0 -> -A (-y), axis 1 -> -S (-z), axis 2 -> +R (+x)
    pir_affine = np.array(
        [
            [0.0, 0.0, 2.0, -20.0],
            [-2.0, 0.0, 0.0, 20.0],
            [0.0, -2.0, 0.0, 20.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    vol = np.zeros((12, 14, 16), dtype=np.float32)
    vol[3:9, 4:10, 5:11] = 500.0
    img = nib.Nifti1Image(vol, pir_affine)

    ras_data, _ras_aff, _zooms, ornt = canonicalize_to_ras_with_transform(img)
    assert ras_data.shape == (16, 12, 14)

    # Original i (axis 0) -> RAS axis 1 (A-P); original j (axis 1) -> RAS axis 2 (S-I)
    pe, ro, sl = resolve_encoding_axes(
        {"PhaseEncodingDirection": "i", "SliceEncodingDirection": "j"},
        modality="dwi",
        ornt_transform=ornt,
    )
    assert (pe, ro, sl) == (1, 0, 2)

    bvecs_pir = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    bvecs_ras = reorient_bvecs_to_ras(bvecs_pir, ornt)
    expected = np.array(
        [
            [0.0, 0.0, 1.0],  # +k -> +R
            [-1.0, 0.0, 0.0],  # +i -> -A
            [0.0, -1.0, 0.0],  # +j -> -S
        ],
        dtype=np.float32,
    )
    assert np.allclose(bvecs_ras, expected, atol=1e-6)


def test_passthrough_site_preserves_exact_real_baseline_at_zero_severity(
    bids_phantom_dir: Path, tmp_path: Path
) -> None:
    """With sites=[] in dose_response mode, NativeSite(passthrough=True) leaves s=0 bit-for-bit identical."""
    cfg = SimulationConfig(
        global_seed=12,
        modalities=["T1w"],
        sampling=SamplingPlanConfig(
            mode="dose_response",
            include_clean_baseline=True,
            coupled_trajectories=True,
            dose_severities=[0.0, 0.40],
            artifacts=[ArtifactSpec(name="rician_noise")],
        ),
        sites=[],
        output=OutputConfig(naming_scheme="acq_blinded"),
    )
    out_dir = tmp_path / "passthrough_out"
    res = run_simulation_plan(bids_phantom_dir, cfg, out_dir=out_dir, workers=1, device="cpu")

    base_rows = [r for r in res.rows if r["target_artifact"] == "baseline"]
    assert len(base_rows) == 2  # 2 subjects, once per scan
    for row in base_rows:
        assert row["site_passthrough"] is True
        src_img = nib.load(str(row["source_nifti"]))
        src_data, _, _ = canonicalize_to_ras(src_img)  # type: ignore[arg-type]
        out_img = nib.load(str(out_dir / row["nifti_path"]))
        out_data = np.asarray(out_img.get_fdata(), dtype=np.float32)
        assert np.allclose(out_data, src_data, atol=1e-5)


def test_excess_variance_noise_calibration_subtracts_baseline_noise(
    bids_phantom_dir: Path,
) -> None:
    t1_scan = discover_bids_scans(bids_phantom_dir, modalities=["T1w"])[0]
    loaded = load_bids_scan(t1_scan)
    data, affine, zooms = canonicalize_to_ras(loaded.img)
    priors = compute_anatomical_priors(data)

    # Inject substantial native noise sigma_0 to simulate a noisier real clinical scan
    rng0 = np.random.default_rng(5)
    n1 = rng0.normal(0.0, 45.0, size=data.shape).astype(np.float32)
    n2 = rng0.normal(0.0, 45.0, size=data.shape).astype(np.float32)
    noisy_real = np.sqrt((data + n1) ** 2 + n2**2).astype(np.float32)
    priors_noisy = compute_anatomical_priors(noisy_real)
    sigma_0 = estimate_background_noise_sigma(noisy_real, priors_noisy.bg_mask, modality="T1w")
    assert 30.0 < sigma_0 < 60.0

    def _run(mode: str) -> dict[str, object]:
        ctx = ScanContext(
            data=noisy_real.copy(),
            affine=affine.copy(),
            zooms=zooms,
            modality="T1w",
            pe_axis=1,
            readout_axis=0,
            slice_axis=2,
            priors=priors,
            rng=np.random.default_rng(42),
            seed=42,
            device="cpu",
            baseline_noise_sigma=sigma_0,
            noise_calibration=mode,  # type: ignore[arg-type]
        )
        return create_artifact("rician_noise", severity=0.45).apply(ctx).params

    p_excess = _run("excess_variance")
    p_add = _run("additive")
    assert float(p_excess["sigma_abs"]) < float(p_add["sigma_abs"])  # type: ignore[arg-type]
    expected_eff = float(
        np.sqrt(max(float(p_add["sigma_target_abs"]) ** 2 - sigma_0**2, 0.0))  # type: ignore[arg-type]
    )
    assert abs(float(p_excess["sigma_abs"]) - expected_eff) < 1e-2  # type: ignore[arg-type]
