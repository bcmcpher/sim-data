"""Tests for continuous tissue-aware T1w -> T2w / EPI contrast conversion (Options 1 + 2)."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage, stats

from ghibli.config import ContrastConversionConfig, StructuralSynthesisConfig
from ghibli.io.bids import discover_bids_scans, load_bids_scan
from ghibli.io.orientation import canonicalize_to_ras, compute_anatomical_priors
from ghibli.synthesis.contrast import invert_t1w_to_t2w
from ghibli.synthesis.modalities import synthesize_structural_modality
from ghibli.synthesis.tissue import estimate_tissue_pve, sample_tissue_mri_params


def test_invert_t1w_to_t2w_flips_contrast_keeps_air_dark_and_inverts_texture(
    bids_phantom_dir: Path,
) -> None:
    t1_scan = discover_bids_scans(bids_phantom_dir, modalities=["T1w"])[0]
    loaded = load_bids_scan(t1_scan)
    t1_data, affine, _ = canonicalize_to_ras(loaded.img)

    res_t2 = invert_t1w_to_t2w(
        t1_data,
        affine=affine,
        target="T2w",
        cfg=ContrastConversionConfig(n4_bias_correct=True),
        seed=42,
    )
    res_bold = invert_t1w_to_t2w(
        t1_data,
        affine=affine,
        target="bold_s0",
        cfg=ContrastConversionConfig(n4_bias_correct=True),
        seed=42,
    )

    assert res_t2.data.shape == t1_data.shape
    assert np.all(np.isfinite(res_t2.data))
    assert not res_t2.used_external_mask

    # Inside ICV: CSF (low T1) > GM (mid T1) > WM (high T1) after conversion
    icv = res_t2.icv_mask
    t1_icv = res_t2.t1_n4[icv]
    t2_icv = res_t2.data[icv]
    p20, p45, p65, p85 = np.percentile(t1_icv, [20.0, 45.0, 65.0, 85.0])
    csf_med = float(np.median(t2_icv[t1_icv <= p20]))
    gm_med = float(np.median(t2_icv[(t1_icv >= p45) & (t1_icv <= p65)]))
    wm_med = float(np.median(t2_icv[t1_icv >= p85]))
    assert csf_med > gm_med > wm_med

    # Background air & cortical bone stay strictly dark
    assert float(np.mean(res_t2.data[res_t2.bone_air_mask])) < 1.0

    # EPI BOLD S0 suppresses extracranial scalp/fat strictly more than clinical FSE T2w
    if np.any(res_t2.scalp_mask):
        assert float(np.median(res_bold.data[res_bold.scalp_mask])) < float(
            np.median(res_t2.data[res_t2.scalp_mask])
        )

    # High-frequency local micro-texture inside ICV is genuinely inverted relative to T1w
    hp_t1 = res_t2.t1_n4 - ndimage.gaussian_filter(res_t2.t1_n4, sigma=2.0)
    hp_t2 = res_t2.data - ndimage.gaussian_filter(res_t2.data, sigma=2.0)
    interior = ndimage.binary_erosion(icv, iterations=2)
    r_val, _ = stats.pearsonr(hp_t1[interior], hp_t2[interior])
    assert float(r_val) < -0.55


def test_invert_t1w_to_t2w_respects_external_brain_mask(
    bids_phantom_dir: Path, tmp_path: Path
) -> None:
    t1_scan = discover_bids_scans(bids_phantom_dir, modalities=["T1w"])[0]
    loaded = load_bids_scan(t1_scan)
    t1_data, affine, _ = canonicalize_to_ras(loaded.img)

    priors = compute_anatomical_priors(t1_data)
    ext_mask = ndimage.binary_erosion(priors.brain_mask, iterations=2).astype(np.uint8)
    mask_path = tmp_path / "sub-01_desc-brain_mask.nii.gz"
    nib.save(nib.Nifti1Image(ext_mask, affine), str(mask_path))

    res = invert_t1w_to_t2w(
        t1_data,
        affine=affine,
        target="dwi_b0",
        brain_mask_path=mask_path,
        seed=99,
    )
    assert res.used_external_mask is True
    assert np.array_equal(res.icv_mask, ext_mask > 0)
    assert float(np.mean(res.data[res.bone_air_mask])) < 1.0


def test_flair_polarity_aware_texture_preserves_csf_suppression_and_gm_gt_wm(
    bids_phantom_dir: Path,
) -> None:
    t1_scan = discover_bids_scans(bids_phantom_dir, modalities=["T1w"])[0]
    loaded = load_bids_scan(t1_scan)
    t1_data, affine, _ = canonicalize_to_ras(loaded.img)

    pve = estimate_tissue_pve(t1_data, target_affine=affine, seed=7)
    rng = np.random.default_rng(7)
    params = sample_tissue_mri_params(rng, jitter=0.0)
    flair, timing = synthesize_structural_modality(
        pve=pve,
        warped_t1=t1_data,
        warped_t2=None,
        modality="FLAIR",
        params=params,
        cfg=StructuralSynthesisConfig(preserve_seed_texture_weight=0.65, baseline_snr=90.0),
        rng=rng,
        device="cpu",
        affine=affine,
    )
    assert "InversionTime" in timing
    gm_pure = pve.gm > 0.6
    wm_pure = pve.wm > 0.6
    csf_pure = pve.csf > 0.6
    assert float(np.median(flair[gm_pure])) > float(np.median(flair[wm_pure]))
    assert float(np.median(flair[csf_pure])) < float(np.median(flair[gm_pure]))


def test_standalone_flair_pchip_conversion_nulls_csf_and_keeps_gm_above_wm(
    bids_phantom_dir: Path,
) -> None:
    t1_scan = discover_bids_scans(bids_phantom_dir, modalities=["T1w"])[0]
    loaded = load_bids_scan(t1_scan)
    t1_data, affine, _ = canonicalize_to_ras(loaded.img)

    res = invert_t1w_to_t2w(
        t1_data,
        affine=affine,
        target="FLAIR",
        cfg=ContrastConversionConfig(n4_bias_correct=True, brain_extraction="morphological"),
        seed=42,
    )
    assert res.data.shape == t1_data.shape
    assert np.all(np.isfinite(res.data))
    icv = res.icv_mask
    t1_icv = res.t1_n4[icv]
    fl_icv = res.data[icv]
    p15, p45, p65, p88 = np.percentile(t1_icv, [15.0, 45.0, 65.0, 88.0])
    csf_med = float(np.median(fl_icv[t1_icv <= p15]))
    gm_med = float(np.median(fl_icv[(t1_icv >= p45) & (t1_icv <= p65)]))
    wm_med = float(np.median(fl_icv[t1_icv >= p88]))
    assert gm_med > wm_med > csf_med
    assert csf_med < 0.45 * gm_med
