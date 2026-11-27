"""Step 02: Convert the staged T1w into every supported contrast & modality.

Tests both modality conversion code paths in ghibli:
1. Standalone `convert-contrast` (`invert_t1w_to_t2w`) for all 3 targets: `T2w`, `bold_s0`, `dwi_b0`,
   rendering the internal N4ITK + ICV/scalp/bone compartment separation so masking gaps are visible.
2. Full multimodal completion via `synthesize_cohort(..., preserve_real_modalities=True)` keeping
   the input T1w bit-for-bit untouched while synthesizing `T2w`, `FLAIR`, 4D `bold`, and 4D `dwi`.
3. Blind NiPreps IQM extraction (`extract_iqms_for_bids_dir`) and `render_scan_views` across all 5
   modalities, plus an annotated visual overview figure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from dipy.io.gradients import read_bvals_bvecs

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

from ghibli.config import (  # noqa: E402
    BoldSynthesisConfig,
    CohortSynthesisConfig,
    ContrastConversionConfig,
    DwiSynthesisConfig,
)
from ghibli.io.bids import discover_bids_scans  # noqa: E402
from ghibli.io.orientation import compute_anatomical_priors  # noqa: E402
from ghibli.iqm.extractor import extract_iqms_for_bids_dir  # noqa: E402
from ghibli.synthesis.cohort import synthesize_cohort  # noqa: E402
from ghibli.synthesis.contrast import ContrastTarget, invert_t1w_to_t2w  # noqa: E402
from ghibli.views.mosaic import render_scan_views  # noqa: E402


def _render_compartment_figure(
    t1_n4: np.ndarray,
    icv_mask: np.ndarray,
    scalp_mask: np.ndarray,
    bone_air_mask: np.ndarray,
    converted: dict[str, np.ndarray],
    out_path: Path,
    mask_method: str,
) -> Path:
    """Visualize N4-corrected T1w, ICV/scalp/bone-air compartment masks, and all 4 converted contrasts."""
    # Anchor slices to ICV midpoint so axial slice shows mid-ventricles/cortex instead of neck/orbits
    if np.any(icv_mask):
        coords = np.argwhere(icv_mask)
        mid_x = int((coords[:, 0].min() + coords[:, 0].max()) // 2)
        mid_z = int((coords[:, 2].min() + coords[:, 2].max()) // 2)
    else:
        mid_x, _mid_y, mid_z = common.midline_indices(t1_n4)

    # RGB compartment overlay: green=ICV, orange=scalp/fat, dark blue=bone/air
    comp_rgb = np.zeros((*t1_n4.shape, 3), dtype=np.float32)
    comp_rgb[icv_mask] = [0.20, 0.80, 0.45]
    comp_rgb[scalp_mask] = [0.95, 0.55, 0.15]
    comp_rgb[bone_air_mask] = [0.10, 0.14, 0.28]

    panels: list[tuple[str, np.ndarray, bool]] = [
        ("T1w (N4-corrected)", t1_n4, False),
        (f"Compartments\n(mask={mask_method})", comp_rgb, True),
        ("Converted T2w", converted["T2w"], False),
        ("Converted FLAIR", converted["FLAIR"], False),
        ("bold_s0 (skull-free)", converted["bold_s0"], False),
        ("dwi_b0 (skull-free)", converted["dwi_b0"], False),
    ]

    fig, axes = plt.subplots(2, len(panels), figsize=(17.5, 6.4))
    for col, (title, arr, is_rgb) in enumerate(panels):
        if is_rgb:
            ax_sl = np.flipud(np.transpose(arr[:, :, mid_z, :], (1, 0, 2)))
            sag_sl = np.flipud(np.transpose(arr[mid_x, :, :, :], (1, 0, 2)))
            axes[0, col].imshow(ax_sl)
            axes[1, col].imshow(sag_sl)
        else:
            vmin, vmax = common.compute_window(arr)
            axes[0, col].imshow(
                common.orient_2d(common.normalize_slice(arr[:, :, mid_z], vmin, vmax)),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            axes[1, col].imshow(
                common.orient_2d(common.normalize_slice(arr[mid_x, :, :], vmin, vmax)),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
        axes[0, col].set_title(title, fontsize=9)
        axes[0, col].axis("off")
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Axial (ICV mid-Z)", fontsize=9)
    axes[1, 0].set_ylabel("Sagittal (ICV mid-X)", fontsize=9)
    fig.suptitle(
        f"Step 02a: Standalone convert-contrast (mask={mask_method} | green=ICV, orange=scalp/fat, navy=bone/air)",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    return common.save_figure(fig, out_path)


def _render_modalities_overview(
    mod_volumes: dict[str, tuple[np.ndarray, str]],
    out_path: Path,
    icv_mask_3d: np.ndarray | None = None,
) -> Path:
    """Render 3-plane midline comparison across all 5 completed modalities (+ DWI high-b)."""
    ordered_keys = [k for k in ["T1w", "T2w", "FLAIR", "bold", "dwi_b0", "dwi_highb"] if k in mod_volumes]
    fig, axes = plt.subplots(3, len(ordered_keys), figsize=(3.1 * len(ordered_keys), 8.8))
    if len(ordered_keys) == 1:
        axes = axes[:, None]

    icv_mid: tuple[int, int, int] | None = None
    if icv_mask_3d is not None and np.any(icv_mask_3d):
        c = np.argwhere(icv_mask_3d)
        icv_mid = (
            int((c[:, 0].min() + c[:, 0].max()) // 2),
            int((c[:, 1].min() + c[:, 1].max()) // 2),
            int((c[:, 2].min() + c[:, 2].max()) // 2),
        )

    for col, key in enumerate(ordered_keys):
        vol, subtitle = mod_volumes[key]
        if icv_mid is not None and icv_mask_3d is not None and vol.shape == icv_mask_3d.shape:
            mx, my, mz = icv_mid
        else:
            mx, my, mz = common.midline_indices(vol)
        vmin, vmax = common.compute_window(vol)
        slices = [
            common.orient_2d(common.normalize_slice(vol[:, :, mz], vmin, vmax)),
            common.orient_2d(common.normalize_slice(vol[:, my, :], vmin, vmax)),
            common.orient_2d(common.normalize_slice(vol[mx, :, :], vmin, vmax)),
        ]
        for row, sl in enumerate(slices):
            axes[row, col].imshow(sl, cmap="gray", vmin=0.0, vmax=1.0)
            axes[row, col].axis("off")
        axes[0, col].set_title(f"{key}\n{subtitle}", fontsize=8.2)

    for row, plane in enumerate(["Axial", "Coronal", "Sagittal"]):
        axes[row, 0].text(
            -0.08,
            0.5,
            plane,
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=9,
            fontweight="bold",
        )

    fig.suptitle(
        "Step 02b: Completed Multimodal Subject (Real T1w + Synthesized T2w, FLAIR, 4D BOLD, 4D DWI)",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    return common.save_figure(fig, out_path)


def run_step(
    out_dir: Path = common.DEFAULT_OUT_DIR,
    workers: int = 4,
    device: str = "auto",
    full_4d: bool = False,
    brain_mask: Path | None = None,
) -> dict[str, Any]:
    """Run standalone contrast conversion + full 5-modality synthesis + blind IQM extraction."""
    out_root = Path(out_dir).resolve()
    seed_diag_path = out_root / "00_seed_diagnostics.json"
    if not seed_diag_path.exists():
        raise FileNotFoundError(
            f"Run Step 01 first ({seed_diag_path} not found)."
        )
    seed_diag = common.read_json(seed_diag_path)
    seed_bids = Path(seed_diag["seed_bids_dir"])
    staged_t1w = Path(seed_diag["staged_t1w"])
    eff_mask = brain_mask or (
        Path(seed_diag["external_brain_mask"]) if seed_diag.get("external_brain_mask") else None
    )
    checks: list[common.CheckResult] = []

    with common.StepTimer() as timer:
        # --- 2a. Standalone convert-contrast for all 4 targets (T2w, FLAIR, bold_s0, dwi_b0) ---
        t1_ras, t1_affine, _t1_zooms = common.load_ras_volume(staged_t1w)
        contrast_dir = out_root / "01_contrast_standalone"
        contrast_dir.mkdir(parents=True, exist_ok=True)
        contrast_cfg = ContrastConversionConfig(n4_bias_correct=True)

        converted_arrays: dict[str, np.ndarray] = {}
        contrast_meta: dict[str, Any] = {}
        first_res = None
        cached_icv_mask: np.ndarray | None = None
        for target_str in ("T2w", "FLAIR", "bold_s0", "dwi_b0"):
            res = invert_t1w_to_t2w(
                t1_data=t1_ras,
                affine=t1_affine,
                target=cast(ContrastTarget, target_str),
                cfg=contrast_cfg,
                brain_mask_path=eff_mask,
                brain_mask_array=cached_icv_mask,
                seed=42,
            )
            if first_res is None:
                first_res = res
                cached_icv_mask = res.icv_mask
            converted_arrays[target_str] = res.data
            nib.save(
                nib.Nifti1Image(res.data, t1_affine),
                str(contrast_dir / f"converted_{target_str}.nii.gz"),
            )
            icv_frac = round(float(res.icv_mask.mean()), 4)
            scalp_frac = round(float(res.scalp_mask.mean()), 4)
            contrast_meta[target_str] = {
                "backend": res.backend,
                "mask_method": first_res.mask_method,
                "used_external_mask": first_res.used_external_mask,
                "tissue_anchors_t1": res.tissue_anchors_t1,
                "tissue_targets_out": res.tissue_targets_out,
                "icv_fraction": icv_frac,
                "scalp_fraction": scalp_frac,
            }

        assert first_res is not None
        # Preserve the original extraction method name when reusing Step 01's cached mask
        eff_mask_method = (
            str(seed_diag.get("brain_mask_method", first_res.mask_method))
            if (brain_mask is None and first_res.mask_method == "external")
            else first_res.mask_method
        )
        # Persist the extracted ICV mask for downstream visual slicing and inspection
        icv_mask_path = contrast_dir / "extracted_icv_mask.nii.gz"
        nib.save(nib.Nifti1Image(first_res.icv_mask.astype(np.uint8), t1_affine), str(icv_mask_path))

        fig_comp = _render_compartment_figure(
            t1_n4=first_res.t1_n4,
            icv_mask=first_res.icv_mask,
            scalp_mask=first_res.scalp_mask,
            bone_air_mask=first_res.bone_air_mask,
            converted=converted_arrays,
            out_path=out_root / "figures" / "02a_contrast_compartments.png",
            mask_method=eff_mask_method,
        )

        scalp_ratio = float(first_res.scalp_mask.sum()) / max(1.0, float(first_res.icv_mask.sum()))
        good_mask = first_res.used_external_mask or eff_mask_method == "evac" or scalp_ratio >= 0.12
        checks.append(
            common.CheckResult(
                step="02_convert_modalities",
                name="icv_vs_scalp_separation",
                status="pass" if good_mask else "warn",
                metric=f"method={eff_mask_method}, ICV={first_res.icv_mask.mean():.3f}, scalp={first_res.scalp_mask.mean():.3f} (scalp/ICV={scalp_ratio:.3f})",
                detail=(
                    f"Intracranial brain mask ({eff_mask_method}) separated brain parenchyma cleanly from cortical skull bone and extracranial scalp/neck."
                    if good_mask
                    else "Morphological fallback classified only a thin rim as scalp; supply --brain-mask or enable EVAC+ brain extraction."
                ),
            )
        )

        # --- 2b. Full multimodal completion via synthesize-cohort --preserve-real ---
        multimodal_dir = out_root / "01_multimodal_bids"
        bold_cfg = (
            BoldSynthesisConfig(voxel_size_mm=(3.0, 3.0, 3.0), n_trs=60)
            if full_4d
            else BoldSynthesisConfig(voxel_size_mm=(3.0, 3.0, 3.0), n_trs=30)
        )
        dwi_cfg = (
            DwiSynthesisConfig(voxel_size_mm=(2.5, 2.5, 2.5), n_b0=3, shells=[(1000.0, 20), (2000.0, 20)])
            if full_4d
            else DwiSynthesisConfig(voxel_size_mm=(2.5, 2.5, 2.5), n_b0=2, shells=[(1000.0, 12), (2000.0, 12)])
        )
        synth_cfg = CohortSynthesisConfig(
            global_seed=1024,
            n_subjects=1,
            preserve_real_modalities=True,
            modalities=["T1w", "T2w", "FLAIR", "bold", "dwi"],
            bold=bold_cfg,
            dwi=dwi_cfg,
        )
        if eff_mask is not None:
            synth_cfg.templates.brain_mask_nifti = eff_mask
        else:
            # Reuse the freshly extracted ICV mask from Step 2a so EVAC+ only runs once
            synth_cfg.templates.brain_mask_nifti = icv_mask_path

        synth_res = synthesize_cohort(
            seed_bids_dir=seed_bids,
            config=synth_cfg,
            out_dir=multimodal_dir,
            workers=1,
            device=device,
        )

        # Verify all 5 modalities exist, T1w is untouched, and extract blind IQMs
        scans = discover_bids_scans(multimodal_dir)
        by_mod = {s.modality: s for s in scans}
        iqm_tsv = multimodal_dir / "baseline_iqms.tsv"
        iqms_df = extract_iqms_for_bids_dir(
            bids_dir=multimodal_dir, out_tsv=iqm_tsv, workers=min(workers, max(1, len(scans)))
        )

        # Render built-in views for BOLD carpet & DWI dropout summary, plus our unified overview
        mod_volumes: dict[str, tuple[np.ndarray, str]] = {}
        mod_shapes: dict[str, list[int]] = {}
        mod_zooms: dict[str, list[float]] = {}
        rendered_extra_views: dict[str, str] = {}

        for mod in ("T1w", "T2w", "FLAIR", "bold", "dwi"):
            scan = by_mod.get(mod)
            if scan is None:
                continue
            arr, _aff, zooms = common.load_ras_volume(scan.nifti_path)
            mod_shapes[mod] = [int(x) for x in arr.shape]
            mod_zooms[mod] = [round(float(z), 2) for z in zooms]

            if mod == "T1w":
                t1_exact = bool(np.allclose(arr, t1_ras, atol=1e-5))
                checks.append(
                    common.CheckResult(
                        step="02_convert_modalities",
                        name="real_t1w_preserved",
                        status="pass" if t1_exact else "fail",
                        metric=f"allclose={t1_exact}, shape={arr.shape}",
                        detail="preserve_real_modalities=True must keep the original T1w bit-for-bit untouched.",
                    )
                )
                mod_volumes["T1w"] = (arr, f"real {zooms[0]:.1f}mm {arr.shape}")
            elif mod in ("T2w", "FLAIR"):
                mod_volumes[mod] = (arr, f"synth {zooms[0]:.1f}mm {arr.shape}")
            elif mod == "bold":
                mean_bold = np.mean(arr, axis=3).astype(np.float32)
                mod_volumes["bold"] = (mean_bold, f"EPI {zooms[0]:.1f}mm {arr.shape}")
                views = render_scan_views(
                    data=arr,
                    modality="bold",
                    zooms=zooms,
                    out_dir=multimodal_dir,
                    scan_id=scan.base_id,
                    render_midline=False,
                    render_enhanced=True,
                )
                if "fmri_carpet_summary" in views.get("enhanced", {}):
                    rendered_extra_views["bold_carpet"] = str(
                        (multimodal_dir / views["enhanced"]["fmri_carpet_summary"]).relative_to(out_root)
                    )
            elif mod == "dwi":
                bvals = None
                if scan.bval_path and scan.bval_path.exists():
                    raw_bvals, _ = read_bvals_bvecs(str(scan.bval_path), None)
                    bvals = np.asarray(raw_bvals, dtype=np.float32)
                b0_idx = np.where(bvals <= 50.0)[0] if bvals is not None else np.array([0])
                hi_idx = np.where(bvals > 150.0)[0] if bvals is not None else np.arange(1, arr.shape[3])
                b0_vol = np.mean(arr[..., b0_idx], axis=3).astype(np.float32)
                hi_vol = np.mean(arr[..., hi_idx], axis=3).astype(np.float32)
                mod_volumes["dwi_b0"] = (b0_vol, f"b=0 {zooms[0]:.1f}mm {arr.shape[:3]}")
                mod_volumes["dwi_highb"] = (hi_vol, f"high-b {zooms[0]:.1f}mm ({len(hi_idx)}d)")
                b0_priors = compute_anatomical_priors(b0_vol)
                b0_med = float(np.median(b0_vol[b0_priors.brain_mask])) if np.any(b0_priors.brain_mask) else 1.0
                hi_med = float(np.median(hi_vol[b0_priors.brain_mask])) if np.any(b0_priors.brain_mask) else 0.0
                atten_ratio = hi_med / max(b0_med, 1e-6)
                checks.append(
                    common.CheckResult(
                        step="02_convert_modalities",
                        name="dwi_highb_attenuation",
                        status="pass" if 0.05 < atten_ratio < 0.75 else "warn",
                        metric=f"median(high-b)/median(b=0) = {atten_ratio:.3f}",
                        detail="Stejskal-Tanner diffusion attenuation should substantially reduce high-b signal relative to b=0.",
                    )
                )
                views = render_scan_views(
                    data=arr,
                    modality="dwi",
                    zooms=zooms,
                    out_dir=multimodal_dir,
                    scan_id=scan.base_id,
                    render_midline=False,
                    render_enhanced=True,
                    bvals=bvals,
                )
                if "dwi_shell_dropout_summary" in views.get("enhanced", {}):
                    rendered_extra_views["dwi_summary"] = str(
                        (multimodal_dir / views["enhanced"]["dwi_shell_dropout_summary"]).relative_to(out_root)
                    )

        fig_mods = _render_modalities_overview(
            mod_volumes,
            out_path=out_root / "figures" / "02b_all_modalities_overview.png",
            icv_mask_3d=first_res.icv_mask,
        )

        # Summarize key blind IQMs per modality & check for NaN/inf
        iqm_summary_by_mod: dict[str, dict[str, float]] = {}
        nan_mods: list[str] = []
        for _, row in iqms_df.iterrows():
            mod_name = str(row["modality"])
            vals: dict[str, float] = {}
            for col in iqms_df.columns:
                if not col.startswith("iqm_"):
                    continue
                v = row[col]
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    continue
                try:
                    fv = float(v)
                    if np.isfinite(fv):
                        vals[col.removeprefix("iqm_")] = round(fv, 4)
                except (TypeError, ValueError):
                    pass
            if not vals:
                nan_mods.append(mod_name)
            iqm_summary_by_mod[mod_name] = vals

        checks.append(
            common.CheckResult(
                step="02_convert_modalities",
                name="all_5_modalities_and_iqms",
                status="pass" if (len(by_mod) == 5 and not nan_mods) else "fail",
                metric=f"modalities={sorted(by_mod.keys())} ({len(by_mod)}/5), nan_mods={nan_mods}",
                detail="All 5 modalities (T1w, T2w, FLAIR, bold, dwi) synthesized and blind IQMs extracted.",
            )
        )
        dwi_clean_fd = iqm_summary_by_mod.get("dwi", {}).get("mean_fd")
        if dwi_clean_fd is not None:
            checks.append(
                common.CheckResult(
                    step="02_convert_modalities",
                    name="clean_dwi_fd_sanity",
                    status="pass" if float(dwi_clean_fd) < 0.5 else "warn",
                    metric=f"clean DWI mean_fd={dwi_clean_fd:.3f} mm",
                    detail="Stationary clean multi-shell DWI should report near-zero shell-aware mean_fd (< 0.5 mm).",
                )
            )

    summary: dict[str, Any] = {
        "step": "02_convert_modalities",
        "elapsed_sec": timer.elapsed_sec,
        "full_4d": full_4d,
        "multimodal_bids_dir": str(multimodal_dir),
        "synthesis_manifest": str(synth_res.manifest_path),
        "extracted_icv_mask": str(icv_mask_path),
        "brain_mask_method": eff_mask_method,
        "baseline_iqms_tsv": str(iqm_tsv),
        "contrast_conversion": contrast_meta,
        "modality_shapes": mod_shapes,
        "modality_zooms_mm": mod_zooms,
        "baseline_iqms_by_modality": iqm_summary_by_mod,
        "figures": {
            "contrast_compartments": str(fig_comp.relative_to(out_root)),
            "all_modalities_overview": str(fig_mods.relative_to(out_root)),
            **rendered_extra_views,
        },
        "checks": common.checks_to_dicts(checks),
    }
    common.write_json(out_root / "01_modalities_summary.json", summary)
    print(f"[Step 02] Completed 5 modalities & baseline IQMs ({timer.elapsed_sec:.1f}s)")
    for chk in checks:
        print(f"  [{chk.status.upper():4s}] {chk.name}: {chk.metric}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 02: Convert T1w into all supported modalities.")
    parser.add_argument("--out-dir", type=Path, default=common.DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker count.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: auto|cpu|cuda.")
    parser.add_argument("--full-4d", action="store_true", help="Use full 60 TRs / 40 DWI dirs.")
    parser.add_argument("--brain-mask", type=Path, default=None, help="Optional external brain mask.")
    args = parser.parse_args()
    run_step(
        out_dir=args.out_dir,
        workers=args.workers,
        device=args.device,
        full_4d=args.full_4d,
        brain_mask=args.brain_mask,
    )


if __name__ == "__main__":
    main()
