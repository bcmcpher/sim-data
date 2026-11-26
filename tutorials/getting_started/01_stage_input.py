"""Step 01: Stage any single 3D T1w NIfTI into a clean 1-subject BIDS seed & inspect diagnostics.

Why this step matters:
- `sim-data` operates on BIDS directories. If you start from a standalone T1w NIfTI—or from a
  multi-acquisition dataset like MR-ART where `acq-headmotion1` sorts alphabetically before
  `acq-standard`—staging a clean single-subject BIDS directory ensures the exact chosen
  no-motion T1w scan is used as the baseline seed.
- Prints and plots orientation/RAS+ canonicalization, resolved encoding axes, anatomical priors
  (head/brain/WM/GM/background masks), and estimated native background noise sigma_0.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

from sim_data.io.orientation import (  # noqa: E402
    canonicalize_to_ras_with_transform,
    compute_anatomical_priors,
    estimate_background_noise_sigma,
    resolve_encoding_axes,
)


def _render_input_priors_figure(
    data: np.ndarray,
    priors: Any,
    out_path: Path,
    title: str,
) -> Path:
    """Plot 3-plane midline slices with brain/head mask contour and background air mask overlay."""
    mid_x = (priors.bbox_min[0] + priors.bbox_max[0]) // 2
    mid_y = (priors.bbox_min[1] + priors.bbox_max[1]) // 2
    mid_z = (priors.bbox_min[2] + priors.bbox_max[2]) // 2
    vmin, vmax = common.compute_window(data)

    planes = [
        ("Sagittal (mid-X)", data[mid_x, :, :], priors.brain_mask[mid_x, :, :], priors.bg_mask[mid_x, :, :]),
        ("Coronal (mid-Y)", data[:, mid_y, :], priors.brain_mask[:, mid_y, :], priors.bg_mask[:, mid_y, :]),
        ("Axial (mid-Z)", data[:, :, mid_z], priors.brain_mask[:, :, mid_z], priors.bg_mask[:, :, mid_z]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.6))
    for col, (label, sl, bmask, bgmask) in enumerate(planes):
        img_2d = common.orient_2d(common.normalize_slice(sl, vmin, vmax))
        bm_2d = common.orient_2d(bmask.astype(np.float32))
        bg_2d = common.orient_2d(bgmask.astype(np.float32))

        ax_top = axes[0, col]
        ax_top.imshow(img_2d, cmap="gray", vmin=0.0, vmax=1.0)
        ax_top.contour(bm_2d, levels=[0.5], colors=["#00e5ff"], linewidths=1.0)
        ax_top.set_title(f"{label}\n(cyan = priors.brain_mask)", fontsize=9)
        ax_top.axis("off")

        ax_bot = axes[1, col]
        ax_bot.imshow(common.orient_2d(common.bg_boost_slice(sl, ref_p95=priors.fg_p95)), cmap="gray", vmin=0.0, vmax=1.0)
        overlay = np.zeros((*bg_2d.shape, 4), dtype=np.float32)
        overlay[bg_2d > 0.5] = [0.2, 0.6, 1.0, 0.25]
        ax_bot.imshow(overlay)
        ax_bot.set_title("BG-boost + blue = priors.bg_mask", fontsize=8.5)
        ax_bot.axis("off")

    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return common.save_figure(fig, out_path)


def run_step(
    t1w: Path | None = None,
    json_sidecar: Path | None = None,
    brain_mask: Path | None = None,
    subject_id: str = "01",
    out_dir: Path = common.DEFAULT_OUT_DIR,
    use_phantom: bool = False,
) -> dict[str, Any]:
    """Stage the input T1w into `<out_dir>/00_seed_bids` and run baseline diagnostics."""
    out_root = Path(out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    checks: list[common.CheckResult] = []

    with common.StepTimer() as timer:
        src_t1w, auto_json, source_kind = common.resolve_input_t1w(
            t1w=t1w, use_phantom=use_phantom, out_dir=out_root
        )
        src_json = json_sidecar.expanduser().resolve() if json_sidecar is not None else auto_json
        src_mask = brain_mask.expanduser().resolve() if brain_mask is not None else None

        sub_clean = subject_id.removeprefix("sub-")
        seed_bids = out_root / "00_seed_bids"
        anat_dir = seed_bids / f"sub-{sub_clean}" / "anat"
        anat_dir.mkdir(parents=True, exist_ok=True)

        ext = ".nii.gz" if src_t1w.name.endswith(".nii.gz") else ".nii"
        dst_t1w = anat_dir / f"sub-{sub_clean}_T1w{ext}"
        shutil.copy2(src_t1w, dst_t1w)

        sidecar: dict[str, Any] = {}
        dst_json = anat_dir / f"sub-{sub_clean}_T1w.json"
        if src_json is not None and src_json.exists():
            sidecar = json.loads(src_json.read_text(encoding="utf-8"))
            shutil.copy2(src_json, dst_json)
        else:
            sidecar = {"Modality": "MR"}
            dst_json.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

        dst_mask: Path | None = None
        if src_mask is not None:
            if not src_mask.exists():
                raise FileNotFoundError(f"External brain mask not found: {src_mask}")
            m_ext = ".nii.gz" if src_mask.name.endswith(".nii.gz") else ".nii"
            dst_mask = anat_dir / f"sub-{sub_clean}_desc-brain_mask{m_ext}"
            shutil.copy2(src_mask, dst_mask)

        (seed_bids / "dataset_description.json").write_text(
            json.dumps(
                {"Name": "Tutorial Staged Clean Seed BIDS", "BIDSVersion": "1.9.0", "DatasetType": "raw"},
                indent=2,
            ),
            encoding="utf-8",
        )

        # Inspect native geometry, orientation, encoding axes, and anatomical priors
        raw_img = nib.load(str(dst_t1w))
        if not isinstance(raw_img, nib.Nifti1Image):
            raw_img = nib.Nifti1Image(np.asarray(raw_img.get_fdata(), dtype=np.float32), raw_img.affine)
        orig_axcodes = "".join(nib.aff2axcodes(raw_img.affine))
        orig_shape = [int(s) for s in raw_img.shape]
        orig_dtype = str(raw_img.get_data_dtype())

        ras_data, _affine, zooms, ornt_transform = canonicalize_to_ras_with_transform(raw_img)
        if ras_data.ndim != 3:
            raise ValueError(f"Expected 3D structural T1w NIfTI, got shape {ras_data.shape}")

        pe_axis, ro_axis, sl_axis = resolve_encoding_axes(
            sidecar, modality="T1w", ornt_transform=ornt_transform
        )
        priors = compute_anatomical_priors(ras_data)
        sigma_0 = estimate_background_noise_sigma(ras_data, priors.bg_mask, modality="T1w")

        bbox_span = [
            int(priors.bbox_max[i] - priors.bbox_min[i] + 1) for i in range(3)
        ]
        bbox_frac = [round(bbox_span[i] / max(1, ras_data.shape[i]), 3) for i in range(3)]
        fg_frac = round(float(priors.brain_mask.mean()), 4)
        bg_frac = round(float(priors.bg_mask.mean()), 4)
        snr_est = round(float(priors.fg_median / max(sigma_0, 1e-6)), 2) if sigma_0 > 1e-6 else float("inf")

        has_pe = "PhaseEncodingDirection" in sidecar
        has_sl = "SliceEncodingDirection" in sidecar
        checks.append(
            common.CheckResult(
                step="01_stage_input",
                name="orientation_ras",
                status="pass",
                metric=f"{orig_axcodes} -> RAS ({ras_data.shape}, {[round(z, 2) for z in zooms]} mm)",
                detail=f"Loaded {src_t1w.name} ({orig_dtype}); canonicalized to RAS+ float32.",
            )
        )
        checks.append(
            common.CheckResult(
                step="01_stage_input",
                name="sidecar_encoding_axes",
                status="pass" if (has_pe and has_sl) else "warn",
                metric=f"PE={pe_axis}, RO={ro_axis}, SL={sl_axis} (explicit PE={has_pe}, SL={has_sl})",
                detail=(
                    "Explicit PhaseEncodingDirection and SliceEncodingDirection found in JSON."
                    if (has_pe and has_sl)
                    else "Sidecar lacks PhaseEncodingDirection/SliceEncodingDirection (common in DICOM-converted 3D T1w such as MR-ART); sim-data defaulted to PE=1 (A-P), SL=2 (S-I)."
                ),
            )
        )
        checks.append(
            common.CheckResult(
                step="01_stage_input",
                name="priors_bbox_extent",
                status="warn" if bbox_frac[2] > 0.82 else "pass",
                metric=f"fg_frac={fg_frac}, bbox_frac={bbox_frac} (Z={priors.bbox_min[2]}..{priors.bbox_max[2]})",
                detail=(
                    "Foreground mask spans >82% of Z-axis (typical full-head/neck scan). Because priors.brain_mask covers the whole head+neck, bbox-anchored artifacts (fov_cropping, defacing_damage, susceptibility_metal) and mask-free T1w->T2w conversion may be affected by neck/face extent."
                    if bbox_frac[2] > 0.82
                    else "Foreground bounding box is compact within FOV."
                ),
            )
        )
        checks.append(
            common.CheckResult(
                step="01_stage_input",
                name="native_noise_sigma0",
                status="pass" if sigma_0 > 0.0 else "warn",
                metric=f"sigma_0={sigma_0:.3f}, fg_median={priors.fg_median:.1f}, est_SNR={snr_est}",
                detail=f"Background air fraction={bg_frac}; used by excess_variance noise calibration.",
            )
        )

        fig_path = _render_input_priors_figure(
            ras_data,
            priors=priors,
            out_path=out_root / "figures" / "01_input_priors.png",
            title=f"Step 01: Staged Input T1w ({src_t1w.name}) & Anatomical Priors",
        )

    summary: dict[str, Any] = {
        "step": "01_stage_input",
        "elapsed_sec": timer.elapsed_sec,
        "source_kind": source_kind,
        "source_t1w": str(src_t1w),
        "source_json": str(src_json) if src_json else None,
        "external_brain_mask": str(dst_mask) if dst_mask else None,
        "seed_bids_dir": str(seed_bids),
        "staged_t1w": str(dst_t1w),
        "subject": sub_clean,
        "orig_axcodes": orig_axcodes,
        "orig_shape": orig_shape,
        "ras_shape": [int(s) for s in ras_data.shape],
        "zooms_mm": [round(float(z), 4) for z in zooms],
        "encoding_axes_ras": {"pe": pe_axis, "ro": ro_axis, "sl": sl_axis},
        "sidecar_has_pe_dir": has_pe,
        "sidecar_has_sl_dir": has_sl,
        "priors": {
            "bbox_min": list(priors.bbox_min),
            "bbox_max": list(priors.bbox_max),
            "bbox_frac": bbox_frac,
            "fg_fraction": fg_frac,
            "bg_fraction": bg_frac,
            "fg_median": round(float(priors.fg_median), 3),
            "fg_p95": round(float(priors.fg_p95), 3),
            "native_noise_sigma0": round(float(sigma_0), 5),
        },
        "figures": {"input_priors": str(fig_path.relative_to(out_root))},
        "checks": common.checks_to_dicts(checks),
    }
    common.write_json(out_root / "00_seed_diagnostics.json", summary)
    print(f"[Step 01] Staged {src_t1w.name} -> {seed_bids} ({timer.elapsed_sec:.1f}s)")
    for chk in checks:
        print(f"  [{chk.status.upper():4s}] {chk.name}: {chk.metric}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 01: Stage a single T1w NIfTI into BIDS.")
    parser.add_argument("--t1w", type=Path, default=None, help="Path to clean 3D T1w NIfTI.")
    parser.add_argument("--json", type=Path, default=None, help="Optional BIDS JSON sidecar.")
    parser.add_argument("--brain-mask", type=Path, default=None, help="Optional external brain mask.")
    parser.add_argument("--subject-id", type=str, default="01", help="Subject ID (default: 01).")
    parser.add_argument("--out-dir", type=Path, default=common.DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument("--use-phantom", action="store_true", help="Use synthetic phantom T1w.")
    args = parser.parse_args()
    run_step(
        t1w=args.t1w,
        json_sidecar=args.json,
        brain_mask=args.brain_mask,
        subject_id=args.subject_id,
        out_dir=args.out_dir,
        use_phantom=args.use_phantom,
    )


if __name__ == "__main__":
    main()
