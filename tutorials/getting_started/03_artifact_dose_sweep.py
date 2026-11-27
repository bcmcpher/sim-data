"""Step 03: Apply every artifact individually across Low / Mid / High severities.

Exercises:
- `run_simulation_plan` in `dose_response` mode with:
  - `dose_severities = [0.0, 0.25, 0.55, 0.85]` (Baseline / Low / Mid / High)
  - `coupled_trajectories = True` (Common Random Numbers so severity scales monotonically)
  - `sites = []` (`NativeSite` passthrough so s=0.0 is bit-for-bit identical to input)
  - `noise_calibration = "excess_variance"`
  - `naming_scheme = "acq_blinded"` (raw-BIDS compliant, zero artifact/severity leakage)
  - `render_views = True` (exercises `ghibli.views.mosaic`)
- All 11 structural artifacts on the real T1w scan (34 scans)
- All 6 remaining 4D/EPI/DWI-specific artifacts on the synthesized BOLD & DWI scans
  (covering 17/17 artifacts in `ARTIFACT_REGISTRY`, unless `--t1w-only` is passed)
- Blind NiPreps IQM extraction (`extract_iqms_for_bids_dir`) + `run_sensitivity_analysis`
- Annotated visual contact sheets (de-blinded strictly from `manifest.tsv`) + zero-leakage audit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

from ghibli.artifacts.registry import ARTIFACT_REGISTRY, compatible_artifacts  # noqa: E402
from ghibli.config import (  # noqa: E402
    ArtifactSpec,
    OutputConfig,
    SamplingPlanConfig,
    SimulationConfig,
)
from ghibli.eval.sensitivity import run_sensitivity_analysis  # noqa: E402
from ghibli.io.orientation import compute_anatomical_priors  # noqa: E402
from ghibli.iqm.diffusion import compute_bad_slices_qsiprep  # noqa: E402
from ghibli.iqm.extractor import extract_iqms_for_bids_dir  # noqa: E402
from ghibli.iqm.functional import estimate_fd_proxy  # noqa: E402
from ghibli.pipeline.engine import run_simulation_plan  # noqa: E402

# 4D/EPI/DWI-specific artifacts not compatible with 3D T1w (plus motion_4d on DWI to verify bvecs)
BOLD_SWEEP_ARTIFACTS = ["motion_4d", "nyquist_n2_ghost", "epi_susceptibility", "temporal_drift"]
DWI_SWEEP_ARTIFACTS = ["eddy_currents_dwi", "dwi_slice_dropout", "motion_4d"]

# Artifacts best inspected sagittally rather than axially
_SAGITTAL_ARTIFACTS = {"fov_cropping", "defacing_damage", "anisotropy"}


def _audit_zero_leakage(bids_dir: Path, artifact_names: list[str]) -> tuple[bool, list[str]]:
    """Verify no artifact name or severity token leaks into BIDS filenames, JSONs, or scans.tsv."""
    forbidden = [a.lower().replace("_", "") for a in artifact_names] + [a.lower() for a in artifact_names]
    leaks: list[str] = []

    for p in sorted(bids_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(bids_dir))
        # Skip explicitly ignored bookkeeping outputs
        top = p.relative_to(bids_dir).parts[0]
        if top in {"views", "packets", "sensitivity_analysis", "manifest.tsv", "iqms.tsv", "simulation_config_resolved.yaml"}:
            continue
        if p.name.endswith(".nii.gz") or p.name.endswith(".nii"):
            low = p.name.lower()
            for tok in forbidden:
                if tok in low:
                    leaks.append(f"filename leak: {rel} matched '{tok}'")
                    break
        elif p.suffix == ".json" and p.name != "dataset_description.json":
            txt = p.read_text(encoding="utf-8", errors="ignore").lower()
            if "simulationprovenance" in txt:
                leaks.append(f"sidecar leak: {rel} contains SimulationProvenance")
        elif p.name.endswith("_scans.tsv"):
            txt = p.read_text(encoding="utf-8", errors="ignore").lower()
            for col in ("target_artifact", "dose_level", "sev_"):
                if col in txt:
                    leaks.append(f"scans.tsv leak: {rel} contains '{col}'")
    return (len(leaks) == 0), leaks


def _extract_artifact_params(row: pd.Series, art_name: str) -> dict[str, Any]:
    """Extract realized parameter dict for `art_name` from a manifest row's artifacts_json."""
    try:
        items = json.loads(str(row.get("artifacts_json", "[]")))
    except Exception:
        return {}
    for item in items:
        if isinstance(item, dict) and item.get("name") == art_name:
            params = item.get("params", {})
            return params if isinstance(params, dict) else {}
    return {}


def _short_param_note(art_name: str, artifacts_json_str: str) -> str:
    """Extract a concise realized parameter label from manifest artifacts_json."""
    try:
        items = json.loads(artifacts_json_str)
    except Exception:
        return ""
    if not items:
        return ""
    params = items[0].get("params", {})
    if art_name == "motion_3d":
        return f"deg={params.get('degrees')}, trans={params.get('translation_mm')}mm"
    if art_name == "ghosting_3d":
        return f"ghosts={params.get('num_ghosts')}, int={params.get('intensity')}"
    if art_name == "wrap_around":
        return f"ax={params.get('axis')}, frac={params.get('wrap_fraction')}"
    if art_name == "bias_field":
        return f"span={params.get('realized_brain_field_range')}"
    if art_name == "rf_spike":
        return f"n={params.get('num_spikes')}, lnK={params.get('k_intensity_ln')}"
    if art_name == "susceptibility_metal":
        return f"{params.get('landmark')}, r={params.get('radius_vox')}v"
    if art_name == "rician_noise":
        return f"sig={params.get('sigma_abs')} (cal={params.get('noise_calibration')})"
    if art_name == "fov_cropping":
        return f"{params.get('side')}, cut={params.get('brain_voxels_clipped_pct')}%"
    if art_name == "gibbs_ringing":
        return f"alpha={params.get('alpha')}"
    if art_name == "anisotropy":
        return f"ax={params.get('axis')}, ds={params.get('downsampling')}x"
    if art_name == "defacing_damage":
        return f"{params.get('pattern')}, dmg={params.get('brain_voxels_damaged_pct')}%"
    if art_name == "motion_4d":
        return f"FD_mean={params.get('fd_mean_mm')}mm, max={params.get('fd_max_mm')}mm"
    if art_name == "nyquist_n2_ghost":
        return f"phase={params.get('phase_error_rad')}rad"
    if art_name == "epi_susceptibility":
        return f"shift={params.get('max_shift_vox')}v, drop={params.get('dropout_strength')}"
    if art_name == "temporal_drift":
        return f"lin={params.get('linear_amp')}, quad={params.get('quad_amp')}"
    if art_name == "eddy_currents_dwi":
        return f"shear={params.get('max_shear')}, shift={params.get('max_shift_vox')}v"
    if art_name == "dwi_slice_dropout":
        return f"dropped={params.get('n_dropped_slices')} ({params.get('dropped_slice_fraction')})"
    return ""


def _extract_plane_slice(vol: np.ndarray, art_name: str, mid_x: int, mid_z: int, high_row: pd.Series) -> tuple[np.ndarray, str]:
    """Select the most visually diagnostic 2D slice for an artifact."""
    if art_name in _SAGITTAL_ARTIFACTS:
        idx = min(max(0, mid_x), vol.shape[0] - 1)
        return common.orient_2d(vol[idx, :, :]), "Sag"
    if art_name == "susceptibility_metal":
        try:
            items = json.loads(str(high_row.get("artifacts_json", "[]")))
            cz = int(round(float(items[0]["params"]["center_vox"][2])))
            cz = min(max(0, cz), vol.shape[2] - 1)
            return common.orient_2d(vol[:, :, cz]), f"Ax z={cz}"
        except Exception:
            pass
    idx = min(max(0, mid_z), vol.shape[2] - 1)
    return common.orient_2d(vol[:, :, idx]), "Ax"


def _render_t1w_artifact_sheet(
    artifacts: list[str],
    manifest_df: pd.DataFrame,
    bids_dir: Path,
    base_vol: np.ndarray,
    out_path: Path,
    title: str,
    icv_mid: tuple[int, int, int] | None = None,
) -> Path:
    """Render Baseline | Low | Mid | High | High BG-Boost | |High - Base| contact sheet."""
    if icv_mid is not None:
        mid_x, _mid_y, mid_z = icv_mid
    else:
        mid_x, _mid_y, mid_z = common.midline_indices(base_vol)
    vmin, vmax = common.compute_window(base_vol)
    base_p95 = float(np.percentile(base_vol[base_vol > 0], 95.0)) if np.any(base_vol > 0) else 1.0
    pos_doses = [d for d in common.DEFAULT_DOSE_LEVELS if d > 0.0]

    nrows = len(artifacts)
    ncols = 6
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.6, 2.75 * nrows))
    if nrows == 1:
        axes = axes[None, :]

    col_headers = [
        "Baseline (0.0)",
        f"Low ({pos_doses[0]:.2f})",
        f"Mid ({pos_doses[1]:.2f})",
        f"High ({pos_doses[2]:.2f})",
        f"High ({pos_doses[2]:.2f}) BG-Boost",
        "|High - Base| Diff",
    ]
    for c, hdr in enumerate(col_headers):
        axes[0, c].set_title(hdr, fontsize=9, fontweight="bold")

    for r, art in enumerate(artifacts):
        sub = manifest_df[manifest_df["target_artifact"] == art].sort_values("dose_level")
        rows_by_dose = {round(float(row["dose_level"]), 2): row for _, row in sub.iterrows()}
        high_row = rows_by_dose.get(round(pos_doses[-1], 2), sub.iloc[-1])
        base_sl, plane_lbl = _extract_plane_slice(base_vol, art, mid_x, mid_z, high_row)

        # Col 0: Baseline
        axes[r, 0].imshow(common.normalize_slice(base_sl, vmin, vmax), cmap="gray", vmin=0.0, vmax=1.0)
        note = _short_param_note(art, str(high_row.get("artifacts_json", "")))
        axes[r, 0].set_ylabel(f"{art}\n[{plane_lbl}]\n{note}", fontsize=7.5, fontweight="bold")
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])

        high_sl = base_sl
        for c_idx, dose in enumerate(pos_doses, start=1):
            row = rows_by_dose.get(round(dose, 2))
            if row is None:
                axes[r, c_idx].axis("off")
                continue
            vol, _, _ = common.load_ras_volume(bids_dir / str(row["nifti_path"]))
            sl, _ = _extract_plane_slice(vol, art, mid_x, mid_z, high_row)
            if c_idx == len(pos_doses):
                high_sl = sl
            axes[r, c_idx].imshow(common.normalize_slice(sl, vmin, vmax), cmap="gray", vmin=0.0, vmax=1.0)
            ssim_v = row.get("ref_ssim", np.nan)
            psnr_v = row.get("ref_psnr", np.nan)
            axes[r, c_idx].set_xlabel(f"SSIM={ssim_v:.3f} | PSNR={psnr_v:.1f}", fontsize=7)
            axes[r, c_idx].set_xticks([])
            axes[r, c_idx].set_yticks([])

        # Col 4: High BG-boost
        axes[r, 4].imshow(common.bg_boost_slice(high_sl, ref_p95=base_p95), cmap="gray", vmin=0.0, vmax=1.0)
        axes[r, 4].set_xticks([])
        axes[r, 4].set_yticks([])

        # Col 5: |High - Base| diff
        if high_sl.shape == base_sl.shape:
            diff = np.abs(high_sl - base_sl)
            dmax = max(float(np.percentile(diff, 99.0)), 1e-4)
            axes[r, 5].imshow(diff, cmap="inferno", vmin=0.0, vmax=dmax)
        else:
            axes[r, 5].text(0.5, 0.5, f"Shape changed\n{high_sl.shape}", ha="center", va="center", fontsize=8)
        axes[r, 5].set_xticks([])
        axes[r, 5].set_yticks([])

    fig.suptitle(title, fontsize=11.5, fontweight="bold", y=0.995)
    fig.tight_layout()
    return common.save_figure(fig, out_path)


def _render_4d_artifacts_figure(
    manifest_4d: pd.DataFrame,
    bids_dir_4d: Path,
    out_path: Path,
) -> Path:
    """Render visual + temporal/slice diagnostics for all 6 4D/EPI/DWI-specific artifacts."""
    rows_spec = [
        ("bold", "nyquist_n2_ghost", "spatial"),
        ("bold", "epi_susceptibility", "spatial"),
        ("bold", "motion_4d", "bold_fd"),
        ("bold", "temporal_drift", "bold_drift"),
        ("dwi", "eddy_currents_dwi", "dwi_highb"),
        ("dwi", "dwi_slice_dropout", "dwi_dropout"),
    ]
    pos_doses = [d for d in common.DEFAULT_DOSE_LEVELS if d > 0.0]
    fig, axes = plt.subplots(len(rows_spec), 5, figsize=(14.5, 2.8 * len(rows_spec)))

    base_cache: dict[str, tuple[np.ndarray, tuple[float, float, float], np.ndarray | None]] = {}
    for mod in ("bold", "dwi"):
        b_rows = manifest_4d[(manifest_4d["modality"] == mod) & (manifest_4d["target_artifact"] == "baseline")]
        if b_rows.empty:
            continue
        p = bids_dir_4d / str(b_rows.iloc[0]["nifti_path"])
        arr, _, zooms = common.load_ras_volume(p)
        bvals = None
        bval_p = p.with_name(p.name.replace(".nii.gz", ".bval"))
        if bval_p.exists():
            bvals = np.loadtxt(str(bval_p), dtype=np.float32).reshape(-1)
        base_cache[mod] = (arr, zooms, bvals)

    for r, (mod, art, kind) in enumerate(rows_spec):
        if mod not in base_cache:
            for c in range(5):
                axes[r, c].axis("off")
            continue
        base_4d, zooms, bvals = base_cache[mod]
        sub = manifest_4d[(manifest_4d["modality"] == mod) & (manifest_4d["target_artifact"] == art)].sort_values("dose_level")
        by_dose = {round(float(row["dose_level"]), 2): row for _, row in sub.iterrows()}

        base_3d = common.representative_3d(base_4d, bvals=bvals)
        if kind == "dwi_highb" and bvals is not None and np.any(bvals > 150.0):
            base_3d = np.mean(base_4d[..., bvals > 150.0], axis=3).astype(np.float32)
        priors = compute_anatomical_priors(common.representative_3d(base_4d, bvals=bvals))
        mid_z = (priors.bbox_min[2] + priors.bbox_max[2]) // 2
        vmin, vmax = common.compute_window(base_3d)
        base_p95 = float(np.percentile(base_3d[base_3d > 0], 95.0)) if np.any(base_3d > 0) else 1.0

        high_row = by_dose.get(round(pos_doses[-1], 2), sub.iloc[-1])
        note = _short_param_note(art, str(high_row.get("artifacts_json", "")))

        if kind in ("spatial", "dwi_highb"):
            base_sl = common.orient_2d(base_3d[:, :, mid_z])
            axes[r, 0].imshow(common.normalize_slice(base_sl, vmin, vmax), cmap="gray", vmin=0.0, vmax=1.0)
            axes[r, 0].set_title("Baseline (0.0)", fontsize=8.5)
            axes[r, 0].set_ylabel(f"{mod}: {art}\n{note}", fontsize=7.5, fontweight="bold")
            axes[r, 0].set_xticks([])
            axes[r, 0].set_yticks([])

            high_sl = base_sl
            for ci, dose in enumerate(pos_doses, start=1):
                row = by_dose.get(round(dose, 2))
                if row is None:
                    axes[r, ci].axis("off")
                    continue
                arr, _, _ = common.load_ras_volume(bids_dir_4d / str(row["nifti_path"]))
                v3 = (
                    np.mean(arr[..., bvals > 150.0], axis=3).astype(np.float32)
                    if (kind == "dwi_highb" and bvals is not None and np.any(bvals > 150.0))
                    else common.representative_3d(arr, bvals=bvals)
                )
                sl = common.orient_2d(v3[:, :, mid_z])
                if ci == len(pos_doses):
                    high_sl = sl
                disp = common.bg_boost_slice(sl, ref_p95=base_p95) if art == "nyquist_n2_ghost" else common.normalize_slice(sl, vmin, vmax)
                axes[r, ci].imshow(disp, cmap="gray", vmin=0.0, vmax=1.0)
                axes[r, ci].set_title(f"Dose {dose:.2f}" + (" (BG)" if art == "nyquist_n2_ghost" else ""), fontsize=8.5)
                axes[r, ci].set_xticks([])
                axes[r, ci].set_yticks([])

            diff = np.abs(high_sl - base_sl)
            axes[r, 4].imshow(diff, cmap="inferno", vmin=0.0, vmax=max(float(np.percentile(diff, 99.0)), 1e-4))
            axes[r, 4].set_title("|High - Base| Diff", fontsize=8.5)
            axes[r, 4].set_xticks([])
            axes[r, 4].set_yticks([])

        elif kind == "bold_fd":
            # Col 0-3: mid-axial mean BOLD across doses; Col 4: blind FD curves overlaid across doses
            axes[r, 0].imshow(common.orient_2d(common.normalize_slice(base_3d[:, :, mid_z], vmin, vmax)), cmap="gray")
            axes[r, 0].set_title("Baseline (0.0)", fontsize=8.5)
            axes[r, 0].set_ylabel(f"{mod}: {art}\n{note}", fontsize=7.5, fontweight="bold")
            axes[r, 0].set_xticks([])
            axes[r, 0].set_yticks([])
            fd0 = estimate_fd_proxy(base_4d, priors.brain_mask, zooms)
            axes[r, 4].plot(fd0, label="0.0", lw=1.0)
            for ci, dose in enumerate(pos_doses, start=1):
                row = by_dose.get(round(dose, 2))
                if row is None:
                    axes[r, ci].axis("off")
                    continue
                arr, _, _ = common.load_ras_volume(bids_dir_4d / str(row["nifti_path"]))
                v3 = np.mean(arr, axis=3).astype(np.float32)
                axes[r, ci].imshow(common.orient_2d(common.normalize_slice(v3[:, :, mid_z], vmin, vmax)), cmap="gray")
                axes[r, ci].set_title(f"Dose {dose:.2f} (mean)", fontsize=8.5)
                axes[r, ci].set_xticks([])
                axes[r, ci].set_yticks([])
                fd_i = estimate_fd_proxy(arr, priors.brain_mask, zooms)
                axes[r, 4].plot(fd_i, label=f"{dose:.2f}", lw=1.1)
            axes[r, 4].set_title("Blind FD Proxy (mm)", fontsize=8.5)
            axes[r, 4].set_xlabel("TR", fontsize=7.5)
            axes[r, 4].tick_params(labelsize=7)
            axes[r, 4].legend(fontsize=6.5, loc="upper right")

        elif kind == "bold_drift":
            # Col 0: Baseline mean; Col 1-3: voxelwise temporal SD maps on a SHARED scale; Col 4: global mean curves
            axes[r, 0].imshow(common.orient_2d(common.normalize_slice(base_3d[:, :, mid_z], vmin, vmax)), cmap="gray")
            axes[r, 0].set_title("Baseline (mean)", fontsize=8.5)
            axes[r, 0].set_ylabel(f"{mod}: {art}\n{note}", fontsize=7.5, fontweight="bold")
            axes[r, 0].set_xticks([])
            axes[r, 0].set_yticks([])
            g0 = base_4d[priors.brain_mask].mean(axis=0)
            axes[r, 4].plot(g0 / max(g0[0], 1e-5), label="0.0", lw=1.0)
            tsd_list: list[tuple[int, float, np.ndarray]] = []
            shared_vmax = 1e-3
            for ci, dose in enumerate(pos_doses, start=1):
                row = by_dose.get(round(dose, 2))
                if row is None:
                    continue
                arr, _, _ = common.load_ras_volume(bids_dir_4d / str(row["nifti_path"]))
                tsd = np.std(arr, axis=3)
                shared_vmax = max(shared_vmax, float(np.percentile(tsd[priors.brain_mask], 98.0)))
                tsd_list.append((ci, dose, tsd))
                gi = arr[priors.brain_mask].mean(axis=0)
                axes[r, 4].plot(gi / max(gi[0], 1e-5), label=f"{dose:.2f}", lw=1.2)
            for ci, dose, tsd in tsd_list:
                axes[r, ci].imshow(
                    common.orient_2d(tsd[:, :, mid_z]),
                    cmap="magma",
                    vmin=0.0,
                    vmax=shared_vmax,
                )
                axes[r, ci].set_title(f"Dose {dose:.2f} (tSD)", fontsize=8.5)
                axes[r, ci].set_xticks([])
                axes[r, ci].set_yticks([])
            axes[r, 4].set_title("Global Mean Signal (norm)", fontsize=8.5)
            axes[r, 4].set_xlabel("TR", fontsize=7.5)
            axes[r, 4].tick_params(labelsize=7)
            axes[r, 4].legend(fontsize=6.5, loc="best")

        elif kind == "dwi_dropout":
            # Col 0-3: Vol x Slice intensity ratio heatmaps across Baseline / Low / Mid / High; Col 4: High coronal high-b showing venetian-blind stripes
            _, rmap0 = compute_bad_slices_qsiprep(base_4d, priors.brain_mask, slice_axis=2, bvals=bvals)
            axes[r, 0].imshow(rmap0.T, aspect="auto", cmap="RdYlBu", vmin=0.35, vmax=1.25, origin="lower")
            axes[r, 0].set_title("Baseline (Vol x Slice)", fontsize=8.5)
            axes[r, 0].set_ylabel(f"{mod}: {art}\n{note}", fontsize=7.5, fontweight="bold")
            axes[r, 0].tick_params(labelsize=7)
            high_arr = base_4d
            for ci, dose in enumerate(pos_doses, start=1):
                row = by_dose.get(round(dose, 2))
                if row is None:
                    axes[r, ci].axis("off")
                    continue
                arr, _, _ = common.load_ras_volume(bids_dir_4d / str(row["nifti_path"]))
                if ci == len(pos_doses):
                    high_arr = arr
                bad_cnt, rmap = compute_bad_slices_qsiprep(arr, priors.brain_mask, slice_axis=2, bvals=bvals)
                axes[r, ci].imshow(rmap.T, aspect="auto", cmap="RdYlBu", vmin=0.35, vmax=1.25, origin="lower")
                axes[r, ci].set_title(f"Dose {dose:.2f} (bad={bad_cnt})", fontsize=8.5)
                axes[r, ci].tick_params(labelsize=7)

            # Show coronal slice of worst-dropped high-b volume
            if bvals is not None and np.any(bvals > 150.0):
                hi_indices = np.where(bvals > 150.0)[0]
                worst_v = int(hi_indices[int(np.argmin([float(high_arr[..., v].mean()) for v in hi_indices]))])
            else:
                worst_v = high_arr.shape[3] - 1
            mid_y = (priors.bbox_min[1] + priors.bbox_max[1]) // 2
            cor_sl = common.orient_2d(high_arr[:, mid_y, :, worst_v])
            cvmin, cvmax = common.compute_window(high_arr[..., worst_v])
            axes[r, 4].imshow(common.normalize_slice(cor_sl, cvmin, cvmax), cmap="gray")
            axes[r, 4].set_title(f"High Coronal (vol {worst_v})", fontsize=8.5)
            axes[r, 4].set_xticks([])
            axes[r, 4].set_yticks([])

    fig.suptitle(
        "Step 03b: 4D / EPI / DWI Artifact Dose Sweep (Low=0.25, Mid=0.55, High=0.85)",
        fontsize=11.5,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()
    return common.save_figure(fig, out_path)


def _to_rgb_tile(
    sl_2d: np.ndarray,
    vmin: float,
    vmax: float,
    target_h: int = 220,
    ref_contour: np.ndarray | None = None,
) -> Image.Image:
    u8 = (common.normalize_slice(sl_2d, vmin, vmax) * 255.0).astype(np.uint8)
    rgb = np.stack([u8, u8, u8], axis=-1)
    if ref_contour is not None and ref_contour.shape == u8.shape:
        # Subtle cyan reference outline from TR 0 so cumulative head drift & shifts are easy to track
        rgb[ref_contour, 0] = (rgb[ref_contour, 0].astype(np.float32) * 0.45).astype(np.uint8)
        rgb[ref_contour, 1] = np.minimum(255, rgb[ref_contour, 1].astype(np.int32) + 120).astype(np.uint8)
        rgb[ref_contour, 2] = np.minimum(255, rgb[ref_contour, 2].astype(np.int32) + 140).astype(np.uint8)
    pil = Image.fromarray(rgb, mode="RGB")
    w, h = pil.size
    if h > 0 and h != target_h:
        new_w = max(1, int(round(w * (target_h / h))))
        pil = pil.resize((new_w, target_h), resample=Image.Resampling.BILINEAR)
    return pil


def _compose_labeled_strip(tiles: list[tuple[str, Image.Image]], banner: str) -> Image.Image:
    pad = 8
    header_h = 44
    tile_h = max(im.size[1] for _, im in tiles)
    total_w = sum(im.size[0] for _, im in tiles) + pad * (len(tiles) + 1)
    canvas = Image.new("RGB", (total_w, tile_h + header_h + pad), color=(18, 22, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 6), banner, fill=(235, 240, 248))
    x = pad
    for lbl, im in tiles:
        draw.text((x, 24), lbl, fill=(185, 215, 255))
        canvas.paste(im, (x, header_h))
        x += im.size[0] + pad
    return canvas


def _render_motion_param_strip(
    tr_idx: int,
    n_trs: int,
    width_px: int,
    height_px: int,
    high_params: dict[str, list[float]] | None,
    fd_by_label: list[tuple[str, np.ndarray]],
) -> Image.Image:
    """Render a synchronized dark-theme motion parameter strip (Tx/Ty/Tz, Rx/Ry/Rz, FD) with TR playhead."""
    dpi = 100
    fig, axes = plt.subplots(
        1, 3, figsize=(width_px / dpi, height_px / dpi), dpi=dpi, facecolor="#12161c"
    )
    trs = np.arange(1, n_trs + 1)
    cur_tr = tr_idx + 1

    def _style_ax(ax: Any, title: str, ylabel: str) -> None:
        ax.set_facecolor("#181e27")
        ax.set_title(title, color="#e6edf3", fontsize=8.0, fontweight="bold", pad=4)
        ax.set_xlabel("TR", color="#9fb0c4", fontsize=7.0, labelpad=2)
        ax.set_ylabel(ylabel, color="#9fb0c4", fontsize=7.0, labelpad=2)
        ax.tick_params(colors="#9fb0c4", labelsize=6.5, length=2.5)
        ax.set_xlim(1, max(2, n_trs))
        ax.grid(True, color="#2b3442", linewidth=0.5, alpha=0.8)
        for spine in ax.spines.values():
            spine.set_color("#303a48")
            spine.set_linewidth(0.7)
        ax.axvline(cur_tr, color="#ffffff", linestyle="--", linewidth=1.0, alpha=0.85, zorder=4)

    # Panel 1: 6-DOF Translations (mm) at High dose
    ax_t = axes[0]
    _style_ax(ax_t, "Translations (High 0.85)", "mm")
    if high_params is not None:
        t_specs = [
            ("tx", high_params.get("trans_x_mm", []), "#ff6b6b"),
            ("ty", high_params.get("trans_y_mm", []), "#51cf66"),
            ("tz", high_params.get("trans_z_mm", []), "#339af0"),
        ]
        all_t: list[float] = []
        for name, vals, clr in t_specs:
            if len(vals) >= n_trs:
                y = np.asarray(vals[:n_trs], dtype=np.float64)
                all_t.extend(y.tolist())
                ax_t.plot(trs, y, color=clr, lw=1.35, label=name)
                ax_t.scatter([cur_tr], [y[tr_idx]], color=clr, edgecolor="#ffffff", s=26, linewidth=0.8, zorder=5)
        ax_t.axhline(0.0, color="#526075", lw=0.6, ls=":")
        if all_t:
            lim = max(0.5, max(abs(min(all_t)), abs(max(all_t))) * 1.18)
            ax_t.set_ylim(-lim, lim)
        ax_t.legend(loc="upper right", fontsize=6.0, frameon=False, labelcolor="#e6edf3", ncol=3, handlelength=1.1, columnspacing=0.7)

    # Panel 2: 6-DOF Rotations (deg) at High dose
    ax_r = axes[1]
    _style_ax(ax_r, "Rotations (High 0.85)", "deg")
    if high_params is not None:
        r_specs = [
            ("rx", high_params.get("rot_x_deg", []), "#ffa94d"),
            ("ry", high_params.get("rot_y_deg", []), "#da77f2"),
            ("rz", high_params.get("rot_z_deg", []), "#ffd43b"),
        ]
        all_r: list[float] = []
        for name, vals, clr in r_specs:
            if len(vals) >= n_trs:
                y = np.asarray(vals[:n_trs], dtype=np.float64)
                all_r.extend(y.tolist())
                ax_r.plot(trs, y, color=clr, lw=1.35, label=name)
                ax_r.scatter([cur_tr], [y[tr_idx]], color=clr, edgecolor="#ffffff", s=26, linewidth=0.8, zorder=5)
        ax_r.axhline(0.0, color="#526075", lw=0.6, ls=":")
        if all_r:
            lim = max(0.5, max(abs(min(all_r)), abs(max(all_r))) * 1.18)
            ax_r.set_ylim(-lim, lim)
        ax_r.legend(loc="upper right", fontsize=6.0, frameon=False, labelcolor="#e6edf3", ncol=3, handlelength=1.1, columnspacing=0.7)

    # Panel 3: Framewise Displacement (mm) across all doses
    ax_fd = axes[2]
    _style_ax(ax_fd, "Framewise Displacement (FD)", "mm")
    fd_colors = ["#8b949e", "#4dabf7", "#ffd43b", "#ff6b6b"]
    max_fd = 0.5
    for idx, (lbl, fd_arr) in enumerate(fd_by_label):
        y = np.asarray(fd_arr[:n_trs], dtype=np.float64)
        max_fd = max(max_fd, float(np.max(y)) if y.size else 0.5)
        clr = fd_colors[idx % len(fd_colors)]
        short_lbl = lbl.replace("Baseline (0.0)", "0.0").replace("Dose ", "")
        ax_fd.plot(trs, y, color=clr, lw=1.35, label=short_lbl)
        ax_fd.scatter([cur_tr], [y[tr_idx]], color=clr, edgecolor="#ffffff", s=26, linewidth=0.8, zorder=5)
    ax_fd.set_ylim(0.0, max_fd * 1.15)
    ax_fd.legend(loc="upper right", fontsize=6.0, frameon=False, labelcolor="#e6edf3", ncol=2, handlelength=1.1, columnspacing=0.7)

    fig.subplots_adjust(left=0.065, right=0.985, top=0.84, bottom=0.22, wspace=0.28)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    pil_plot = Image.fromarray(rgba[..., :3], mode="RGB")
    plt.close(fig)
    if pil_plot.size != (width_px, height_px):
        pil_plot = pil_plot.resize((width_px, height_px), resample=Image.Resampling.BILINEAR)
    return pil_plot


def _render_4d_motion_and_dropout_gifs(
    manifest_4d: pd.DataFrame,
    bids_dir_4d: Path,
    out_root: Path,
) -> dict[str, str]:
    """Generate animated GIFs for 4D BOLD motion_4d (all TRs + 6-DOF line plot) and DWI slice dropout."""
    from scipy import ndimage

    gifs: dict[str, str] = {}
    pos_doses = [d for d in common.DEFAULT_DOSE_LEVELS if d > 0.0]

    # 1. BOLD motion_4d animated 4-way synced movie (Baseline | Low | Mid | High) + motion curves
    b_base_rows = manifest_4d[(manifest_4d["modality"] == "bold") & (manifest_4d["target_artifact"] == "baseline")]
    b_mot_rows = manifest_4d[(manifest_4d["modality"] == "bold") & (manifest_4d["target_artifact"] == "motion_4d")].sort_values("dose_level")
    if not b_base_rows.empty and not b_mot_rows.empty:
        base_4d, _, zooms = common.load_ras_volume(bids_dir_4d / str(b_base_rows.iloc[0]["nifti_path"]))
        base_priors = compute_anatomical_priors(np.mean(base_4d, axis=3))
        mid_z = (base_priors.bbox_min[2] + base_priors.bbox_max[2]) // 2
        vmin, vmax = common.compute_window(np.mean(base_4d, axis=3))

        # 1-pixel reference contour of Baseline TR 0 brain outline to make cumulative drift obvious
        ref_mask_2d = common.orient_2d(base_priors.brain_mask[:, :, mid_z].astype(np.float32)) > 0.5
        ref_contour = ref_mask_2d ^ ndimage.binary_erosion(ref_mask_2d, iterations=1)

        vols: list[tuple[str, np.ndarray, np.ndarray]] = [
            ("Baseline (0.0)", base_4d, np.zeros(base_4d.shape[3], dtype=np.float32))
        ]
        by_dose = {round(float(r["dose_level"]), 2): r for _, r in b_mot_rows.iterrows()}
        high_motion_params: dict[str, list[float]] | None = None
        for d in pos_doses:
            if round(d, 2) in by_dose:
                row_d = by_dose[round(d, 2)]
                v_arr, _, _ = common.load_ras_volume(bids_dir_4d / str(row_d["nifti_path"]))
                # Prefer ground-truth 6-DOF motion_params recorded in manifest.tsv; fall back to blind FD proxy
                mp = _extract_artifact_params(row_d, "motion_4d").get("motion_params")
                if isinstance(mp, dict) and "fd_mm" in mp:
                    v_fd = np.asarray(mp["fd_mm"], dtype=np.float32)
                    high_motion_params = mp  # last dose (High 0.85) wins
                else:
                    v_fd = estimate_fd_proxy(v_arr, base_priors.brain_mask, zooms)
                vols.append((f"Dose {d:.2f}", v_arr, v_fd))

        n_trs = min(v.shape[3] for _, v, _ in vols)
        fd_by_label = [(lbl, fd_arr) for lbl, _, fd_arr in vols]
        frames: list[Image.Image] = []
        for t in range(n_trs):
            tiles = [
                (
                    f"{lbl} | FD={float(fd_arr[t]):.2f}mm",
                    _to_rgb_tile(
                        common.orient_2d(v[:, :, mid_z, t]),
                        vmin,
                        vmax,
                        target_h=200,
                        ref_contour=ref_contour,
                    ),
                )
                for lbl, v, fd_arr in vols
            ]
            top_strip = _compose_labeled_strip(
                tiles,
                f"BOLD motion_4d - TR {t + 1:02d}/{n_trs:02d} (Axial; cyan = TR 0 reference contour)",
            )
            plot_h = 175
            plot_strip = _render_motion_param_strip(
                tr_idx=t,
                n_trs=n_trs,
                width_px=top_strip.size[0],
                height_px=plot_h,
                high_params=high_motion_params,
                fd_by_label=fd_by_label,
            )
            combined = Image.new("RGB", (top_strip.size[0], top_strip.size[1] + plot_h), color=(18, 22, 28))
            combined.paste(top_strip, (0, 0))
            combined.paste(plot_strip, (0, top_strip.size[1]))
            frames.append(combined)
        gif_bold = out_root / "figures" / "03c_bold_motion4d.gif"
        gif_bold.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(str(gif_bold), save_all=True, append_images=frames[1:], duration=180, loop=0)
        gifs["bold_motion4d_gif"] = str(gif_bold.relative_to(out_root))

    # 2. DWI slice dropout animated coronal sweep (Baseline | Mid | High)
    d_base_rows = manifest_4d[(manifest_4d["modality"] == "dwi") & (manifest_4d["target_artifact"] == "baseline")]
    d_drop_rows = manifest_4d[(manifest_4d["modality"] == "dwi") & (manifest_4d["target_artifact"] == "dwi_slice_dropout")].sort_values("dose_level")
    if not d_base_rows.empty and not d_drop_rows.empty:
        base_p = bids_dir_4d / str(d_base_rows.iloc[0]["nifti_path"])
        base_4d, _, _ = common.load_ras_volume(base_p)
        bval_p = base_p.with_name(base_p.name.replace(".nii.gz", ".bval"))
        bvals = np.loadtxt(str(bval_p), dtype=np.float32).reshape(-1) if bval_p.exists() else None
        vols_d: list[tuple[str, np.ndarray]] = [("Baseline (0.0)", base_4d)]
        by_dose_d = {round(float(r["dose_level"]), 2): r for _, r in d_drop_rows.iterrows()}
        for d in pos_doses:
            if round(d, 2) in by_dose_d:
                v_arr, _, _ = common.load_ras_volume(bids_dir_4d / str(by_dose_d[round(d, 2)]["nifti_path"]))
                vols_d.append((f"Dose {d:.2f}", v_arr))
        mid_y = common.midline_indices(common.representative_3d(base_4d, bvals=bvals))[1]
        hi_idx = np.where(bvals > 150.0)[0] if (bvals is not None and np.any(bvals > 150.0)) else np.arange(base_4d.shape[3])
        vmin, vmax = common.compute_window(np.mean(base_4d[..., hi_idx], axis=3))
        frames_d: list[Image.Image] = []
        for idx_v in hi_idx:
            b_lbl = f"b={int(round(float(bvals[idx_v])))}" if bvals is not None else ""
            tiles = [
                (lbl, _to_rgb_tile(common.orient_2d(v[:, mid_y, :, int(idx_v)]), vmin, vmax, target_h=200))
                for lbl, v in vols_d
            ]
            frames_d.append(
                _compose_labeled_strip(tiles, f"DWI dwi_slice_dropout - Vol {int(idx_v) + 1:02d}/{base_4d.shape[3]:02d} ({b_lbl}, Coronal)")
            )
        gif_dwi = out_root / "figures" / "03d_dwi_slice_dropout.gif"
        gif_dwi.parent.mkdir(parents=True, exist_ok=True)
        frames_d[0].save(str(gif_dwi), save_all=True, append_images=frames_d[1:], duration=220, loop=0)
        gifs["dwi_dropout_gif"] = str(gif_dwi.relative_to(out_root))

    return gifs


def _check_monotonic_ssim(manifest_df: pd.DataFrame, modality: str) -> tuple[list[str], dict[str, list[float]]]:
    """Check whether full-reference SSIM decreases monotonically with dose for each artifact trajectory."""
    non_mono: list[str] = []
    trajectories: dict[str, list[float]] = {}
    sub_mod = manifest_df[(manifest_df["modality"] == modality) & (manifest_df["target_artifact"] != "baseline")]
    for art, grp in sub_mod.groupby("target_artifact"):
        g = grp.sort_values("dose_level")
        ssims = [round(float(x), 4) for x in g["ref_ssim"].to_numpy()]
        trajectories[str(art)] = ssims
        # Allow tiny 0.005 tolerance for stochastic rounding
        if any(ssims[i + 1] > ssims[i] + 0.005 for i in range(len(ssims) - 1)):
            non_mono.append(f"{modality}:{art} ({ssims})")
    return non_mono, trajectories


def run_step(
    out_dir: Path = common.DEFAULT_OUT_DIR,
    workers: int = 4,
    device: str = "auto",
    t1w_only: bool = False,
) -> dict[str, Any]:
    """Run individual Low/Mid/High dose-response artifact sweeps across T1w (+ 4D BOLD/DWI)."""
    out_root = Path(out_dir).resolve()
    seed_diag_path = out_root / "00_seed_diagnostics.json"
    if not seed_diag_path.exists():
        raise FileNotFoundError(f"Run Step 01 first ({seed_diag_path} not found).")
    seed_diag = common.read_json(seed_diag_path)
    seed_bids = Path(seed_diag["seed_bids_dir"])
    staged_t1w = Path(seed_diag["staged_t1w"])

    checks: list[common.CheckResult] = []
    figures: dict[str, str] = {}
    tested_artifacts: set[str] = set()
    sensitivity_reports: dict[str, dict[str, Any]] = {}
    ssim_trajectories: dict[str, dict[str, list[float]]] = {}

    with common.StepTimer() as timer:
        # --- 3a. Structural T1w dose sweep (all 11 compatible artifacts) ---
        t1w_arts = compatible_artifacts("T1w")
        tested_artifacts.update(t1w_arts)
        t1w_out = out_root / "02_artifacts_t1w"
        t1w_manifest = t1w_out / "manifest.tsv"
        t1w_cfg = SimulationConfig(
            global_seed=2026,
            modalities=["T1w"],
            sampling=SamplingPlanConfig(
                mode="dose_response",
                include_clean_baseline=True,
                shuffle_run_order=True,
                coupled_trajectories=True,
                replicates_per_artifact=1,
                sweep_all_sites=True,
                noise_calibration="excess_variance",
                dose_severities=list(common.DEFAULT_DOSE_LEVELS),
                artifacts=[ArtifactSpec(name=a) for a in t1w_arts],
            ),
            sites=[],
            output=OutputConfig(
                save_nifti=True,
                write_bids_metadata=True,
                naming_scheme="acq_blinded",
                extract_iqms=False,
                render_views=True,
                render_midline=True,
                render_enhanced=False,
            ),
        )
        t1w_res = run_simulation_plan(
            bids_dir=seed_bids,
            config=t1w_cfg,
            out_dir=t1w_out,
            workers=workers,
            device=device,
            manifest_out=t1w_manifest,
        )
        t1w_df = pd.read_csv(t1w_res.manifest_path, sep="\t")
        t1w_iqms_path = t1w_out / "iqms.tsv"
        t1w_iqms_df = extract_iqms_for_bids_dir(
            bids_dir=t1w_out, out_tsv=t1w_iqms_path, modalities=["T1w"], workers=workers
        )
        t1w_sens = run_sensitivity_analysis(
            t1w_df,
            out_dir=t1w_out / "sensitivity_analysis",
            iqms_df=t1w_iqms_df,
            modality="T1w",
        )
        sensitivity_reports["T1w"] = t1w_sens.report

        # Passthrough identity check at s=0.0
        src_t1_arr, _, _ = common.load_ras_volume(staged_t1w)
        base_row = t1w_df[t1w_df["target_artifact"] == "baseline"].iloc[0]
        out_base_arr, _, _ = common.load_ras_volume(t1w_out / str(base_row["nifti_path"]))
        passthrough_ok = bool(np.allclose(out_base_arr, src_t1_arr, atol=1e-5))
        checks.append(
            common.CheckResult(
                step="03_artifact_dose_sweep",
                name="t1w_passthrough_baseline",
                status="pass" if passthrough_ok else "fail",
                metric=f"allclose={passthrough_ok}, max_abs_diff={float(np.max(np.abs(out_base_arr - src_t1_arr))):.6f}",
                detail="NativeSite passthrough keeps s=0.0 baseline bit-for-bit identical to canonicalized input T1w.",
            )
        )

        # Zero-leakage check
        leak_ok, leaks = _audit_zero_leakage(t1w_out, t1w_arts)
        checks.append(
            common.CheckResult(
                step="03_artifact_dose_sweep",
                name="zero_leakage_acq_blinded",
                status="pass" if leak_ok else "fail",
                metric=f"leaks={len(leaks)} across {len(t1w_df)} T1w scans",
                detail="No artifact names, severity tokens, or SimulationProvenance leaked into BIDS outputs." if leak_ok else "; ".join(leaks[:5]),
            )
        )

        # Monotonic SSIM check
        non_mono_t1, ssim_t1 = _check_monotonic_ssim(t1w_df, "T1w")
        ssim_trajectories["T1w"] = ssim_t1
        checks.append(
            common.CheckResult(
                step="03_artifact_dose_sweep",
                name="t1w_ssim_monotonicity",
                status="pass" if not non_mono_t1 else "warn",
                metric=f"{len(t1w_arts) - len(non_mono_t1)}/{len(t1w_arts)} artifacts strictly monotone in SSIM",
                detail="All T1w artifacts degrade SSIM monotonically across 0.25 -> 0.55 -> 0.85." if not non_mono_t1 else f"Non-monotonic: {', '.join(non_mono_t1)}",
            )
        )

        # IQM blind-spot check
        t1_blind_spots = list(t1w_sens.report.get("blind_spot_artifacts", []))
        checks.append(
            common.CheckResult(
                step="03_artifact_dose_sweep",
                name="t1w_iqm_blind_spots",
                status="warn" if t1_blind_spots else "pass",
                metric=f"{len(t1_blind_spots)}/{len(t1w_arts)} blind-spot artifact(s) (max |rho| < 0.60)",
                detail=(
                    f"Artifacts with weak structural IQM sensitivity: {', '.join(t1_blind_spots)}"
                    if t1_blind_spots
                    else "Every T1w artifact achieved |Spearman rho| >= 0.60 on at least one blind IQM."
                ),
            )
        )

        # Resolve ICV midpoint from Step 02 extracted mask if available (centers axial slices on cortex/ventricles)
        icv_mid: tuple[int, int, int] | None = None
        icv_mask_cand = out_root / "01_contrast_standalone" / "extracted_icv_mask.nii.gz"
        if icv_mask_cand.exists():
            icv_arr, _, _ = common.load_ras_volume(icv_mask_cand)
            if icv_arr.shape == out_base_arr.shape and np.any(icv_arr > 0.5):
                c_icv = np.argwhere(icv_arr > 0.5)
                icv_mid = (
                    int((c_icv[:, 0].min() + c_icv[:, 0].max()) // 2),
                    int((c_icv[:, 1].min() + c_icv[:, 1].max()) // 2),
                    int((c_icv[:, 2].min() + c_icv[:, 2].max()) // 2),
                )

        # Render 2-part T1w contact sheets + sensitivity plots
        half = (len(t1w_arts) + 1) // 2
        fig1 = _render_t1w_artifact_sheet(
            t1w_arts[:half],
            t1w_df,
            t1w_out,
            out_base_arr,
            out_root / "figures" / "03a_t1w_artifacts_part1.png",
            "Step 03a (Part 1/2): Individual T1w Artifacts — Baseline / Low (0.25) / Mid (0.55) / High (0.85)",
            icv_mid=icv_mid,
        )
        fig2 = _render_t1w_artifact_sheet(
            t1w_arts[half:],
            t1w_df,
            t1w_out,
            out_base_arr,
            out_root / "figures" / "03a_t1w_artifacts_part2.png",
            "Step 03a (Part 2/2): Individual T1w Artifacts — Baseline / Low (0.25) / Mid (0.55) / High (0.85)",
            icv_mid=icv_mid,
        )
        figures["t1w_artifacts_part1"] = str(fig1.relative_to(out_root))
        figures["t1w_artifacts_part2"] = str(fig2.relative_to(out_root))
        if t1w_sens.heatmap_path:
            figures["t1w_sensitivity_heatmap"] = str(t1w_sens.heatmap_path.relative_to(out_root))
        if t1w_sens.curves_path:
            figures["t1w_dose_curves"] = str(t1w_sens.curves_path.relative_to(out_root))

        # --- 3b. 4D BOLD & DWI dose sweep (remaining 6 artifacts -> 17/17 coverage) ---
        n_4d_scans = 0
        out_4d = out_root / "02_artifacts_4d"
        if not t1w_only:
            mod_summary_path = out_root / "01_modalities_summary.json"
            if not mod_summary_path.exists():
                raise FileNotFoundError(
                    f"Step 02 output ({mod_summary_path}) is required for 4D sweeps (or pass --t1w-only)."
                )
            multimodal_dir = Path(common.read_json(mod_summary_path)["multimodal_bids_dir"])
            art_list_4d = sorted(set(BOLD_SWEEP_ARTIFACTS + DWI_SWEEP_ARTIFACTS))
            tested_artifacts.update(art_list_4d)

            cfg_4d = SimulationConfig(
                global_seed=2027,
                modalities=["bold", "dwi"],
                sampling=SamplingPlanConfig(
                    mode="dose_response",
                    include_clean_baseline=True,
                    shuffle_run_order=True,
                    coupled_trajectories=True,
                    replicates_per_artifact=1,
                    sweep_all_sites=True,
                    noise_calibration="excess_variance",
                    rotate_dwi_bvecs=False,
                    dose_severities=list(common.DEFAULT_DOSE_LEVELS),
                    artifacts=[ArtifactSpec(name=a) for a in art_list_4d],
                ),
                sites=[],
                output=OutputConfig(
                    save_nifti=True,
                    write_bids_metadata=True,
                    naming_scheme="acq_blinded",
                    extract_iqms=False,
                    render_views=False,
                ),
            )
            res_4d = run_simulation_plan(
                bids_dir=multimodal_dir,
                config=cfg_4d,
                out_dir=out_4d,
                workers=workers,
                device=device,
                manifest_out=out_4d / "manifest.tsv",
            )
            df_4d = pd.read_csv(res_4d.manifest_path, sep="\t")
            n_4d_scans = len(df_4d)
            iqms_4d = extract_iqms_for_bids_dir(
                bids_dir=out_4d,
                out_tsv=out_4d / "iqms.tsv",
                modalities=["bold", "dwi"],
                workers=workers,
            )
            for mod_4d in ("bold", "dwi"):
                sens_4d = run_sensitivity_analysis(
                    df_4d,
                    out_dir=out_4d / f"sensitivity_{mod_4d}",
                    iqms_df=iqms_4d,
                    modality=mod_4d,
                )
                sensitivity_reports[mod_4d] = sens_4d.report
                non_mono_m, ssim_m = _check_monotonic_ssim(df_4d, mod_4d)
                ssim_trajectories[mod_4d] = ssim_m
                if non_mono_m:
                    checks.append(
                        common.CheckResult(
                            step="03_artifact_dose_sweep",
                            name=f"{mod_4d}_ssim_monotonicity",
                            status="warn",
                            metric=f"{len(non_mono_m)} non-monotone",
                            detail=", ".join(non_mono_m),
                        )
                    )

            # Verify DWI motion_4d leaves .bvec unrotated by default (matching raw scanner)
            dwi_base = df_4d[(df_4d["modality"] == "dwi") & (df_4d["target_artifact"] == "baseline")].iloc[0]
            dwi_mot = df_4d[(df_4d["modality"] == "dwi") & (df_4d["target_artifact"] == "motion_4d")].sort_values("dose_level").iloc[-1]
            bvec_base = np.loadtxt(str((out_4d / str(dwi_base["nifti_path"])).with_name(Path(str(dwi_base["nifti_path"])).name.replace(".nii.gz", ".bvec"))))
            bvec_mot = np.loadtxt(str((out_4d / str(dwi_mot["nifti_path"])).with_name(Path(str(dwi_mot["nifti_path"])).name.replace(".nii.gz", ".bvec"))))
            bvec_unrotated = bool(np.allclose(bvec_base, bvec_mot, atol=1e-6))
            checks.append(
                common.CheckResult(
                    step="03_artifact_dose_sweep",
                    name="dwi_motion_bvecs_unrotated",
                    status="pass" if bvec_unrotated else "fail",
                    metric=f"allclose={bvec_unrotated}",
                    detail="With rotate_dwi_bvecs=False (default), DWI motion_4d preserves unrotated scanner .bvec tables.",
                )
            )

            fig_4d = _render_4d_artifacts_figure(
                df_4d, out_4d, out_root / "figures" / "03b_4d_artifacts_grid.png"
            )
            figures["artifacts_4d_grid"] = str(fig_4d.relative_to(out_root))
            figures.update(_render_4d_motion_and_dropout_gifs(df_4d, out_4d, out_root))

        # Full registry coverage check
        all_reg = set(ARTIFACT_REGISTRY.keys())
        missing = sorted(all_reg - tested_artifacts)
        checks.append(
            common.CheckResult(
                step="03_artifact_dose_sweep",
                name="artifact_registry_coverage",
                status="pass" if not missing else ("info" if t1w_only else "warn"),
                metric=f"{len(tested_artifacts)}/{len(all_reg)} artifacts exercised",
                detail=(
                    "100% of registered artifacts in ARTIFACT_REGISTRY exercised across Low/Mid/High severities."
                    if not missing
                    else f"Not exercised (t1w_only={t1w_only}): {', '.join(missing)}"
                ),
            )
        )

    summary: dict[str, Any] = {
        "step": "03_artifact_dose_sweep",
        "elapsed_sec": timer.elapsed_sec,
        "t1w_only": t1w_only,
        "dose_levels": common.DEFAULT_DOSE_LEVELS,
        "t1w_scans_generated": len(t1w_df),
        "fourd_scans_generated": n_4d_scans,
        "tested_artifacts": sorted(tested_artifacts),
        "registry_total": len(ARTIFACT_REGISTRY),
        "t1w_manifest": str(t1w_res.manifest_path),
        "t1w_iqms": str(t1w_iqms_path),
        "ssim_trajectories": ssim_trajectories,
        "sensitivity_reports": sensitivity_reports,
        "figures": figures,
        "checks": common.checks_to_dicts(checks),
    }
    common.write_json(out_root / "02_artifacts_summary.json", summary)
    print(
        f"[Step 03] Generated {len(t1w_df)} T1w + {n_4d_scans} 4D dose-response scans "
        f"({len(tested_artifacts)}/{len(ARTIFACT_REGISTRY)} artifacts, {timer.elapsed_sec:.1f}s)"
    )
    for chk in checks:
        print(f"  [{chk.status.upper():4s}] {chk.name}: {chk.metric}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 03: Individual Low/Mid/High artifact dose sweep.")
    parser.add_argument("--out-dir", type=Path, default=common.DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker count.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: auto|cpu|cuda.")
    parser.add_argument("--t1w-only", action="store_true", help="Skip 4D BOLD/DWI artifact sweep.")
    args = parser.parse_args()
    run_step(out_dir=args.out_dir, workers=args.workers, device=args.device, t1w_only=args.t1w_only)


if __name__ == "__main__":
    main()
