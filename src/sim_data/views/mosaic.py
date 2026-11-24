"""2D visual-QC views: qc-studio 3 midline slices + enhanced mosaics, BOLD carpet, DWI dropout."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sim_data.config import Modality
from sim_data.io.orientation import compute_anatomical_priors
from sim_data.iqm.diffusion import compute_bad_slices_qsiprep
from sim_data.iqm.functional import compute_dvars_series, estimate_fd_proxy


def _window_to_uint8(arr: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.5) -> np.ndarray:
    a = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    pos = a[a > 0]
    if pos.size < 8:
        return np.zeros(a.shape, dtype=np.uint8)
    lo = float(np.percentile(pos, lo_pct))
    hi = float(np.percentile(pos, hi_pct))
    if hi <= lo + 1e-6:
        hi = lo + 1.0
    norm = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def _bg_boost_uint8(arr: np.ndarray) -> np.ndarray:
    """Log/gamma-compressed intensity view highlighting ghosting, spikes, wrap-around, and noise floor."""
    a = np.maximum(np.nan_to_num(arr.astype(np.float32), nan=0.0), 0.0)
    p95 = float(np.percentile(a[a > 0], 95.0)) if np.any(a > 0) else 1.0
    norm = np.clip(a / max(p95, 1e-6), 0.0, 1.5)
    # Log-gamma boost lifts subtle 1%-15% background artifacts cleanly into mid-gray
    boosted = np.log1p(22.0 * norm) / np.log1p(22.0 * 1.5)
    return np.asarray(np.clip(boosted, 0.0, 1.0) * 255.0, dtype=np.uint8)


def _orient_slice_2d(sl: np.ndarray) -> np.ndarray:
    """Transpose & flip vertically so superior/anterior appear at top in standard radiological display."""
    return np.flipud(sl.T)


def _save_gray_image(img_u8: np.ndarray, path: Path, target_min_size: int = 256) -> None:
    pil = Image.fromarray(img_u8, mode="L")
    w, h = pil.size
    if min(w, h) < target_min_size and min(w, h) > 0:
        scale = max(1, int(round(target_min_size / min(w, h))))
        pil = pil.resize((w * scale, h * scale), resample=Image.Resampling.BILINEAR)
    pil.save(str(path))


def _pad_to_square(sl: np.ndarray, size: int) -> np.ndarray:
    out = np.zeros((size, size), dtype=sl.dtype)
    h, w = sl.shape
    y0 = max(0, (size - h) // 2)
    x0 = max(0, (size - w) // 2)
    ch = min(h, size)
    cw = min(w, size)
    out[y0 : y0 + ch, x0 : x0 + cw] = sl[:ch, :cw]
    return out


def _render_ortho_grid(vol: np.ndarray, bg_boost: bool, path: Path, n_slices: int = 5) -> None:
    """Render 3-row (axial, coronal, sagittal) x n_slices orthogonal mosaic."""
    priors = compute_anatomical_priors(vol)
    bmin, bmax = priors.bbox_min, priors.bbox_max
    converter = _bg_boost_uint8 if bg_boost else _window_to_uint8

    # Normalize full volume once so slice intensities stay consistent across the mosaic
    vol_u8 = converter(vol)
    max_dim = max(vol.shape)
    rows: list[np.ndarray] = []

    for axis in (2, 1, 0):  # axial (z), coronal (y), sagittal (x)
        lo = max(1, bmin[axis] + int(0.08 * (bmax[axis] - bmin[axis])))
        hi = min(vol.shape[axis] - 2, bmax[axis] - int(0.08 * (bmax[axis] - bmin[axis])))
        if hi <= lo:
            indices = np.linspace(0, vol.shape[axis] - 1, n_slices).astype(int)
        else:
            indices = np.linspace(lo, hi, n_slices).astype(int)

        tiles: list[np.ndarray] = []
        for idx in indices:
            sl = np.take(vol_u8, int(idx), axis=axis)
            tiles.append(_pad_to_square(_orient_slice_2d(sl), max_dim))
        rows.append(np.concatenate(tiles, axis=1))

    grid = np.concatenate(rows, axis=0)
    pil = Image.fromarray(grid, mode="L")
    if grid.shape[1] < 900:
        scale = 900.0 / max(1, grid.shape[1])
        pil = pil.resize(
            (int(round(grid.shape[1] * scale)), int(round(grid.shape[0] * scale))),
            resample=Image.Resampling.BILINEAR,
        )
    pil.save(str(path))


def _render_fmri_carpet_summary(
    data_4d: np.ndarray,
    zooms: tuple[float, float, float],
    path: Path,
) -> None:
    """Render fMRIPrep-inspired BOLD summary: blind FD + std_dvars traces, carpet plot, and tSNR slice."""
    mean_vol = np.mean(data_4d, axis=3)
    priors = compute_anatomical_priors(mean_vol)
    brain = priors.brain_mask

    _, dvars_std, _ = compute_dvars_series(data_4d, brain)
    fd = estimate_fd_proxy(data_4d, brain, zooms)

    ts = data_4d[brain].astype(np.float32)  # (V, T)
    # Sample up to 600 voxels sorted by mean intensity (tissue proxy grouping: CSF/GM/WM)
    v_means = np.mean(ts, axis=1)
    order = np.argsort(v_means)
    if order.size > 600:
        order = order[np.linspace(0, order.size - 1, 600).astype(int)]
    carpet = ts[order]
    c_mean = np.mean(carpet, axis=1, keepdims=True)
    c_std = np.std(carpet, axis=1, keepdims=True) + 1e-5
    carpet_z = np.clip((carpet - c_mean) / c_std, -3.0, 3.0)

    # Voxelwise tSNR midline axial map
    t_std = np.std(data_4d, axis=3) + 1e-5
    tsnr_map = np.where(brain, mean_vol / t_std, 0.0)
    mid_z = (priors.bbox_min[2] + priors.bbox_max[2]) // 2
    tsnr_slice = _orient_slice_2d(tsnr_map[:, :, mid_z])

    fig = plt.figure(figsize=(10, 6), dpi=110)
    gs = fig.add_gridspec(3, 2, width_ratios=[2.6, 1.0], height_ratios=[1.0, 1.0, 2.2], hspace=0.35)

    ax_fd = fig.add_subplot(gs[0, 0])
    ax_fd.plot(fd, color="#1f77b4", lw=1.2)
    ax_fd.axhline(0.2, color="#d62728", ls="--", lw=0.8, label="0.2 mm")
    ax_fd.set_ylabel("FD (mm)", fontsize=8)
    ax_fd.set_title("fMRIPrep-Style BOLD QC Summary", fontsize=10)
    ax_fd.tick_params(labelsize=7)

    ax_dv = fig.add_subplot(gs[1, 0], sharex=ax_fd)
    ax_dv.plot(dvars_std, color="#ff7f0e", lw=1.2)
    ax_dv.set_ylabel("stdDVARS", fontsize=8)
    ax_dv.tick_params(labelsize=7)

    ax_cp = fig.add_subplot(gs[2, 0], sharex=ax_fd)
    ax_cp.imshow(carpet_z, aspect="auto", cmap="gray", vmin=-2.5, vmax=2.5)
    ax_cp.set_ylabel("Voxels (tissue)", fontsize=8)
    ax_cp.set_xlabel("Volume (TR)", fontsize=8)
    ax_cp.tick_params(labelsize=7)

    ax_tsnr = fig.add_subplot(gs[:, 1])
    im = ax_tsnr.imshow(tsnr_slice, cmap="inferno", vmin=0, vmax=max(10.0, float(np.percentile(tsnr_slice[tsnr_slice > 0], 95)) if np.any(tsnr_slice > 0) else 20.0))
    ax_tsnr.set_title("Mid-Axial tSNR", fontsize=9)
    ax_tsnr.axis("off")
    fig.colorbar(im, ax=ax_tsnr, fraction=0.046, pad=0.04)

    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)


def _render_dwi_dropout_summary(
    data_4d: np.ndarray,
    bvals: np.ndarray | None,
    slice_axis: int,
    path: Path,
) -> None:
    """Render QSIPrep-inspired DWI summary: b=0 vs high-b axial slices + Volume x Slice intensity heatmap."""
    n_vols = data_4d.shape[3]
    if bvals is not None and len(bvals) == n_vols:
        b0_idx = np.where(bvals <= 50.0)[0]
        dw_idx = np.where(bvals > 150.0)[0]
    else:
        b0_idx = np.array([0])
        dw_idx = np.arange(1, n_vols) if n_vols > 1 else np.array([0])

    b0_vol = np.mean(data_4d[..., b0_idx], axis=3) if b0_idx.size > 0 else np.mean(data_4d, axis=3)
    dw_vol = np.mean(data_4d[..., dw_idx], axis=3) if dw_idx.size > 0 else b0_vol
    priors = compute_anatomical_priors(b0_vol)
    _, ratio_map = compute_bad_slices_qsiprep(
        data_4d, priors.brain_mask, slice_axis=slice_axis, bvals=bvals
    )

    mid_z = (priors.bbox_min[2] + priors.bbox_max[2]) // 2
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), dpi=110, gridspec_kw={"width_ratios": [1, 1, 1.8]})

    axes[0].imshow(_orient_slice_2d(_window_to_uint8(b0_vol[:, :, mid_z])), cmap="gray")
    axes[0].set_title("Mean b=0 (Axial)", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(_orient_slice_2d(_window_to_uint8(dw_vol[:, :, mid_z])), cmap="gray")
    axes[1].set_title("Mean High-b (Axial)", fontsize=9)
    axes[1].axis("off")

    im = axes[2].imshow(ratio_map.T, aspect="auto", cmap="RdYlBu", vmin=0.35, vmax=1.25, origin="lower")
    axes[2].set_title("Slice Intensity Ratio (Vol x Slice)", fontsize=9)
    axes[2].set_xlabel("DWI Volume", fontsize=8)
    axes[2].set_ylabel("Slice Index", fontsize=8)
    axes[2].tick_params(labelsize=7)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="S_z / neighbor")

    fig.tight_layout()
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)


def render_scan_views(
    data: np.ndarray,
    modality: Modality,
    zooms: tuple[float, float, float],
    out_dir: str | Path,
    scan_id: str,
    image_format: str = "png",
    render_midline: bool = True,
    render_enhanced: bool = True,
    bvals: np.ndarray | None = None,
    slice_axis: int = 2,
) -> dict[str, dict[str, str]]:
    """Render qc-studio midline 3-plane slices and enhanced diagnostic mosaics.

    Returns nested dictionary {"midline": {...}, "enhanced": {...}} with paths relative to out_dir.
    """
    out_root = Path(out_dir)
    views_dir = out_root / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    ext = image_format.lower().lstrip(".")

    # Select representative 3D volume for spatial views
    if data.ndim == 4:
        if modality == "dwi" and bvals is not None and len(bvals) == data.shape[3]:
            b0_idx = np.where(bvals <= 50.0)[0]
            vol_3d = np.mean(data[..., b0_idx], axis=3) if b0_idx.size > 0 else np.mean(data, axis=3)
        else:
            vol_3d = np.mean(data, axis=3)
    else:
        vol_3d = data

    priors = compute_anatomical_priors(vol_3d)
    mid_x = (priors.bbox_min[0] + priors.bbox_max[0]) // 2
    mid_y = (priors.bbox_min[1] + priors.bbox_max[1]) // 2
    mid_z = (priors.bbox_min[2] + priors.bbox_max[2]) // 2

    views_out: dict[str, dict[str, str]] = {"midline": {}, "enhanced": {}}

    if render_midline:
        vol_u8 = _window_to_uint8(vol_3d)
        sag_p = views_dir / f"{scan_id}_mid_sag.{ext}"
        ax_p = views_dir / f"{scan_id}_mid_ax.{ext}"
        cor_p = views_dir / f"{scan_id}_mid_cor.{ext}"
        _save_gray_image(_orient_slice_2d(vol_u8[mid_x, :, :]), sag_p)
        _save_gray_image(_orient_slice_2d(vol_u8[:, :, mid_z]), ax_p)
        _save_gray_image(_orient_slice_2d(vol_u8[:, mid_y, :]), cor_p)
        views_out["midline"] = {
            "sagittal": str(sag_p.relative_to(out_root)),
            "axial": str(ax_p.relative_to(out_root)),
            "coronal": str(cor_p.relative_to(out_root)),
        }

    if render_enhanced:
        ortho_p = views_dir / f"{scan_id}_ortho.{ext}"
        bg_p = views_dir / f"{scan_id}_bg_boost.{ext}"
        _render_ortho_grid(vol_3d, bg_boost=False, path=ortho_p)
        _render_ortho_grid(vol_3d, bg_boost=True, path=bg_p)
        views_out["enhanced"] = {
            "ortho_mosaic": str(ortho_p.relative_to(out_root)),
            "bg_boost_mosaic": str(bg_p.relative_to(out_root)),
        }
        if modality == "bold" and data.ndim == 4 and data.shape[3] >= 3:
            carpet_p = views_dir / f"{scan_id}_fmri_carpet.{ext}"
            _render_fmri_carpet_summary(data, zooms=zooms, path=carpet_p)
            views_out["enhanced"]["fmri_carpet_summary"] = str(carpet_p.relative_to(out_root))
        elif modality == "dwi" and data.ndim == 4 and data.shape[3] >= 2:
            dwi_p = views_dir / f"{scan_id}_dwi_summary.{ext}"
            _render_dwi_dropout_summary(data, bvals=bvals, slice_axis=slice_axis, path=dwi_p)
            views_out["enhanced"]["dwi_shell_dropout_summary"] = str(dwi_p.relative_to(out_root))

    return views_out
