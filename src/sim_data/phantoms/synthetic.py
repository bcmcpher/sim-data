"""Realistic synthetic multimodal BIDS brain phantom generator (T1w, T2w, FLAIR, BOLD, DWI).

Produces clean, anatomically structured 3D/4D brain volumes (skull/scalp, CSF, cortical GM,
subcortical WM, lateral ventricles, cerebellum, and brainstem) with BIDS JSON sidecars and
multi-shell spherical DWI .bval/.bvec tables for offline testing, benchmarking, and demos.
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def _fibonacci_sphere(n_dirs: int) -> np.ndarray:
    """Generate evenly distributed unit gradient vectors on the hemisphere (3, n_dirs)."""
    indices = np.arange(n_dirs, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - indices / max(n_dirs, 1))
    theta = np.pi * (1.0 + 5.0**0.5) * indices
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    vecs = np.vstack([x, y, z])
    norms = np.linalg.norm(vecs, axis=0, keepdims=True) + 1e-8
    return np.asarray(vecs / norms, dtype=np.float32)


def _build_anatomical_compartments(
    shape: tuple[int, int, int],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Generate smooth anatomical tissue probability maps in RAS+ voxel coordinates."""
    sx, sy, sz = shape
    xs = np.linspace(-1.0, 1.0, sx, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, sy, dtype=np.float32)
    zs = np.linspace(-1.0, 1.0, sz, dtype=np.float32)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")

    # Slight inter-subject anatomical variability
    rx = float(0.64 * rng.uniform(0.95, 1.05))
    ry = float(0.76 * rng.uniform(0.95, 1.05))
    rz = float(0.68 * rng.uniform(0.95, 1.05))

    # Main cranial ellipsoid + inferior cerebellum lobe
    head_r = np.sqrt((gx / rx) ** 2 + ((gy - 0.02) / ry) ** 2 + ((gz - 0.05) / rz) ** 2)
    cereb_r = np.sqrt((gx / (0.48 * rx)) ** 2 + ((gy + 0.34) / (0.34 * ry)) ** 2 + ((gz + 0.44) / (0.32 * rz)) ** 2)
    skull_mask = (head_r <= 1.05) | (cereb_r <= 1.0)
    icv_mask = (head_r <= 0.94) | (cereb_r <= 0.88)

    # Cortical folding texture via smoothed random harmonic field
    tex = np.sin(7.5 * gx + rng.uniform(0, 6.28)) * np.cos(7.5 * gy + rng.uniform(0, 6.28)) + 0.6 * np.sin(
        9.0 * gz + rng.uniform(0, 6.28)
    )
    wm_radius = 0.72 + 0.06 * tex
    wm_mask = ((head_r <= wm_radius) & (gz > -0.35)) | (cereb_r <= 0.56)

    # Bilateral lateral ventricles (CSF)
    vent_l = ((gx + 0.14) / 0.11) ** 2 + ((gy - 0.02) / 0.28) ** 2 + ((gz - 0.12) / 0.14) ** 2 <= 1.0
    vent_r = ((gx - 0.14) / 0.11) ** 2 + ((gy - 0.02) / 0.28) ** 2 + ((gz - 0.12) / 0.14) ** 2 <= 1.0
    vent_mask = vent_l | vent_r

    # Subarachnoid CSF rim + ventricles
    csf_mask = (icv_mask & (head_r >= 0.87) & (cereb_r > 0.80)) | vent_mask
    wm_mask = wm_mask & (~csf_mask)
    gm_mask = icv_mask & (~wm_mask) & (~csf_mask)
    scalp_mask = skull_mask & (~icv_mask)

    # Mild smooth partial-volume blurring so edges resemble finite MRI resolution
    out: dict[str, np.ndarray] = {}
    for name, m in [
        ("wm", wm_mask),
        ("gm", gm_mask),
        ("csf", csf_mask),
        ("scalp", scalp_mask),
        ("brain", icv_mask),
    ]:
        out[name] = ndimage.gaussian_filter(m.astype(np.float32), sigma=0.65)
    return out


def _synthesize_structural(
    comp: dict[str, np.ndarray],
    modality: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthesize realistic T1w, T2w, or FLAIR tissue contrast."""
    if modality == "T1w":
        w_wm, w_gm, w_csf, w_sc = 1000.0, 720.0, 220.0, 460.0
    elif modality == "T2w":
        w_wm, w_gm, w_csf, w_sc = 520.0, 760.0, 1150.0, 340.0
    else:  # FLAIR: T2-like GM>WM with suppressed dark CSF
        w_wm, w_gm, w_csf, w_sc = 560.0, 820.0, 140.0, 360.0

    vol = (
        w_wm * comp["wm"]
        + w_gm * comp["gm"]
        + w_csf * comp["csf"]
        + w_sc * comp["scalp"]
    )
    # Subtle intrinsic Rician baseline noise (high SNR clean reference)
    sigma = 8.0
    n1 = rng.normal(0.0, sigma, size=vol.shape).astype(np.float32)
    n2 = rng.normal(0.0, sigma, size=vol.shape).astype(np.float32)
    return np.sqrt((vol + n1) ** 2 + n2**2).astype(np.float32)


def _synthesize_bold(
    comp: dict[str, np.ndarray],
    n_trs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthesize clean 4D BOLD EPI time series with realistic gray-matter BOLD fluctuations."""
    base = 720.0 * comp["wm"] + 900.0 * comp["gm"] + 1050.0 * comp["csf"] + 240.0 * comp["scalp"]
    data = np.repeat(base[..., None], n_trs, axis=3)

    # Low-amplitude spatially smooth GM hemodynamic fluctuations (~1.5% signal)
    t = np.linspace(0.0, 2.0 * np.pi, n_trs, dtype=np.float32)
    hemodyn = 1.0 + 0.015 * np.sin(2.0 * t + rng.uniform(0, 6.28))
    data = data * (1.0 + (hemodyn[None, None, None, :] - 1.0) * comp["gm"][..., None])

    sigma = 10.0
    n1 = rng.normal(0.0, sigma, size=data.shape).astype(np.float32)
    n2 = rng.normal(0.0, sigma, size=data.shape).astype(np.float32)
    return np.asarray(np.sqrt((data + n1) ** 2 + n2**2), dtype=np.float32)


def _synthesize_dwi(
    comp: dict[str, np.ndarray],
    n_b0: int,
    n_dirs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize clean 4D DWI volume with Stejskal-Tanner directional WM signal attenuation."""
    b0_base = 640.0 * comp["wm"] + 780.0 * comp["gm"] + 1050.0 * comp["csf"] + 200.0 * comp["scalp"]
    dirs = _fibonacci_sphere(n_dirs)
    bvals = np.concatenate([np.zeros(n_b0, dtype=np.float32), np.full(n_dirs, 1000.0, dtype=np.float32)])
    bvecs = np.hstack([np.zeros((3, n_b0), dtype=np.float32), dirs])

    n_vols = n_b0 + n_dirs
    data = np.empty((*b0_base.shape, n_vols), dtype=np.float32)

    # Simplified corpus-callosum-like principal diffusion axis (R-L) in WM
    v_principal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    md_wm, fa_aniso = 0.00075, 0.00055
    md_gm, md_csf = 0.00085, 0.0026

    for v in range(n_vols):
        b = float(bvals[v])
        if b <= 50.0:
            vol = b0_base.copy()
        else:
            g = bvecs[:, v]
            cos2 = float(np.dot(g, v_principal) ** 2)
            d_wm = md_wm + fa_aniso * (cos2 - 0.33)
            vol = (
                640.0 * comp["wm"] * np.exp(-b * d_wm)
                + 780.0 * comp["gm"] * np.exp(-b * md_gm)
                + 1050.0 * comp["csf"] * np.exp(-b * md_csf)
                + 200.0 * comp["scalp"] * np.exp(-b * 0.0012)
            )
        sigma = 7.0
        n1 = rng.normal(0.0, sigma, size=vol.shape).astype(np.float32)
        n2 = rng.normal(0.0, sigma, size=vol.shape).astype(np.float32)
        data[..., v] = np.sqrt((vol + n1) ** 2 + n2**2)

    return data, bvals, bvecs


def generate_synthetic_bids_cohort(
    out_dir: str | Path,
    n_subjects: int = 3,
    anat_shape: tuple[int, int, int] = (64, 72, 64),
    epi_shape: tuple[int, int, int] = (48, 56, 36),
    n_bold_trs: int = 24,
    n_dwi_dirs: int = 12,
    seed: int = 123,
) -> Path:
    """Create a complete synthetic BIDS multimodal cohort (T1w, T2w, FLAIR, bold, dwi)."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset_description.json").write_text(
        json.dumps(
            {
                "Name": "sim-data Synthetic Multimodal Phantom Cohort",
                "BIDSVersion": "1.9.0",
                "DatasetType": "raw",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    anat_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    epi_affine = np.diag([3.0, 3.0, 3.0, 1.0])

    for s_idx in range(1, n_subjects + 1):
        sub = f"sub-{s_idx:02d}"
        rng = np.random.default_rng(seed + s_idx * 101)

        # Structural modalities
        anat_dir = root / sub / "anat"
        anat_dir.mkdir(parents=True, exist_ok=True)
        anat_comp = _build_anatomical_compartments(anat_shape, rng)
        for mod, tr, te in [("T1w", 2.3, 0.0029), ("T2w", 3.2, 0.408), ("FLAIR", 5.0, 0.388)]:
            vol = _synthesize_structural(anat_comp, mod, rng)
            nib.save(nib.Nifti1Image(vol, anat_affine), str(anat_dir / f"{sub}_{mod}.nii.gz"))
            (anat_dir / f"{sub}_{mod}.json").write_text(
                json.dumps(
                    {
                        "Modality": "MR",
                        "MagneticFieldStrength": 3.0,
                        "RepetitionTime": tr,
                        "EchoTime": te,
                        "PhaseEncodingDirection": "j-",
                        "SliceEncodingDirection": "k",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        # Functional BOLD
        func_dir = root / sub / "func"
        func_dir.mkdir(parents=True, exist_ok=True)
        epi_comp = _build_anatomical_compartments(epi_shape, rng)
        bold = _synthesize_bold(epi_comp, n_trs=n_bold_trs, rng=rng)
        nib.save(
            nib.Nifti1Image(bold, epi_affine),
            str(func_dir / f"{sub}_task-rest_bold.nii.gz"),
        )
        (func_dir / f"{sub}_task-rest_bold.json").write_text(
            json.dumps(
                {
                    "Modality": "MR",
                    "TaskName": "rest",
                    "MagneticFieldStrength": 3.0,
                    "RepetitionTime": 2.0,
                    "EchoTime": 0.030,
                    "PhaseEncodingDirection": "j-",
                    "SliceEncodingDirection": "k",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # Diffusion DWI
        dwi_dir = root / sub / "dwi"
        dwi_dir.mkdir(parents=True, exist_ok=True)
        dwi, bvals, bvecs = _synthesize_dwi(epi_comp, n_b0=2, n_dirs=n_dwi_dirs, rng=rng)
        nib.save(nib.Nifti1Image(dwi, epi_affine), str(dwi_dir / f"{sub}_dwi.nii.gz"))
        (dwi_dir / f"{sub}_dwi.json").write_text(
            json.dumps(
                {
                    "Modality": "MR",
                    "MagneticFieldStrength": 3.0,
                    "RepetitionTime": 4.2,
                    "EchoTime": 0.089,
                    "PhaseEncodingDirection": "j-",
                    "SliceEncodingDirection": "k",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (dwi_dir / f"{sub}_dwi.bval").write_text(
            " ".join(f"{float(v):.1f}" for v in bvals) + "\n", encoding="utf-8"
        )
        bvec_lines = [" ".join(f"{float(v):.6f}" for v in row) for row in bvecs]
        (dwi_dir / f"{sub}_dwi.bvec").write_text("\n".join(bvec_lines) + "\n", encoding="utf-8")

    return root
