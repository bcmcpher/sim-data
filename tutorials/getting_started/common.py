"""Shared helpers for the ghibli Getting Started tutorial & verification suite."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from ghibli.io.orientation import canonicalize_to_ras, compute_anatomical_priors
from ghibli.phantoms.synthetic import generate_synthetic_bids_cohort

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MRART_T1W = (
    REPO_ROOT.parent
    / "data"
    / "mrart"
    / "bids"
    / "sub-000103"
    / "anat"
    / "sub-000103_acq-standard_T1w.nii.gz"
)
DEFAULT_SITES_CATALOG = REPO_ROOT / "configs" / "sites_catalog.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "tutorial_output" / "getting_started"

# Standard Low / Mid / High dose schedule used across Step 03
DEFAULT_DOSE_LEVELS: list[float] = [0.0, 0.25, 0.55, 0.85]


@dataclass
class CheckResult:
    """Structured empirical verification check emitted by each tutorial step."""

    step: str
    name: str
    status: str  # "pass" | "warn" | "fail" | "info"
    metric: str
    detail: str


class StepTimer:
    """Context manager measuring wall-clock runtime of a step or sub-stage."""

    def __init__(self) -> None:
        self.t0: float = 0.0
        self.elapsed_sec: float = 0.0

    def __enter__(self) -> StepTimer:
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *_args: object) -> None:
        self.elapsed_sec = round(time.perf_counter() - self.t0, 2)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def checks_to_dicts(checks: list[CheckResult]) -> list[dict[str, Any]]:
    return [asdict(c) for c in checks]


def resolve_input_t1w(
    t1w: Path | None,
    use_phantom: bool,
    out_dir: Path,
    phantom_seed: int = 123,
) -> tuple[Path, Path | None, str]:
    """Resolve input T1w NIfTI + optional sibling JSON sidecar + provenance label.

    Priority:
    1. Explicit `--t1w` path (works with any 3D T1w NIfTI).
    2. `--use-phantom` (generates a clean synthetic T1w via `ghibli init-phantom`).
    3. Local MR-ART no-motion example (`sub-000103_acq-standard_T1w.nii.gz`) if present.
    """
    if use_phantom:
        phantom_dir = out_dir / "_phantom_source"
        generate_synthetic_bids_cohort(out_dir=phantom_dir, n_subjects=1, seed=phantom_seed)
        t1_path = phantom_dir / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
        json_path = phantom_dir / "sub-01" / "anat" / "sub-01_T1w.json"
        return t1_path, json_path if json_path.exists() else None, "synthetic_phantom"

    if t1w is not None:
        resolved = t1w.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Input T1w NIfTI not found: {resolved}")
        sibling_json = find_sibling_json(resolved)
        return resolved, sibling_json, "user_t1w"

    if DEFAULT_MRART_T1W.exists():
        sibling_json = find_sibling_json(DEFAULT_MRART_T1W)
        return DEFAULT_MRART_T1W.resolve(), sibling_json, "mrart_default_sub000103"

    raise FileNotFoundError(
        "No --t1w path provided and default MR-ART scan was not found at "
        f"{DEFAULT_MRART_T1W}. Pass --t1w /path/to/your_T1w.nii.gz or --use-phantom."
    )


def find_sibling_json(nifti_path: Path) -> Path | None:
    """Locate the matching BIDS .json sidecar alongside a .nii or .nii.gz file."""
    name = nifti_path.name
    if name.endswith(".nii.gz"):
        cand = nifti_path.with_name(name[:-7] + ".json")
    elif name.endswith(".nii"):
        cand = nifti_path.with_name(name[:-4] + ".json")
    else:
        cand = nifti_path.with_suffix(".json")
    return cand if cand.exists() else None


def load_ras_volume(nifti_path: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Load any NIfTI file and canonicalize to RAS+ float32 array."""
    img = nib.load(str(nifti_path))
    if not isinstance(img, nib.Nifti1Image):
        img = nib.Nifti1Image(np.asarray(img.get_fdata(), dtype=np.float32), img.affine)
    return canonicalize_to_ras(img)


def representative_3d(data: np.ndarray, bvals: np.ndarray | None = None) -> np.ndarray:
    """Return a single representative 3D volume (mean b=0 for DWI, mean across time for BOLD)."""
    if data.ndim == 3:
        return data
    if bvals is not None and len(bvals) == data.shape[3]:
        b0_idx = np.where(bvals <= 50.0)[0]
        if b0_idx.size > 0:
            return np.asarray(np.mean(data[..., b0_idx], axis=3), dtype=np.float32)
    return np.asarray(np.mean(data, axis=3), dtype=np.float32)


def orient_2d(sl: np.ndarray) -> np.ndarray:
    """Standard radiological-like 2D orientation with Superior/Anterior toward top."""
    return np.flipud(sl.T)


def midline_indices(vol_3d: np.ndarray) -> tuple[int, int, int]:
    """Compute foreground bounding-box midpoint (mid_x, mid_y, mid_z)."""
    priors = compute_anatomical_priors(vol_3d)
    mid_x = (priors.bbox_min[0] + priors.bbox_max[0]) // 2
    mid_y = (priors.bbox_min[1] + priors.bbox_max[1]) // 2
    mid_z = (priors.bbox_min[2] + priors.bbox_max[2]) // 2
    return int(mid_x), int(mid_y), int(mid_z)


def compute_window(vol: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 98.5) -> tuple[float, float]:
    """Robust foreground intensity window (vmin, vmax)."""
    a = np.nan_to_num(vol.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    pos = a[a > 0]
    if pos.size < 16:
        return (0.0, 1.0)
    lo = float(np.percentile(pos, lo_pct))
    hi = float(np.percentile(pos, hi_pct))
    if hi <= lo + 1e-6:
        hi = lo + 1.0
    return (lo, hi)


def normalize_slice(sl: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Linearly map a 2D slice into [0, 1] using fixed (vmin, vmax)."""
    return np.clip((sl.astype(np.float32) - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)


def bg_boost_slice(sl: np.ndarray, ref_p95: float | None = None) -> np.ndarray:
    """Log-gamma background-boosted [0, 1] slice exposing ghosting, spikes, and noise floor."""
    a = np.maximum(np.nan_to_num(sl.astype(np.float32), nan=0.0), 0.0)
    if ref_p95 is None or ref_p95 <= 1e-6:
        pos = a[a > 0]
        ref_p95 = float(np.percentile(pos, 95.0)) if pos.size > 8 else 1.0
    norm = np.clip(a / max(ref_p95, 1e-6), 0.0, 1.5)
    boosted = np.log1p(22.0 * norm) / np.log1p(22.0 * 1.5)
    return np.asarray(np.clip(boosted, 0.0, 1.0), dtype=np.float32)


def save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), bbox_inches="tight", dpi=130)
    plt.close(fig)
    return path
