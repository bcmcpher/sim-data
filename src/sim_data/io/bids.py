"""BIDS input discovery and BIDS-derivative writing for multimodal MRI."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from sim_data.config import Modality

_MODALITY_FOLDERS: dict[str, str] = {
    "T1w": "anat",
    "T2w": "anat",
    "FLAIR": "anat",
    "bold": "func",
    "dwi": "dwi",
}

_ENTITY_RE = re.compile(r"(sub|ses|task|acq|run|dir|echo)-([A-Za-z0-9]+)")


@dataclass
class BIDSScan:
    """Reference to a discovered BIDS MRI scan and its companion files."""

    subject: str
    session: str | None
    modality: Modality
    datatype: str
    nifti_path: Path
    json_path: Path | None = None
    bval_path: Path | None = None
    bvec_path: Path | None = None
    entities: dict[str, str] = field(default_factory=dict)

    @property
    def base_id(self) -> str:
        stem = self.nifti_path.name
        if stem.endswith(".nii.gz"):
            return stem[:-7]
        if stem.endswith(".nii"):
            return stem[:-4]
        return self.nifti_path.stem


@dataclass
class LoadedBIDSScan:
    """Loaded NIfTI image with optional JSON sidecar and DWI gradient tables."""

    scan: BIDSScan
    img: nib.Nifti1Image
    sidecar: dict[str, Any]
    bvals: np.ndarray | None = None
    bvecs: np.ndarray | None = None  # shape (3, N)


def _strip_nifti_suffix(path: Path) -> Path:
    name = path.name
    if name.endswith(".nii.gz"):
        return path.with_name(name[:-7])
    if name.endswith(".nii"):
        return path.with_name(name[:-4])
    return path.with_suffix("")


def _parse_modality(path: Path) -> Modality | None:
    stem = _strip_nifti_suffix(path).name
    parts = stem.split("_")
    suffix = parts[-1] if parts else ""
    if suffix in _MODALITY_FOLDERS:
        return suffix  # type: ignore[return-value]
    return None


def discover_bids_scans(
    bids_dir: str | Path,
    modalities: list[Modality] | None = None,
) -> list[BIDSScan]:
    """Recursively discover BIDS MRI scans under a BIDS root directory."""
    root = Path(bids_dir)
    if not root.exists():
        raise FileNotFoundError(f"BIDS directory not found: {root}")

    allowed = set(modalities) if modalities else set(_MODALITY_FOLDERS.keys())
    scans: list[BIDSScan] = []

    for path in sorted(root.rglob("*.nii*")):
        if not (path.name.endswith(".nii.gz") or path.name.endswith(".nii")):
            continue
        # Skip derivatives folder if scanning a BIDS root containing derivatives
        rel_parts = path.relative_to(root).parts
        if "derivatives" in rel_parts:
            continue

        modality = _parse_modality(path)
        if modality is None or modality not in allowed:
            continue

        entities = dict(_ENTITY_RE.findall(path.name))
        subject = entities.get("sub", "unknown")
        session = entities.get("ses")
        datatype = _MODALITY_FOLDERS[modality]

        prefix = _strip_nifti_suffix(path)
        json_p = prefix.with_suffix(".json")
        bval_p = prefix.with_suffix(".bval")
        bvec_p = prefix.with_suffix(".bvec")

        scans.append(
            BIDSScan(
                subject=subject,
                session=session,
                modality=modality,
                datatype=datatype,
                nifti_path=path,
                json_path=json_p if json_p.exists() else None,
                bval_path=bval_p if bval_p.exists() else None,
                bvec_path=bvec_p if bvec_p.exists() else None,
                entities=entities,
            )
        )

    return scans


def load_bids_scan(scan: BIDSScan) -> LoadedBIDSScan:
    """Load NIfTI image, JSON sidecar, and optional DWI .bval/.bvec files."""
    img = nib.load(str(scan.nifti_path))
    if not isinstance(img, nib.Nifti1Image):
        if isinstance(img, nib.spatialimages.SpatialImage):
            img = nib.Nifti1Image(np.asarray(img.get_fdata(), dtype=np.float32), img.affine)
        else:
            raise TypeError(f"Unsupported non-spatial image type: {type(img).__name__}")

    sidecar: dict[str, Any] = {}
    if scan.json_path and scan.json_path.exists():
        sidecar = json.loads(scan.json_path.read_text(encoding="utf-8"))

    bvals: np.ndarray | None = None
    bvecs: np.ndarray | None = None
    if scan.bval_path and scan.bval_path.exists():
        bvals = np.loadtxt(scan.bval_path, dtype=np.float32).reshape(-1)
    if scan.bvec_path and scan.bvec_path.exists():
        raw_bvecs = np.loadtxt(scan.bvec_path, dtype=np.float32)
        if raw_bvecs.ndim == 2:
            if raw_bvecs.shape[0] != 3 and raw_bvecs.shape[1] == 3:
                raw_bvecs = raw_bvecs.T
            bvecs = raw_bvecs

    return LoadedBIDSScan(scan=scan, img=img, sidecar=sidecar, bvals=bvals, bvecs=bvecs)


def _sanitize_entity(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", value)


def build_derivative_stem(
    scan: BIDSScan,
    recipe_id: str,
    site_id: str,
) -> tuple[Path, str]:
    """Construct relative BIDS-derivative directory and filename stem."""
    sub_dir = Path(f"sub-{scan.subject}")
    if scan.session:
        sub_dir = sub_dir / f"ses-{scan.session}"
    rel_dir = sub_dir / scan.datatype

    parts = [f"sub-{scan.subject}"]
    if scan.session:
        parts.append(f"ses-{scan.session}")
    if "task" in scan.entities:
        parts.append(f"task-{scan.entities['task']}")
    parts.append(f"acq-{_sanitize_entity(recipe_id)}{_sanitize_entity(site_id)}")
    if "run" in scan.entities:
        parts.append(f"run-{scan.entities['run']}")
    parts.append(scan.modality)
    return rel_dir, "_".join(parts)


def write_bids_derivative(
    out_dir: str | Path,
    scan: BIDSScan,
    recipe_id: str,
    site_id: str,
    data: np.ndarray,
    affine: np.ndarray,
    sidecar: dict[str, Any],
    provenance: dict[str, Any],
    bvals: np.ndarray | None = None,
    bvecs: np.ndarray | None = None,
) -> dict[str, Path]:
    """Write simulated NIfTI, JSON sidecar with provenance, and optional .bval/.bvec."""
    out_root = Path(out_dir)
    rel_dir, stem = build_derivative_stem(scan, recipe_id=recipe_id, site_id=site_id)
    target_dir = out_root / rel_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    nifti_out = target_dir / f"{stem}.nii.gz"
    json_out = target_dir / f"{stem}.json"

    out_img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)
    nib.save(out_img, str(nifti_out))

    merged_sidecar = dict(sidecar)
    merged_sidecar["SimulationProvenance"] = provenance
    json_out.write_text(json.dumps(merged_sidecar, indent=2), encoding="utf-8")

    written: dict[str, Path] = {"nifti": nifti_out, "json": json_out}

    if bvals is not None:
        bval_out = target_dir / f"{stem}.bval"
        bval_out.write_text(
            " ".join(f"{float(v):.1f}" for v in bvals.reshape(-1)) + "\n",
            encoding="utf-8",
        )
        written["bval"] = bval_out

    if bvecs is not None:
        bvec_out = target_dir / f"{stem}.bvec"
        arr = np.asarray(bvecs, dtype=np.float64)
        if arr.shape[0] != 3 and arr.shape[1] == 3:
            arr = arr.T
        lines = [" ".join(f"{float(v):.6f}" for v in row) for row in arr]
        bvec_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written["bvec"] = bvec_out

    return written
