"""BIDS input discovery and zero-leakage blinded BIDS writing for multimodal MRI."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from dipy.core.sphere import HemiSphere, disperse_charges
from dipy.io.gradients import read_bvals_bvecs

from sim_data.config import Modality, SiteNaming

_MODALITY_FOLDERS: dict[str, str] = {
    "T1w": "anat",
    "T2w": "anat",
    "FLAIR": "anat",
    "bold": "func",
    "dwi": "dwi",
}

_ENTITY_RE = re.compile(r"(sub|ses|task|acq|run|dir|echo|desc)-([A-Za-z0-9]+)")


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
    bval_str = str(scan.bval_path) if (scan.bval_path and scan.bval_path.exists()) else None
    bvec_str = str(scan.bvec_path) if (scan.bvec_path and scan.bvec_path.exists()) else None
    if bval_str is not None or bvec_str is not None:
        raw_bvals, raw_bvecs = read_bvals_bvecs(bval_str, bvec_str)
        if raw_bvals is not None:
            bvals = np.asarray(raw_bvals, dtype=np.float32).reshape(-1)
        if raw_bvecs is not None:
            bv = np.asarray(raw_bvecs, dtype=np.float32)
            # DIPY read_bvals_bvecs returns (N, 3); store as BIDS-standard (3, N)
            if bv.ndim == 2 and bv.shape[1] == 3 and bv.shape[0] != 3:
                bv = bv.T
            elif bv.ndim == 2 and bv.shape[1] == 3 and bvals is not None and len(bvals) == bv.shape[0]:
                bv = bv.T
            bvecs = np.ascontiguousarray(bv, dtype=np.float32)

    return LoadedBIDSScan(scan=scan, img=img, sidecar=sidecar, bvals=bvals, bvecs=bvecs)


def sample_electrostatic_directions(
    n_dirs: int,
    rng: np.random.Generator,
    iters: int = 300,
) -> np.ndarray:
    """Generate uniformly dispersed unit gradient directions (3, n_dirs) via DIPY electrostatic repulsion."""
    pts = rng.normal(0.0, 1.0, size=(max(1, n_dirs), 3)).astype(np.float64)
    norms = np.linalg.norm(pts, axis=1, keepdims=True) + 1e-8
    pts = pts / norms
    hemi = HemiSphere(xyz=pts)
    dispersed, _potential = disperse_charges(hemi, iters=iters)
    return np.ascontiguousarray(dispersed.vertices.T, dtype=np.float32)


def _sanitize_entity(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", value)


def build_blinded_bids_stem(
    scan: BIDSScan,
    run_index: int,
    site_id: str | None = None,
    site_naming: SiteNaming = "desc",
) -> tuple[Path, str]:
    """Construct a zero-leakage blinded BIDS relative directory and filename stem.

    Preserves input `ses-*` and `task-*` entities untouched, assigns an opaque sequential
    `run-XX` index, and optionally isolates site identity via `desc-<site>` when
    `site_naming == "desc"`. Never writes recipe, artifact, or severity labels into filenames.
    """
    sub_dir = Path(f"sub-{scan.subject}")
    if scan.session:
        sub_dir = sub_dir / f"ses-{scan.session}"
    rel_dir = sub_dir / scan.datatype

    parts = [f"sub-{scan.subject}"]
    if scan.session:
        parts.append(f"ses-{scan.session}")
    if "task" in scan.entities:
        parts.append(f"task-{scan.entities['task']}")
    parts.append(f"run-{run_index:02d}")
    if site_naming == "desc" and site_id:
        sanitized_site = _sanitize_entity(site_id)
        if sanitized_site:
            parts.append(f"desc-{sanitized_site}")
    parts.append(scan.modality)
    return rel_dir, "_".join(parts)


def write_bids_scan(
    out_dir: str | Path,
    scan: BIDSScan,
    run_index: int,
    site_id: str | None,
    site_naming: SiteNaming,
    data: np.ndarray,
    affine: np.ndarray,
    sidecar: dict[str, Any],
    bvals: np.ndarray | None = None,
    bvecs: np.ndarray | None = None,
) -> dict[str, Path]:
    """Write a blinded simulated BIDS NIfTI, clean zero-leakage JSON sidecar, and DWI tables.

    Strips any SimulationProvenance metadata so no artifact or ground-truth information
    leaks into the image sidecar during manual or AI visual-QC evaluation.
    """
    out_root = Path(out_dir)
    rel_dir, stem = build_blinded_bids_stem(
        scan, run_index=run_index, site_id=site_id, site_naming=site_naming
    )
    target_dir = out_root / rel_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    nifti_out = target_dir / f"{stem}.nii.gz"
    json_out = target_dir / f"{stem}.json"

    out_img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)
    nib.save(out_img, str(nifti_out))

    # Write clean BIDS sidecar with zero simulation/artifact leakage
    clean_sidecar = {k: v for k, v in sidecar.items() if k != "SimulationProvenance"}
    json_out.write_text(json.dumps(clean_sidecar, indent=2), encoding="utf-8")

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


def write_bids_root_files(
    out_dir: str | Path,
    rows: list[dict[str, Any]],
    site_naming: SiteNaming = "desc",
) -> dict[str, Path]:
    """Write standard BIDS root files: dataset_description.json, participants.tsv, *_scans.tsv, .bidsignore."""
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    is_derivative = site_naming == "desc"
    desc: dict[str, Any] = {
        "Name": "Simulated Multimodal MRI Dataset",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative" if is_derivative else "raw",
    }
    if is_derivative:
        desc["GeneratedBy"] = [{"Name": "sim-data", "Version": "0.1.0"}]
    desc_path = out_root / "dataset_description.json"
    desc_path.write_text(json.dumps(desc, indent=2), encoding="utf-8")
    written["dataset_description"] = desc_path

    # Build participants.tsv with site_id when a subject maps to a single site (else comma-joined sites)
    sub_sites: dict[str, set[str]] = {}
    for row in rows:
        sub = str(row["subject"])
        site = str(row.get("site_id", ""))
        sub_sites.setdefault(sub, set())
        if site:
            sub_sites[sub].add(site)

    part_rows = []
    for sub in sorted(sub_sites):
        sites_sorted = sorted(sub_sites[sub])
        part_rows.append(
            {
                "participant_id": f"sub-{sub}",
                "site_id": sites_sorted[0] if len(sites_sorted) == 1 else ",".join(sites_sorted),
            }
        )
    part_path = out_root / "participants.tsv"
    pd.DataFrame(part_rows).to_csv(part_path, sep="\t", index=False)
    written["participants"] = part_path

    # Per-subject/session BIDS *_scans.tsv listing only filename and site_id (zero artifact leakage)
    scans_grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        nifti_rel = str(row.get("nifti_path", ""))
        if not nifti_rel:
            continue
        sub = str(row["subject"])
        ses = str(row.get("session", ""))
        sub_root = f"sub-{sub}/ses-{ses}/" if ses else f"sub-{sub}/"
        rel_inside = nifti_rel[len(sub_root) :] if nifti_rel.startswith(sub_root) else nifti_rel
        scans_grouped.setdefault((sub, ses), []).append(
            {"filename": rel_inside, "site_id": str(row.get("site_id", ""))}
        )

    for (sub, ses), scan_entries in sorted(scans_grouped.items()):
        target_dir = out_root / f"sub-{sub}" / f"ses-{ses}" if ses else out_root / f"sub-{sub}"
        target_dir.mkdir(parents=True, exist_ok=True)
        fname = f"sub-{sub}_ses-{ses}_scans.tsv" if ses else f"sub-{sub}_scans.tsv"
        scan_entries_sorted = sorted(scan_entries, key=lambda d: d["filename"])
        pd.DataFrame(scan_entries_sorted).to_csv(target_dir / fname, sep="\t", index=False)

    # .bidsignore so BIDS validators & NiPreps tools ignore bookkeeping/evaluation files
    bidsignore_path = out_root / ".bidsignore"
    bidsignore_path.write_text(
        "\n".join(
            [
                "manifest.tsv",
                "iqms.tsv",
                "simulation_config_resolved.yaml",
                "views/",
                "packets/",
                "harmonization_audit/",
                "qc_benchmark/",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    written["bidsignore"] = bidsignore_path

    return written
