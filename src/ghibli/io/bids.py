"""BIDS input discovery, per-subject mask discovery, and zero-leakage BIDS writing."""

from __future__ import annotations

import fnmatch
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

from ghibli.config import Modality, NamingScheme, SiteNaming

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
    acq: str | None = None,
    filename_glob: str | None = None,
) -> list[BIDSScan]:
    """Recursively discover BIDS MRI scans under a BIDS root directory.

    Optional filters:
    - `acq`: exact or fnmatch glob pattern on the BIDS `acq-<label>` entity (e.g. `"standard"`
      to select clean `acq-standard` scans in multi-acquisition datasets like MR-ART).
    - `filename_glob`: fnmatch glob pattern against the NIfTI filename (e.g. `"*acq-standard*"`).
    """
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

        if filename_glob and not fnmatch.fnmatch(path.name, filename_glob):
            continue

        modality = _parse_modality(path)
        if modality is None or modality not in allowed:
            continue

        entities = dict(_ENTITY_RE.findall(path.name))
        if acq is not None:
            scan_acq = entities.get("acq", "")
            if not scan_acq or not fnmatch.fnmatch(scan_acq, acq):
                continue

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


def discover_subject_brain_mask(
    bids_dir: str | Path,
    subject: str,
    session: str | None = None,
    suffix: str = "desc-brain_mask",
) -> Path | None:
    """Auto-discover an external brain/ICV mask NIfTI for a BIDS subject (raw anat/ or derivatives/)."""
    root = Path(bids_dir)
    sub_part = f"sub-{subject}"
    ses_rel = Path(sub_part) / f"ses-{session}" / "anat" if session else Path(sub_part) / "anat"

    search_dirs: list[Path] = [root / ses_rel]
    deriv_root = root / "derivatives"
    if deriv_root.exists():
        for pipeline_dir in sorted(deriv_root.iterdir()):
            if pipeline_dir.is_dir():
                search_dirs.append(pipeline_dir / ses_rel)

    patterns = [f"*{suffix}.nii.gz", f"*{suffix}.nii", "*brain_mask*.nii.gz", "*brain_mask*.nii"]
    for sdir in search_dirs:
        if not sdir.exists():
            continue
        for pat in patterns:
            matches = sorted(sdir.glob(pat))
            if matches:
                return matches[0]
    return None


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


def build_output_bids_stem(
    scan: BIDSScan,
    run_index: int = 1,
    site_id: str | None = None,
    site_naming: SiteNaming = "desc",
    naming_scheme: NamingScheme = "blinded",
    acq_label: str | None = None,
) -> tuple[Path, str]:
    """Construct a BIDS relative directory and filename stem under the chosen naming scheme.

    - `blinded`: Shuffled `run-XX` (+ optional `desc-<Site>`), zero artifact/severity leakage.
    - `acq_blinded`: Raw-BIDS compliant (`DatasetType: "raw"`) opaque shuffled `acq-vXX`
      with zero artifact, severity, or run-order leakage.
    - `unblinded_validation`: Raw-BIDS compliant `acq-<Profile>_run-<Step>` strictly for
      non-blinded validation/debugging (explicit opt-in only).
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

    if naming_scheme == "blinded":
        parts.append(f"run-{run_index:02d}")
        if site_naming == "desc" and site_id:
            sanitized_site = _sanitize_entity(site_id)
            if sanitized_site:
                parts.append(f"desc-{sanitized_site}")
    elif naming_scheme == "acq_blinded":
        sanitized_acq = _sanitize_entity(acq_label or f"v{max(1, run_index):02d}")
        if sanitized_acq:
            parts.append(f"acq-{sanitized_acq}")
    else:  # unblinded_validation
        if acq_label:
            sanitized_acq = _sanitize_entity(acq_label)
            if sanitized_acq:
                parts.append(f"acq-{sanitized_acq}")
        if run_index > 0:
            parts.append(f"run-{run_index:02d}")

    parts.append(scan.modality)
    return rel_dir, "_".join(parts)


def build_blinded_bids_stem(
    scan: BIDSScan,
    run_index: int,
    site_id: str | None = None,
    site_naming: SiteNaming = "desc",
) -> tuple[Path, str]:
    """Backward-compatible wrapper constructing a zero-leakage blinded BIDS stem."""
    return build_output_bids_stem(
        scan=scan,
        run_index=run_index,
        site_id=site_id,
        site_naming=site_naming,
        naming_scheme="blinded",
    )


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
    naming_scheme: NamingScheme = "blinded",
    acq_label: str | None = None,
) -> dict[str, Path]:
    """Write a simulated BIDS NIfTI, clean zero-leakage JSON sidecar, and DWI tables.

    Always strips SimulationProvenance metadata so no artifact or ground-truth information
    leaks into the image sidecar under any naming scheme.
    """
    out_root = Path(out_dir)
    rel_dir, stem = build_output_bids_stem(
        scan,
        run_index=run_index,
        site_id=site_id,
        site_naming=site_naming,
        naming_scheme=naming_scheme,
        acq_label=acq_label,
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
    naming_scheme: NamingScheme = "blinded",
) -> dict[str, Path]:
    """Write standard BIDS root files: dataset_description.json, participants.tsv, *_scans.tsv, .bidsignore."""
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    is_derivative = naming_scheme == "blinded" and site_naming == "desc"
    desc: dict[str, Any] = {
        "Name": "Simulated Multimodal MRI Dataset",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative" if is_derivative else "raw",
    }
    if is_derivative:
        desc["GeneratedBy"] = [{"Name": "ghibli", "Version": "0.1.0"}]
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

    # Per-subject/session BIDS *_scans.tsv: strictly zero-leakage (filename + site_id only)
    # under blinded and acq_blinded modes; optional dose columns only in unblinded_validation.
    scans_grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        nifti_rel = str(row.get("nifti_path", ""))
        if not nifti_rel:
            continue
        sub = str(row["subject"])
        ses = str(row.get("session", ""))
        sub_root = f"sub-{sub}/ses-{ses}/" if ses else f"sub-{sub}/"
        rel_inside = nifti_rel[len(sub_root) :] if nifti_rel.startswith(sub_root) else nifti_rel
        entry: dict[str, Any] = {
            "filename": rel_inside,
            "site_id": str(row.get("site_id", "")),
        }
        if naming_scheme == "unblinded_validation":
            entry["trajectory_id"] = str(row.get("trajectory_id", ""))
            entry["dose_step"] = int(row.get("dose_step", 0))
            entry["dose_level"] = float(row.get("dose_level", 0.0))
        scans_grouped.setdefault((sub, ses), []).append(entry)

    for (sub, ses), scan_entries in sorted(scans_grouped.items()):
        target_dir = out_root / f"sub-{sub}" / f"ses-{ses}" if ses else out_root / f"sub-{sub}"
        target_dir.mkdir(parents=True, exist_ok=True)
        fname = f"sub-{sub}_ses-{ses}_scans.tsv" if ses else f"sub-{sub}_scans.tsv"
        scan_entries_sorted = sorted(scan_entries, key=lambda d: str(d["filename"]))
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
                "sensitivity_analysis/",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    written["bidsignore"] = bidsignore_path

    return written
