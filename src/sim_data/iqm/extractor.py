"""Unified strictly-blind modality-aware NiPreps IQM extractor and BIDS batch runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sim_data.config import Modality
from sim_data.io.bids import BIDSScan, discover_bids_scans, load_bids_scan
from sim_data.io.orientation import canonicalize_to_ras, resolve_encoding_axes
from sim_data.iqm.diffusion import extract_diffusion_iqms
from sim_data.iqm.functional import extract_functional_iqms
from sim_data.iqm.structural import extract_structural_iqms


def iqm_schema_name(modality: Modality) -> str:
    if modality in ("T1w", "T2w", "FLAIR"):
        return "nipreps_mriqc_structural_v1"
    if modality == "bold":
        return "nipreps_mriqc_fmriprep_bold_v1"
    return "nipreps_qsiprep_mriqc_dwi_v1"


def extract_iqms(
    data: np.ndarray,
    modality: Modality,
    zooms: tuple[float, float, float],
    bvals: np.ndarray | None = None,
    bvecs: np.ndarray | None = None,
    slice_axis: int = 2,
) -> dict[str, float]:
    """Dispatch to the strictly-blind modality-appropriate NiPreps-aligned IQM extractor."""
    if modality in ("T1w", "T2w", "FLAIR"):
        vol = data[..., 0] if data.ndim == 4 else data
        return extract_structural_iqms(vol, zooms=zooms)
    if modality == "bold":
        return extract_functional_iqms(data, zooms=zooms)
    if modality == "dwi":
        return extract_diffusion_iqms(
            data,
            zooms=zooms,
            bvals=bvals,
            bvecs=bvecs,
            slice_axis=slice_axis,
        )
    raise ValueError(f"Unsupported modality for IQM extraction: {modality}")


def _extract_single_bids_scan(scan: BIDSScan, bids_root: Path) -> dict[str, Any]:
    loaded = load_bids_scan(scan)
    data, _affine, zooms = canonicalize_to_ras(loaded.img)
    _pe, _ro, sl_axis = resolve_encoding_axes(loaded.sidecar, scan.modality)
    iqms = extract_iqms(
        data,
        modality=scan.modality,
        zooms=zooms,
        bvals=loaded.bvals,
        bvecs=loaded.bvecs,
        slice_axis=sl_axis,
    )
    row: dict[str, Any] = {
        "scan_id": scan.base_id,
        "nifti_path": str(scan.nifti_path.relative_to(bids_root)),
        "subject": scan.subject,
        "session": scan.session or "",
        "modality": scan.modality,
    }
    for k, v in iqms.items():
        row[f"iqm_{k}"] = v
    return row


def extract_iqms_for_bids_dir(
    bids_dir: str | Path,
    out_tsv: str | Path,
    modalities: list[Modality] | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    """Independently extract blind NiPreps IQMs from all BIDS scans on disk and save as TSV."""
    root = Path(bids_dir)
    scans = discover_bids_scans(root, modalities=modalities)
    if not scans:
        raise ValueError(f"No matching BIDS scans found under {root}")

    if workers <= 1 or len(scans) <= 1:
        rows = [_extract_single_bids_scan(s, root) for s in scans]
    else:
        rows = Parallel(n_jobs=workers, prefer="processes")(
            delayed(_extract_single_bids_scan)(s, root) for s in scans
        )

    df = pd.DataFrame(rows)
    out_path = Path(out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    return df
