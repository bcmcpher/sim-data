"""Unified modality-aware NiPreps IQM extractor dispatcher."""

from __future__ import annotations

from typing import Any

import numpy as np

from sim_data.config import Modality
from sim_data.io.orientation import AnatomicalPriors
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
    clean_priors: AnatomicalPriors | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Dispatch to the modality-appropriate NiPreps-aligned IQM extractor."""
    if modality in ("T1w", "T2w", "FLAIR"):
        vol = data[..., 0] if data.ndim == 4 else data
        return extract_structural_iqms(vol, zooms=zooms, clean_priors=clean_priors)
    if modality == "bold":
        return extract_functional_iqms(data, zooms=zooms, metadata=metadata)
    if modality == "dwi":
        return extract_diffusion_iqms(
            data,
            zooms=zooms,
            bvals=bvals,
            bvecs=bvecs,
            slice_axis=slice_axis,
            metadata=metadata,
        )
    raise ValueError(f"Unsupported modality for IQM extraction: {modality}")
