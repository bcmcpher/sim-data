"""BIDS I/O, orientation canonicalization, and anatomical priors."""

from sim_data.io.bids import (
    BIDSScan,
    build_blinded_bids_stem,
    build_output_bids_stem,
    discover_bids_scans,
    discover_subject_brain_mask,
    load_bids_scan,
    write_bids_root_files,
    write_bids_scan,
)
from sim_data.io.orientation import (
    AnatomicalPriors,
    canonicalize_to_ras,
    canonicalize_to_ras_with_transform,
    compute_anatomical_priors,
    estimate_background_noise_sigma,
    reorient_bvecs_to_ras,
    resolve_encoding_axes,
)

__all__ = [
    "AnatomicalPriors",
    "BIDSScan",
    "build_blinded_bids_stem",
    "build_output_bids_stem",
    "canonicalize_to_ras",
    "canonicalize_to_ras_with_transform",
    "compute_anatomical_priors",
    "discover_bids_scans",
    "discover_subject_brain_mask",
    "estimate_background_noise_sigma",
    "load_bids_scan",
    "reorient_bvecs_to_ras",
    "resolve_encoding_axes",
    "write_bids_root_files",
    "write_bids_scan",
]
