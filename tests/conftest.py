"""Shared pytest fixtures generating fast synthetic multimodal BIDS phantoms offline."""

from __future__ import annotations

from pathlib import Path

import pytest

from ghibli.phantoms.synthetic import generate_synthetic_bids_cohort


@pytest.fixture(scope="session")
def bids_phantom_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("bids_phantom")
    return generate_synthetic_bids_cohort(
        out_dir=root,
        n_subjects=2,
        anat_shape=(48, 56, 48),
        epi_shape=(40, 44, 28),
        n_bold_trs=18,
        n_dwi_dirs=10,
        seed=777,
    )
