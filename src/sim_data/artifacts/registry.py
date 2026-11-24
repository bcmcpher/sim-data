"""Artifact registry and modality compatibility lookup."""

from __future__ import annotations

from typing import Any

from sim_data.artifacts.base import BaseArtifactTransform
from sim_data.artifacts.dynamic4d import DWISliceDropout, Motion4D
from sim_data.artifacts.kspace import NyquistN2Ghost, WrapAround
from sim_data.artifacts.monai_wrappers import BiasField, GibbsRinging, RFSpike, RicianNoiseArtifact
from sim_data.artifacts.noise import TemporalDrift
from sim_data.artifacts.physics import (
    EddyCurrentsDWI,
    EPISusceptibilityDistortion,
    SusceptibilityMetal3D,
)
from sim_data.artifacts.spatial import DefacingDamage, FOVCropping
from sim_data.artifacts.torchio_wrappers import Anisotropy, Ghosting3D, Motion3D
from sim_data.config import Modality

ARTIFACT_REGISTRY: dict[str, type[BaseArtifactTransform]] = {
    "motion_3d": Motion3D,
    "motion_4d": Motion4D,
    "ghosting_3d": Ghosting3D,
    "nyquist_n2_ghost": NyquistN2Ghost,
    "wrap_around": WrapAround,
    "bias_field": BiasField,
    "rf_spike": RFSpike,
    "susceptibility_metal": SusceptibilityMetal3D,
    "epi_susceptibility": EPISusceptibilityDistortion,
    "eddy_currents_dwi": EddyCurrentsDWI,
    "dwi_slice_dropout": DWISliceDropout,
    "rician_noise": RicianNoiseArtifact,
    "temporal_drift": TemporalDrift,
    "fov_cropping": FOVCropping,
    "gibbs_ringing": GibbsRinging,
    "anisotropy": Anisotropy,
    "defacing_damage": DefacingDamage,
}


def create_artifact(
    name: str,
    severity: float,
    params: dict[str, Any] | None = None,
) -> BaseArtifactTransform:
    if name not in ARTIFACT_REGISTRY:
        known = ", ".join(sorted(ARTIFACT_REGISTRY))
        raise KeyError(f"Unknown artifact '{name}'. Available: {known}")
    return ARTIFACT_REGISTRY[name](severity=severity, params=params)


def compatible_artifacts(modality: Modality) -> list[str]:
    return [
        name
        for name, cls in ARTIFACT_REGISTRY.items()
        if modality in cls.supported_modalities
    ]
