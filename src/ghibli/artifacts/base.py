"""Core artifact transform interfaces, physical stages, and scan context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np
import torch

from ghibli.config import Modality, NoiseCalibrationMode
from ghibli.io.orientation import AnatomicalPriors


class PipelineStage(IntEnum):
    """Strict causal ordering for physical MRI simulation."""

    SITE_PRE = 10
    SPATIAL_FIELD_4D = 20
    KSPACE = 30
    BIAS_FIELD = 40
    NOISE_DRIFT = 50
    POST_RECON = 60


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve user device flag ('auto', 'cpu', 'cuda') into a torch.device."""
    dev = (device or "auto").lower().strip()
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(dev)


@dataclass
class ArtifactRecord:
    """Recorded provenance for a single applied artifact."""

    name: str
    severity: float
    stage: int
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanContext:
    """Mutable MRI volume state flowing through the 7-stage physical pipeline."""

    data: np.ndarray  # float32 (X, Y, Z) or (X, Y, Z, T) in RAS+
    affine: np.ndarray  # 4x4 float64
    zooms: tuple[float, float, float]
    modality: Modality
    pe_axis: int
    readout_axis: int
    slice_axis: int
    priors: AnatomicalPriors
    rng: np.random.Generator
    seed: int
    device: str = "auto"
    bvals: np.ndarray | None = None
    bvecs: np.ndarray | None = None  # shape (3, T)
    rotate_dwi_bvecs: bool = False
    coupled_trajectory: bool = False
    baseline_noise_sigma: float = 0.0
    noise_calibration: NoiseCalibrationMode = "excess_variance"
    site_seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_4d(self) -> bool:
        return self.data.ndim == 4

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        return (int(self.data.shape[0]), int(self.data.shape[1]), int(self.data.shape[2]))

    @property
    def torch_device(self) -> torch.device:
        return resolve_device(self.device)

    def spawn_seed(self) -> int:
        return int(self.rng.integers(0, 2**31 - 1))


class BaseArtifactTransform(ABC):
    """Base class for all severity-calibrated MRI artifact transforms."""

    name: str = "base"
    stage: PipelineStage = PipelineStage.KSPACE
    supported_modalities: tuple[Modality, ...] = ("T1w", "T2w", "FLAIR", "bold", "dwi")
    qc_weight: float = 1.0

    def __init__(self, severity: float, params: dict[str, Any] | None = None) -> None:
        self.severity = float(np.clip(severity, 0.0, 1.0))
        self.params = dict(params or {})

    def apply(self, ctx: ScanContext) -> ArtifactRecord:
        if self.severity <= 0.0:
            return ArtifactRecord(
                name=self.name,
                severity=0.0,
                stage=int(self.stage),
                params={"skipped": True},
            )
        realized = self._apply_impl(ctx)
        # Ensure finite float32 output after every stage
        ctx.data = np.nan_to_num(
            ctx.data.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0
        )
        return ArtifactRecord(
            name=self.name,
            severity=self.severity,
            stage=int(self.stage),
            params=realized,
        )

    @abstractmethod
    def _apply_impl(self, ctx: ScanContext) -> dict[str, Any]:
        """Mutate ctx in-place and return a dictionary of realized physical parameters."""
