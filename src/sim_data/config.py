"""Pydantic configuration schemas for sim-data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

Modality = Literal["T1w", "T2w", "FLAIR", "bold", "dwi"]
SamplingMode = Literal["dose_response", "combinatorial", "multisite_factorial"]
SiteNaming = Literal["desc", "none"]


def _default_modalities() -> list[Modality]:
    return ["T1w", "T2w", "FLAIR", "bold", "dwi"]


class ArtifactSpec(BaseModel):
    """Specification for an artifact eligible for simulation."""

    name: str
    weight: float = Field(default=1.0, gt=0.0)
    severity_range: tuple[float, float] | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("severity_range")
    @classmethod
    def _validate_severity_range(
        cls, value: tuple[float, float] | None
    ) -> tuple[float, float] | None:
        if value is not None:
            lo, hi = value
            if not (0.0 <= lo <= hi <= 1.0):
                raise ValueError(f"severity_range must satisfy 0 <= lo <= hi <= 1, got {value}")
        return value


class SiteProfileConfig(BaseModel):
    """Image-domain scanner physics profile inducing location, scale, and covariance effects."""

    site_id: str
    description: str = ""
    contrast_gamma: float = Field(default=1.0, gt=0.2, lt=3.0)
    intensity_gain: float = Field(default=1.0, gt=0.2, lt=5.0)
    intensity_offset: float = Field(default=0.0, ge=-0.25, le=0.25)
    psf_sigma_vox: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bias_order: int = Field(default=2, ge=1, le=4)
    bias_coeffs: tuple[float, float, float] = (0.0, 0.0, 0.0)
    coil_count: int = Field(default=16, ge=1, le=128)
    noise_sigma_rel: float = Field(default=0.0, ge=0.0, le=0.2)
    g_factor_strength: float = Field(default=0.0, ge=0.0, le=1.5)
    voxel_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    jitter_scale: float = Field(default=0.05, ge=0.0, le=0.5)


class QCThresholdConfig(BaseModel):
    """Optional thresholds mapping defect_burden into 3-tier proxy labels (when opted in)."""

    pass_max_severity: float = Field(default=0.22, ge=0.0, le=1.0)
    borderline_max_severity: float = Field(default=0.56, ge=0.0, le=1.0)

    @field_validator("borderline_max_severity")
    @classmethod
    def _check_order(cls, v: float, info: Any) -> float:
        pass_max = info.data.get("pass_max_severity", 0.22)
        if v <= pass_max:
            raise ValueError("borderline_max_severity must exceed pass_max_severity")
        return v


class SamplingPlanConfig(BaseModel):
    """Controls how clean scans, sites, and artifacts are combined into recipes."""

    mode: SamplingMode = "combinatorial"
    include_clean_baseline: bool = True
    shuffle_run_order: bool = True
    dose_severities: list[float] = Field(default_factory=lambda: [0.0, 0.25, 0.55, 0.85])
    samples_per_scan: int = Field(default=4, ge=1)
    samples_per_scan_per_site: int = Field(default=2, ge=1)
    min_artifacts: int = Field(default=1, ge=0)
    max_artifacts: int = Field(default=3, ge=1)
    severity_range: tuple[float, float] = (0.20, 0.90)
    rotate_dwi_bvecs: bool = False
    artifacts: list[ArtifactSpec] = Field(default_factory=list)


class OutputConfig(BaseModel):
    """Controls blinded BIDS outputs and optional downstream diagnostics.

    By default, `sim-data generate` writes only blinded BIDS NIfTI/JSON/bval/bvec files,
    BIDS root metadata (dataset_description.json, participants.tsv, *_scans.tsv, .bidsignore),
    and the official ground-truth `manifest.tsv`. IQM extraction, 2D view rendering,
    packet.json assembly, and categorical proxy QC labels are opt-in.
    """

    save_nifti: bool = True
    write_bids_metadata: bool = True
    site_naming: SiteNaming = "desc"
    extract_iqms: bool = False
    render_views: bool = False
    render_midline: bool = True
    render_enhanced: bool = True
    image_format: Literal["png", "webp"] = "png"
    write_packets: bool = False
    include_proxy_qc_labels: bool = False
    qc_thresholds: QCThresholdConfig = Field(default_factory=QCThresholdConfig)


class SimulationConfig(BaseModel):
    """Top-level configuration for a sim-data run."""

    global_seed: int = 42
    modalities: list[Modality] = Field(default_factory=_default_modalities)
    sampling: SamplingPlanConfig = Field(default_factory=SamplingPlanConfig)
    sites: list[SiteProfileConfig] = Field(default_factory=list)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SimulationConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(raw or {})

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
