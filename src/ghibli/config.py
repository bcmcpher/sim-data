"""Pydantic configuration schemas for ghibli."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

Modality = Literal["T1w", "T2w", "FLAIR", "bold", "dwi"]
SamplingMode = Literal["dose_response", "combinatorial", "multisite_factorial"]
SiteNaming = Literal["desc", "none"]
NamingScheme = Literal["blinded", "acq_blinded", "unblinded_validation"]
NoiseCalibrationMode = Literal["excess_variance", "additive"]
ContrastBackend = Literal["pchip_bloch"]
BrainExtractionMode = Literal["auto", "evac", "morphological", "none"]


def _default_modalities() -> list[Modality]:
    return ["T1w", "T2w", "FLAIR", "bold", "dwi"]


def _default_dwi_shells() -> list[tuple[float, int]]:
    return [(1000.0, 20), (2000.0, 20)]


class ArtifactSpec(BaseModel):
    """Specification for an artifact eligible for simulation."""

    name: str
    weight: float = Field(default=1.0, gt=0.0)
    severity_range: tuple[float, float] | None = None
    fixed_severity: float | None = Field(default=None, ge=0.0, le=1.0)
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


class CompoundTrajectorySpec(BaseModel):
    """Multi-artifact coupled trajectory swept along a shared dose curve.

    Each member ArtifactSpec scales as `clip(dose * weight, 0, 1)` unless
    `fixed_severity` is set (which holds that artifact constant as a background
    nuisance confounder across the trajectory).
    """

    name: str
    artifacts: list[ArtifactSpec] = Field(default_factory=list)


class SiteProfileConfig(BaseModel):
    """Image-domain scanner physics profile inducing location, scale, and covariance effects.

    Set `passthrough=True` for native real-scanner baselines where `severity=0.0` must
    remain bit-for-bit identical to the input QC-passing scan.
    """

    site_id: str
    description: str = ""
    passthrough: bool = False
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
    coupled_trajectories: bool = True
    replicates_per_artifact: int = Field(default=1, ge=1)
    sweep_all_sites: bool = True
    noise_calibration: NoiseCalibrationMode = "excess_variance"
    compound_trajectories: list[CompoundTrajectorySpec] = Field(default_factory=list)
    samples_per_scan: int = Field(default=4, ge=1)
    samples_per_scan_per_site: int = Field(default=2, ge=1)
    min_artifacts: int = Field(default=1, ge=0)
    max_artifacts: int = Field(default=3, ge=1)
    severity_range: tuple[float, float] = (0.20, 0.90)
    rotate_dwi_bvecs: bool = False
    artifacts: list[ArtifactSpec] = Field(default_factory=list)


class OutputConfig(BaseModel):
    """Controls BIDS outputs, blinding scheme, and optional downstream diagnostics.

    Naming schemes:
    - `blinded` (default): Shuffled sequential `run-01..run-N` (+ optional `desc-<Site>`).
      Zero artifact or severity leakage in filenames, JSON sidecars, or `*_scans.tsv`.
    - `acq_blinded`: Raw-BIDS compliant (`DatasetType: "raw"`, no `desc-`) with shuffled
      opaque `acq-v01..acq-vNN` labels. Zero artifact or severity leakage anywhere in the
      BIDS tree; only `manifest.tsv` maps `acq-vXX` to trajectory and severity.
    - `unblinded_validation` (explicit opt-in ONLY): Readable `acq-<Artifact>` and monotonic
      `run-<Step>` labels strictly for internal debugging/validation where blinding is not
      needed. NEVER use for human or AI QC rating studies because filenames leak severity.
    """

    save_nifti: bool = True
    write_bids_metadata: bool = True
    site_naming: SiteNaming = "desc"
    naming_scheme: NamingScheme = "blinded"
    include_severity_in_acq: bool = False
    extract_iqms: bool = False
    render_views: bool = False
    render_midline: bool = True
    render_enhanced: bool = True
    image_format: Literal["png", "webp"] = "png"
    write_packets: bool = False
    include_proxy_qc_labels: bool = False
    qc_thresholds: QCThresholdConfig = Field(default_factory=QCThresholdConfig)

    @model_validator(mode="after")
    def _validate_blinding_guardrails(self) -> OutputConfig:
        if self.include_severity_in_acq and self.naming_scheme != "unblinded_validation":
            raise ValueError(
                "include_severity_in_acq=True is only allowed when "
                "naming_scheme='unblinded_validation'. Severity must never appear in "
                "filenames under blinded or acq_blinded modes."
            )
        return self


class SimulationConfig(BaseModel):
    """Top-level configuration for a ghibli run."""

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


class DiffeomorphicWarpConfig(BaseModel):
    """Per-subject coherent 3D diffeomorphic deformation parameters."""

    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    smooth_sigma_mm: float = Field(default=16.0, gt=2.0, le=60.0)
    affine_scale_range: tuple[float, float] = (0.94, 1.06)
    affine_rot_deg: float = Field(default=6.0, ge=0.0, le=25.0)
    affine_shear: float = Field(default=0.03, ge=0.0, le=0.15)
    flip_lr_prob: float = Field(default=0.5, ge=0.0, le=1.0)


class OptionalTemplateConfig(BaseModel):
    """Optional external NIfTI templates/masks (all None by default for template-free mode)."""

    brain_mask_nifti: Path | None = None
    tissue_pve_nifti: Path | None = None
    rsn_parcellation_nifti: Path | None = None
    dti_eigenvector_nifti: Path | None = None


class ContrastConversionConfig(BaseModel):
    """Tissue-aware continuous T1w -> T2w / FLAIR / EPI contrast conversion parameters.

    `backend="pchip_bloch"` implements Options 1 + 2 (N4ITK + internal EVAC+/external brain
    mask + PCHIP transfer anchored to Bloch tissue signals).
    Reserved extension hook for a future pretrained neural backend (Option 3).
    """

    backend: ContrastBackend = "pchip_bloch"
    brain_extraction: BrainExtractionMode = "auto"
    n4_bias_correct: bool = True
    preserve_micro_texture: bool = True
    fat_retention_t2w: float = Field(default=0.65, ge=0.0, le=1.2)
    fat_retention_flair: float = Field(default=0.35, ge=0.0, le=1.0)
    fat_retention_epi: float = Field(default=0.0, ge=0.0, le=1.0)
    epi_skull_strip: bool = True
    epi_crop_to_brain: bool = True
    epi_crop_pad_mm: float = Field(default=16.0, ge=0.0, le=60.0)
    brain_mask_suffix: str = "desc-brain_mask"


class StructuralSynthesisConfig(BaseModel):
    """Forward Bloch / contrast-converted structural synthesis parameters."""

    preserve_seed_texture_weight: float = Field(default=0.65, ge=0.0, le=1.0)
    tissue_param_jitter: float = Field(default=0.06, ge=0.0, le=0.25)
    baseline_snr: float = Field(default=80.0, ge=10.0, le=300.0)


class BoldSynthesisConfig(BaseModel):
    """4D BOLD EPI spatiotemporal synthesis parameters."""

    voxel_size_mm: tuple[float, float, float] = (3.0, 3.0, 3.0)
    n_trs: int = Field(default=60, ge=6, le=2000)
    tr_sec: float = Field(default=2.0, gt=0.1, le=10.0)
    te_sec: float = Field(default=0.030, gt=0.005, le=0.15)
    gm_bold_pct: float = Field(default=0.025, ge=0.0, le=0.15)
    physio_pct: float = Field(default=0.015, ge=0.0, le=0.15)
    ar1_phi: float = Field(default=0.35, ge=0.0, le=0.9)
    baseline_snr: float = Field(default=65.0, ge=10.0, le=300.0)


class DwiSynthesisConfig(BaseModel):
    """4D multi-shell multi-tensor DWI synthesis parameters."""

    voxel_size_mm: tuple[float, float, float] = (2.5, 2.5, 2.5)
    n_b0: int = Field(default=3, ge=1, le=20)
    shells: list[tuple[float, int]] = Field(default_factory=_default_dwi_shells)
    crossing_fiber_fraction: float = Field(default=0.30, ge=0.0, le=1.0)
    tr_sec: float = Field(default=4.2, gt=0.5, le=20.0)
    te_sec: float = Field(default=0.089, gt=0.01, le=0.25)
    baseline_snr: float = Field(default=55.0, ge=10.0, le=300.0)


class CohortSynthesisConfig(BaseModel):
    """Top-level configuration for `ghibli synthesize-cohort`."""

    global_seed: int = 1024
    n_subjects: int = Field(default=12, ge=1)
    min_seed_subjects: int = Field(default=3, ge=1)
    allow_few_seeds: bool = False
    preserve_real_modalities: bool = False
    modalities: list[Modality] = Field(default_factory=_default_modalities)
    warp: DiffeomorphicWarpConfig = Field(default_factory=DiffeomorphicWarpConfig)
    templates: OptionalTemplateConfig = Field(default_factory=OptionalTemplateConfig)
    contrast: ContrastConversionConfig = Field(default_factory=ContrastConversionConfig)
    structural: StructuralSynthesisConfig = Field(default_factory=StructuralSynthesisConfig)
    bold: BoldSynthesisConfig = Field(default_factory=BoldSynthesisConfig)
    dwi: DwiSynthesisConfig = Field(default_factory=DwiSynthesisConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> CohortSynthesisConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(raw or {})

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
