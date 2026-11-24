# `sim-data`: Multimodal MRI Artifact & Multi-Site QC Simulator

`sim-data` generates realistic, ground-truthed multimodal MRI quality-control datasets (**T1w, T2w, FLAIR, 4D BOLD fMRI, and 4D DWI**) from clean BIDS seed scans.

It combines **MONAI**, **TorchIO**, and **custom MRI-physics operators** in a strict 7-stage causal acquisition order, extracts **NiPreps-aligned Image Quality Metrics (IQMs)** matching **MRIQC**, **fMRIPrep**, and **QSIPrep**, simulates **image-domain scanner/site effects** verifiable via **ComBat** and **CovBat**, and outputs **`qc-studio`-compatible visual-QC packets** (`packet.json` v1.1 + midline and enhanced diagnostic views).

---

## Installation

`sim-data` uses [`uv`](https://docs.astral.sh/uv/) and provides mutually exclusive PyTorch extras so the same `pyproject.toml` supports both **CPU-only** machines and **NVIDIA RTX 50-series (Blackwell `sm_120` / CUDA 12.8)** GPUs.

### Prerequisites
- Python `>= 3.10`
- `uv` (`>= 0.5`)

### CPU-Only Workstation
```bash
uv sync --extra cpu --extra harmonization --group dev
```

### NVIDIA RTX 5080 / Blackwell GPU Workstation (CUDA 12.8)
```bash
uv sync --extra cu128 --extra harmonization --group dev
```

> **Tip (`uv run`)**: After syncing `--extra cpu` or `--extra cu128`, pass `--no-sync` (or the same `--extra`) to `uv run` so `uv` preserves your chosen PyTorch wheel backend instead of re-resolving the default PyPI `torch` wheel.

---

## Input BIDS Structure

Provide a clean baseline BIDS dataset containing any subset of the supported modalities (`T1w`, `T2w`, `FLAIR`, `bold`, `dwi`):

```text
my_bids/
├── dataset_description.json
└── sub-01/
    ├── anat/
    │   ├── sub-01_T1w.nii.gz
    │   ├── sub-01_T1w.json          # Optional: PhaseEncodingDirection, SliceEncodingDirection
    │   ├── sub-01_T2w.nii.gz
    │   └── sub-01_FLAIR.nii.gz
    ├── func/
    │   ├── sub-01_task-rest_bold.nii.gz
    │   └── sub-01_task-rest_bold.json
    └── dwi/
        ├── sub-01_dwi.nii.gz
        ├── sub-01_dwi.json
        ├── sub-01_dwi.bval
        └── sub-01_dwi.bvec
```

- Volumes are canonicalized internally to **RAS+**, and encoding axes are resolved from BIDS JSON sidecars (`PhaseEncodingDirection`, `SliceEncodingDirection`) with standard modality fallbacks.
- **DWI motion & `.bvec` behavior**: By default (`rotate_dwi_bvecs: false`), simulated inter-volume DWI motion leaves `.bvec` files **unrotated**, mirroring raw scanner acquisitions. Set `sampling.rotate_dwi_bvecs: true` when simulating motion-corrected/preprocessed datasets.

---

## Quickstart (CLI)

### 1. (Optional) Generate an Offline Synthetic Multimodal BIDS Cohort
For smoke tests, CI, or experimenting without real scans:
```bash
uv run --no-sync sim-data init-phantom --out ./demo_bids --n-subjects 4
```

### 2. Simulate Artifacts, Multi-Site Effects, IQMs & Visual-QC Packets
```bash
uv run --no-sync sim-data generate \
  --bids-dir ./demo_bids \
  --config configs/default_multimodal.yaml \
  --out-dir ./demo_out \
  --workers 4 \
  --device auto
```
- Use `--device auto` (uses CUDA on RTX 5080, CPU otherwise), `--device cpu`, or `--device cuda`.
- Included YAML templates in [`configs/`](configs/):
  - `configs/default_multimodal.yaml`: Multi-site factorial (`multisite_factorial`) across all 5 modalities and 3 scanner profiles.
  - `configs/dose_response_structural.yaml`: Single-artifact severity sweep (`dose_response`) across all 11 structural artifacts.
  - `configs/sites_catalog.yaml`: Reusable catalog of 4 image-domain scanner physics profiles (`Siemens_3T_32ch`, `GE_15T_8ch`, `Philips_3T_16ch`, `Siemens_3T_64ch_HighAccel`).

### 3. Verify ComBat vs. CovBat Multi-Site Harmonization
```bash
uv run --no-sync sim-data verify-harmonization \
  --manifest ./demo_out/manifest.parquet \
  --out-dir ./demo_out/harmonization_audit \
  --modality T1w
```
Audits Raw vs. ComBat vs. CovBat across ANOVA $F$/$\eta^2$ (location), Levene variance (scale), within-site residual covariance Frobenius distance, off-diagonal correlation Frobenius distance, linear & RF site-classifier AUC, and QC classification preservation, plus saves `site_pca_comparison.png`.

### 4. Benchmark Deterministic IQM Baseline & Enrich Visual-QC Packets
```bash
uv run --no-sync sim-data benchmark-deterministic \
  --manifest ./demo_out/manifest.parquet \
  --out-dir ./demo_out/qc_benchmark \
  --modality T1w \
  --update-packets ./demo_out
```
Fits stratified cross-validated regularized Logistic Regression over NiPreps IQMs (`pass`/`borderline`/`fail`), reports Macro-F1, Cohen's $\kappa$, quadratic weighted $\kappa$, and per-artifact detection recall (highlighting deterministic blind spots), and optionally writes `deterministic_priors` into each scan's `packet.json`.

---

## Supported Artifacts & Modality Compatibility

All artifacts map normalized severity $s \in [0, 1]$ monotonically to physical parameters and execute in strict physical causal order (**Site Pre $\to$ Spatial/Field/4D $\to$ K-Space $\to$ Bias Field $\to$ Noise/Drift $\to$ Post-Recon Cropping/Defacing**):

| Artifact Name | Modalities | Engine | Physical Phenomenon |
| :--- | :--- | :--- | :--- |
| `motion_3d` | `T1w`, `T2w`, `FLAIR` | TorchIO (`RandomMotion`) | Intra-scan 3D k-space rigid head motion |
| `motion_4d` | `bold`, `dwi` | Custom (`dynamic4d`) | Inter-volume 6-DOF drift + jerks, Power FD, BOLD spin-history, unrotated DWI `.bvec` |
| `ghosting_3d` | `T1w`, `T2w`, `FLAIR` | TorchIO (`RandomGhosting`) | Periodic PE-axis k-space ghosting |
| `nyquist_n2_ghost` | `bold`, `dwi` | Custom (`kspace`) | EPI odd/even $k_{\text{PE}}$ phase-error $N/2$-shifted ghost |
| `wrap_around` | All 5 | Custom (`kspace`) | PE-axis FOV fold-over / circular aliasing |
| `bias_field` | All 5 | MONAI (`RandBiasField`) | Smooth 3D polynomial $B_1$ intensity inhomogeneity |
| `rf_spike` | All 5 | MONAI (`KSpaceSpikeNoise`) | Controlled k-space RF spikes (herringbone striping; transient in 4D) |
| `susceptibility_metal` | `T1w`, `T2w`, `FLAIR` | Custom (`physics`) | Localized 3D magnetic dipole void + bright pile-up rim + readout warp |
| `epi_susceptibility` | `bold`, `dwi` | Custom (`physics`) | Orbitofrontal/temporal $B_0$ PE distortion + Jacobian modulation + $T_2^*$ dropout |
| `eddy_currents_dwi` | `dwi` | Custom (`physics`) | Gradient-direction (`bvec`/`bval`) correlated PE shear/scale/translation |
| `dwi_slice_dropout` | `dwi` | Custom (`dynamic4d`) | High-$b$ intra-TR slice-wise signal dropout ("venetian blind") |
| `rician_noise` | All 5 | MONAI (`RandRicianNoise`) + Custom NC-$\chi$ | Magnitude Rician / multi-coil Non-Central $\chi$ noise degrading SNR |
| `temporal_drift` | `bold` | Custom (`noise`) | Low-frequency polynomial & sinusoidal BOLD scanner drift |
| `fov_cropping` | All 5 | Custom (`spatial`) | Superior/inferior/anterior/posterior brain slab cutoff (`zero_slab` or `truncate_matrix`) |
| `gibbs_ringing` | All 5 | MONAI (`GibbsNoise`) | Hard k-space low-pass truncation edge ringing |
| `anisotropy` | All 5 | TorchIO (`RandomAnisotropy`) | Thick-slice through-plane downsampling |
| `defacing_damage` | `T1w`, `T2w`, `FLAIR` | Custom (`spatial`) | Post-noise zero-valued RAS+ facial/orbitofrontal/ear defacing cuts |

---

## Output Directory Layout

```text
demo_out/
├── simulation_config_resolved.yaml
├── manifest.parquet                  # Unified metadata, artifact severities, fidelity & NiPreps IQMs
├── manifest.tsv
├── sub-01/                           # BIDS-derivative simulated NIfTI + JSON provenance (+ .bval/.bvec)
│   ├── anat/
│   ├── func/
│   └── dwi/
├── views/
│   ├── *_mid_{sag,ax,cor}.png        # qc-studio prototype-compatible 3 midline slices
│   ├── *_ortho.png                   # 3-plane x 5-slice orthogonal mosaic
│   ├── *_bg_boost.png                # Log/gamma-boosted background artifact view
│   ├── *_fmri_carpet.png             # fMRIPrep-style BOLD carpet plot + FD/stdDVARS + tSNR
│   └── *_dwi_summary.png             # QSIPrep-style b=0 vs high-b + Volume x Slice dropout heatmap
└── packets/
    └── *_packet.json                 # Visual-QC payload v1.1 (IQMs + midline/enhanced views + ground_truth)
```

---

## Python API Example

```python
from sim_data.config import SimulationConfig
from sim_data.harmonization import run_harmonization_audit
from sim_data.pipeline import run_simulation_plan
import pandas as pd

cfg = SimulationConfig.from_yaml("configs/default_multimodal.yaml")
result = run_simulation_plan(
    bids_dir="./demo_bids",
    config=cfg,
    out_dir="./demo_out",
    workers=4,
    device="auto",
)

df = pd.read_parquet(result.manifest_paths["parquet"])
audit = run_harmonization_audit(df, out_dir="./demo_out/harmonization_audit", modality="T1w")
print(audit.report["deltas"])
```

---

## Development & Quality Checks

```bash
# Static type checking (mypy with pydantic plugin)
uv run --no-sync mypy src/sim_data

# Linting (ruff)
uv run --no-sync ruff check src tests

# Full offline unit & integration test suite
uv run --no-sync pytest -v
```
