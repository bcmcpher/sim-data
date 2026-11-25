# `sim-data`: Multimodal MRI Artifact & Multi-Site BIDS Simulator

`sim-data` generates **blinded, leakage-free multimodal MRI BIDS datasets** (**T1w, T2w, FLAIR, 4D BOLD fMRI, and 4D DWI**) with controlled scanner/site effects, continuous T1w→T2w sequential contrast synthesis, and **coupled Common-Random-Numbers (CRN) dose-response artifact trajectories** from real QC-passing BIDS scans.

It combines **MONAI**, **TorchIO**, **DIPY**, **SimpleITK**, and **custom MRI-physics operators** in a strict 7-stage causal acquisition order, records all injected artifact and site weights exclusively in a ground-truth **`manifest.tsv`** (keeping filenames and JSON sidecars completely clean for unbiased manual and AI evaluations), and provides decoupled post-hoc tools for **blind NiPreps-aligned IQM extraction** (MRIQC, fMRIPrep, QSIPrep), **IQM dose-response sensitivity ranking**, and **ComBat / CovBat multi-site harmonization auditing**.

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

## Input BIDS Structure & Real QC-Passing Scans

Provide a clean baseline BIDS dataset containing one or more real QC-passing subjects with any subset of the supported modalities (`T1w`, `T2w`, `FLAIR`, `bold`, `dwi`). Existing `ses-*` sessions are preserved untouched, and non-RAS scans (e.g., LPS/PIR) automatically have their `PhaseEncodingDirection`, `SliceEncodingDirection`, and voxel-space `.bvec` axes remapped into canonical RAS+ coordinates:

```text
my_bids/
├── dataset_description.json
└── sub-01/
    ├── anat/
    │   ├── sub-01_T1w.nii.gz
    │   ├── sub-01_T1w.json          # Recommended: PhaseEncodingDirection, SliceEncodingDirection
    │   ├── sub-01_desc-brain_mask.nii.gz  # Optional external brain/ICV mask (Option 2)
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

### Naming Schemes & Zero-Leakage Policy (`output.naming_scheme`)

| Naming Scheme | Output Filename Example | Raw BIDS (`DatasetType`) | Leakage-Free? | Intended Use |
| :--- | :--- | :--- | :--- | :--- |
| `blinded` *(default)* | `sub-01_run-01[_desc-<Site>]_T1w.nii.gz` | `derivative` (`desc`) / `raw` (`none`) | **Yes** (shuffled `run-XX`) | Multi-site & combinatorial human/AI QC benchmarks |
| `acq_blinded` *(dose-response default)* | `sub-01_acq-v01_T1w.nii.gz` | **`raw`** (MRIQC / fMRIPrep / QSIPrep ready) | **Yes** (shuffled opaque `acq-vXX`) | Raw-BIDS dose-response & sensitivity sweeps with **zero severity leakage** |
| `unblinded_validation` *(opt-in only)* | `sub-01_acq-motion3d_run-01_T1w.nii.gz` | **`raw`** | **No** (explicitly embeds artifact & step order) | Internal debugging & visual validation ONLY |

> **Why `unblinded_validation` (and `include_severity_in_acq`) is disabled by default:**
> Embedding artifact identity, monotonic dose order, or severity values in BIDS filenames creates severe label leakage for human raters and vision-language models. Under both `blinded` and `acq_blinded`, filenames, JSON sidecars, and `*_scans.tsv` contain **zero** artifact or severity information—only `manifest.tsv` (saved outside the BIDS root via `--manifest-out`) maps each `acq-vXX` / `run-XX` scan to its `trajectory_id`, `target_artifact`, `dose_step`, `dose_level`, `baseline_scan_id`, and `sev_*` weights. Setting `include_severity_in_acq: true` is rejected by Pydantic unless `naming_scheme: unblinded_validation` is explicitly chosen.

- **DWI motion & `.bvec` behavior**: By default (`rotate_dwi_bvecs: false`), simulated inter-volume DWI motion leaves `.bvec` files **unrotated**, mirroring raw scanner acquisitions. Set `sampling.rotate_dwi_bvecs: true` when simulating motion-corrected/preprocessed datasets.
- **Real-scan noise calibration (`sampling.noise_calibration`)**: Defaults to `"excess_variance"`, estimating native baseline noise $\sigma_0$ on real QC-passing scans and subtracting it in quadrature when injecting `rician_noise` (while keeping $s=0$ untouched bit-for-bit under `sites: []` / `passthrough: true`).

---

## Quickstart (CLI)

### 1. (Optional) Standalone Continuous T1w → T2w / EPI Contrast Conversion
Convert a clean T1w NIfTI into continuous `T2w`, fat-suppressed $T_2^*$ `bold_s0`, or spin-echo `dwi_b0` contrast using N4ITK bias correction + ICV/bone/scalp separation (with optional external brain mask) + monotone decreasing PCHIP spline anchored to Bloch tissue signals:
```bash
uv run --no-sync sim-data convert-contrast \
  --t1w ./sub-01_T1w.nii.gz \
  --brain-mask ./sub-01_desc-brain_mask.nii.gz \
  --target T2w \
  --out ./sub-01_converted_T2w.nii.gz
```

### 2. (Optional) Complete Missing Modalities on Real Subjects or Synthesize a Cohort
Preserve real subjects' existing QC-passing modalities untouched (zero warp) and sequentially synthesize only missing modalities (`T1w → T2w → anti-aliased downsample → BOLD / DWI`):
```bash
uv run --no-sync sim-data synthesize-cohort \
  --seed-bids ./real_qc_passing_bids \
  --out ./completed_multimodal_bids \
  --preserve-real \
  --workers 4 \
  --device auto
```
Or expand **3–5+ structural seeds** into $N$ anatomically distinct warped subjects:
```bash
uv run --no-sync sim-data synthesize-cohort \
  --seed-bids ./clean_structural_seeds \
  --out ./synthetic_clean_cohort \
  --config configs/synthesize_cohort.yaml \
  --n-subjects 12 \
  --workers 4 \
  --device auto
```
*(Or create a quick offline synthetic phantom cohort for smoke tests via `uv run --no-sync sim-data init-phantom --out ./demo_bids --n-subjects 4`.)*

### 3. Generate Coupled Dose-Response or Multi-Site Blinded BIDS Scans + `manifest.tsv`
**Coupled dose-response sensitivity sweep (real QC-passing scans, raw-BIDS `acq_blinded`, passthrough $s=0$ baseline):**
```bash
uv run --no-sync sim-data generate \
  --bids-dir ./completed_multimodal_bids \
  --config configs/dose_response_sensitivity.yaml \
  --out-dir ./dose_bids \
  --manifest-out ./dose_manifest.tsv \
  --workers 4 \
  --device auto
```

**Multi-site factorial benchmark:**
```bash
uv run --no-sync sim-data generate \
  --bids-dir ./demo_bids \
  --config configs/default_multimodal.yaml \
  --out-dir ./demo_out \
  --manifest-out ./demo_manifest.tsv \
  --workers 4 \
  --device auto
```

### 4. Independently Extract Blind NiPreps IQMs (or Run Real MRIQC / fMRIPrep / QSIPrep)
```bash
uv run --no-sync sim-data extract-iqms \
  --bids-dir ./dose_bids \
  --out ./dose_iqms.tsv \
  --workers 4
```

### 5. Analyze IQM Sensitivity Across Increasing Artifact Intensities
```bash
uv run --no-sync sim-data analyze-sensitivity \
  --manifest ./dose_manifest.tsv \
  --iqms ./dose_iqms.tsv \
  --out-dir ./dose_sensitivity_report \
  --modality T1w
```
Works with both `sim-data extract-iqms` TSVs and external **MRIQC / fMRIPrep / QSIPrep** group TSVs. Outputs `sensitivity_summary.tsv`, `sensitivity_matrix.tsv`, `sensitivity_report.json`, `sensitivity_heatmap.png`, and `dose_response_curves.png` (within-trajectory Spearman $\rho$, standardized slope, Cohen's $d$, Minimum Detectable Severity / LoD, and cross-artifact specificity).

### 6. Audit ComBat vs. CovBat Multi-Site Harmonization
```bash
uv run --no-sync sim-data verify-harmonization \
  --manifest ./demo_manifest.tsv \
  --iqms ./demo_iqms.tsv \
  --out-dir ./demo_harmonization_audit \
  --modality T1w
```

### 7. (Optional) Benchmark Deterministic IQM Baseline
```bash
uv run --no-sync sim-data benchmark-deterministic \
  --manifest ./demo_manifest.tsv \
  --iqms ./demo_iqms.tsv \
  --out-dir ./demo_qc_benchmark \
  --modality T1w
```

---

## Supported Artifacts & Modality Compatibility

All artifacts map normalized severity $s \in [0, 1]$ monotonically to physical parameters (with Common-Random-Numbers trajectory coupling in `dose_response` mode) and execute in strict physical causal order (**Site Pre $\to$ Spatial/Field/4D $\to$ K-Space $\to$ Bias Field $\to$ Noise/Drift $\to$ Post-Recon Cropping/Defacing**):

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
| `rician_noise` | All 5 | MONAI (`RandRicianNoise`) + Custom NC-$\chi$ | Magnitude Rician / multi-coil Non-Central $\chi$ noise (`excess_variance` or `additive`) |
| `temporal_drift` | `bold` | Custom (`noise`) | Low-frequency polynomial & sinusoidal BOLD scanner drift |
| `fov_cropping` | All 5 | Custom (`spatial`) | Superior/inferior/anterior/posterior brain slab cutoff (`zero_slab` or `truncate_matrix`) |
| `gibbs_ringing` | All 5 | MONAI (`GibbsNoise`) | Hard k-space low-pass truncation edge ringing |
| `anisotropy` | All 5 | TorchIO (`RandomAnisotropy`) | Thick-slice through-plane downsampling |
| `defacing_damage` | `T1w`, `T2w`, `FLAIR` | Custom (`spatial`) | Post-noise zero-valued RAS+ facial/orbitofrontal/ear defacing cuts |

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
