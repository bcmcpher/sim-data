# `sim-data`: Multimodal MRI Artifact & Multi-Site BIDS Simulator

`sim-data` generates **blinded, leakage-free multimodal MRI BIDS datasets** (**T1w, T2w, FLAIR, 4D BOLD fMRI, and 4D DWI**) with controlled scanner/site effects and acquisition artifacts from clean BIDS seed scans.

It combines **MONAI**, **TorchIO**, and **custom MRI-physics operators** in a strict 7-stage causal acquisition order, records all injected artifact and site weights exclusively in a ground-truth **`manifest.tsv`** (keeping filenames and JSON sidecars completely clean for unbiased manual and AI evaluations), and provides decoupled post-hoc tools for **blind NiPreps-aligned IQM extraction** (MRIQC, fMRIPrep, QSIPrep) and **ComBat / CovBat multi-site harmonization auditing**.

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

Provide a clean baseline BIDS dataset containing any subset of the supported modalities (`T1w`, `T2w`, `FLAIR`, `bold`, `dwi`). Existing `ses-*` sessions are preserved untouched:

```text
my_bids/
├── dataset_description.json
└── sub-01/
    ├── anat/
    │   ├── sub-01_T1w.nii.gz
    │   ├── sub-01_T1w.json          # Recommended: PhaseEncodingDirection, SliceEncodingDirection
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

- **Zero-leakage blinded output**: Output filenames use shuffled sequential `run-01`, `run-02`, ... indices with **no recipe, artifact, or severity names**, and `.json` sidecars contain **no simulation provenance**.
- **Multi-site isolation (`output.site_naming`)**:
  - `desc` (default for multi-site): `sub-01[_ses-01][_task-rest]_run-01_desc-<Site>_<modality>.nii.gz`
  - `none` (strict raw-BIDS compliant): `sub-01[_ses-01][_task-rest]_run-01_<modality>.nii.gz` (site recorded in `participants.tsv`, `*_scans.tsv`, and `manifest.tsv`)
- **DWI motion & `.bvec` behavior**: By default (`rotate_dwi_bvecs: false`), simulated inter-volume DWI motion leaves `.bvec` files **unrotated**, mirroring raw scanner acquisitions. Set `sampling.rotate_dwi_bvecs: true` when simulating motion-corrected/preprocessed datasets.

---

## Quickstart (CLI)

### 1. (Optional) Synthesize a Clean Multimodal Cohort from 3–5+ Structural Seeds
If you only have clean structural scans (`T1w`, optionally paired with `T2w`), you can expand **3–5+ seed subjects** into $N$ anatomically distinct, cross-modality-aligned subjects (`T1w`, `T2w`, `FLAIR`, 4D `bold`, 4D `dwi`) via coherent diffeomorphic warps—template-free by default, with optional PVE / RSN / DTI templates supported:
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

### 2. Generate Blinded Simulated BIDS Files + Ground-Truth `manifest.tsv`
```bash
uv run --no-sync sim-data generate \
  --bids-dir ./demo_bids \
  --config configs/default_multimodal.yaml \
  --out-dir ./demo_out \
  --manifest-out ./demo_manifest.tsv \
  --workers 4 \
  --device auto
```
- By default (`extract_iqms: false`, `render_views: false`, `write_packets: false`), `generate` writes **only** the blinded BIDS dataset (`sub-*/`, `dataset_description.json`, `participants.tsv`, `*_scans.tsv`, `.bidsignore`) and the official ground-truth **`manifest.tsv`** (recording all injected artifact weights `sev_*`, realized physical parameters, site physics parameters, and continuous `defect_burden`).
- Pass `--manifest-out` to save the ground-truth answer key completely outside the BIDS directory for blinded manual or AI evaluations (defaults to `<out-dir>/manifest.tsv`, ignored by `.bidsignore`).

### 3. Independently Extract Blind NiPreps IQMs (or Run Real MRIQC / fMRIPrep / QSIPrep)
```bash
uv run --no-sync sim-data extract-iqms \
  --bids-dir ./demo_out \
  --out ./demo_iqms.tsv \
  --workers 4
```
Quantifies MRIQC / fMRIPrep / QSIPrep-aligned IQMs **strictly from the modified NIfTI images on disk** with zero access to clean baselines or simulation metadata.

### 4. Audit ComBat vs. CovBat Multi-Site Harmonization
```bash
uv run --no-sync sim-data verify-harmonization \
  --manifest ./demo_manifest.tsv \
  --iqms ./demo_iqms.tsv \
  --out-dir ./demo_harmonization_audit \
  --modality T1w
```
Works with both `sim-data extract-iqms` TSVs and external **MRIQC / QSIPrep** group TSVs (joined via `scan_id` or `bids_name`). Outputs `harmonization_report.json`, `iqms_combat.tsv`, `iqms_covbat.tsv`, and `site_pca_comparison.png`.

### 5. (Optional) Benchmark Deterministic IQM Baseline
```bash
uv run --no-sync sim-data benchmark-deterministic \
  --manifest ./demo_manifest.tsv \
  --iqms ./demo_iqms.tsv \
  --out-dir ./demo_qc_benchmark \
  --modality T1w
```

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
├── dataset_description.json          # BIDS dataset metadata
├── participants.tsv                  # Subject IDs & site mapping
├── .bidsignore                       # Ignores non-BIDS bookkeeping files
├── simulation_config_resolved.yaml
├── manifest.tsv                      # Official ground-truth ledger (or saved via --manifest-out)
└── sub-01/
    ├── sub-01_scans.tsv              # Clean BIDS scans table (filename + site_id only)
    ├── anat/
    │   ├── sub-01_run-01_desc-Siemens3T32ch_T1w.nii.gz
    │   └── sub-01_run-01_desc-Siemens3T32ch_T1w.json   # Clean sidecar (zero simulation provenance)
    ├── func/
    └── dwi/
```

*(Setting `render_views: true` and `write_packets: true` optionally adds `views/` with 3 midline slices + enhanced mosaics/carpet/DWI summaries and `packets/*_packet.json`.)*

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
