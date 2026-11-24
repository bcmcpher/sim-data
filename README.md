# `sim-data`: Multimodal MRI Artifact & Multi-Site QC Simulator

`sim-data` generates realistic, ground-truthed multimodal MRI quality-control datasets (**T1w, T2w, FLAIR, 4D BOLD fMRI, and 4D DWI**) from clean BIDS seed scans. It combines **MONAI**, **TorchIO**, and **custom MRI physics transforms** in a strict 7-stage causal acquisition order, extracts **NiPreps-aligned IQMs** (MRIQC, fMRIPrep, QSIPrep), simulates **image-domain scanner/site effects** verifiable by **ComBat** and **CovBat**, and produces **`qc-studio`-compatible visual-QC packets**.

## Installation

Choose either the **CPU-only** or **CUDA 12.8 (`cu128`, NVIDIA RTX 5080 / Blackwell)** installation:

```bash
# CPU-only workstation
uv sync --extra cpu --extra harmonization --group dev

# NVIDIA RTX 5080 (Blackwell sm_120 / CUDA 12.8) workstation
uv sync --extra cu128 --extra harmonization --group dev
```

## Quickstart

```bash
# 1. (Optional) Create an offline synthetic multimodal BIDS dataset for smoke testing
uv run sim-data init-phantom --out ./demo_bids --n-subjects 4

# 2. Generate simulated multi-site, multi-artifact BIDS derivatives, IQMs, views & packets
uv run sim-data generate \
  --bids-dir ./demo_bids \
  --config configs/default_multimodal.yaml \
  --out-dir ./demo_out \
  --workers 4 \
  --device auto

# 3. Verify ComBat vs. CovBat site harmonization on extracted IQMs
uv run sim-data verify-harmonization \
  --manifest ./demo_out/manifest.parquet \
  --out-dir ./demo_out/harmonization_audit

# 4. Run cross-validated deterministic IQM logistic regression baseline
uv run sim-data benchmark-deterministic \
  --manifest ./demo_out/manifest.parquet \
  --out-dir ./demo_out/qc_benchmark
```
