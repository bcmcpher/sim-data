# `ghibli` Getting Started Tutorial & End-to-End Verification

This tutorial walks through **every major working path** of `ghibli` starting from **any single clean 3D T1w NIfTI image**. It is designed both as a hands-on onboarding guide and as a diagnostic verification harness that surfaces calibration, anatomical masking, and IQM sensitivity gaps on real clinical/research MRI data.

## Quickstart (One Command)

### Option A: Run on the MR-ART No-Motion Example (`sub-000103`, score = 1)
```bash
uv run --no-sync python tutorials/getting_started/run_all.py \
  --t1w ../data/mrart/bids/sub-000103/anat/sub-000103_acq-standard_T1w.nii.gz \
  --out-dir tutorial_output/mrart_sub000103 \
  --workers 4
```

### Option B: Run on Any T1w NIfTI Image
```bash
uv run --no-sync python tutorials/getting_started/run_all.py \
  --t1w /path/to/your_clean_T1w.nii.gz \
  --out-dir tutorial_output/my_subject \
  --workers 4
```
*(Optional: pass `--brain-mask /path/to/brain_mask.nii.gz` to test Option 2 external ICV masking during modality synthesis, or `--t1w-only` to skip the 4D BOLD/DWI artifact sweep for faster execution.)*

### Option C: Zero-Data Offline Smoke Run (Synthetic Phantom)
```bash
uv run --no-sync python tutorials/getting_started/run_all.py \
  --use-phantom \
  --out-dir tutorial_output/phantom_demo \
  --workers 2
```

---

## Step-by-Step Scripts

Each script can also be executed independently on the same `--out-dir` to inspect or re-run individual stages:

| Script | What It Tests | Key `ghibli` Paths Exercised |
| :--- | :--- | :--- |
| `01_stage_input.py` | Stages any 3D T1w (+ optional JSON/mask) into a clean 1-subject BIDS seed & inspects orientation, encoding axes, foreground mask, and native noise $\sigma_0$. | `canonicalize_to_ras_with_transform`, `resolve_encoding_axes`, `compute_anatomical_priors`, `estimate_background_noise_sigma` |
| `02_convert_modalities.py` | Converts the T1w into all supported contrasts & modalities (`T2w`, `FLAIR`, 4D `bold`, 4D `dwi`) while keeping the real T1w untouched. | `convert-contrast` (`invert_t1w_to_t2w`), `synthesize-cohort --preserve-real`, `extract-iqms`, `render_scan_views` |
| `03_artifact_dose_sweep.py` | Applies **all 17/17 artifacts** individually across **Baseline / Low (0.25) / Mid (0.55) / High (0.85)** severities (11 structural on real T1w + 6 4D/EPI/DWI on synthesized BOLD/DWI) with CRN coupling & zero-leakage `acq_blinded` naming. | `generate` (`dose_response`, `NativeSite` passthrough, `excess_variance`), `extract-iqms`, `analyze-sensitivity` |
| `04_multisite_simulation.py` | Simulates the same subject across `NativeSite` + all **4 curated scanner profiles** (`Siemens_3T_32ch`, `GE_15T_8ch`, `Philips_3T_16ch`, `Siemens_3T_64ch_HighAccel`), plus a multi-site factorial mini-batch. | `sites_catalog.yaml`, `generate` (`multisite_factorial`, `blinded` + `desc-<Site>`), `verify-harmonization` (ComBat vs. CovBat), `benchmark-deterministic` |
| `05_build_report.py` | Compiles all timings, tables, visual contact sheets, and automated sanity checks into `report.html`, `report.md`, and `verification_audit.json`. | End-to-end diagnostic summary |

---

## Output Directory Layout

```text
<out-dir>/
├── 00_seed_bids/                 # Clean staged 1-subject BIDS input
├── 01_contrast_standalone/       # Standalone convert-contrast outputs (T2w, FLAIR, bold_s0, dwi_b0)
├── 01_multimodal_bids/           # Completed 5-modality BIDS (real T1w + synthesized T2w/FLAIR/bold/dwi)
├── 02_artifacts_t1w/             # 34 blinded T1w dose-response scans + manifest + sensitivity report
├── 02_artifacts_4d/              # 29 blinded BOLD/DWI dose-response scans + manifests + sensitivity reports
├── 03_sites_clean/               # 5 clean same-subject scans (Native + 4 curated sites) + IQM shifts
├── 03_sites_factorial/           # 24 multi-site scans + ComBat/CovBat audit + deterministic QC benchmark
├── figures/                      # Annotated PNG visual inspection sheets + 4D animated GIFs
├── verification_audit.json       # Machine-readable check results, timings, and flagged gaps
├── report.html                   # Browser-ready HTML report with clickable full-res figures
└── report.md                     # Markdown version of the verification report
```

## Interpreting the Report & Discovering Gaps

Open `<out-dir>/report.html` in any web browser (e.g., `firefox <out-dir>/report.html`; pass `--embed-images` to base64-inline all PNGs into a single portable HTML file) to inspect:
- **Anatomical Priors (`01_input_priors.png`)**: Does `compute_anatomical_priors` include long neck/jaw/face regions that shift bounding-box-based artifacts (`fov_cropping`, `defacing_damage`, `susceptibility_metal`)?
- **Contrast & Modality Synthesis (`02a_contrast_compartments.png`, `02b_all_modalities_overview.png`)**: Without an external `--brain-mask`, how cleanly are skull, scalp, and neck separated from intracranial CSF/GM/WM?
- **Artifact Contact Sheets (`03a_t1w_artifacts_grid_*.png`, `03b_4d_artifacts_grid.png`)**: Are Low (0.25), Mid (0.55), and High (0.85) visually distinct, realistic, and monotonic on real 1 mm resolution? Which artifacts are IQM blind spots?
- **Multi-Site Comparison (`04a_sites_comparison.png`)**: How do site `contrast_gamma`, `intensity_gain`, `intensity_offset`, `psf_sigma_vox`, and `g_factor_strength` reshape tissue histograms, background air floor, and IQMs?
