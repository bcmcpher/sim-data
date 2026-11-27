# `ghibli` Empirical Benchmarks

Reproducible benchmark scripts and configs validating `ghibli`'s built-in IQMs, artifact physics, multi-site harmonization, and interoperability against real datasets (such as MR-ART + MRIQC 24.0.2).

> **Repository Hygiene:** Only lightweight Python scripts, YAML configs, and documentation live under `benchmarks/`. All generated datasets, extracted IQMs, MRIQC derivatives, figures, and reports are written to `benchmark_output/` (gitignored) or any custom `--out-dir` outside the repository.

## Benchmark Suite

| Script | Stage | Description |
| :--- | :--- | :--- |
| `01_mriqc_alignment.py` | **Stage 1A** | Cross-validates `ghibli extract-iqms` against official **MRIQC 24.0.2** (`group_T1w.tsv`) and human ratings (`scores.tsv`) across all 436 MR-ART scans, reporting cross-sectional & within-subject motion-delta alignment, Score 1/2/3 ROC-AUC parity, and empirical motion-delta quantiles for Stage 1B calibration. |

## Quickstart (Stage 1A)

```bash
# Conservative default (2 CPU workers, resumable cache in benchmark_output/01_mriqc_alignment)
uv run --no-sync python benchmarks/01_mriqc_alignment.py \
  --bids-dir ../data/mrart/bids \
  --mriqc-tsv ../data/nipoppy/derivatives/mriqc/24.0.2/output/group_T1w.tsv \
  --scores-tsv ../data/mrart/derivatives/scores.tsv \
  --out-dir benchmark_output/01_mriqc_alignment \
  --workers 2
```

Pass `--limit-subjects 20` for a quick 60-scan preview before running all 148 subjects (436 scans). Re-running with the same `--out-dir` automatically reuses cached `ghibli_iqms.tsv` rows.
