"""Typer CLI for sim-data: init-phantom, convert-contrast, synthesize-cohort, generate, extract-iqms, analyze-sensitivity, verify-harmonization, benchmark-deterministic."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, cast

import nibabel as nib
import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from sim_data.config import (
    CohortSynthesisConfig,
    ContrastConversionConfig,
    Modality,
    SimulationConfig,
)
from sim_data.eval.deterministic_qc import evaluate_deterministic_qc
from sim_data.eval.sensitivity import run_sensitivity_analysis
from sim_data.harmonization.audit import run_harmonization_audit
from sim_data.io.orientation import canonicalize_to_ras
from sim_data.iqm.extractor import extract_iqms_for_bids_dir
from sim_data.phantoms.synthetic import generate_synthetic_bids_cohort
from sim_data.pipeline.engine import run_simulation_plan
from sim_data.synthesis.cohort import synthesize_cohort
from sim_data.synthesis.contrast import ContrastTarget, invert_t1w_to_t2w

app = typer.Typer(
    name="sim-data",
    help="Physics-ordered multimodal MRI artifact & multi-site BIDS simulator.",
    no_args_is_help=True,
)
console = Console()


def _load_tsv(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_csv(path, sep="\t")


@app.command("init-phantom")
def init_phantom(
    out: Path = typer.Option(..., "--out", help="Output BIDS directory for synthetic phantom cohort."),
    n_subjects: int = typer.Option(4, "--n-subjects", min=1, help="Number of synthetic subjects."),
    seed: int = typer.Option(123, "--seed", help="Random seed."),
) -> None:
    """Create an offline synthetic multimodal BIDS dataset (T1w, T2w, FLAIR, bold, dwi)."""
    root = generate_synthetic_bids_cohort(out_dir=out, n_subjects=n_subjects, seed=seed)
    console.print(f"[green]Created synthetic BIDS cohort ({n_subjects} subjects) at:[/green] {root}")


_VALID_CONTRAST_TARGETS: set[str] = {"T2w", "FLAIR", "bold_s0", "dwi_b0"}


@app.command("convert-contrast")
def convert_contrast_cmd(
    t1w: Path = typer.Option(..., "--t1w", exists=True, help="Input clean 3D T1w NIfTI file."),
    out: Path = typer.Option(..., "--out", help="Output NIfTI file path for converted contrast."),
    target: str = typer.Option(
        "T2w", "--target", help="Target contrast polarity: T2w, FLAIR, bold_s0, or dwi_b0."
    ),
    brain_mask: Optional[Path] = typer.Option(
        None, "--brain-mask", exists=True, help="Optional external brain/ICV mask NIfTI (Option 2)."
    ),
    no_n4: bool = typer.Option(
        False, "--no-n4", help="Skip SimpleITK N4ITK bias correction before contrast conversion."
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for GMM tissue anchoring."),
) -> None:
    """Convert a clean T1w NIfTI into continuous T2w, BOLD S0, or DWI b=0 contrast."""
    if target not in _VALID_CONTRAST_TARGETS:
        raise typer.BadParameter(
            f"Unknown target '{target}'. Expected one of {sorted(_VALID_CONTRAST_TARGETS)}"
        )

    img = nib.load(str(t1w))
    if not isinstance(img, nib.Nifti1Image):
        if isinstance(img, nib.spatialimages.SpatialImage):
            img = nib.Nifti1Image(np.asarray(img.get_fdata(), dtype=np.float32), img.affine)
        else:
            raise typer.BadParameter(f"Unsupported image type: {type(img).__name__}")

    t1_data, affine, _zooms = canonicalize_to_ras(img)
    cfg = ContrastConversionConfig(n4_bias_correct=not no_n4)
    res = invert_t1w_to_t2w(
        t1_data=t1_data,
        affine=affine,
        target=cast(ContrastTarget, target),
        cfg=cfg,
        brain_mask_path=brain_mask,
        seed=seed,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(res.data, affine), str(out))
    console.print(
        f"[green]Converted T1w -> {target} ({res.backend}, external_mask={res.used_external_mask}) ->[/green] {out}"
    )
    console.print(f"  T1 anchors (CSF/GM/WM): {res.tissue_anchors_t1}")


@app.command("synthesize-cohort")
def synthesize_cohort_cmd(
    seed_bids: Path = typer.Option(
        ...,
        "--seed-bids",
        exists=True,
        help="Clean real BIDS directory (T1w required; optional T2w/FLAIR/bold/dwi/brain masks).",
    ),
    out: Path = typer.Option(..., "--out", help="Output clean synthesized multimodal BIDS directory."),
    config: Optional[Path] = typer.Option(
        None, "--config", exists=True, help="Optional YAML synthesis config (defaults to CohortSynthesisConfig)."
    ),
    n_subjects: Optional[int] = typer.Option(
        None, "--n-subjects", min=1, help="Override number of synthetic subjects to generate."
    ),
    warp_strength: Optional[float] = typer.Option(
        None, "--warp-strength", min=0.0, max=1.0, help="Override diffeomorphic warp strength [0..1]."
    ),
    allow_few_seeds: bool = typer.Option(
        False, "--allow-few-seeds", help="Allow running with fewer than min_seed_subjects (default 3)."
    ),
    preserve_real: bool = typer.Option(
        False,
        "--preserve-real",
        help="Keep real subjects' existing modalities untouched (zero warp) and synthesize only missing modalities.",
    ),
    brain_mask: Optional[Path] = typer.Option(
        None, "--brain-mask", exists=True, help="Optional external brain/ICV mask NIfTI (Option 2)."
    ),
    tissue_pve: Optional[Path] = typer.Option(
        None, "--tissue-pve", exists=True, help="Optional 4D CSF/GM/WM[/Scalp] PVE template NIfTI."
    ),
    rsn_template: Optional[Path] = typer.Option(
        None, "--rsn-template", exists=True, help="Optional 3D/4D RSN parcellation NIfTI for BOLD."
    ),
    dti_template: Optional[Path] = typer.Option(
        None, "--dti-template", exists=True, help="Optional 4D eigenvector/tensor NIfTI for DWI."
    ),
    workers: int = typer.Option(1, "--workers", min=1, help="Parallel worker processes."),
    device: str = typer.Option("auto", "--device", help="PyTorch compute device: auto, cpu, or cuda."),
) -> None:
    """Expand structural BIDS seeds into N multimodal subjects or complete missing modalities (--preserve-real)."""
    cfg = CohortSynthesisConfig.from_yaml(config) if config is not None else CohortSynthesisConfig()
    if n_subjects is not None:
        cfg.n_subjects = n_subjects
    if warp_strength is not None:
        cfg.warp.strength = warp_strength
    if allow_few_seeds:
        cfg.allow_few_seeds = True
    if preserve_real:
        cfg.preserve_real_modalities = True
    if brain_mask is not None:
        cfg.templates.brain_mask_nifti = brain_mask
    if tissue_pve is not None:
        cfg.templates.tissue_pve_nifti = tissue_pve
    if rsn_template is not None:
        cfg.templates.rsn_parcellation_nifti = rsn_template
    if dti_template is not None:
        cfg.templates.dti_eigenvector_nifti = dti_template

    res = synthesize_cohort(
        seed_bids_dir=seed_bids,
        config=cfg,
        out_dir=out,
        workers=workers,
        device=device,
    )
    mode_lbl = "Completed native real" if cfg.preserve_real_modalities else "Synthesized"
    console.print(
        f"[green]{mode_lbl} {res.n_subjects} multimodal subject(s) from {res.n_seed_subjects} seed(s) in:[/green] {res.out_dir}"
    )
    console.print(f"  synthesis manifest: {res.manifest_path}")


@app.command("generate")
def generate(
    bids_dir: Path = typer.Option(..., "--bids-dir", exists=True, help="Clean input BIDS directory."),
    config: Path = typer.Option(..., "--config", exists=True, help="YAML simulation configuration."),
    out_dir: Path = typer.Option(..., "--out-dir", help="Output simulated BIDS directory."),
    manifest_out: Optional[Path] = typer.Option(
        None,
        "--manifest-out",
        help="Optional path for manifest.tsv (defaults to <out-dir>/manifest.tsv, ignored by .bidsignore).",
    ),
    workers: int = typer.Option(1, "--workers", min=1, help="Parallel worker processes."),
    device: str = typer.Option("auto", "--device", help="PyTorch compute device: auto, cpu, or cuda."),
) -> None:
    """Generate a simulated BIDS dataset and official ground-truth manifest.tsv."""
    cfg = SimulationConfig.from_yaml(config)
    if cfg.output.naming_scheme == "unblinded_validation":
        console.print(
            "[yellow]WARNING: output.naming_scheme='unblinded_validation' embeds artifact/dose labels "
            "in filenames. Use ONLY for internal validation/debugging—never for blinded human or AI QC evaluation.[/yellow]"
        )
    res = run_simulation_plan(
        bids_dir=bids_dir,
        config=cfg,
        out_dir=out_dir,
        workers=workers,
        device=device,
        manifest_out=manifest_out,
    )
    console.print(
        f"[green]Generated {res.n_scans_generated} BIDS scans (naming={cfg.output.naming_scheme}) in:[/green] {res.out_dir}"
    )
    console.print(f"  manifest (tsv): {res.manifest_path}")


_VALID_MODALITIES: set[str] = {"T1w", "T2w", "FLAIR", "bold", "dwi"}


@app.command("extract-iqms")
def extract_iqms_cmd(
    bids_dir: Path = typer.Option(..., "--bids-dir", exists=True, help="BIDS directory to evaluate blindly."),
    out: Path = typer.Option(..., "--out", help="Output TSV file path for extracted NiPreps IQMs."),
    modality: Optional[list[str]] = typer.Option(
        None, "--modality", help="Optional modality filter (repeatable: T1w, T2w, FLAIR, bold, dwi)."
    ),
    workers: int = typer.Option(1, "--workers", min=1, help="Parallel worker processes."),
) -> None:
    """Independently compute blind NiPreps-aligned IQMs purely from NIfTI files in a BIDS directory."""
    mods: list[Modality] | None = None
    if modality:
        invalid = [m for m in modality if m not in _VALID_MODALITIES]
        if invalid:
            raise typer.BadParameter(
                f"Unknown modality: {invalid}. Expected subset of {sorted(_VALID_MODALITIES)}"
            )
        mods = [cast(Modality, m) for m in modality]

    df = extract_iqms_for_bids_dir(
        bids_dir=bids_dir,
        out_tsv=out,
        modalities=mods,
        workers=workers,
    )
    console.print(f"[green]Extracted blind IQMs for {len(df)} scans ->[/green] {out}")


@app.command("analyze-sensitivity")
def analyze_sensitivity_cmd(
    manifest: Path = typer.Option(..., "--manifest", exists=True, help="Path to ground-truth manifest.tsv."),
    out_dir: Path = typer.Option(..., "--out-dir", help="Output directory for sensitivity summary, matrix & plots."),
    iqms: Optional[Path] = typer.Option(
        None,
        "--iqms",
        exists=True,
        help="Optional external IQMs TSV (from `sim-data extract-iqms` or MRIQC/fMRIPrep/QSIPrep).",
    ),
    modality: Optional[str] = typer.Option(None, "--modality", help="Modality subset to analyze."),
) -> None:
    """Rank IQM sensitivity, monotonicity (Spearman ρ), MDS/LoD, and specificity across dose trajectories."""
    df = _load_tsv(manifest)
    iqms_df = _load_tsv(iqms) if iqms is not None else None
    res = run_sensitivity_analysis(df, out_dir=out_dir, iqms_df=iqms_df, modality=modality)

    table = Table(
        title=f"Top Sensitive IQMs ({res.report['modality']}, trajectories={res.report['n_trajectories']})"
    )
    table.add_column("Artifact")
    table.add_column("Top IQM")
    table.add_column("Spearman ρ", justify="right")
    table.add_column("Max Cohen's d", justify="right")
    table.add_column("MDS / LoD", justify="right")
    table.add_column("Specificity", justify="right")
    for art, top_list in res.report["top_iqms_per_artifact"].items():
        if not top_list:
            continue
        best = top_list[0]
        table.add_row(
            art,
            str(best["iqm"]),
            f"{best['mean_spearman_rho']:+.3f}",
            f"{best['max_effect_cohens_d']:+.2f}",
            str(best["min_detectable_severity"]) if best["min_detectable_severity"] is not None else "-",
            f"{best['specificity_ratio']:.2f}",
        )
    console.print(table)
    console.print(f"Sensitivity summary saved to: {res.summary_path}")


@app.command("verify-harmonization")
def verify_harmonization(
    manifest: Path = typer.Option(..., "--manifest", exists=True, help="Path to ground-truth manifest.tsv."),
    out_dir: Path = typer.Option(..., "--out-dir", help="Output directory for ComBat/CovBat audit report & plots."),
    iqms: Optional[Path] = typer.Option(
        None,
        "--iqms",
        exists=True,
        help="Optional external IQMs TSV (from `sim-data extract-iqms` or MRIQC/QSIPrep).",
    ),
    modality: Optional[str] = typer.Option(None, "--modality", help="Modality subset to audit."),
    backend: str = typer.Option("auto", "--backend", help="Harmonization backend: auto or combatlearn."),
) -> None:
    """Run ComBat vs. CovBat multi-site harmonization audit on blind IQMs."""
    df = _load_tsv(manifest)
    iqms_df = _load_tsv(iqms) if iqms is not None else None
    res = run_harmonization_audit(
        df, out_dir=out_dir, iqms_df=iqms_df, modality=modality, backend=backend
    )

    table = Table(title=f"ComBat vs. CovBat Audit ({res.report['modality']}, n={res.report['n_scans']})")
    table.add_column("Metric")
    table.add_column("Raw", justify="right")
    table.add_column("ComBat", justify="right")
    table.add_column("CovBat", justify="right")
    for key in [
        "mean_anova_f",
        "mean_site_eta2",
        "mean_levene_stat",
        "covariance_frobenius_dist",
        "correlation_frobenius_dist",
        "site_linear_auc",
        "site_rf_auc",
        "defect_burden_r2",
    ]:
        if key in res.report["raw"]:
            table.add_row(
                key,
                str(res.report["raw"].get(key, "-")),
                str(res.report["combat"].get(key, "-")),
                str(res.report["covbat"].get(key, "-")),
            )
    console.print(table)
    console.print(f"Report saved to: {res.report_path}")


@app.command("benchmark-deterministic")
def benchmark_deterministic(
    manifest: Path = typer.Option(..., "--manifest", exists=True, help="Path to ground-truth manifest.tsv."),
    out_dir: Path = typer.Option(..., "--out-dir", help="Output directory for deterministic QC report."),
    iqms: Optional[Path] = typer.Option(
        None,
        "--iqms",
        exists=True,
        help="Optional external IQMs TSV (from `sim-data extract-iqms` or MRIQC/QSIPrep).",
    ),
    modality: Optional[str] = typer.Option(None, "--modality", help="Modality subset to benchmark."),
    label_col: Optional[str] = typer.Option(
        None, "--label-col", help="Optional rating column name (defaults to qc_label/proxy_qc_label or binned defect_burden)."
    ),
    update_packets: Optional[Path] = typer.Option(
        None,
        "--update-packets",
        help="Optional simulation output root whose packet.json files will be enriched with deterministic_priors.",
    ),
) -> None:
    """Benchmark cross-validated Logistic Regression on blind NiPreps IQMs."""
    df = _load_tsv(manifest)
    iqms_df = _load_tsv(iqms) if iqms is not None else None
    res = evaluate_deterministic_qc(
        df,
        out_dir=out_dir,
        iqms_df=iqms_df,
        modality=modality,
        label_col=label_col,
        update_packets_root=update_packets,
    )
    console.print(
        f"[green]Deterministic CV Logistic Regression ({res.report['modality']}, n={res.report['n_scans']}):[/green] "
        f"Macro-F1={res.report['macro_f1']}, Kappa={res.report['cohen_kappa']}, QWK={res.report['quadratic_weighted_kappa']}"
    )
    console.print(f"Report saved to: {res.report_path}")


if __name__ == "__main__":
    app()
