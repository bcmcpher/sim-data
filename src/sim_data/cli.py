"""Typer CLI for sim-data: init-phantom, generate, extract-iqms, verify-harmonization, benchmark-deterministic."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, cast

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from sim_data.config import Modality, SimulationConfig
from sim_data.eval.deterministic_qc import evaluate_deterministic_qc
from sim_data.harmonization.audit import run_harmonization_audit
from sim_data.iqm.extractor import extract_iqms_for_bids_dir
from sim_data.phantoms.synthetic import generate_synthetic_bids_cohort
from sim_data.pipeline.engine import run_simulation_plan

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


@app.command("generate")
def generate(
    bids_dir: Path = typer.Option(..., "--bids-dir", exists=True, help="Clean input BIDS directory."),
    config: Path = typer.Option(..., "--config", exists=True, help="YAML simulation configuration."),
    out_dir: Path = typer.Option(..., "--out-dir", help="Output blinded BIDS directory."),
    manifest_out: Optional[Path] = typer.Option(
        None,
        "--manifest-out",
        help="Optional path for manifest.tsv (defaults to <out-dir>/manifest.tsv, ignored by .bidsignore).",
    ),
    workers: int = typer.Option(1, "--workers", min=1, help="Parallel worker processes."),
    device: str = typer.Option("auto", "--device", help="PyTorch compute device: auto, cpu, or cuda."),
) -> None:
    """Generate a blinded simulated BIDS dataset and official ground-truth manifest.tsv."""
    cfg = SimulationConfig.from_yaml(config)
    res = run_simulation_plan(
        bids_dir=bids_dir,
        config=cfg,
        out_dir=out_dir,
        workers=workers,
        device=device,
        manifest_out=manifest_out,
    )
    console.print(f"[green]Generated {res.n_scans_generated} blinded BIDS scans in:[/green] {res.out_dir}")
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
