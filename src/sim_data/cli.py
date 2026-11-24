"""Typer CLI for sim-data: init-phantom, generate, verify-harmonization, benchmark-deterministic."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from sim_data.config import SimulationConfig
from sim_data.eval.deterministic_qc import evaluate_deterministic_qc
from sim_data.harmonization.audit import run_harmonization_audit
from sim_data.phantoms.synthetic import generate_synthetic_bids_cohort
from sim_data.pipeline.engine import run_simulation_plan

app = typer.Typer(
    name="sim-data",
    help="Physics-ordered multimodal MRI artifact & multi-site QC simulator.",
    no_args_is_help=True,
)
console = Console()


def _load_manifest(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_json(path)


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
    out_dir: Path = typer.Option(..., "--out-dir", help="Output directory for derivatives, IQMs, views & packets."),
    workers: int = typer.Option(1, "--workers", min=1, help="Parallel worker processes."),
    device: str = typer.Option("auto", "--device", help="PyTorch compute device: auto, cpu, or cuda."),
) -> None:
    """Run multimodal MRI artifact & site-effect simulation."""
    cfg = SimulationConfig.from_yaml(config)
    res = run_simulation_plan(
        bids_dir=bids_dir,
        config=cfg,
        out_dir=out_dir,
        workers=workers,
        device=device,
    )
    console.print(f"[green]Generated {res.n_scans_generated} simulated scans in:[/green] {res.out_dir}")
    for fmt, p in res.manifest_paths.items():
        console.print(f"  manifest ({fmt}): {p}")


@app.command("verify-harmonization")
def verify_harmonization(
    manifest: Path = typer.Option(..., "--manifest", exists=True, help="Path to manifest.parquet or manifest.tsv."),
    out_dir: Path = typer.Option(..., "--out-dir", help="Output directory for ComBat/CovBat audit report & plots."),
    modality: str | None = typer.Option(None, "--modality", help="Modality subset to audit (defaults to largest group)."),
    backend: str = typer.Option("auto", "--backend", help="Harmonization backend: auto or combatlearn."),
) -> None:
    """Run ComBat vs. CovBat multi-site harmonization audit on extracted NiPreps IQMs."""
    df = _load_manifest(manifest)
    res = run_harmonization_audit(df, out_dir=out_dir, modality=modality, backend=backend)

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
        "qc_macro_f1",
        "qc_cohen_kappa",
    ]:
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
    manifest: Path = typer.Option(..., "--manifest", exists=True, help="Path to manifest.parquet or manifest.tsv."),
    out_dir: Path = typer.Option(..., "--out-dir", help="Output directory for deterministic QC report."),
    modality: str | None = typer.Option(None, "--modality", help="Modality subset to benchmark."),
    update_packets: Path | None = typer.Option(
        None,
        "--update-packets",
        help="Optional simulation output root whose packet.json files will be enriched with deterministic_priors.",
    ),
) -> None:
    """Benchmark cross-validated Logistic Regression on NiPreps IQMs (pass/borderline/fail)."""
    df = _load_manifest(manifest)
    res = evaluate_deterministic_qc(
        df, out_dir=out_dir, modality=modality, update_packets_root=update_packets
    )
    console.print(
        f"[green]Deterministic CV Logistic Regression ({res.report['modality']}, n={res.report['n_scans']}):[/green] "
        f"Macro-F1={res.report['macro_f1']}, Kappa={res.report['cohen_kappa']}, QWK={res.report['quadratic_weighted_kappa']}"
    )
    console.print(f"Report saved to: {res.report_path}")


if __name__ == "__main__":
    app()
