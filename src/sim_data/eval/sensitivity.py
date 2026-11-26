"""Decoupled dose-response IQM sensitivity & specificity analysis.

Consumes `manifest.tsv` (with `trajectory_id`, `target_artifact`, `dose_level`, `baseline_scan_id`)
plus any blind IQM TSV (`sim-data extract-iqms`, MRIQC, fMRIPrep, or QSIPrep) to rank which
metrics are most monotonically sensitive and specific to each simulated MRI artifact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from sim_data.harmonization.audit import merge_manifest_and_iqms


@dataclass
class SensitivityAnalysisResult:
    report: dict[str, Any]
    report_path: Path
    summary_path: Path
    matrix_path: Path
    heatmap_path: Path | None
    curves_path: Path | None
    summary_df: pd.DataFrame
    matrix_df: pd.DataFrame


def _select_iqm_columns(df: pd.DataFrame) -> list[str]:
    iqm_cols = [c for c in df.columns if c.startswith("iqm_")]
    usable: list[str] = []
    for col in iqm_cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().mean() < 0.2 and float(vals.std(skipna=True)) > 1e-8:
            usable.append(col)
    return usable


def _plot_sensitivity_heatmap(matrix_df: pd.DataFrame, out_path: Path) -> None:
    if matrix_df.empty:
        return
    # Keep up to 18 most responsive IQMs for readable labels
    max_abs = matrix_df.abs().max(axis=0).sort_values(ascending=False)
    keep_cols = max_abs.index.tolist()[:18]
    sub = matrix_df[keep_cols]
    clean_cols = [str(c).removeprefix("iqm_") for c in sub.columns]

    fig_w = max(7.0, 0.55 * len(clean_cols) + 2.5)
    fig_h = max(3.5, 0.45 * len(sub.index) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)
    im = ax.imshow(sub.to_numpy(dtype=np.float64), cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(clean_cols)))
    ax.set_xticklabels(clean_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(sub.index)))
    ax.set_yticklabels([str(i) for i in sub.index], fontsize=8)
    ax.set_title("Within-Trajectory IQM Sensitivity (Mean Spearman ρ vs. Dose)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Spearman ρ")
    fig.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


def _plot_dose_response_curves(
    curve_records: dict[str, dict[str, list[tuple[float, float]]]],
    top_by_artifact: dict[str, list[str]],
    out_path: Path,
) -> None:
    artifacts = [a for a in sorted(top_by_artifact) if top_by_artifact[a]]
    if not artifacts:
        return

    n_art = len(artifacts)
    ncols = min(3, n_art)
    nrows = int(np.ceil(n_art / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.2 * nrows), dpi=115, squeeze=False)

    for idx, art in enumerate(artifacts):
        ax = axes[idx // ncols][idx % ncols]
        for col in top_by_artifact[art][:3]:
            pts = curve_records.get(art, {}).get(col, [])
            if not pts:
                continue
            by_dose: dict[float, list[float]] = {}
            for d, z in pts:
                by_dose.setdefault(round(d, 4), []).append(z)
            xs = sorted(by_dose)
            ys = [float(np.mean(by_dose[x])) for x in xs]
            ax.plot(xs, ys, marker="o", lw=1.6, ms=3.5, label=col.removeprefix("iqm_"))
        ax.axhline(0.0, color="#888888", ls="--", lw=0.7)
        ax.set_title(art, fontsize=9)
        ax.set_xlabel("Dose Severity", fontsize=8)
        ax.set_ylabel("Δ IQM (SD units)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="best")

    for idx in range(n_art, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("Dose-Response IQM Sensitivity Curves (Top-3 IQMs per Artifact)", fontsize=11)
    fig.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


def run_sensitivity_analysis(
    manifest_df: pd.DataFrame,
    out_dir: str | Path,
    iqms_df: pd.DataFrame | None = None,
    modality: str | None = None,
) -> SensitivityAnalysisResult:
    """Quantify IQM monotonicity, effect size, MDS/LoD, and specificity across dose trajectories."""
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = merge_manifest_and_iqms(manifest_df, iqms_df)
    if modality is not None and "modality" in df.columns:
        df = df[df["modality"] == modality].reset_index(drop=True)
    elif "modality" in df.columns and df["modality"].nunique() > 1:
        modality = str(df["modality"].value_counts().index[0])
        df = df[df["modality"] == modality].reset_index(drop=True)

    iqm_cols = _select_iqm_columns(df)
    if not iqm_cols:
        raise ValueError(
            "No valid numeric iqm_* columns found. Pass --iqms from `sim-data extract-iqms` or MRIQC/QSIPrep."
        )

    for col in iqm_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "target_artifact" not in df.columns or "dose_level" not in df.columns:
        raise ValueError(
            "Manifest must contain 'target_artifact' and 'dose_level' columns "
            "(generated by `sim-data generate` in `dose_response` mode)."
        )

    df["dose_level"] = pd.to_numeric(df["dose_level"], errors="coerce").fillna(0.0)
    df["target_artifact"] = df["target_artifact"].fillna("none").astype(str)
    if "trajectory_id" not in df.columns:
        df["trajectory_id"] = df["subject"].astype(str) + "__" + df["target_artifact"]
    if "baseline_scan_id" not in df.columns:
        df["baseline_scan_id"] = ""

    # Map baseline scan_id -> row for fast s=0 attachment
    baseline_rows = df[(df["target_artifact"] == "baseline") | (df["dose_level"] <= 0.0)]
    base_lookup: dict[str, pd.Series] = {}
    for _, brow in baseline_rows.iterrows():
        base_lookup[str(brow["scan_id"])] = brow

    # Reference scale per IQM (baseline SD when >=2 baselines vary, else overall SD)
    ref_sd: dict[str, float] = {}
    for col in iqm_cols:
        b_sd = float(baseline_rows[col].std(skipna=True)) if len(baseline_rows) >= 2 else 0.0
        o_sd = float(df[col].std(skipna=True))
        ref_sd[col] = b_sd if (np.isfinite(b_sd) and b_sd > 1e-6) else max(o_sd, 1e-6)

    dose_rows = df[
        (df["target_artifact"] != "baseline")
        & (df["target_artifact"] != "none")
        & (df["dose_level"] > 0.0)
    ].copy()
    if dose_rows.empty:
        raise ValueError("No positive dose-response trajectory rows found in manifest.")

    artifacts = sorted(dose_rows["target_artifact"].unique().tolist())

    # Collect per-(artifact, iqm) trajectory metrics & curve points
    raw_stats: dict[tuple[str, str], dict[str, Any]] = {}
    curve_records: dict[str, dict[str, list[tuple[float, float]]]] = {}

    for art in artifacts:
        art_df = dose_rows[dose_rows["target_artifact"] == art]
        curve_records[art] = {}
        trajs = list(art_df.groupby("trajectory_id"))

        for col in iqm_cols:
            rhos: list[float] = []
            slopes: list[float] = []
            max_ds: list[float] = []
            dose_deltas: dict[float, list[float]] = {}
            pts_for_plot: list[tuple[float, float]] = []
            sd_scale = ref_sd[col]

            for _tid, grp in trajs:
                g = grp.sort_values("dose_level")
                doses = g["dose_level"].to_numpy(dtype=np.float64).tolist()
                vals = g[col].to_numpy(dtype=np.float64).tolist()

                # Prepend shared s=0 baseline if linked
                b_id = str(g["baseline_scan_id"].iloc[0]) if "baseline_scan_id" in g.columns else ""
                if b_id and b_id in base_lookup and np.isfinite(base_lookup[b_id][col]):
                    b_val = float(base_lookup[b_id][col])
                    doses = [0.0] + doses
                    vals = [b_val] + vals

                valid_pairs = [(d, v) for d, v in zip(doses, vals, strict=True) if np.isfinite(v)]
                if len(valid_pairs) < 2:
                    continue
                d_arr = np.array([p[0] for p in valid_pairs], dtype=np.float64)
                v_arr = np.array([p[1] for p in valid_pairs], dtype=np.float64)
                if np.std(d_arr) <= 1e-9:
                    continue

                if np.std(v_arr) <= 1e-12:
                    rho = 0.0
                else:
                    r_val, _ = stats.spearmanr(d_arr, v_arr)
                    rho = float(r_val) if np.isfinite(r_val) else 0.0
                rhos.append(rho)

                delta_z = (v_arr - v_arr[0]) / sd_scale
                slope = float(np.polyfit(d_arr, delta_z, 1)[0])
                slopes.append(slope)
                max_ds.append(float(delta_z[-1]))

                for d_i, z_i in zip(d_arr, delta_z, strict=True):
                    pts_for_plot.append((float(d_i), float(z_i)))
                    if d_i > 0.0:
                        dose_deltas.setdefault(round(float(d_i), 4), []).append(float(z_i))

            curve_records[art][col] = pts_for_plot

            # Minimum Detectable Severity (MDS / LoD): lowest dose where |mean delta_z| >= 2.0
            mds: float | None = None
            for d_key in sorted(dose_deltas):
                if abs(float(np.mean(dose_deltas[d_key]))) >= 2.0:
                    mds = float(d_key)
                    break

            raw_stats[(art, col)] = {
                "n_trajectories": len(rhos),
                "mean_spearman_rho": float(np.mean(rhos)) if rhos else 0.0,
                "std_spearman_rho": float(np.std(rhos)) if rhos else 0.0,
                "standardized_slope": float(np.mean(slopes)) if slopes else 0.0,
                "max_effect_cohens_d": float(np.mean(max_ds)) if max_ds else 0.0,
                "min_detectable_severity": mds,
            }

    # Build summary rows with cross-artifact specificity ratio
    summary_rows: list[dict[str, Any]] = []
    for art in artifacts:
        for col in iqm_cols:
            st = raw_stats[(art, col)]
            abs_self = abs(st["mean_spearman_rho"])
            if len(artifacts) > 1:
                other_max = max(
                    abs(raw_stats[(o_art, col)]["mean_spearman_rho"])
                    for o_art in artifacts
                    if o_art != art
                )
                spec = abs_self / max(other_max, 0.05)
            else:
                spec = 1.0

            summary_rows.append(
                {
                    "modality": modality or "all",
                    "target_artifact": art,
                    "iqm": col.removeprefix("iqm_"),
                    "iqm_col": col,
                    "n_trajectories": int(st["n_trajectories"]),
                    "mean_spearman_rho": round(st["mean_spearman_rho"], 4),
                    "abs_mean_rho": round(abs_self, 4),
                    "std_spearman_rho": round(st["std_spearman_rho"], 4),
                    "standardized_slope": round(st["standardized_slope"], 4),
                    "max_effect_cohens_d": round(st["max_effect_cohens_d"], 4),
                    "abs_max_d": round(abs(float(st["max_effect_cohens_d"])), 4),
                    "min_detectable_severity": (
                        round(float(st["min_detectable_severity"]), 4)
                        if st["min_detectable_severity"] is not None
                        else np.nan
                    ),
                    "specificity_ratio": round(float(spec), 3),
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["target_artifact", "abs_mean_rho", "abs_max_d", "specificity_ratio"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    matrix_df = summary_df.pivot(
        index="target_artifact", columns="iqm_col", values="mean_spearman_rho"
    ).fillna(0.0)

    top_by_artifact_cols: dict[str, list[str]] = {}
    top_iqms_report: dict[str, list[dict[str, Any]]] = {}
    blind_spots: list[str] = []

    for art in artifacts:
        sub = summary_df[summary_df["target_artifact"] == art].head(5)
        top_by_artifact_cols[art] = sub["iqm_col"].tolist()
        top_iqms_report[art] = [
            {
                "iqm": str(r["iqm"]),
                "mean_spearman_rho": float(r["mean_spearman_rho"]),
                "standardized_slope": float(r["standardized_slope"]),
                "max_effect_cohens_d": float(r["max_effect_cohens_d"]),
                "min_detectable_severity": (
                    float(r["min_detectable_severity"])
                    if pd.notna(r["min_detectable_severity"])
                    else None
                ),
                "specificity_ratio": float(r["specificity_ratio"]),
            }
            for _, r in sub.iterrows()
        ]
        if sub.empty or float(sub["abs_mean_rho"].max()) < 0.60:
            blind_spots.append(art)

    summary_path = out_root / "sensitivity_summary.tsv"
    matrix_path = out_root / "sensitivity_matrix.tsv"
    report_path = out_root / "sensitivity_report.json"
    heatmap_path = out_root / "sensitivity_heatmap.png"
    curves_path = out_root / "dose_response_curves.png"

    summary_df.to_csv(summary_path, sep="\t", index=False)
    matrix_df.to_csv(matrix_path, sep="\t")
    _plot_sensitivity_heatmap(matrix_df, heatmap_path)
    _plot_dose_response_curves(curve_records, top_by_artifact_cols, curves_path)

    report: dict[str, Any] = {
        "modality": modality or "all",
        "n_scans": int(len(df)),
        "n_trajectories": int(dose_rows["trajectory_id"].nunique()),
        "n_iqm_features": len(iqm_cols),
        "artifacts": artifacts,
        "blind_spot_artifacts": blind_spots,
        "top_iqms_per_artifact": top_iqms_report,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return SensitivityAnalysisResult(
        report=report,
        report_path=report_path,
        summary_path=summary_path,
        matrix_path=matrix_path,
        heatmap_path=heatmap_path if heatmap_path.exists() else None,
        curves_path=curves_path if curves_path.exists() else None,
        summary_df=summary_df,
        matrix_df=matrix_df,
    )
