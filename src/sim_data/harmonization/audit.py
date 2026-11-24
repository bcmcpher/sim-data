"""Statistical & ML verification audit comparing Raw vs. ComBat vs. CovBat IQMs."""

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
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sim_data.harmonization.combat import harmonize_combat, harmonize_covbat


@dataclass
class HarmonizationAuditResult:
    report: dict[str, Any]
    report_path: Path
    plot_path: Path | None
    combat_df: pd.DataFrame
    covbat_df: pd.DataFrame


def _select_iqm_columns(df: pd.DataFrame) -> list[str]:
    iqm_cols = [c for c in df.columns if c.startswith("iqm_")]
    usable: list[str] = []
    for col in iqm_cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().mean() < 0.1 and float(vals.std(skipna=True)) > 1e-8:
            usable.append(col)
    return usable


def _site_location_scale_stats(
    X_std: np.ndarray,
    sites: np.ndarray,
) -> dict[str, float]:
    unique_sites = sorted(pd.unique(sites).tolist())
    n_feat = X_std.shape[1]
    f_stats: list[float] = []
    eta2_list: list[float] = []
    levene_stats: list[float] = []

    for j in range(n_feat):
        groups = [X_std[sites == s, j] for s in unique_sites if np.sum(sites == s) >= 2]
        if len(groups) < 2:
            continue
        f_val, _ = stats.f_oneway(*groups)
        if np.isfinite(f_val):
            f_stats.append(float(f_val))
        # Eta-squared (fraction of variance explained by site)
        grand = float(np.mean(X_std[:, j]))
        ss_between = sum(len(g) * (float(np.mean(g)) - grand) ** 2 for g in groups)
        ss_total = float(np.sum((X_std[:, j] - grand) ** 2)) + 1e-10
        eta2_list.append(float(ss_between / ss_total))

        lev_val, _ = stats.levene(*groups)
        if np.isfinite(lev_val):
            levene_stats.append(float(lev_val))

    return {
        "mean_anova_f": round(float(np.mean(f_stats)) if f_stats else 0.0, 4),
        "max_anova_f": round(float(np.max(f_stats)) if f_stats else 0.0, 4),
        "mean_site_eta2": round(float(np.mean(eta2_list)) if eta2_list else 0.0, 5),
        "mean_levene_stat": round(float(np.mean(levene_stats)) if levene_stats else 0.0, 4),
    }


def _covariance_and_correlation_divergence(
    X: np.ndarray,
    sites: np.ndarray,
    covars: np.ndarray | None = None,
) -> tuple[float, float]:
    """Mean pairwise Frobenius distance between per-site residual covariance & correlation matrices.

    Standardizes residuals by the pooled within-site SD (not total pooled SD) so that
    removing between-site location shifts does not artificially inflate covariance scale.
    """
    unique_sites = sorted(pd.unique(sites).tolist())
    n_samples = X.shape[0]
    if covars is not None and covars.size > 0:
        cov = np.asarray(covars, dtype=np.float64)
        if cov.ndim == 1:
            cov = cov[:, None]
        X_des = np.hstack([np.ones((n_samples, 1), dtype=np.float64), cov - cov.mean(axis=0)])
        B, _, _, _ = np.linalg.lstsq(X_des, X, rcond=None)
        E = X - X_des @ B
    else:
        E = X - np.mean(X, axis=0, keepdims=True)

    # Pooled within-site SD
    E_centered = np.empty_like(E)
    for s in unique_sites:
        m = sites == s
        E_centered[m] = E[m] - np.mean(E[m], axis=0, keepdims=True)
    within_sd = np.std(E_centered, axis=0, keepdims=True) + 1e-8
    E_std = E / within_sd

    covs: list[np.ndarray] = []
    corrs: list[np.ndarray] = []
    for s in unique_sites:
        sub = E_std[sites == s]
        if sub.shape[0] >= 3:
            covs.append(np.cov(sub, rowvar=False))
            corrs.append(np.nan_to_num(np.corrcoef(sub, rowvar=False), nan=0.0))

    if len(covs) < 2:
        return (0.0, 0.0)

    cov_dists: list[float] = []
    corr_dists: list[float] = []
    for i in range(len(covs)):
        for j in range(i + 1, len(covs)):
            cov_dists.append(float(np.linalg.norm(covs[i] - covs[j], ord="fro")))
            corr_dists.append(float(np.linalg.norm(corrs[i] - corrs[j], ord="fro")))
    return (round(float(np.mean(cov_dists)), 4), round(float(np.mean(corr_dists)), 4))


def _cv_site_predictability(X: np.ndarray, sites: np.ndarray, seed: int = 42) -> tuple[float, float]:
    """Cross-validated multi-class OVR AUC predicting site_id via Linear (LR) and Nonlinear (RF)."""
    y = pd.Categorical(sites).codes
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or int(counts.min()) < 3:
        return (0.5, 0.5)
    n_splits = min(5, int(counts.min()))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def _eval_model(model: Pipeline) -> float:
        probs = np.zeros((len(y), len(classes)), dtype=np.float64)
        for tr, te in skf.split(X, y):
            model.fit(X[tr], y[tr])
            probs[te] = model.predict_proba(X[te])
        if len(classes) == 2:
            return round(float(roc_auc_score(y, probs[:, 1])), 4)
        return round(float(roc_auc_score(y, probs, multi_class="ovr", average="macro")), 4)

    lr_auc = _eval_model(
        Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, C=1.0))])
    )
    rf_auc = _eval_model(
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("rf", RandomForestClassifier(n_estimators=120, max_depth=6, random_state=seed)),
            ]
        )
    )
    return (lr_auc, rf_auc)


def _cv_qc_preservation(X: np.ndarray, qc_labels: np.ndarray, seed: int = 42) -> dict[str, float]:
    """Cross-validated LogisticRegression predicting pass/borderline/fail from IQMs."""
    y = pd.Categorical(qc_labels).codes
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or int(counts.min()) < 2:
        return {"qc_macro_f1": 0.0, "qc_cohen_kappa": 0.0}
    n_splits = min(5, int(counts.min()))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.zeros_like(y)

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, C=1.0)),
        ]
    )
    for tr, te in skf.split(X, y):
        clf.fit(X[tr], y[tr])
        preds[te] = clf.predict(X[te])

    return {
        "qc_macro_f1": round(float(f1_score(y, preds, average="macro")), 4),
        "qc_cohen_kappa": round(float(cohen_kappa_score(y, preds)), 4),
    }


def _evaluate_stage(
    df: pd.DataFrame,
    feature_cols: list[str],
    site_col: str,
    qc_col: str,
    covar_cols: list[str] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    X = df[feature_cols].to_numpy(dtype=np.float64)
    sites = df[site_col].astype(str).to_numpy()
    qc = df[qc_col].astype(str).to_numpy() if qc_col in df.columns else np.zeros(len(df), dtype=str)
    covars = df[covar_cols].to_numpy(dtype=np.float64) if covar_cols else None

    X_std = StandardScaler().fit_transform(X)
    loc_scale = _site_location_scale_stats(X_std, sites)
    cov_fro, corr_fro = _covariance_and_correlation_divergence(X, sites, covars=covars)
    site_linear_auc, site_rf_auc = _cv_site_predictability(X, sites, seed=seed)
    qc_metrics = _cv_qc_preservation(X, qc, seed=seed) if qc_col in df.columns else {}

    return {
        **loc_scale,
        "covariance_frobenius_dist": cov_fro,
        "correlation_frobenius_dist": corr_fro,
        "site_linear_auc": site_linear_auc,
        "site_rf_auc": site_rf_auc,
        **qc_metrics,
    }


def _plot_pca_comparison(
    raw_df: pd.DataFrame,
    combat_df: pd.DataFrame,
    covbat_df: pd.DataFrame,
    feature_cols: list[str],
    site_col: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=120)
    stages = [("Raw IQMs", raw_df), ("After ComBat", combat_df), ("After CovBat", covbat_df)]
    unique_sites = sorted(raw_df[site_col].astype(str).unique().tolist())

    for ax, (title, frame) in zip(axes, stages, strict=True):
        X = StandardScaler().fit_transform(frame[feature_cols].to_numpy(dtype=np.float64))
        pcs = PCA(n_components=min(2, X.shape[1])).fit_transform(X)
        for s in unique_sites:
            mask = frame[site_col].astype(str).to_numpy() == s
            ax.scatter(pcs[mask, 0], pcs[mask, 1] if pcs.shape[1] > 1 else np.zeros(mask.sum()), label=s, alpha=0.75, s=26)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("PC1", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle("Multi-Site IQM Space: Raw vs. ComBat vs. CovBat", fontsize=11)
    fig.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


def run_harmonization_audit(
    manifest_df: pd.DataFrame,
    out_dir: str | Path,
    modality: str | None = None,
    site_col: str = "site_id",
    qc_col: str = "qc_label",
    protect_covars: list[str] | None = None,
    backend: str = "auto",
    seed: int = 42,
) -> HarmonizationAuditResult:
    """Run ComBat and CovBat on extracted IQMs and emit a quantitative audit report + PCA plot."""
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = manifest_df.copy()
    if modality is not None and "modality" in df.columns:
        df = df[df["modality"] == modality].reset_index(drop=True)
    elif "modality" in df.columns and df["modality"].nunique() > 1:
        # Default to most abundant structural modality (or largest modality group)
        counts = df["modality"].value_counts()
        modality = str(counts.index[0])
        df = df[df["modality"] == modality].reset_index(drop=True)

    feature_cols = _select_iqm_columns(df)
    if len(feature_cols) < 2:
        raise ValueError(f"Need at least 2 varying IQM columns for harmonization, found {feature_cols}")

    # Impute any sporadic NaNs cleanly with column medians
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    covar_cols = protect_covars
    if covar_cols is None:
        covar_cols = [c for c in ("defect_burden",) if c in df.columns and df[c].nunique() > 1]

    combat_df = harmonize_combat(
        df, feature_cols=feature_cols, site_col=site_col, covar_cols=covar_cols, backend=backend
    )
    covbat_df = harmonize_covbat(
        df, feature_cols=feature_cols, site_col=site_col, covar_cols=covar_cols, backend=backend
    )

    raw_stats = _evaluate_stage(df, feature_cols, site_col, qc_col, covar_cols=covar_cols, seed=seed)
    combat_stats = _evaluate_stage(combat_df, feature_cols, site_col, qc_col, covar_cols=covar_cols, seed=seed)
    covbat_stats = _evaluate_stage(covbat_df, feature_cols, site_col, qc_col, covar_cols=covar_cols, seed=seed)

    report: dict[str, Any] = {
        "modality": modality or "all",
        "n_scans": len(df),
        "sites": sorted(df[site_col].astype(str).unique().tolist()),
        "n_iqm_features": len(feature_cols),
        "iqm_features": feature_cols,
        "protected_covariates": covar_cols,
        "raw": raw_stats,
        "combat": combat_stats,
        "covbat": covbat_stats,
        "deltas": {
            "anova_f_reduction_combat_pct": round(
                100.0 * (1.0 - combat_stats["mean_anova_f"] / max(raw_stats["mean_anova_f"], 1e-6)),
                2,
            ),
            "cov_frobenius_reduction_combat_pct": round(
                100.0
                * (
                    1.0
                    - combat_stats["covariance_frobenius_dist"]
                    / max(raw_stats["covariance_frobenius_dist"], 1e-6)
                ),
                2,
            ),
            "cov_frobenius_reduction_covbat_pct": round(
                100.0
                * (
                    1.0
                    - covbat_stats["covariance_frobenius_dist"]
                    / max(raw_stats["covariance_frobenius_dist"], 1e-6)
                ),
                2,
            ),
            "site_linear_auc_drop_covbat": round(
                raw_stats["site_linear_auc"] - covbat_stats["site_linear_auc"], 4
            ),
            "site_rf_auc_drop_covbat": round(
                raw_stats["site_rf_auc"] - covbat_stats["site_rf_auc"], 4
            ),
        },
    }

    report_path = out_root / "harmonization_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    combat_df.to_parquet(out_root / "iqms_combat.parquet", index=False)
    covbat_df.to_parquet(out_root / "iqms_covbat.parquet", index=False)

    plot_path = out_root / "site_pca_comparison.png"
    _plot_pca_comparison(df, combat_df, covbat_df, feature_cols, site_col, plot_path)

    return HarmonizationAuditResult(
        report=report,
        report_path=report_path,
        plot_path=plot_path,
        combat_df=combat_df,
        covbat_df=covbat_df,
    )
