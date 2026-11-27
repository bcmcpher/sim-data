"""Stage 1A Benchmark: Cross-validate ghibli structural IQMs against official MRIQC 24.0.2 on MR-ART.

Evaluates all 436 MR-ART T1w scans (148 subjects x acq-standard / headmotion1 / headmotion2):
1. Extracts blind `ghibli` structural IQMs (resumable/cached to `<out-dir>/ghibli_iqms.tsv`).
2. Joins with official MRIQC 24.0.2 `group_T1w.tsv` and human motion ratings `scores.tsv` (1/2/3).
3. Computes:
   - Cross-sectional Spearman rho, Pearson r, and scale ratio across all 22 shared IQMs.
   - Within-subject paired motion-delta correlation (motion scan minus subject's own acq-standard).
   - Human clinical Score 1 vs. 2 and Score 1 vs. 3 Cohen's d and ROC-AUC (MRIQC vs. ghibli),
     plus 5-fold subject-grouped Logistic Regression CV (Macro-F1, QWK, binary pass/fail AUC).
   - Empirical MR-ART Score 1 -> 2 -> 3 delta-IQM reference quantiles (`mrart_motion_reference.json`)
     for direct Stage 1B `motion_3d` calibration.
4. Generates publication-style PNG figures and a browser-ready `report.html` + `report.md`.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import time
from pathlib import Path
from typing import Any

# Prevent ITK/N4/OpenBLAS from spawning extra threads per worker on shared machines
os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ghibli.io.bids import BIDSScan, discover_bids_scans, load_bids_scan
from ghibli.io.orientation import canonicalize_to_ras
from ghibli.iqm.structural import extract_structural_iqms

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BIDS_DIR = REPO_ROOT.parent / "data" / "mrart" / "bids"
DEFAULT_MRIQC_TSV = (
    REPO_ROOT.parent / "data" / "nipoppy" / "derivatives" / "mriqc" / "24.0.2" / "output" / "group_T1w.tsv"
)
DEFAULT_SCORES_TSV = REPO_ROOT.parent / "data" / "mrart" / "derivatives" / "scores.tsv"
DEFAULT_OUT_DIR = REPO_ROOT / "benchmark_output" / "01_mriqc_alignment"

# Mapping: (canonical_label, mriqc_col, ghibli_col, is_core_qc, expected_motion_direction)
# expected_motion_direction: +1 = increases with motion/worse QC, -1 = decreases, 0 = neutral/summary
SHARED_IQM_MAP: list[tuple[str, str, str, bool, int]] = [
    ("cjv", "cjv", "iqm_cjv", True, +1),
    ("cnr", "cnr", "iqm_cnr", True, -1),
    ("efc", "efc", "iqm_efc", True, +1),
    ("fber", "fber", "iqm_fber", True, -1),
    ("fwhm_avg", "fwhm_avg", "iqm_fwhm_avg", True, +1),
    ("fwhm_x", "fwhm_x", "iqm_fwhm_x", False, +1),
    ("fwhm_y", "fwhm_y", "iqm_fwhm_y", False, +1),
    ("fwhm_z", "fwhm_z", "iqm_fwhm_z", False, +1),
    ("inu_med", "inu_med", "iqm_inu_med", False, 0),
    ("inu_range", "inu_range", "iqm_inu_range", True, +1),
    ("qi_1", "qi_1", "iqm_qi1", True, +1),
    ("qi_2", "qi_2", "iqm_qi2", True, +1),
    ("snr_total", "snr_total", "iqm_snr_total", True, -1),
    ("snr_wm", "snr_wm", "iqm_snr_wm", True, -1),
    ("snr_gm", "snr_gm", "iqm_snr_gm", True, -1),
    ("snrd_total", "snrd_total", "iqm_snrd_total", True, -1),
    ("wm2max", "wm2max", "iqm_wm2max", True, -1),
    ("summary_bg_mean", "summary_bg_mean", "iqm_summary_bg_mean", False, +1),
    ("summary_bg_stdv", "summary_bg_stdv", "iqm_summary_bg_stdv", False, +1),
    ("summary_bg_p95", "summary_bg_p95", "iqm_summary_bg_p95", False, +1),
    ("summary_wm_mean", "summary_wm_mean", "iqm_summary_wm_mean", False, 0),
    ("summary_gm_mean", "summary_gm_mean", "iqm_summary_gm_mean", False, 0),
]

CORE_SCATTER_METRICS = [
    "cjv",
    "cnr",
    "efc",
    "fwhm_avg",
    "inu_range",
    "snrd_total",
    "snr_wm",
    "snr_gm",
    "snr_total",
    "wm2max",
    "qi_1",
    "qi_2",
]


def _extract_one_scan(scan: BIDSScan) -> dict[str, Any]:
    """Compute blind ghibli structural IQMs for one BIDS T1w scan (strictly single-threaded per worker)."""
    import SimpleITK as sitk

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    loaded = load_bids_scan(scan)
    data, _affine, zooms = canonicalize_to_ras(loaded.img)
    iqms = extract_structural_iqms(data, zooms=zooms)
    row: dict[str, Any] = {
        "bids_name": scan.base_id,
        "subject": scan.subject,
        "acq": scan.entities.get("acq", ""),
        "nifti_path": str(scan.nifti_path),
    }
    for k, v in iqms.items():
        row[f"iqm_{k}"] = float(v)
    return row


def _extract_or_resume_ghibli_iqms(
    bids_dir: Path,
    target_bids_names: set[str],
    cache_tsv: Path,
    workers: int,
) -> pd.DataFrame:
    """Extract ghibli IQMs for target scans, reusing any rows already cached in `cache_tsv`."""
    cached_df = pd.DataFrame()
    done_names: set[str] = set()
    if cache_tsv.exists():
        cached_df = pd.read_csv(cache_tsv, sep="\t")
        if "bids_name" in cached_df.columns:
            done_names = set(cached_df["bids_name"].astype(str))

    all_scans = discover_bids_scans(bids_dir, modalities=["T1w"])
    todo = [s for s in all_scans if s.base_id in target_bids_names and s.base_id not in done_names]

    if todo:
        print(
            f"[Stage 1A] Extracting blind ghibli structural IQMs for {len(todo)} scan(s) "
            f"(cached={len(done_names & target_bids_names)}, workers={workers})..."
        )
        t0 = time.perf_counter()
        if workers <= 1 or len(todo) == 1:
            new_rows = [_extract_one_scan(s) for s in todo]
        else:
            new_rows = Parallel(n_jobs=workers, prefer="processes")(
                delayed(_extract_one_scan)(s) for s in todo
            )
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([cached_df, new_df], ignore_index=True) if not cached_df.empty else new_df
        combined = combined.drop_duplicates(subset=["bids_name"], keep="last").sort_values("bids_name")
        cache_tsv.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(cache_tsv, sep="\t", index=False)
        elapsed = time.perf_counter() - t0
        print(
            f"[Stage 1A] Extraction complete ({len(todo)} scans in {elapsed:.1f}s, "
            f"{elapsed / max(1, len(todo)):.2f}s/scan effective) -> {cache_tsv}"
        )
    else:
        combined = cached_df
        print(f"[Stage 1A] Reused {len(done_names & target_bids_names)} cached scans from {cache_tsv}")

    return combined[combined["bids_name"].isin(target_bids_names)].copy()


def _valid_pair_mask(m_vals: np.ndarray, g_vals: np.ndarray, metric_name: str) -> np.ndarray:
    mask = np.isfinite(m_vals) & np.isfinite(g_vals)
    if metric_name == "fber":
        # MRIQC emits -1.0 when median background air is 0 (uninformative sentinel)
        mask &= (m_vals > 0.0) & (g_vals > 0.0)
    return mask


def _safe_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size < 5 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return (float("nan"), float("nan"))
    rho, _ = stats.spearmanr(x, y)
    r, _ = stats.pearsonr(x, y)
    return (float(rho), float(r))


def _binary_separation(vals: np.ndarray, scores: np.ndarray, pos_score: int) -> tuple[float, float]:
    """Compute Cohen's d (pos - score1) and direction-agnostic ROC-AUC (max(auc, 1-auc)) for score 1 vs pos_score."""
    s1 = vals[(scores == 1) & np.isfinite(vals)]
    sp = vals[(scores == pos_score) & np.isfinite(vals)]
    if s1.size < 3 or sp.size < 3:
        return (float("nan"), float("nan"))
    pooled_sd = float(np.sqrt(0.5 * (np.var(s1, ddof=1) + np.var(sp, ddof=1))))
    cohens_d = float((np.mean(sp) - np.mean(s1)) / pooled_sd) if pooled_sd > 1e-12 else 0.0
    y_true = np.concatenate([np.zeros(s1.size, dtype=int), np.ones(sp.size, dtype=int)])
    y_score = np.concatenate([s1, sp])
    if float(np.std(y_score)) <= 1e-12:
        return (cohens_d, 0.5)
    raw_auc = float(roc_auc_score(y_true, y_score))
    return (cohens_d, max(raw_auc, 1.0 - raw_auc))


def _compute_alignment_tables(
    merged: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compute cross-sectional alignment, within-subject delta alignment, and MR-ART motion quantiles."""
    # Build within-subject paired deltas against each subject's own acq-standard (score == 1)
    std_df = (
        merged[merged["acq"] == "standard"]
        .drop_duplicates(subset=["subject"])
        .set_index("subject")
    )
    motion_df = merged[merged["acq"] != "standard"].copy()
    delta_rows: list[dict[str, Any]] = []
    for _, row in motion_df.iterrows():
        sub = str(row["subject"])
        if sub not in std_df.index:
            continue
        base = std_df.loc[sub]
        d_entry: dict[str, Any] = {
            "bids_name": str(row["bids_name"]),
            "subject": sub,
            "acq": str(row["acq"]),
            "score": int(row["score"]),
            "base_score": int(base["score"]),
        }
        for name, m_col, g_col, _is_core, _dir in SHARED_IQM_MAP:
            mv = float(row[m_col]) if m_col in row else float("nan")
            mb = float(base[m_col]) if m_col in base else float("nan")
            gv = float(row[g_col]) if g_col in row else float("nan")
            gb = float(base[g_col]) if g_col in base else float("nan")
            if name == "fber" and (mv <= 0 or mb <= 0 or gv <= 0 or gb <= 0):
                d_entry[f"dm_{name}"] = float("nan")
                d_entry[f"dg_{name}"] = float("nan")
            else:
                d_entry[f"dm_{name}"] = mv - mb
                d_entry[f"dg_{name}"] = gv - gb
                d_entry[f"rel_m_{name}"] = (mv - mb) / max(abs(mb), 1e-6)
                d_entry[f"rel_g_{name}"] = (gv - gb) / max(abs(gb), 1e-6)
        delta_rows.append(d_entry)
    deltas_df = pd.DataFrame(delta_rows)

    summary_rows: list[dict[str, Any]] = []
    scores_all = merged["score"].to_numpy(dtype=int)

    for name, m_col, g_col, is_core, _exp_dir in SHARED_IQM_MAP:
        if m_col not in merged.columns or g_col not in merged.columns:
            continue
        m_vals = merged[m_col].to_numpy(dtype=np.float64)
        g_vals = merged[g_col].to_numpy(dtype=np.float64)
        valid = _valid_pair_mask(m_vals, g_vals, name)

        rho_all, r_all = _safe_corr(m_vals[valid], g_vals[valid])
        # Compute median ratio on positive pairs
        pos_m = valid & (np.abs(m_vals) > 1e-6)
        scale_ratio = float(np.median(g_vals[pos_m] / m_vals[pos_m])) if np.any(pos_m) else float("nan")

        # Within-subject delta correlations
        if not deltas_df.empty and f"dm_{name}" in deltas_df.columns:
            dm = deltas_df[f"dm_{name}"].to_numpy(dtype=np.float64)
            dg = deltas_df[f"dg_{name}"].to_numpy(dtype=np.float64)
            v_d = np.isfinite(dm) & np.isfinite(dg)
            rho_delta, r_delta = _safe_corr(dm[v_d], dg[v_d])
            n_delta = int(v_d.sum())
        else:
            rho_delta, r_delta, n_delta = float("nan"), float("nan"), 0

        # Score 1 vs 2 and Score 1 vs 3 separability (masking FBER sentinels)
        m_clean = np.where(valid, m_vals, np.nan)
        g_clean = np.where(valid, g_vals, np.nan)
        d_m_12, auc_m_12 = _binary_separation(m_clean, scores_all, pos_score=2)
        d_g_12, auc_g_12 = _binary_separation(g_clean, scores_all, pos_score=2)
        d_m_13, auc_m_13 = _binary_separation(m_clean, scores_all, pos_score=3)
        d_g_13, auc_g_13 = _binary_separation(g_clean, scores_all, pos_score=3)

        # Tier classification based on cross-sectional & delta Spearman alignment
        best_rho = max(
            rho_all if np.isfinite(rho_all) else -1.0,
            rho_delta if np.isfinite(rho_delta) else -1.0,
        )
        if np.isfinite(rho_all) and rho_all >= 0.75:
            tier = "STRONG"
        elif best_rho >= 0.45:
            tier = "MODERATE"
        else:
            tier = "GAP"

        summary_rows.append(
            {
                "metric": name,
                "mriqc_col": m_col,
                "ghibli_col": g_col,
                "is_core": is_core,
                "n_valid": int(valid.sum()),
                "spearman_rho": round(rho_all, 4) if np.isfinite(rho_all) else float("nan"),
                "pearson_r": round(r_all, 4) if np.isfinite(r_all) else float("nan"),
                "delta_spearman_rho": round(rho_delta, 4) if np.isfinite(rho_delta) else float("nan"),
                "delta_pearson_r": round(r_delta, 4) if np.isfinite(r_delta) else float("nan"),
                "n_deltas": n_delta,
                "median_ratio_ghibli_over_mriqc": round(scale_ratio, 4) if np.isfinite(scale_ratio) else float("nan"),
                "mriqc_d_1vs2": round(d_m_12, 3) if np.isfinite(d_m_12) else float("nan"),
                "ghibli_d_1vs2": round(d_g_12, 3) if np.isfinite(d_g_12) else float("nan"),
                "mriqc_auc_1vs2": round(auc_m_12, 4) if np.isfinite(auc_m_12) else float("nan"),
                "ghibli_auc_1vs2": round(auc_g_12, 4) if np.isfinite(auc_g_12) else float("nan"),
                "mriqc_d_1vs3": round(d_m_13, 3) if np.isfinite(d_m_13) else float("nan"),
                "ghibli_d_1vs3": round(d_g_13, 3) if np.isfinite(d_g_13) else float("nan"),
                "mriqc_auc_1vs3": round(auc_m_13, 4) if np.isfinite(auc_m_13) else float("nan"),
                "ghibli_auc_1vs3": round(auc_g_13, 4) if np.isfinite(auc_g_13) else float("nan"),
                "alignment_tier": tier,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    # Empirical MR-ART motion-delta reference quantiles for Stage 1B motion_3d calibration
    motion_ref: dict[str, Any] = {
        "n_scans_total": int(len(merged)),
        "n_subjects": int(merged["subject"].nunique()),
        "score_counts": {str(k): int(v) for k, v in merged["score"].value_counts().sort_index().items()},
        "acq_counts": {str(k): int(v) for k, v in merged["acq"].value_counts().sort_index().items()},
        "delta_quantiles_by_score": {},
    }
    if not deltas_df.empty:
        for sc in (2, 3):
            sub_d = deltas_df[(deltas_df["score"] == sc) & (deltas_df["base_score"] == 1)]
            sc_block: dict[str, Any] = {"n_pairs": int(len(sub_d)), "metrics": {}}
            for name in ("cjv", "cnr", "efc", "fwhm_avg", "inu_range", "qi_1", "qi_2", "snrd_total", "snr_wm", "wm2max"):
                if f"dm_{name}" not in sub_d.columns:
                    continue
                dm = sub_d[f"dm_{name}"].dropna().to_numpy(dtype=np.float64)
                dg = sub_d[f"dg_{name}"].dropna().to_numpy(dtype=np.float64)
                rm = sub_d[f"rel_m_{name}"].dropna().to_numpy(dtype=np.float64)
                rg = sub_d[f"rel_g_{name}"].dropna().to_numpy(dtype=np.float64)

                def _qstats(arr: np.ndarray) -> dict[str, float]:
                    if arr.size == 0:
                        return {}
                    return {
                        "p25": round(float(np.percentile(arr, 25.0)), 5),
                        "median": round(float(np.median(arr)), 5),
                        "p75": round(float(np.percentile(arr, 75.0)), 5),
                        "mean": round(float(np.mean(arr)), 5),
                        "std": round(float(np.std(arr)), 5),
                    }

                sc_block["metrics"][name] = {
                    "mriqc_abs_delta": _qstats(dm),
                    "mriqc_rel_delta": _qstats(rm),
                    "ghibli_abs_delta": _qstats(dg),
                    "ghibli_rel_delta": _qstats(rg),
                }
            motion_ref["delta_quantiles_by_score"][f"score_{sc}_vs_1"] = sc_block

    return summary_df, deltas_df, motion_ref


def _evaluate_multivariate_qc(merged: pd.DataFrame) -> dict[str, dict[str, float]]:
    """5-fold subject-grouped LogisticRegression CV predicting MR-ART clinical score (1/2/3)."""
    y = merged["score"].to_numpy(dtype=int)
    groups = merged["subject"].astype(str).to_numpy()
    n_groups = int(len(np.unique(groups)))
    n_splits = min(5, n_groups)
    if n_splits < 2 or len(np.unique(y)) < 2:
        return {}

    mriqc_cols = [m for _n, m, _g, _c, _d in SHARED_IQM_MAP if m in merged.columns and m != "fber"]
    ghibli_cols = [g for _n, _m, g, _c, _d in SHARED_IQM_MAP if g in merged.columns and g != "iqm_fber"]
    ghibli_all_cols = [
        c for c in merged.columns if c.startswith("iqm_") and c != "iqm_fber"
    ]

    results: dict[str, dict[str, float]] = {}
    gkf = GroupKFold(n_splits=n_splits)

    for label, cols in [
        ("MRIQC_24.0.2_shared", mriqc_cols),
        ("ghibli_shared", ghibli_cols),
        ("ghibli_all_structural", ghibli_all_cols),
    ]:
        x_raw = merged[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        col_meds = np.nanmedian(x_raw, axis=0)
        col_meds = np.where(np.isfinite(col_meds), col_meds, 0.0)
        inds = np.where(~np.isfinite(x_raw))
        x_raw[inds] = np.take(col_meds, inds[1])

        preds = np.zeros_like(y)
        prob_bad = np.zeros(len(y), dtype=np.float64)

        for tr_idx, te_idx in gkf.split(x_raw, y, groups):
            clf = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")),
                ]
            )
            clf.fit(x_raw[tr_idx], y[tr_idx])
            preds[te_idx] = clf.predict(x_raw[te_idx])
            proba = clf.predict_proba(x_raw[te_idx])
            classes = list(clf.classes_)
            # P(motion artifact: score >= 2)
            bad_cols = [i for i, c in enumerate(classes) if c >= 2]
            prob_bad[te_idx] = proba[:, bad_cols].sum(axis=1)

        y_bin_any = (y >= 2).astype(int)
        y_bin_sev = (y == 3).astype(int)
        results[label] = {
            "n_features": float(len(cols)),
            "macro_f1": round(float(f1_score(y, preds, average="macro")), 4),
            "qwk": round(float(cohen_kappa_score(y, preds, weights="quadratic")), 4),
            "auc_score1_vs_23": round(float(roc_auc_score(y_bin_any, prob_bad)), 4),
            "auc_score3_vs_12": round(float(roc_auc_score(y_bin_sev, prob_bad)), 4),
        }

    return results


def _render_scatter_grid(
    merged: pd.DataFrame,
    summary_df: pd.DataFrame,
    out_path: Path,
) -> Path:
    """3x4 cross-sectional scatter grid of core IQMs (ghibli vs. MRIQC 24.0.2), colored by MR-ART score."""
    fig, axes = plt.subplots(3, 4, figsize=(15.5, 10.5))
    score_colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#d62728"}
    score_labels = {1: "Score 1 (No motion)", 2: "Score 2 (Moderate)", 3: "Score 3 (Severe)"}
    meta_by_metric = {str(r["metric"]): r for _, r in summary_df.iterrows()}

    for ax, metric in zip(axes.ravel(), CORE_SCATTER_METRICS, strict=True):
        row_m = meta_by_metric.get(metric)
        if row_m is None:
            ax.axis("off")
            continue
        m_col = str(row_m["mriqc_col"])
        g_col = str(row_m["ghibli_col"])
        for sc in (1, 2, 3):
            sub = merged[merged["score"] == sc]
            mv = sub[m_col].to_numpy(dtype=np.float64)
            gv = sub[g_col].to_numpy(dtype=np.float64)
            ok = _valid_pair_mask(mv, gv, metric)
            ax.scatter(
                mv[ok],
                gv[ok],
                s=16,
                alpha=0.62,
                color=score_colors[sc],
                edgecolors="none",
                label=score_labels[sc],
            )
        rho = row_m["spearman_rho"]
        r_val = row_m["pearson_r"]
        tier = row_m["alignment_tier"]
        ax.set_title(f"{metric} [{tier}]\nρ={rho:+.3f}, r={r_val:+.3f}", fontsize=9.2, fontweight="bold")
        ax.set_xlabel("MRIQC 24.0.2", fontsize=8)
        ax.set_ylabel("ghibli", fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle(
        f"Stage 1A: Cross-Sectional IQM Alignment — ghibli vs. MRIQC 24.0.2 (MR-ART n={len(merged)} scans)",
        fontsize=12.5,
        fontweight="bold",
        y=1.015,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _render_delta_grid(
    deltas_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    out_path: Path,
) -> Path:
    """2x4 scatter grid of within-subject motion deltas (motion - acq-standard): Δghibli vs. ΔMRIQC."""
    delta_metrics = ["cjv", "cnr", "efc", "fwhm_avg", "inu_range", "snrd_total", "snr_wm", "wm2max"]
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.2))
    score_colors = {2: "#ff7f0e", 3: "#d62728"}
    score_labels = {2: "Δ (Score 2 − Std)", 3: "Δ (Score 3 − Std)"}
    meta_by_metric = {str(r["metric"]): r for _, r in summary_df.iterrows()}

    for ax, metric in zip(axes.ravel(), delta_metrics, strict=True):
        row_m = meta_by_metric.get(metric)
        if row_m is None or f"dm_{metric}" not in deltas_df.columns:
            ax.axis("off")
            continue
        for sc in (2, 3):
            sub = deltas_df[deltas_df["score"] == sc]
            dm = sub[f"dm_{metric}"].to_numpy(dtype=np.float64)
            dg = sub[f"dg_{metric}"].to_numpy(dtype=np.float64)
            ok = np.isfinite(dm) & np.isfinite(dg)
            ax.scatter(
                dm[ok],
                dg[ok],
                s=18,
                alpha=0.65,
                color=score_colors[sc],
                edgecolors="none",
                label=score_labels[sc],
            )
        ax.axhline(0.0, color="#666666", lw=0.7, ls=":")
        ax.axvline(0.0, color="#666666", lw=0.7, ls=":")
        drho = row_m["delta_spearman_rho"]
        dr = row_m["delta_pearson_r"]
        ax.set_title(f"Δ{metric}\nΔρ={drho:+.3f}, Δr={dr:+.3f}", fontsize=9.2, fontweight="bold")
        ax.set_xlabel("Δ MRIQC 24.0.2", fontsize=8)
        ax.set_ylabel("Δ ghibli", fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle(
        f"Stage 1A: Within-Subject Motion Delta Alignment (n={len(deltas_df)} paired MR-ART scans)",
        fontsize=12.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _render_auc_comparison(
    summary_df: pd.DataFrame,
    out_path: Path,
) -> Path:
    """Side-by-side ROC-AUC comparison (MRIQC 24.0.2 vs. ghibli) for Score 1 vs. 2 and Score 1 vs. 3."""
    core = summary_df[summary_df["is_core"]].copy()
    core = core.sort_values("mriqc_auc_1vs3", ascending=True)
    y_pos = np.arange(len(core))
    h = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharey=True)
    for ax, (title, m_col, g_col) in zip(
        axes,
        [
            ("Score 1 (No Motion) vs. Score 2 (Moderate)", "mriqc_auc_1vs2", "ghibli_auc_1vs2"),
            ("Score 1 (No Motion) vs. Score 3 (Severe)", "mriqc_auc_1vs3", "ghibli_auc_1vs3"),
        ],
        strict=True,
    ):
        ax.barh(y_pos + h / 2, core[m_col].to_numpy(dtype=float), height=h, color="#1f77b4", label="MRIQC 24.0.2")
        ax.barh(y_pos - h / 2, core[g_col].to_numpy(dtype=float), height=h, color="#e67e22", label="ghibli")
        ax.axvline(0.5, color="#888888", ls="--", lw=0.8)
        ax.set_xlim(0.45, 1.02)
        ax.set_xlabel("ROC-AUC (higher = better motion separation)", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.25)

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(core["metric"].tolist(), fontsize=9)
    axes[1].legend(loc="lower right", fontsize=9)
    fig.suptitle(
        "Stage 1A: Univariate Human Motion Rating Discrimination — MRIQC 24.0.2 vs. ghibli",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _write_reports(
    out_root: Path,
    summary_df: pd.DataFrame,
    cv_results: dict[str, dict[str, float]],
    motion_ref: dict[str, Any],
    figures: dict[str, str],
) -> tuple[Path, Path]:
    """Write Markdown and browser-ready HTML alignment reports."""
    strong = summary_df[summary_df["alignment_tier"] == "STRONG"]["metric"].tolist()
    moderate = summary_df[summary_df["alignment_tier"] == "MODERATE"]["metric"].tolist()
    gaps = summary_df[summary_df["alignment_tier"] == "GAP"]["metric"].tolist()

    # Markdown report
    md_lines: list[str] = [
        "# Stage 1A Benchmark: `ghibli` vs. Official MRIQC 24.0.2 on MR-ART",
        "",
        f"- **Dataset:** MR-ART (`n={motion_ref['n_scans_total']}` scans across `{motion_ref['n_subjects']}` subjects)",
        f"- **Score distribution:** `{motion_ref['score_counts']}`",
        f"- **Alignment summary (22 shared IQMs):** **{len(strong)} STRONG** (`|ρ| ≥ 0.75`), "
        f"**{len(moderate)} MODERATE** (`0.45 ≤ |ρ| < 0.75`), **{len(gaps)} GAP** (`|ρ| < 0.45`)",
        f"- **Flagged alignment gaps:** `{gaps}`" if gaps else "- **Flagged alignment gaps:** None",
        "",
        "## Per-Metric Alignment & Human Score Discrimination",
        "",
        "| Metric | Tier | Spearman ρ | Pearson r | Δ-Motion ρ | Scale (ghibli/MRIQC) | MRIQC AUC (1v2 / 1v3) | ghibli AUC (1v2 / 1v3) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in summary_df.iterrows():
        md_lines.append(
            f"| `{r['metric']}` | `{r['alignment_tier']}` | {r['spearman_rho']:+.3f} | {r['pearson_r']:+.3f} | "
            f"{r['delta_spearman_rho']:+.3f} | {r['median_ratio_ghibli_over_mriqc']:.3f} | "
            f"{r['mriqc_auc_1vs2']:.3f} / {r['mriqc_auc_1vs3']:.3f} | "
            f"{r['ghibli_auc_1vs2']:.3f} / {r['ghibli_auc_1vs3']:.3f} |"
        )

    if cv_results:
        md_lines.extend(
            [
                "",
                "## Multivariate 5-Fold Subject-Grouped QC Prediction (MR-ART Score 1/2/3)",
                "",
                "| Feature Set | Features | Macro-F1 | QWK | AUC (1 vs 2+3) | AUC (3 vs 1+2) |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for lbl, m in cv_results.items():
            md_lines.append(
                f"| `{lbl}` | {int(m['n_features'])} | {m['macro_f1']:.4f} | {m['qwk']:.4f} | "
                f"{m['auc_score1_vs_23']:.4f} | {m['auc_score3_vs_12']:.4f} |"
            )

    for title, key in [
        ("Cross-Sectional IQM Scatter Alignment", "scatter"),
        ("Within-Subject Motion Delta Alignment", "delta"),
        ("Human Motion Rating ROC-AUC Comparison", "auc"),
    ]:
        if key in figures:
            md_lines.extend(["", f"## {title}", "", f"![{title}]({figures[key]})"])

    md_path = out_root / "report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # HTML report
    def _badge(tier: str) -> str:
        cls = {"STRONG": "pass", "MODERATE": "warn", "GAP": "fail"}.get(tier, "info")
        return f'<span class="badge badge-{cls}">{html.escape(tier)}</span>'

    rows_html = []
    for _, r in summary_df.iterrows():
        rows_html.append(
            "<tr>"
            f"<td><code>{html.escape(str(r['metric']))}</code></td>"
            f"<td>{_badge(str(r['alignment_tier']))}</td>"
            f"<td>{r['spearman_rho']:+.3f}</td>"
            f"<td>{r['pearson_r']:+.3f}</td>"
            f"<td>{r['delta_spearman_rho']:+.3f}</td>"
            f"<td>{r['median_ratio_ghibli_over_mriqc']:.3f}</td>"
            f"<td>{r['mriqc_auc_1vs2']:.3f} / {r['mriqc_auc_1vs3']:.3f}</td>"
            f"<td>{r['ghibli_auc_1vs2']:.3f} / {r['ghibli_auc_1vs3']:.3f}</td>"
            "</tr>"
        )

    cv_html = ""
    if cv_results:
        cv_rows = "".join(
            f"<tr><td><code>{html.escape(lbl)}</code></td><td>{int(m['n_features'])}</td>"
            f"<td>{m['macro_f1']:.4f}</td><td>{m['qwk']:.4f}</td>"
            f"<td>{m['auc_score1_vs_23']:.4f}</td><td>{m['auc_score3_vs_12']:.4f}</td></tr>"
            for lbl, m in cv_results.items()
        )
        cv_html = (
            "<h2>Multivariate 5-Fold Subject-Grouped QC Prediction (Score 1/2/3)</h2>"
            "<table><thead><tr><th>Feature Set</th><th>Features</th><th>Macro-F1</th>"
            "<th>QWK</th><th>AUC (1 vs 2+3)</th><th>AUC (3 vs 1+2)</th></tr></thead>"
            f"<tbody>{cv_rows}</tbody></table>"
        )

    figs_html = "".join(
        f'<h2>{html.escape(title)}</h2><figure><a href="{html.escape(figures[key])}" target="_blank">'
        f'<img src="{html.escape(figures[key])}" alt="{html.escape(title)}" /></a></figure>'
        for title, key in [
            ("Cross-Sectional IQM Scatter Alignment", "scatter"),
            ("Within-Subject Motion Delta Alignment", "delta"),
            ("Human Motion Rating ROC-AUC Comparison", "auc"),
        ]
        if key in figures
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8" />
<title>GHIBLI Stage 1A: MRIQC 24.0.2 Alignment Benchmark</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 1260px; margin: 0 auto; padding: 24px; color: #1f2328; }}
h1, h2 {{ border-bottom: 1px solid #d1d9e0; padding-bottom: 6px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; margin: 12px 0 22px; }}
th, td {{ border: 1px solid #d1d9e0; padding: 6px 10px; text-align: left; }}
th {{ background: #f6f8fa; }}
code {{ background: #f6f8fa; padding: 2px 5px; border-radius: 4px; }}
.badge {{ padding: 2px 8px; border-radius: 999px; font-size: 0.75rem; font-weight: 700; }}
.badge-pass {{ background: #dafbe1; color: #116329; }}
.badge-warn {{ background: #fff8c5; color: #7d4e00; }}
.badge-fail {{ background: #ffebe9; color: #a40e26; }}
figure {{ margin: 14px 0 28px; padding: 12px; background: #f6f8fa; border: 1px solid #d1d9e0; border-radius: 8px; text-align: center; }}
figure img {{ max-width: 100%; height: auto; }}
</style></head><body>
<h1><code>ghibli</code> Stage 1A: MRIQC 24.0.2 Alignment Benchmark (MR-ART)</h1>
<p><strong>Scans:</strong> {motion_ref['n_scans_total']} ({motion_ref['n_subjects']} subjects) &nbsp;|&nbsp;
<strong>Scores:</strong> <code>{html.escape(str(motion_ref['score_counts']))}</code> &nbsp;|&nbsp;
<strong>Tiers:</strong> {_badge('STRONG')} {len(strong)} &nbsp; {_badge('MODERATE')} {len(moderate)} &nbsp; {_badge('GAP')} {len(gaps)}</p>
<h2>Per-Metric Alignment &amp; Human Score Discrimination</h2>
<table><thead><tr><th>Metric</th><th>Tier</th><th>Spearman &rho;</th><th>Pearson r</th>
<th>&Delta;-Motion &rho;</th><th>Scale (ghibli/MRIQC)</th><th>MRIQC AUC (1v2 / 1v3)</th><th>ghibli AUC (1v2 / 1v3)</th></tr></thead>
<tbody>{"".join(rows_html)}</tbody></table>
{cv_html}
{figs_html}
</body></html>
"""
    html_path = out_root / "report.html"
    html_path.write_text(html_doc, encoding="utf-8")
    return md_path, html_path


def run_benchmark(
    bids_dir: Path = DEFAULT_BIDS_DIR,
    mriqc_tsv: Path = DEFAULT_MRIQC_TSV,
    scores_tsv: Path = DEFAULT_SCORES_TSV,
    out_dir: Path = DEFAULT_OUT_DIR,
    workers: int = 2,
    limit_subjects: int | None = None,
) -> dict[str, Any]:
    """Run Stage 1A MR-ART alignment benchmark between ghibli extract-iqms and MRIQC 24.0.2."""
    out_root = Path(out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    mriqc_df = pd.read_csv(mriqc_tsv, sep="\t")
    scores_df = pd.read_csv(scores_tsv, sep="\t")
    ref_df = mriqc_df.merge(scores_df, on="bids_name", how="inner")
    ref_df["subject"] = ref_df["bids_name"].str.extract(r"sub-([A-Za-z0-9]+)")
    ref_df["acq"] = ref_df["bids_name"].str.extract(r"acq-([A-Za-z0-9]+)").fillna("")

    if limit_subjects is not None and limit_subjects > 0:
        keep_subs = sorted(ref_df["subject"].dropna().unique().tolist())[:limit_subjects]
        ref_df = ref_df[ref_df["subject"].isin(keep_subs)].copy()

    target_names = set(ref_df["bids_name"].astype(str))
    ghibli_df = _extract_or_resume_ghibli_iqms(
        bids_dir=bids_dir,
        target_bids_names=target_names,
        cache_tsv=out_root / "ghibli_iqms.tsv",
        workers=workers,
    )

    merged = ref_df.merge(
        ghibli_df.drop(columns=["subject", "acq"], errors="ignore"),
        on="bids_name",
        how="inner",
    )
    if merged.empty:
        raise RuntimeError("No overlapping scans between MRIQC TSV, scores TSV, and ghibli IQMs.")

    merged.to_csv(out_root / "merged_alignment.tsv", sep="\t", index=False)

    summary_df, deltas_df, motion_ref = _compute_alignment_tables(merged)
    summary_df.to_csv(out_root / "metric_alignment_summary.tsv", sep="\t", index=False)
    if not deltas_df.empty:
        deltas_df.to_csv(out_root / "within_subject_deltas.tsv", sep="\t", index=False)

    cv_results = _evaluate_multivariate_qc(merged)
    motion_ref["multivariate_cv"] = cv_results
    (out_root / "mrart_motion_reference.json").write_text(
        json.dumps(motion_ref, indent=2), encoding="utf-8"
    )

    fig_dir = out_root / "figures"
    figures: dict[str, str] = {}
    figures["scatter"] = str(
        _render_scatter_grid(merged, summary_df, fig_dir / "01a_scatter_alignment.png").relative_to(out_root)
    )
    if not deltas_df.empty:
        figures["delta"] = str(
            _render_delta_grid(deltas_df, summary_df, fig_dir / "01b_delta_alignment.png").relative_to(out_root)
        )
    figures["auc"] = str(
        _render_auc_comparison(summary_df, fig_dir / "01c_score_discrimination.png").relative_to(out_root)
    )

    md_path, html_path = _write_reports(out_root, summary_df, cv_results, motion_ref, figures)
    print(f"[Stage 1A] Wrote report -> {html_path} (and {md_path})")
    return {
        "n_scans": int(len(merged)),
        "report_html": str(html_path),
        "report_md": str(md_path),
        "summary_tsv": str(out_root / "metric_alignment_summary.tsv"),
        "motion_reference_json": str(out_root / "mrart_motion_reference.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1A: Benchmark ghibli structural IQMs against official MRIQC 24.0.2 on MR-ART."
    )
    parser.add_argument("--bids-dir", type=Path, default=DEFAULT_BIDS_DIR)
    parser.add_argument("--mriqc-tsv", type=Path, default=DEFAULT_MRIQC_TSV)
    parser.add_argument("--scores-tsv", type=Path, default=DEFAULT_SCORES_TSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="CPU worker processes (default: 2 to stay polite on shared machines).",
    )
    parser.add_argument(
        "--limit-subjects",
        type=int,
        default=None,
        help="Optional cap on number of subjects (e.g., 20 for a 60-scan preview).",
    )
    args = parser.parse_args()
    run_benchmark(
        bids_dir=args.bids_dir,
        mriqc_tsv=args.mriqc_tsv,
        scores_tsv=args.scores_tsv,
        out_dir=args.out_dir,
        workers=args.workers,
        limit_subjects=args.limit_subjects,
    )


if __name__ == "__main__":
    main()
