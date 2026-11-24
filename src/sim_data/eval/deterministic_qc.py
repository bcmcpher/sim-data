"""Cross-validated Logistic Regression baseline on NiPreps IQMs & blind-spot diagnostics.

Directly supports qc-studio's hybrid AI + deterministic direction by:
1. Fitting out-of-fold regularized Logistic Regression on NiPreps IQMs for pass/borderline/fail.
2. Computing multi-rater reliability metrics (Cohen's kappa, quadratic weighted kappa, macro-F1).
3. Reporting per-artifact detection recall to pinpoint deterministic IQM blind spots where
   visual views / VLM reasoning contribute the highest marginal lift.
4. Optionally enriching packet.json files with deterministic_priors (probabilities + outlier Z-scores).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LABELS_ORDER = ["pass", "borderline", "fail"]


@dataclass
class DeterministicQCReport:
    report: dict[str, Any]
    report_path: Path
    oof_predictions: pd.DataFrame


def evaluate_deterministic_qc(
    manifest_df: pd.DataFrame,
    out_dir: str | Path,
    modality: str | None = None,
    update_packets_root: str | Path | None = None,
    seed: int = 42,
) -> DeterministicQCReport:
    """Run stratified CV Logistic Regression on IQMs and report per-artifact recall & kappa."""
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = manifest_df.copy()
    if modality is not None and "modality" in df.columns:
        df = df[df["modality"] == modality].reset_index(drop=True)
    elif "modality" in df.columns and df["modality"].nunique() > 1:
        modality = str(df["modality"].value_counts().index[0])
        df = df[df["modality"] == modality].reset_index(drop=True)

    iqm_cols = [
        c
        for c in df.columns
        if c.startswith("iqm_")
        and pd.to_numeric(df[c], errors="coerce").isna().mean() < 0.1
        and float(pd.to_numeric(df[c], errors="coerce").std(skipna=True)) > 1e-8
    ]
    if not iqm_cols:
        raise ValueError("No valid numeric iqm_* columns found in manifest")

    X = np.column_stack(
        [
            pd.to_numeric(df[c], errors="coerce").fillna(pd.to_numeric(df[c], errors="coerce").median()).to_numpy(dtype=np.float64)
            for c in iqm_cols
        ]
    )

    present_labels = [lbl for lbl in LABELS_ORDER if lbl in set(df["qc_label"].astype(str))]
    label_to_idx = {lbl: i for i, lbl in enumerate(present_labels)}
    y = np.array([label_to_idx[str(lbl)] for lbl in df["qc_label"]], dtype=int)

    _, counts = np.unique(y, return_counts=True)
    n_splits = max(2, min(5, int(counts.min()))) if len(counts) >= 2 and int(counts.min()) >= 2 else 1

    probs = np.zeros((len(df), len(present_labels)), dtype=np.float64)
    preds = np.zeros(len(df), dtype=int)

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1500, C=1.0, class_weight="balanced")),
        ]
    )

    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, y):
            clf.fit(X[tr], y[tr])
            probs[te] = clf.predict_proba(X[te])
            preds[te] = clf.predict(X[te])
    else:
        clf.fit(X, y)
        probs = clf.predict_proba(X)
        preds = clf.predict(X)

    # Fit once on clean reference scans (or full cohort) for interpretable IQM Z-scores
    clean_mask = (df["qc_label"].astype(str) == "pass").to_numpy()
    ref_X = X[clean_mask] if int(clean_mask.sum()) >= 3 else X
    ref_mean = np.mean(ref_X, axis=0)
    ref_std = np.std(ref_X, axis=0) + 1e-6
    z_scores = (X - ref_mean[None, :]) / ref_std[None, :]

    pred_labels = [present_labels[i] for i in preds]
    macro_f1 = float(f1_score(y, preds, average="macro")) if len(present_labels) >= 2 else 1.0
    kappa = float(cohen_kappa_score(y, preds)) if len(present_labels) >= 2 else 1.0
    qwk = (
        float(cohen_kappa_score(y, preds, weights="quadratic"))
        if len(present_labels) >= 2
        else 1.0
    )

    # Per-artifact defect detection rate (flagged as borderline/fail when artifact present)
    artifact_recall: dict[str, dict[str, Any]] = {}
    if "primary_artifact" in df.columns:
        for art, grp in df.groupby("primary_artifact"):
            if str(art) == "none":
                continue
            idx = grp.index.to_numpy()
            true_bad = (grp["qc_label"].astype(str) != "pass").to_numpy()
            pred_bad = np.array([pred_labels[i] != "pass" for i in idx], dtype=bool)
            exact_acc = float(
                np.mean([pred_labels[i] == str(df.loc[i, "qc_label"]) for i in idx])
            )
            defect_det = (
                float(np.mean(pred_bad[true_bad])) if int(true_bad.sum()) > 0 else 1.0
            )
            artifact_recall[str(art)] = {
                "n_scans": len(idx),
                "exact_3class_accuracy": round(exact_acc, 4),
                "non_pass_detection_rate": round(defect_det, 4),
            }

    oof_df = pd.DataFrame(
        {
            "scan_id": df["scan_id"],
            "modality": df["modality"],
            "site_id": df["site_id"],
            "true_qc_label": df["qc_label"],
            "pred_qc_label": pred_labels,
        }
    )
    for idx_lbl, lbl in enumerate(present_labels):
        oof_df[f"prob_{lbl}"] = np.round(probs[:, idx_lbl], 4)

    # Optionally update packet.json files with deterministic_priors
    updated_packets = 0
    if update_packets_root is not None and "packet_path" in df.columns:
        root_p = Path(update_packets_root)
        for i, row in df.iterrows():
            rel = str(row.get("packet_path", ""))
            if not rel:
                continue
            pkt_file = root_p / rel
            if not pkt_file.exists():
                continue
            pkt = json.loads(pkt_file.read_text(encoding="utf-8"))
            abs_z = np.abs(z_scores[i])
            top_idx = np.argsort(abs_z)[::-1][:5]
            top_outliers = [
                {
                    "metric": iqm_cols[int(j)].removeprefix("iqm_"),
                    "z_score": round(float(z_scores[i, int(j)]), 3),
                }
                for j in top_idx
            ]
            prob_map = {lbl: 0.0 for lbl in LABELS_ORDER}
            for idx_lbl, lbl in enumerate(present_labels):
                prob_map[lbl] = round(float(probs[i, idx_lbl]), 4)
            pkt["deterministic_priors"] = {
                "model": "cv_logistic_regression",
                "probabilities": prob_map,
                "predicted_label": pred_labels[i],
                "top_outlier_iqms": top_outliers,
            }
            pkt_file.write_text(json.dumps(pkt, indent=2), encoding="utf-8")
            updated_packets += 1

    report: dict[str, Any] = {
        "modality": modality or "all",
        "n_scans": len(df),
        "n_iqm_features": len(iqm_cols),
        "cv_folds": n_splits,
        "macro_f1": round(macro_f1, 4),
        "cohen_kappa": round(kappa, 4),
        "quadratic_weighted_kappa": round(qwk, 4),
        "classification_report": classification_report(
            y, preds, target_names=present_labels, output_dict=True, zero_division=0
        ),
        "per_primary_artifact_performance": artifact_recall,
        "updated_packets": updated_packets,
    }

    report_path = out_root / "deterministic_qc_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    oof_df.to_csv(out_root / "oof_predictions.tsv", sep="\t", index=False)

    return DeterministicQCReport(report=report, report_path=report_path, oof_predictions=oof_df)
