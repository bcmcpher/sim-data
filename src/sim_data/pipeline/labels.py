"""Ground-truth QC target scoring, full-reference fidelity metrics, and manifest writers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim

from sim_data.artifacts.base import ArtifactRecord
from sim_data.artifacts.registry import ARTIFACT_REGISTRY
from sim_data.config import QCThresholdConfig
from sim_data.io.orientation import AnatomicalPriors, compute_anatomical_priors


def compute_qc_targets(
    records: list[ArtifactRecord],
    thresholds: QCThresholdConfig | None = None,
) -> dict[str, Any]:
    """Compute composite defect burden, continuous QC score [0, 1], and 3-tier qc-studio label.

    Combines worst-defect dominance with cumulative multi-artifact burden so either
    one severe artifact OR multiple moderate artifacts trigger 'fail'.
    """
    thresholds = thresholds or QCThresholdConfig()
    active = [r for r in records if r.severity > 0.0 and not r.params.get("skipped")]

    if not active:
        return {
            "qc_label": "pass",
            "qc_rating_int": 1,
            "qc_score": 1.0,
            "defect_burden": 0.0,
            "max_artifact_severity": 0.0,
            "num_artifacts": 0,
            "primary_artifact": "none",
            "artifact_names": [],
        }

    weighted: list[tuple[float, ArtifactRecord]] = []
    for rec in active:
        cls = ARTIFACT_REGISTRY.get(rec.name)
        w = cls.qc_weight if cls is not None else 1.0
        weighted.append((float(np.clip(rec.severity * w, 0.0, 1.0)), rec))

    weighted.sort(key=lambda item: item[0], reverse=True)
    worst = weighted[0][0]
    secondary_sum = sum(val for val, _ in weighted[1:])
    # Non-linear aggregation: worst defect dominates, additional defects compound burden smoothly
    defect_burden = float(np.clip(worst + 0.30 * secondary_sum * (1.0 - 0.5 * worst), 0.0, 1.0))
    qc_score = float(np.clip(1.0 - defect_burden, 0.0, 1.0))

    if defect_burden <= thresholds.pass_max_severity:
        label, rating_int = "pass", 1
    elif defect_burden <= thresholds.borderline_max_severity:
        label, rating_int = "borderline", 2
    else:
        label, rating_int = "fail", 3

    return {
        "qc_label": label,
        "qc_rating_int": rating_int,
        "qc_score": round(qc_score, 4),
        "defect_burden": round(defect_burden, 4),
        "max_artifact_severity": round(max(r.severity for r in active), 4),
        "num_artifacts": len(active),
        "primary_artifact": weighted[0][1].name,
        "artifact_names": [r.name for r in active],
    }


def compute_fidelity_metrics(
    clean_data: np.ndarray,
    sim_data: np.ndarray,
    clean_priors: AnatomicalPriors,
) -> dict[str, float]:
    """Compute full-reference image fidelity (PSNR, SSIM, NRMSE, brain_mask_retention)."""
    ref = np.mean(clean_data, axis=3) if clean_data.ndim == 4 else clean_data
    cur = np.mean(sim_data, axis=3) if sim_data.ndim == 4 else sim_data

    # Handle matrix-truncated cropping cleanly
    if cur.shape != ref.shape:
        cur_priors = compute_anatomical_priors(cur)
        retention = float(
            np.clip(
                cur_priors.brain_mask.sum() / max(1, clean_priors.brain_mask.sum()),
                0.0,
                1.0,
            )
        )
        return {
            "ref_psnr": 0.0,
            "ref_ssim": 0.0,
            "ref_nrmse": 1.0,
            "brain_mask_retention": round(retention, 4),
        }

    mask = clean_priors.brain_mask
    r_fg = ref[mask].astype(np.float64)
    c_fg = cur[mask].astype(np.float64)
    # Scale-invariant normalization so pure site gain shifts don't artificially nuke PSNR/SSIM
    r_scale = max(float(np.percentile(r_fg, 95.0)), 1e-6)
    c_scale = max(float(np.percentile(c_fg, 95.0)), 1e-6)
    r_norm = np.clip(ref / r_scale, 0.0, 2.0)
    c_norm = np.clip(cur / c_scale, 0.0, 2.0)

    mse = float(np.mean((r_norm[mask] - c_norm[mask]) ** 2))
    nrmse = float(np.sqrt(mse))
    psnr = float(min(60.0, 20.0 * np.log10(1.0 / max(nrmse, 1e-4))))

    # Compute 2D SSIM on midline axial slice for speed + stability across small phantom shapes
    mid_z = (clean_priors.bbox_min[2] + clean_priors.bbox_max[2]) // 2
    sl_r = r_norm[:, :, mid_z]
    sl_c = c_norm[:, :, mid_z]
    win = min(7, sl_r.shape[0], sl_r.shape[1])
    if win % 2 == 0:
        win = max(3, win - 1)
    try:
        ssim_val = float(ssim(sl_r, sl_c, data_range=2.0, win_size=win))
    except Exception:
        ssim_val = 0.0

    # Retained positive foreground brain fraction (drops under cropping/defacing/severe voids)
    thresh_alive = 0.05 * c_scale
    retention = float(np.mean(cur[mask] > thresh_alive))

    return {
        "ref_psnr": round(psnr, 3),
        "ref_ssim": round(float(np.clip(ssim_val, -1.0, 1.0)), 4),
        "ref_nrmse": round(nrmse, 4),
        "brain_mask_retention": round(retention, 4),
    }


def write_manifest(
    rows: list[dict[str, Any]],
    out_dir: str | Path,
    formats: Sequence[str],
) -> dict[str, Path]:
    """Persist unified simulation manifest as Parquet, TSV, and/or JSON."""
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    written: dict[str, Path] = {}

    for fmt in formats:
        if fmt == "parquet":
            p = out_root / "manifest.parquet"
            df.to_parquet(p, index=False)
            written["parquet"] = p
        elif fmt == "tsv":
            p = out_root / "manifest.tsv"
            df.to_csv(p, sep="\t", index=False)
            written["tsv"] = p
        elif fmt == "json":
            p = out_root / "manifest.json"
            p.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            written["json"] = p

    return written
