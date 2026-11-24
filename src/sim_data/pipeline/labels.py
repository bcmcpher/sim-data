"""Defect burden calculation, full-reference fidelity metrics, and TSV manifest writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim

from sim_data.artifacts.base import ArtifactRecord
from sim_data.artifacts.registry import ARTIFACT_REGISTRY
from sim_data.config import QCThresholdConfig
from sim_data.io.orientation import AnatomicalPriors, compute_anatomical_priors


def compute_defect_summary(
    records: list[ArtifactRecord],
    thresholds: QCThresholdConfig | None = None,
    include_proxy_labels: bool = False,
) -> dict[str, Any]:
    """Compute tunable continuous `defect_burden` and artifact counts from applied records.

    Discrete proxy QC tiers (`proxy_qc_label`, `proxy_qc_rating_int`) are included only
    when `include_proxy_labels=True`; true categorical QC ratings should come downstream
    from manual or AI raters.
    """
    thresholds = thresholds or QCThresholdConfig()
    active = [r for r in records if r.severity > 0.0 and not r.params.get("skipped")]

    if not active:
        summary: dict[str, Any] = {
            "defect_burden": 0.0,
            "max_artifact_severity": 0.0,
            "num_artifacts": 0,
            "primary_artifact": "none",
            "artifact_names": [],
        }
        if include_proxy_labels:
            summary["proxy_qc_label"] = "pass"
            summary["proxy_qc_rating_int"] = 1
        return summary

    weighted: list[tuple[float, ArtifactRecord]] = []
    for rec in active:
        cls = ARTIFACT_REGISTRY.get(rec.name)
        w = cls.qc_weight if cls is not None else 1.0
        weighted.append((float(np.clip(rec.severity * w, 0.0, 1.0)), rec))

    weighted.sort(key=lambda item: item[0], reverse=True)
    worst = weighted[0][0]
    secondary_sum = sum(val for val, _ in weighted[1:])
    # Tunable continuous defect burden equation (worst-defect dominant + cumulative secondary load)
    defect_burden = float(np.clip(worst + 0.30 * secondary_sum * (1.0 - 0.5 * worst), 0.0, 1.0))

    summary = {
        "defect_burden": round(defect_burden, 4),
        "max_artifact_severity": round(max(r.severity for r in active), 4),
        "num_artifacts": len(active),
        "primary_artifact": weighted[0][1].name,
        "artifact_names": [r.name for r in active],
    }
    if include_proxy_labels:
        if defect_burden <= thresholds.pass_max_severity:
            label, rating_int = "pass", 1
        elif defect_burden <= thresholds.borderline_max_severity:
            label, rating_int = "borderline", 2
        else:
            label, rating_int = "fail", 3
        summary["proxy_qc_label"] = label
        summary["proxy_qc_rating_int"] = rating_int
    return summary


def compute_fidelity_metrics(
    clean_data: np.ndarray,
    sim_data: np.ndarray,
    clean_priors: AnatomicalPriors,
) -> dict[str, float]:
    """Compute full-reference image fidelity (PSNR, SSIM, NRMSE, brain_mask_retention)."""
    ref = np.mean(clean_data, axis=3) if clean_data.ndim == 4 else clean_data
    cur = np.mean(sim_data, axis=3) if sim_data.ndim == 4 else sim_data

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
    r_scale = max(float(np.percentile(r_fg, 95.0)), 1e-6)
    c_scale = max(float(np.percentile(c_fg, 95.0)), 1e-6)
    r_norm = np.clip(ref / r_scale, 0.0, 2.0)
    c_norm = np.clip(cur / c_scale, 0.0, 2.0)

    mse = float(np.mean((r_norm[mask] - c_norm[mask]) ** 2))
    nrmse = float(np.sqrt(mse))
    psnr = float(min(60.0, 20.0 * np.log10(1.0 / max(nrmse, 1e-4))))

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

    thresh_alive = 0.05 * c_scale
    retention = float(np.mean(cur[mask] > thresh_alive))

    return {
        "ref_psnr": round(psnr, 3),
        "ref_ssim": round(float(np.clip(ssim_val, -1.0, 1.0)), 4),
        "ref_nrmse": round(nrmse, 4),
        "brain_mask_retention": round(retention, 4),
    }


def write_manifest_tsv(
    rows: list[dict[str, Any]],
    manifest_path: str | Path,
) -> Path:
    """Persist official ground-truth simulation manifest as a single BIDS-friendly TSV."""
    target = Path(manifest_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(target, sep="\t", index=False)
    return target
