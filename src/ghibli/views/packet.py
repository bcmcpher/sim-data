"""qc-studio Visual-QC packet.json assembler (v1.1: midline + enhanced + NiPreps IQMs)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ghibli.config import Modality
from ghibli.iqm.extractor import iqm_schema_name


def assemble_qc_packet(
    scan_id: str,
    modality: Modality,
    site_id: str,
    iqms: dict[str, float],
    views: dict[str, dict[str, str]],
    ground_truth: dict[str, Any],
    nifti_rel_path: str | None = None,
    deterministic_priors: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a structured visual-QC packet compatible with qc-studio AI + deterministic raters."""
    packet: dict[str, Any] = {
        "packet_version": "1.1",
        "scan_id": scan_id,
        "modality": modality,
        "site_id": site_id,
        "iqm_schema": iqm_schema_name(modality),
        "nifti_path": nifti_rel_path,
        "rating_scale": {
            "labels": ["pass", "borderline", "fail"],
            "int_map": {"pass": 1, "borderline": 2, "fail": 3},
        },
        "iqms": iqms,
        "views": views,
        "ground_truth": ground_truth,
    }
    if deterministic_priors is not None:
        packet["deterministic_priors"] = deterministic_priors
    return packet


def write_qc_packet(packet: dict[str, Any], out_dir: str | Path) -> Path:
    """Write packet.json under <out_dir>/packets/<scan_id>_packet.json."""
    packets_dir = Path(out_dir) / "packets"
    packets_dir.mkdir(parents=True, exist_ok=True)
    path = packets_dir / f"{packet['scan_id']}_packet.json"
    path.write_text(json.dumps(packet, indent=2), encoding="utf-8")
    return path
