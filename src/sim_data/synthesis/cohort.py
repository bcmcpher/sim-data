"""End-to-end multimodal BIDS cohort synthesis from 3-5+ clean structural seed subjects."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sim_data.config import CohortSynthesisConfig
from sim_data.io.bids import BIDSScan, discover_bids_scans, load_bids_scan
from sim_data.io.orientation import canonicalize_to_ras
from sim_data.pipeline.sampler import derive_seed
from sim_data.synthesis.modalities import (
    synthesize_bold_4d,
    synthesize_dwi_4d,
    synthesize_structural_modality,
)
from sim_data.synthesis.tissue import (
    TissuePVEMaps,
    estimate_tissue_pve,
    sample_tissue_mri_params,
)
from sim_data.synthesis.warp import sample_diffeomorphic_field, warp_volume


@dataclass
class SeedSubject:
    subject: str
    t1_scan: BIDSScan
    t2_scan: BIDSScan | None = None


@dataclass
class CohortSynthesisResult:
    out_dir: Path
    n_subjects: int
    n_seed_subjects: int
    manifest_path: Path
    rows: list[dict[str, Any]]


def discover_structural_seeds(seed_bids_dir: str | Path) -> list[SeedSubject]:
    """Discover clean structural seed subjects (T1w + optional paired T2w)."""
    t1_scans = discover_bids_scans(seed_bids_dir, modalities=["T1w"])
    t2_scans = discover_bids_scans(seed_bids_dir, modalities=["T2w"])
    t2_by_sub: dict[str, BIDSScan] = {}
    for s in t2_scans:
        t2_by_sub.setdefault(s.subject, s)

    seen_subs: set[str] = set()
    seeds: list[SeedSubject] = []
    for t1 in t1_scans:
        if t1.subject in seen_subs:
            continue
        seen_subs.add(t1.subject)
        seeds.append(
            SeedSubject(
                subject=t1.subject,
                t1_scan=t1,
                t2_scan=t2_by_sub.get(t1.subject),
            )
        )
    return seeds


def _synthesize_single_subject(
    synth_idx: int,
    seed_subj: SeedSubject,
    config: CohortSynthesisConfig,
    out_root: Path,
    device: str,
) -> dict[str, Any]:
    sub_label = f"{synth_idx:03d}"
    sub_id = f"sub-{sub_label}"
    seed = derive_seed(config.global_seed, f"synth:{sub_id}:{seed_subj.subject}")
    rng = np.random.default_rng(seed)

    loaded_t1 = load_bids_scan(seed_subj.t1_scan)
    t1_data, affine, zooms = canonicalize_to_ras(loaded_t1.img)

    t2_data: np.ndarray | None = None
    if seed_subj.t2_scan is not None:
        loaded_t2 = load_bids_scan(seed_subj.t2_scan)
        t2_raw, _t2_aff, _ = canonicalize_to_ras(loaded_t2.img)
        if t2_raw.shape == t1_data.shape:
            t2_data = t2_raw

    # 1. Estimate soft tissue PVE maps in seed space (template-free or optional template)
    seed_pve = estimate_tissue_pve(
        t1_data,
        target_affine=affine,
        t2_data=t2_data,
        template_pve_path=config.templates.tissue_pve_nifti,
        seed=seed,
    )

    # 2. Sample ONE coherent 3D diffeomorphic warp shared across ALL modalities of this subject
    shape = (int(t1_data.shape[0]), int(t1_data.shape[1]), int(t1_data.shape[2]))
    field = sample_diffeomorphic_field(shape, zooms, config.warp, rng)

    warped_t1 = warp_volume(t1_data, field, order=1)
    warped_t2 = warp_volume(t2_data, field, order=1) if t2_data is not None else None
    w_csf = np.clip(warp_volume(seed_pve.csf, field, order=1), 0.0, 1.0)
    w_gm = np.clip(warp_volume(seed_pve.gm, field, order=1), 0.0, 1.0)
    w_wm = np.clip(warp_volume(seed_pve.wm, field, order=1), 0.0, 1.0)
    w_sc = np.clip(warp_volume(seed_pve.scalp, field, order=1), 0.0, 1.0)

    total = w_csf + w_gm + w_wm + w_sc
    head_mask = total > 0.15
    denom = np.where(head_mask, np.maximum(total, 1e-5), 1.0)
    warped_pve = TissuePVEMaps(
        csf=np.where(head_mask, w_csf / denom, 0.0).astype(np.float32),
        gm=np.where(head_mask, w_gm / denom, 0.0).astype(np.float32),
        wm=np.where(head_mask, w_wm / denom, 0.0).astype(np.float32),
        scalp=np.where(head_mask, w_sc / denom, 0.0).astype(np.float32),
        brain_mask=np.asarray((w_csf + w_gm + w_wm) > 0.35, dtype=bool),
        head_mask=np.asarray(head_mask, dtype=bool),
    )

    # 3. Sample subject-specific biological MRI tissue parameters
    mri_params = sample_tissue_mri_params(rng, jitter=config.structural.tissue_param_jitter)

    # 4. Synthesize each requested modality coherently from the warped anatomy
    written_mods: list[str] = []
    for mod in config.modalities:
        if mod in ("T1w", "T2w", "FLAIR"):
            anat_dir = out_root / sub_id / "anat"
            anat_dir.mkdir(parents=True, exist_ok=True)
            vol, timing = synthesize_structural_modality(
                warped_pve,
                warped_t1=warped_t1,
                warped_t2=warped_t2,
                modality=mod,
                params=mri_params,
                cfg=config.structural,
                rng=rng,
                device=device,
            )
            stem = f"{sub_id}_{mod}"
            nib.save(nib.Nifti1Image(vol, affine), str(anat_dir / f"{stem}.nii.gz"))
            sidecar = {
                "Modality": "MR",
                "MagneticFieldStrength": 3.0,
                "PhaseEncodingDirection": str(
                    loaded_t1.sidecar.get("PhaseEncodingDirection", "j-")
                ),
                "SliceEncodingDirection": str(loaded_t1.sidecar.get("SliceEncodingDirection", "k")),
                **timing,
            }
            (anat_dir / f"{stem}.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
            written_mods.append(mod)

        elif mod == "bold":
            func_dir = out_root / sub_id / "func"
            func_dir.mkdir(parents=True, exist_ok=True)
            bold_4d, bold_aff = synthesize_bold_4d(
                warped_pve,
                src_affine=affine,
                src_zooms=zooms,
                params=mri_params,
                cfg=config.bold,
                warp_field=field,
                rsn_template_path=config.templates.rsn_parcellation_nifti,
                rng=rng,
                device=device,
            )
            stem = f"{sub_id}_task-rest_bold"
            nib.save(nib.Nifti1Image(bold_4d, bold_aff), str(func_dir / f"{stem}.nii.gz"))
            bold_json = {
                "Modality": "MR",
                "TaskName": "rest",
                "MagneticFieldStrength": 3.0,
                "RepetitionTime": config.bold.tr_sec,
                "EchoTime": config.bold.te_sec,
                "PhaseEncodingDirection": "j-",
                "SliceEncodingDirection": "k",
            }
            (func_dir / f"{stem}.json").write_text(
                json.dumps(bold_json, indent=2), encoding="utf-8"
            )
            written_mods.append(mod)

        elif mod == "dwi":
            dwi_dir = out_root / sub_id / "dwi"
            dwi_dir.mkdir(parents=True, exist_ok=True)
            dwi_4d, dwi_aff, bvals, bvecs = synthesize_dwi_4d(
                warped_pve,
                src_affine=affine,
                src_zooms=zooms,
                params=mri_params,
                cfg=config.dwi,
                warp_field=field,
                dti_template_path=config.templates.dti_eigenvector_nifti,
                rng=rng,
                device=device,
            )
            stem = f"{sub_id}_dwi"
            nib.save(nib.Nifti1Image(dwi_4d, dwi_aff), str(dwi_dir / f"{stem}.nii.gz"))
            dwi_json = {
                "Modality": "MR",
                "MagneticFieldStrength": 3.0,
                "RepetitionTime": config.dwi.tr_sec,
                "EchoTime": config.dwi.te_sec,
                "PhaseEncodingDirection": "j-",
                "SliceEncodingDirection": "k",
            }
            (dwi_dir / f"{stem}.json").write_text(json.dumps(dwi_json, indent=2), encoding="utf-8")
            (dwi_dir / f"{stem}.bval").write_text(
                " ".join(f"{float(v):.1f}" for v in bvals) + "\n", encoding="utf-8"
            )
            bvec_lines = [" ".join(f"{float(v):.6f}" for v in row) for row in bvecs]
            (dwi_dir / f"{stem}.bvec").write_text("\n".join(bvec_lines) + "\n", encoding="utf-8")
            written_mods.append(mod)

    seed_brain_vox = max(1, int(seed_pve.brain_mask.sum()))
    brain_vol_ratio = float(warped_pve.brain_mask.sum() / seed_brain_vox)

    return {
        "participant_id": sub_id,
        "source_seed_subject": f"sub-{seed_subj.subject}",
        "used_seed_t2": seed_subj.t2_scan is not None,
        "seed": seed,
        "flipped_lr": field.flipped_lr,
        "warp_max_disp_mm": field.max_disp_mm,
        "warp_mean_jac_det": field.mean_jac_det,
        "warp_min_jac_det": field.min_jac_det,
        "brain_volume_ratio": round(brain_vol_ratio, 4),
        "modalities": ",".join(written_mods),
    }


def synthesize_cohort(
    seed_bids_dir: str | Path,
    config: CohortSynthesisConfig,
    out_dir: str | Path,
    workers: int = 1,
    device: str = "auto",
) -> CohortSynthesisResult:
    """Expand 3-5+ clean structural seed subjects into N distinct multimodal BIDS subjects."""
    seeds = discover_structural_seeds(seed_bids_dir)
    if not seeds:
        raise ValueError(f"No T1w seed scans found under {seed_bids_dir}")

    if len(seeds) < config.min_seed_subjects and not config.allow_few_seeds:
        raise ValueError(
            f"Found only {len(seeds)} structural seed subject(s), but min_seed_subjects="
            f"{config.min_seed_subjects} (3-5+ distinct clean subjects are recommended to prevent "
            f"texture overfitting). Provide more seeds or set allow_few_seeds=True / --allow-few-seeds."
        )

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    config.to_yaml(out_root / "synthesis_config_resolved.yaml")

    # Assign seeds round-robin (balanced across all seed subjects)
    assignments = [(i + 1, seeds[i % len(seeds)]) for i in range(config.n_subjects)]

    if workers <= 1 or len(assignments) <= 1:
        rows = [
            _synthesize_single_subject(idx, s, config, out_root, device) for idx, s in assignments
        ]
    else:
        rows = Parallel(n_jobs=workers, prefer="processes")(
            delayed(_synthesize_single_subject)(idx, s, config, out_root, device)
            for idx, s in assignments
        )

    # Write standard raw BIDS root files so the output feeds directly into `sim-data generate` or NiPreps
    (out_root / "dataset_description.json").write_text(
        json.dumps(
            {
                "Name": "Synthesized Clean Multimodal MRI Cohort",
                "BIDSVersion": "1.9.0",
                "DatasetType": "raw",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.DataFrame([{"participant_id": r["participant_id"]} for r in rows]).to_csv(
        out_root / "participants.tsv", sep="\t", index=False
    )
    (out_root / ".bidsignore").write_text(
        "synthesis_manifest.tsv\nsynthesis_config_resolved.yaml\n", encoding="utf-8"
    )

    manifest_path = out_root / "synthesis_manifest.tsv"
    pd.DataFrame(rows).to_csv(manifest_path, sep="\t", index=False)

    return CohortSynthesisResult(
        out_dir=out_root,
        n_subjects=len(rows),
        n_seed_subjects=len(seeds),
        manifest_path=manifest_path,
        rows=rows,
    )
