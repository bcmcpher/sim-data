"""End-to-end multimodal BIDS cohort synthesis & real-subject modality completion."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import nibabel.processing as nib_proc
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ghibli.config import CohortSynthesisConfig, DiffeomorphicWarpConfig, Modality
from ghibli.io.bids import (
    BIDSScan,
    discover_bids_scans,
    discover_subject_brain_mask,
    load_bids_scan,
)
from ghibli.io.orientation import canonicalize_to_ras
from ghibli.pipeline.sampler import derive_seed
from ghibli.synthesis.brain_mask import extract_brain_mask
from ghibli.synthesis.contrast import get_contrast_converter
from ghibli.synthesis.modalities import (
    synthesize_bold_4d,
    synthesize_dwi_4d,
    synthesize_structural_modality,
)
from ghibli.synthesis.tissue import (
    TissuePVEMaps,
    estimate_tissue_pve,
    sample_tissue_mri_params,
)
from ghibli.synthesis.warp import DiffeomorphicField, sample_diffeomorphic_field, warp_volume


@dataclass
class SeedSubject:
    subject: str
    t1_scan: BIDSScan
    session: str | None = None
    t2_scan: BIDSScan | None = None
    flair_scan: BIDSScan | None = None
    bold_scan: BIDSScan | None = None
    dwi_scan: BIDSScan | None = None
    brain_mask_path: Path | None = None

    def get_real_scan(self, modality: Modality) -> BIDSScan | None:
        lookup: dict[Modality, BIDSScan | None] = {
            "T1w": self.t1_scan,
            "T2w": self.t2_scan,
            "FLAIR": self.flair_scan,
            "bold": self.bold_scan,
            "dwi": self.dwi_scan,
        }
        return lookup[modality]


@dataclass
class CohortSynthesisResult:
    out_dir: Path
    n_subjects: int
    n_seed_subjects: int
    manifest_path: Path
    rows: list[dict[str, Any]]


def discover_structural_seeds(
    seed_bids_dir: str | Path,
    brain_mask_suffix: str = "desc-brain_mask",
    acq: str | None = None,
    filename_glob: str | None = None,
) -> list[SeedSubject]:
    """Discover clean real seed subjects (requiring T1w + optional T2w/FLAIR/bold/dwi/brain mask)."""
    all_scans = discover_bids_scans(seed_bids_dir, acq=acq, filename_glob=filename_glob)
    by_sub_mod: dict[tuple[str, Modality], BIDSScan] = {}
    t1_scans: list[BIDSScan] = []
    for s in all_scans:
        by_sub_mod.setdefault((s.subject, s.modality), s)
        if s.modality == "T1w":
            t1_scans.append(s)

    seen_subs: set[str] = set()
    seeds: list[SeedSubject] = []
    for t1 in t1_scans:
        if t1.subject in seen_subs:
            continue
        seen_subs.add(t1.subject)
        mask_path = discover_subject_brain_mask(
            seed_bids_dir, subject=t1.subject, session=t1.session, suffix=brain_mask_suffix
        )
        seeds.append(
            SeedSubject(
                subject=t1.subject,
                session=t1.session,
                t1_scan=t1,
                t2_scan=by_sub_mod.get((t1.subject, "T2w")),
                flair_scan=by_sub_mod.get((t1.subject, "FLAIR")),
                bold_scan=by_sub_mod.get((t1.subject, "bold")),
                dwi_scan=by_sub_mod.get((t1.subject, "dwi")),
                brain_mask_path=mask_path,
            )
        )
    return seeds


def _copy_real_scan_to_out(scan: BIDSScan, out_root: Path, sub_id: str, ses_id: str | None) -> None:
    """Copy an existing real QC-passing BIDS scan and its companion files untouched."""
    base_dir = out_root / sub_id / f"ses-{ses_id}" / scan.datatype if ses_id else out_root / sub_id / scan.datatype
    base_dir.mkdir(parents=True, exist_ok=True)

    prefix_parts = [sub_id]
    if ses_id:
        prefix_parts.append(f"ses-{ses_id}")
    if "task" in scan.entities:
        prefix_parts.append(f"task-{scan.entities['task']}")
    elif scan.modality == "bold":
        prefix_parts.append("task-rest")
    prefix_parts.append(scan.modality)
    stem = "_".join(prefix_parts)

    nifti_ext = ".nii.gz" if scan.nifti_path.name.endswith(".nii.gz") else ".nii"
    shutil.copy2(scan.nifti_path, base_dir / f"{stem}{nifti_ext}")

    if scan.json_path and scan.json_path.exists():
        shutil.copy2(scan.json_path, base_dir / f"{stem}.json")
    else:
        default_json: dict[str, Any] = {
            "Modality": "MR",
            "MagneticFieldStrength": 3.0,
            "PhaseEncodingDirection": "j-",
            "SliceEncodingDirection": "k",
        }
        if scan.modality == "bold":
            default_json["TaskName"] = scan.entities.get("task", "rest")
        (base_dir / f"{stem}.json").write_text(json.dumps(default_json, indent=2), encoding="utf-8")

    if scan.bval_path and scan.bval_path.exists():
        shutil.copy2(scan.bval_path, base_dir / f"{stem}.bval")
    if scan.bvec_path and scan.bvec_path.exists():
        shutil.copy2(scan.bvec_path, base_dir / f"{stem}.bvec")


def _identity_field(shape: tuple[int, int, int]) -> DiffeomorphicField:
    sx, sy, sz = shape
    gx, gy, gz = np.meshgrid(
        np.arange(sx, dtype=np.float32),
        np.arange(sy, dtype=np.float32),
        np.arange(sz, dtype=np.float32),
        indexing="ij",
    )
    coords = np.stack([gx, gy, gz], axis=0)
    return DiffeomorphicField(
        coords=coords,
        displacement_vox=np.zeros_like(coords),
        affine_3x3=np.eye(3, dtype=np.float64),
        flipped_lr=False,
        max_disp_mm=0.0,
        mean_jac_det=1.0,
        min_jac_det=1.0,
    )


def _synthesize_single_subject(
    synth_idx: int,
    seed_subj: SeedSubject,
    config: CohortSynthesisConfig,
    out_root: Path,
    device: str,
) -> dict[str, Any]:
    preserve_real = config.preserve_real_modalities
    if preserve_real:
        sub_id = f"sub-{seed_subj.subject}"
        ses_id = seed_subj.session
    else:
        sub_id = f"sub-{synth_idx:03d}"
        ses_id = None

    seed = derive_seed(config.global_seed, f"synth:{sub_id}:{seed_subj.subject}")
    rng = np.random.default_rng(seed)

    loaded_t1 = load_bids_scan(seed_subj.t1_scan)
    t1_data, affine, zooms = canonicalize_to_ras(loaded_t1.img)
    shape = (int(t1_data.shape[0]), int(t1_data.shape[1]), int(t1_data.shape[2]))

    # Resolve optional real T2w (resampled onto T1w grid if shapes differ)
    t2_data: np.ndarray | None = None
    if seed_subj.t2_scan is not None:
        loaded_t2 = load_bids_scan(seed_subj.t2_scan)
        t2_raw, t2_aff, _ = canonicalize_to_ras(loaded_t2.img)
        if t2_raw.shape == t1_data.shape and np.allclose(t2_aff, affine, atol=1e-2):
            t2_data = t2_raw
        else:
            res_t2 = nib_proc.resample_from_to(
                nib.Nifti1Image(t2_raw, t2_aff), (shape, affine), order=1
            )
            t2_data = np.maximum(
                np.asarray(res_t2.get_fdata(dtype=np.float32), dtype=np.float32), 0.0
            )

    # Check if we even need to synthesize any missing modalities in preserve_real mode
    missing_mods = [
        m for m in config.modalities if not (preserve_real and seed_subj.get_real_scan(m) is not None)
    ]

    # Resolve brain/ICV mask ONCE per subject (external mask -> internal EVAC+ -> morphological)
    eff_brain_mask_path = config.templates.brain_mask_nifti or seed_subj.brain_mask_path
    seed_brain_mask_arr: np.ndarray | None = None
    brain_mask_method = "none"
    if missing_mods:
        bm_res = extract_brain_mask(
            t1_corr=t1_data,
            affine=affine,
            method=config.contrast.brain_extraction,
            brain_mask_path=eff_brain_mask_path,
            device=device,
        )
        seed_brain_mask_arr = bm_res.icv_mask
        brain_mask_method = bm_res.method

    # 1. Estimate soft tissue PVE maps in seed space (when any modality needs synthesis)
    if missing_mods:
        seed_pve = estimate_tissue_pve(
            t1_data,
            target_affine=affine,
            t2_data=t2_data,
            template_pve_path=config.templates.tissue_pve_nifti,
            brain_mask_path=eff_brain_mask_path,
            brain_mask_array=seed_brain_mask_arr,
            seed=seed,
        )
    else:
        seed_pve = None

    # 2. Coherent 3D diffeomorphic warp (strictly identity when preserving real native observations)
    if preserve_real:
        field = _identity_field(shape)
        warped_t1 = t1_data
        warped_t2 = t2_data
        warped_brain_mask = seed_brain_mask_arr
        warped_pve = seed_pve
    else:
        warp_cfg: DiffeomorphicWarpConfig = config.warp
        field = sample_diffeomorphic_field(shape, zooms, warp_cfg, rng)
        warped_t1 = warp_volume(t1_data, field, order=1)
        warped_t2 = warp_volume(t2_data, field, order=1) if t2_data is not None else None
        warped_brain_mask = (
            (warp_volume(seed_brain_mask_arr.astype(np.float32), field, order=0) > 0.5)
            if seed_brain_mask_arr is not None
            else None
        )
        assert seed_pve is not None
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

    # Pre-compute sequential high-res T2-like contrast bases for BOLD / DWI when needed
    converter = get_contrast_converter(config.contrast)
    bold_t2_base: np.ndarray | None = None
    dwi_t2_base: np.ndarray | None = None
    bold_base_fat_suppressed = False
    dwi_base_fat_suppressed = False

    if "bold" in missing_mods:
        if warped_t2 is not None:
            bold_t2_base = warped_t2
            bold_base_fat_suppressed = False
        else:
            bold_conv = converter.convert(
                t1_data=warped_t1,
                affine=affine,
                target="bold_s0",
                params=mri_params,
                cfg=config.contrast,
                brain_mask_array=warped_brain_mask,
                te_ms=config.bold.te_sec * 1000.0,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            bold_t2_base = bold_conv.data
            bold_base_fat_suppressed = True

    if "dwi" in missing_mods:
        if warped_t2 is not None:
            dwi_t2_base = warped_t2
            dwi_base_fat_suppressed = False
        else:
            dwi_conv = converter.convert(
                t1_data=warped_t1,
                affine=affine,
                target="dwi_b0",
                params=mri_params,
                cfg=config.contrast,
                brain_mask_array=warped_brain_mask,
                te_ms=config.dwi.te_sec * 1000.0,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            dwi_t2_base = dwi_conv.data
            dwi_base_fat_suppressed = True

    sub_base = out_root / sub_id / f"ses-{ses_id}" if ses_id else out_root / sub_id
    stem_prefix = f"{sub_id}_ses-{ses_id}" if ses_id else sub_id

    # 4. Copy real modalities (when preserve_real) or synthesize missing modalities sequentially
    written_mods: list[str] = []
    mod_provenance: dict[str, str] = {}

    for mod in config.modalities:
        real_scan = seed_subj.get_real_scan(mod) if preserve_real else None
        if real_scan is not None:
            _copy_real_scan_to_out(real_scan, out_root, sub_id=sub_id, ses_id=ses_id)
            written_mods.append(mod)
            mod_provenance[mod] = "real"
            continue

        assert warped_pve is not None
        if mod in ("T1w", "T2w", "FLAIR"):
            anat_dir = sub_base / "anat"
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
                affine=affine,
                contrast_cfg=config.contrast,
                brain_mask_array=warped_brain_mask,
            )
            stem = f"{stem_prefix}_{mod}"
            nib.save(nib.Nifti1Image(vol, affine), str(anat_dir / f"{stem}.nii.gz"))
            sidecar = {
                "Modality": "MR",
                "MagneticFieldStrength": 3.0,
                "PhaseEncodingDirection": str(
                    loaded_t1.sidecar.get("PhaseEncodingDirection", "j-")
                ),
                "SliceEncodingDirection": str(
                    loaded_t1.sidecar.get("SliceEncodingDirection", "k")
                ),
                **timing,
            }
            (anat_dir / f"{stem}.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
            written_mods.append(mod)
            if mod == "T2w" and warped_t2 is None:
                mod_provenance[mod] = "contrast_converted_from_t1w"
            elif warped_t2 is not None and mod in ("T2w", "FLAIR"):
                mod_provenance[mod] = "synthesized_from_t2w"
            else:
                mod_provenance[mod] = "synthesized_from_t1w"

        elif mod == "bold":
            func_dir = sub_base / "func"
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
                t2_base_vol=bold_t2_base,
                fat_retention_epi=config.contrast.fat_retention_epi,
                t2_base_is_fat_suppressed=bold_base_fat_suppressed,
                epi_skull_strip=config.contrast.epi_skull_strip,
                epi_crop_to_brain=config.contrast.epi_crop_to_brain,
                epi_crop_pad_mm=config.contrast.epi_crop_pad_mm,
            )
            stem = f"{stem_prefix}_task-rest_bold"
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
            mod_provenance[mod] = (
                "synthesized_from_t2w" if warped_t2 is not None else "synthesized_from_t1w_contrast"
            )

        elif mod == "dwi":
            dwi_dir = sub_base / "dwi"
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
                t2_base_vol=dwi_t2_base,
                fat_retention_epi=config.contrast.fat_retention_epi,
                t2_base_is_fat_suppressed=dwi_base_fat_suppressed,
                epi_skull_strip=config.contrast.epi_skull_strip,
                epi_crop_to_brain=config.contrast.epi_crop_to_brain,
                epi_crop_pad_mm=config.contrast.epi_crop_pad_mm,
            )
            stem = f"{stem_prefix}_dwi"
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
            mod_provenance[mod] = (
                "synthesized_from_t2w" if warped_t2 is not None else "synthesized_from_t1w_contrast"
            )

    if seed_pve is not None and warped_pve is not None:
        seed_brain_vox = max(1, int(seed_pve.brain_mask.sum()))
        brain_vol_ratio = float(warped_pve.brain_mask.sum() / seed_brain_vox)
    else:
        brain_vol_ratio = 1.0

    return {
        "participant_id": sub_id,
        "source_seed_subject": f"sub-{seed_subj.subject}",
        "preserve_real_modalities": preserve_real,
        "used_seed_t2": seed_subj.t2_scan is not None,
        "used_external_brain_mask": eff_brain_mask_path is not None,
        "brain_mask_method": brain_mask_method,
        "seed": seed,
        "flipped_lr": field.flipped_lr,
        "warp_max_disp_mm": field.max_disp_mm,
        "warp_mean_jac_det": field.mean_jac_det,
        "warp_min_jac_det": field.min_jac_det,
        "brain_volume_ratio": round(brain_vol_ratio, 4),
        "modalities": ",".join(written_mods),
        "modality_provenance_json": json.dumps(mod_provenance),
    }


def synthesize_cohort(
    seed_bids_dir: str | Path,
    config: CohortSynthesisConfig,
    out_dir: str | Path,
    workers: int = 1,
    device: str = "auto",
) -> CohortSynthesisResult:
    """Expand structural seeds into N multimodal subjects OR complete missing modalities on real subjects."""
    seeds = discover_structural_seeds(
        seed_bids_dir,
        brain_mask_suffix=config.contrast.brain_mask_suffix,
        acq=config.seed_acq,
        filename_glob=config.seed_glob,
    )
    if not seeds:
        raise ValueError(f"No T1w seed scans found under {seed_bids_dir}")

    if (
        not config.preserve_real_modalities
        and len(seeds) < config.min_seed_subjects
        and not config.allow_few_seeds
    ):
        raise ValueError(
            f"Found only {len(seeds)} structural seed subject(s), but min_seed_subjects="
            f"{config.min_seed_subjects} (3-5+ distinct clean subjects are recommended to prevent "
            f"texture overfitting). Provide more seeds, set allow_few_seeds=True / --allow-few-seeds, "
            f"or use preserve_real_modalities=True / --preserve-real for native modality completion."
        )

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    config.to_yaml(out_root / "synthesis_config_resolved.yaml")

    if config.preserve_real_modalities:
        assignments = [(i + 1, s) for i, s in enumerate(seeds)]
    else:
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
