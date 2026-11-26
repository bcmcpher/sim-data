"""Step 04: Simulate the same subject across multiple MRI scanner sites & audit harmonization.

Exercises:
- Loading curated scanner profiles from `configs/sites_catalog.yaml`
  (`Siemens_3T_32ch`, `GE_15T_8ch`, `Philips_3T_16ch`, `Siemens_3T_64ch_HighAccel`)
  plus `NativeSite(passthrough=True)` as an untouched reference.
- Part 4a (Clean Same-Subject Site Comparison):
  1 clean scan per site -> blind IQM extraction -> visual comparison (shared Native window,
  per-scan window, BG-boost noise/offset floor) + overlaid brain/background histograms + IQM shifts.
- Part 4b (Multi-Site Factorial + ComBat/CovBat & Deterministic QC):
  `multisite_factorial` mode (6 scans/site x 4 curated sites = 24 scans) -> `extract-iqms` ->
  `verify-harmonization` (ComBat vs. CovBat) -> `benchmark-deterministic` (CV Logistic Regression).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

from ghibli.config import (  # noqa: E402
    ArtifactSpec,
    OutputConfig,
    SamplingPlanConfig,
    SimulationConfig,
    SiteProfileConfig,
)
from ghibli.eval.deterministic_qc import evaluate_deterministic_qc  # noqa: E402
from ghibli.harmonization.audit import merge_manifest_and_iqms, run_harmonization_audit  # noqa: E402
from ghibli.io.orientation import compute_anatomical_priors  # noqa: E402
from ghibli.iqm.extractor import extract_iqms_for_bids_dir  # noqa: E402
from ghibli.pipeline.engine import run_simulation_plan  # noqa: E402
from ghibli.sites.generator import load_sites_catalog  # noqa: E402


def _render_sites_comparison(
    site_order: list[str],
    site_volumes: dict[str, np.ndarray],
    site_meta: dict[str, dict[str, Any]],
    out_path: Path,
) -> Path:
    """Render 4-row cross-site visual + intensity histogram comparison for the same subject."""
    native_vol = site_volumes[site_order[0]]
    priors = compute_anatomical_priors(native_vol)
    mid_z = (priors.bbox_min[2] + priors.bbox_max[2]) // 2
    nat_vmin, nat_vmax = common.compute_window(native_vol)
    nat_p95 = float(np.percentile(native_vol[priors.brain_mask], 95.0))

    n_sites = len(site_order)
    fig = plt.figure(figsize=(3.3 * n_sites, 11.2))
    gs = fig.add_gridspec(4, n_sites, height_ratios=[1.0, 1.0, 1.0, 1.05], hspace=0.28, wspace=0.12)

    colors = ["#222222", "#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    ax_hist_brain = fig.add_subplot(gs[3, : max(1, n_sites // 2 + 1)])
    ax_hist_bg = fig.add_subplot(gs[3, max(1, n_sites // 2 + 1) :])

    brain_clip_hi = nat_p95 * 1.45
    bg_clip_hi = max(nat_p95 * 0.12, 1.0)

    for col, sid in enumerate(site_order):
        vol = site_volumes[sid]
        meta = site_meta[sid]
        sl = common.orient_2d(vol[:, :, mid_z])
        own_vmin, own_vmax = common.compute_window(vol)
        own_p95 = float(np.percentile(vol[priors.brain_mask], 95.0))

        # Row 0: Shared Native window (shows true gain/gamma/offset differences)
        ax0 = fig.add_subplot(gs[0, col])
        ax0.imshow(common.normalize_slice(sl, nat_vmin, nat_vmax), cmap="gray", vmin=0.0, vmax=1.0)
        ax0.set_title(
            f"{sid}\nγ={meta['gamma']:.2f} g={meta['gain']:.2f} off={meta['offset']:+.3f}\n"
            f"PSF={meta['psf_z']:.2f} L={meta['coils']}ch",
            fontsize=8,
        )
        ax0.set_xticks([])
        ax0.set_yticks([])
        if col == 0:
            ax0.set_ylabel("Shared Native\nWindow", fontsize=8.5, fontweight="bold")

        # Row 1: Per-scan window (shows tissue contrast, sharpness, bias)
        ax1 = fig.add_subplot(gs[1, col])
        ax1.imshow(common.normalize_slice(sl, own_vmin, own_vmax), cmap="gray", vmin=0.0, vmax=1.0)
        ax1.set_xticks([])
        ax1.set_yticks([])
        if col == 0:
            ax1.set_ylabel("Per-Site\nWindow", fontsize=8.5, fontweight="bold")

        # Row 2: BG-boost (reveals noise floor, g-factor shading, and air offset pedestal!)
        ax2 = fig.add_subplot(gs[2, col])
        ax2.imshow(common.bg_boost_slice(sl, ref_p95=own_p95), cmap="gray", vmin=0.0, vmax=1.0)
        ax2.set_xlabel(f"BG med={meta['bg_median']:.1f} | std={meta['bg_std']:.1f}", fontsize=7.5)
        ax2.set_xticks([])
        ax2.set_yticks([])
        if col == 0:
            ax2.set_ylabel("BG-Boost\n(Noise/Offset)", fontsize=8.5, fontweight="bold")

        # Row 3: Histograms
        clr = colors[col % len(colors)]
        b_vals = vol[priors.brain_mask]
        b_vals = b_vals[(b_vals > 0) & (b_vals <= brain_clip_hi)]
        ax_hist_brain.hist(b_vals, bins=70, density=True, histtype="step", lw=1.6, color=clr, label=sid)

        bg_vals = vol[priors.bg_mask]
        bg_vals = bg_vals[bg_vals <= bg_clip_hi]
        ax_hist_bg.hist(bg_vals, bins=50, density=True, histtype="step", lw=1.6, color=clr, label=sid)

    ax_hist_brain.set_title("Foreground / Brain Intensity Distribution Across Sites", fontsize=9)
    ax_hist_brain.set_xlabel("Voxel Intensity (Native Scale)", fontsize=8)
    ax_hist_brain.set_ylabel("Density", fontsize=8)
    ax_hist_brain.tick_params(labelsize=7.5)
    ax_hist_brain.legend(fontsize=7.5, loc="upper right")

    ax_hist_bg.set_title("Background Air Intensity Distribution (Noise Floor & Offset)", fontsize=9)
    ax_hist_bg.set_xlabel("Voxel Intensity (Low Range)", fontsize=8)
    ax_hist_bg.tick_params(labelsize=7.5)
    ax_hist_bg.legend(fontsize=7.5, loc="upper right")

    fig.suptitle(
        "Step 04a: Same-Subject Multi-Site Simulation (NativeSite vs. 4 Curated Scanner Profiles)",
        fontsize=11.5,
        fontweight="bold",
        y=0.995,
    )
    return common.save_figure(fig, out_path)


def run_step(
    out_dir: Path = common.DEFAULT_OUT_DIR,
    workers: int = 4,
    device: str = "auto",
    skip_factorial: bool = False,
    sites_catalog: Path = common.DEFAULT_SITES_CATALOG,
) -> dict[str, Any]:
    """Run clean cross-site comparison + multi-site factorial ComBat/CovBat harmonization check."""
    out_root = Path(out_dir).resolve()
    seed_diag_path = out_root / "00_seed_diagnostics.json"
    if not seed_diag_path.exists():
        raise FileNotFoundError(f"Run Step 01 first ({seed_diag_path} not found).")
    seed_diag = common.read_json(seed_diag_path)
    seed_bids = Path(seed_diag["seed_bids_dir"])

    checks: list[common.CheckResult] = []
    figures: dict[str, str] = {}

    with common.StepTimer() as timer:
        curated_sites = load_sites_catalog(sites_catalog)
        native_site = SiteProfileConfig(
            site_id="NativeSite",
            description="Untouched native real-scanner baseline (passthrough)",
            passthrough=True,
        )
        all_clean_sites = [native_site, *curated_sites]

        # --- 4a. Clean Same-Subject Site Comparison ---
        clean_out = out_root / "03_sites_clean"
        clean_cfg = SimulationConfig(
            global_seed=3030,
            modalities=["T1w"],
            sampling=SamplingPlanConfig(
                mode="dose_response",
                include_clean_baseline=True,
                shuffle_run_order=False,
                coupled_trajectories=True,
                sweep_all_sites=True,
                dose_severities=[0.0],
                artifacts=[ArtifactSpec(name="rician_noise")],  # unused since pos_sevs is empty
            ),
            sites=all_clean_sites,
            output=OutputConfig(
                save_nifti=True,
                write_bids_metadata=True,
                naming_scheme="blinded",
                site_naming="desc",
                extract_iqms=False,
                render_views=False,
            ),
        )
        clean_res = run_simulation_plan(
            bids_dir=seed_bids,
            config=clean_cfg,
            out_dir=clean_out,
            workers=min(workers, len(all_clean_sites)),
            device=device,
            manifest_out=clean_out / "manifest.tsv",
        )
        clean_manifest = pd.read_csv(clean_res.manifest_path, sep="\t")
        clean_iqms = extract_iqms_for_bids_dir(
            bids_dir=clean_out,
            out_tsv=clean_out / "iqms.tsv",
            modalities=["T1w"],
            workers=min(workers, len(all_clean_sites)),
        )
        merged_clean = merge_manifest_and_iqms(clean_manifest, clean_iqms)
        merged_clean.to_csv(clean_out / "site_iqm_shifts.tsv", sep="\t", index=False)

        site_order = [s.site_id for s in all_clean_sites]
        site_volumes: dict[str, np.ndarray] = {}
        site_meta: dict[str, dict[str, Any]] = {}
        native_priors = None
        pedestal_sites: list[str] = []

        for sid in site_order:
            row = merged_clean[merged_clean["site_id"] == sid].iloc[0]
            vol, _, _ = common.load_ras_volume(clean_out / str(row["nifti_path"]))
            site_volumes[sid] = vol
            if native_priors is None:
                native_priors = compute_anatomical_priors(vol)
            bg_vals = vol[native_priors.bg_mask]
            bg_med = float(np.median(bg_vals))
            bg_std = float(np.std(bg_vals))
            site_meta[sid] = {
                "gamma": float(row["site_gamma"]),
                "gain": float(row["site_gain"]),
                "offset": float(row["site_offset"]),
                "psf_z": float(row["site_psf_z"]),
                "coils": int(row["site_coil_count"]),
                "noise_rel": float(row["site_noise_sigma_rel"]),
                "g_factor": float(row["site_g_factor"]),
                "bg_median": round(bg_med, 3),
                "bg_std": round(bg_std, 3),
                "ref_ssim": round(float(row["ref_ssim"]), 4),
            }
            if sid != "NativeSite" and bg_med > max(5.0 * site_meta["NativeSite"]["bg_median"], 10.0):
                pedestal_sites.append(f"{sid} (bg_med={bg_med:.1f}, offset={float(row['site_offset']):+.3f})")

        fig_sites = _render_sites_comparison(
            site_order=site_order,
            site_volumes=site_volumes,
            site_meta=site_meta,
            out_path=out_root / "figures" / "04a_sites_comparison.png",
        )
        figures["sites_comparison"] = str(fig_sites.relative_to(out_root))

        # Collect concise IQM comparison across sites
        key_iqms = ["iqm_snr_total", "iqm_cnr", "iqm_cjv", "iqm_efc", "iqm_fwhm_avg", "iqm_inu_range", "iqm_qi_1"]
        clean_iqm_table: list[dict[str, Any]] = []
        for sid in site_order:
            row = merged_clean[merged_clean["site_id"] == sid].iloc[0]
            entry: dict[str, Any] = {"site_id": sid, "bg_median": site_meta[sid]["bg_median"]}
            for col in key_iqms:
                if col in merged_clean.columns:
                    entry[col.removeprefix("iqm_")] = round(float(row[col]), 4)
            clean_iqm_table.append(entry)

        checks.append(
            common.CheckResult(
                step="04_multisite_simulation",
                name="clean_site_diversity",
                status="pass" if len(site_order) == 5 else "fail",
                metric=f"{len(site_order)} site conditions generated (Native + {len(curated_sites)} curated)",
                detail="All curated scanner profiles applied cleanly to the same baseline subject.",
            )
        )
        checks.append(
            common.CheckResult(
                step="04_multisite_simulation",
                name="site_background_offset_pedestal",
                status="warn" if pedestal_sites else "pass",
                metric=f"Native bg_med={site_meta['NativeSite']['bg_median']:.1f}; elevated={len(pedestal_sites)} site(s)",
                detail=(
                    f"Positive site intensity_offset adds a uniform pedestal to background air voxels: {', '.join(pedestal_sites)}. Consider masking offset to foreground or applying after background air check."
                    if pedestal_sites
                    else "Background air median remained physically low across all simulated sites."
                ),
            )
        )

        # --- 4b. Multi-site factorial batch -> ComBat vs. CovBat & Deterministic QC ---
        harm_report: dict[str, Any] | None = None
        det_report: dict[str, Any] | None = None
        n_factorial_scans = 0

        if not skip_factorial:
            fact_out = out_root / "03_sites_factorial"
            fact_arts = [
                ArtifactSpec(name="motion_3d", weight=1.2),
                ArtifactSpec(name="ghosting_3d", weight=1.0),
                ArtifactSpec(name="bias_field", weight=1.1),
                ArtifactSpec(name="rician_noise", weight=1.2),
                ArtifactSpec(name="gibbs_ringing", weight=0.9),
                ArtifactSpec(name="anisotropy", weight=0.9),
            ]
            fact_cfg = SimulationConfig(
                global_seed=4040,
                modalities=["T1w"],
                sampling=SamplingPlanConfig(
                    mode="multisite_factorial",
                    include_clean_baseline=True,
                    shuffle_run_order=True,
                    samples_per_scan_per_site=5,
                    min_artifacts=1,
                    max_artifacts=2,
                    severity_range=(0.20, 0.85),
                    artifacts=fact_arts,
                ),
                sites=curated_sites,
                output=OutputConfig(
                    save_nifti=True,
                    write_bids_metadata=True,
                    naming_scheme="blinded",
                    site_naming="desc",
                    extract_iqms=False,
                    render_views=False,
                ),
            )
            fact_res = run_simulation_plan(
                bids_dir=seed_bids,
                config=fact_cfg,
                out_dir=fact_out,
                workers=workers,
                device=device,
                manifest_out=fact_out / "manifest.tsv",
            )
            fact_manifest = pd.read_csv(fact_res.manifest_path, sep="\t")
            n_factorial_scans = len(fact_manifest)
            fact_iqms = extract_iqms_for_bids_dir(
                bids_dir=fact_out,
                out_tsv=fact_out / "iqms.tsv",
                modalities=["T1w"],
                workers=workers,
            )

            harm_res = run_harmonization_audit(
                fact_manifest,
                out_dir=fact_out / "harmonization_audit",
                iqms_df=fact_iqms,
                modality="T1w",
            )
            harm_report = harm_res.report
            if harm_res.plot_path and harm_res.plot_path.exists():
                dst_pca = out_root / "figures" / "04b_site_pca_harmonization.png"
                dst_pca.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(harm_res.plot_path, dst_pca)
                figures["site_pca_harmonization"] = str(dst_pca.relative_to(out_root))

            det_res = evaluate_deterministic_qc(
                fact_manifest,
                out_dir=fact_out / "qc_benchmark",
                iqms_df=fact_iqms,
                modality="T1w",
            )
            det_report = det_res.report

            raw_f = float(harm_report["raw"]["mean_anova_f"])
            covbat_f = float(harm_report["covbat"]["mean_anova_f"])
            checks.append(
                common.CheckResult(
                    step="04_multisite_simulation",
                    name="harmonization_combat_covbat",
                    status="pass" if covbat_f < raw_f else "warn",
                    metric=f"ANOVA F: raw={raw_f:.2f} -> combat={harm_report['combat']['mean_anova_f']:.2f} -> covbat={covbat_f:.2f}; site_rf_auc: {harm_report['raw']['site_rf_auc']:.2f} -> {harm_report['covbat']['site_rf_auc']:.2f}",
                    detail="Verified ComBat & CovBat harmonization pipeline end-to-end (note: 1-subject factorial smoke check; multi-subject cohorts are required for population benchmarks).",
                )
            )
            checks.append(
                common.CheckResult(
                    step="04_multisite_simulation",
                    name="deterministic_qc_benchmark",
                    status="pass",
                    metric=f"Macro-F1={det_report['macro_f1']}, Kappa={det_report['cohen_kappa']}, QWK={det_report['quadratic_weighted_kappa']}",
                    detail=f"Cross-validated LogisticRegression baseline ran cleanly on n={det_report['n_scans']} scans.",
                )
            )

    summary: dict[str, Any] = {
        "step": "04_multisite_simulation",
        "elapsed_sec": timer.elapsed_sec,
        "skip_factorial": skip_factorial,
        "clean_sites_count": len(site_order),
        "factorial_scans_generated": n_factorial_scans,
        "site_metadata": site_meta,
        "clean_site_iqm_table": clean_iqm_table,
        "harmonization_report": harm_report,
        "deterministic_qc_report": det_report,
        "figures": figures,
        "checks": common.checks_to_dicts(checks),
    }
    common.write_json(out_root / "03_sites_summary.json", summary)
    print(
        f"[Step 04] Completed {len(site_order)} clean sites + {n_factorial_scans} factorial scans ({timer.elapsed_sec:.1f}s)"
    )
    for chk in checks:
        print(f"  [{chk.status.upper():4s}] {chk.name}: {chk.metric}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 04: Multi-site simulation & harmonization audit.")
    parser.add_argument("--out-dir", type=Path, default=common.DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker count.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: auto|cpu|cuda.")
    parser.add_argument("--skip-factorial", action="store_true", help="Only run clean site comparison.")
    parser.add_argument("--sites-catalog", type=Path, default=common.DEFAULT_SITES_CATALOG, help="Sites YAML.")
    args = parser.parse_args()
    run_step(
        out_dir=args.out_dir,
        workers=args.workers,
        device=args.device,
        skip_factorial=args.skip_factorial,
        sites_catalog=args.sites_catalog,
    )


if __name__ == "__main__":
    main()
