"""Step 05: Aggregate all tutorial diagnostics, visual contact sheets, and gap findings into report.md & report.html."""

from __future__ import annotations

import argparse
import base64
import html
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def _fmt_md_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No rows available._"
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def _badge_html(status: str) -> str:
    st = status.lower().strip()
    cls = st if st in ("pass", "warn", "fail", "info") else "info"
    return f'<span class="badge badge-{cls}">{html.escape(st.upper())}</span>'


def _fmt_html_table(
    headers: list[str],
    rows: list[list[Any]],
    status_col_idx: int | None = None,
) -> str:
    if not rows:
        return "<p><em>No rows available.</em></p>"
    ths = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    trs: list[str] = []
    for r in rows:
        tds: list[str] = []
        for idx, val in enumerate(r):
            if status_col_idx is not None and idx == status_col_idx:
                tds.append(f"<td>{_badge_html(str(val))}</td>")
            else:
                tds.append(f"<td>{html.escape(str(val))}</td>")
        trs.append("<tr>" + "".join(tds) + "</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{ths}</tr></thead><tbody>{"".join(trs)}</tbody></table></div>'


def _img_html(out_root: Path, rel_path: str, caption: str, embed_images: bool) -> str:
    full = out_root / rel_path
    if not full.exists():
        return ""
    if embed_images:
        mime = "image/gif" if full.suffix.lower() == ".gif" else "image/png"
        b64 = base64.b64encode(full.read_bytes()).decode("ascii")
        src = f"data:{mime};base64,{b64}"
    else:
        src = html.escape(rel_path)
    cap_esc = html.escape(caption)
    hint = "(animated GIF — click to open)" if full.suffix.lower() == ".gif" else "(click to inspect full resolution)"
    return (
        f'<figure class="report-fig">'
        f'<a href="{src}" target="_blank" title="Open full size in new tab">'
        f'<img src="{src}" alt="{cap_esc}" loading="lazy" />'
        f"</a>"
        f"<figcaption>{cap_esc} "
        f'<span class="fig-hint">{hint}</span>'
        f"</figcaption></figure>"
    )


_HTML_CSS = """
:root {
  --bg: #ffffff;
  --fg: #1f2328;
  --muted: #59636e;
  --border: #d1d9e0;
  --card: #f6f8fa;
  --accent: #0969da;
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background: var(--bg);
  color: var(--fg);
  line-height: 1.55;
  margin: 0;
  padding: 28px 24px 64px;
}
main { max-width: 1280px; margin: 0 auto; }
h1 { margin-top: 0; font-size: 1.85rem; border-bottom: 1px solid var(--border); padding-bottom: 10px; }
h2 { margin-top: 36px; font-size: 1.35rem; border-bottom: 1px solid var(--border); padding-bottom: 6px; }
h3 { margin-top: 22px; font-size: 1.08rem; }
code { background: var(--card); padding: 2px 5px; border-radius: 4px; font-size: 0.9em; }
nav.toc {
  display: flex; flex-wrap: wrap; gap: 10px;
  background: var(--card); border: 1px solid var(--border); border-radius: 8px;
  padding: 10px 14px; margin: 16px 0 24px; font-size: 0.92rem;
}
nav.toc a { color: var(--accent); text-decoration: none; font-weight: 500; }
nav.toc a:hover { text-decoration: underline; }
.meta-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px; margin: 16px 0 24px;
}
.meta-card {
  background: var(--card); border: 1px solid var(--border); border-radius: 8px;
  padding: 12px 14px;
}
.meta-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }
.meta-value { font-size: 1.02rem; font-weight: 600; margin-top: 4px; word-break: break-word; }
.table-wrap { overflow-x: auto; margin: 12px 0 20px; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
th, td { border: 1px solid var(--border); padding: 7px 10px; text-align: left; vertical-align: top; }
th { background: var(--card); font-weight: 600; white-space: nowrap; }
tr:nth-child(even) td { background: #fbfcfd; }
.badge {
  display: inline-block; padding: 2px 8px; border-radius: 999px;
  font-size: 0.75rem; font-weight: 700; letter-spacing: 0.03em;
}
.badge-pass { background: #dafbe1; color: #116329; }
.badge-warn { background: #fff8c5; color: #7d4e00; }
.badge-fail { background: #ffebe9; color: #a40e26; }
.badge-info { background: #ddf4ff; color: #0550ae; }
figure.report-fig {
  margin: 18px 0 26px; padding: 12px;
  background: var(--card); border: 1px solid var(--border); border-radius: 8px;
  text-align: center;
}
figure.report-fig img {
  max-width: 100%; height: auto; border-radius: 4px; cursor: zoom-in;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
}
figure.report-fig figcaption { margin-top: 8px; font-size: 0.9rem; color: var(--fg); font-weight: 500; }
.fig-hint { color: var(--muted); font-weight: 400; font-size: 0.82rem; }
.callout {
  border-left: 4px solid #bf8700; background: #fff8c5; padding: 10px 14px;
  border-radius: 0 6px 6px 0; margin: 12px 0;
}
.guide-box {
  border-left: 4px solid var(--accent); background: #f0f6fc; padding: 12px 16px;
  border-radius: 0 6px 6px 0; margin: 14px 0 18px; font-size: 0.93rem;
}
.guide-box p { margin: 6px 0; }
.guide-box ul { margin: 6px 0 6px 20px; padding: 0; }
"""


def _build_html_report(
    out_root: Path,
    s1: dict[str, Any],
    s2: dict[str, Any],
    s3: dict[str, Any],
    s4: dict[str, Any],
    all_checks: list[dict[str, Any]],
    flagged_gaps: list[dict[str, Any]],
    counts: dict[str, int],
    total_sec: float,
    embed_images: bool = False,
) -> str:
    """Render a clean, browser-ready HTML report with clickable figures and status badges."""
    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8" />',
        '<meta name="viewport" content="width=device-width, initial-scale=1" />',
        "<title>sim-data Getting Started Verification Report</title>",
        f"<style>{_HTML_CSS}</style>",
        "</head><body><main>",
        "<h1><code>sim-data</code> Getting Started Verification Report</h1>",
        '<nav class="toc">'
        '<a href="#gaps">Flagged Gaps</a>'
        '<a href="#checks">All Checks</a>'
        '<a href="#step1">Step 01: Input Priors</a>'
        '<a href="#step2">Step 02: Modalities</a>'
        '<a href="#step3">Step 03: Artifacts (17/17)</a>'
        '<a href="#step4">Step 04: Multi-Site &amp; Harmonization</a>'
        "</nav>",
        '<div class="meta-grid">',
        f'<div class="meta-card"><div class="meta-label">Input T1w</div>'
        f'<div class="meta-value">{html.escape(Path(str(s1.get("source_t1w", "N/A"))).name)}</div></div>',
        f'<div class="meta-card"><div class="meta-label">Geometry (RAS+)</div>'
        f'<div class="meta-value">{html.escape(str(s1.get("ras_shape")))} @ {html.escape(str(s1.get("zooms_mm")))} mm</div></div>',
        f'<div class="meta-card"><div class="meta-label">Total Wall Time</div>'
        f'<div class="meta-value">{total_sec:.1f} s</div></div>',
        f'<div class="meta-card"><div class="meta-label">Automated Checks</div>'
        f'<div class="meta-value">{_badge_html("pass")} {counts["pass"]} &nbsp; '
        f'{_badge_html("warn")} {counts["warn"]} &nbsp; '
        f'{_badge_html("fail")} {counts["fail"]}</div></div>',
        "</div>",
        f'<p><strong>Full input path:</strong> <code>{html.escape(str(s1.get("source_t1w", "N/A")))}</code> '
        f'({html.escape(str(s1.get("source_kind", "unknown")))}, native orientation '
        f'<code>{html.escape(str(s1.get("orig_axcodes", "?")))}</code> &rarr; <code>RAS+</code>)</p>',
        '<div class="guide-box">'
        "<p><strong>About this report:</strong> Starting from a single clean 3D T1w scan, this tutorial exercises every major "
        "working path in <code>sim-data</code>: (1) RAS+ canonicalization &amp; anatomical priors, (2) internal brain extraction "
        "and sequential modality completion into <code>T2w</code>, <code>FLAIR</code>, lower-resolution skull-free 4D <code>bold</code>, "
        "and 4D <code>dwi</code>, (3) coupled Common-Random-Numbers (CRN) dose-response trajectories across all 17/17 registered "
        "artifacts at Low (0.25), Mid (0.55), and High (0.85) severity with zero-leakage <code>acq_blinded</code> BIDS naming, "
        "and (4) multi-site scanner simulation &amp; ComBat/CovBat harmonization auditing.</p>"
        "</div>",
        '<h2 id="gaps">Executive Summary: Flagged Gaps &amp; Refinement Candidates</h2>',
    ]

    if flagged_gaps:
        gap_rows = [
            [g["status"], g["step"], g["name"], g["metric"], g["detail"]]
            for g in flagged_gaps
        ]
        parts.append(
            _fmt_html_table(
                ["Status", "Step", "Check", "Observed Metric", "Interpretation"],
                gap_rows,
                status_col_idx=0,
            )
        )
    else:
        parts.append("<p><em>No automated warnings or failures were triggered.</em></p>")

    parts.append('<h2 id="checks">All Automated Verification Checks</h2>')
    chk_rows = [[c["status"], c["step"], c["name"], c["metric"]] for c in all_checks]
    parts.append(_fmt_html_table(["Status", "Step", "Check", "Metric"], chk_rows, status_col_idx=0))

    # --- Step 01 ---
    if s1:
        p = s1.get("priors", {})
        parts.append('<h2 id="step1">Step 01 &mdash; Input Staging &amp; Anatomical Priors</h2>')
        parts.append(
            '<div class="guide-box"><p><strong>What this step verifies:</strong> Any standalone 3D T1w NIfTI is staged into a '
            "clean single-subject BIDS directory, reoriented to canonical <code>RAS+</code>, and inspected for phase/slice encoding "
            "axes (falling back to DICOM orientation metadata when BIDS <code>PhaseEncodingDirection</code> is absent) plus native "
            "scanner background noise (&sigma;<sub>0</sub>, used by <code>excess_variance</code> noise calibration). "
            "The top row shows the foreground envelope (cyan); the bottom row applies a log-gamma <strong>BG-Boost</strong> stretch "
            "to expose faint air noise, ghosting, or pre-existing defacing cuts.</p></div>"
        )
        parts.append(
            f"<ul><li>Resolved RAS+ axes: <code>{html.escape(str(s1.get('encoding_axes_ras')))}</code> "
            f"(explicit PE={s1.get('sidecar_has_pe_dir')}, SL={s1.get('sidecar_has_sl_dir')})</li>"
            f"<li>Foreground mask: <code>{p.get('fg_fraction')}</code> of FOV, bbox fraction "
            f"<code>{p.get('bbox_frac')}</code>, median=<code>{p.get('fg_median')}</code>, "
            f"p95=<code>{p.get('fg_p95')}</code>, native noise &sigma;<sub>0</sub>=<code>{p.get('native_noise_sigma0')}</code></li></ul>"
        )
        fig1 = s1.get("figures", {}).get("input_priors")
        if fig1:
            parts.append(_img_html(out_root, fig1, "Step 01: Staged Input T1w & Anatomical Priors", embed_images))

    # --- Step 02 ---
    if s2:
        parts.append('<h2 id="step2">Step 02 &mdash; Modality Conversion &amp; Completion (T1w &rarr; T2w, FLAIR, BOLD, DWI)</h2>')
        parts.append(
            '<div class="guide-box"><p><strong>What this step verifies:</strong> First, standalone <code>convert-contrast</code> '
            f"(brain extraction method: <code>{html.escape(str(s2.get('brain_mask_method', 'auto')))}</code>) separates intracranial "
            "brain (ICV) from cortical skull bone and extracranial scalp/neck, then maps N4-corrected T1w intensities through "
            "Bloch-anchored PCHIP splines into <code>T2w</code> (bright CSF &gt; GM &gt; WM), CSF-nulled <code>FLAIR</code> "
            "(bright GM &gt; WM &gt;&gt; dark CSF), and skull-stripped <code>bold_s0</code> / <code>dwi_b0</code>. Next, "
            "<code>synthesize-cohort --preserve-real</code> keeps the input <code>T1w</code> bit-for-bit untouched while completing "
            "high-resolution <code>T2w</code>/<code>FLAIR</code> (1 mm) and lower-resolution, brain-cropped 4D <code>bold</code> (3 mm) "
            "and multi-shell 4D <code>dwi</code> (2.5 mm). The BOLD carpet plot groups voxels into <strong>GM</strong>, "
            "<strong>WM</strong>, and <strong>CSF</strong> bands to help spot global vs. tissue-specific signal fluctuations.</p></div>"
        )
        mod_rows = []
        for mod, shp in s2.get("modality_shapes", {}).items():
            zm = s2.get("modality_zooms_mm", {}).get(mod, [])
            iqms = s2.get("baseline_iqms_by_modality", {}).get(mod, {})
            top_k = list(iqms.items())[:5]
            mod_rows.append([mod, shp, zm, ", ".join(f"{k}={v}" for k, v in top_k)])
        parts.append(_fmt_html_table(["Modality", "Shape", "Zooms (mm)", "Sample Blind IQMs"], mod_rows))
        for label, key in [
            ("Standalone Contrast Conversion & Compartments", "contrast_compartments"),
            ("All 5 Completed Modalities Overview", "all_modalities_overview"),
            ("Synthesized BOLD fMRIPrep-Style Carpet & tSNR", "bold_carpet"),
            ("Synthesized DWI Shell & Slice Summary", "dwi_summary"),
        ]:
            rel = s2.get("figures", {}).get(key)
            if rel:
                parts.append(f"<h3>{html.escape(label)}</h3>")
                parts.append(_img_html(out_root, rel, label, embed_images))

    # --- Step 03 ---
    if s3:
        parts.append('<h2 id="step3">Step 03 &mdash; Individual Artifacts Across Low / Mid / High Severity</h2>')
        parts.append(
            '<div class="guide-box">'
            "<p><strong>How to read this section &amp; IQM sensitivity rankings:</strong> Each artifact is injected individually "
            "along a coupled Common-Random-Numbers (CRN) trajectory at <code>[0.0, 0.25, 0.55, 0.85]</code> under zero-leakage "
            "<code>acq_blinded</code> BIDS naming, and blind NiPreps-aligned IQMs are extracted purely from the output NIfTIs.</p>"
            "<ul>"
            "<li><strong>Top IQM ranking:</strong> Metrics are ranked first by monotonic rank correlation "
            "(<code>|Spearman &rho;|</code> across dose steps), breaking ties by standardized effect size "
            "(<code>|Max Cohen's d|</code> at High severity) and then cross-artifact <code>Specificity</code> "
            "(<code>|&rho;_self| / max(|&rho;_other|)</code>; values &gt; 1.0 indicate specificity to that defect).</li>"
            "<li><strong>MDS / LoD (Minimum Detectable Severity):</strong> Lowest dose where the mean IQM shift exceeds "
            "2.0 SD from baseline (<code>None</code> on small single-subject runs when overall SD is wide).</li>"
            "<li><strong>Common IQM directionality:</strong> "
            "<em>Higher = better quality:</em> <code>snr_*</code>, <code>snrd_total</code>, <code>cnr</code>, <code>fber</code>, "
            "<code>tsnr</code>, <code>raw_neighbor_corr</code>, <code>raw_b0_snr</code>, <code>snr_shell_high</code>. "
            "<em>Higher = worse / more artifact:</em> <code>efc</code> (blur/ghosting), <code>cjv</code> (poor GM/WM contrast), "
            "<code>fwhm_*</code> (blurriness), <code>inu_range</code> (bias shading), <code>qi1</code>/<code>qi2</code> (background artifacts), "
            "<code>gsr_*</code> (Nyquist/ghosting), <code>fd_mean</code>/<code>dvars_*</code>/<code>aqi</code> (4D motion), "
            "<code>raw_num_bad_slices</code> (DWI dropout), <code>kspace_spike_ratio</code> (RF spikes).</li>"
            "<li><strong>Visual strips:</strong> Columns progress from Baseline &rarr; Low &rarr; Mid &rarr; High (fixed baseline window), "
            "followed by <strong>BG-Boost</strong> (highlights faint ghosts/spikes/noise in air) and <strong>|High &minus; Base| Diff</strong> "
            "(pinpoints exact voxelwise changes). Full-reference <code>SSIM</code> (3-plane average) and <code>PSNR</code> are shown below each tile.</li>"
            "</ul></div>"
        )
        parts.append(
            f"<p>Exercised <strong>{len(s3.get('tested_artifacts', []))}/{s3.get('registry_total', 17)}</strong> "
            f"registered artifacts across doses <code>{html.escape(str(s3.get('dose_levels')))}</code> "
            f"(<code>{s3.get('t1w_scans_generated')}</code> T1w + <code>{s3.get('fourd_scans_generated')}</code> 4D scans).</p>"
        )
        for mod_key, sens in s3.get("sensitivity_reports", {}).items():
            parts.append(f"<h3>IQM Dose-Response Sensitivity ({html.escape(mod_key)})</h3>")
            s_rows = []
            ssims_mod = s3.get("ssim_trajectories", {}).get(mod_key, {})
            for art, top_list in sens.get("top_iqms_per_artifact", {}).items():
                best = top_list[0] if top_list else {}
                ssim_str = " -> ".join(f"{v:.3f}" for v in ssims_mod.get(art, []))
                s_rows.append([
                    art,
                    ssim_str,
                    best.get("iqm", "-"),
                    f"{best.get('mean_spearman_rho', 0.0):+.3f}",
                    f"{best.get('max_effect_cohens_d', 0.0):+.2f}",
                    str(best.get("min_detectable_severity", "-")),
                    f"{best.get('specificity_ratio', 0.0):.2f}",
                ])
            parts.append(
                _fmt_html_table(
                    ["Artifact", "SSIM (Low->Mid->High)", "Top IQM", "Spearman rho", "Max Cohen's d", "MDS/LoD", "Specificity"],
                    s_rows,
                )
            )
            if sens.get("blind_spot_artifacts"):
                parts.append(
                    f'<div class="callout"><strong>IQM Blind Spots ({html.escape(mod_key)}):</strong> '
                    f"<code>{html.escape(str(sens['blind_spot_artifacts']))}</code></div>"
                )

        for label, key in [
            ("T1w Artifact Contact Sheet (Part 1/2)", "t1w_artifacts_part1"),
            ("T1w Artifact Contact Sheet (Part 2/2)", "t1w_artifacts_part2"),
            ("Animated T1w motion_3d Flipbook (Baseline -> Low -> Mid -> High)", "t1w_motion3d_gif"),
            ("4D / EPI / DWI Artifact Contact Sheet", "artifacts_4d_grid"),
            ("Animated 4D BOLD motion_4d Movie (Synced Across TRs)", "bold_motion4d_gif"),
            ("Animated 4D DWI dwi_slice_dropout Sweep (High-b Volumes)", "dwi_dropout_gif"),
            ("T1w IQM Sensitivity Heatmap", "t1w_sensitivity_heatmap"),
            ("T1w Dose-Response IQM Curves", "t1w_dose_curves"),
        ]:
            rel = s3.get("figures", {}).get(key)
            if rel:
                parts.append(f"<h3>{html.escape(label)}</h3>")
                parts.append(_img_html(out_root, rel, label, embed_images))

    # --- Step 04 ---
    if s4:
        parts.append('<h2 id="step4">Step 04 &mdash; Multi-Site Simulation &amp; Harmonization Audit</h2>')
        parts.append(
            '<div class="guide-box"><p><strong>What this step verifies:</strong> The exact same baseline subject is imaged across '
            "<code>NativeSite</code> (passthrough) plus 4 curated scanner profiles (<code>Siemens_3T_32ch</code>, "
            "<code>GE_15T_8ch</code>, <code>Philips_3T_16ch</code>, <code>Siemens_3T_64ch_HighAccel</code>) differing in contrast "
            "&gamma;, gain, PSF sharpness, coil B<sub>1</sub> bias, coil count, and parallel-imaging g-factor noise. "
            "Part 4a isolates pure scanner physics on clean scans (shared window vs. per-site window vs. BG-boost &amp; histograms). "
            "Part 4b runs a 24-scan multi-site factorial batch through <strong>ComBat</strong> (location/scale adjustment) and "
            "<strong>CovBat</strong> (location/scale + covariance harmonization), auditing site separability (<code>ANOVA F</code>, "
            "<code>site_rf_auc</code> &rarr; ~0.50 chance) while preserving defect signal.</p></div>"
        )
        tbl = s4.get("clean_site_iqm_table", [])
        if tbl:
            headers = list(tbl[0].keys())
            rows = [[row.get(h, "-") for h in headers] for row in tbl]
            parts.append("<h3>Clean Same-Subject IQM Shifts Across Curated Scanner Profiles</h3>")
            parts.append(_fmt_html_table(headers, rows))
        rel_sites = s4.get("figures", {}).get("sites_comparison")
        if rel_sites:
            parts.append(_img_html(out_root, rel_sites, "Step 04a: Multi-Site Comparison & Intensity Histograms", embed_images))

        harm = s4.get("harmonization_report")
        if harm:
            parts.append("<h3>ComBat vs. CovBat Smoke Audit (Single-Subject Factorial Batch)</h3>")
            h_rows = []
            for metric in [
                "mean_anova_f",
                "mean_site_eta2",
                "covariance_frobenius_dist",
                "site_linear_auc",
                "site_rf_auc",
                "defect_burden_r2",
            ]:
                if metric in harm.get("raw", {}):
                    h_rows.append([
                        metric,
                        harm["raw"].get(metric, "-"),
                        harm["combat"].get(metric, "-"),
                        harm["covbat"].get(metric, "-"),
                    ])
            parts.append(_fmt_html_table(["Metric", "Raw", "ComBat", "CovBat"], h_rows))
            rel_pca = s4.get("figures", {}).get("site_pca_harmonization")
            if rel_pca:
                parts.append(_img_html(out_root, rel_pca, "Step 04b: Multi-Site IQM Space (Raw vs. ComBat vs. CovBat)", embed_images))

        det = s4.get("deterministic_qc_report")
        if det:
            parts.append(
                f"<h3>Deterministic QC Baseline</h3>"
                f"<p>Macro-F1=<code>{det.get('macro_f1')}</code>, "
                f"Kappa=<code>{det.get('cohen_kappa')}</code>, "
                f"QWK=<code>{det.get('quadratic_weighted_kappa')}</code> (n={det.get('n_scans')})</p>"
            )

    parts.append("</main></body></html>")
    return "\n".join(parts) + "\n"


def run_step(
    out_dir: Path = common.DEFAULT_OUT_DIR,
    embed_images: bool = False,
) -> dict[str, Any]:
    """Assemble `<out_dir>/report.md`, `<out_dir>/report.html`, and `<out_dir>/verification_audit.json`."""
    out_root = Path(out_dir).resolve()
    s1 = common.read_json(out_root / "00_seed_diagnostics.json") if (out_root / "00_seed_diagnostics.json").exists() else {}
    s2 = common.read_json(out_root / "01_modalities_summary.json") if (out_root / "01_modalities_summary.json").exists() else {}
    s3 = common.read_json(out_root / "02_artifacts_summary.json") if (out_root / "02_artifacts_summary.json").exists() else {}
    s4 = common.read_json(out_root / "03_sites_summary.json") if (out_root / "03_sites_summary.json").exists() else {}

    all_checks: list[dict[str, Any]] = []
    for block in (s1, s2, s3, s4):
        all_checks.extend(block.get("checks", []))

    counts = {k: sum(1 for c in all_checks if c.get("status") == k) for k in ("pass", "warn", "fail", "info")}
    flagged_gaps = [c for c in all_checks if c.get("status") in ("warn", "fail")]
    total_sec = round(sum(float(b.get("elapsed_sec", 0.0)) for b in (s1, s2, s3, s4)), 2)

    lines: list[str] = [
        "# `sim-data` Getting Started Verification Report",
        "",
        f"- **Input T1w**: `{s1.get('source_t1w', 'N/A')}` (`{s1.get('source_kind', 'unknown')}`)",
        f"- **Geometry**: shape `{s1.get('ras_shape')}`, zooms `{s1.get('zooms_mm')} mm`, native orientation `{s1.get('orig_axcodes')}` → `RAS+`",
        f"- **Total wall time**: `{total_sec:.1f} s`",
        f"- **Automated Checks**: **{counts['pass']} pass**, **{counts['warn']} warn**, **{counts['fail']} fail**, {counts['info']} info",
        "",
        "---",
        "",
        "## Executive Summary: Flagged Gaps & Refinement Candidates",
        "",
    ]

    if flagged_gaps:
        gap_rows = [
            [f"`{g['status'].upper()}`", f"`{g['step']}`", f"`{g['name']}`", g["metric"], g["detail"]]
            for g in flagged_gaps
        ]
        lines.append(_fmt_md_table(["Status", "Step", "Check", "Observed Metric", "Interpretation"], gap_rows))
    else:
        lines.append("_No automated warnings or failures were triggered._")

    lines.extend(["", "## All Automated Verification Checks", ""])
    chk_rows = [
        [f"`{c['status'].upper()}`", f"`{c['step']}`", f"`{c['name']}`", c["metric"]]
        for c in all_checks
    ]
    lines.append(_fmt_md_table(["Status", "Step", "Check", "Metric"], chk_rows))

    # --- Step 01 ---
    if s1:
        lines.extend(["", "---", "", "## Step 01 — Input Staging & Anatomical Priors", ""])
        p = s1.get("priors", {})
        lines.append(
            f"- Resolved RAS+ axes: `{s1.get('encoding_axes_ras')}` "
            f"(explicit PE={s1.get('sidecar_has_pe_dir')}, SL={s1.get('sidecar_has_sl_dir')})"
        )
        lines.append(
            f"- Foreground mask: `{p.get('fg_fraction')}` of FOV, bbox fraction `{p.get('bbox_frac')}`, "
            f"median=`{p.get('fg_median')}`, p95=`{p.get('fg_p95')}`, native noise $\\sigma_0$=`{p.get('native_noise_sigma0')}`"
        )
        fig1 = s1.get("figures", {}).get("input_priors")
        if fig1:
            lines.extend(["", f"![Input Priors]({fig1})"])

    # --- Step 02 ---
    if s2:
        lines.extend(["", "---", "", "## Step 02 — Modality Conversion & Completion (`T1w → T2w, FLAIR, BOLD, DWI`)", ""])
        mod_rows = []
        for mod, shp in s2.get("modality_shapes", {}).items():
            zm = s2.get("modality_zooms_mm", {}).get(mod, [])
            iqms = s2.get("baseline_iqms_by_modality", {}).get(mod, {})
            top_k = list(iqms.items())[:5]
            iqm_str = ", ".join(f"{k}={v}" for k, v in top_k)
            mod_rows.append([f"`{mod}`", f"`{shp}`", f"`{zm}`", iqm_str])
        lines.append(_fmt_md_table(["Modality", "Shape", "Zooms (mm)", "Sample Blind IQMs"], mod_rows))
        for label, key in [
            ("Standalone Contrast Conversion & Compartments", "contrast_compartments"),
            ("All 5 Completed Modalities Overview", "all_modalities_overview"),
            ("Synthesized BOLD fMRIPrep-Style Carpet & tSNR", "bold_carpet"),
            ("Synthesized DWI Shell & Slice Summary", "dwi_summary"),
        ]:
            rel = s2.get("figures", {}).get(key)
            if rel:
                lines.extend(["", f"### {label}", "", f"![{label}]({rel})"])

    # --- Step 03 ---
    if s3:
        lines.extend(["", "---", "", "## Step 03 — Individual Artifacts Across Low / Mid / High Severity", ""])
        lines.append(
            f"- Exercised **{len(s3.get('tested_artifacts', []))}/{s3.get('registry_total', 17)}** registered artifacts "
            f"across doses `{s3.get('dose_levels')}` (`{s3.get('t1w_scans_generated')}` T1w + `{s3.get('fourd_scans_generated')}` 4D scans)."
        )
        for mod_key, sens in s3.get("sensitivity_reports", {}).items():
            lines.extend(["", f"### IQM Dose-Response Sensitivity ({mod_key})", ""])
            s_rows = []
            ssims_mod = s3.get("ssim_trajectories", {}).get(mod_key, {})
            for art, top_list in sens.get("top_iqms_per_artifact", {}).items():
                best = top_list[0] if top_list else {}
                ssim_str = " → ".join(f"{v:.3f}" for v in ssims_mod.get(art, []))
                s_rows.append([
                    f"`{art}`",
                    ssim_str,
                    f"`{best.get('iqm', '-')}`",
                    f"{best.get('mean_spearman_rho', 0.0):+.3f}",
                    f"{best.get('max_effect_cohens_d', 0.0):+.2f}",
                    str(best.get("min_detectable_severity", "-")),
                    f"{best.get('specificity_ratio', 0.0):.2f}",
                ])
            lines.append(
                _fmt_md_table(
                    ["Artifact", "SSIM (Low→Mid→High)", "Top IQM", "Spearman ρ", "Max Cohen's d", "MDS/LoD", "Specificity"],
                    s_rows,
                )
            )
            if sens.get("blind_spot_artifacts"):
                lines.append(f"\n> **IQM Blind Spots ({mod_key}):** `{sens['blind_spot_artifacts']}`")

        for label, key in [
            ("T1w Artifact Contact Sheet (Part 1/2)", "t1w_artifacts_part1"),
            ("T1w Artifact Contact Sheet (Part 2/2)", "t1w_artifacts_part2"),
            ("Animated T1w motion_3d Flipbook", "t1w_motion3d_gif"),
            ("4D / EPI / DWI Artifact Contact Sheet", "artifacts_4d_grid"),
            ("Animated 4D BOLD motion_4d Movie", "bold_motion4d_gif"),
            ("Animated 4D DWI dwi_slice_dropout Sweep", "dwi_dropout_gif"),
            ("T1w IQM Sensitivity Heatmap", "t1w_sensitivity_heatmap"),
            ("T1w Dose-Response IQM Curves", "t1w_dose_curves"),
        ]:
            rel = s3.get("figures", {}).get(key)
            if rel:
                lines.extend(["", f"### {label}", "", f"![{label}]({rel})"])

    # --- Step 04 ---
    if s4:
        lines.extend(["", "---", "", "## Step 04 — Multi-Site Simulation & Harmonization Audit", ""])
        tbl = s4.get("clean_site_iqm_table", [])
        if tbl:
            headers = list(tbl[0].keys())
            rows = [[row.get(h, "-") for h in headers] for row in tbl]
            lines.append("### Clean Same-Subject IQM Shifts Across Curated Scanner Profiles")
            lines.append("")
            lines.append(_fmt_md_table(headers, rows))
        rel_sites = s4.get("figures", {}).get("sites_comparison")
        if rel_sites:
            lines.extend(["", f"![Multi-Site Comparison]({rel_sites})"])

        harm = s4.get("harmonization_report")
        if harm:
            lines.extend(["", "### ComBat vs. CovBat Smoke Audit (Single-Subject Factorial Batch)", ""])
            h_rows = []
            for metric in [
                "mean_anova_f",
                "mean_site_eta2",
                "covariance_frobenius_dist",
                "site_linear_auc",
                "site_rf_auc",
                "defect_burden_r2",
            ]:
                if metric in harm.get("raw", {}):
                    h_rows.append([
                        f"`{metric}`",
                        harm["raw"].get(metric, "-"),
                        harm["combat"].get(metric, "-"),
                        harm["covbat"].get(metric, "-"),
                    ])
            lines.append(_fmt_md_table(["Metric", "Raw", "ComBat", "CovBat"], h_rows))
            rel_pca = s4.get("figures", {}).get("site_pca_harmonization")
            if rel_pca:
                lines.extend(["", f"![ComBat vs CovBat PCA]({rel_pca})"])

        det = s4.get("deterministic_qc_report")
        if det:
            lines.extend([
                "",
                f"### Deterministic QC Baseline: Macro-F1=`{det.get('macro_f1')}`, "
                f"Kappa=`{det.get('cohen_kappa')}`, QWK=`{det.get('quadratic_weighted_kappa')}` (n={det.get('n_scans')})",
            ])

    report_md_path = out_root / "report.md"
    report_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report_html_path = out_root / "report.html"
    report_html_path.write_text(
        _build_html_report(
            out_root=out_root,
            s1=s1,
            s2=s2,
            s3=s3,
            s4=s4,
            all_checks=all_checks,
            flagged_gaps=flagged_gaps,
            counts=counts,
            total_sec=total_sec,
            embed_images=embed_images,
        ),
        encoding="utf-8",
    )

    audit = {
        "report_path": str(report_md_path),
        "report_html_path": str(report_html_path),
        "total_elapsed_sec": total_sec,
        "check_counts": counts,
        "flagged_gaps": flagged_gaps,
        "all_checks": all_checks,
        "step_timings_sec": {
            "01_stage_input": s1.get("elapsed_sec"),
            "02_convert_modalities": s2.get("elapsed_sec"),
            "03_artifact_dose_sweep": s3.get("elapsed_sec"),
            "04_multisite_simulation": s4.get("elapsed_sec"),
        },
    }
    audit_path = out_root / "verification_audit.json"
    common.write_json(audit_path, audit)
    print(
        f"[Step 05] Wrote report -> {report_html_path} (and {report_md_path}) "
        f"({counts['pass']} pass, {counts['warn']} warn, {counts['fail']} fail)"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 05: Build consolidated tutorial verification report (HTML + Markdown)."
    )
    parser.add_argument("--out-dir", type=Path, default=common.DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument(
        "--embed-images",
        action="store_true",
        help="Base64-inline all PNG figures inside report.html for single-file portability.",
    )
    args = parser.parse_args()
    run_step(out_dir=args.out_dir, embed_images=args.embed_images)


if __name__ == "__main__":
    main()
