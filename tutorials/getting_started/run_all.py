"""Master runner executing Steps 01 -> 05 of the ghibli Getting Started tutorial.

Example usage:
    # Real MR-ART no-motion T1w scan (default if ../data/mrart exists)
    uv run --no-sync python tutorials/getting_started/run_all.py \
        --t1w ../data/mrart/bids/sub-000103/anat/sub-000103_acq-standard_T1w.nii.gz \
        --out-dir tutorial_output/mrart_sub000103 \
        --workers 4

    # Any clean 3D T1w NIfTI
    uv run --no-sync python tutorials/getting_started/run_all.py \
        --t1w /path/to/your_T1w.nii.gz \
        --out-dir tutorial_output/custom \
        --workers 4

    # Zero-data synthetic phantom smoke run
    uv run --no-sync python tutorials/getting_started/run_all.py \
        --use-phantom \
        --out-dir tutorial_output/phantom_demo
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def run_all(
    t1w: Path | None = None,
    json_sidecar: Path | None = None,
    brain_mask: Path | None = None,
    subject_id: str = "01",
    out_dir: Path = common.DEFAULT_OUT_DIR,
    workers: int = 4,
    device: str = "auto",
    use_phantom: bool = False,
    t1w_only: bool = False,
    full_4d: bool = False,
    skip_factorial: bool = False,
    embed_images: bool = False,
) -> dict[str, Any]:
    """Execute the full Getting Started verification workflow end-to-end."""
    step01 = importlib.import_module("01_stage_input")
    step02 = importlib.import_module("02_convert_modalities")
    step03 = importlib.import_module("03_artifact_dose_sweep")
    step04 = importlib.import_module("04_multisite_simulation")
    step05 = importlib.import_module("05_build_report")

    out_root = Path(out_dir).resolve()
    step01.run_step(
        t1w=t1w,
        json_sidecar=json_sidecar,
        brain_mask=brain_mask,
        subject_id=subject_id,
        out_dir=out_root,
        use_phantom=use_phantom,
    )
    step02.run_step(
        out_dir=out_root,
        workers=workers,
        device=device,
        full_4d=full_4d,
        brain_mask=brain_mask,
    )
    step03.run_step(
        out_dir=out_root,
        workers=workers,
        device=device,
        t1w_only=t1w_only,
    )
    step04.run_step(
        out_dir=out_root,
        workers=workers,
        device=device,
        skip_factorial=skip_factorial,
    )
    return dict(step05.run_step(out_dir=out_root, embed_images=embed_images))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete ghibli Getting Started tutorial & verification suite."
    )
    parser.add_argument("--t1w", type=Path, default=None, help="Path to any clean 3D T1w NIfTI.")
    parser.add_argument("--json", type=Path, default=None, help="Optional BIDS JSON sidecar.")
    parser.add_argument("--brain-mask", type=Path, default=None, help="Optional external brain mask.")
    parser.add_argument("--subject-id", type=str, default="01", help="Staged BIDS subject ID.")
    parser.add_argument("--out-dir", type=Path, default=common.DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker count.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: auto|cpu|cuda.")
    parser.add_argument("--use-phantom", action="store_true", help="Generate a synthetic phantom T1w.")
    parser.add_argument("--t1w-only", action="store_true", help="Skip 4D BOLD/DWI artifact sweeps.")
    parser.add_argument("--full-4d", action="store_true", help="Use full 60 TRs / 40 DWI dirs.")
    parser.add_argument("--skip-factorial", action="store_true", help="Skip Part 4b harmonization batch.")
    parser.add_argument("--embed-images", action="store_true", help="Base64-inline PNGs into report.html.")
    args = parser.parse_args()

    audit = run_all(
        t1w=args.t1w,
        json_sidecar=args.json,
        brain_mask=args.brain_mask,
        subject_id=args.subject_id,
        out_dir=args.out_dir,
        workers=args.workers,
        device=args.device,
        use_phantom=args.use_phantom,
        t1w_only=args.t1w_only,
        full_4d=args.full_4d,
        skip_factorial=args.skip_factorial,
        embed_images=args.embed_images,
    )
    print(f"\nDone! Open your browser report at: {audit['report_html_path']}")
    print(f"      (Markdown version: {audit['report_path']})")


if __name__ == "__main__":
    main()
