"""Physics-ordered simulation pipeline, recipe samplers, and ground-truth labeling."""

from sim_data.pipeline.engine import SimulationResult, run_simulation_plan, simulate_single_recipe
from sim_data.pipeline.labels import compute_fidelity_metrics, compute_qc_targets, write_manifest
from sim_data.pipeline.sampler import ScanRecipe, build_recipes

__all__ = [
    "ScanRecipe",
    "SimulationResult",
    "build_recipes",
    "compute_fidelity_metrics",
    "compute_qc_targets",
    "run_simulation_plan",
    "simulate_single_recipe",
    "write_manifest",
]
