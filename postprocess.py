"""Standalone post-processing script — runs evaluation and plotting from saved params.

Use this to re-generate the full-field and near-hole plots without re-training.
Requires that training has already been run at least once so that either
``results/best_params.pkl`` or ``results/final_params.pkl`` exists.

Usage
-----
    # Use best-loss checkpoint (recommended):
    python postprocess.py

    # Use final-epoch checkpoint instead:
    python postprocess.py --checkpoint final

    # Custom results directory:
    python postprocess.py --results-dir /path/to/results
"""

import argparse
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import Config
from network import build_model
from train import load_params
from evaluate import evaluate_on_grid, evaluate_near_hole, compute_summary_metrics
from sampler import get_batch
import jax
from visualize import (
    plot_fields,
    plot_hole_zoom,
    plot_loss_history,
    plot_principal_fields,
    plot_deformed_fields,
    plot_sampling_points,
)


def postprocess(results_dir: str, checkpoint: str = "best") -> None:
    """Load saved params and regenerate all evaluation outputs.

    Parameters
    ----------
    results_dir : str
        Directory containing ``best_params.pkl`` / ``final_params.pkl``
        and where output plots will be written.
    checkpoint : str
        ``"best"`` (default) to load ``best_params.pkl``,
        ``"final"`` to load ``final_params.pkl``.
    """
    fname = "best_params.pkl" if checkpoint == "best" else "final_params.pkl"
    params_path = os.path.join(results_dir, fname)

    if not os.path.exists(params_path):
        # Fall back to the other checkpoint if the requested one is missing
        alt = "final_params.pkl" if checkpoint == "best" else "best_params.pkl"
        alt_path = os.path.join(results_dir, alt)
        if os.path.exists(alt_path):
            print(f"  ⚠  '{fname}' not found — falling back to '{alt}'")
            params_path = alt_path
        else:
            raise FileNotFoundError(
                f"No checkpoint found in '{results_dir}'. "
                "Run main.py first to train the model."
            )

    print(f"\nLoading params from  '{params_path}' …")
    params = load_params(params_path)

    cfg = Config()
    cfg.training.save_dir = results_dir

    model = build_model(
        cfg.network,
        eta_max=cfg.problem.eta_max,
        hole_xi_c=cfg.problem.hole_xi_c,
        hole_eta_c=cfg.problem.hole_eta_c,
        hole_rc=cfg.problem.hole_rc,
    )

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    print("Evaluating on default full and near-hole grids …")
    results = evaluate_on_grid(params, model, cfg.problem, cfg.network)
    results_zoom = evaluate_near_hole(params, model, cfg.problem, cfg.network)
    metrics = compute_summary_metrics(results)
    print(f"  max sigma_xx = {metrics['sxx_max']:.4e} {cfg.problem.stress_unit}")
    print(f"  max |tau_xy| = {metrics['txy_abs_max']:.4e} {cfg.problem.stress_unit}")

    # ------------------------------------------------------------------
    # Visualise
    # ------------------------------------------------------------------
    print("\nSaving plots …")

    loss_path = os.path.join(results_dir, "loss_history.npz")
    if os.path.exists(loss_path):
        import numpy as np
        raw = np.load(loss_path)
        history = {k: raw[k].tolist() for k in raw.files}
        plot_loss_history(history, results_dir)
    else:
        print("  (loss_history.npz not found — skipping loss plot)")

    plot_fields(results, results_dir,
                length_unit=cfg.problem.length_unit,
                stress_unit=cfg.problem.stress_unit)
    plot_principal_fields(results, results_dir,
                          length_unit=cfg.problem.length_unit,
                          stress_unit=cfg.problem.stress_unit)
    plot_deformed_fields(results, results_dir,
                         deformation_scale=cfg.plotting.deformation_scale,
                         length_unit=cfg.problem.length_unit,
                         stress_unit=cfg.problem.stress_unit)
    plot_hole_zoom(results_zoom, results_dir,
                   length_unit=cfg.problem.length_unit,
                   stress_unit=cfg.problem.stress_unit)

    batch = get_batch(cfg.problem, cfg.training, jax.random.PRNGKey(cfg.training.seed))
    plot_sampling_points(batch, cfg.problem, results_dir,
                         length_unit=cfg.problem.length_unit)

    print(f"\n✓  All results saved to  '{results_dir}/'")


def main():
    parser = argparse.ArgumentParser(
        description="Re-run post-processing from a saved PINN checkpoint."
    )
    parser.add_argument(
        "--checkpoint", choices=["best", "final"], default="best",
        help="Which checkpoint to load: 'best' (lowest training loss) or "
             "'final' (last epoch). Default: best",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Path to the results directory (default: ./results/ relative to this script)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir or os.path.join(os.path.dirname(__file__), "results")
    postprocess(results_dir, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
