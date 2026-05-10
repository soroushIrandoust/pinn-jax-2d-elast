"""Entry point for the plate-with-hole elasticity PINN using JAX + Flax.

Problem
-------
    Domain     :  [0, L] × [0, H] with L = 10 000 mm, H = 1 000 mm
    Hole       :  centred at (L/2, H/2) with radius 100 mm
    Material   :  plane stress, E = 20 000 MPa, ν = 0.3
    BCs        :
        Left  (x = 0)   :  u = v = 0  (hard-enforced)
        Right (x = L)   :  σ_xx = σ₀ = 2 MPa, τ_xy = 0
        Top / Bottom    :  σ_yy = 0, τ_xy = 0
        Hole surface    :  σ·n = 0
        Midline y=H/2   :  v = 0  (soft symmetry penalty)

Usage
-----
    python main.py
"""

import os
import sys
import io
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import jax

from config import Config
from train import train
from evaluate import evaluate_on_grid, evaluate_near_hole, compute_summary_metrics
from sampler import get_batch
from visualize import (
    plot_fields,
    plot_hole_zoom,
    plot_loss_history,
    plot_principal_fields,
    plot_principal_vectors,
    plot_deformed_fields,
    plot_sampling_points,
)


class _TeeStream:
    """Write console output to both terminal and in-memory buffer."""

    def __init__(self, stream, buffer: io.StringIO):
        self._stream = stream
        self._buffer = buffer

    def write(self, data):
        self._stream.write(data)
        self._buffer.write(data)

    def flush(self):
        self._stream.flush()
        self._buffer.flush()


def _clean_run_log(text: str) -> str:
    """Remove tqdm progress-bar redraw artifacts from captured output.

    Keeps explicit training summary lines (printed via tqdm.write) and normal
    print output, while dropping dynamic progress-bar updates.
    """
    # tqdm updates are written with carriage returns; normalize first.
    normalized = text.replace("\r", "\n")
    lines = normalized.splitlines()

    # Match typical tqdm bar fragments like " 76%|###...".
    pbar_re = re.compile(r"\s*\d{1,3}%\|.*")
    cleaned = [ln for ln in lines if ln and not pbar_re.match(ln)]
    return "\n".join(cleaned) + "\n"


def main():
    save_dir = os.path.join(os.path.dirname(__file__), "results")

    os.makedirs(save_dir, exist_ok=True)
    log_buffer = io.StringIO()
    log_path = os.path.join(save_dir, "run.log")

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = _TeeStream(orig_stdout, log_buffer)
    sys.stderr = _TeeStream(orig_stderr, log_buffer)

    try:
        cfg = Config()
        cfg.training.save_dir = save_dir

        _print_banner(cfg)

        # -------------------------------------------------------------------
        # Train
        # -------------------------------------------------------------------
        params, model, history = train(cfg)

        # -------------------------------------------------------------------
        # Evaluate
        # -------------------------------------------------------------------
        print("\nEvaluating on default full and near-hole grids …")
        results = evaluate_on_grid(params, model, cfg.problem, cfg.network)
        results_zoom = evaluate_near_hole(params, model, cfg.problem, cfg.network)
        metrics = compute_summary_metrics(results)

        print("\n  Summary metrics:")
        print(f"    max |u|        :  {metrics['u_max']:.4e} {cfg.problem.length_unit}")
        print(f"    max |v|        :  {metrics['v_max']:.4e} {cfg.problem.length_unit}")
        print(f"    max sigma_xx   :  {metrics['sxx_max']:.4e} {cfg.problem.stress_unit}")
        print(f"    min sigma_yy   :  {metrics['syy_min']:.4e} {cfg.problem.stress_unit}")
        print(f"    max |σ_xy|     :  {metrics['sxy_abs_max']:.4e} {cfg.problem.stress_unit}")

        # -------------------------------------------------------------------
        # Visualise
        # -------------------------------------------------------------------
        print("\nSaving plots …")
        plot_loss_history(history, cfg.training.save_dir, plot_cfg=cfg.plotting)
        plot_fields(
            results,
            cfg.training.save_dir,
            length_unit=cfg.problem.length_unit,
            stress_unit=cfg.problem.stress_unit,
            plot_cfg=cfg.plotting,
        )
        plot_principal_fields(
            results,
            cfg.training.save_dir,
            length_unit=cfg.problem.length_unit,
            stress_unit=cfg.problem.stress_unit,
            plot_cfg=cfg.plotting,
        )
        plot_principal_vectors(
            results,
            cfg.training.save_dir,
            length_unit=cfg.problem.length_unit,
            stress_unit=cfg.problem.stress_unit,
            plot_cfg=cfg.plotting,
        )
        plot_deformed_fields(
            results,
            cfg.training.save_dir,
            deformation_scale=cfg.plotting.deformation_scale,
            length_unit=cfg.problem.length_unit,
            stress_unit=cfg.problem.stress_unit,
            plot_cfg=cfg.plotting,
        )
        plot_hole_zoom(
            results_zoom,
            cfg.training.save_dir,
            length_unit=cfg.problem.length_unit,
            stress_unit=cfg.problem.stress_unit,
            plot_cfg=cfg.plotting,
        )
        _batch = get_batch(cfg.problem, cfg.training,
                           jax.random.PRNGKey(cfg.training.seed))
        plot_sampling_points(_batch, cfg.problem, cfg.training.save_dir,
                             length_unit=cfg.problem.length_unit,
                             plot_cfg=cfg.plotting)

        print(f"\n✓  All results saved to  '{cfg.training.save_dir}/'")
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        cleaned_log = _clean_run_log(log_buffer.getvalue())
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(cleaned_log)
        print(f"Run log saved to: {log_path}")


def _print_banner(cfg: Config):
    p = cfg.problem
    n = cfg.network
    t = cfg.training
    sep = "─" * 58
    print(f"\n{sep}")
    print("  2-D Elasticity PINN  ·  Uniaxial Stretch  ·  JAX")
    print(sep)
    print(f"  Domain   : {p.L} mm × {p.H} mm  ({p.mode})")
    print(f"  Hole     : radius = {p.hole_radius} mm at ({p.L / 2:.1f}, {p.H / 2:.1f}) mm")
    print(f"  Material : E = {p.E:.2e} MPa,  ν = {p.nu}")
    print(f"  Loading  : σ₀ = {p.sigma0:.2e} MPa")
    print(f"  u_ref    : {p.u_ref:.4e} mm  (characteristic displacement)")
    print(
        f"  Network  : FourierMLP{list(n.hidden_dims)}, act={n.activation}, "
        f"hard_bc={n.use_hard_bc}, n_fourier={n.n_fourier}"
    )
    print(
        f"  Training : Adam {t.epochs_adam} steps,  "
        f"lr {t.lr_init:.0e}→{t.lr_final:.0e}  (warmup {t.warmup_steps} steps)"
    )
    print(sep)


if __name__ == "__main__":
    main()
