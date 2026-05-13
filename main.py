"""Entry point for the plate-with-hole elasticity PINN using JAX + Flax.

Problem
-------
    Domain     :  [0, L] x [0, H] in physical units (mm)
    Hole       :  centred at (L/2, H/2) with radius hole_radius mm
    Material   :  plane stress, E, nu
    BCs        :
        Left  (x = 0)   :  u = v = 0  (hard-enforced via ansatz)
        Right (x = L)   :  sigma_xx = sigma0, tau_xy = 0
        Top / Bottom    :  sigma_yy = 0, tau_xy = 0
        Hole surface    :  sigma*n = 0
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
from postprocess import run_evaluation_and_plots
from generate_report import build_report


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
        # Evaluate and plot (shared with postprocess.py)
        # -------------------------------------------------------------------
        print("\nPost-processing ...")
        run_evaluation_and_plots(params, model, history, cfg)

        # -------------------------------------------------------------------
        # Technical report
        # -------------------------------------------------------------------
        print("\nGenerating technical report ...")
        try:
            build_report(save_dir)
        except Exception as exc:
            print(f"Warning: report generation failed: {exc}")

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
    sep = "-" * 58
    print(f"\n{sep}")
    print("  2-D Elasticity PINN  *  Uniaxial Stretch  *  JAX")
    print(sep)
    print(f"  Domain   : {p.L} mm x {p.H} mm  ({p.mode})")
    print(f"  Hole     : radius = {p.hole_radius} mm at ({p.L / 2:.1f}, {p.H / 2:.1f}) mm")
    print(f"  Material : E = {p.E:.2e} MPa,  nu = {p.nu}")
    print(f"  Loading  : sigma0 = {p.sigma0:.2e} MPa")
    print(f"  u_ref    : {p.u_ref:.4e} mm  (characteristic displacement)")
    print(
        f"  Network  : FourierMLP{list(n.hidden_dims)}, act={n.activation}, "
        f"hard_bc={n.use_hard_bc}, n_fourier={n.n_fourier}"
    )
    print(
        f"  Training : Adam {t.epochs_adam} steps,  "
        f"lr {t.lr_init:.0e}-->{t.lr_final:.0e}  (warmup {t.warmup_steps} steps)"
    )
    print(sep)


if __name__ == "__main__":
    main()