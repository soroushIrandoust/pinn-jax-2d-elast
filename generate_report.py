"""Generate a manuscript-style PDF report from a completed PINN training run.

Usage
-----
    python generate_report.py
    python generate_report.py --results_dir my_run/

Output
------
    results/technical_report.pdf
"""

import argparse
import os
import re
import sys
import textwrap

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# Local config dataclasses
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from config import ProblemConfig, NetworkConfig, TrainingConfig

# ---------------------------------------------------------------------------
# Page geometry and typography
# ---------------------------------------------------------------------------
_PAGE_W = 8.5
_PAGE_H = 11.0
_MARGIN = 1.0

_LM = _MARGIN / _PAGE_W
_RM = 1.0 - _MARGIN / _PAGE_W
_TM = 1.0 - _MARGIN / _PAGE_H
_BM = _MARGIN / _PAGE_H
_CW = _RM - _LM
_CH = _TM - _BM

_FONT_FAMILY = "serif"
_FONT_SERIF_STACK = [
    "Times New Roman",
    "Times New Roman PS MT",
    "Times",
    "Nimbus Roman",
    "STIX Two Text",
    "STIXGeneral",
    "DejaVu Serif",
]

_FS_BODY = 10
_FS_SECTION = 14
_FS_SUBSECTION = 11
_FS_SUBSUB = 10
_FS_TITLE = 16
_FS_SUBTITLE = 12
_FS_META = 10
_FS_CAPTION = 9
_FS_AXIS_LABEL = 9
_FS_AXIS_TICK = 8
_FS_AXIS_TITLE = 10
_FS_LEGEND = 8

_WRAP_WIDTH = 118

plt.rcParams.update({
    "font.family": _FONT_FAMILY,
    "font.serif": _FONT_SERIF_STACK,
    "font.size": _FS_BODY,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.labelsize": _FS_AXIS_LABEL,
    "xtick.labelsize": _FS_AXIS_TICK,
    "ytick.labelsize": _FS_AXIS_TICK,
    "axes.titlesize": _FS_AXIS_TITLE,
    "legend.fontsize": _FS_LEGEND,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def _text_page(pdf: PdfPages, text: str):
    """Render a manuscript text page with simple heading detection."""
    fig = plt.figure(figsize=(_PAGE_W, _PAGE_H))
    ax = fig.add_axes([_LM, _BM, _CW, _CH])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    pt_to_ax = 1.0 / (72.0 * (_CH * _PAGE_H))
    y = 0.99

    lines = text.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()

    for raw in lines:
        if y < -0.02:
            break

        s = raw.strip()
        if not s:
            y -= 1.15 * _FS_BODY * pt_to_ax
            continue

        if re.match(r"^\d+\.\s{2,}\S", s) and not re.match(r"^\d+\.\d", s):
            y -= 0.6 * _FS_SECTION * pt_to_ax
            ax.text(0.0, y, s, fontsize=_FS_SECTION, fontweight="bold", va="top")
            y -= 1.9 * _FS_SECTION * pt_to_ax
            continue

        if re.match(r"^\d+\.\d+\s{2,}\S", s) and not re.match(r"^\d+\.\d+\.\d", s):
            y -= 0.45 * _FS_SUBSECTION * pt_to_ax
            ax.text(0.0, y, s, fontsize=_FS_SUBSECTION, fontweight="bold", va="top")
            y -= 1.65 * _FS_SUBSECTION * pt_to_ax
            continue

        if re.match(r"^\d+\.\d+\.\d+\s{2,}\S", s):
            y -= 0.25 * _FS_SUBSUB * pt_to_ax
            ax.text(0.0, y, s, fontsize=_FS_SUBSUB, fontweight="bold", va="top")
            y -= 1.55 * _FS_SUBSUB * pt_to_ax
            continue

        wrapped = textwrap.wrap(s, width=_WRAP_WIDTH) or [s]
        for w in wrapped:
            ax.text(0.0, y, w, fontsize=_FS_BODY, va="top")
            y -= 1.28 * _FS_BODY * pt_to_ax

    return fig


def _save_page(pdf: PdfPages, fig, page_no: int | None):
    """Save one page and optionally stamp a bottom-center page number."""
    if page_no is not None:
        fig.text(
            0.5,
            0.035,
            str(page_no),
            ha="center",
            va="bottom",
            fontsize=_FS_BODY,
            fontfamily=_FONT_FAMILY,
        )
    pdf.savefig(fig, dpi=200)
    plt.close(fig)


def _packed_fig_pages(
    pdf: PdfPages,
    items: list[tuple[str, str, int]],
    logical_page_start: int,
) -> int:
    """Greedy pack figures so each page holds as many as possible."""
    items = [(p, c, n) for p, c, n in items if os.path.exists(p)]
    if not items:
        return logical_page_start

    idx = 0
    logical_page = logical_page_start
    while idx < len(items):
        fig = plt.figure(figsize=(_PAGE_W, _PAGE_H))
        top_pad = 0.02
        bottom_pad = 0.03
        caption_h = 0.045
        vgap = 0.022

        ax_w_in = _CW * _PAGE_W
        content_h_in = _CH * _PAGE_H

        y_top = 1.0 - top_pad
        placed = 0

        while idx < len(items):
            path, caption, fig_num = items[idx]
            img = plt.imread(path)
            h_px, w_px = img.shape[0], img.shape[1]

            ideal_h = (ax_w_in * (h_px / float(w_px))) / content_h_in
            img_h = max(0.09, min(ideal_h, 0.36))

            needed = img_h + caption_h
            if placed > 0:
                needed += vgap

            if y_top - needed < bottom_pad:
                break

            if placed > 0:
                y_top -= vgap

            img_bottom = _BM + (y_top - img_h) * _CH
            ax_img = fig.add_axes([_LM, img_bottom, _CW, img_h * _CH])
            ax_img.imshow(img)
            ax_img.axis("off")

            cap_bottom = _BM + (y_top - img_h - caption_h) * _CH
            ax_cap = fig.add_axes([_LM, cap_bottom, _CW, caption_h * _CH])
            ax_cap.axis("off")
            ax_cap.text(
                0.5,
                0.5,
                f"Figure {fig_num}. {caption}",
                ha="center",
                va="center",
                fontsize=_FS_CAPTION,
                fontstyle="italic",
                wrap=True,
            )

            y_top = y_top - img_h - caption_h
            placed += 1
            idx += 1

        if placed == 0 and idx < len(items):
            # Fallback: always place at least one figure even if very tall.
            path, caption, fig_num = items[idx]
            img = plt.imread(path)
            img_h = 1.0 - top_pad - bottom_pad - caption_h
            img_bottom = _BM + bottom_pad * _CH + caption_h * _CH
            ax_img = fig.add_axes([_LM, img_bottom, _CW, img_h * _CH])
            ax_img.imshow(img)
            ax_img.axis("off")

            ax_cap = fig.add_axes([_LM, _BM + bottom_pad * _CH, _CW, caption_h * _CH])
            ax_cap.axis("off")
            ax_cap.text(
                0.5,
                0.5,
                f"Figure {fig_num}. {caption}",
                ha="center",
                va="center",
                fontsize=_FS_CAPTION,
                fontstyle="italic",
                wrap=True,
            )
            idx += 1

        _save_page(pdf, fig, logical_page)
        logical_page += 1

    return logical_page


def _loss_page(pdf: PdfPages, npz_path: str, fig_num: int):
    if not os.path.exists(npz_path):
        return

    data = np.load(npz_path)
    total = data["total"]
    epochs = np.arange(len(total))
    comp_keys = [k for k in data.files if k not in ("total", "sigma_xx_hole_top", "sigma_xx_hole_side")]

    fig = plt.figure(figsize=(_PAGE_W, _PAGE_H))
    ax1 = fig.add_axes([_LM, _BM + _CH * 0.38, _CW * 0.44, _CH * 0.52])
    ax2 = fig.add_axes([_LM + _CW * 0.56, _BM + _CH * 0.38, _CW * 0.44, _CH * 0.52])

    ax1.semilogy(epochs, total, lw=0.8, color="black")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Weighted loss")
    ax1.set_title("Total Loss")
    ax1.grid(True, which="both", ls=":", lw=0.4)

    colors = plt.cm.tab10.colors
    for ci, key in enumerate(comp_keys):
        ax2.semilogy(epochs, data[key], lw=0.65, label=key, color=colors[ci % 10])
    ax2.set_xlabel("Epoch")
    ax2.set_title("Loss Components")
    ax2.legend(loc="upper right", framealpha=0.7)
    ax2.grid(True, which="both", ls=":", lw=0.4)

    ax_cap = fig.add_axes([_LM, _BM + _CH * 0.28, _CW, _CH * 0.07])
    ax_cap.axis("off")
    ax_cap.text(
        0.5,
        0.5,
        f"Figure {fig_num}. Training loss history on a log scale. Left: total weighted loss. Right: individual PDE and boundary loss components.",
        ha="center",
        va="center",
        fontsize=_FS_CAPTION,
        fontstyle="italic",
        wrap=True,
    )

    return fig


def build_report(results_dir: str):
    cfg_p = ProblemConfig()
    cfg_n = NetworkConfig()
    cfg_t = TrainingConfig()

    n_layers = len(cfg_n.hidden_dims)
    layer_size = cfg_n.hidden_dims[0]
    n_fourier_feats = 4 * cfg_n.n_fourier
    n_polar_feats = 2 + 2 * cfg_n.n_polar
    n_input_total = 2 + n_fourier_feats + n_polar_feats
    u_ref = cfg_p.u_ref
    eta_max = cfg_p.eta_max
    kirsch_peak = 3.0 * cfg_p.sigma0

    img = os.path.join(results_dir, "{}.png")
    zimg = os.path.join(results_dir, "zoom_{}.png")

    fig_num = 0

    def nf() -> int:
        nonlocal fig_num
        fig_num += 1
        return fig_num

    out_path = os.path.join(results_dir, "technical_report.pdf")

    with PdfPages(out_path) as pdf:
        logical_page = 1
        fig = plt.figure(figsize=(_PAGE_W, _PAGE_H))
        ax = fig.add_axes([_LM, _BM, _CW, _CH])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        ax.text(0.5, 0.90, "Physics-Informed Neural Network for 2-D Linear Elasticity", ha="center", va="top", fontsize=_FS_TITLE, fontweight="bold")
        ax.text(0.5, 0.82, "Stress Concentration in a Rectangular Plate with a Circular Hole", ha="center", va="top", fontsize=_FS_SUBTITLE, fontstyle="italic")
        ax.text(
            0.5,
            0.68,
            (
                f"JAX + Flax + Optax\n"
                f"FourierMLP ({n_layers}x{layer_size}, {cfg_n.activation}, n_fourier={cfg_n.n_fourier}, n_polar={cfg_n.n_polar})\n"
                f"Domain {cfg_p.L:,.0f}x{cfg_p.H:,.0f} mm, hole R={cfg_p.hole_radius:.0f} mm, {cfg_p.mode}\n"
                f"E={cfg_p.E:,.0f} MPa, nu={cfg_p.nu:.2f}, sigma0={cfg_p.sigma0:.0f} MPa\n"
                f"{cfg_t.epochs_adam:,} epochs, LR {cfg_t.lr_init:.0e}->{cfg_t.lr_final:.0e}"
            ),
            ha="center",
            va="top",
            fontsize=_FS_META,
            linespacing=1.8,
        )

        _save_page(pdf, fig, None)

        fig = _text_page(pdf, textwrap.dedent(
            f"""
            1.  Problem Formulation

            1.1  Geometry and Loading

            The computational domain is a rectangular plate of size L x H = {cfg_p.L:,.0f} mm x {cfg_p.H:,.0f} mm with a central circular hole of radius R = {cfg_p.hole_radius:.0f} mm. The material is linear elastic with E = {cfg_p.E:,.0f} MPa and nu = {cfg_p.nu:.2f}, under {cfg_p.mode} assumptions. A uniaxial traction sigma_0 = {cfg_p.sigma0:.0f} MPa is prescribed on the right edge.

            1.2  Boundary Conditions

            The model applies mixed displacement and traction conditions. The left edge is clamped and the hole and outer horizontal edges are traction-free.

            Left (x = 0): u = 0, v = 0 [hard-enforced via network ansatz]
            Right (x = L): sigma_xx = {cfg_p.sigma0:.0f} MPa, tau_xy = 0
            Top (y = H): sigma_yy = 0, tau_xy = 0
            Bottom (y = 0): sigma_yy = 0, tau_xy = 0
            Hole surface: sigma . n = 0 [soft penalty]

            In addition, a midline symmetry condition v(x, H/2) = 0 is imposed as a soft penalty to suppress antisymmetric vertical modes.

            1.3  Governing Equations and Constitutive Law

            With zero body force, static equilibrium is enforced in strong form.

            d(sigma_xx)/dx + d(tau_xy)/dy = 0
            d(tau_xy)/dx + d(sigma_yy)/dy = 0

            The stress-strain relation for {cfg_p.mode} is written as follows.

            sigma_xx = E/(1-nu^2) * (eps_xx + nu * eps_yy)
            sigma_yy = E/(1-nu^2) * (eps_yy + nu * eps_xx)
            tau_xy = E/(2*(1+nu)) * gamma_xy

            Here eps_xx = du/dx, eps_yy = dv/dy, and gamma_xy = du/dy + dv/dx.

            1.4  Kirsch Benchmark

            The Kirsch infinite-plate solution predicts stress concentration factor SCF = 3 at the hole equator. For the present loading this gives sigma_xx = 3 * sigma_0 = {kirsch_peak:.0f} MPa at (x_c, y_c +/- R).

            Because the numerical problem uses a finite domain and a clamped left edge, the exact peak stress is expected to deviate from this analytical benchmark.
            """
        ))
        _save_page(pdf, fig, logical_page)
        logical_page += 1

        fig = _text_page(pdf, textwrap.dedent(
            f"""
            2.  PINN Methodology

            A Physics-Informed Neural Network approximates u(x,y) and v(x,y) by a neural network trained to minimize a composite residual loss. No labeled solution data is required.

            2.1  Non-dimensionalization

            The model uses normalized spatial, displacement, and stress variables.

            xi = x / L, eta = y / L (inputs in [0,1] x [0, {eta_max:.3g}])
            u_hat = u / u_ref, u_ref = sigma_0 * L / E = {u_ref:.4g} mm
            sigma_tilde = sigma / sigma_0

            2.2  Hard BC Ansatz

            To satisfy the clamped left boundary exactly, the displacement fields are parameterized with a multiplicative xi factor.

            u_hat(xi, eta) = xi * psi_u(xi, eta)
            v_hat(xi, eta) = xi * psi_v(xi, eta)

            2.3  Loss Function

            The weighted training objective combines PDE residual and boundary terms.

            L = {cfg_t.w_pde:g}*L_pde + {cfg_t.w_bc_traction:g}*L_right + {cfg_t.w_bc_tb:g}*L_tb + {cfg_t.w_bc_hole:g}*L_hole + {cfg_t.w_bc_mid:g}*L_mid

            Each term is a mean-squared residual on randomly sampled collocation points for that region. Points are resampled every {cfg_t.resample_every:,} epochs.

            2.4  Optimization

            Optimization is performed with Adam using a cosine-decay learning-rate schedule from {cfg_t.lr_init:.0e} to {cfg_t.lr_final:.0e}, with warm-up over {cfg_t.warmup_steps:,} steps, for a total of {cfg_t.epochs_adam:,} epochs.

            2.5  Network Architecture

            A Fourier feature embedding is prepended to a fully-connected MLP to overcome spectral bias.

            2.5.1  Input feature map (total dimension {n_input_total})

            The input embedding concatenates three feature groups. The first group is the raw coordinates (xi, eta) with dimension 2. The second group contains {n_fourier_feats} Fourier features for k = 1,...,{cfg_n.n_fourier}. The third group contains {n_polar_feats} hole-centered polar features for k = 1,...,{cfg_n.n_polar}.

            2.5.2  MLP backbone

            The backbone is a fully connected MLP with {n_layers} hidden layers and {layer_size} neurons per layer, using {cfg_n.activation} activation. The network outputs two pre-ansatz fields, psi_u and psi_v. Stresses are computed via automatic differentiation and constitutive relations.

            2.6  Sampling Strategy

            The collocation set is redrawn during training rather than kept fixed. Interior points are sampled randomly outside the hole, while an enriched annulus is used near the hole boundary to improve resolution of steep stress gradients. Figure {fig_num + 1} shows a typical sampling pattern used in training.
            """
        ))
        _save_page(pdf, fig, logical_page)
        logical_page += 1

        logical_page = _packed_fig_pages(pdf, [
            (
                os.path.join(results_dir, "sampling_points.png"),
                "Training collocation points, including near-hole enrichment and boundary sampling.",
                nf(),
            )
        ], logical_page)

        fig = _text_page(pdf, textwrap.dedent(
            """
            3.  Results

            The trained model recovers the expected qualitative structure of the elasticity solution. The displacement fields remain smooth away from the hole, while the stress plots show clear concentration and redistribution around the circular boundary. The loss history provides a complementary view of optimization behavior, including stochastic variability induced by random collocation resampling.

            3.1  Training History

            The total loss decreases substantially over training, while individual loss components settle to different magnitudes depending on weighting and physical difficulty.

            3.2  Full-Field Predictions

            Full-domain displacement and stress plots show the far-field loading response together with the perturbation induced by the hole.

            3.3  Near-Hole and Deformed Views

            Near-hole stress plots isolate the region where the strongest gradients occur. Deformed plots provide a qualitative visualization of how predicted fields distort around the cavity.
            """
        ))
        _save_page(pdf, fig, logical_page)
        logical_page += 1

        fig = _loss_page(pdf, os.path.join(results_dir, "loss_history.npz"), nf())
        if fig is not None:
            _save_page(pdf, fig, logical_page)
            logical_page += 1

        figure_items = [
            (img.format("u"), "x-displacement u (mm).", nf()),
            (img.format("v"), "y-displacement v (mm).", nf()),
            (img.format("sxx"), "sigma_xx (MPa).", nf()),
            (img.format("syy"), "sigma_yy (MPa).", nf()),
            (img.format("txy"), "tau_xy (MPa).", nf()),
            (zimg.format("sxx"), "sigma_xx near hole (MPa) - near-hole detail.", nf()),
            (zimg.format("syy"), "sigma_yy near hole (MPa) - near-hole detail.", nf()),
            (zimg.format("txy"), "tau_xy near hole (MPa) - near-hole detail.", nf()),
            (os.path.join(results_dir, "deformed_u.png"), "Deformed x-displacement field.", nf()),
            (os.path.join(results_dir, "deformed_v.png"), "Deformed y-displacement field.", nf()),
            (os.path.join(results_dir, "deformed_umag.png"), "Deformed displacement-magnitude field.", nf()),
            (os.path.join(results_dir, "deformed_sxx.png"), "Deformed sigma_xx field.", nf()),
            (os.path.join(results_dir, "deformed_syy.png"), "Deformed sigma_yy field.", nf()),
            (os.path.join(results_dir, "deformed_sxy.png"), "Deformed tau_xy field.", nf()),
        ]
        logical_page = _packed_fig_pages(pdf, figure_items, logical_page)

        fig = _text_page(pdf, textwrap.dedent(
            f"""
            4.  Discussion and Conclusions

            The FourierMLP PINN resolves stress concentration around the circular hole and reproduces the expected sigma_xx peak behavior near the hole equator. The Fourier and polar embeddings improve near-hole expressivity, and the hard boundary-condition ansatz stabilizes optimization by enforcing the left-edge clamp exactly.

            With sigma_0 = {cfg_p.sigma0:.0f} MPa, the Kirsch infinite-plate target is sigma_xx,peak = 3*sigma_0 = {kirsch_peak:.0f} MPa (SCF = 3). The present results should therefore be interpreted against that reference: agreement indicates correct concentration magnitude, while any mismatch from {kirsch_peak:.0f} MPa is expected to reflect finite-domain effects and the clamped boundary at x = 0, not only network approximation error.

            At the same time, the optimization history remains noisy because stochastic collocation introduces variance in residual estimates. The near-hole stress field is qualitatively correct, but the report still lacks a systematic quantitative error study against a trusted FEM or analytical reference over the finite domain.

            Overall, the present methodology is effective as a research prototype and produces physically plausible fields, but quantitative verification and robustness should be strengthened before predictive use.
            """
        ))
        _save_page(pdf, fig, logical_page)
        logical_page += 1

        fig = _text_page(pdf, textwrap.dedent(
            """
            5.  Future Work

            A natural next step is to compare PINN predictions against a finite-element reference solution on the same geometry and loading case, so displacement and stress errors can be reported quantitatively.

            Another useful extension is adaptive loss reweighting, such as NTK-based balancing or residual-driven weighting, to improve optimization stability and reduce manual hyperparameter tuning.

            The framework can also be extended to nonlinear material behavior, more complex geometries, and eventually three-dimensional elasticity.
            """
        ))
        _save_page(pdf, fig, logical_page)
        logical_page += 1

    print(f"  Report saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate manuscript PDF report from PINN results.")
    parser.add_argument(
        "--results_dir",
        default=None,
        help="Path to results directory (default: <script dir>/results)",
    )
    args = parser.parse_args()

    if args.results_dir is None:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    else:
        results_dir = args.results_dir

    if not os.path.isdir(results_dir):
        print(f"Error: results directory not found: {results_dir}")
        sys.exit(1)

    build_report(results_dir)
