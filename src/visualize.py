"""Plotting utilities for full-field and near-hole PINN outputs."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless backend — safe on any system
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _field_panel(ax, x, y, z, title: str, cmap: str = "RdBu_r",
                 xlabel: str = "x", ylabel: str = "y",
                 xlim=None, ylim=None, aspect: str = "equal"):
    """Single contourf panel with colour-bar."""
    z_ma = np.ma.array(z)
    finite = z_ma.compressed()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        finite = np.array([0.0])
    levels = np.linspace(finite.min(), finite.max(), 64) if finite.min() != finite.max() else 64
    cf = ax.contourf(x, y, z_ma, levels=levels, cmap=cmap)
    ax.set_xlabel(xlabel, fontsize=6)
    ax.set_ylabel(ylabel, fontsize=6)
    ax.set_title(title, fontsize=7)
    ax.tick_params(labelsize=5)
    ax.set_aspect(aspect, adjustable="box")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Attach colorbar to the axes itself so its height matches the plate axis,
    # not the full figure canvas.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.0%", pad=0.2)
    cb = ax.figure.colorbar(cf, cax=cax)
    cb.ax.tick_params(labelsize=5)


def _draw_hole(ax, center, radius):
    ax.add_patch(Circle(center, radius, facecolor="white", edgecolor="black", linewidth=0.8, zorder=10))


def _deformed_hole_boundary(center, radius, scale: float, results: dict,
                            n_theta: int = 721):
    """Return deformed hole boundary coordinates and deformed centre."""
    if (
        "hole_boundary_x" in results
        and "hole_boundary_y" in results
        and "hole_boundary_u" in results
        and "hole_boundary_v" in results
    ):
        xb = np.asarray(results["hole_boundary_x"])
        yb = np.asarray(results["hole_boundary_y"])
        ub = np.asarray(results["hole_boundary_u"])
        vb = np.asarray(results["hole_boundary_v"])
        xb_d = xb + scale * ub
        yb_d = yb + scale * vb
        cx_d = float(np.mean(xb_d))
        cy_d = float(np.mean(yb_d))
        return xb_d, yb_d, cx_d, cy_d

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    xb = center[0] + radius * np.cos(theta)
    yb = center[1] + radius * np.sin(theta)

    xg = results["x"][:, 0]
    yg = results["y"][0, :]
    x_hi = np.clip(np.searchsorted(xg, xb), 1, len(xg) - 1)
    y_hi = np.clip(np.searchsorted(yg, yb), 1, len(yg) - 1)
    x_lo = x_hi - 1
    y_lo = y_hi - 1

    x0 = xg[x_lo]
    x1 = xg[x_hi]
    y0 = yg[y_lo]
    y1 = yg[y_hi]
    tx = np.where(x1 > x0, (xb - x0) / (x1 - x0), 0.0)
    ty = np.where(y1 > y0, (yb - y0) / (y1 - y0), 0.0)

    u = np.nan_to_num(results["u"], nan=0.0)
    v = np.nan_to_num(results["v"], nan=0.0)

    u00 = u[x_lo, y_lo]
    u10 = u[x_hi, y_lo]
    u01 = u[x_lo, y_hi]
    u11 = u[x_hi, y_hi]
    v00 = v[x_lo, y_lo]
    v10 = v[x_hi, y_lo]
    v01 = v[x_lo, y_hi]
    v11 = v[x_hi, y_hi]

    ub = (1.0 - tx) * (1.0 - ty) * u00 + tx * (1.0 - ty) * u10 + (1.0 - tx) * ty * u01 + tx * ty * u11
    vb = (1.0 - tx) * (1.0 - ty) * v00 + tx * (1.0 - ty) * v10 + (1.0 - tx) * ty * v01 + tx * ty * v11

    xb_d = xb + scale * ub
    yb_d = yb + scale * vb
    cx_d = float(np.mean(xb_d))
    cy_d = float(np.mean(yb_d))
    return xb_d, yb_d, cx_d, cy_d


def _draw_deformed_hole(ax, center, radius, scale: float, results: dict):
    """Draw deformed hole boundary from displaced circumference samples."""
    xb_d, yb_d, _, _ = _deformed_hole_boundary(center, radius, scale, results)
    verts = np.column_stack([xb_d, yb_d])
    ax.add_patch(Polygon(verts, closed=True, facecolor="white", edgecolor="black", linewidth=1.0, zorder=10))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_fields(results: dict, save_dir: str = "results",
                length_unit: str = "m", stress_unit: str = "Pa"):
    """Save full-domain field plots.

    Parameters
    ----------
    results     : dict returned by evaluate_on_grid()
    save_dir    : directory to save PNG files
    length_unit : unit label for spatial axes and displacements (e.g. "mm")
    stress_unit : unit label for stress components (e.g. "MPa")
    """
    os.makedirs(save_dir, exist_ok=True)
    x, y = results["x"], results["y"]
    hole_center = results["hole_center"]
    hole_radius = results["hole_radius"]

    lu = length_unit
    su = stress_unit
    umag = np.sqrt(results["u"] ** 2 + results["v"] ** 2)

    field_specs = [
        ("u",   f"u — displacement ({lu})",  "RdBu_r"),
        ("v",   f"v — displacement ({lu})",  "RdBu_r"),
        ("umag", f"|u| — displacement magnitude ({lu})", "viridis"),
        ("sxx", f"σ_xx ({su})",              "RdYlGn_r"),
        ("syy", f"σ_yy ({su})",              "coolwarm"),
        ("txy", f"τ_xy ({su})",              "PuOr"),
    ]

    xlabel = f"x ({lu})"
    ylabel = f"y ({lu})"

    plot_data = dict(results)
    plot_data["umag"] = umag

    for key, label, cmap in field_specs:
        field_plot = np.nan_to_num(plot_data[key], nan=0.0)
        fig, ax = plt.subplots(1, 1, figsize=(12.0, 2.0))
        _field_panel(ax, x, y, field_plot, label, cmap, xlabel, ylabel, aspect="equal")
        _draw_hole(ax, hole_center, hole_radius)
        plt.tight_layout()
        path = os.path.join(save_dir, f"{key}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Field plots → {save_dir}/")


def plot_principal_fields(results: dict, save_dir: str = "results",
                          length_unit: str = "m", stress_unit: str = "Pa"):
    """Save principal stress and strain contour plots on undeformed geometry."""
    os.makedirs(save_dir, exist_ok=True)
    x, y = results["x"], results["y"]
    hole_center = results["hole_center"]
    hole_radius = results["hole_radius"]

    field_specs = [
        ("s1", f"σ1 ({stress_unit})", "RdYlGn_r"),
        ("s2", f"σ2 ({stress_unit})", "RdYlGn_r"),
        ("e1", "ε1 (-)", "viridis"),
        ("e2", "ε2 (-)", "viridis"),
    ]
    xlabel = f"x ({length_unit})"
    ylabel = f"y ({length_unit})"

    for key, label, cmap in field_specs:
        field_plot = np.nan_to_num(results[key], nan=0.0)
        fig, ax = plt.subplots(1, 1, figsize=(12.0, 2.0))
        _field_panel(ax, x, y, field_plot, label, cmap, xlabel, ylabel, aspect="equal")
        _draw_hole(ax, hole_center, hole_radius)
        plt.tight_layout()
        path = os.path.join(save_dir, f"{key}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Principal plots → {save_dir}/")


def plot_deformed_fields(results: dict, save_dir: str = "results",
                         deformation_scale: float = 100.0,
                         length_unit: str = "m", stress_unit: str = "Pa"):
    """Save selected contours on the deformed configuration.

    Required set:
        u, v, |u|, sxx, syy, sxy
    where sxy is sourced from the model shear-stress field txy.
    """
    os.makedirs(save_dir, exist_ok=True)
    x, y = results["x"], results["y"]
    # Keep deformed coordinates finite even inside the masked hole so contourf
    # can render the full domain extents reliably.
    u_def = np.nan_to_num(results["u"], nan=0.0)
    v_def = np.nan_to_num(results["v"], nan=0.0)
    x_def = x + deformation_scale * u_def
    y_def = y + deformation_scale * v_def
    hole_center = results["hole_center"]
    hole_radius = results["hole_radius"]
    umag = np.sqrt(results["u"] ** 2 + results["v"] ** 2)

    # spec: (field_array, output_stem, title, cmap)
    field_specs = [
        (results["u"],   "deformed_u",    f"u on deformed shape ({length_unit}), mag={deformation_scale:g}", "RdBu_r"),
        (results["v"],   "deformed_v",    f"v on deformed shape ({length_unit}), mag={deformation_scale:g}", "RdBu_r"),
        (umag,            "deformed_umag", f"|u| on deformed shape ({length_unit}), mag={deformation_scale:g}", "viridis"),
        (results["sxx"], "deformed_sxx",  f"σ_xx on deformed shape ({stress_unit}), mag={deformation_scale:g}", "RdYlGn_r"),
        (results["syy"], "deformed_syy",  f"σ_yy on deformed shape ({stress_unit}), mag={deformation_scale:g}", "coolwarm"),
        (results["txy"], "deformed_sxy",  f"σ_xy on deformed shape ({stress_unit}), mag={deformation_scale:g}", "PuOr"),
    ]
    xlabel = f"x_deformed ({length_unit})"
    ylabel = f"y_deformed ({length_unit})"
    xlim = (float(np.nanmin(x_def)), float(np.nanmax(x_def)))
    ylim = (float(np.nanmin(y_def)), float(np.nanmax(y_def)))

    # Build mask from the deformed hole boundary polygon so the plotted hole
    # matches the deformed geometry rather than the undeformed circular mask.
    xb_d, yb_d, _, _ = _deformed_hole_boundary(
        hole_center, hole_radius, deformation_scale, results
    )
    hole_path = Path(np.column_stack([xb_d, yb_d]), closed=True)
    xy_def = np.column_stack([x_def.ravel(), y_def.ravel()])
    hole_mask_def = hole_path.contains_points(xy_def).reshape(x_def.shape)

    for field, stem, label, cmap in field_specs:
        # Remove undeformed hole NaNs first, then apply fresh deformed mask.
        field_filled = np.nan_to_num(field, nan=0.0)
        field_plot = np.ma.array(field_filled, mask=hole_mask_def)
        fig, ax = plt.subplots(1, 1, figsize=(12.0, 2.0))
        _field_panel(ax, x_def, y_def, field_plot, label, cmap, xlabel, ylabel, xlim=xlim, ylim=ylim, aspect="equal")
        _draw_deformed_hole(ax, hole_center, hole_radius, deformation_scale, results)
        plt.tight_layout()
        path = os.path.join(save_dir, f"{stem}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Deformed plots → {save_dir}/")


def plot_hole_zoom(results: dict, save_dir: str = "results",
                   length_unit: str = "m", stress_unit: str = "Pa"):
    """Save zoomed plots around the hole."""
    os.makedirs(save_dir, exist_ok=True)
    x, y = results["x"], results["y"]
    hole_center = results["hole_center"]
    hole_radius = results["hole_radius"]
    lu = length_unit
    su = stress_unit

    field_specs = [
        ("sxx", f"σ_xx near hole ({su})", "RdYlGn_r"),
        ("syy", f"σ_yy near hole ({su})", "coolwarm"),
        ("txy", f"τ_xy near hole ({su})", "PuOr"),
    ]
    xlabel = f"x ({lu})"
    ylabel = f"y ({lu})"

    for key, label, cmap in field_specs:
        field_plot = np.nan_to_num(results[key], nan=0.0)
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.0))
        _field_panel(ax, x, y, field_plot, label, cmap, xlabel, ylabel, aspect="equal")
        _draw_hole(ax, hole_center, hole_radius)
        plt.tight_layout()
        path = os.path.join(save_dir, f"zoom_{key}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Zoom plots  → {save_dir}/")


def plot_loss_history(history: dict, save_dir: str = "results"):
    """Log-scale loss curves for total loss and individual components."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = np.arange(1, len(history["total"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.semilogy(epochs, history["total"], "k-", lw=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Total Loss")
    ax1.grid(True, which="both", alpha=0.3)

    component_colors = {
        "pde":       "royalblue",
        "bc_left":   "tomato",
        "bc_right":  "mediumseagreen",
        "bc_tb":     "orchid",
        "bc_hole":   "darkorange",
        "bc_mid":    "cadetblue",
    }
    for comp, col in component_colors.items():
        if comp in history and len(history[comp]):
            ax2.semilogy(epochs, history[comp], color=col, lw=1.2, label=comp)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss Components")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "loss_history.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Loss curve  → {path}")


def plot_sampling_points(batch: tuple, cfg_p, save_dir: str = "results",
                         length_unit: str = "mm"):
    """Save a diagnostic plot showing geometry boundaries and all collocation points.

    Parameters
    ----------
    batch    : output of ``sampler.get_batch()`` —
               (pts_int, pts_left, pts_right, pts_bot, pts_top, pts_hole, pts_mid)
    cfg_p    : ProblemConfig — used to recover physical dimensions
    save_dir : directory to write the PNG
    """
    import numpy as np  # already imported at module level but kept explicit here
    os.makedirs(save_dir, exist_ok=True)

    pts_int, pts_left, pts_right, pts_bot, pts_top, pts_hole, pts_mid = batch

    L = float(cfg_p.L)
    H = float(cfg_p.H)
    xc = float(cfg_p.hole_xi_c) * L
    yc = float(cfg_p.hole_eta_c) * L
    r  = float(cfg_p.hole_radius)

    # Convert all point sets from normalised to physical coordinates
    def phys(pts):
        pts = np.asarray(pts)
        return pts[:, 0] * L, pts[:, 1] * L

    hole_pts = np.asarray(pts_hole)

    categories = [
        ("Interior (bulk + near-hole)", phys(pts_int),        "steelblue",   4, 0.35),
        ("Left BC",                     phys(pts_left),        "tomato",      6, 0.8),
        ("Right BC",                    phys(pts_right),       "mediumseagreen", 6, 0.8),
        ("Top / Bottom BC",             phys(np.concatenate([pts_top, pts_bot])), "orchid", 6, 0.8),
        ("Hole BC",                     (hole_pts[:, 0] * L, hole_pts[:, 1] * L), "darkorange", 7, 1.0),
        ("Midline BC",                  phys(pts_mid),         "cadetblue",   6, 0.8),
    ]

    fig, ax = plt.subplots(figsize=(14, 3.5))

    # Draw domain rectangle
    rect = plt.Polygon(
        [[0, 0], [L, 0], [L, H], [0, H]],
        closed=True, fill=False, edgecolor="black", linewidth=1.5
    )
    ax.add_patch(rect)

    # Draw exact circular hole
    ax.add_patch(Circle((xc, yc), r, facecolor="#dddddd", edgecolor="black",
                         linewidth=1.2, zorder=5, label="Hole geometry"))

    # Plot each point set
    for label, (px, py), color, ms, alpha in categories:
        ax.scatter(px, py, s=ms, color=color, alpha=alpha, linewidths=0,
                   label=f"{label} ({len(px)})", zorder=6)

    # Axis formatting
    ax.set_xlim(-0.03 * L, 1.03 * L)
    ax.set_ylim(-0.15 * H, 1.15 * H)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"x ({length_unit})", fontsize=8)
    ax.set_ylabel(f"y ({length_unit})", fontsize=8)
    ax.set_title("Collocation / boundary-condition sampling points", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(loc="upper right", fontsize=6.5, markerscale=3, framealpha=0.85)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = os.path.join(save_dir, "sampling_points.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Sampling plot → {path}")
