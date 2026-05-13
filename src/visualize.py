"""Plotting utilities for full-field and near-hole PINN outputs."""

import os
import importlib
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless backend — safe on any system
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, Polygon
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable

from config import PlotConfig

go = None
make_subplots = None
_PARULA_RGB = [
    (0.2081, 0.1663, 0.5292),
    (0.2116, 0.1898, 0.5777),
    (0.2123, 0.2742, 0.6743),
    (0.1959, 0.3665, 0.7394),
    (0.1707, 0.4700, 0.7720),
    (0.1253, 0.5773, 0.7682),
    (0.1579, 0.6834, 0.7081),
    (0.3692, 0.7883, 0.5394),
    (0.6780, 0.8637, 0.1895),
    (0.9022, 0.8890, 0.1057),
    (0.9763, 0.9831, 0.0538),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _interactive_available() -> bool:
    global go, make_subplots
    if go is not None and make_subplots is not None:
        return True
    try:
        go = importlib.import_module("plotly.graph_objects")
        make_subplots = importlib.import_module("plotly.subplots").make_subplots
        return True
    except Exception:
        go = None
        make_subplots = None
        return False


def _interactive_dir(save_dir: str) -> str:
    parent = os.path.dirname(os.path.abspath(save_dir))
    path = os.path.join(parent, "results_interactive")
    os.makedirs(path, exist_ok=True)
    if not _interactive_available():
        note_path = os.path.join(path, "_interactive_disabled.txt")
        with open(note_path, "w", encoding="utf-8") as fh:
            fh.write(
                "Interactive plots were skipped because Plotly is not available.\n"
                "Install dependencies and rerun postprocess:\n"
                "    pip install -r requirements.txt\n"
                "    python postprocess.py\n"
            )
    return path


def _get_cmap(cmap_name: str):
    if cmap_name == "parula":
        if "parula" in plt.colormaps():
            return plt.get_cmap("parula")
        return LinearSegmentedColormap.from_list("parula_custom", _PARULA_RGB, N=256)
    return plt.get_cmap(cmap_name)


def _finite_min_max(values) -> tuple[float, float]:
    arr = np.asarray(values)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 0.0
    return float(np.min(finite)), float(np.max(finite))


def _field_stats_text(values, digits: int = 4) -> str:
    vmin, vmax = _finite_min_max(values)
    return f"min={vmin:.{digits}e}\nmax={vmax:.{digits}e}"


def _annotate_field_stats_mpl(ax, values, plot_cfg: PlotConfig | None,
                              extra_lines: str | None = None):
    cfg = _resolve_plot_cfg(plot_cfg)
    if not cfg.annotate_field_minmax:
        return
    txt = _field_stats_text(values, digits=int(cfg.field_stats_digits))
    if extra_lines:
        txt = f"{txt}\n{extra_lines}"
    fig = ax.figure
    fig.text(
        0.985,
        0.02,
        txt,
        transform=fig.transFigure,
        ha="right",
        va="bottom",
        fontsize=6,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="none", edgecolor="none", alpha=0.0),
        zorder=20,
    )


def _canonical_field_key(field_key: str) -> str:
    key = str(field_key)
    for prefix in (
        "undeformed_zoom_",
        "deformed_zoom_",
        "undeformed_vector_",
        "deformed_vector_",
        "undeformed_",
        "deformed_",
        "zoom_",
        "vector_",
    ):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _resolve_levels(field_key: str, values, plot_cfg: PlotConfig | None):
    data_min, data_max = _finite_min_max(values)

    if plot_cfg is None:
        return data_min, data_max

    key = _canonical_field_key(field_key)

    # Critical fixed-range fields should always honor configured limits.
    if key in {"syy", "sxy", "s2"}:
        vmin_cfg, vmax_cfg = plot_cfg.field_level_limits.get(key, (None, None))
        if vmin_cfg is not None or vmax_cfg is not None:
            vmin = data_min if vmin_cfg is None else float(vmin_cfg)
            vmax = data_max if vmax_cfg is None else float(vmax_cfg)
            if vmin > vmax:
                vmin, vmax = vmax, vmin
            return vmin, vmax

    if plot_cfg.auto_levels:
        mode = "auto"
    else:
        mode = plot_cfg.field_level_mode.get(key, "auto")

    lims = plot_cfg.field_level_limits.get(key, (None, None))
    vmin_cfg, vmax_cfg = lims

    if mode == "fixed":
        vmin = data_min if vmin_cfg is None else float(vmin_cfg)
        vmax = data_max if vmax_cfg is None else float(vmax_cfg)
    elif mode == "nonnegative_auto":
        vmin = 0.0
        vmax = data_max
    elif mode == "symmetric_auto":
        abs_max = max(abs(data_min), abs(data_max))
        vmin, vmax = -abs_max, abs_max
    else:  # "auto"
        vmin, vmax = data_min, data_max

    if vmin > vmax:
        vmin, vmax = vmax, vmin
    return vmin, vmax


def _resolve_plot_cfg(plot_cfg: PlotConfig | None) -> PlotConfig:
    return plot_cfg if plot_cfg is not None else PlotConfig()


def _mpl_to_plotly_colorscale(cmap_name: str, n: int = 256, discrete_levels: int | None = None):
    cmap = _get_cmap(cmap_name)

    if discrete_levels is not None and discrete_levels >= 2:
        # Piecewise-constant colorscale: duplicate stops at each bin edge so
        # Plotly renders discrete color bands instead of continuous gradients.
        n_bins = int(discrete_levels)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        mids = 0.5 * (edges[:-1] + edges[1:])
        scale = []
        for i in range(n_bins):
            r, g, b, _ = cmap(mids[i])
            color = f"rgb({int(255 * r)},{int(255 * g)},{int(255 * b)})"
            scale.append([float(edges[i]), color])
            scale.append([float(edges[i + 1]), color])
        return scale

    scale = []
    for i in range(n):
        t = i / max(n - 1, 1)
        r, g, b, _ = cmap(t)
        scale.append([t, f"rgb({int(255 * r)},{int(255 * g)},{int(255 * b)})"])
    return scale


def _plotly_colorbar_len(x, y, fig_width: int, fig_height: int,
                         min_len: float = 0.18, max_len: float = 0.92) -> float:
    """Estimate colorbar height so it matches equal-aspect plotted data height."""
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    xf = x_arr[np.isfinite(x_arr)]
    yf = y_arr[np.isfinite(y_arr)]
    if xf.size == 0 or yf.size == 0:
        return max_len

    dx = float(np.max(xf) - np.min(xf))
    dy = float(np.max(yf) - np.min(yf))
    if dx <= 1e-12 or dy <= 1e-12 or fig_width <= 0 or fig_height <= 0:
        return max_len

    # For equal-aspect axes: occupied vertical fraction ~ (dy/dx)*(W/H), capped to 1.
    frac = min(1.0, (dy / dx) * (float(fig_width) / float(fig_height)))
    return float(np.clip(frac, min_len, max_len))


def _plotly_xy_ranges(x, y, pad_frac: float = 0.02):
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    xf = x_arr[np.isfinite(x_arr)]
    yf = y_arr[np.isfinite(y_arr)]
    if xf.size == 0 or yf.size == 0:
        return None, None

    xmin = float(np.min(xf))
    xmax = float(np.max(xf))
    ymin = float(np.min(yf))
    ymax = float(np.max(yf))

    dx = xmax - xmin
    dy = ymax - ymin
    xpad = pad_frac * dx if dx > 0.0 else 1.0
    ypad = pad_frac * dy if dy > 0.0 else 1.0

    return [xmin - xpad, xmax + xpad], [ymin - ypad, ymax + ypad]


def _save_interactive_field(x, y, z, title: str, cmap: str,
                            xlabel: str, ylabel: str, html_path: str,
                            vmin=None, vmax=None, hole_line=None,
                            plot_cfg: PlotConfig | None = None,
                            irregular_coords: bool = False,
                            ref_bc_overlay: dict | None = None,
                            colorbar_title: str | None = None,
                            extra_annotation_lines: str | None = None,
                            xlim: tuple[float, float] | None = None,
                            ylim: tuple[float, float] | None = None):
    if not _interactive_available():
        return

    cfg = _resolve_plot_cfg(plot_cfg)
    cbar_title = colorbar_title if colorbar_title is not None else title
    n_discrete = max(int(cfg.png_contour_levels), 2)
    plotly_scale = _mpl_to_plotly_colorscale(cmap, discrete_levels=n_discrete)

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    z_ma = np.ma.array(z)
    if cfg.interactive_colorbar_len_fraction is None:
        cb_len = _plotly_colorbar_len(x_arr, y_arr, cfg.interactive_width, cfg.interactive_field_height)
    else:
        cb_len = float(np.clip(cfg.interactive_colorbar_len_fraction, 0.18, 0.995))

    if irregular_coords:
        x_flat = x_arr.ravel()
        y_flat = y_arr.ravel()
        z_flat = np.ma.ravel(z_ma)
        z_mask = np.ma.getmaskarray(z_flat)
        z_vals = np.asarray(np.ma.filled(z_flat, np.nan))
        mask = (~z_mask) & np.isfinite(x_flat) & np.isfinite(y_flat) & np.isfinite(z_vals)
        x_plot = x_flat[mask]
        y_plot = y_flat[mask]
        z_plot = z_vals[mask]

        fig = go.Figure(
            data=go.Scattergl(
                x=x_plot,
                y=y_plot,
                mode="markers",
                marker=dict(
                    color=z_plot,
                    colorscale=plotly_scale,
                    autocolorscale=False,
                    cmin=vmin,
                    cmax=vmax,
                    cauto=(vmin is None and vmax is None),
                    size=3,
                    opacity=1.0,
                    colorbar=dict(
                        title=dict(text=cbar_title, side="top"),
                        lenmode="fraction",
                        len=cb_len,
                        y=0.5,
                        yanchor="middle",
                    ),
                ),
                hovertemplate=(
                    "x=%{x:.4g}<br>"
                    "y=%{y:.4g}<br>"
                    "value=%{marker.color:.6g}<extra></extra>"
                ),
                showlegend=False,
            )
        )
    else:
        x1 = x_arr[:, 0]
        y1 = y_arr[0, :]
        z_plot = np.ma.filled(z_ma, np.nan).T
        custom = np.dstack(np.meshgrid(x1, y1, indexing="xy"))

        fig = go.Figure(
            data=go.Heatmap(
                x=x1,
                y=y1,
                z=z_plot,
                zmin=vmin,
                zmax=vmax,
                zauto=(vmin is None and vmax is None),
                colorscale=plotly_scale,
                autocolorscale=False,
                colorbar=dict(
                    title=dict(text=cbar_title, side="top"),
                    lenmode="fraction",
                    len=cb_len,
                    y=0.5,
                    yanchor="middle",
                ),
                customdata=custom,
                hovertemplate=(
                    "x=%{customdata[0]:.4g}<br>"
                    "y=%{customdata[1]:.4g}<br>"
                    "value=%{z:.6g}<extra></extra>"
                ),
            )
        )

    if hole_line is not None:
        hx, hy = hole_line
        hx = np.asarray(hx)
        hy = np.asarray(hy)
        if hx.size > 0 and hy.size > 0:
            if hx[0] != hx[-1] or hy[0] != hy[-1]:
                hx = np.append(hx, hx[0])
                hy = np.append(hy, hy[0])
        fig.add_trace(
            go.Scatter(
                x=hx,
                y=hy,
                mode="lines",
                fill="toself",
                fillcolor="white",
                line=dict(color="black", width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if ref_bc_overlay is not None:
        xl = ref_bc_overlay["x_left"]
        xr = ref_bc_overlay["x_right"]
        yb = ref_bc_overlay["y_bottom"]
        yt = ref_bc_overlay["y_top"]
        hxo = ref_bc_overlay["hole_x"]
        hyo = ref_bc_overlay["hole_y"]
        dash_style = dict(color="rgba(0,0,0,0.22)", width=1.2, dash="dash")
        for xs, ys in [
            (xl[0], xl[1]),
            (xr[0], xr[1]),
            (yb[0], yb[1]),
            (yt[0], yt[1]),
            (hxo, hyo),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dash_style,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    x_range, y_range = _plotly_xy_ranges(x_arr, y_arr)
    if xlim is not None:
        x_range = [float(xlim[0]), float(xlim[1])]
    if ylim is not None:
        y_range = [float(ylim[0]), float(ylim[1])]

    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis=dict(
            range=x_range,
            showgrid=False,
            showline=False,
            showticklabels=False,
            ticks="",
        ),
        yaxis=dict(
            range=y_range,
            showgrid=False,
            showline=False,
            showticklabels=False,
            ticks="",
        ),
        template="plotly_white",
        width=cfg.interactive_width,
        height=cfg.interactive_field_height,
        dragmode="pan",
    )
    if cfg.annotate_field_minmax:
        # Position annotation under colorbar (on the right side)
        # Colorbar is at x ≈ 0.98-1.0, so annotation goes at x ≈ 0.98, lower y
        fig.add_annotation(
            x=0.98,
            y=0.15,
            xref="paper",
            yref="paper",
            xanchor="right",
            yanchor="top",
            align="right",
            text=(
                _field_stats_text(z, digits=int(cfg.field_stats_digits))
                + (f"\n{extra_annotation_lines}" if extra_annotation_lines else "")
            ).replace("\n", "<br>"),
            showarrow=False,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=12),
        )
    if cfg.interactive_lock_aspect:
        # Keep 1:1 physical scaling while using axis-range expansion
        # instead of shrinking the drawing domain (which creates white bands).
        fig.update_yaxes(scaleanchor="x", scaleratio=1, constrain="range")
    fig.write_html(
        html_path,
        include_plotlyjs="cdn",
        config={
            "responsive": cfg.interactive_responsive,
            "scrollZoom": True,
            "modeBarButtonsToAdd": ["pan2d"],
            "displayModeBar": True,
        },
    )


def _principal_directions(a11: np.ndarray, a22: np.ndarray, a12: np.ndarray):
    theta = 0.5 * np.arctan2(2.0 * a12, a11 - a22)
    c = np.cos(theta)
    s = np.sin(theta)
    d1x, d1y = c, s
    d2x, d2y = -s, c
    return d1x, d1y, d2x, d2y


def _vector_segments(x, y, u, v, half_len: float):
    x0 = x - half_len * u
    x1 = x + half_len * u
    y0 = y - half_len * v
    y1 = y + half_len * v
    xs = np.column_stack([x0, x1, np.full_like(x0, np.nan)]).ravel()
    ys = np.column_stack([y0, y1, np.full_like(y0, np.nan)]).ravel()
    return xs, ys


def _vector_target_length(x, y, frac: float = 0.62) -> float:
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not np.any(finite):
        return 1.0

    xf = np.asarray(x_arr[finite], dtype=float)
    yf = np.asarray(y_arr[finite], dtype=float)
    xu = np.unique(np.round(xf, 12))
    yu = np.unique(np.round(yf, 12))

    dx = np.diff(np.sort(xu)) if xu.size > 1 else np.array([], dtype=float)
    dy = np.diff(np.sort(yu)) if yu.size > 1 else np.array([], dtype=float)
    dx = dx[dx > 0.0]
    dy = dy[dy > 0.0]

    candidates = []
    if dx.size:
        candidates.append(float(np.median(dx)))
    if dy.size:
        candidates.append(float(np.median(dy)))

    if candidates:
        return max(min(candidates) * frac, 1e-12)

    bbox = max(float(np.nanmax(xf) - np.nanmin(xf)), float(np.nanmax(yf) - np.nanmin(yf)))
    return max(0.03 * bbox, 1e-12)


def _vector_length_scale(mag, vmin=None, vmax=None,
                         min_frac: float = 1.10, max_frac: float = 3.00):
    m = np.abs(np.asarray(mag, dtype=float))
    finite = np.isfinite(m)
    if not np.any(finite):
        return np.full_like(m, min_frac, dtype=float)

    if vmin is not None or vmax is not None:
        lim = max(abs(float(vmin)) if vmin is not None else 0.0,
                  abs(float(vmax)) if vmax is not None else 0.0)
    else:
        lim = float(np.nanmax(m[finite]))

    if lim <= 1e-12:
        norm = np.zeros_like(m, dtype=float)
    else:
        norm = np.clip(m / lim, 0.0, 1.0)
    return min_frac + (max_frac - min_frac) * norm


def _save_vector_plot_png(x, y, vx, vy, mag, title, out_path,
                          xlabel, ylabel, hole_center, hole_radius,
                          cmap="RdYlGn_r", vmin=None, vmax=None,
                          hole_outline=None,
                          plot_cfg: PlotConfig | None = None,
                          colorbar_title: str | None = None):
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 3.2))
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(vx) & np.isfinite(vy) & np.isfinite(mag)
    x_f = x[finite]
    y_f = y[finite]
    vx_f = vx[finite]
    vy_f = vy[finite]
    mag_f = mag[finite]
    arrow_len = _vector_target_length(x_f, y_f, frac=1.10)
    len_scale = _vector_length_scale(mag_f, vmin=vmin, vmax=vmax)
    u_plot = arrow_len * len_scale * vx_f
    v_plot = arrow_len * len_scale * vy_f
    q = ax.quiver(
        x_f, y_f, u_plot, v_plot, mag_f,
        cmap=_get_cmap(cmap), pivot="tail", angles="xy", scale_units="xy", scale=1.0,
        width=0.0028, headwidth=2.4, headlength=3.0, headaxislength=2.7,
        minshaft=2.0,
        minlength=0.0,
    )
    if vmin is not None or vmax is not None:
        q.set_clim(vmin, vmax)
    if hole_outline is not None:
        hx, hy = hole_outline
        hx = np.asarray(hx)
        hy = np.asarray(hy)
        if hx.size > 0 and hy.size > 0:
            verts = np.column_stack([hx, hy])
            ax.add_patch(Polygon(verts, closed=True, facecolor="white", edgecolor="black", linewidth=1.0, zorder=10))
    else:
        _draw_hole(ax, hole_center, hole_radius)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Axis-locked colorbar: height always matches the plate axis height.
    cax = ax.inset_axes([1.02, 0.0, 0.025, 1.0])
    cb = fig.colorbar(q, cax=cax)
    cb.ax.set_title(colorbar_title if colorbar_title is not None else "value", fontsize=8, pad=3)
    cb.ax.tick_params(labelsize=7)
    _annotate_field_stats_mpl(ax, mag_f, plot_cfg)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _save_vector_plot_html(x, y, vx, vy, mag, title, out_path,
                           xlabel, ylabel, hole_line, vmin=None, vmax=None,
                           cmap="RdYlGn_r", plot_cfg: PlotConfig | None = None,
                           colorbar_title: str | None = None):
    if not _interactive_available():
        return

    cfg = _resolve_plot_cfg(plot_cfg)
    cbar_title = colorbar_title if colorbar_title is not None else "value"
    n_discrete = max(int(cfg.png_contour_levels), 2)
    plotly_scale = _mpl_to_plotly_colorscale(cmap, discrete_levels=n_discrete)

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(vx) & np.isfinite(vy) & np.isfinite(mag)
    x_f = x[finite]
    y_f = y[finite]
    vx_f = vx[finite]
    vy_f = vy[finite]
    m_f = mag[finite]

    arrow_len = _vector_target_length(x_f, y_f, frac=1.10)
    len_scale = _vector_length_scale(m_f, vmin=vmin, vmax=vmax)
    half = 0.80 * arrow_len * len_scale
    x0 = x_f - half * vx_f
    y0 = y_f - half * vy_f
    x1 = x_f + half * vx_f
    y1 = y_f + half * vy_f

    cmin = float(np.nanmin(m_f)) if vmin is None else float(vmin)
    cmax = float(np.nanmax(m_f)) if vmax is None else float(vmax)
    if cmax <= cmin:
        cmax = cmin + 1e-12
    if cfg.interactive_colorbar_len_fraction is None:
        cb_len = _plotly_colorbar_len(x_f, y_f, cfg.interactive_width, cfg.interactive_vector_height)
    else:
        cb_len = float(np.clip(cfg.interactive_colorbar_len_fraction, 0.18, 0.995))

    t = np.clip((m_f - cmin) / (cmax - cmin), 0.0, 1.0)
    cmap_obj = _get_cmap(cmap)
    rgba = cmap_obj(t)
    arrow_colors = [f"rgba({int(255*r)},{int(255*g)},{int(255*b)},0.95)" for r, g, b, _ in rgba]
    xs = np.column_stack([x0, x1, np.full_like(x0, np.nan)]).ravel()
    ys = np.column_stack([y0, y1, np.full_like(y0, np.nan)]).ravel()

    fig = go.Figure()

    # Always draw shafts explicitly so vectors never appear as head-only markers.
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color="rgba(35,35,35,0.92)", width=2.4),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Invisible trace used only to display a proper colorbar in Plotly.
    fig.add_trace(
        go.Scatter(
            x=x_f,
            y=y_f,
            mode="markers",
            marker=dict(
                color=m_f,
                colorscale=plotly_scale,
                autocolorscale=False,
                cmin=cmin,
                cmax=cmax,
                showscale=True,
                size=1,
                opacity=0.0,
                colorbar=dict(
                    title=dict(text=cbar_title, side="top"),
                    lenmode="fraction",
                    len=cb_len,
                    y=0.5,
                    yanchor="middle",
                ),
            ),
            customdata=np.column_stack([x_f, y_f, vx_f, vy_f]),
            hovertemplate=(
                "x=%{customdata[0]:.4g}<br>"
                "y=%{customdata[1]:.4g}<br>"
                "principal=%{marker.color:.6g}<br>"
                "dir=(%{customdata[2]:.4g}, %{customdata[3]:.4g})"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    for xi0, yi0, xi1, yi1, col, ls in zip(x0, y0, x1, y1, arrow_colors, len_scale):
        fig.add_annotation(
            x=float(xi1), y=float(yi1),
            ax=float(xi0), ay=float(yi0),
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=0.70,
            arrowwidth=float(1.0 + 0.8 * np.clip(ls, 0.0, 2.0)),
            arrowcolor=col,
            text="",
        )

    if hole_line is not None:
        hx, hy = hole_line
        hx = np.asarray(hx)
        hy = np.asarray(hy)
        if hx.size > 0 and hy.size > 0:
            if hx[0] != hx[-1] or hy[0] != hy[-1]:
                hx = np.append(hx, hx[0])
                hy = np.append(hy, hy[0])
        fig.add_trace(
            go.Scatter(
                x=hx,
                y=hy,
                mode="lines",
                fill="toself",
                fillcolor="white",
                line=dict(color="black", width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    x_range, y_range = _plotly_xy_ranges(x_f, y_f)

    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis=dict(
            range=x_range,
            showgrid=False,
            showline=False,
            showticklabels=False,
            ticks="",
        ),
        yaxis=dict(
            range=y_range,
            showgrid=False,
            showline=False,
            showticklabels=False,
            ticks="",
        ),
        template="plotly_white",
        width=cfg.interactive_width,
        height=cfg.interactive_vector_height,
        dragmode="pan",
    )
    if cfg.annotate_field_minmax:
        fig.add_annotation(
            x=0.985,
            y=0.01,
            xref="paper",
            yref="paper",
            xanchor="right",
            yanchor="bottom",
            align="right",
            text=_field_stats_text(m_f, digits=int(cfg.field_stats_digits)).replace("\n", "<br>"),
            showarrow=False,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=12),
        )
    if cfg.interactive_lock_aspect:
        fig.update_yaxes(scaleanchor="x", scaleratio=1, constrain="range")
    fig.write_html(
        out_path,
        include_plotlyjs="cdn",
        config={
            "responsive": cfg.interactive_responsive,
            "modeBarButtonsToAdd": ["pan2d"],
            "displayModeBar": True,
        },
    )

def _field_panel(ax, x, y, z, title: str, cmap: str = "RdBu_r",
                 xlabel: str = "x", ylabel: str = "y",
                 xlim=None, ylim=None, vmin=None, vmax=None, aspect: str = "equal",
                 plot_cfg: PlotConfig | None = None):
    """Single contourf panel with colour-bar."""
    z_ma = np.ma.array(z)
    finite = z_ma.compressed()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        finite = np.array([0.0])
    
    # Use provided limits or auto
    cfg = _resolve_plot_cfg(plot_cfg)

    if vmin is None:
        vmin = finite.min()
    if vmax is None:
        vmax = finite.max()
    
    if vmax <= vmin:
        vmax = vmin + 1e-12

    # Discrete contour bands for PNG: controlled by png_contour_levels.
    n_levels = max(int(cfg.png_contour_levels), 2)
    levels = np.linspace(vmin, vmax, n_levels)
    z_plot = np.ma.clip(z_ma, vmin, vmax)
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    cf = ax.contourf(x, y, z_plot, levels=levels, cmap=_get_cmap(cmap), norm=norm)
    ax.set_xlabel(xlabel, fontsize=6)
    ax.set_ylabel(ylabel, fontsize=6)
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
    cb = ax.figure.colorbar(cf, cax=cax, boundaries=levels)
    cb.ax.set_title(title, fontsize=6, pad=2)
    cb.mappable.set_clim(vmin, vmax)
    cb.set_ticks(np.linspace(vmin, vmax, min(7, n_levels)))
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


def _original_bc_overlay(results: dict, n: int = 721) -> dict:
    x_min = float(np.nanmin(results["x"]))
    x_max = float(np.nanmax(results["x"]))
    y_min = float(np.nanmin(results["y"]))
    y_max = float(np.nanmax(results["y"]))
    cx, cy = results["hole_center"]
    r = float(results["hole_radius"])
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    return {
        "x_left": ([x_min, x_min], [y_min, y_max]),
        "x_right": ([x_max, x_max], [y_min, y_max]),
        "y_bottom": ([x_min, x_max], [y_min, y_min]),
        "y_top": ([x_min, x_max], [y_max, y_max]),
        "hole_x": (cx + r * np.cos(theta)).tolist(),
        "hole_y": (cy + r * np.sin(theta)).tolist(),
    }


def _draw_original_bc_overlay(ax, overlay: dict):
    style = dict(color="black", linewidth=0.9, linestyle=(0, (4, 3)), alpha=0.22, zorder=9)
    for xs, ys in [
        overlay["x_left"],
        overlay["x_right"],
        overlay["y_bottom"],
        overlay["y_top"],
        (overlay["hole_x"], overlay["hole_y"]),
    ]:
        ax.plot(xs, ys, **style)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_fields(results: dict, save_dir: str = "results",
                length_unit: str = "m", stress_unit: str = "Pa",
                plot_cfg: PlotConfig | None = None):
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

    _cfg = _resolve_plot_cfg(plot_cfg)
    _cs = _cfg.cmap_stress
    _cd = _cfg.cmap_displacement
    _ce = _cfg.cmap_strain
    field_specs = [
        ("u", f"u ({lu})", _cd),
        ("v", f"v ({lu})", _cd),
        ("umag", f"|u| ({lu})", _cd),
        ("sxx", f"σ_xx ({su})", _cs),
        ("syy", f"σ_yy ({su})", _cs),
        ("sxy", f"σ_xy ({su})", _cs),
        ("exx", "ε_xx", _ce),
        ("eyy", "ε_yy", _ce),
        ("exy", "ε_xy", _ce),
    ]

    xlabel = f"x ({lu})"
    ylabel = f"y ({lu})"

    interactive_dir = _interactive_dir(save_dir)
    hx = hole_center[0] + hole_radius * np.cos(np.linspace(0.0, 2.0 * np.pi, 361))
    hy = hole_center[1] + hole_radius * np.sin(np.linspace(0.0, 2.0 * np.pi, 361))

    plot_data = dict(results)
    plot_data["umag"] = umag

    for key, label, cmap in field_specs:
        if key not in plot_data:
            continue
        field_plot = np.nan_to_num(plot_data[key], nan=0.0)
        vmin, vmax = _resolve_levels(key, field_plot, plot_cfg)
        fig, ax = plt.subplots(1, 1, figsize=(12.0, 2.0))
        _field_panel(
            ax,
            x,
            y,
            field_plot,
            label,
            cmap,
            xlabel,
            ylabel,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
            plot_cfg=plot_cfg,
        )
        _annotate_field_stats_mpl(
            ax,
            field_plot,
            plot_cfg,
        )
        _draw_hole(ax, hole_center, hole_radius)
        plt.tight_layout()
        path = os.path.join(save_dir, f"undeformed_{key}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        html_path = os.path.join(interactive_dir, f"undeformed_{key}.html")
        _save_interactive_field(
            x, y, field_plot, label, cmap,
            xlabel, ylabel, html_path,
            vmin=vmin, vmax=vmax,
            hole_line=(hx, hy),
            plot_cfg=plot_cfg,
            colorbar_title=label,
        )

    print(f"  Field plots -> {save_dir}/")
    if _interactive_available():
        print(f"  Interactive field plots -> {interactive_dir}/")
    else:
        print("  Interactive field plots skipped (install plotly)")


def plot_principal_fields(results: dict, save_dir: str = "results",
                          length_unit: str = "m", stress_unit: str = "Pa",
                          plot_cfg: PlotConfig | None = None):
    """Save principal stress and strain contour plots on undeformed geometry."""
    os.makedirs(save_dir, exist_ok=True)
    x, y = results["x"], results["y"]
    hole_center = results["hole_center"]
    hole_radius = results["hole_radius"]

    _cfg = _resolve_plot_cfg(plot_cfg)
    _cs = _cfg.cmap_stress
    _ce = _cfg.cmap_strain
    field_specs = [
        ("s1", f"σ1 ({stress_unit})", _cs),
        ("s2", f"σ2 ({stress_unit})", _cs),
        ("e1", "ε1", _ce),
        ("e2", "ε2", _ce),
    ]
    xlabel = f"x ({length_unit})"
    ylabel = f"y ({length_unit})"
    interactive_dir = _interactive_dir(save_dir)
    hx = hole_center[0] + hole_radius * np.cos(np.linspace(0.0, 2.0 * np.pi, 361))
    hy = hole_center[1] + hole_radius * np.sin(np.linspace(0.0, 2.0 * np.pi, 361))

    for key, label, cmap in field_specs:
        if key not in results:
            continue
        field_plot = np.nan_to_num(results[key], nan=0.0)
        vmin, vmax = _resolve_levels(key, field_plot, plot_cfg)
        fig, ax = plt.subplots(1, 1, figsize=(12.0, 2.0))
        _field_panel(
            ax,
            x,
            y,
            field_plot,
            label,
            cmap,
            xlabel,
            ylabel,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
            plot_cfg=plot_cfg,
        )
        _annotate_field_stats_mpl(
            ax,
            field_plot,
            plot_cfg,
        )
        _draw_hole(ax, hole_center, hole_radius)
        plt.tight_layout()
        path = os.path.join(save_dir, f"undeformed_{key}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        html_path = os.path.join(interactive_dir, f"undeformed_{key}.html")
        _save_interactive_field(
            x, y, field_plot, label, cmap,
            xlabel, ylabel, html_path,
            vmin=vmin, vmax=vmax,
            hole_line=(hx, hy),
            plot_cfg=plot_cfg,
            colorbar_title=label,
        )

    print(f"  Principal plots -> {save_dir}/")
    if _interactive_available():
        print(f"  Interactive principal plots -> {interactive_dir}/")


def plot_principal_vectors(results: dict, save_dir: str = "results",
                           length_unit: str = "m", stress_unit: str = "Pa",
                           strain_unit: str = "",
                           stride: int = 16,
                           plot_cfg: PlotConfig | None = None):
    """Save principal stress/strain direction vector plots in PNG and HTML."""
    os.makedirs(save_dir, exist_ok=True)

    x = np.asarray(results["x"])
    y = np.asarray(results["y"])
    hole_center = results["hole_center"]
    hole_radius = results["hole_radius"]
    interactive_dir = _interactive_dir(save_dir)

    sxx = np.asarray(results.get("sxx"))
    syy = np.asarray(results.get("syy"))
    sxy = np.asarray(results.get("sxy"))
    exx = np.asarray(results.get("exx"))
    eyy = np.asarray(results.get("eyy"))
    exy = np.asarray(results.get("exy"))

    ds1x, ds1y, ds2x, ds2y = _principal_directions(sxx, syy, sxy)
    de1x, de1y, de2x, de2y = _principal_directions(exx, eyy, exy)

    xq = x[::stride, ::stride]
    yq = y[::stride, ::stride]

    s1 = np.asarray(results.get("s1"))[::stride, ::stride]
    s2 = np.asarray(results.get("s2"))[::stride, ::stride]
    e1 = np.asarray(results.get("e1"))[::stride, ::stride]
    e2 = np.asarray(results.get("e2"))[::stride, ::stride]

    ds1x = ds1x[::stride, ::stride]
    ds1y = ds1y[::stride, ::stride]
    ds2x = ds2x[::stride, ::stride]
    ds2y = ds2y[::stride, ::stride]
    de1x = de1x[::stride, ::stride]
    de1y = de1y[::stride, ::stride]
    de2x = de2x[::stride, ::stride]
    de2y = de2y[::stride, ::stride]

    hx = hole_center[0] + hole_radius * np.cos(np.linspace(0.0, 2.0 * np.pi, 361))
    hy = hole_center[1] + hole_radius * np.sin(np.linspace(0.0, 2.0 * np.pi, 361))

    e1_title = "ε1" if not strain_unit else f"ε1 ({strain_unit})"
    e2_title = "ε2" if not strain_unit else f"ε2 ({strain_unit})"

    specs = [
        ("s1", "undeformed_vector_s1", f"σ1 ({stress_unit})", ds1x, ds1y, s1, _resolve_plot_cfg(plot_cfg).cmap_stress),
        ("s2", "undeformed_vector_s2", f"σ2 ({stress_unit})", ds2x, ds2y, s2, _resolve_plot_cfg(plot_cfg).cmap_stress),
        ("e1", "undeformed_vector_e1", e1_title, de1x, de1y, e1, _resolve_plot_cfg(plot_cfg).cmap_strain),
        ("e2", "undeformed_vector_e2", e2_title, de2x, de2y, e2, _resolve_plot_cfg(plot_cfg).cmap_strain),
    ]

    xlabel = f"x ({length_unit})"
    ylabel = f"y ({length_unit})"

    for field_key, stem, title, vx, vy, mag, cmap in specs:
        vmin, vmax = _resolve_levels(field_key, mag, plot_cfg)
        png_path = os.path.join(save_dir, f"{stem}.png")
        _save_vector_plot_png(
            xq, yq, vx, vy, mag, title, png_path,
            xlabel, ylabel, hole_center, hole_radius,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            plot_cfg=plot_cfg,
            colorbar_title=title,
        )

        html_path = os.path.join(interactive_dir, f"{stem}.html")
        _save_vector_plot_html(
            xq, yq, vx, vy, mag, title, html_path,
            xlabel, ylabel, hole_line=(hx, hy),
            vmin=vmin, vmax=vmax,
            cmap=cmap,
            plot_cfg=plot_cfg,
            colorbar_title=title,
        )

    print(f"  Principal vector plots -> {save_dir}/")
    if _interactive_available():
        print(f"  Interactive principal vectors -> {interactive_dir}/")


def plot_deformed_fields(results: dict, save_dir: str = "results",
                         deformation_scale: float = 500.0,
                         length_unit: str = "m", stress_unit: str = "Pa",
                         plot_cfg: PlotConfig | None = None):
    """Save selected contours on the deformed configuration.

    Required set:
        u, v, |u|, sxx, syy, sxy
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

    sxy_field = results.get("sxy")

    cfg_plot = _resolve_plot_cfg(plot_cfg)

    # spec: (field_key, field_array, output_stem, title, cmap)
    field_specs = [
        ("u", results["u"],   "deformed_u",    f"u ({length_unit})", cfg_plot.cmap_displacement),
        ("v", results["v"],   "deformed_v",    f"v ({length_unit})", cfg_plot.cmap_displacement),
        ("umag", umag,            "deformed_umag", f"|u| ({length_unit})", cfg_plot.cmap_displacement),
        ("sxx", results["sxx"], "deformed_sxx",  f"σ_xx ({stress_unit})", cfg_plot.cmap_stress),
        ("syy", results["syy"], "deformed_syy",  f"σ_yy ({stress_unit})", cfg_plot.cmap_stress),
        ("sxy", sxy_field,       "deformed_sxy",  f"σ_xy ({stress_unit})", cfg_plot.cmap_stress),
        ("s1", results.get("s1"), "deformed_s1",  f"σ1 ({stress_unit})", cfg_plot.cmap_stress),
        ("s2", results.get("s2"), "deformed_s2",  f"σ2 ({stress_unit})", cfg_plot.cmap_stress),
        ("exx", results.get("exx"), "deformed_exx", "ε_xx", cfg_plot.cmap_strain),
        ("eyy", results.get("eyy"), "deformed_eyy", "ε_yy", cfg_plot.cmap_strain),
        ("exy", results.get("exy"), "deformed_exy", "ε_xy", cfg_plot.cmap_strain),
        ("e1", results.get("e1"), "deformed_e1", "ε1", cfg_plot.cmap_strain),
        ("e2", results.get("e2"), "deformed_e2", "ε2", cfg_plot.cmap_strain),
    ]
    xlabel = f"x_deformed ({length_unit})"
    ylabel = f"y_deformed ({length_unit})"
    xlim = (float(np.nanmin(x_def)), float(np.nanmax(x_def)))
    ylim = (float(np.nanmin(y_def)), float(np.nanmax(y_def)))
    interactive_dir = _interactive_dir(save_dir)

    # Build mask from the deformed hole boundary polygon so the plotted hole
    # matches the deformed geometry rather than the undeformed circular mask.
    xb_d, yb_d, _, _ = _deformed_hole_boundary(
        hole_center, hole_radius, deformation_scale, results
    )
    hole_path = Path(np.column_stack([xb_d, yb_d]), closed=True)
    xy_def = np.column_stack([x_def.ravel(), y_def.ravel()])
    hole_mask_def = hole_path.contains_points(xy_def).reshape(x_def.shape)
    bc_overlay = _original_bc_overlay(results) if cfg_plot.show_deformed_reference_bc else None

    for field_data in field_specs:
        if len(field_data) == 5:
            field_key, field, stem, label, cmap = field_data
        else:
            continue
        if field is None:
            continue
        # Keep original NaN mask (undeformed hole and any invalid points) and
        # combine it with the deformed-hole mask to avoid zero-filled streaks.
        field_arr = np.asarray(field)
        orig_nan_mask = np.isnan(field_arr)
        field_filled = np.nan_to_num(field_arr, nan=0.0)
        field_plot = np.ma.array(field_filled, mask=(hole_mask_def | orig_nan_mask))
        vmin, vmax = _resolve_levels(field_key, field_plot, plot_cfg)
        fig, ax = plt.subplots(1, 1, figsize=(12.0, 2.0))
        _field_panel(
            ax,
            x_def,
            y_def,
            field_plot,
            label,
            cmap,
            xlabel,
            ylabel,
            xlim=xlim,
            ylim=ylim,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
            plot_cfg=plot_cfg,
        )
        _annotate_field_stats_mpl(
            ax,
            field_plot,
            plot_cfg,
            extra_lines=f"deformation scale={deformation_scale:g}",
        )
        if bc_overlay is not None:
            _draw_original_bc_overlay(ax, bc_overlay)
        _draw_deformed_hole(ax, hole_center, hole_radius, deformation_scale, results)
        plt.tight_layout()
        path = os.path.join(save_dir, f"{stem}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        html_path = os.path.join(interactive_dir, f"{stem}.html")
        _save_interactive_field(
            x_def, y_def, field_plot, label, cmap,
            xlabel, ylabel, html_path,
            vmin=vmin, vmax=vmax,
            hole_line=(xb_d, yb_d),
            plot_cfg=plot_cfg,
            irregular_coords=True,
            ref_bc_overlay=bc_overlay,
            colorbar_title=label,
            extra_annotation_lines=f"deformation scale={deformation_scale:g}",
        )

    print(f"  Deformed plots -> {save_dir}/")
    if _interactive_available():
        print(f"  Interactive deformed plots -> {interactive_dir}/")


def plot_hole_zoom(results: dict, save_dir: str = "results",
                   length_unit: str = "m", stress_unit: str = "Pa",
                   plot_cfg: PlotConfig | None = None):
    """Save zoomed stress/strain plots around the hole."""
    os.makedirs(save_dir, exist_ok=True)
    x, y = results["x"], results["y"]
    hole_center = results["hole_center"]
    hole_radius = results["hole_radius"]
    lu = length_unit
    su = stress_unit

    _cfg = _resolve_plot_cfg(plot_cfg)
    field_specs = [
        ("sxx", f"σ_xx ({stress_unit})", _cfg.cmap_stress),
        ("syy", f"σ_yy ({stress_unit})", _cfg.cmap_stress),
        ("sxy", f"σ_xy ({stress_unit})", _cfg.cmap_stress),
        ("s1", f"σ1 ({stress_unit})", _cfg.cmap_stress),
        ("s2", f"σ2 ({stress_unit})", _cfg.cmap_stress),
        ("exx", "ε_xx", _cfg.cmap_strain),
        ("eyy", "ε_yy", _cfg.cmap_strain),
        ("exy", "ε_xy", _cfg.cmap_strain),
        ("e1", "ε1", _cfg.cmap_strain),
        ("e2", "ε2", _cfg.cmap_strain),
    ]
    xlabel = f"x ({lu})"
    ylabel = f"y ({lu})"
    interactive_dir = _interactive_dir(save_dir)
    hx = hole_center[0] + hole_radius * np.cos(np.linspace(0.0, 2.0 * np.pi, 361))
    hy = hole_center[1] + hole_radius * np.sin(np.linspace(0.0, 2.0 * np.pi, 361))

    x_min = float(np.nanmin(np.asarray(x)))
    x_max = float(np.nanmax(np.asarray(x)))
    y_min = float(np.nanmin(np.asarray(y)))
    y_max = float(np.nanmax(np.asarray(y)))
    zoom_half = float(_cfg.hole_zoom_radius_factor) * float(hole_radius)
    xlim = (
        max(x_min, float(hole_center[0]) - zoom_half),
        min(x_max, float(hole_center[0]) + zoom_half),
    )
    ylim = (
        max(y_min, float(hole_center[1]) - zoom_half),
        min(y_max, float(hole_center[1]) + zoom_half),
    )

    for key, label, cmap in field_specs:
        if key not in results:
            continue
        field_plot = np.nan_to_num(results[key], nan=0.0)
        vmin, vmax = _resolve_levels(key, field_plot, plot_cfg)
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.0))
        _field_panel(
            ax,
            x,
            y,
            field_plot,
            label,
            cmap,
            xlabel,
            ylabel,
            xlim=xlim,
            ylim=ylim,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
            plot_cfg=plot_cfg,
        )
        _annotate_field_stats_mpl(ax, field_plot, plot_cfg)
        _draw_hole(ax, hole_center, hole_radius)
        plt.tight_layout()
        path = os.path.join(save_dir, f"undeformed_zoom_{key}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        html_path = os.path.join(interactive_dir, f"undeformed_zoom_{key}.html")
        _save_interactive_field(
            x, y, field_plot, label, cmap,
            xlabel, ylabel, html_path,
            vmin=vmin, vmax=vmax,
            hole_line=(hx, hy),
            plot_cfg=plot_cfg,
            colorbar_title=label,
            xlim=xlim,
            ylim=ylim,
        )

    print(f"  Zoom plots  -> {save_dir}/")
    if _interactive_available():
        print(f"  Interactive zoom plots -> {interactive_dir}/")


def plot_loss_history(history: dict, save_dir: str = "results",
                      plot_cfg: PlotConfig | None = None):
    """Log-scale loss curves for total loss and individual components."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = np.arange(1, len(history["total"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.semilogy(epochs, history["total"], "k-", lw=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
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
    ax2.legend(loc="right", fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "loss_history.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Loss curve  -> {path}")

    if _interactive_available():
        epochs_list = epochs.tolist()
        interactive_dir = _interactive_dir(save_dir)
        html_path = os.path.join(interactive_dir, "loss_history.html")

        fig_html = make_subplots(rows=1, cols=2, subplot_titles=("", ""))
        fig_html.add_trace(
            go.Scatter(x=epochs_list, y=history["total"], mode="lines", name="total", line=dict(color="black")),
            row=1, col=1,
        )
        for comp, col in component_colors.items():
            if comp in history and len(history[comp]):
                fig_html.add_trace(
                    go.Scatter(x=epochs_list, y=history[comp], mode="lines", name=comp, line=dict(color=col)),
                    row=1, col=2,
                )

        fig_html.update_yaxes(type="log", row=1, col=1)
        fig_html.update_yaxes(type="log", row=1, col=2)
        fig_html.update_xaxes(title_text="Epoch", row=1, col=1)
        fig_html.update_xaxes(title_text="Epoch", row=1, col=2)
        fig_html.update_yaxes(title_text="Loss", row=1, col=1)
        fig_html.update_yaxes(title_text="Loss", row=1, col=2)
        cfg = _resolve_plot_cfg(plot_cfg)
        fig_html.update_layout(
            template="plotly_white",
            width=cfg.interactive_width,
            height=cfg.interactive_misc_height,
            dragmode="pan",
        )
        fig_html.update_xaxes(showgrid=False, showline=False, showticklabels=False, ticks="")
        fig_html.update_yaxes(showgrid=False, showline=False, showticklabels=False, ticks="")
        fig_html.write_html(
            html_path,
            include_plotlyjs="cdn",
            config={
                "responsive": cfg.interactive_responsive,
                "modeBarButtonsToAdd": ["pan2d"],
                "displayModeBar": True,
            },
        )
        print(f"  Interactive loss curve -> {html_path}")


def plot_sampling_points(batch: tuple, cfg_p, save_dir: str = "results",
                         length_unit: str = "mm",
                         plot_cfg: PlotConfig | None = None):
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
    ax.tick_params(labelsize=7)
    ax.legend(loc="right", fontsize=6.5, markerscale=3, framealpha=0.85)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = os.path.join(save_dir, "sampling_points.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Sampling plot -> {path}")

    if _interactive_available():
        interactive_dir = _interactive_dir(save_dir)
        html_path = os.path.join(interactive_dir, "sampling_points.html")

        fig_html = go.Figure()
        for label, (px, py), color, ms, alpha in categories:
            fig_html.add_trace(
                go.Scattergl(
                    x=px,
                    y=py,
                    mode="markers",
                    name=label,
                    marker=dict(color=color, size=max(4, ms), opacity=alpha),
                    hovertemplate="x=%{x:.4g}<br>y=%{y:.4g}<extra></extra>",
                )
            )

        theta = np.linspace(0.0, 2.0 * np.pi, 361)
        hx = xc + r * np.cos(theta)
        hy = yc + r * np.sin(theta)
        fig_html.add_trace(
            go.Scatter(
                x=hx,
                y=hy,
                mode="lines",
                name="Hole geometry",
                line=dict(color="black", width=2),
                hoverinfo="skip",
            )
        )

        cfg = _resolve_plot_cfg(plot_cfg)
        fig_html.update_layout(
            xaxis_title=f"x ({length_unit})",
            yaxis_title=f"y ({length_unit})",
            template="plotly_white",
            width=cfg.interactive_width,
            height=cfg.interactive_misc_height,
            dragmode="pan",
        )
        fig_html.update_xaxes(showgrid=False, showline=False, showticklabels=False, ticks="")
        fig_html.update_yaxes(showgrid=False, showline=False, showticklabels=False, ticks="")
        if cfg.interactive_lock_aspect:
            fig_html.update_yaxes(scaleanchor="x", scaleratio=1, constrain="range")
        fig_html.write_html(
            html_path,
            include_plotlyjs="cdn",
            config={
                "responsive": cfg.interactive_responsive,
                "modeBarButtonsToAdd": ["pan2d"],
                "displayModeBar": True,
            },
        )
        print(f"  Interactive sampling plot -> {html_path}")
