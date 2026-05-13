"""Evaluation helpers for the plate-with-hole elasticity PINN.

The trained model is evaluated on regular grids in physical units. Since the
problem contains a circular hole and no closed-form analytical solution is
used in this code path, the outputs focus on physically meaningful summaries:
field plots, near-hole zooms, and stress metrics.
"""

import jax
import jax.numpy as jnp
import numpy as np

from config import ProblemConfig, NetworkConfig
from physics import _net_uv, _stress_at, _stiffness


def _principal_values_2x2(a11: np.ndarray, a22: np.ndarray, a12: np.ndarray):
    """Return principal values (max/min eigenvalues) of 2x2 symmetric tensors.

    Tensor form:
        [a11  a12]
        [a12  a22]
    """
    avg = 0.5 * (a11 + a22)
    rad = np.sqrt((0.5 * (a11 - a22)) ** 2 + a12 ** 2)
    return avg + rad, avg - rad


def _mask_inside_hole(cfg_p: ProblemConfig, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return boolean mask for points inside the physical circular hole."""
    xc = 0.5 * cfg_p.L
    yc = 0.5 * cfg_p.H
    return (x - xc) ** 2 + (y - yc) ** 2 <= cfg_p.hole_radius ** 2


def _evaluate_grid(params, model, cfg_p: ProblemConfig, cfg_net: NetworkConfig,
                   xi_1d: np.ndarray, eta_1d: np.ndarray):
    """Evaluate fields on a supplied normalised grid."""
    use_hard_bc = cfg_net.use_hard_bc
    C11, C12, C33 = _stiffness(cfg_p.nu, cfg_p.mode)
    u_ref = cfg_p.u_ref
    s0 = cfg_p.sigma0

    XI, ETA = np.meshgrid(xi_1d, eta_1d, indexing="ij")
    pts = jnp.array(np.stack([XI.ravel(), ETA.ravel()], axis=-1))

    uv_hat = jax.vmap(lambda xy: _net_uv(params, model, xy, use_hard_bc))(pts)
    stresses = jax.vmap(
        lambda xy: _stress_at(params, model, xy, C11, C12, C33, use_hard_bc)
    )(pts)

    shape = XI.shape
    x = XI * cfg_p.L
    y = ETA * cfg_p.L
    fields = {
        "xi": XI,
        "eta": ETA,
        "x": x,
        "y": y,
        "u": np.array(uv_hat[:, 0]).reshape(shape) * u_ref,
        "v": np.array(uv_hat[:, 1]).reshape(shape) * u_ref,
        "sxx": np.array(stresses[:, 0]).reshape(shape) * s0,
        "syy": np.array(stresses[:, 1]).reshape(shape) * s0,
        "sxy": np.array(stresses[:, 2]).reshape(shape) * s0,
    }

    # Recover normalised strain from normalised stress via constitutive inverse.
    # Then convert to physical small strain using eps = (u_ref / L) * eps_hat.
    sxx_hat = np.array(stresses[:, 0]).reshape(shape)
    syy_hat = np.array(stresses[:, 1]).reshape(shape)
    txy_hat = np.array(stresses[:, 2]).reshape(shape)

    det = C11 * C11 - C12 * C12
    exx_hat = (C11 * sxx_hat - C12 * syy_hat) / det
    eyy_hat = (-C12 * sxx_hat + C11 * syy_hat) / det
    gxy_hat = txy_hat / C33

    strain_scale = u_ref / cfg_p.L
    exx = exx_hat * strain_scale
    eyy = eyy_hat * strain_scale
    exy = 0.5 * gxy_hat * strain_scale

    s1, s2 = _principal_values_2x2(fields["sxx"], fields["syy"], fields["sxy"])
    e1, e2 = _principal_values_2x2(exx, eyy, exy)

    fields["exx"] = exx
    fields["eyy"] = eyy
    fields["exy"] = exy
    fields["s1"] = s1
    fields["s2"] = s2
    fields["e1"] = e1
    fields["e2"] = e2

    hole_mask = _mask_inside_hole(cfg_p, x, y)
    fields["hole_mask"] = hole_mask
    fields["hole_center"] = (0.5 * cfg_p.L, 0.5 * cfg_p.H)
    fields["hole_radius"] = cfg_p.hole_radius

    # Evaluate displacement directly on a dense hole circumference so
    # post-processing can use a smooth deformed hole boundary independent of
    # coarse grid interpolation artifacts.
    theta = np.linspace(0.0, 2.0 * np.pi, 1440, endpoint=False)
    xb = fields["hole_center"][0] + cfg_p.hole_radius * np.cos(theta)
    yb = fields["hole_center"][1] + cfg_p.hole_radius * np.sin(theta)
    pts_b = jnp.array(np.stack([xb / cfg_p.L, yb / cfg_p.L], axis=-1))
    uv_b_hat = jax.vmap(lambda xy: _net_uv(params, model, xy, use_hard_bc))(pts_b)
    fields["hole_boundary_x"] = xb
    fields["hole_boundary_y"] = yb
    fields["hole_boundary_u"] = np.array(uv_b_hat[:, 0]) * u_ref
    fields["hole_boundary_v"] = np.array(uv_b_hat[:, 1]) * u_ref

    for key in ("u", "v", "sxx", "syy", "sxy", "exx", "eyy", "exy", "s1", "s2", "e1", "e2"):
        fields[key] = np.where(hole_mask, np.nan, fields[key])

    return fields


def evaluate_on_grid(params, model,
                     cfg_p: ProblemConfig, cfg_net: NetworkConfig,
                     nx: int = 401, ny: int = 161):
    """Evaluate fields on a full-domain grid."""
    xi_1d = np.linspace(0.0, 1.0, nx)
    eta_1d = np.linspace(0.0, cfg_p.eta_max, ny)
    return _evaluate_grid(params, model, cfg_p, cfg_net, xi_1d, eta_1d)


def evaluate_near_hole(params, model,
                       cfg_p: ProblemConfig, cfg_net: NetworkConfig,
                       half_width_radii: float = 2.5,
                       nx: int = 401, ny: int = 401):
    """Evaluate fields on a zoomed grid around the hole."""
    xc = cfg_p.hole_xi_c
    yc = cfg_p.hole_eta_c
    radius = cfg_p.hole_rc
    xi_1d = np.linspace(xc - half_width_radii * radius, xc + half_width_radii * radius, nx)
    eta_1d = np.linspace(yc - half_width_radii * radius, yc + half_width_radii * radius, ny)
    xi_1d = np.clip(xi_1d, 0.0, 1.0)
    eta_1d = np.clip(eta_1d, 0.0, cfg_p.eta_max)
    return _evaluate_grid(params, model, cfg_p, cfg_net, xi_1d, eta_1d)


def compute_summary_metrics(res: dict) -> dict:
    """Return coarse summary metrics for logging and sanity checks."""
    return {
        "u_max": float(np.nanmax(np.abs(res["u"]))),
        "v_max": float(np.nanmax(np.abs(res["v"]))),
        "sxx_max": float(np.nanmax(res["sxx"])),
        "syy_min": float(np.nanmin(res["syy"])),
        "sxy_abs_max": float(np.nanmax(np.abs(res["sxy"]))),
    }
