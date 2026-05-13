"""Collocation-point sampling in normalised coordinates.

Normalisation:  xi = x / L,  eta = y / L
Domain:         xi  in [0, 1],  eta in [0, H/L]
Hole:           circle centred at (0.5, eta_max/2) with normalised radius R/L

Near-hole enrichment
--------------------
The Kirsch stress concentration has a characteristic length scale R (the hole
radius).  With R/L = 0.01 the hole region occupies only ≈0.03 % of the domain
area, so a purely uniform interior sample gives fewer than ~15 PDE points
within 3R of the hole — far too few to constrain the equilibrium equations
there.  To fix this, `sample_interior` dedicates a configurable fraction of
its points to a dedicated near-hole annulus (R < r < kR). These points are
mixed into the same interior batch and weighted identically by the PDE loss.
"""

import numpy as np
import jax
import jax.numpy as jnp

from config import ProblemConfig, TrainingConfig


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _sample_bulk_interior(cfg_p: ProblemConfig, n: int, key: jax.Array) -> jnp.ndarray:
    """Sample *n* points uniformly inside the full domain, excluding the hole.

    Uses 4× oversampling then rejection.  Shape: (n, 2).
    """
    xi_c  = cfg_p.hole_xi_c
    eta_c = cfg_p.hole_eta_c
    r_c   = cfg_p.hole_rc

    n_over = max(4 * n, 512)
    k1, k2 = jax.random.split(key)
    xi  = np.array(jax.random.uniform(k1, (n_over,), minval=0.0, maxval=1.0))
    eta = np.array(jax.random.uniform(k2, (n_over,), minval=0.0, maxval=cfg_p.eta_max))

    outside = (xi - xi_c) ** 2 + (eta - eta_c) ** 2 > r_c ** 2
    xi_ok  = xi[outside][:n]
    eta_ok = eta[outside][:n]

    if len(xi_ok) < n:
        raise RuntimeError(
            f"_sample_bulk_interior: only {len(xi_ok)} valid points after hole "
            f"rejection (need {n}). Increase oversampling factor."
        )

    return jnp.array(np.stack([xi_ok, eta_ok], axis=-1))


def _sample_near_hole_annulus(cfg_p: ProblemConfig, n: int, key: jax.Array,
                              outer_radius_mult: float = 3.0) -> jnp.ndarray:
    """Sample *n* points uniformly (by area) in the annulus R < r < kR.

    Uses the inverse-CDF trick for 2-D polar uniform sampling:
        r ~ sqrt( U(r_min², r_max²) )
    which ensures equal probability per unit area.  Shape: (n, 2).
    """
    xi_c    = cfg_p.hole_xi_c
    eta_c   = cfg_p.hole_eta_c
    r_c     = cfg_p.hole_rc
    r_outer = outer_radius_mult * r_c

    k1, k2 = jax.random.split(key)
    r     = jnp.sqrt(jax.random.uniform(k1, (n,), minval=r_c ** 2, maxval=r_outer ** 2))
    theta = jax.random.uniform(k2, (n,), minval=0.0, maxval=2.0 * jnp.pi)

    xi  = xi_c  + r * jnp.cos(theta)
    eta = eta_c + r * jnp.sin(theta)

    xi  = jnp.clip(xi,  0.0, 1.0)
    eta = jnp.clip(eta, 0.0, cfg_p.eta_max)

    return jnp.stack([xi, eta], axis=-1)


# ---------------------------------------------------------------------------
# Public sampling functions
# ---------------------------------------------------------------------------

def sample_interior(cfg_p: ProblemConfig, n: int, key: jax.Array, cfg_t: TrainingConfig) -> jnp.ndarray:
    """Sample *n* interior collocation points with near-hole enrichment.

    A configurable fraction of points are drawn from the near-hole annulus
    R < r < kR; the remaining points are drawn uniformly from the bulk domain
    (excluding the hole circle).  All points share the same PDE loss weight.
    Shape: (n, 2).
    """
    near_frac = float(np.clip(cfg_t.near_hole_fraction, 0.0, 0.95))
    outer_mult = float(max(cfg_t.near_hole_outer_mult, 1.05))
    n_near = int(round(n * near_frac))
    n_bulk = n - n_near

    k_bulk, k_near = jax.random.split(key)
    bulk_pts = _sample_bulk_interior(cfg_p, n_bulk, k_bulk)
    near_pts = _sample_near_hole_annulus(cfg_p, n_near, k_near, outer_radius_mult=outer_mult)
    return jnp.concatenate([bulk_pts, near_pts], axis=0)


def sample_hole_boundary(cfg_p: ProblemConfig, n: int, key: jax.Array) -> jnp.ndarray:
    """Sample *n* points uniformly on the circular hole circumference.

    Returns (n, 4) array with columns  [xi, eta, nx, ny]  where (nx, ny) is
    the outward unit normal to the hole surface.
    """
    xi_c  = cfg_p.hole_xi_c
    eta_c = cfg_p.hole_eta_c
    r_c   = cfg_p.hole_rc

    theta = jax.random.uniform(key, (n,), minval=0.0, maxval=2.0 * jnp.pi)
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)

    xi  = xi_c  + r_c * cos_t
    eta = eta_c + r_c * sin_t
    return jnp.stack([xi, eta, cos_t, sin_t], axis=-1)   # (n, 4)


def sample_midline(cfg_p: ProblemConfig, n: int, key: jax.Array) -> jnp.ndarray:
    """Sample *n* points on the symmetry line y = H/2, *excluding* the hole.

    The midline eta = hole_eta_c passes through the hole centre.  Points with
    |xi − xi_c| < r_c lie inside the hole (no material) and must be excluded
    before applying the v = 0 symmetry BC.  Uses 2× oversampling + rejection.
    Shape: (n, 2).
    """
    xi_c = cfg_p.hole_xi_c
    r_c  = cfg_p.hole_rc

    n_over = 2 * n
    xi_all = np.array(jax.random.uniform(key, (n_over,), minval=0.0, maxval=1.0))
    outside = np.abs(xi_all - xi_c) > r_c      # exclude hole interior on midline
    xi_ok   = xi_all[outside][:n]

    if len(xi_ok) < n:
        raise RuntimeError(
            f"sample_midline: only {len(xi_ok)} valid points after hole "
            f"exclusion (need {n})."
        )

    eta_ok = np.full(n, cfg_p.hole_eta_c)
    return jnp.array(np.stack([xi_ok, eta_ok], axis=-1))


def sample_boundaries(cfg_p: ProblemConfig, n: int, key: jax.Array):
    """Sample *n* points on each of the four boundary edges.

    Returns
    -------
    pts_left  : (n, 2)  xi = 0
    pts_right : (n, 2)  xi = 1
    pts_bot   : (n, 2)  eta = 0
    pts_top   : (n, 2)  eta = H/L
    """
    keys = jax.random.split(key, 4)
    eta_max = cfg_p.eta_max

    eta_l = jax.random.uniform(keys[0], (n,), minval=0.0, maxval=eta_max)
    pts_left = jnp.stack([jnp.zeros(n), eta_l], axis=-1)

    eta_r = jax.random.uniform(keys[1], (n,), minval=0.0, maxval=eta_max)
    pts_right = jnp.stack([jnp.ones(n), eta_r], axis=-1)

    xi_b = jax.random.uniform(keys[2], (n,), minval=0.0, maxval=1.0)
    pts_bot = jnp.stack([xi_b, jnp.zeros(n)], axis=-1)

    xi_t = jax.random.uniform(keys[3], (n,), minval=0.0, maxval=1.0)
    pts_top = jnp.stack([xi_t, jnp.full(n, eta_max)], axis=-1)

    return pts_left, pts_right, pts_bot, pts_top


def get_batch(cfg_p: ProblemConfig, cfg_t: TrainingConfig, key: jax.Array):
    """Return a complete batch:
    (pts_int, pts_left, pts_right, pts_bot, pts_top, pts_hole, pts_mid)
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)
    pts_int  = sample_interior(cfg_p, cfg_t.n_interior, k1, cfg_t)
    pts_left, pts_right, pts_bot, pts_top = sample_boundaries(cfg_p, cfg_t.n_boundary, k2)
    pts_hole = sample_hole_boundary(cfg_p, cfg_t.n_hole, k3)
    pts_mid = sample_midline(cfg_p, cfg_t.n_midline, k4)
    return pts_int, pts_left, pts_right, pts_bot, pts_top, pts_hole, pts_mid
