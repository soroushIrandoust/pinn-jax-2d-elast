"""PDE residuals and boundary-condition losses for 2-D linear elasticity.

Normalisation
-------------
    xi   = x / L          (xi  in [0, 1])
    eta  = y / L          (eta in [0, H/L])
    u_hat = u / u_ref     u_ref = sigma0 * L / E
    v_hat = v / u_ref
    s_hat = sigma / sigma0

With this choice every term in the governing equations is O(1) and the
equilibrium equations take the parameter-free form:

    d(s_xx) / d(xi)  +  d(t_xy) / d(eta)  =  0
    d(t_xy) / d(xi)  +  d(s_yy) / d(eta)  =  0

Plane-stress constitutive relations (normalised):
    s_xx = C11 * exx + C12 * eyy
    s_yy = C12 * exx + C11 * eyy
    t_xy = C33 * gxy          (gxy = du_hat/deta + dv_hat/dxi)

where  C11 = 1/(1-nu^2),  C12 = nu/(1-nu^2),  C33 = 1/(2*(1+nu)).

Hard-BC ansatz
--------------
When use_hard_bc=True:
    u_hat(xi, eta) = xi * psi_u(xi, eta)    --> guarantees u_hat(0, eta) = 0
    v_hat(xi, eta) = xi * psi_v(xi, eta)    --> guarantees v_hat(0, eta) = 0
Both u=0 and v=0 are enforced exactly along the entire left edge (x=0).
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from config import ProblemConfig, TrainingConfig


# ---------------------------------------------------------------------------
# Constitutive model
# ---------------------------------------------------------------------------

def _stiffness(nu: float, mode: str):
    """Normalised stiffness scalars C11, C12, C33."""
    if mode == "plane_stress":
        C11 = 1.0 / (1.0 - nu ** 2)
        C12 = nu  / (1.0 - nu ** 2)
        C33 = 0.5 / (1.0 + nu)
    else:  # plane_strain
        lam = nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu  = 0.5 / (1.0 + nu)
        C11 = lam + 2.0 * mu
        C12 = lam
        C33 = mu
    return C11, C12, C33


def _stresses_from_jacobian(J, C11, C12, C33):
    """Compute normalised stress vector [s_xx, s_yy, t_xy] from the 2x2
    displacement Jacobian  J[i, j] = d(uv_hat[i]) / d(xi_eta[j])."""
    exx = J[0, 0]
    eyy = J[1, 1]
    gxy = J[0, 1] + J[1, 0]   # engineering shear strain
    s_xx = C11 * exx + C12 * eyy
    s_yy = C12 * exx + C11 * eyy
    t_xy = C33 * gxy
    return jnp.array([s_xx, s_yy, t_xy])


# ---------------------------------------------------------------------------
# Network evaluation  (single point, functional API)
# ---------------------------------------------------------------------------

def _net_uv(params, model: nn.Module, xi_eta: jnp.ndarray,
            use_hard_bc: bool) -> jnp.ndarray:
    """Return normalised displacements [u_hat, v_hat] at a single point."""
    raw = model.apply(params, xi_eta[None])[0]   # shape (2,)
    scaled = jnp.array([xi_eta[0] * raw[0], xi_eta[0] * raw[1]])
    return scaled if use_hard_bc else raw


# ---------------------------------------------------------------------------
# Stress field  (single point, differentiable)
# ---------------------------------------------------------------------------

def _stress_at(params, model, xi_eta, C11, C12, C33, use_hard_bc):
    """Return normalised stress vector [s_xx, s_yy, t_xy] at one point.

    This function is differentiable w.r.t. xi_eta (used in pde_residuals).
    """
    def uv_fn(xy):
        return _net_uv(params, model, xy, use_hard_bc)

    J = jax.jacfwd(uv_fn)(xi_eta)              # (2, 2)  — first derivatives
    return _stresses_from_jacobian(J, C11, C12, C33)


# ---------------------------------------------------------------------------
# PDE residuals  (interior collocation points)
# ---------------------------------------------------------------------------

def pde_residuals(params, model, pts_int, C11, C12, C33, use_hard_bc):
    """Equilibrium residuals [rx, ry] at every interior point.

    Parameters
    ----------
    pts_int : (n, 2) normalised interior coordinates
    Returns
    -------
    residuals : (n, 2)
    """
    def single(xi_eta):
        def stress_fn(xy):
            return _stress_at(params, model, xy, C11, C12, C33, use_hard_bc)

        # J_s[i, j] = d(stress[i]) / d(xi_eta[j])  — second derivatives
        J_s = jax.jacfwd(stress_fn)(xi_eta)    # (3, 2)
        rx = J_s[0, 0] + J_s[2, 1]            # d(s_xx)/d(xi) + d(t_xy)/d(eta)
        ry = J_s[2, 0] + J_s[1, 1]            # d(t_xy)/d(xi) + d(s_yy)/d(eta)
        return jnp.array([rx, ry])

    return jax.vmap(single)(pts_int)           # (n, 2)


# ---------------------------------------------------------------------------
# Boundary condition losses
# ---------------------------------------------------------------------------

def bc_left_displacement(params, model, pts_left, use_hard_bc):
    """Left BC  u_hat = 0.  Zero by construction when use_hard_bc=True."""
    if use_hard_bc:
        return 0.0
    uvs = jax.vmap(lambda xy: _net_uv(params, model, xy, False))(pts_left)
    return jnp.mean(uvs[:, 0] ** 2)


def bc_hole_traction(params, model, pts_hole, C11, C12, C33, use_hard_bc):
    """Hole traction-free BC  sigma·n = 0 on the circular boundary."""
    def single(pt):
        xi_eta = pt[:2]
        nx, ny = pt[2], pt[3]
        s_xx, s_yy, t_xy = _stress_at(
            params, model, xi_eta, C11, C12, C33, use_hard_bc
        )
        tx = s_xx * nx + t_xy * ny
        ty = t_xy * nx + s_yy * ny
        return tx ** 2 + ty ** 2

    return jnp.mean(jax.vmap(single)(pts_hole))


def bc_midline_v(params, model, pts_mid, use_hard_bc):
    """Symmetry BC  v_hat(xi, eta_mid) = 0."""
    v_vals = jax.vmap(lambda xy: _net_uv(params, model, xy, use_hard_bc)[1])(pts_mid)
    return jnp.mean(v_vals ** 2)


def bc_right_traction(params, model, pts_right, C11, C12, C33, use_hard_bc):
    """Right BC  s_xx = 1  (normalised applied stress),  t_xy = 0."""
    stresses = jax.vmap(
        lambda xy: _stress_at(params, model, xy, C11, C12, C33, use_hard_bc)
    )(pts_right)
    loss_sxx = jnp.mean((stresses[:, 0] - 1.0) ** 2)
    loss_sxy = jnp.mean(stresses[:, 2] ** 2)
    return loss_sxx + loss_sxy


def bc_traction_free(params, model, pts, C11, C12, C33, use_hard_bc):
    """Top / bottom traction-free BCs  s_yy = 0,  t_xy = 0."""
    stresses = jax.vmap(
        lambda xy: _stress_at(params, model, xy, C11, C12, C33, use_hard_bc)
    )(pts)
    return jnp.mean(stresses[:, 1] ** 2 + stresses[:, 2] ** 2)


# ---------------------------------------------------------------------------
# Aggregate weighted loss
# ---------------------------------------------------------------------------

def total_loss(params, model, batch, cfg_p: ProblemConfig, cfg_t: TrainingConfig,
               use_hard_bc: bool):
    """Return (scalar_loss, info_dict).

    batch = (pts_int, pts_left, pts_right, pts_bot, pts_top, pts_hole, pts_mid)
    """
    pts_int, pts_left, pts_right, pts_bot, pts_top, pts_hole, pts_mid = batch
    C11, C12, C33 = _stiffness(cfg_p.nu, cfg_p.mode)

    # PDE
    res = pde_residuals(params, model, pts_int, C11, C12, C33, use_hard_bc)
    loss_pde = jnp.mean(res ** 2)

    # Left Dirichlet
    loss_left = bc_left_displacement(params, model, pts_left, use_hard_bc)

    # Right traction
    loss_right = bc_right_traction(
        params, model, pts_right, C11, C12, C33, use_hard_bc
    )

    # Top + bottom traction-free
    pts_tb = jnp.concatenate([pts_top, pts_bot], axis=0)
    loss_tb = bc_traction_free(params, model, pts_tb, C11, C12, C33, use_hard_bc)

    # Hole traction-free
    loss_hole = bc_hole_traction(
        params, model, pts_hole, C11, C12, C33, use_hard_bc
    )

    # Midline symmetry
    loss_mid = bc_midline_v(params, model, pts_mid, use_hard_bc)

    total = (
        cfg_t.w_pde          * loss_pde
        + cfg_t.w_bc_disp    * loss_left
        + cfg_t.w_bc_traction * loss_right
        + cfg_t.w_bc_tb      * loss_tb
        + cfg_t.w_bc_hole    * loss_hole
        + cfg_t.w_bc_mid     * loss_mid
    )

    info = {
        "pde":       loss_pde,
        "bc_left":   loss_left,
        "bc_right":  loss_right,
        "bc_tb":     loss_tb,
        "bc_hole":   loss_hole,
        "bc_mid":    loss_mid,
    }
    return total, info
