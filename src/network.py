"""Flax neural-network architectures for the elasticity PINN.

Two architectures are provided:

MLP
    Plain fully-connected network.  Susceptible to spectral bias — it
    preferentially learns low-frequency functions and struggles to represent
    the sharp stress-concentration gradients near the hole.

FourierMLP  (default when NetworkConfig.n_fourier > 0)
    Input coordinates are lifted to a high-dimensional Fourier feature
    space before entering the MLP.  This is the standard cure for
    spectral bias in PINNs and is essential for resolving the Kirsch-type
    stress concentration around the hole.

    Both normalised coordinates are first mapped to [0, 1]:
        xi_n  = xi                    (already in [0, 1])
        eta_n = eta / eta_max         (eta_max = H/L)

    Then the feature vector is formed from the raw coordinates plus Fourier
    bands:
        phi = [xi_n, eta_n,
               sin(pi*xi_n),   cos(pi*xi_n),   sin(pi*eta_n),   cos(pi*eta_n),
               sin(2pi*xi_n),  cos(2pi*xi_n),  sin(2pi*eta_n),  cos(2pi*eta_n),
               ...  up to n_fourier bands  ...]
        dim(phi) = 2 + 4 * n_fourier

    With n_fourier=8 the input dimension is 34; a [128,128,128,128] MLP on
    top has ~80 k parameters — well within GPU memory.
"""

from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn

from config import NetworkConfig, ProblemConfig

_ACTIVATIONS: dict = {
    "tanh":  nn.tanh,
    "swish": nn.swish,
    "gelu":  nn.gelu,
}


class MLP(nn.Module):
    """Plain fully-connected MLP with Glorot-normal initialisation."""

    hidden_dims: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x is a batch of normalised coordinates with columns [xi, eta]."""
        act = _ACTIVATIONS[self.activation]
        for dim in self.hidden_dims:
            x = nn.Dense(dim, kernel_init=nn.initializers.glorot_normal())(x)
            x = act(x)
        return nn.Dense(2, kernel_init=nn.initializers.glorot_normal())(x)


class FourierMLP(nn.Module):
    """MLP with deterministic Fourier feature input embedding.
    Polar-enhanced Fourier MLP.

    Combines two complementary input embeddings:

    1. Global axis-aligned Fourier features (n_fourier bands in xi and eta)
       to capture smooth far-field variations.

    2. Hole-centred polar features that directly encode the Kirsch angular
       modes (sin/cos k*theta, k=1..n_polar) and radial Kirsch decay terms
       (1/r_hat, 1/r_hat²).  This is the critical addition: the stress
       concentration has a dominant cos(2*theta) signature that the global
       Fourier basis cannot represent efficiently when the hole radius is
       small relative to the domain.
    """

    hidden_dims: Sequence[int]
    n_fourier: int
    eta_max: float       # H / L; used to rescale eta into [0, 1]
    n_polar: int         # angular harmonics for hole-centred embedding
    hole_xi_c: float     # normalised hole centre x
    hole_eta_c: float    # normalised hole centre y
    hole_rc: float       # normalised hole radius
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x : (batch, 2)  —  columns are [xi, eta]."""
        xi_n  = x[:, 0:1]                    # (batch, 1)  in [0, 1]
        eta_n = x[:, 1:2] / self.eta_max     # (batch, 1)  in [0, 1]

        # Raw coordinates + sinusoidal bands
        parts = [xi_n, eta_n]
        for k in range(1, self.n_fourier + 1):
            parts.extend([
                jnp.sin(k * jnp.pi * xi_n),
                jnp.cos(k * jnp.pi * xi_n),
                jnp.sin(k * jnp.pi * eta_n),
                jnp.cos(k * jnp.pi * eta_n),
            ])

        # ------------------------------------------------------------------ #
        # Hole-centred polar embedding                                         #
        # Provides radial decay (1/r_hat, 1/r_hat²) and angular harmonics     #
        # (sin k*θ, cos k*θ) that are the building blocks of the Kirsch       #
        # stress-concentration field.  Without these, the global Fourier      #
        # basis cannot resolve the 2-θ pattern when hole_rc ≪ 1.             #
        # ------------------------------------------------------------------ #
        if self.n_polar > 0:
            dxi  = x[:, 0:1] - self.hole_xi_c
            deta = x[:, 1:2] - self.hole_eta_c
            # Radial distance; eps avoids zero when a point sits at the centre
            r = jnp.sqrt(dxi**2 + deta**2 + 1e-12)
            # Normalised to hole radius; clamped at 1 so we don't divide by
            # values inside the hole (those points are masked in eval anyway)
            r_hat = r / self.hole_rc
            r_hat_c = jnp.maximum(r_hat, 1.0)
            # Radial decay: 1/r_hat encodes the Kirsch O(R/r) perturbation;
            # 1/r_hat² encodes the O(R²/r²) stress terms
            parts.append(1.0 / r_hat_c)
            parts.append(1.0 / r_hat_c ** 2)
            # Angular harmonics: Kirsch primary mode is k=2 (cos 2θ / sin 2θ)
            theta = jnp.arctan2(deta, dxi)
            for k in range(1, self.n_polar + 1):
                parts.extend([
                    jnp.sin(k * theta),
                    jnp.cos(k * theta),
                ])

        inp = jnp.concatenate(parts, axis=-1)

        act = _ACTIVATIONS[self.activation]
        for dim in self.hidden_dims:
            inp = nn.Dense(dim, kernel_init=nn.initializers.glorot_normal())(inp)
            inp = act(inp)
        return nn.Dense(2, kernel_init=nn.initializers.glorot_normal())(inp)


def build_model(
    cfg: NetworkConfig,
    problem: ProblemConfig,
) -> nn.Module:
    """Construct the network from config.

    Parameters
    ----------
    cfg     : network hyper-parameters
    eta_max : normalised domain height H/L, passed to FourierMLP so that the
              y-direction Fourier frequencies are correctly scaled.
              Required even when n_fourier == 0 (ignored in that case).
    hole_xi_c, hole_eta_c, hole_rc : normalised hole geometry for the polar
              embedding.  Ignored when cfg.n_polar == 0.
    """
    if cfg.n_fourier > 0:
        return FourierMLP(
            hidden_dims=cfg.hidden_dims,
            activation=cfg.activation,
            n_fourier=cfg.n_fourier,
            eta_max=problem.eta_max,
            n_polar=cfg.n_polar,
            hole_xi_c=problem.hole_xi_c,
            hole_eta_c=problem.hole_eta_c,
            hole_rc=problem.hole_rc,
        )
    return MLP(hidden_dims=cfg.hidden_dims, activation=cfg.activation)


def init_params(model: nn.Module, key: jax.Array) -> dict:
    """Return initial parameter tree using a dummy (1, 2) input."""
    dummy = jnp.zeros((1, 2))
    return model.init(key, dummy)
