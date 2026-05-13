"""Configuration dataclasses for the 2D elasticity PINN."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class ProblemConfig:
    """Physical problem parameters — uniaxial stretch of a 2-D rectangle."""

    L: float = 10.0e3          # Domain length in x (mm)
    H: float = 1.0e3           # Domain height in y (mm)
    E: float = 1.0e3           # Young's modulus (MPa)
    nu: float = 0.3            # Poisson's ratio (dimensionless)
    sigma0: float = 1.0        # Applied normal stress at right boundary (MPa)
    hole_radius: float = 200.0 # Circular hole radius (mm)
    mode: str = "plane_stress" # "plane_stress" | "plane_strain"
    length_unit: str = "mm"    # Label used in plots for spatial axes and displacements
    stress_unit: str = "MPa"   # Label used in plots for stress components

    @property
    def u_ref(self) -> float:
        """Characteristic displacement: u_ref = sigma0 * L / E."""
        return self.sigma0 * self.L / self.E

    @property
    def eta_max(self) -> float:
        """Maximum normalised y-coordinate: H / L."""
        return self.H / self.L

    @property
    def hole_xi_c(self) -> float:
        """Normalised x-coordinate of the hole centre."""
        return 0.5

    @property
    def hole_eta_c(self) -> float:
        """Normalised y-coordinate of the hole centre."""
        return 0.5 * self.H / self.L

    @property
    def hole_rc(self) -> float:
        """Normalised hole radius."""
        return self.hole_radius / self.L


@dataclass
class NetworkConfig:
    """Neural-network architecture parameters."""

    hidden_dims: Tuple[int, ...] = (128, 128, 128, 128)
    activation: str = "tanh"        # "tanh" | "swish" | "gelu"
    use_hard_bc: bool = True        # Hard enforcement of u(0, y) = 0
    n_fourier: int = 8              # Fourier feature bands (0 = plain MLP)
    n_polar: int = 6                # Angular harmonics for hole-centred polar embedding (0 = disabled)


@dataclass
class TrainingConfig:
    """Optimiser and sampling hyper-parameters."""

    # Collocation sampling
    n_interior: int = 6144          # Interior points
    n_boundary: int = 512           # Points per boundary side
    n_hole: int = 1536              # Points on hole circumference
    n_midline: int = 512            # Points on y = H/2 symmetry line
    near_hole_fraction: float = 0.50    # Fraction of interior points in near-hole annulus
    near_hole_outer_mult: float = 3.0   # Annulus outer radius in units of hole radius

    # Optimisation
    epochs_adam: int = 700000
    lr_init: float = 1e-3           # Peak learning rate
    lr_final: float = 5e-6          # Final learning rate
    warmup_steps: int = 1000        # Linear warm-up steps
    lr_decay_steps: int = 700000    # Reach lr_final by this step
    seed: int = 42
    resample_every: int = 2000      # Periodic refresh to avoid fixed-batch collapse

    # Adaptive loss weighting (stabilised inverse-EMA balancing)
    use_adaptive_weights: bool = True
    adaptive_ema_decay: float = 0.99
    adaptive_alpha: float = 0.5      # 0=no adapt, 1=full inverse-EMA adapt
    adaptive_min_mult: float = 0.25  # Lower bound for per-term multiplier
    adaptive_max_mult: float = 4.0   # Upper bound for per-term multiplier
    adaptive_eps: float = 1e-8

    # Early stopping (based on base-weighted loss, not adaptive objective)
    early_stop_enable: bool = True
    early_stop_min_epochs: int = 120000
    early_stop_patience: int = 80000
    early_stop_rel_tol: float = 1e-3

    # Loss weights (balanced for hole-stress concentration stability)
    w_pde: float = 10.0
    w_bc_disp: float = 100.0        # Left-boundary u = 0
    w_bc_traction: float = 60.0     # Right-boundary traction BC
    w_bc_tb: float = 50.0           # Top / bottom traction-free BCs
    w_bc_hole: float = 12.0         # Hole traction-free BC
    w_bc_mid: float = 30.0          # Midline symmetry v(x, H/2) = 0

    # Logging
    log_every: int = 500
    save_dir: str = "results"


@dataclass
class PlotConfig:
    """Post-processing and visualisation parameters."""

    deformation_scale: float = 250.0   # Deformed-configuration magnification
    interactive_width: int = 1800      # HTML figure width in px (initial)
    interactive_field_height: int = 900
    interactive_vector_height: int = 900
    interactive_misc_height: int = 900
    interactive_responsive: bool = True
    interactive_lock_aspect: bool = True  # True: preserve x/y physical scale in HTML
    # Colorbar length in interactive plots. Set to None to auto-fit by aspect.
    interactive_colorbar_len_fraction: float | None = 0.995
    hole_zoom_radius_factor: float = 3.0  # Zoom window half-size = factor * hole_radius
    png_contour_levels: int = 32      # More levels -> smoother PNG gradients
    annotate_field_minmax: bool = True # Add min/max text to field figures
    field_stats_digits: int = 4
    show_deformed_reference_bc: bool = False
    auto_levels: bool = False          # If True, all fields use local min/max

    # Per-field level modes:
    #   fixed            -> use field_level_limits values
    #   auto             -> [min(data), max(data)]
    #   nonnegative_auto -> [0, max(data)]
    #   symmetric_auto   -> [-max(abs(data)), +max(abs(data))]
    field_level_mode: Dict[str, str] = field(default_factory=lambda: {
        "u": "nonnegative_auto",
        "v": "symmetric_auto",
        "umag": "auto",
        "sxx": "fixed",
        "syy": "fixed",
        "sxy": "fixed",
        "exx": "symmetric_auto",
        "eyy": "symmetric_auto",
        "exy": "symmetric_auto",
        "s1": "fixed",
        "s2": "fixed",
        "e1": "auto",
        "e2": "auto",
    })

    # Colormap selections — applied to all stress, strain, and displacement plots.
    # Stress options  : "RdYlGn_r", "parula", "hot_r", "plasma", "inferno", "RdBu_r", "coolwarm", "seismic", "jet"
    # Strain options  : "parula", "viridis", "plasma", "magma", "YlOrRd", "coolwarm", "RdBu_r"
    # Displacement options: "RdBu", "parula", "coolwarm", "seismic", "bwr", "PiYG", "PRGn"
    cmap_stress: str = "RdYlGn_r"
    cmap_strain: str = "parula"
    cmap_displacement: str = "RdBu"

    # Limits used when mode is "fixed".
    field_level_limits: Dict[str, Tuple[Optional[float], Optional[float]]] = field(
        default_factory=lambda: {
            "u": (0.0, None),
            "v": (None, None),
            "umag": (None, None),
            "sxx": (0.0, 3.5),
            "syy": (-1.0, 1.0),
            "sxy": (-1.0, 1.0),
            "exx": (None, None),
            "eyy": (None, None),
            "exy": (None, None),
            "s1": (0.0, 3.5),
            "s2": (-1.0, 1.0),
            "e1": (None, None),
            "e2": (None, None),
        }
    )


@dataclass
class Config:
    problem: ProblemConfig = field(default_factory=ProblemConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    plotting: PlotConfig = field(default_factory=PlotConfig)
