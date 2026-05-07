"""Configuration dataclasses for the 2D elasticity PINN."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ProblemConfig:
    """Physical problem parameters — uniaxial stretch of a 2-D rectangle."""

    L: float = 10.0e3          # Domain length in x (mm)
    H: float = 1.0e3           # Domain height in y (mm)
    E: float = 2.0e3           # Young's modulus (MPa) — 2 GPa = 2,000 MPa
    nu: float = 0.3            # Poisson's ratio (dimensionless)
    sigma0: float = 2.0        # Applied normal stress at right boundary (MPa)
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
    n_hole: int = 1024              # Points on hole circumference
    n_midline: int = 512            # Points on y = H/2 symmetry line
    near_hole_fraction: float = 0.30  # Fraction of interior points in near-hole annulus
    near_hole_outer_mult: float = 3.0   # Annulus outer radius in units of hole radius

    # Optimisation
    epochs_adam: int = 300000
    lr_init: float = 1e-3           # Peak learning rate
    lr_final: float = 1e-5          # Final learning rate
    warmup_steps: int = 1000        # Linear warm-up steps
    seed: int = 42
    resample_every: int = 2000      # Periodic refresh to avoid fixed-batch collapse

    # Loss weights (balanced for hole-stress concentration stability)
    w_pde: float = 10.0
    w_bc_disp: float = 100.0        # Left-boundary u = 0
    w_bc_traction: float = 60.0     # Right-boundary traction BC
    w_bc_tb: float = 50.0           # Top / bottom traction-free BCs
    w_bc_hole: float = 3.0          # Hole traction-free BC
    w_bc_mid: float = 30.0          # Midline symmetry v(x, H/2) = 0

    # Logging
    log_every: int = 500
    save_dir: str = "results"


@dataclass
class PlotConfig:
    """Post-processing and visualisation parameters."""

    deformation_scale: float = 100.0   # Deformed-configuration magnification


@dataclass
class Config:
    problem: ProblemConfig = field(default_factory=ProblemConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    plotting: PlotConfig = field(default_factory=PlotConfig)
