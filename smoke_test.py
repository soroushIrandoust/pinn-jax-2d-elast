"""Quick 100-epoch smoke test — validates JIT pipeline without GPU warmup wait."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import Config, ProblemConfig, NetworkConfig, TrainingConfig

cfg = Config(
    problem=ProblemConfig(L=10.0e3, H=1.0e3, E=20.0e3, nu=0.3, sigma0=2.0, hole_radius=100.0),
    network=NetworkConfig(hidden_dims=(32, 32, 32), activation="tanh", use_hard_bc=True, n_fourier=4),
    training=TrainingConfig(
        n_interior=256,
        n_boundary=64,
        n_hole=64,
        n_midline=64,
        epochs_adam=100,
        lr_init=1e-3,
        lr_final=1e-4,
        warmup_steps=10,
        seed=0,
        resample_every=1000,
        w_pde=1.0,
        w_bc_disp=100.0,
        w_bc_traction=20.0,
        w_bc_tb=20.0,
        w_bc_hole=10.0,
        w_bc_mid=100.0,
        log_every=25,
        save_dir="results_test",
    ),
)

from train import train
params, model, history = train(cfg)
print("Final total loss:", history["total"][-1])
print("SMOKE TEST PASSED")
