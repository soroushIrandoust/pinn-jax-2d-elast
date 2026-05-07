"""Training loop with Adam optimiser and cosine LR schedule."""

import os
import time
import pickle
import subprocess

import jax
import jax.numpy as jnp
import optax
import numpy as np
from tqdm import tqdm

from config import Config
from network import build_model, init_params
from sampler import get_batch
from physics import total_loss, _stiffness, _stress_at


def _format_gpu_specs() -> list[str]:
    """Collect GPU model/spec lines for the startup report.

    Uses JAX device metadata first, then enriches with nvidia-smi output when
    available.
    """
    lines = []

    # JAX-visible devices
    jax_devs = [d for d in jax.devices() if getattr(d, "platform", "") == "gpu"]
    if jax_devs:
        lines.append("GPU(s) from JAX:")
        for idx, dev in enumerate(jax_devs):
            kind = getattr(dev, "device_kind", str(dev))
            proc = getattr(dev, "process_index", "?")
            did = getattr(dev, "id", "?")
            lines.append(f"  [{idx}] {kind}  (id={did}, process={proc})")

    # NVIDIA runtime details (best-effort)
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,cuda_version",
            "--format=csv,noheader,nounits",
        ]
        raw = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=5)
        rows = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if rows:
            lines.append("GPU specs from nvidia-smi:")
            for idx, row in enumerate(rows):
                parts = [p.strip() for p in row.split(",")]
                if len(parts) >= 4:
                    name, mem_mb, drv, cuda = parts[:4]
                    lines.append(
                        f"  [{idx}] {name} | VRAM={mem_mb} MiB | driver={drv} | cuda={cuda}"
                    )
                else:
                    lines.append(f"  [{idx}] {row}")
    except Exception:
        # Keep startup robust even when nvidia-smi is absent in environment.
        lines.append("GPU specs from nvidia-smi: unavailable")

    if not lines:
        lines.append("GPU specs: no GPU detected by JAX")
    return lines


def _build_schedule(cfg_t):
    """Warm-up then cosine decay."""
    warmup = optax.linear_schedule(
        init_value=0.0,
        end_value=cfg_t.lr_init,
        transition_steps=cfg_t.warmup_steps,
    )
    decay = optax.cosine_decay_schedule(
        init_value=cfg_t.lr_init,
        decay_steps=max(cfg_t.epochs_adam - cfg_t.warmup_steps, 1),
        alpha=cfg_t.lr_final / cfg_t.lr_init,
    )
    return optax.join_schedules(
        schedules=[warmup, decay],
        boundaries=[cfg_t.warmup_steps],
    )


def train(cfg: Config):
    """Run the full Adam training loop.

    Returns
    -------
    params  : trained parameter tree
    model   : Flax module
    history : dict of loss arrays
    """
    rng = jax.random.PRNGKey(cfg.training.seed)
    rng, key_init, key_batch = jax.random.split(rng, 3)

    model = build_model(
        cfg.network,
        eta_max=cfg.problem.eta_max,
        hole_xi_c=cfg.problem.hole_xi_c,
        hole_eta_c=cfg.problem.hole_eta_c,
        hole_rc=cfg.problem.hole_rc,
    )
    params = init_params(model, key_init)

    use_hard_bc = cfg.network.use_hard_bc

    schedule  = _build_schedule(cfg.training)
    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(params)

    # -----------------------------------------------------------------------
    # JIT-compiled single training step
    # -----------------------------------------------------------------------
    @jax.jit
    def step(params, opt_state, batch):
        def loss_fn(p):
            return total_loss(
                p, model, batch,
                cfg.problem, cfg.training, use_hard_bc,
            )

        (loss_val, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val, info

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    os.makedirs(cfg.training.save_dir, exist_ok=True)

    batch = get_batch(cfg.problem, cfg.training, key_batch)

    keys = [
        "total", "pde", "bc_left", "bc_right", "bc_tb", "bc_hole", "bc_mid",
        "sigma_xx_hole_top", "sigma_xx_hole_side"
    ]
    history = {k: [] for k in keys}

    hole_top = jnp.array([cfg.problem.hole_xi_c, cfg.problem.hole_eta_c + cfg.problem.hole_rc])
    hole_side = jnp.array([cfg.problem.hole_xi_c + cfg.problem.hole_rc, cfg.problem.hole_eta_c])
    C11, C12, C33 = _stiffness(cfg.problem.nu, cfg.problem.mode)

    @jax.jit
    def sigma_xx_at_hole_top(p):
        return _stress_at(p, model, hole_top, C11, C12, C33, use_hard_bc)[0]

    @jax.jit
    def sigma_xx_at_hole_side(p):
        return _stress_at(p, model, hole_side, C11, C12, C33, use_hard_bc)[0]

    best_loss = float("inf")
    best_params = params

    print(f"\nBackend : {jax.default_backend().upper()}")
    print(f"Devices : {jax.devices()}")
    print(f"Params  : {sum(x.size for x in jax.tree_util.tree_leaves(params)):,}")
    for ln in _format_gpu_specs():
        print(ln)
    print(f"\nCompiling JIT step (first call may take ~30 s)…")

    t0 = time.time()
    for epoch in tqdm(range(1, cfg.training.epochs_adam + 1), ncols=90):

        if epoch % cfg.training.resample_every == 0:
            rng, key_batch = jax.random.split(rng)
            batch = get_batch(cfg.problem, cfg.training, key_batch)

        params, opt_state, loss_val, info = step(params, opt_state, batch)
        sigma_xx_top = sigma_xx_at_hole_top(params) * cfg.problem.sigma0
        sigma_xx_side = sigma_xx_at_hole_side(params) * cfg.problem.sigma0

        history["total"].append(float(loss_val))
        for k, v in info.items():
            history[k].append(float(v))
        history["sigma_xx_hole_top"].append(float(sigma_xx_top))
        history["sigma_xx_hole_side"].append(float(sigma_xx_side))

        loss_scalar = float(loss_val)
        if loss_scalar < best_loss:
            best_loss = loss_scalar
            best_params = params

        if epoch % cfg.training.log_every == 0:
            elapsed = time.time() - t0
            tqdm.write(
                f"  epoch {epoch:6d}  total={loss_val:.3e}  "
                f"pde={info['pde']:.3e}  right={info['bc_right']:.3e}  "
                f"tb={info['bc_tb']:.3e}  hole={info['bc_hole']:.3e}  "
                f"mid={info['bc_mid']:.3e}  sxx@hole_top={sigma_xx_top:.3f} MPa  "
                f"sxx@hole_side={sigma_xx_side:.3f} MPa  ({elapsed:.0f}s)"
            )

    with open(os.path.join(cfg.training.save_dir, "best_params.pkl"), "wb") as fh:
        pickle.dump(best_params, fh)
    with open(os.path.join(cfg.training.save_dir, "final_params.pkl"), "wb") as fh:
        pickle.dump(params, fh)

    # Persist loss history
    np.savez(
        os.path.join(cfg.training.save_dir, "loss_history.npz"),
        **{k: np.array(v) for k, v in history.items()},
    )
    print(f"\nTraining complete in {time.time() - t0:.1f} s")
    return params, model, history


def load_params(path: str):
    """Load a saved Flax parameter tree from pickle."""
    with open(path, "rb") as fh:
        return pickle.load(fh)
