"""Training loop with Adam optimiser and optional L-BFGS refinement."""

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
    """Build the warmup-cosine learning-rate schedule.

    Linear warm-up from 0 to ``lr_init`` over ``warmup_steps`` steps, then
    cosine decay to ``lr_final`` by ``lr_decay_steps``.

    If ``lr_decay_steps`` is set below ``epochs_adam``, the schedule holds
    ``lr_final`` for the remaining steps.
    """
    warmup = optax.linear_schedule(
        init_value=0.0,
        end_value=cfg_t.lr_init,
        transition_steps=cfg_t.warmup_steps,
    )

    decay_end = min(cfg_t.lr_decay_steps, cfg_t.epochs_adam)
    decay_steps = max(decay_end - cfg_t.warmup_steps, 1)
    decay = optax.cosine_decay_schedule(
        init_value=cfg_t.lr_init,
        decay_steps=decay_steps,
        alpha=cfg_t.lr_final / cfg_t.lr_init,
    )

    if decay_end <= cfg_t.warmup_steps or decay_end >= cfg_t.epochs_adam:
        return optax.join_schedules(
            schedules=[warmup, decay],
            boundaries=[cfg_t.warmup_steps],
        )

    hold_final = optax.constant_schedule(cfg_t.lr_final)
    return optax.join_schedules(
        schedules=[warmup, decay, hold_final],
        boundaries=[cfg_t.warmup_steps, decay_end],
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
        cfg.problem,
    )
    params = init_params(model, key_init)

    use_hard_bc = cfg.network.use_hard_bc

    schedule  = _build_schedule(cfg.training)
    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(params)

    # -----------------------------------------------------------------------
    # Adaptive loss weighting: bounded, normalised inverse-EMA balancing
    # -----------------------------------------------------------------------
    loss_component_names = ["pde", "bc_left", "bc_right", "bc_tb", "bc_hole", "bc_mid"]
    moving_avg = {name: 1.0 for name in loss_component_names}
    base_weights = {
        "pde":      cfg.training.w_pde,
        "bc_left":  cfg.training.w_bc_disp,
        "bc_right": cfg.training.w_bc_traction,
        "bc_tb":    cfg.training.w_bc_tb,
        "bc_hole":  cfg.training.w_bc_hole,
        "bc_mid":   cfg.training.w_bc_mid,
    }
    use_adaptive = cfg.training.use_adaptive_weights
    ema_decay = cfg.training.adaptive_ema_decay
    alpha = cfg.training.adaptive_alpha
    min_mult = cfg.training.adaptive_min_mult
    max_mult = cfg.training.adaptive_max_mult
    eps = cfg.training.adaptive_eps

    def _base_weighted_total(info_dict):
        return sum(base_weights[k] * float(info_dict[k]) for k in loss_component_names)

    def _raw_unweighted_total(info_dict):
        return sum(float(info_dict[k]) for k in loss_component_names)

    def compute_adaptive_weights(losses_dict):
        """Compute bounded, normalised adaptive weights."""
        if (not use_adaptive) or (losses_dict is None):
            return base_weights

        # Update exponential moving average
        for key in loss_component_names:
            val = abs(float(losses_dict.get(key, 1.0)))
            moving_avg[key] = ema_decay * moving_avg[key] + (1.0 - ema_decay) * val

        mean_ema = float(np.mean([moving_avg[k] for k in loss_component_names]))

        # Relative inverse scaling around mean EMA with tempered exponent.
        raw_mult = {
            k: ((mean_ema + eps) / (moving_avg[k] + eps)) ** alpha
            for k in loss_component_names
        }

        # Normalise multipliers to keep average around 1 and avoid objective drift.
        mean_mult = float(np.mean(list(raw_mult.values())))
        raw_mult = {k: (raw_mult[k] / (mean_mult + eps)) for k in loss_component_names}

        # Bound multipliers to prevent weight explosion/collapse.
        clipped_mult = {
            k: float(np.clip(raw_mult[k], min_mult, max_mult))
            for k in loss_component_names
        }

        adaptive_w = {
            k: base_weights[k] * clipped_mult[k]
            for k in loss_component_names
        }
        return adaptive_w

    def loss_and_info(p, batch, adaptive_weights=None):
        return total_loss(
            p, model, batch,
            cfg.problem, cfg.training, use_hard_bc,
            adaptive_weights=adaptive_weights,
        )

    # -----------------------------------------------------------------------
    # JIT-compiled single training step
    # -----------------------------------------------------------------------
    @jax.jit
    def step_jit(params, opt_state, batch, adaptive_weights):
        (loss_val, info), grads = jax.value_and_grad(
            lambda p: loss_and_info(p, batch, adaptive_weights),
            has_aux=True,
        )(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val, info

    # Wrapper to handle adaptive weight computation in Python
    def step(params, opt_state, batch, losses_dict_prev=None):
        """Execute one training step with adaptive/base weighting."""
        adaptive_w = compute_adaptive_weights(losses_dict_prev)
        return step_jit(params, opt_state, batch, adaptive_w)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    os.makedirs(cfg.training.save_dir, exist_ok=True)

    batch = get_batch(cfg.problem, cfg.training, key_batch)

    keys = [
        "total", "total_adaptive", "total_unweighted", "pde", "bc_left", "bc_right", "bc_tb", "bc_hole", "bc_mid",
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
    best_epoch = 0
    last_improve_epoch = 0
    best_params = params
    final_base_loss = float("nan")

    print(f"\nBackend : {jax.default_backend().upper()}")
    print(f"Devices : {jax.devices()}")
    print(f"Params  : {sum(x.size for x in jax.tree_util.tree_leaves(params)):,}")
    for ln in _format_gpu_specs():
        print(ln)
    print(f"\nCompiling JIT step (first call may take ~30 s)…")

    t0 = time.time()
    losses_dict_prev = None  # Initialize for first step
    for epoch in tqdm(range(1, cfg.training.epochs_adam + 1), ncols=90):

        if epoch % cfg.training.resample_every == 0:
            rng, key_batch = jax.random.split(rng)
            batch = get_batch(cfg.problem, cfg.training, key_batch)

        params, opt_state, loss_val, info = step(params, opt_state, batch, losses_dict_prev)
        losses_dict_prev = info  # Save for next iteration's adaptive weighting
        sigma_xx_top = sigma_xx_at_hole_top(params) * cfg.problem.sigma0
        sigma_xx_side = sigma_xx_at_hole_side(params) * cfg.problem.sigma0

        base_total = _base_weighted_total(info)
        unweighted_total = _raw_unweighted_total(info)

        # Keep "total" as base-weighted objective for interpretable plotting.
        history["total"].append(base_total)
        history["total_adaptive"].append(float(loss_val))
        history["total_unweighted"].append(unweighted_total)
        for k, v in info.items():
            history[k].append(float(v))
        history["sigma_xx_hole_top"].append(float(sigma_xx_top))
        history["sigma_xx_hole_side"].append(float(sigma_xx_side))

        # Track best model using base-weighted loss, not adaptive objective scale.
        loss_scalar = float(base_total)
        final_base_loss = loss_scalar
        if not np.isfinite(best_loss):
            improved = True
        else:
            rel_improve = (best_loss - loss_scalar) / max(abs(best_loss), 1e-12)
            improved = (loss_scalar < best_loss) and (rel_improve > cfg.training.early_stop_rel_tol)
        if improved:
            best_loss = loss_scalar
            best_epoch = epoch
            last_improve_epoch = epoch
            best_params = params

        if epoch % cfg.training.log_every == 0:
            elapsed = time.time() - t0
            tqdm.write(
                f"  epoch {epoch:6d}  total_base={base_total:.3e}  total_adapt={float(loss_val):.3e}  "
                f"pde={info['pde']:.3e}  right={info['bc_right']:.3e}  "
                f"tb={info['bc_tb']:.3e}  hole={info['bc_hole']:.3e}  "
                f"mid={info['bc_mid']:.3e}  sxx@hole_top={sigma_xx_top:.3f} MPa  "
                f"sxx@hole_side={sigma_xx_side:.3f} MPa  ({elapsed:.0f}s)"
            )

        # Early-stop on prolonged base-loss plateau.
        if cfg.training.early_stop_enable:
            if epoch >= cfg.training.early_stop_min_epochs and (epoch - last_improve_epoch) >= cfg.training.early_stop_patience:
                tqdm.write(
                    f"Early stopping at epoch {epoch}: no base-loss improvement for "
                    f"{cfg.training.early_stop_patience} epochs "
                    f"(best epoch={best_epoch}, best base loss={best_loss:.3e})."
                )
                break

    with open(os.path.join(cfg.training.save_dir, "best_params.pkl"), "wb") as fh:
        pickle.dump(best_params, fh)
    with open(os.path.join(cfg.training.save_dir, "final_params.pkl"), "wb") as fh:
        pickle.dump(params, fh)

    # Persist loss history
    np.savez(
        os.path.join(cfg.training.save_dir, "loss_history.npz"),
        **{k: np.array(v) for k, v in history.items()},
    )
    print(
        f"Best checkpoint: epoch={best_epoch}, base_loss={best_loss:.3e}; "
        f"final base_loss={final_base_loss:.3e}"
    )
    print(f"\nTraining complete in {time.time() - t0:.1f} s")
    # Return best parameters for postprocessing/evaluation.
    return best_params, model, history


def load_params(path: str):
    """Load a saved Flax parameter tree from pickle."""
    with open(path, "rb") as fh:
        return pickle.load(fh)
