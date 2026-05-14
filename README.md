# 2-D Elasticity PINN

Physics-informed neural network (PINN) for 2-D linear elasticity in a rectangular plate with a circular hole, implemented with JAX, Flax, and Optax.

## Current Default Setup

<!-- AUTO-CONFIG-START -->
_This section is auto-generated from `src/config.py` by `python sync_readme_config.py`._

### ProblemConfig

| Key | Value |
|---|---|
| `L` | `10000.0` |
| `H` | `1000.0` |
| `E` | `1000.0` |
| `nu` | `0.3` |
| `sigma0` | `1.0` |
| `hole_radius` | `200.0` |
| `mode` | `plane_stress` |
| `length_unit` | `mm` |
| `stress_unit` | `MPa` |

### ProblemConfig Derived Values

| Key | Value |
|---|---|
| `u_ref` | `10.0` |
| `eta_max` | `0.1` |
| `hole_xi_c` | `0.5` |
| `hole_eta_c` | `0.05` |
| `hole_rc` | `0.02` |

### NetworkConfig

| Key | Value |
|---|---|
| `hidden_dims` | `(128, 128, 128, 128)` |
| `activation` | `tanh` |
| `use_hard_bc` | `True` |
| `n_fourier` | `8` |
| `n_polar` | `6` |

### TrainingConfig

| Key | Value |
|---|---|
| `n_interior` | `6144` |
| `n_boundary` | `512` |
| `n_hole` | `1536` |
| `n_midline` | `512` |
| `near_hole_fraction` | `0.5` |
| `near_hole_outer_mult` | `3.0` |
| `epochs_adam` | `700000` |
| `lr_init` | `0.001` |
| `lr_final` | `5e-06` |
| `warmup_steps` | `1000` |
| `lr_decay_steps` | `700000` |
| `seed` | `42` |
| `resample_every` | `2000` |
| `use_adaptive_weights` | `True` |
| `adaptive_ema_decay` | `0.99` |
| `adaptive_alpha` | `0.5` |
| `adaptive_min_mult` | `0.25` |
| `adaptive_max_mult` | `4.0` |
| `adaptive_eps` | `1e-08` |
| `early_stop_enable` | `True` |
| `early_stop_min_epochs` | `120000` |
| `early_stop_patience` | `80000` |
| `early_stop_rel_tol` | `0.001` |
| `w_pde` | `10.0` |
| `w_bc_disp` | `100.0` |
| `w_bc_traction` | `60.0` |
| `w_bc_tb` | `50.0` |
| `w_bc_hole` | `12.0` |
| `w_bc_mid` | `30.0` |
| `log_every` | `500` |
| `save_dir` | `results` |

### PlotConfig

| Key | Value |
|---|---|
| `deformation_scale` | `250.0` |
| `interactive_width` | `1800` |
| `interactive_field_height` | `900` |
| `interactive_vector_height` | `900` |
| `interactive_misc_height` | `900` |
| `interactive_responsive` | `True` |
| `interactive_lock_aspect` | `True` |
| `interactive_colorbar_len_fraction` | `0.995` |
| `hole_zoom_radius_factor` | `3.0` |
| `png_contour_levels` | `32` |
| `annotate_field_minmax` | `True` |
| `field_stats_digits` | `4` |
| `show_deformed_reference_bc` | `False` |
| `auto_levels` | `False` |
| `field_level_mode` | `{'u': 'nonnegative_auto', 'v': 'symmetric_auto', 'umag': 'auto', 'sxx': 'fixed', 'syy': 'fixed', 'sxy': 'fixed', 'exx': 'symmetric_auto', 'eyy': 'symmetric_auto', 'exy': 'symmetric_auto', 's1': 'fixed', 's2': 'fixed', 'e1': 'auto', 'e2': 'auto'}` |
| `cmap_stress` | `RdYlGn_r` |
| `cmap_strain` | `parula` |
| `cmap_displacement` | `RdBu` |
| `field_level_limits` | `{'u': (0.0, None), 'v': (None, None), 'umag': (None, None), 'sxx': (0.0, 3.5), 'syy': (-1.0, 1.0), 'sxy': (-1.0, 1.0), 'exx': (None, None), 'eyy': (None, None), 'exy': (None, None), 's1': (0.0, 3.5), 's2': (-1.0, 1.0), 'e1': (None, None), 'e2': (None, None)}` |

<!-- AUTO-CONFIG-END -->

To refresh this section after changing `src/config.py`, run:

```bash
python sync_readme_config.py
```

## Features

- JAX JIT-based PINN training
- Fourier + polar embeddings for near-hole stress concentration resolution
- Random resampling of collocation points every fixed interval
- Full-domain, deformed-domain, near-hole zoom, and principal-field plotting
- Interactive HTML plots with hover values (`results_interactive/`)
- Principal stress/strain direction vector plots on undeformed geometry (PNG + interactive HTML)
- PDF report generation from saved outputs

## Plot Configuration Highlights

Key plotting options are in `src/config.py` under `PlotConfig`:

- `deformation_scale`
- `interactive_width`, `interactive_field_height`, `interactive_vector_height`, `interactive_misc_height`
- `show_deformed_reference_bc`
- `auto_levels`, `field_level_mode`, `field_level_limits`
- `cmap_stress`, `cmap_strain`, `cmap_displacement`

All contour/vector limits and colormaps are applied consistently to both PNG and Plotly outputs.

## Repository Layout

- `main.py`: training + evaluation + plotting pipeline
- `postprocess.py`: regenerate evaluation/plots from saved checkpoint
- `generate_report.py`: build PDF report from saved results
- `requirements.txt`: dependencies
- `src/config.py`: dataclass configuration
- `src/train.py`: training loop and checkpoint/history saving
- `src/evaluate.py`: grid evaluation and derived fields
- `src/network.py`: model architectures
	- `build_model(cfg.network, cfg.problem)` derives the network's geometric inputs from the problem config
- `src/physics.py`: PDE and BC residual/loss terms
- `src/sampler.py`: collocation/boundary sampling
- `src/visualize.py`: static and interactive plotting

## Installation

```bash
pip install -r requirements.txt
```

For GPU support on Linux/WSL with CUDA 12:

```bash
pip install "jax[cuda12]" -r requirements.txt
```

## Usage

Train from scratch:

```bash
python main.py
```

Re-run postprocessing from saved checkpoints (no retraining):

```bash
python postprocess.py
python postprocess.py --checkpoint final
```

Generate a PDF report:

```bash
python generate_report.py
python generate_report.py --results_dir path/to/results
```

## Output Folders

`results/` contains static artifacts:

- `best_params.pkl`, `final_params.pkl`
- `loss_history.npz`
- static PNG plots (undeformed fields, undeformed principal fields, undeformed principal vectors, deformed fields, undeformed zoom fields, sampling)
- `run.log`

Current naming convention:

- Undeformed full/principal fields: `undeformed_<field>.png`
- Undeformed principal vectors: `undeformed_vector_<field>.png`
- Undeformed near-hole zoom fields: `undeformed_zoom_<field>.png`
- Deformed fields: `deformed_<field>.png`

`results_interactive/` contains interactive HTML plots:

- field contours with hover values (`x`, `y`, `value`)
- deformed and near-hole interactive contours
- interactive loss history
- interactive sampling map
- interactive undeformed principal-direction vector plots

Current naming convention mirrors PNG stems, e.g.:

- `undeformed_<field>.html`
- `undeformed_vector_<field>.html`
- `undeformed_zoom_<field>.html`
- `deformed_<field>.html`

## Notes

- Results can be regenerated post training if the solution parameters are saved in the results folder (.pkl) by running `postprocess.py`.
- `main.py` and `postprocess.py` both call the same evaluation-and-plotting function.
- Interactive outputs require `plotly` (included in `requirements.txt`).
