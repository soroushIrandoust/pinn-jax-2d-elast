# 2-D Elasticity PINN

Physics-informed neural network (PINN) for 2-D linear elasticity in a rectangular plate with a circular hole, implemented with JAX, Flax, and Optax.

## Current Default Setup

- Domain: `L=10000 mm`, `H=1000 mm`
- Hole: centered at `(L/2, H/2)` with radius `200 mm`
- Material: plane stress, `E=1000 MPa`, `nu=0.3`
- Load: right-edge traction `sigma0=1 MPa`
- Network: Fourier + hole-centered polar feature MLP (`[128,128,128,128]`, `tanh`, `n_fourier=8`, `n_polar=6`)
- Hard BC ansatz: enforces `u=v=0` on the left edge when enabled
- Training: Adam-only, `460000` epochs, warmup-cosine schedule

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

- If a training run is already in progress, you can apply code changes and then regenerate updated figures by running `postprocess.py` after training finishes.
- Interactive outputs require `plotly` (included in `requirements.txt`).