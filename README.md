# 2-D Elasticity PINN

Physics-informed neural network for 2-D linear elasticity in a rectangular plate with a circular hole, implemented with JAX, Flax, and Optax.

The current problem setup is a plate of size `10000 mm x 1000 mm` with a centered hole of radius `200 mm`, loaded in uniaxial tension on the right edge. The model uses a Fourier-feature MLP with additional hole-centered polar features to better capture the Kirsch-type stress concentration near the hole boundary.

## Features

- JAX-based PINN training with JIT compilation
- Fourier and polar feature embeddings for improved stress-field resolution
- Hard enforcement of the left-edge displacement boundary condition
- Random collocation resampling to avoid fixed-grid aliasing
- Full-domain and near-hole postprocessing plots
- PDF report generation from saved results

## Repository Layout

- `main.py`: main training entry point
- `postprocess.py`: regenerate plots from saved checkpoints
- `generate_report.py`: build a PDF report from results
- `requirements.txt`: Python dependencies
- `src/config.py`: problem, network, and training dataclasses
- `src/train.py`: training loop
- `src/evaluate.py`: grid evaluation and summary metrics
- `src/network.py`: neural network definitions
- `src/physics.py`: elasticity equations and residual/stress helpers
- `src/sampler.py`: collocation-point sampling
- `src/visualize.py`: plotting utilities

## Installation

Create and activate your Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

For GPU support on Linux/WSL with CUDA 12:

```bash
pip install "jax[cuda12]" -r requirements.txt
```

For CPU-only Windows usage:

```bash
pip install -r requirements.txt
```

## Usage

Train the default model:

```bash
python main.py
```

Regenerate plots from saved checkpoints:

```bash
python postprocess.py
python postprocess.py --checkpoint final
```

Generate the PDF report from the default results directory:

```bash
python generate_report.py
```

Or point the report script to another results directory:

```bash
python generate_report.py --results_dir path/to/results
```

## Outputs

The default training run writes outputs to `results/`, including:

- model checkpoints
- loss history arrays
- full-domain field plots
- near-hole zoom plots
- generated PDF report

## Notes

- The current training configuration is defined in `src/config.py`.
- The report text is generated dynamically from configuration defaults.