# 2-D Elasticity PINN — Full Codebase Walkthrough

This document describes the current active codebase file by file, using line ranges from the source as it exists now.

The scope covered here is the full non-empty codebase:

- `main.py`
- `postprocess.py`
- `smoke_test.py`
- `requirements.txt`
- `src/config.py`
- `src/evaluate.py`
- `src/network.py`
- `src/physics.py`
- `src/sampler.py`
- `src/train.py`
- `src/visualize.py`
- `generate_report.py`

Two files are currently empty and are therefore documented only as empty stubs:

- `_patch_docs.py`
- `src/check_imports.py`

## 1. main.py

Purpose: top-level training entry point. It builds the default configuration, trains the model, evaluates the trained network on full and zoomed grids, and saves plots.

- Lines 1-18: Module docstring. It defines the physical problem at a glance: plate dimensions, centered hole, material, boundary conditions, symmetry penalty, and the standard `python main.py` entry command.
- Lines 20-21: Import `os` and `sys` for path handling and module path injection.
- Line 23: Inserts the local `src/` directory at the front of `sys.path` so this script can import sibling modules without packaging the project.
- Lines 25-28: Import configuration dataclasses, the training routine, the evaluation helpers, and the plotting functions.
- Lines 31-32: Start `main()` and compute the output directory as `<project>/results`.
- Lines 34-70: Construct the default `Config` object.
  - Lines 35-43: Build `ProblemConfig` with `L=10000 mm`, `H=1000 mm`, `E=20000 MPa`, `nu=0.3`, `sigma0=2 MPa`, `hole_radius=100 mm`, and `plane_stress`.
  - Lines 44-49: Build `NetworkConfig` with four 128-unit hidden layers, `tanh`, hard BCs enabled, and `n_fourier=8`.
  - Lines 50-69: Build `TrainingConfig` with the recovered stable regime: 4096 interior points, 512 boundary/hole/midline points where applicable, 150000 Adam epochs, warmup-plus-cosine schedule, effectively fixed sampling via `resample_every=200000`, and the current loss weights.
- Line 72: Print the startup banner via `_print_banner(cfg)`.
- Lines 74-77: Training block header comment and actual call to `train(cfg)` returning trained parameters, the model object, and the saved history arrays.
- Lines 79-85: Evaluation block. It evaluates the trained PINN on a full grid, a near-hole zoom grid, and computes a compact metric summary.
- Lines 87-92: Print scalar summary metrics for displacement and stress magnitudes.
- Lines 94-110: Visualization block. It writes the loss history plot, the full-domain field plots, and the near-hole zoom plots.
- Line 112: Final success message with the output directory.
- Lines 115-134: `_print_banner(cfg)`.
  - Lines 116-118: Alias the problem, network, and training configs for shorter access.
  - Line 119: Create a separator line.
  - Lines 120-134: Print problem size, hole geometry, material properties, characteristic displacement, network description, and training length.
- Lines 137-138: Standard script guard so `main()` only runs when the file is executed directly.

## 2. postprocess.py

Purpose: reload saved model checkpoints and regenerate plots without retraining.

- Lines 1-17: Module docstring describing the script, its assumptions, and CLI usage.
- Lines 19-21: Import `argparse`, `os`, and `sys`.
- Line 23: Suppress TensorFlow C++ log noise if present in the environment.
- Line 24: Add `src/` to `sys.path` so this script can import project modules.
- Lines 26-30: Import config, model builder, checkpoint loader, evaluation helpers, and plot helpers.
- Lines 33-44: Define `postprocess(results_dir, checkpoint='best')` and document its arguments.
- Lines 45-46: Choose `best_params.pkl` or `final_params.pkl` based on the requested checkpoint.
- Lines 48-59: Fallback logic. If the requested checkpoint is missing, try the other checkpoint; if neither exists, raise `FileNotFoundError`.
- Lines 61-62: Print the checkpoint path and load parameters from disk.
- Lines 64-67: Build a default config object, redirect `save_dir` to the requested results directory, and instantiate the model architecture.
- Lines 69-77: Evaluate the loaded checkpoint on both the full domain and the near-hole zoom grid, then print two summary metrics: maximum `sigma_xx` and maximum `|tau_xy|`.
- Lines 79-91: Load `loss_history.npz` if available and regenerate the loss plot. If it is missing, print a warning and continue.
- Lines 93-98: Regenerate the full-domain field plots and the near-hole zoom plots.
- Line 100: Final success message.
- Lines 103-119: `main()` CLI entry point.
  - Lines 104-115: Define two optional command-line arguments: `--checkpoint` and `--results-dir`.
  - Lines 116-119: Resolve the effective results directory and call `postprocess()`.
- Lines 122-123: Standard script guard.

## 3. smoke_test.py

Purpose: short end-to-end sanity check that runs a tiny training job.

- Line 1: Module docstring describing the smoke test as a quick pipeline validator.
- Lines 2-3: Import `sys` and `os`, then prepend `src/` to `sys.path`.
- Line 5: Import the four configuration dataclasses.
- Lines 7-30: Build a much smaller `Config` object for a fast debug run.
  - Line 8: Same physical problem as the main run, including the 100 mm hole.
  - Line 9: Smaller network with three 32-unit layers and only four Fourier bands.
  - Lines 10-29: Smaller sample counts, only 100 epochs, lighter LR schedule, and a reduced save directory `results_test`.
- Line 32: Import `train` only after `sys.path` is configured.
- Line 33: Run training with the reduced configuration.
- Lines 34-35: Print the final total loss and a success message.

## 4. requirements.txt

Purpose: list Python dependencies and installation guidance.

- Lines 1-7: Human-readable installation notes. They explain standard install, CUDA-enabled install for Linux/WSL2, and CPU-only install on Windows.
- Lines 9-15: Actual dependency pins and minimum versions for `jax`, `jaxlib`, `flax`, `optax`, `numpy`, `matplotlib`, and `tqdm`.

## 5. src/config.py

Purpose: define all physical, network, and training hyperparameters as dataclasses.

- Line 1: Module docstring.
- Lines 3-4: Import `dataclass`, `field`, and `Tuple`.
- Lines 7-44: `ProblemConfig` dataclass.
  - Lines 8-9: Declare the physical problem configuration class.
  - Lines 11-19: Store the default physical parameters and unit labels.
  - Lines 21-24: `u_ref` property computes the characteristic displacement `sigma0 * L / E`.
  - Lines 26-29: `eta_max` property computes normalized domain height `H / L`.
  - Lines 31-34: `hole_xi_c` property fixes the normalized hole center in `x` at `0.5`.
  - Lines 36-39: `hole_eta_c` property fixes the normalized hole center in `y` at `0.5 * H / L`.
  - Lines 41-44: `hole_rc` property computes the normalized hole radius.
- Lines 47-54: `NetworkConfig` dataclass.
  - Lines 51-54: Default architecture: four 128-unit layers, `tanh`, hard BCs on, eight Fourier bands.
- Lines 57-85: `TrainingConfig` dataclass.
  - Lines 61-65: Default sample counts.
  - Lines 67-73: Optimization schedule parameters and seed.
  - Lines 75-81: Loss weights for PDE, left BC, right traction, top/bottom traction-free, hole traction-free, and midline symmetry.
  - Lines 83-85: Logging cadence and save directory.
- Lines 88-92: `Config` wrapper dataclass that nests `problem`, `network`, and `training` using default factories.

## 6. src/evaluate.py

Purpose: evaluate a trained network on regular grids in physical units and compute summary metrics.

- Lines 1-7: Module docstring. It explicitly says this evaluator is for the plate-with-hole case and does not use a closed-form analytical reference.
- Lines 9-11: Import `jax`, `jax.numpy`, and `numpy`.
- Lines 13-14: Import the problem/network configs plus low-level field/stress helpers from `physics.py`.
- Lines 17-21: `_mask_inside_hole(cfg_p, x, y)` returns a boolean mask selecting points inside the physical circular hole.
- Lines 24-63: `_evaluate_grid(...)` is the core evaluation routine.
  - Lines 27-30: Pull flags and normalization constants from the configs.
  - Lines 32-33: Build the meshgrid and flatten it into point coordinates.
  - Lines 35-38: Evaluate displacements and stresses at every point using `jax.vmap`.
  - Lines 40-53: Reshape the outputs and convert them from normalized units to physical units.
  - Lines 55-58: Compute and store the hole mask, hole center, and hole radius.
  - Lines 60-61: Replace values inside the hole with `NaN` so downstream plotting ignores them.
  - Line 63: Return the assembled results dictionary.
- Lines 66-72: `evaluate_on_grid(...)` creates a standard full-domain grid of `201 × 81` points and forwards to `_evaluate_grid`.
- Lines 75-87: `evaluate_near_hole(...)` builds a clipped zoom window around the hole, defaulting to `2.5` radii in each direction, then forwards to `_evaluate_grid`.
- Lines 90-98: `compute_summary_metrics(res)` returns max/min summary values used in `main.py` and `postprocess.py`.

## 7. src/network.py

Purpose: define the neural architectures used by the PINN.

- Lines 1-29: Module docstring describing the plain MLP, the FourierMLP, the reason Fourier features are needed, and the resulting feature dimension.
- Line 31: Import `Sequence` for typed layer-size declarations.
- Lines 33-35: Import `jax`, `jax.numpy`, and `flax.linen`.
- Line 37: Import `NetworkConfig` for `build_model`.
- Lines 39-43: `_ACTIVATIONS` lookup table mapping configuration strings to Flax activation functions.
- Lines 46-59: `MLP` class.
  - Lines 49-50: Store hidden dimensions and activation name.
  - Lines 52-59: `__call__` builds a dense stack with Glorot-normal initialization and returns a 2-component output.
- Lines 62-100: `FourierMLP` class.
  - Lines 74-77: Store layer sizes, activation, Fourier band count, and `eta_max`.
  - Lines 79-83: Normalize input columns into `xi_n` and `eta_n`.
  - Lines 85-94: Build the deterministic sinusoidal feature embedding `[raw coords + sin/cos bands]`.
  - Lines 96-100: Run the embedded features through the dense stack and output two raw displacement channels.
- Lines 103-120: `build_model(cfg, eta_max)`.
  - Lines 113-119: Return `FourierMLP` if `n_fourier > 0`.
  - Line 120: Otherwise fall back to the plain `MLP`.
- Lines 123-126: `init_params(model, key)` builds an initial Flax parameter tree from a dummy `(1, 2)` input.

## 8. src/sampler.py

Purpose: generate interior, boundary, hole, and symmetry-line collocation points.

- Lines 1-6: Module docstring describing the normalized coordinate system and the hole location.
- Lines 8-10: Import `numpy`, `jax`, and `jax.numpy`.
- Line 12: Import the problem and training configs.
- Lines 15-40: `sample_interior(...)`.
  - Lines 21-23: Read the hole center and radius in normalized coordinates.
  - Lines 25-28: Draw twice as many raw points as needed.
  - Lines 30-32: Reject points that fall inside the hole.
  - Lines 34-38: Fail loudly if oversampling was still insufficient.
  - Line 40: Return the accepted points as a JAX array.
- Lines 43-59: `sample_hole_boundary(...)`.
  - Lines 49-51: Read hole geometry.
  - Lines 53-55: Draw random angles and compute unit-circle trigonometric values.
  - Lines 57-59: Convert the angles into point coordinates and outward normals `[xi, eta, nx, ny]`.
- Lines 62-66: `sample_midline(...)` samples random `xi` and fixes `eta` to the hole center height, which is the normalized midline.
- Lines 69-94: `sample_boundaries(...)` samples each of the four outer rectangular boundaries independently.
- Lines 97-106: `get_batch(...)` orchestrates the full batch: interior, four outer edges, hole boundary, and midline samples.

## 9. src/physics.py

Purpose: define the constitutive model, stress reconstruction, PDE residual, all boundary-condition losses, and the final weighted loss.

- Lines 1-30: Module docstring. It explains the nondimensionalization, normalized equilibrium equations, normalized constitutive constants, and the hard-BC ansatz for both displacement components.
- Lines 32-34: Import `jax`, `jax.numpy`, and `flax.linen`.
- Line 36: Import problem and training configs for typing.
- Lines 43-55: `_stiffness(nu, mode)` returns normalized constitutive constants.
  - Lines 45-48: Plane-stress branch.
  - Lines 49-54: Plane-strain branch.
- Lines 58-67: `_stresses_from_jacobian(...)` maps the displacement Jacobian to `[s_xx, s_yy, t_xy]`.
- Lines 74-79: `_net_uv(...)` evaluates the network at a single point.
  - Line 77: Get the raw network outputs.
  - Line 78: Apply the hard-BC scaling `xi * raw` to both components.
  - Line 79: Return scaled or raw output depending on `use_hard_bc`.
- Lines 86-95: `_stress_at(...)` differentiates `_net_uv` with respect to spatial coordinates and converts the Jacobian into normalized stresses.
- Lines 102-122: `pde_residuals(...)` computes the equilibrium residuals at interior points.
  - Lines 112-120: For each point, differentiate the stress field a second time and assemble the two equilibrium equations.
  - Line 122: Vectorize over all interior points.
- Lines 129-134: `bc_left_displacement(...)` penalizes left-edge `u_hat` only when the hard-BC ansatz is disabled.
- Lines 137-149: `bc_hole_traction(...)` enforces `sigma · n = 0` on the hole by projecting the stress tensor onto the outward normal at each sampled point.
- Lines 152-155: `bc_midline_v(...)` penalizes nonzero vertical displacement on the symmetry line.
- Lines 158-165: `bc_right_traction(...)` enforces normalized `s_xx = 1` and `t_xy = 0` on the right edge.
- Lines 168-173: `bc_traction_free(...)` enforces `s_yy = 0` and `t_xy = 0` on the top and bottom edges.
- Lines 180-230: `total_loss(...)`.
  - Lines 186-187: Unpack the full seven-part training batch and compute normalized stiffness constants.
  - Lines 189-191: PDE loss.
  - Lines 193-194: Left-edge loss.
  - Lines 196-199: Right-edge traction loss.
  - Lines 201-203: Top/bottom traction-free loss.
  - Lines 205-208: Hole traction-free loss.
  - Lines 210-211: Midline symmetry loss.
  - Lines 213-220: Weighted sum using the configuration weights.
  - Lines 222-229: Return the scalar total plus a dictionary of individual loss components.

## 10. src/train.py

Purpose: run optimization, save checkpoints, and record training history.

- Line 1: Module docstring.
- Lines 3-5: Import filesystem, timing, and pickle helpers.
- Lines 7-11: Import JAX, Optax, NumPy, and `tqdm`.
- Lines 13-16: Import config, model helpers, sampling, and low-level physics functions.
- Lines 19-34: `_build_schedule(cfg_t)` creates the warmup-plus-cosine learning-rate schedule.
  - Lines 21-25: Linear warmup from `0` to `lr_init`.
  - Lines 26-30: Cosine decay from `lr_init` to `lr_final`.
  - Lines 31-34: Join both schedules at `warmup_steps`.
- Lines 37-139: `train(cfg)`.
  - Lines 46-47: Seed and split the PRNG state.
  - Lines 49-50: Build the model and initialize parameters.
  - Lines 52-56: Build the optimizer and its state.
  - Lines 61-72: Define the JIT-compiled `step(...)` function that computes loss, gradients, optimizer updates, and the next parameter tree.
  - Lines 77-79: Ensure the results directory exists and sample the initial batch.
  - Lines 81-82: Initialize the history dictionary, including `sigma_xx_hole_top`.
  - Lines 84-89: Define the top-of-hole stress probe and a JIT helper to evaluate it.
  - Lines 91-92: Initialize best-loss tracking.
  - Lines 94-97: Print backend, devices, parameter count, and compile notice.
  - Lines 99-126: Main epoch loop.
    - Lines 102-104: Optional resampling hook.
    - Lines 106-107: Run one optimization step and evaluate the stress probe.
    - Lines 109-112: Append losses and probe values to history.
    - Lines 114-117: Update the best checkpoint when total loss improves.
    - Lines 119-126: Periodic console logging of key losses plus `sxx@hole_top`.
  - Lines 128-131: Save best and final parameter checkpoints as pickle files.
  - Lines 133-137: Save the full history dictionary into `loss_history.npz`.
  - Lines 138-139: Print total runtime and return the final artifacts.
- Lines 142-145: `load_params(path)` reads a pickled Flax parameter tree back from disk.

## 11. src/visualize.py

Purpose: save full-field plots, near-hole zoom plots, and loss-history plots.

- Line 1: Module docstring.
- Lines 3-8: Import filesystem, NumPy, Matplotlib, configure the headless backend, and import `Circle` for drawing the hole boundary.
- Lines 15-28: `_field_panel(...)` draws one contour plot with automatic finite-value level selection and a colorbar.
- Lines 31-32: `_draw_hole(ax, center, radius)` overlays a white circle with a black edge so the hole is visually clear.
- Lines 39-78: `plot_fields(...)` saves the full-domain field images.
  - Lines 50-53: Pull the domain coordinates and hole geometry from the evaluated results.
  - Lines 58-64: Define which fields to plot and which colormaps to use.
  - Lines 69-76: Loop through fields, render each panel, draw the hole, and save as `u.png`, `v.png`, `sxx.png`, `syy.png`, and `txy.png`.
  - Line 78: Print the destination directory.
- Lines 81-109: `plot_hole_zoom(...)` saves the near-hole stress zoom plots `zoom_sxx.png`, `zoom_syy.png`, and `zoom_txy.png`.
- Lines 112-146: `plot_loss_history(...)` saves a two-panel semilog plot of total loss and individual loss components.

## 12. generate_report.py

Purpose: build a PDF report from the result images and the saved training history.

- Lines 1-18: Module docstring explaining usage, inputs, output file, and the fact that no LaTeX installation is needed.
- Lines 20-31: Import CLI helpers, filesystem utilities, text wrapping, Matplotlib PDF tools, and NumPy.
- Lines 37-57: `_text_page(...)` creates a PDF page made of text blocks at different font sizes.
- Lines 60-71: `_image_page(...)` renders one image plus caption on a PDF page.
- Lines 74-93: `_two_image_page(...)` renders one or two vertically stacked images with captions.
- Lines 96-134: `_loss_page(...)` reconstructs the loss-history figure directly from `loss_history.npz`.
  - Lines 104-111: Define colors and labels for each tracked loss component.
  - Lines 113-133: Draw total loss and the component curves, then write the page to the PDF.
- Lines 137-164: `_grid_page(...)` assembles several saved PNGs into a grid and writes them as a report page.
- Lines 171-409: `build_report(results_dir)` builds the whole PDF.
  - Lines 172-174: Define the output file and open a `PdfPages` context.
  - Lines 176-205: Build the title page with problem, material, and architecture metadata.
  - Lines 207-248: Build the “Problem Formulation” text page.
  - Lines 250-301: Build the “PINN Methodology” text page.
  - Lines 303-341: Build the “Network Architecture” text page.
  - Line 346: Insert the loss-history page.
  - Lines 348-354: Insert the displacement-field page.
  - Lines 356-363: Insert the full-domain stress-component page.
  - Lines 365-372: Insert the near-hole stress zoom page.
  - Lines 374-407: Build the conclusions page.
  - Line 409: Print the final report path.
- Lines 412-434: CLI entry point.
  - Lines 416-422: Build the `argparse` interface.
  - Lines 424-428: Resolve the results directory.
  - Lines 430-432: Check that the directory exists.
  - Line 434: Call `build_report(results_dir)`.

## 13. Empty Stubs

Two files exist but currently contain no code:

- `_patch_docs.py`: empty placeholder.
- `src/check_imports.py`: empty placeholder.

## 14. End-to-End Flow

The active execution path is:

1. `main.py` builds the default config.
2. `src/train.py` builds the model from `src/network.py`, samples batches from `src/sampler.py`, and computes losses from `src/physics.py`.
3. Checkpoints and history are written into `results/`.
4. `src/evaluate.py` converts the trained model into dense field arrays.
5. `src/visualize.py` turns those arrays into PNG plots.
6. `postprocess.py` can rerun steps 4-5 from saved checkpoints.
7. `generate_report.py` assembles the stored PNGs and `loss_history.npz` into `technical_report.pdf`.

## 15. Notes On Current Version

This document describes the current hole-aware version of the codebase, not the older no-hole analytical comparison variant.

The most important behavior choices in the current implementation are:

- hard enforcement of both `u = 0` and `v = 0` on the left edge
- explicit hole traction-free loss
- explicit midline symmetry loss `v(x, H/2) = 0`
- fixed default hole radius of `100 mm`
- right-edge traction of `2 MPa`
- `E = 20_000 MPa`
- stable long-run training regime with `epochs_adam = 150000` and `resample_every = 200000`
