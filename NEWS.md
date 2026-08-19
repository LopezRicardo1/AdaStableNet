# AdaStableNet 0.4.2

* Moved the paper-scale Monte Carlo runner and journal-specific simulation
  protocol to the separate paper-analysis repository. Package installations now
  contain only the reusable estimator, diagnostics, tests, and compact package
  documentation.
* Added build and Git exclusions that prevent manuscript sources, paper results,
  figures, and analysis directories from being included accidentally in the R
  package repository or source tarball.

# AdaStableNet 0.4.1

* Standardized non-oracle prediction across all four branches: the two-stage
  branch now starts from the B-spline-reconstructed state at the training
  origin, while the modal branches retain their profiled fitted initial state.
* Added `"two_stage"` support to `coef()`, `fitted()`, `residuals()`,
  `predict()`, `plot()`, and `stability_diagnostics()` for a single, auditable
  prediction path.
* The simulation workflow now explicitly sets `fit_ode2fd = TRUE`, records the
  functional-data prediction contract, and invalidates older checkpoints that
  used a noisy first observation for the two-stage forecast.
* Trajectory plots now distinguish observed points, the B-spline
  reconstruction, and the fitted ODE curve.

# AdaStableNet 0.4.0

* Added `AdaStableNet_WaldNetwork()` for off-diagonal, BH-adjusted directed
  network recovery. It returns binary adjacency, signed selected weights, a
  dynamic matrix that retains self-dynamics, a complete edge table, and
  stability diagnostics before and after sparsification.
* Added `summarize_wald_networks()` for replicate or bootstrap selection
  frequency, directional selection frequency, and sign consistency in
  observed-data applications where support truth and ROC curves are absent.
* Extended coefficient-Wald simulation evaluation to all four estimators with
  off-diagonal ROC AUC, precision-recall AUC, power, false-positive and false-
  discovery rates, precision, F1, sign recovery, sparse-matrix error, and
  post-sparsification stability and forecasting metrics.
* Simulation checkpoints now retain every edge-level Wald result and
  sparsified matrix in `wald-networks.rds` for reproducible threshold curves.
* Stability classification now uses a documented square-root machine-precision
  tolerance by default, avoiding false instability labels from reconstruction
  roundoff while retaining the raw abscissa.

# AdaStableNet 0.3.0

* Added optional `backend = "torch"` optimization with float64 autograd,
  Adam, optional L-BFGS refinement, CPU/CUDA device selection, reproducible
  starts, and the same fitted-object contract as the base-R backend.
* Added `backend = "auto"` for Torch selection when both the R package and
  LibTorch runtime are available. Torch remains an optional suggested package.
* Extended the sourceable sparse simulation runner with backend, device,
  learning-rate, refinement, and patience controls. Results record the resolved
  backend, device, and Torch version, and checkpoints distinguish Torch tuning
  configurations.

# AdaStableNet 0.2.2

* Added an installed, sourceable simulation-study runner with the canonical
  sparse p = 15 and p = 16 designs, per-replication checkpoints, failure and
  warning capture, matrix and trajectory metrics, CSV summaries, and plots.

# AdaStableNet 0.2.1

* Changed `simulate_adastablenet()` to use the sparse block-embedding,
  permutation, and median-thresholding construction from the eigen-bound study
  by default. The prior dense conditioned generator remains available through
  `matrix_structure = "dense"`.

# AdaStableNet 0.2.0

* Reimplemented the profiled modal estimator with a deterministic base-R
  optimization backend and no external tensor runtime requirement.
* Added explicit unconstrained, Wald-screened, and stability-constrained model
  stages with multi-start optimization.
* Added input validation, tolerance-based eigenvalue classification, rank and
  conditioning diagnostics, convergence histories, and spectral checks.
* Added `coef()`, `fitted()`, `residuals()`, `predict()`, `plot()`, `print()`, and
  `summary()` methods.
* Added controlled simulation utilities and a reproducible simulation-study
  vignette.
* Corrected smoothing-parameter selection and removed ad hoc real-to-complex
  mode conversion and initializer clipping.
* Added testthat coverage, README documentation, and citation metadata.
