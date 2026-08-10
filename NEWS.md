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
