#' Fit an AdaStableNet Model from Modal Initial Values
#'
#' Fits a profiled nonlinear least-squares model for a linear ODE from supplied
#' modal initial values. For each candidate spectrum, the modal loading matrix is
#' estimated analytically; only the eigenvalue parameters are optimized.
#'
#' @param Y Numeric matrix with states in rows and time points in columns.
#' @param tt Strictly increasing observation times.
#' @param initial_a Initial real parts for complex-conjugate modes.
#' @param initial_b Initial positive imaginary parts for complex-conjugate modes.
#' @param initial_cc Optional initial real eigenvalues.
#' @param eigen_real_wald Apply an approximate Wald screen to modal real parts.
#' @param wald_critical Absolute normal critical value used by the screen.
#' @param eigen_bound Fit the stability-constrained branch.
#' @param stability_margin Upper bound for active modal real parts. The default
#'   zero enforces nonpositive real parts.
#' @param backend Optimization backend. `"base"` uses [stats::optim()],
#'   `"torch"` uses float64 automatic differentiation, and `"auto"` selects
#'   Torch when its runtime is available and otherwise uses base R.
#' @param optimizer Optimization method passed to [stats::optim()].
#' @param num_iter Maximum iterations per optimization start.
#' @param tol Relative optimization tolerance.
#' @param ridge.pen Nonnegative scale-adjusted ridge multiplier.
#' @param n_starts Number of optimization starts.
#' @param start_jitter Standard deviation of parameter jitter after the first
#'   start.
#' @param seed Optional seed for optimization starts.
#' @param wald_nsteps Integration intervals for approximate Wald variances.
#' @param variance_ridge Ridge multiplier for Fisher-information inversion.
#' @param lr Learning rate for the Torch Adam optimizer; ignored by base R.
#'   `NULL` uses `0.01`.
#' @param torch_device Torch device: automatic CUDA selection, CPU, or CUDA.
#' @param torch_refine Refine the Adam solution with Torch L-BFGS.
#' @param torch_refine_iter Maximum L-BFGS refinement iterations.
#' @param torch_patience Consecutive small relative-loss changes required for
#'   Adam convergence.
#' @param verbose Print concise progress messages.
#'
#' @return A list with `Unbounded`, `Wald_Real`, and `Eigen_Bound` model stages.
#' @export
AdaStableNet <- function(Y, tt, initial_a, initial_b, initial_cc = NULL,
                         eigen_real_wald = TRUE, wald_critical = 2,
                         eigen_bound = TRUE, stability_margin = 0,
                         backend = c("base", "torch", "auto"),
                         optimizer = "BFGS", num_iter = 1000L, tol = 1e-8,
                         ridge.pen = 1e-3, n_starts = 1L,
                         start_jitter = 0.05, seed = NULL,
                         wald_nsteps = 20L, variance_ridge = 1e-6,
                         lr = NULL, torch_device = c("auto", "cpu", "cuda"),
                         torch_refine = TRUE, torch_refine_iter = 20L,
                         torch_patience = 5L, verbose = TRUE) {
  tt <- .validate_time(tt)
  Y <- .validate_data(Y, tt, "state_by_time")
  initial_a <- as.numeric(initial_a)
  initial_b <- as.numeric(initial_b)
  initial_cc_numeric <- if (is.null(initial_cc)) numeric() else as.numeric(initial_cc)
  if (length(initial_a) != length(initial_b)) {
    stop("`initial_a` and `initial_b` must have equal lengths.", call. = FALSE)
  }
  if (any(!is.finite(c(initial_a, initial_b, initial_cc_numeric))) ||
      any(initial_b <= 0)) {
    stop("Initial modal values must be finite and `initial_b` must be positive.",
         call. = FALSE)
  }
  if (2L * length(initial_a) + length(initial_cc_numeric) != nrow(Y)) {
    stop(
      "Initial modal values must describe exactly one mode per observed state.",
      call. = FALSE
    )
  }
  if (!is.logical(eigen_real_wald) || length(eigen_real_wald) != 1L ||
      !is.logical(eigen_bound) || length(eigen_bound) != 1L) {
    stop("`eigen_real_wald` and `eigen_bound` must be logical scalars.",
         call. = FALSE)
  }
  .validate_scalar(wald_critical, "wald_critical", lower = 0, lower_open = TRUE)
  .validate_scalar(stability_margin, "stability_margin")
  .validate_scalar(num_iter, "num_iter", lower = 1, integer = TRUE)
  .validate_scalar(tol, "tol", lower = 0, lower_open = TRUE)
  .validate_scalar(ridge.pen, "ridge.pen", lower = 0)
  .validate_scalar(n_starts, "n_starts", lower = 1, integer = TRUE)
  .validate_scalar(start_jitter, "start_jitter", lower = 0)
  .validate_scalar(wald_nsteps, "wald_nsteps", lower = 2, integer = TRUE)
  .validate_scalar(variance_ridge, "variance_ridge", lower = 0)
  backend_requested <- match.arg(backend)
  backend <- .resolve_optimizer_backend(backend_requested)
  torch_device <- match.arg(torch_device)
  learning_rate <- lr %||% 0.01
  .validate_scalar(learning_rate, "lr", lower = 0, lower_open = TRUE)
  if (!is.logical(torch_refine) || length(torch_refine) != 1L ||
      is.na(torch_refine)) {
    stop("`torch_refine` must be TRUE or FALSE.", call. = FALSE)
  }
  .validate_scalar(torch_refine_iter, "torch_refine_iter", lower = 0,
                   integer = TRUE)
  .validate_scalar(torch_patience, "torch_patience", lower = 1,
                   integer = TRUE)
  if (!optimizer %in% c("BFGS", "Nelder-Mead", "CG")) {
    stop("`optimizer` must be one of 'BFGS', 'Nelder-Mead', or 'CG'.",
         call. = FALSE)
  }

  time_origin <- min(tt)
  relative_time <- tt - time_origin
  common <- list(
    Y = Y, tt = relative_time, ridge.pen = ridge.pen,
    optimizer = optimizer, num_iter = num_iter, tol = tol,
    n_starts = n_starts, start_jitter = start_jitter,
    seed = seed, verbose = verbose,
    backend = backend, lr = learning_rate,
    torch_device = torch_device, torch_refine = torch_refine,
    torch_refine_iter = torch_refine_iter,
    torch_patience = torch_patience
  )

  if (verbose) message("Fitting the unconstrained modal model.")
  unbounded_fit <- do.call(.fit_modal_stage, c(list(
    a = initial_a,
    b = initial_b,
    cc = initial_cc_numeric,
    stable = FALSE
  ), common))
  Unbounded <- ada_output(
    a = unbounded_fit$a, b = unbounded_fit$b, cc = unbounded_fit$cc,
    tt = relative_time, Y = Y, ridge.pen = ridge.pen,
    diagnostics = unbounded_fit$diagnostics, time_origin = time_origin
  )

  Wald_Real <- NULL
  wald <- NULL
  if (eigen_real_wald) {
    if (verbose) message("Applying the approximate modal Wald screen.")
    wald <- .eigen_wald_masks(
      Unbounded, Y, critical = wald_critical,
      nsteps = wald_nsteps, variance_ridge = variance_ridge
    )
    wald_fit <- do.call(.fit_modal_stage, c(list(
      a = unbounded_fit$a,
      b = unbounded_fit$b,
      cc = unbounded_fit$cc,
      a_mask = wald$a_mask,
      cc_mask = wald$cc_mask,
      stable = FALSE
    ), common))
    Wald_Real <- ada_output(
      a = wald_fit$a, b = wald_fit$b, cc = wald_fit$cc,
      tt = relative_time, Y = Y, ridge.pen = ridge.pen,
      a_wald = wald$a_mask, cc_wald = wald$cc_mask,
      diagnostics = wald_fit$diagnostics, time_origin = time_origin
    )
    Wald_Real$wald <- wald[c("table", "sigma")]
  }

  Eigen_Bound <- NULL
  if (eigen_bound) {
    if (verbose) message("Fitting the stability-constrained modal model.")
    start_stage <- Wald_Real %||% Unbounded
    start_pars <- start_stage$modal_parameters
    stable_fit <- do.call(.fit_modal_stage, c(list(
      a = start_pars$a,
      b = start_pars$b,
      cc = start_pars$cc,
      a_mask = start_pars$a_mask,
      cc_mask = start_pars$cc_mask,
      stable = TRUE,
      stability_margin = stability_margin
    ), common))
    Eigen_Bound <- ada_output(
      a = stable_fit$a, b = stable_fit$b, cc = stable_fit$cc,
      tt = relative_time, Y = Y, ridge.pen = ridge.pen,
      a_wald = stable_fit$a_mask, cc_wald = stable_fit$cc_mask,
      diagnostics = stable_fit$diagnostics, time_origin = time_origin
    )
    if (!is.null(wald)) Eigen_Bound$wald <- wald[c("table", "sigma")]
    if (!Eigen_Bound$diagnostics$full_modal_rank) {
      warning(
        "The stability-constrained loading matrix is rank deficient; the ",
        "reconstructed system matrix need not preserve every target mode.",
        call. = FALSE
      )
    }
    stability_tolerance <- 1e-7 * max(1, max(Mod(Eigen_Bound$eigenvalues)))
    allowed_abscissa <- max(0, stability_margin)
    if (Eigen_Bound$diagnostics$spectral_abscissa >
        allowed_abscissa + stability_tolerance) {
      warning(
        "The reconstructed system matrix exceeds the requested stability ",
        "margin; inspect modal rank and conditioning diagnostics.",
        call. = FALSE
      )
    }
  }

  structure(list(
    Unbounded = Unbounded,
    Wald_Real = Wald_Real,
    Eigen_Bound = Eigen_Bound,
    time = tt,
    relative_time = relative_time,
    time_origin = time_origin,
    control = list(
      eigen_real_wald = eigen_real_wald,
      wald_critical = wald_critical,
      eigen_bound = eigen_bound,
      stability_margin = stability_margin,
      backend_requested = backend_requested,
      backend = backend,
      optimizer = optimizer,
      num_iter = num_iter,
      tol = tol,
      ridge.pen = ridge.pen,
      n_starts = n_starts,
      start_jitter = start_jitter,
      seed = seed,
      lr = learning_rate,
      torch_device = torch_device,
      torch_refine = torch_refine,
      torch_refine_iter = torch_refine_iter,
      torch_patience = torch_patience
    )
  ), class = "adastablenet_stages")
}
