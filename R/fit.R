#' Fit AdaStableNet to a Multivariate Trajectory
#'
#' This is the main user interface. It smooths the observed states, constructs a
#' two-stage gradient-matching initializer, and fits profiled unconstrained,
#' Wald-screened, and stability-constrained modal models.
#'
#' @param Y Numeric matrix with time points in rows and states in columns.
#' @param tt Strictly increasing observation times.
#' @param nbasis Number of B-spline basis functions.
#' @param lambda_range Range for the natural logarithm of the smoothing penalty.
#' @param method Gradient-matching initializer.
#' @param twoSE Use the two-standard-error smoothing rule.
#' @param fit_ode2fd Fit modal models to smoothed data when `TRUE`, otherwise raw
#'   data.
#' @param complex_pairs Deprecated compatibility argument. Modal structure is
#'   inferred from the real initial system matrix.
#' @param eigen_real_wald Apply an approximate Wald screen to modal real parts.
#' @param wald_critical Absolute normal critical value.
#' @param eigen_bound Fit a stability-constrained branch.
#' @param stability_margin Upper bound for active modal real parts.
#' @param optimizer Optimization method passed to [stats::optim()].
#' @param num_iter Maximum optimizer iterations per start.
#' @param tol Relative optimization tolerance.
#' @param ridge.pen Nonnegative scale-adjusted ridge multiplier.
#' @param n_starts Number of optimization starts.
#' @param start_jitter Standard deviation of parameter jitter after the first
#'   start.
#' @param seed Optional seed for optimization starts.
#' @param wald_nsteps Integration intervals for approximate Wald variances.
#' @param variance_ridge Ridge multiplier for Fisher-information inversion.
#' @param lr Deprecated compatibility argument; ignored by the base-R backend.
#' @param verbose Print concise progress messages.
#'
#' @return An object of class `adastablenet_fit`.
#' @examples
#' sim <- simulate_adastablenet(p = 3, n_time = 31, sigma = 0.03, seed = 1)
#' fit <- FitAdaStableNet(
#'   sim$Y, sim$time, nbasis = 12, twoSE = FALSE,
#'   num_iter = 100, eigen_real_wald = FALSE, verbose = FALSE
#' )
#' fit
#' coef(fit)
#' @export
FitAdaStableNet <- function(Y, tt, nbasis = 25,
                            lambda_range = c(-16, 4),
                            method = "two.stage", twoSE = TRUE,
                            fit_ode2fd = TRUE, complex_pairs = NULL,
                            eigen_real_wald = TRUE, wald_critical = 2,
                            eigen_bound = TRUE, stability_margin = 0,
                            optimizer = "BFGS", num_iter = 1000L,
                            tol = 1e-8, ridge.pen = 1e-3,
                            n_starts = 1L, start_jitter = 0.05,
                            seed = NULL, wald_nsteps = 20L,
                            variance_ridge = 1e-6, lr = NULL,
                            verbose = TRUE) {
  tt <- .validate_time(tt)
  Y <- .validate_data(Y, tt, "time_by_state")
  if (!is.null(complex_pairs)) {
    warning(
      "`complex_pairs` is deprecated and ignored; modal structure is inferred ",
      "from the real initial system matrix.", call. = FALSE
    )
  }
  if (verbose) message("Computing the functional-data initializer.")
  initial <- ode2stage(
    Y = Y, tt = tt, nbasis = nbasis, lambda_range = lambda_range,
    est.pen = ridge.pen, method = method, twoSE = twoSE
  )
  modal_dimension <- 2L * length(initial$re.hat) + length(initial$real.hat)
  if (modal_dimension != ncol(Y)) {
    stop(
      "The initialized eigenvalues do not form a complete real modal basis. ",
      "Try a larger ridge penalty or a different smoothing range.", call. = FALSE
    )
  }
  modal_data <- if (fit_ode2fd) initial$Yhat_fd else Y
  stages <- AdaStableNet(
    Y = t(modal_data), tt = tt,
    initial_a = initial$re.hat,
    initial_b = initial$im.hat,
    initial_cc = initial$real.hat,
    eigen_real_wald = eigen_real_wald,
    wald_critical = wald_critical,
    eigen_bound = eigen_bound,
    stability_margin = stability_margin,
    optimizer = optimizer,
    num_iter = num_iter,
    tol = tol,
    ridge.pen = ridge.pen,
    n_starts = n_starts,
    start_jitter = start_jitter,
    seed = seed,
    wald_nsteps = wald_nsteps,
    variance_ridge = variance_ridge,
    lr = lr,
    verbose = verbose
  )
  selected <- if (!is.null(stages$Eigen_Bound)) {
    "stable"
  } else if (!is.null(stages$Wald_Real)) {
    "wald"
  } else {
    "unbounded"
  }
  structure(list(
    Ode2Stage = initial,
    AdaEigenStableNet = stages,
    data = Y,
    fitted_data = modal_data,
    time = tt,
    time_origin = min(tt),
    selected_branch = selected,
    call = match.call(),
    control = list(
      nbasis = nbasis,
      lambda_range = lambda_range,
      method = method,
      twoSE = twoSE,
      fit_ode2fd = fit_ode2fd,
      eigen_real_wald = eigen_real_wald,
      eigen_bound = eigen_bound,
      stability_margin = stability_margin,
      optimizer = optimizer,
      num_iter = num_iter,
      tol = tol,
      ridge.pen = ridge.pen,
      n_starts = n_starts,
      seed = seed
    )
  ), class = "adastablenet_fit")
}
