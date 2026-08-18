#' @export
print.adastablenet_fit <- function(x, ...) {
  stage <- .resolve_branch(x, x$selected_branch)
  cat("AdaStableNet fit\n")
  cat("  States: ", ncol(x$data), "\n", sep = "")
  cat("  Time points: ", nrow(x$data), "\n", sep = "")
  cat("  Backend: ", stage$diagnostics$backend %||% "base", "\n", sep = "")
  cat("  Selected branch: ", x$selected_branch, "\n", sep = "")
  cat("  Training MSE: ", format(stage$diagnostics$loss, digits = 5),
      "\n", sep = "")
  cat("  Spectral abscissa: ",
      format(stage$diagnostics$spectral_abscissa, digits = 5), "\n", sep = "")
  cat("  Modal loading rank: ", stage$diagnostics$modal_rank, "/",
      nrow(stage$P), "\n", sep = "")
  invisible(x)
}

#' Summarize an AdaStableNet Fit
#'
#' @param object An `adastablenet_fit` object.
#' @param ... Unused.
#'
#' @return An object of class `summary.adastablenet_fit` with a branch table.
#' @export
summary.adastablenet_fit <- function(object, ...) {
  keys <- c(unbounded = "Unbounded", wald = "Wald_Real", stable = "Eigen_Bound")
  rows <- lapply(names(keys), function(name) {
    stage <- object$AdaEigenStableNet[[keys[[name]]]]
    if (is.null(stage)) return(NULL)
    data.frame(
      branch = name,
      backend = stage$diagnostics$backend %||% "base",
      mse = stage$diagnostics$loss,
      spectral_abscissa = stage$diagnostics$spectral_abscissa,
      numerical_abscissa = stage$diagnostics$numerical_abscissa,
      modal_rank = stage$diagnostics$modal_rank,
      loading_condition = stage$diagnostics$loading_condition,
      convergence = stage$diagnostics$convergence %||% NA_integer_,
      evaluations = unname(stage$diagnostics$counts[["function"]] %||% NA_integer_)
    )
  })
  structure(list(
    call = object$call,
    dimensions = c(time = nrow(object$data), states = ncol(object$data)),
    selected_branch = object$selected_branch,
    branches = do.call(rbind, rows),
    selected_lambda = object$Ode2Stage$selected_lambda,
    smoothing_method = object$Ode2Stage$selection_method
  ), class = "summary.adastablenet_fit")
}

#' Stability Diagnostics for an AdaStableNet Fit
#'
#' Summarizes asymptotic spectral stability and finite-horizon transient
#' amplification for one fitted branch. A nonpositive spectral abscissa rules
#' out exponentially growing eigenmodes. It does not imply monotone decay of
#' the Euclidean norm when the fitted matrix is nonnormal; the numerical
#' abscissa and maximum matrix-exponential norm diagnose that distinction.
#'
#' @param object An `adastablenet_fit` or `adastablenet_stage` object.
#' @param branch One of `"stable"`, `"wald"`, or `"unbounded"` when `object`
#'   is a complete fit.
#' @param horizon Nonnegative horizon over which transient amplification is
#'   evaluated.
#' @param n_grid Number of equally spaced evaluation times.
#'
#' @return A list containing the spectral abscissa, numerical abscissa,
#'   stability indicators, and finite-horizon amplification curve.
#' @export
stability_diagnostics <- function(
    object, branch = c("stable", "wald", "unbounded"),
    horizon = 1, n_grid = 101L) {
  stage <- if (inherits(object, "adastablenet_stage")) {
    object
  } else if (inherits(object, "adastablenet_fit")) {
    .resolve_branch(object, branch)
  } else {
    stop("`object` must be an AdaStableNet fit or fitted stage.",
         call. = FALSE)
  }
  transient <- .transient_growth(stage$A_hat, horizon, n_grid)
  spectral <- .spectral_abscissa(stage$A_hat)
  numerical <- .numerical_abscissa(stage$A_hat)
  structure(list(
    branch = if (inherits(object, "adastablenet_fit")) match.arg(branch) else NA_character_,
    spectral_abscissa = spectral,
    numerical_abscissa = numerical,
    spectrally_stable = spectral <= 0,
    euclidean_dissipative = numerical <= 0,
    transient = transient
  ), class = "adastablenet_stability_diagnostics")
}

#' @export
print.summary.adastablenet_fit <- function(x, ...) {
  cat("AdaStableNet summary\n")
  cat("  Dimensions: ", x$dimensions[["time"]], " time points x ",
      x$dimensions[["states"]], " states\n", sep = "")
  cat("  Smoothing: ", x$smoothing_method, " (lambda = ",
      format(x$selected_lambda, digits = 4), ")\n\n", sep = "")
  print(x$branches, row.names = FALSE, digits = 5)
  invisible(x)
}

#' Extract the Estimated Dynamic Matrix
#'
#' @param object An `adastablenet_fit` object.
#' @param branch One of `"stable"`, `"wald"`, or `"unbounded"`.
#' @param ... Unused.
#'
#' @return A numeric system matrix.
#' @export
coef.adastablenet_fit <- function(object,
                                  branch = c("stable", "wald", "unbounded"), ...) {
  .resolve_branch(object, branch)$A_hat
}

#' Extract Fitted Trajectories
#'
#' @param object An `adastablenet_fit` object.
#' @param branch One of `"stable"`, `"wald"`, or `"unbounded"`.
#' @param ... Unused.
#'
#' @return A time-by-state numeric matrix.
#' @export
fitted.adastablenet_fit <- function(object,
                                    branch = c("stable", "wald", "unbounded"), ...) {
  t(.resolve_branch(object, branch)$X_hat)
}

#' Extract AdaStableNet Residuals
#'
#' @param object An `adastablenet_fit` object.
#' @param branch One of `"stable"`, `"wald"`, or `"unbounded"`.
#' @param observed If `TRUE`, compute residuals against the original observations;
#'   otherwise use the data supplied to modal fitting.
#' @param ... Unused.
#'
#' @return A time-by-state numeric matrix.
#' @export
residuals.adastablenet_fit <- function(object,
                                       branch = c("stable", "wald", "unbounded"),
                                       observed = TRUE, ...) {
  reference <- if (observed) object$data else object$fitted_data
  reference - stats::fitted(object, branch = branch)
}

#' Predict from an AdaStableNet Fit
#'
#' @param object An `adastablenet_fit` object.
#' @param new_time Numeric prediction times. Defaults to observed times.
#' @param x0 Optional state at `time_origin`; defaults to the fitted state.
#' @param branch One of `"stable"`, `"wald"`, or `"unbounded"`.
#' @param time_origin Time represented by `x0`.
#' @param ... Unused.
#'
#' @return A time-by-state numeric matrix.
#' @export
predict.adastablenet_fit <- function(object, new_time = object$time, x0 = NULL,
                                     branch = c("stable", "wald", "unbounded"),
                                     time_origin = object$time_origin, ...) {
  if (!is.numeric(new_time) || !length(new_time) || any(!is.finite(new_time))) {
    stop("`new_time` must be a nonempty finite numeric vector.", call. = FALSE)
  }
  stage <- .resolve_branch(object, branch)
  x0 <- x0 %||% stage$x0_hat
  if (!is.numeric(x0) || length(x0) != nrow(stage$A_hat) || any(!is.finite(x0))) {
    stop("`x0` must be a finite state vector of the fitted dimension.",
         call. = FALSE)
  }
  .matrix_trajectory(stage$A_hat, new_time, x0, origin = time_origin)
}

#' Plot an AdaStableNet Fit
#'
#' @param x An `adastablenet_fit` object.
#' @param type Plot fitted trajectories or estimated eigenvalues.
#' @param branch One of `"stable"`, `"wald"`, or `"unbounded"`.
#' @param state State indices for trajectory panels.
#' @param ... Additional graphical parameters passed to base plotting functions.
#'
#' @return `x`, invisibly.
#' @export
plot.adastablenet_fit <- function(x,
                                  type = c("trajectories", "eigenvalues"),
                                  branch = c("stable", "wald", "unbounded"),
                                  state = seq_len(min(4L, ncol(x$data))), ...) {
  type <- match.arg(type)
  stage <- .resolve_branch(x, branch)
  if (type == "eigenvalues") {
    z <- stage$eigenvalues
    graphics::plot(Re(z), Im(z), xlab = "Real part", ylab = "Imaginary part",
                   main = paste("AdaStableNet", match.arg(branch), "spectrum"), ...)
    graphics::abline(v = 0, lty = 2, col = "grey60")
  } else {
    if (any(state < 1L | state > ncol(x$data))) {
      stop("`state` contains an invalid state index.", call. = FALSE)
    }
    old_par <- graphics::par(no.readonly = TRUE)
    on.exit(graphics::par(old_par), add = TRUE)
    graphics::par(mfrow = grDevices::n2mfrow(length(state)))
    fitted_values <- t(stage$X_hat)
    for (j in state) {
      graphics::plot(x$time, x$data[, j], type = "p", pch = 16, cex = 0.55,
                     xlab = "Time", ylab = paste("State", j), ...)
      graphics::lines(x$time, fitted_values[, j], col = "#0072B2", lwd = 2)
    }
  }
  invisible(x)
}
