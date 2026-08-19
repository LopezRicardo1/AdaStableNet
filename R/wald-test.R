#' Coefficientwise Wald Tests for an AdaStableNet Fit
#'
#' Computes coefficientwise Wald statistics for the entries of the fitted
#' system matrix using the trajectory-sensitivity tensor and total Fisher
#' information developed for matrix-based linear ODE estimation by Wu et al.
#' (2019), and optionally applies a multiple-testing correction. The current
#' plug-in implementation conditions on the fitted initial state. For a
#' stability-constrained estimate on the boundary, its ordinary normal
#' reference distribution remains a working approximation and should be
#' calibrated by simulation.
#'
#' @param fit An `adastablenet_stage` object or a stage extracted from a fitted
#'   model.
#' @param Y Observed data in state-by-time or time-by-state orientation.
#' @param tt Observation times.
#' @param method Method passed to [stats::p.adjust()].
#' @param alpha Significance level.
#' @param return Return z scores, adjusted p values, or a thresholded matrix.
#' @param nsteps Integration intervals for [TheoVar()].
#' @param variance_ridge Fisher-information ridge multiplier.
#'
#' @return A numeric matrix. For off-diagonal network recovery with separate
#'   treatment of self-dynamics, use [AdaStableNet_WaldNetwork()].
#' @references Wu, L., Qiu, X., Yuan, Y.-X., and Wu, H. (2019).
#'   Parameter estimation and variable selection for big systems of linear
#'   ordinary differential equations: A matrix-based approach. *Journal of the
#'   American Statistical Association*, 114(526), 657-667.
#' @importFrom stats p.adjust pnorm
#' @export
AdaStableNet_WaldTest <- function(fit, Y, tt, method = "BH", alpha = 0.05,
                                  return = c("zscores", "pvals", "A_thresh"),
                                  nsteps = 20L, variance_ridge = 1e-6) {
  return <- match.arg(return)
  if (!inherits(fit, "adastablenet_stage")) {
    stop("`fit` must be an `adastablenet_stage` object.", call. = FALSE)
  }
  tt <- .validate_time(tt)
  if (is.data.frame(Y)) Y <- as.matrix(Y)
  if (!is.matrix(Y) || !is.numeric(Y)) {
    stop("`Y` must be a numeric matrix.", call. = FALSE)
  }
  if (ncol(Y) == length(tt) && nrow(Y) == nrow(fit$A_hat)) {
    Y_state_time <- Y
  } else if (nrow(Y) == length(tt) && ncol(Y) == nrow(fit$A_hat)) {
    Y_state_time <- t(Y)
  } else {
    stop("Dimensions of `Y`, `tt`, and `fit` do not agree.", call. = FALSE)
  }
  if (!method %in% stats::p.adjust.methods) {
    stop("Unknown multiple-testing method `", method, "`.", call. = FALSE)
  }
  .validate_scalar(alpha, "alpha", lower = 0, upper = 1,
                   lower_open = TRUE, upper_open = TRUE)
  residual_df <- max(length(Y_state_time) - nrow(Y_state_time)^2, 1L)
  sigma <- sqrt(sum((Y_state_time - fit$X_hat)^2) / residual_df)
  variance <- TheoVar(
    sigma = max(sigma, sqrt(.Machine$double.eps)),
    X0 = fit$x0_hat,
    A = fit$A_hat,
    tt = tt - min(tt),
    nsteps = nsteps,
    L2 = variance_ridge
  )
  A_sd <- sqrt(pmax(variance$varmat, 0))
  zmat <- fit$A_hat / A_sd
  p_mat <- 2 * pnorm(abs(zmat), lower.tail = FALSE)
  p_adjusted <- matrix(
    p.adjust(as.vector(p_mat), method = method),
    nrow = nrow(p_mat), ncol = ncol(p_mat)
  )
  A_thresholded <- fit$A_hat
  A_thresholded[p_adjusted > alpha | !is.finite(p_adjusted)] <- 0
  switch(return, zscores = zmat, pvals = p_adjusted, A_thresh = A_thresholded)
}
