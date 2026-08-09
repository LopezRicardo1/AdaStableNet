#' Assemble an AdaStableNet Modal Fit
#'
#' Profiles the modal loading matrix and reconstructs the dynamic matrix from a
#' supplied spectrum. This lower-level function is exported for compatibility
#' and methodological inspection.
#'
#' @param a Real parts of complex-conjugate modes.
#' @param b Nonnegative imaginary parts of complex-conjugate modes.
#' @param cc Optional real eigenvalues.
#' @param tt Numeric times, usually shifted to start at zero.
#' @param Y Numeric matrix with states in rows and times in columns.
#' @param ridge.pen Nonnegative ridge multiplier.
#' @param a_wald Optional zero-one mask for `a`.
#' @param cc_wald Optional zero-one mask for `cc`.
#' @param diagnostics Optional optimizer diagnostics.
#' @param time_origin Original time represented by zero in `tt`.
#'
#' @return An object of class `adastablenet_stage` containing `A_hat`,
#'   `x0_hat`, fitted trajectories, modal parameters, and diagnostics.
#' @export
ada_output <- function(a, b, cc = NULL, tt, Y, ridge.pen = 1e-3,
                       a_wald = NULL, cc_wald = NULL,
                       diagnostics = NULL, time_origin = 0) {
  tt <- .validate_time(tt, min_length = 2L)
  Y <- .validate_data(Y, tt, "state_by_time")
  .validate_scalar(ridge.pen, "ridge.pen", lower = 0)
  a <- as.numeric(a)
  b <- as.numeric(b)
  cc_numeric <- if (is.null(cc)) numeric() else as.numeric(cc)
  a_wald <- as.numeric(a_wald %||% rep(1, length(a)))
  cc_wald <- as.numeric(cc_wald %||% rep(1, length(cc_numeric)))
  effective_a <- a * a_wald
  effective_cc <- cc_numeric * cc_wald
  mode_dimension <- 2L * length(a) + length(cc_numeric)
  if (mode_dimension != nrow(Y)) {
    stop(
      "The modal dimension `2 * length(a) + length(cc)` must equal ",
      "the number of states in `Y`.", call. = FALSE
    )
  }

  J <- Jordan(effective_a, b, effective_cc)
  S <- ode_basis(tt, effective_a, b, effective_cc)
  profile <- .profile_projection(Y, S, ridge.pen)
  P <- profile$P
  P_inv <- .safe_pinv(P)
  A_hat <- P %*% J %*% P_inv
  S_origin <- ode_basis(0, effective_a, b, effective_cc)
  x0_hat <- as.numeric(P %*% S_origin)
  actual_eigenvalues <- eigen(A_hat, only.values = TRUE)$values
  rank_P <- qr(P)$rank

  diagnostics <- utils::modifyList(
    diagnostics %||% list(),
    list(
      loss = profile$loss,
      modal_rank = rank_P,
      full_modal_rank = rank_P == nrow(P),
      loading_condition = .safe_kappa(P),
      gram_condition = profile$gram_condition,
      spectral_abscissa = max(Re(actual_eigenvalues))
    )
  )

  structure(list(
    A_hat = unname(A_hat),
    x0_hat = x0_hat,
    X_hat = unname(profile$X_hat),
    residuals = unname(Y - profile$X_hat),
    P = unname(P),
    ODE_Basis = unname(S),
    J = unname(J),
    eigenvalues = actual_eigenvalues,
    target_eigenvalues = .modal_eigenvalues(effective_a, b, effective_cc),
    modal_parameters = list(
      a = effective_a,
      b = b,
      cc = if (length(effective_cc)) effective_cc else NULL,
      a_mask = a_wald,
      cc_mask = if (length(cc_wald)) cc_wald else NULL
    ),
    time = tt + time_origin,
    relative_time = tt,
    time_origin = time_origin,
    diagnostics = diagnostics
  ), class = "adastablenet_stage")
}
