#' Single-Trajectory Identifiability Diagnostics
#'
#' Computes the initial-condition identifiability diagnostics of Qiu et al.
#' using `ode.ident::ICISAnalysis()`. Identifiability is a property of the pair
#' `(A, x0)`, so the supplied initial state must correspond to the same time
#' origin used for the trajectory.
#'
#' @param A A finite numeric square coefficient matrix.
#' @param x0 Initial state with length `nrow(A)`.
#' @param n.digits Number of digits used by `ode.ident` when detecting zero
#'   modal excitation and repeated eigenvalues.
#'
#' @return A one-row data frame with structural identifiability, its two
#'   component checks, the initial-condition identifiability score (ICIS),
#'   minimum eigenvalue gap, and invariant-subspace excitation scores.
#' @export
identifiability_diagnostics <- function(A, x0, n.digits = 6L) {
  if (!requireNamespace("ode.ident", quietly = TRUE)) {
    stop(
      "Package `ode.ident` is required; install it from ",
      "https://github.com/qiuxing/ode.ident.", call. = FALSE
    )
  }
  if (!is.matrix(A) || !is.numeric(A) || nrow(A) != ncol(A) ||
      any(!is.finite(A))) {
    stop("`A` must be a finite numeric square matrix.", call. = FALSE)
  }
  x0 <- as.numeric(x0)
  if (length(x0) != nrow(A) || any(!is.finite(x0))) {
    stop("`x0` must be finite and have length `nrow(A)`.", call. = FALSE)
  }
  .validate_scalar(n.digits, "n.digits", lower = 1, integer = TRUE)
  result <- ode.ident::ICISAnalysis(A, x0, n.digits = as.integer(n.digits))
  data.frame(
    identifiable = isTRUE(result$Identifiable),
    distinct_eigenvalues = isTRUE(result$Ident1),
    all_invariant_subspaces_excited = isTRUE(result$Ident2),
    ICIS = as.numeric(result$ICIS),
    minimum_eigenvalue_gap = as.numeric(result$Jordan$Lgap),
    unexcited_dimension = sum(diag(result$I0)),
    invariant_subspace_excitation = I(list(as.numeric(result$w0k.norm))),
    stringsAsFactors = FALSE
  )
}
