#' Wald-Sparsified Directed Network from an AdaStableNet Fit
#'
#' Converts coefficientwise Wald tests into a directed interaction network for
#' the linear ODE `X'(t) = A X(t)`. Entry `A[i, j]` represents the edge
#' `j -> i`. Multiple-testing adjustment is applied only to the `p * (p - 1)`
#' off-diagonal edges. Diagonal entries describe self-dynamics and are handled
#' separately rather than being counted as network edges.
#'
#' The returned `adjacency` is binary, `A_network` contains the selected signed
#' edge weights with a zero diagonal, and `A_sparse` is a dynamic coefficient
#' matrix. By default, `A_sparse` retains every estimated diagonal coefficient
#' and only the selected off-diagonal coefficients. Because elementwise
#' thresholding does not preserve eigenvalues, stability diagnostics are
#' recomputed after sparsification.
#'
#' @param fit An `adastablenet_fit` or `adastablenet_stage` object.
#' @param Y Observed data in state-by-time or time-by-state orientation.
#' @param tt Observation times.
#' @param branch Branch used when `fit` is a complete fit. The two-stage option
#'   uses the functional-data initializer and the first observed state as its
#'   plug-in initial condition.
#' @param method Multiplicity method applied to off-diagonal p values and passed
#'   to [stats::p.adjust()].
#' @param alpha Edge-selection significance level.
#' @param diagonal Treatment of diagonal coefficients in `A_sparse`: retain all
#'   (`"keep"`), test them as a separate multiplicity family (`"test"`), or
#'   set them to zero (`"zero"`). This choice never changes the edge network.
#' @param nsteps Integration intervals for [TheoVar()].
#' @param variance_ridge Fisher-information ridge multiplier.
#' @param stability_horizon Nonnegative horizon for transient-growth checks
#'   before and after sparsification.
#' @param stability_n_grid Number of times in each transient-growth check.
#'
#' @return An object of class `adastablenet_wald_network` containing the full
#'   coefficient estimate, standard errors, Wald scores, raw and adjusted p
#'   values, edge table, binary adjacency matrix, weighted network,
#'   sparsified dynamic matrix, and pre/post-sparsification stability
#'   diagnostics.
#' @references Wu, L., Qiu, X., Yuan, Y.-X., and Wu, H. (2019).
#'   Parameter estimation and variable selection for big systems of linear
#'   ordinary differential equations: A matrix-based approach. *Journal of the
#'   American Statistical Association*, 114(526), 657-667.
#' @examples
#' sim <- simulate_adastablenet(p = 3, n_time = 21, sigma = 0.03, seed = 4)
#' fit <- FitAdaStableNet(
#'   sim$Y, sim$time, nbasis = 8, twoSE = FALSE,
#'   num_iter = 30, wald_nsteps = 4, verbose = FALSE
#' )
#' net <- AdaStableNet_WaldNetwork(
#'   fit, sim$Y, sim$time, branch = "stable", nsteps = 4,
#'   stability_n_grid = 11
#' )
#' net$adjacency
#' net$A_sparse
#' @export
AdaStableNet_WaldNetwork <- function(
    fit, Y, tt,
    branch = c("stable", "wald", "unbounded", "two_stage"),
    method = "BH", alpha = 0.05,
    diagonal = c("keep", "test", "zero"),
    nsteps = 20L, variance_ridge = 1e-6,
    stability_horizon = 1, stability_n_grid = 101L) {
  diagonal <- match.arg(diagonal)
  tt <- .validate_time(tt)
  Y_input <- Y
  if (is.data.frame(Y_input)) Y_input <- as.matrix(Y_input)
  if (!is.matrix(Y_input) || !is.numeric(Y_input) || any(!is.finite(Y_input))) {
    stop("`Y` must be a finite numeric matrix.", call. = FALSE)
  }

  p <- if (inherits(fit, "adastablenet_stage")) {
    nrow(fit$A_hat)
  } else if (inherits(fit, "adastablenet_fit")) {
    ncol(fit$data)
  } else {
    stop("`fit` must be an AdaStableNet fit or fitted stage.", call. = FALSE)
  }
  orientation <- .orient_wald_data(Y_input, tt, p)
  Y_state_time <- orientation$Y_state_time
  resolved <- .resolve_wald_stage(fit, Y_state_time, tt, branch)
  stage <- resolved$stage
  .validate_wald_controls(method, alpha, nsteps, variance_ridge)
  .validate_scalar(stability_horizon, "stability_horizon", lower = 0)
  .validate_scalar(stability_n_grid, "stability_n_grid", lower = 2,
                   integer = TRUE)

  details <- .coefficient_wald_details(
    stage, Y_state_time, tt,
    nsteps = nsteps, variance_ridge = variance_ridge
  )
  p <- nrow(stage$A_hat)
  off_diagonal <- row(stage$A_hat) != col(stage$A_hat)
  diagonal_mask <- !off_diagonal
  p_adjusted <- matrix(NA_real_, p, p)
  p_adjusted[off_diagonal] <- stats::p.adjust(
    details$p_values[off_diagonal], method = method
  )
  if (diagonal == "test") {
    p_adjusted[diagonal_mask] <- stats::p.adjust(
      details$p_values[diagonal_mask], method = method
    )
  }

  selected <- matrix(FALSE, p, p)
  selected[off_diagonal] <- is.finite(p_adjusted[off_diagonal]) &
    p_adjusted[off_diagonal] <= alpha
  adjacency <- matrix(as.integer(selected), p, p)
  A_network <- stage$A_hat
  A_network[!selected] <- 0
  diag(A_network) <- 0
  A_sparse <- stage$A_hat
  A_sparse[off_diagonal & !selected] <- 0
  if (diagonal == "test") {
    keep_diagonal <- is.finite(p_adjusted[diagonal_mask]) &
      p_adjusted[diagonal_mask] <= alpha
    diagonal_values <- A_sparse[diagonal_mask]
    diagonal_values[!keep_diagonal] <- 0
    A_sparse[diagonal_mask] <- diagonal_values
  } else if (diagonal == "zero") {
    diag(A_sparse) <- 0
  }

  state_names <- orientation$state_names
  if (is.null(state_names) || length(state_names) != p ||
      any(!nzchar(state_names)) || anyDuplicated(state_names)) {
    state_names <- paste0("X", seq_len(p))
  }
  matrix_names <- list(target = state_names, source = state_names)
  matrices <- list(
    A_estimate = stage$A_hat,
    standard_error = details$standard_error,
    z_scores = details$z_scores,
    p_values = details$p_values,
    p_adjusted = p_adjusted,
    selected = selected,
    adjacency = adjacency,
    A_network = A_network,
    A_sparse = A_sparse
  )
  matrices <- lapply(matrices, function(x) {
    dimnames(x) <- matrix_names
    x
  })

  edge_index <- which(off_diagonal, arr.ind = TRUE)
  edges <- data.frame(
    source_index = edge_index[, "col"],
    target_index = edge_index[, "row"],
    source = state_names[edge_index[, "col"]],
    target = state_names[edge_index[, "row"]],
    estimate = stage$A_hat[edge_index],
    standard_error = details$standard_error[edge_index],
    z_score = details$z_scores[edge_index],
    p_value = details$p_values[edge_index],
    p_adjusted = p_adjusted[edge_index],
    selected = selected[edge_index],
    sign = sign(stage$A_hat[edge_index]),
    stringsAsFactors = FALSE
  )
  edges <- edges[order(edges$source_index, edges$target_index), , drop = FALSE]
  rownames(edges) <- NULL

  before <- stability_diagnostics(
    structure(list(A_hat = matrices$A_estimate),
              class = "adastablenet_stage"),
    horizon = stability_horizon, n_grid = stability_n_grid
  )
  after <- stability_diagnostics(
    structure(list(A_hat = matrices$A_sparse),
              class = "adastablenet_stage"),
    horizon = stability_horizon, n_grid = stability_n_grid
  )

  structure(c(list(
    branch = resolved$branch,
    alpha = alpha,
    method = method,
    diagonal = diagonal,
    state_names = state_names,
    x0_hat = as.numeric(stage$x0_hat),
    sigma = details$sigma,
    residual_df = details$residual_df,
    edge_table = edges,
    n_possible_edges = p * (p - 1L),
    n_selected_edges = sum(selected),
    stability = list(before = before, after = after)
  ), matrices), class = "adastablenet_wald_network")
}

#' @export
print.adastablenet_wald_network <- function(x, ...) {
  branch <- if (is.na(x$branch)) "fitted stage" else x$branch
  cat("AdaStableNet Wald-sparsified network\n")
  cat("  Branch: ", branch, "\n", sep = "")
  cat("  Selected edges: ", x$n_selected_edges, "/",
      x$n_possible_edges, " (", x$method, ", alpha = ",
      format(x$alpha), ")\n", sep = "")
  cat("  Diagonal in dynamic matrix: ", x$diagonal, "\n", sep = "")
  cat("  Spectral abscissa before/after: ",
      format(x$stability$before$spectral_abscissa, digits = 5), " / ",
      format(x$stability$after$spectral_abscissa, digits = 5), "\n", sep = "")
  invisible(x)
}

.orient_wald_data <- function(Y, tt, p) {
  if (ncol(Y) == length(tt) && nrow(Y) == p) {
    list(Y_state_time = unname(Y), state_names = rownames(Y))
  } else if (nrow(Y) == length(tt) && ncol(Y) == p) {
    list(Y_state_time = unname(t(Y)), state_names = colnames(Y))
  } else {
    stop("Dimensions of `Y`, `tt`, and `fit` do not agree.", call. = FALSE)
  }
}

.validate_wald_controls <- function(method, alpha, nsteps, variance_ridge) {
  if (length(method) != 1L || !is.character(method) ||
      !method %in% stats::p.adjust.methods) {
    stop("Unknown multiple-testing method `", method, "`.", call. = FALSE)
  }
  .validate_scalar(alpha, "alpha", lower = 0, upper = 1,
                   lower_open = TRUE, upper_open = TRUE)
  .validate_scalar(nsteps, "nsteps", lower = 2, integer = TRUE)
  .validate_scalar(variance_ridge, "variance_ridge", lower = 0)
  invisible(NULL)
}

.resolve_wald_stage <- function(fit, Y_state_time, tt, branch) {
  if (inherits(fit, "adastablenet_stage")) {
    return(list(stage = fit, branch = NA_character_))
  }
  branch <- match.arg(
    branch, c("stable", "wald", "unbounded", "two_stage")
  )
  list(stage = .resolve_branch(fit, branch), branch = branch)
}

.coefficient_wald_details <- function(fit, Y_state_time, tt,
                                      nsteps, variance_ridge) {
  if (!is.matrix(fit$X_hat) || !all(dim(fit$X_hat) == dim(Y_state_time))) {
    stop("The fitted trajectory and `Y` must have matching dimensions.",
         call. = FALSE)
  }
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
  standard_error <- sqrt(pmax(variance$varmat, 0))
  z_scores <- matrix(NA_real_, nrow(fit$A_hat), ncol(fit$A_hat))
  regular <- is.finite(standard_error) & standard_error > 0
  z_scores[regular] <- fit$A_hat[regular] / standard_error[regular]
  degenerate <- is.finite(standard_error) & standard_error == 0 &
    is.finite(fit$A_hat)
  z_scores[degenerate & fit$A_hat == 0] <- 0
  nonzero_degenerate <- degenerate & fit$A_hat != 0
  z_scores[nonzero_degenerate] <- sign(fit$A_hat[nonzero_degenerate]) * Inf
  p_values <- 2 * stats::pnorm(abs(z_scores), lower.tail = FALSE)
  list(
    standard_error = standard_error,
    z_scores = z_scores,
    p_values = p_values,
    sigma = sigma,
    residual_df = residual_df
  )
}
