#' Simulate a Linear ODE for AdaStableNet
#'
#' Generates a diagonalizable real linear ODE with controlled spectral regime,
#' matrix structure, and non-normality. The default sparse construction follows
#' the block-embedding and thresholding design used in the original eigen-bound
#' simulation. The same system matrix is shared across trajectories while
#' initial conditions and observation errors vary.
#'
#' @param p Number of states.
#' @param n_time Number of equally spaced observation times.
#' @param time_range Length-two time interval.
#' @param spectrum Spectral regime: asymptotically stable, marginal, mixed,
#'   near-boundary, or unstable.
#' @param sigma Observation-noise standard deviation.
#' @param matrix_structure Generate an eigen-bound-style sparse system or a
#'   dense conditioned system.
#' @param sparse_threshold Quantile of the nonzero absolute loading entries set
#'   to zero in the sparse construction. The default reproduces median
#'   thresholding from the eigen-bound simulation.
#' @param condition_number For the dense construction, the target condition
#'   number of the modal loading matrix. For the sparse construction, the
#'   maximum accepted condition number; the identity is used when this equals
#'   one.
#' @param n_trajectories Number of independent initial conditions.
#' @param seed Optional random seed.
#'
#' @return An object of class `adastablenet_simulation` containing the system
#'   matrices `A`, `J`, and `Q`; noise-free and observed trajectories `X` and
#'   `Y`; the time grid; modal parameters; eigenvalues; and matrix-structure
#'   diagnostics. `A_sparsity` and `Q_sparsity` are the proportions of entries
#'   with absolute magnitude below `1e-12`. For one trajectory, `X` and `Y`
#'   are time-by-state matrices; otherwise they are time-by-state-by-trajectory
#'   arrays.
#' @examples
#' sim <- simulate_adastablenet(p = 4, n_time = 21, seed = 42)
#' sim
#' @export
simulate_adastablenet <- function(p = 4L, n_time = 51L,
                                  time_range = c(0, 1),
                                  spectrum = c("stable", "marginal", "mixed",
                                               "near_boundary", "unstable"),
                                  sigma = 0.1,
                                  matrix_structure = c("sparse", "dense"),
                                  sparse_threshold = 0.5,
                                  condition_number = 5,
                                  n_trajectories = 1L, seed = NULL) {
  spectrum <- match.arg(spectrum)
  matrix_structure <- match.arg(matrix_structure)
  .validate_scalar(p, "p", lower = 2, integer = TRUE)
  .validate_scalar(n_time, "n_time", lower = 5, integer = TRUE)
  .validate_scalar(sigma, "sigma", lower = 0)
  .validate_scalar(sparse_threshold, "sparse_threshold", lower = 0, upper = 1,
                   upper_open = TRUE)
  .validate_scalar(condition_number, "condition_number", lower = 1)
  .validate_scalar(n_trajectories, "n_trajectories", lower = 1, integer = TRUE)
  if (!is.numeric(time_range) || length(time_range) != 2L ||
      any(!is.finite(time_range)) || time_range[1] >= time_range[2]) {
    stop("`time_range` must be two increasing finite values.", call. = FALSE)
  }

  .with_seed(seed, {
    p <- as.integer(p)
    q <- p %/% 2L
    r <- p - 2L * q
    frequencies <- 2 * pi * seq_len(q)
    if (matrix_structure == "sparse" && q > 1L) {
      frequencies <- frequencies[sample.int(q)]
    }
    a <- switch(
      spectrum,
      stable = -seq(0.08, 0.25, length.out = q),
      marginal = rep(0, q),
      mixed = if (r) rep(0, q) else ifelse(seq_len(q) %% 2L, 0, -0.15),
      near_boundary = rep(-0.01, q),
      unstable = rep(0.08, q)
    )
    cc <- if (r) {
      switch(
        spectrum,
        stable = -0.18,
        marginal = 0,
        mixed = -0.1,
        near_boundary = -0.01,
        unstable = 0.08
      )
    } else {
      numeric()
    }
    J <- Jordan(a, frequencies, cc)
    if (matrix_structure == "sparse") {
      loading <- .eigen_bound_sparse_loading(
        p = p,
        threshold = sparse_threshold,
        max_condition = condition_number
      )
      Q <- loading$Q
      condition_target_met <- loading$condition_target_met
    } else {
      U <- qr.Q(qr(matrix(stats::rnorm(p^2), p, p)))
      V <- qr.Q(qr(matrix(stats::rnorm(p^2), p, p)))
      singular_values <- exp(seq(0, log(condition_number), length.out = p))
      Q <- U %*% diag(singular_values, p) %*% t(V)
      condition_target_met <- TRUE
    }
    A <- Q %*% J %*% solve(Q)
    modal_x0 <- matrix(stats::rnorm(p * n_trajectories), p, n_trajectories)
    x0 <- Q %*% modal_x0
    time <- seq(time_range[1], time_range[2], length.out = n_time)
    relative_time <- time - time_range[1]
    X <- array(NA_real_, dim = c(n_time, p, n_trajectories))
    for (trajectory in seq_len(n_trajectories)) {
      X[, , trajectory] <- .matrix_trajectory(
        A, relative_time, x0[, trajectory], origin = 0
      )
    }
    Y <- X + stats::rnorm(length(X), sd = sigma)
    dim(Y) <- dim(X)
    if (n_trajectories == 1L) {
      X <- X[, , 1L, drop = TRUE]
      Y <- Y[, , 1L, drop = TRUE]
      x0 <- as.numeric(x0[, 1L])
    }
    structure(list(
      time = time,
      X = X,
      Y = Y,
      x0 = x0,
      A = unname(A),
      J = unname(J),
      Q = unname(Q),
      eigenvalues = eigen(A, only.values = TRUE)$values,
      modal_parameters = list(a = a, b = frequencies,
                              cc = if (length(cc)) cc else NULL),
      sigma = sigma,
      spectrum = spectrum,
      matrix_structure = matrix_structure,
      sparse_threshold = sparse_threshold,
      A_sparsity = mean(abs(A) < 1e-12),
      Q_sparsity = mean(abs(Q) < 1e-12),
      condition_number = .safe_kappa(Q),
      requested_condition_number = condition_number,
      condition_target_met = condition_target_met,
      n_trajectories = n_trajectories,
      seed = seed
    ), class = "adastablenet_simulation")
  })
}

.eigen_bound_sparse_loading <- function(p, threshold, max_condition,
                                        max_attempts = 200L) {
  if (max_condition <= 1 + sqrt(.Machine$double.eps)) {
    return(list(Q = diag(p), condition_target_met = TRUE))
  }

  if (p >= 15L) {
    glue_groups <- list(c(2L, 4L, 6L), c(9L, 12L, 15L))
  } else {
    first_group <- unique(as.integer(round(seq(
      1, max(2, ceiling(p / 2)), length.out = min(3L, p)
    ))))
    second_group <- if (p >= 6L) {
      unique(as.integer(round(seq(floor(p / 2) + 1L, p, length.out = 3L))))
    } else {
      integer()
    }
    glue_groups <- list(first_group, second_group)
  }

  best_Q <- NULL
  best_condition <- Inf
  for (attempt in seq_len(max_attempts)) {
    Q <- diag(p)
    for (indices in glue_groups) {
      if (length(indices) >= 2L) {
        Q[indices, indices] <- qr.Q(qr(matrix(
          stats::rnorm(length(indices)^2), length(indices), length(indices)
        )))
      }
    }

    local_rotation <- diag(p)
    block_size <- min(if (p %% 2L) 4L else 5L, p)
    block_indices <- seq_len(block_size)
    local_rotation[block_indices, block_indices] <- qr.Q(qr(matrix(
      stats::rnorm(block_size^2), block_size, block_size
    )))
    permutation <- sample.int(p)
    local_rotation <- local_rotation[permutation, permutation, drop = FALSE]
    Q <- Q %*% local_rotation

    nonzero <- abs(Q[Q != 0])
    cutoff <- stats::quantile(nonzero, threshold, names = FALSE, type = 7)
    Q[abs(Q) < cutoff] <- 0
    if (qr(Q)$rank < p) next

    current_condition <- .safe_kappa(Q)
    if (current_condition < best_condition) {
      best_Q <- Q
      best_condition <- current_condition
    }
    if (current_condition <= max_condition) {
      return(list(Q = Q, condition_target_met = TRUE))
    }
  }

  if (is.null(best_Q)) {
    stop(
      "Unable to construct a full-rank sparse modal loading matrix. ",
      "Reduce `sparse_threshold`.", call. = FALSE
    )
  }
  warning(
    "The sparse loading construction could not meet `condition_number`; ",
    "returning the best full-rank candidate.", call. = FALSE
  )
  list(Q = best_Q, condition_target_met = FALSE)
}

#' @export
print.adastablenet_simulation <- function(x, ...) {
  p <- nrow(x$A)
  n_time <- length(x$time)
  cat("AdaStableNet simulation\n")
  cat("  Spectrum: ", x$spectrum, "\n", sep = "")
  cat("  Matrix structure: ", x$matrix_structure, "\n", sep = "")
  cat("  States: ", p, "\n", sep = "")
  cat("  Time points: ", n_time, "\n", sep = "")
  cat("  Trajectories: ", x$n_trajectories, "\n", sep = "")
  cat("  Noise SD: ", x$sigma, "\n", sep = "")
  cat("  Loading condition number: ",
      format(x$condition_number, digits = 4), "\n", sep = "")
  cat("  A sparsity: ", format(100 * x$A_sparsity, digits = 4), "%\n",
      sep = "")
  invisible(x)
}
