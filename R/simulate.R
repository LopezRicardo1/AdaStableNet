#' Simulate a Linear ODE for AdaStableNet
#'
#' Generates a diagonalizable real linear ODE with controlled spectral regime
#' and non-normality. The same system matrix is shared across trajectories while
#' initial conditions and observation errors vary.
#'
#' @param p Number of states.
#' @param n_time Number of equally spaced observation times.
#' @param time_range Length-two time interval.
#' @param spectrum Spectral regime: asymptotically stable, marginal, mixed,
#'   near-boundary, or unstable.
#' @param sigma Observation-noise standard deviation.
#' @param condition_number Target condition number of the modal loading matrix.
#' @param n_trajectories Number of independent initial conditions.
#' @param seed Optional random seed.
#'
#' @return An object of class `adastablenet_simulation`. For one trajectory,
#'   `X` and `Y` are time-by-state matrices; otherwise they are
#'   time-by-state-by-trajectory arrays.
#' @examples
#' sim <- simulate_adastablenet(p = 4, n_time = 21, seed = 42)
#' sim
#' @export
simulate_adastablenet <- function(p = 4L, n_time = 51L,
                                  time_range = c(0, 1),
                                  spectrum = c("stable", "marginal", "mixed",
                                               "near_boundary", "unstable"),
                                  sigma = 0.1, condition_number = 5,
                                  n_trajectories = 1L, seed = NULL) {
  spectrum <- match.arg(spectrum)
  .validate_scalar(p, "p", lower = 2, integer = TRUE)
  .validate_scalar(n_time, "n_time", lower = 5, integer = TRUE)
  .validate_scalar(sigma, "sigma", lower = 0)
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
    a <- switch(
      spectrum,
      stable = -seq(0.08, 0.25, length.out = q),
      marginal = rep(0, q),
      mixed = ifelse(seq_len(q) %% 2L, 0, -0.15),
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
    U <- qr.Q(qr(matrix(stats::rnorm(p^2), p, p)))
    V <- qr.Q(qr(matrix(stats::rnorm(p^2), p, p)))
    singular_values <- exp(seq(0, log(condition_number), length.out = p))
    Q <- U %*% diag(singular_values, p) %*% t(V)
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
      condition_number = .safe_kappa(Q),
      n_trajectories = n_trajectories,
      seed = seed
    ), class = "adastablenet_simulation")
  })
}

#' @export
print.adastablenet_simulation <- function(x, ...) {
  p <- nrow(x$A)
  n_time <- length(x$time)
  cat("AdaStableNet simulation\n")
  cat("  Spectrum: ", x$spectrum, "\n", sep = "")
  cat("  States: ", p, "\n", sep = "")
  cat("  Time points: ", n_time, "\n", sep = "")
  cat("  Trajectories: ", x$n_trajectories, "\n", sep = "")
  cat("  Noise SD: ", x$sigma, "\n", sep = "")
  cat("  Loading condition number: ",
      format(x$condition_number, digits = 4), "\n", sep = "")
  invisible(x)
}
