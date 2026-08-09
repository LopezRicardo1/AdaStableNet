#' Approximate Variance for a Linear ODE System
#'
#' Approximates the Fisher information for `vec(A)` using trajectory
#' sensitivities and obtains delta-method variances for the real parts of the
#' eigenvalues. The calculation conditions on the supplied initial state.
#'
#' @param sigma Positive residual standard deviation.
#' @param X0 State at time zero on the supplied time scale.
#' @param A Numeric square system matrix.
#' @param tt Nonnegative times relative to the initial state.
#' @param nsteps Number of trapezoidal integration intervals.
#' @param L2 Nonnegative ridge multiplier for Fisher-information inversion.
#'
#' @return A list containing sensitivity arrays, Fisher information,
#'   coefficient variances, eigenvalue variances, and eigenvalues.
#' @importFrom expm expm
#' @export
TheoVar <- function(sigma, X0, A, tt, nsteps = 20L, L2 = 1e-6) {
  .validate_scalar(sigma, "sigma", lower = 0, lower_open = TRUE)
  .validate_scalar(nsteps, "nsteps", lower = 2, integer = TRUE)
  .validate_scalar(L2, "L2", lower = 0)
  if (!is.matrix(A) || !is.numeric(A) || nrow(A) != ncol(A) ||
      any(!is.finite(A))) {
    stop("`A` must be a finite numeric square matrix.", call. = FALSE)
  }
  p <- nrow(A)
  X0 <- as.numeric(X0)
  if (length(X0) != p || any(!is.finite(X0))) {
    stop("`X0` must be a finite vector with length `nrow(A)`.", call. = FALSE)
  }
  tt <- as.numeric(tt)
  if (!length(tt) || any(!is.finite(tt)) || any(tt < 0)) {
    stop("`tt` must contain finite nonnegative relative times.", call. = FALSE)
  }

  Dt <- array(0, dim = c(length(tt), p, p^2))
  grid <- seq(0, 1, length.out = nsteps + 1L)
  weights <- c(0.5, rep(1, nsteps - 1L), 0.5)
  for (j in seq_along(tt)) {
    tj <- tt[j]
    if (tj == 0) next
    Dtj <- matrix(0, p, p^2)
    for (s_idx in seq_along(grid)) {
      s <- grid[s_idx]
      Xts <- expm(tj * s * A) %*% X0
      eAts <- expm(tj * (1 - s) * A)
      Dtj <- Dtj + weights[s_idx] * (t(Xts) %x% eAts)
    }
    Dt[j, , ] <- Dtj * tj / nsteps
  }

  FisherI <- matrix(0, p^2, p^2)
  for (j in seq_along(tt)) {
    FisherI <- FisherI + crossprod(Dt[j, , ])
  }
  FisherI <- FisherI / sigma^2
  FisherI_inv <- ridge.inv(FisherI, lambda.prop = L2)
  varmat <- matrix(diag(FisherI_inv), p, p)

  ee <- eigen(A)
  eigen_variance <- rep(NA_real_, p)
  inverse_vectors <- tryCatch(solve(ee$vectors), error = function(e) NULL)
  if (!is.null(inverse_vectors)) {
    for (k in seq_len(p)) {
      u_k <- ee$vectors[, k]
      v_k <- inverse_vectors[k, , drop = FALSE]
      derivative <- Re(u_k %x% t(v_k))
      eigen_variance[k] <- drop(t(derivative) %*% FisherI_inv %*% derivative)
    }
  }

  list(
    Dt = Dt,
    FisherI = FisherI,
    FisherI_inv = FisherI_inv,
    varmat = varmat,
    re.eigs.sigma2 = eigen_variance,
    eigenvalues = ee$values
  )
}

.greedy_eigen_match <- function(target, source) {
  if (!length(target)) return(integer())
  cost <- outer(target, source, function(x, y) Mod(x - y))
  result <- rep(NA_integer_, length(target))
  available_rows <- seq_along(target)
  available_cols <- seq_along(source)
  while (length(available_rows)) {
    local_cost <- cost[available_rows, available_cols, drop = FALSE]
    position <- arrayInd(which.min(local_cost), dim(local_cost))[1L, ]
    row <- available_rows[position[1L]]
    col <- available_cols[position[2L]]
    result[row] <- col
    available_rows <- setdiff(available_rows, row)
    available_cols <- setdiff(available_cols, col)
  }
  result
}

.eigen_wald_masks <- function(stage, Y, critical, nsteps, variance_ridge) {
  residual_df <- max(length(Y) - nrow(Y)^2, 1L)
  sigma <- sqrt(sum((Y - stage$X_hat)^2) / residual_df)
  variance <- TheoVar(
    sigma = max(sigma, sqrt(.Machine$double.eps)),
    X0 = stage$x0_hat,
    A = stage$A_hat,
    tt = stage$relative_time,
    nsteps = nsteps,
    L2 = variance_ridge
  )
  pars <- stage$modal_parameters
  target <- c(complex(real = pars$a, imaginary = pars$b), as.complex(pars$cc))
  matched <- .greedy_eigen_match(target, variance$eigenvalues)
  se <- sqrt(pmax(variance$re.eigs.sigma2[matched], 0))
  estimates <- c(pars$a, pars$cc)
  z <- abs(estimates) / se
  keep <- ifelse(is.finite(z), z > critical, TRUE)
  q <- length(pars$a)
  list(
    a_mask = as.numeric(utils::head(keep, q)),
    cc_mask = as.numeric(utils::tail(keep, length(pars$cc))),
    table = data.frame(
      type = c(rep("complex_pair_real_part", q), rep("real_eigenvalue", length(pars$cc))),
      estimate = estimates,
      std_error = se,
      z = z,
      retained = keep,
      matched_eigenvalue = variance$eigenvalues[matched]
    ),
    sigma = sigma,
    variance = variance
  )
}
