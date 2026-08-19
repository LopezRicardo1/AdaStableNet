#' Estimate a Linear ODE Matrix from Functional Data
#'
#' Internal gradient-matching estimator used to initialize AdaStableNet.
#'
#' @param Ts Numeric time vector.
#' @param Xt An `fda::fd` object.
#' @param est.pen Nonnegative ridge multiplier.
#' @param method One of `"two.stage0"`, `"two.stage"`, `"pda0"`, or
#'   `"pda"`.
#'
#' @return A numeric system matrix.
#' @keywords internal
#' @importFrom fda deriv.fd eval.fd inprod
.est <- function(Ts, Xt, est.pen = 1e-3, method = "pda") {
  method <- match.arg(method, c("two.stage0", "two.stage", "pda0", "pda"))
  if (method %in% c("two.stage0", "two.stage")) {
    X <- eval.fd(Ts, Xt)
    X_deriv <- eval.fd(Ts, deriv.fd(Xt))
    cross_x <- crossprod(X)
    cross_dx_x <- crossprod(X_deriv, X)
  } else {
    cross_x <- inprod(Xt, Xt)
    cross_dx_x <- inprod(deriv.fd(Xt), Xt)
  }
  if (method %in% c("two.stage0", "pda0")) {
    cross_dx_x %*% .safe_pinv(cross_x)
  } else {
    cross_dx_x %*% ridge.inv(cross_x, lambda.prop = est.pen)
  }
}

#' Two-Stage Initialization for a Linear ODE
#'
#' Smooths each state with a common B-spline penalty, estimates derivatives,
#' and obtains an initial dynamic matrix by gradient matching.
#'
#' @param Y Numeric matrix with time points in rows and states in columns.
#' @param tt Strictly increasing observation times.
#' @param nbasis Number of B-spline basis functions.
#' @param lambda_range Length-two range for the natural logarithm of the
#'   smoothing penalty.
#' @param est.pen Nonnegative ridge multiplier for gradient matching.
#' @param method Gradient-matching method.
#' @param twoSE If `TRUE`, choose the largest smoothing penalty whose mean GCV
#'   is within two standard errors of the minimum. Otherwise choose minimum GCV.
#' @param nlambda Number of smoothing penalties in the grid.
#'
#' @return A list containing the functional-data fit, smoothed values,
#'   initialized system matrix and modal eigenvalues, and GCV diagnostics.
#' @importFrom fda create.bspline.basis eval.fd fdPar smooth.basis
#' @export
ode2stage <- function(Y, tt, nbasis = 25, lambda_range = c(-16, 4),
                      est.pen = 1e-3, method = "two.stage", twoSE = TRUE,
                      nlambda = 30L) {
  tt <- .validate_time(tt)
  Y <- .validate_data(Y, tt, "time_by_state")
  .validate_scalar(nbasis, "nbasis", lower = 4, integer = TRUE)
  .validate_scalar(est.pen, "est.pen", lower = 0)
  .validate_scalar(nlambda, "nlambda", lower = 3, integer = TRUE)
  if (!is.numeric(lambda_range) || length(lambda_range) != 2L ||
      any(!is.finite(lambda_range)) || lambda_range[1] >= lambda_range[2]) {
    stop("`lambda_range` must be two increasing finite values.", call. = FALSE)
  }
  method <- match.arg(method, c("two.stage", "two.stage0", "pda", "pda0"))
  if (!is.logical(twoSE) || length(twoSE) != 1L || is.na(twoSE)) {
    stop("`twoSE` must be TRUE or FALSE.", call. = FALSE)
  }

  effective_nbasis <- min(as.integer(nbasis), max(4L, length(tt) - 2L))
  basis <- create.bspline.basis(range(tt), nbasis = effective_nbasis)
  log_lambda <- seq(lambda_range[1], lambda_range[2], length.out = nlambda)
  lambda_grid <- exp(log_lambda)
  fits <- vector("list", length(lambda_grid))
  gcv_by_state <- matrix(NA_real_, nrow = length(lambda_grid), ncol = ncol(Y))

  for (i in seq_along(lambda_grid)) {
    fd_par <- fdPar(basis, Lfdobj = 2, lambda = lambda_grid[i])
    fits[[i]] <- smooth.basis(tt, Y, fd_par)
    gcv_by_state[i, ] <- as.numeric(fits[[i]]$gcv)
  }
  gcv_score <- rowMeans(gcv_by_state, na.rm = TRUE)
  if (all(!is.finite(gcv_score))) {
    stop("All smoothing fits produced non-finite GCV values.", call. = FALSE)
  }
  idx_min <- which.min(gcv_score)
  if (twoSE) {
    state_scores <- gcv_by_state[idx_min, ]
    state_scores <- state_scores[is.finite(state_scores)]
    gcv_se <- if (length(state_scores) > 1L) {
      stats::sd(state_scores) / sqrt(length(state_scores))
    } else {
      0
    }
    candidates <- which(gcv_score <= gcv_score[idx_min] + 2 * gcv_se)
    idx_selected <- max(candidates)
    selection_method <- "2-SE rule"
  } else {
    gcv_se <- NA_real_
    idx_selected <- idx_min
    selection_method <- "Minimum GCV"
  }

  fdlist <- fits[[idx_selected]]
  Yfd <- fdlist$fd
  Yhat_fd <- eval.fd(tt, Yfd)
  Ahat <- .est(Ts = tt, Xt = Yfd, est.pen = est.pen, method = method)
  modes <- .classify_eigenvalues(Ahat)

  structure(list(
    Bsplines = fdlist,
    Yhat_fd = Yhat_fd,
    x0_hat = as.numeric(Yhat_fd[1L, ]),
    z.complex = modes$values,
    re.hat = modes$re,
    im.hat = modes$im,
    real.hat = if (length(modes$real)) modes$real else NULL,
    Ahat = Ahat,
    effective_df = fdlist$df,
    gcv = gcv_score,
    gcv_by_state = gcv_by_state,
    gcv_se_at_min = gcv_se,
    lambda_grid = lambda_grid,
    selected_lambda = lambda_grid[idx_selected],
    selected_gcv = gcv_score[idx_selected],
    selection_method = selection_method,
    eigen_tolerance = modes$tolerance,
    nbasis = effective_nbasis
  ), class = "adastablenet_initial")
}
