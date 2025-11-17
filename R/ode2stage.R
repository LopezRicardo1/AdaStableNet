#' Two-Stage Estimation of Dynamic Matrix A from Functional Data
#'
#' Estimates the system matrix \( A \) from observed time series \( Y(t) \)
#' using smoothing splines and optional two-stage or PDA regression.
#'
#' @param Y Matrix of observed values (timepoints × variables)
#' @param tt Vector of observation time points
#' @param nbasis Number of B-spline basis functions (default = 100)
#' @param lambda_range Exponent range for lambda grid (default = c(-20, 20))
#' @param est.pen Penalty parameter for ridge inversion in estimation
#' @param method Estimation method for A ("two.stage", "two.stage0", "pda", or "pda0")
#' @param twoSE If TRUE, uses 2SE rule for GCV selection (default = TRUE)
#'
#' @importFrom fda create.bspline.basis fdPar smooth.basis eval.fd
#' @importFrom stats sd
#' @export
#' @return A list containing estimated A, eigenvalues, smoothed Y, B-spline fit object, and GCV metadata
ode2stage <- function(Y, tt, nbasis = 100, lambda_range = c(-20, 20),
                      est.pen = 1e-3, method = "two.stage", twoSE = TRUE) {

  if (length(tt) <= 50) {
    mybasis = create.bspline.basis(range(tt), breaks = tt)
  } else {
    mybasis = create.bspline.basis(range(tt), nbasis = nbasis)
  }

  llams = seq(lambda_range[1], lambda_range[2], length.out = 50)
  gcvs = numeric(length(llams))
  fd_par_list = vector("list", length(llams))

  for (i in seq_along(llams)) {
    lambda_i = exp(llams[i])
    mypar = fdPar(mybasis, Lfdobj = 2, lambda = lambda_i)
    fdlist = smooth.basis(tt, Y, mypar)
    gcvs[i] = sum(fdlist$gcv)
    fd_par_list[[i]] = fdlist
  }

  gcv_se = 2 * sd(gcvs) / sqrt(length(gcvs))
  idx_min = which.min(gcvs)

  if (idx_min == 1) {
    twoSE = FALSE
    message("twoSE is set to FALSE since min GCV lambda index = 1")
  }

  selected_gcv = gcvs[idx_min]

  if (twoSE) {
    candidate_indices = which(gcvs <= (selected_gcv + gcv_se))
    valid_indices = candidate_indices[candidate_indices < idx_min]
    idx_1se = valid_indices[which.max(gcvs[valid_indices])]
    selected_gcv_1se = gcvs[idx_1se]
    lambda_star = exp(llams[idx_1se])
    fdlist = fd_par_list[[idx_1se]]
  } else {
    lambda_star = exp(llams[idx_min])
    selected_gcv_1se = selected_gcv
    fdlist = fd_par_list[[idx_min]]
  }

  Yfd = fdlist$fd
  Yhat_fd = eval.fd(tt, Yfd)
  Ahat <- .est(Ts = tt, Xt = Yfd, est.pen = est.pen, method = method)
  z = eigen(Ahat)$values

  re = Re(z)[Im(z) > 0]
  im = Im(z)[Im(z) > 0]
  real = Re(z)[Im(z) == 0]
  if (length(real) == 0) real = NULL

  return(list(
    Bsplines = fdlist,
    Yhat_fd = Yhat_fd,
    z.complex = z,
    re.hat = re,
    im.hat = im,
    real.hat = real,
    Ahat = Ahat,
    effective_df = fdlist$df,
    gcv = gcvs,
    lambda_grid = exp(llams),
    selected_lambda = lambda_star,
    selected_gcv = selected_gcv_1se,
    selection_method = ifelse(twoSE, "2SE Rule", "Minimum GCV")
  ))
}
