#' Fit AdaStableNet: Adaptive ODE System Fitting Without FPC Basis
#'
#' This function performs a two-stage estimation of the dynamic matrix \( A \)
#' followed by adaptive eigenvalue refinement using `AdaEigen_ODE`, without relying on
#' FPCA basis representations. The model is fit directly to the (smoothed) data.
#'
#' @param Y Matrix of observed data (timepoints × variables)
#' @param tt Vector of timepoints
#' @param nbasis Number of B-spline basis functions (default = 100)
#' @param lambda_range Exponent range for lambda grid (default = c(-20, 10))
#' @param method Method used in `.est()` (e.g., "two.stage", "pda")
#' @param twoSE Logical, if TRUE uses 2SE rule in GCV selection
#' @param fit_ode2fd If TRUE, fit model to smoothed data; else fit to raw data
#' @param complex_pairs If TRUE, augment real eigenvalues into complex pairs
#' @param eigen_real_wald If TRUE, apply Wald thresholding to real eigenvalues
#' @param wald_critical Wald threshold cutoff (default = 2)
#' @param eigen_bound If TRUE, cap real eigenvalues to ±1
#' @param lr Learning rate for AdaEigen_ODE
#' @param num_iter Number of optimization iterations
#' @param tol Tolerance for convergence
#' @param ridge.pen Ridge penalty for matrix inversion
#' @param verbose Logical, if TRUE print progress messages
#'
#' @return A list with components:
#'   \item{Ode2Stage}{Two-stage estimation output (including Ahat, Yhat_fd)}
#'   \item{AdaEigenStableNet}{Adaptive eigenvalue refinement output}
#'
#' @importFrom fda smooth.basis eval.fd deriv.fd
#' @importFrom stats quantile
#' @importFrom torch torch_tensor
#'
#' @export
FitAdaStableNet <- function(Y, tt, nbasis = 100, lambda_range = c(-20, 10),
                            method = "two.stage", twoSE = TRUE,
                            fit_ode2fd = TRUE, complex_pairs = TRUE,
                            eigen_real_wald = TRUE, wald_critical = 2,
                            eigen_bound = TRUE, lr = 0.001, num_iter = 5000,
                            tol = 1e-3, ridge.pen = 0.01, verbose = TRUE) {

  if (verbose) cat("Starting two-stage dynamic estimation...\n")
  ode2stage_results <- ode2stage(Y = Y, tt = tt, nbasis = nbasis,
                                 lambda_range = lambda_range,
                                 est.pen = ridge.pen,
                                 method = method, twoSE = twoSE)

  initial_a <- ode2stage_results$re.hat
  initial_b <- ode2stage_results$im.hat
  initial_cc <- ode2stage_results$real.hat
  if (!is.null(initial_cc)) {
    initial_cc <- ifelse(abs(initial_cc) > 1, sign(initial_cc), initial_cc)
  }

  nreal <- length(initial_cc)

  if (verbose) cat("Initial eigenvalues extracted.\n")

  if (nreal > 1 && complex_pairs) {
    add_pairs <- floor(nreal / 2)
    add_re <- sort(initial_a)[1:add_pairs]
    add_im <- min(initial_b) / 2^(1:add_pairs)
    initial_a <- c(initial_a, add_re)
    initial_b <- c(initial_b, add_im)
    if ((nreal %% 2) == 0) {
      initial_cc <- NULL
    } else {
      initial_cc <- initial_cc[add_pairs + 1]
    }
  }

  if (fit_ode2fd) {
    if (verbose) cat("Fitting to smoothed data...\n")
    Y_input <- t(ode2stage_results$Yhat_fd)
  } else {
    if (verbose) cat("Fitting to raw data...\n")
    Y_input <- t(Y)
  }

  if (verbose) cat("Starting adaptive ODE eigenvalue fitting...\n")
  AdaEigenStableNet_results <- AdaStableNet(
    Y = Y_input, tt = tt,
    initial_a = initial_a, initial_b = initial_b,
    initial_cc = initial_cc,
    eigen_real_wald = eigen_real_wald,
    wald_critical = wald_critical,
    eigen_bound = eigen_bound,
    lr = lr, num_iter = num_iter, tol = tol,
    ridge.pen = ridge.pen, verbose = verbose
  )

  if (verbose) cat("AdaStableNet model fitting completed.\n")

  return(list(
    Ode2Stage = ode2stage_results,
    AdaEigenStableNet = AdaEigenStableNet_results
  ))
}
