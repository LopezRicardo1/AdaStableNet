#' Estimate ODE Coefficient Matrix
#'
#' Internal helper function to estimate the coefficient matrix A in the system dX/dt = AX.
#' Supports two-stage least squares and penalized differential analysis (PDA) estimation.
#'
#' @param Ts Numeric vector of time points.
#' @param Xt An object of class \code{fd} representing smoothed functional data.
#' @param est.pen Numeric penalty for ridge regularization.
#' @param method Estimation method. One of \code{"two.stage0"}, \code{"two.stage"}, \code{"pda0"}, or \code{"pda"}.
#'
#' @return A matrix representing the estimated ODE coefficient matrix.
#'
#' @keywords internal
#'
#' @importFrom fda eval.fd deriv.fd inprod
.est <- function(Ts, Xt, est.pen = 0.1^4, method = "pda") {
  if (method == "two.stage0") {
    X <- eval.fd(Ts, Xt)
    X.deriv <- eval.fd(Ts, deriv.fd(Xt))
    Ahat <- t(X.deriv) %*% X %*% solve(t(X) %*% X)
  } else if (method == "two.stage") {
    X <- eval.fd(Ts, Xt)
    X.deriv <- eval.fd(Ts, deriv.fd(Xt))
    Ahat <- t(X.deriv) %*% X %*% ridge.inv(t(X) %*% X, lambda.prop = est.pen)
  } else if (method == "pda0") {
    Sigma.xx <- inprod(Xt, Xt)
    Sigma.xderivx <- inprod(deriv.fd(Xt), Xt)
    Ahat <- Sigma.xderivx %*% solve(Sigma.xx)
  } else if (method == "pda") {
    Sigma.xx <- inprod(Xt, Xt)
    Sigma.xderivx <- inprod(deriv.fd(Xt), Xt)
    Ahat <- Sigma.xderivx %*% ridge.inv(Sigma.xx, lambda.prop = est.pen)
  } else {
    stop(paste("Method", method, "is not implemented in function .est()!"))
  }
  return(Ahat)
}
