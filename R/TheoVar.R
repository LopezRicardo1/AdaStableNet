#' Theoretical Variance Estimator for Linear ODE System
#'
#' Computes the Fisher information and theoretical variance of the real parts of eigenvalues
#' from the ODE system dX/dt = AX based on sensitivity analysis and ridge-stabilized inversion.
#'
#' @param sigma Noise standard deviation
#' @param X0 Initial condition vector
#' @param A System matrix (p × p)
#' @param tt Vector of timepoints
#' @param nsteps Number of discretization steps for integral approximation (default = 20)
#' @param L2 Ridge penalty for Fisher information inversion (default = 1e-6)
#'
#' @return A list with:
#' \item{Dt}{3D array of sensitivity matrices}
#' \item{FisherI}{Fisher information matrix}
#' \item{varmat}{Diagonal of the inverse Fisher information matrix (p × p)}
#' \item{re.eigs.sigma2}{Estimated variance of real parts of eigenvalues}
#'
#' @importFrom Matrix diag
#' @importFrom expm expm
#' @export
TheoVar <- function(sigma, X0, A, tt, nsteps = 20, L2 = 1e-6) {
  p <- nrow(A)
  J <- length(tt)

  Xt <- t(sapply(tt, function(tj) expm(tj * A) %*% X0))

  Dt <- array(0, c(J, p, p^2))
  for (j in 1:J) {
    tj <- tt[j]
    Dtj <- matrix(0, p, p^2)
    for (ns in 1:nsteps) {
      s <- ns / nsteps
      Xts <- expm(tj * s * A) %*% X0
      eAts <- expm(tj * (1 - s) * A)
      Dtj <- Dtj + t(Xts) %x% eAts
    }
    Dtj <- Dtj * tj / nsteps
    Dt[j, , ] <- as.matrix(Dtj)
  }

  FisherI <- matrix(rowSums(sapply(1:J, function(j) t(Dt[j, , ]) %*% Dt[j, , ])), p^2, p^2) / sigma^2

  ee <- eigen(FisherI)
  FisherI.inv <- ee$vectors %*% diag(1 / pmax(ee$values, L2 * sum(ee$values))) %*% t(ee$vectors)
  varmat <- matrix(diag(FisherI.inv), p, p)

  ee <- eigen(A)
  lambdas <- ee$values
  U <- ee$vectors
  Vstar <- solve(U)

  re.eigs.sigma2 <- rep(0, p)
  for (k in 1:p) {
    u.k <- U[, k]
    vstar.k <- Vstar[k, , drop = FALSE]
    uxv <- u.k %x% t(vstar.k)
    ck <- Re(uxv)
    re.eigs.sigma2[k] <- drop(t(ck) %*% FisherI.inv %*% ck)
  }

  return(list(
    Dt = Dt,
    FisherI = FisherI,
    varmat = varmat,
    re.eigs.sigma2 = re.eigs.sigma2
  ))
}
