#' Ridge-Inverse of a Symmetric Matrix
#'
#' Computes the stabilized inverse of a symmetric matrix using ridge regularization.
#' Used in PCODE and AdaStableNet for basis inversion.
#'
#' @param SymMat A symmetric square matrix
#' @param lambda.prop Proportion of trace used as ridge penalty (default = 0.0001)
#'
#' @return A regularized inverse of SymMat
#' @export
ridge.inv <- function(SymMat, lambda.prop = 0.1^4) {
  ee <- eigen(SymMat)
  LL <- ee[["values"]]
  TT <- ee[["vectors"]]
  lambda <- sum(LL) * lambda.prop
  inv.mat <- TT %*% diag(1 / pmax(LL, lambda)) %*% t(TT)
  return(inv.mat)
}
