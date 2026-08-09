#' Ridge-Regularized Inverse of a Symmetric Matrix
#'
#' Computes a numerically stable inverse of a symmetric matrix after adding a
#' scale-adjusted ridge term. This function is mainly exposed for compatibility;
#' package users normally do not need to call it directly.
#'
#' @param SymMat A finite symmetric square matrix.
#' @param lambda.prop Nonnegative ridge multiplier relative to the average
#'   absolute eigenvalue.
#'
#' @return A symmetric numeric matrix.
#' @export
ridge.inv <- function(SymMat, lambda.prop = 1e-4) {
  if (!is.matrix(SymMat) || !is.numeric(SymMat) || nrow(SymMat) != ncol(SymMat)) {
    stop("`SymMat` must be a numeric square matrix.", call. = FALSE)
  }
  if (any(!is.finite(SymMat))) {
    stop("`SymMat` must contain only finite values.", call. = FALSE)
  }
  .validate_scalar(lambda.prop, "lambda.prop", lower = 0)
  SymMat <- (SymMat + t(SymMat)) / 2
  ee <- eigen(SymMat, symmetric = TRUE)
  scale <- max(mean(abs(ee$values)), .Machine$double.eps)
  ridge <- lambda.prop * scale
  floor_value <- max(ridge, .Machine$double.eps * scale)
  inv_values <- 1 / pmax(ee$values + ridge, floor_value)
  tcrossprod(sweep(ee$vectors, 2L, inv_values, `*`), ee$vectors)
}

.safe_pinv <- function(x, tol = sqrt(.Machine$double.eps)) {
  if (!is.matrix(x)) {
    x <- as.matrix(x)
  }
  sx <- svd(x)
  if (!length(sx$d)) {
    return(matrix(0, ncol(x), nrow(x)))
  }
  cutoff <- tol * max(dim(x)) * max(sx$d)
  keep <- sx$d > cutoff
  if (!any(keep)) {
    return(matrix(0, ncol(x), nrow(x)))
  }
  sx$v[, keep, drop = FALSE] %*%
    (t(sx$u[, keep, drop = FALSE]) / sx$d[keep])
}

.profile_projection <- function(Y, S, ridge.pen) {
  gram <- tcrossprod(S)
  gram_inv <- ridge.inv(gram, lambda.prop = ridge.pen)
  P <- tcrossprod(Y, S) %*% gram_inv
  X_hat <- P %*% S
  list(
    P = P,
    X_hat = X_hat,
    loss = mean((Y - X_hat)^2),
    gram = gram,
    gram_condition = .safe_kappa(gram)
  )
}

#' @importFrom expm expm
.matrix_trajectory <- function(A, tt, x0, origin = 0) {
  tt <- as.numeric(tt)
  x0 <- as.numeric(x0)
  t(vapply(
    tt,
    function(ti) as.numeric(expm((ti - origin) * A) %*% x0),
    numeric(length(x0))
  ))
}
