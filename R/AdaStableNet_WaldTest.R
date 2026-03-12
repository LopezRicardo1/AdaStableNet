#' AdaStableNet_WaldTest
#'
#' Perform Wald hypothesis testing on each A[i,j] coefficient from an ODE system
#' estimated by AdaStableNet. Computes z-scores, p-values, and applies FDR/FWER
#' multiple testing correction.
#'
#' @param fit     Output list from AdaStableNet()$Unbounded, $Wald_Real, or $Eigen_Bound
#' @param Y       Observed data matrix (states × time)
#' @param tt      Time vector
#' @param method  Multiple testing correction method ("BH", "Holm", etc.)
#' @param alpha   Significance level (default = 0.05)
#' @param return  Type of result: "zscores", "pvals", or "A_thresh"
#'
#' @return Matrix of z-scores, adjusted p-values, or thresholded A matrix
#'
#' @export
AdaStableNet_WaldTest <- function(fit,
                                  Y,
                                  tt,
                                  method = "BH",
                                  alpha = 0.05,
                                  return = c("zscores", "pvals", "A_thresh")) {

  return <- match.arg(return)

  # Extract model outputs
  Ahat  <- fit$A_hat
  Xhat  <- fit$X_hat
  x0hat <- fit$x0_hat

  # Residual standard deviation
  RES   <- as.numeric(Y) - as.numeric(Xhat)
  sigma <- sqrt(mean(RES^2))

  # Theoretical variance and standard error
  A.var <- TheoVar(sigma, x0hat, Ahat, tt)
  A.sd  <- sqrt(A.var$varmat)

  # Wald z-scores
  zmat <- Ahat / A.sd

  # Two-sided p-values
  p.mat <- 2 * pnorm(abs(zmat), lower.tail = FALSE)

  # Multiple testing correction
  padj.vec  <- p.adjust(as.vector(p.mat), method = method)
  p.adj.mat <- matrix(padj.vec, nrow = nrow(p.mat), ncol = ncol(p.mat))

  # Thresholded A matrix
  A.thresh <- Ahat
  A.thresh[p.adj.mat > alpha] <- 0

  # Return selected output
  switch(return,
         zscores  = zmat,
         pvals    = p.adj.mat,
         A_thresh = A.thresh)
}
