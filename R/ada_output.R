#' Full-Rank ODE Solver via Exponential Basis Projection
#'
#' Estimates the coefficient matrix A for the dynamic system dX/dt = AX,
#' using a projected basis representation \( X = PS \) with basis matrix \( S \) constructed
#' from damped exponential and sinusoidal terms parameterized by eigenvalues.
#'
#' @param a Numeric vector of real parts of complex eigenvalues.
#' @param b Numeric vector of imaginary parts of complex eigenvalues.
#' @param cc Numeric vector of real eigenvalues.
#' @param tt Numeric time vector (as torch tensor).
#' @param Y Tensor of observed data (states x time).
#' @param ridge.pen Scalar ridge penalty for basis inversion stability.
#' @param a_wald Optional mask tensor for \code{a}; defaults to vector of 1s.
#' @param cc_wald Optional mask tensor for \code{cc}; defaults to vector of 1s.
#'
#' @return A list containing:
#' \describe{
#'   \item{\code{A_hat}}{Estimated system matrix \( A \) (as base R matrix).}
#'   \item{\code{x0_hat}}{Estimated initial condition vector \( x_0 \).}
#'   \item{\code{X_hat}}{Fitted latent trajectory matrix \( X = PS \).}
#'   \item{\code{P}}{Estimated projection matrix from basis \( S \) to data \( Y \).}
#'   \item{\code{ODE_Basis}}{Evaluated basis matrix \( S \) (states x time).}
#' }
#'
#' @importFrom torch torch_mm torch_maximum torch_pinverse linalg_eig
#' @export
ada_output <- function(a, b, cc, tt, Y, ridge.pen, a_wald = NULL, cc_wald = NULL) {
  if (dim(Y)[2] != dim(tt)[1]) {
    Y <- Y$t()
  }
  # Step 1: Jordan matrix from eigenvalues
  J <- Jordan(a, abs(b), cc)

  # Step 2: ODE basis S
  S <- ode_basis(tt, a, b, cc, a_wald, cc_wald)

  # Step 3: Compute SS and regularized inverse
  SS <- torch_mm(S, S$t())
  SS_eig <- linalg_eig(SS)
  values <- SS_eig[[1]]$real
  vectors <- SS_eig[[2]]$real
  SS_inv <- vectors$mm((1 / torch_maximum(values, ridge.pen * values$sum()))$diag())$mm(vectors$t())

  # Step 4: Estimate P (projection matrix)
  P <- Y$mm(S$t())$mm(SS_inv)

  # Step 5: Construct fitted latent state X_hat = P S
  X_hat <- P$mm(S)

  # Step 6: A_hat = P J P^+
  A_hat <- P$mm(torch_mm(J, torch_pinverse(P)))

  # Return full-rank outputs
  list(
    A_hat = as.matrix(A_hat),
    x0_hat = as.numeric(P[,1]),
    X_hat = as.matrix(X_hat),
    P = as.matrix(P),
    ODE_Basis = as.matrix(S)
  )
}
