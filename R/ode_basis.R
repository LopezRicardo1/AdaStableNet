#' Construct Exponential-Sinusoidal ODE Basis
#'
#' Generates a basis matrix composed of damped exponential and sinusoidal
#' components, corresponding to complex and real eigenvalue modes of
#' the system matrix  A  in dX/dt = AX
#'
#' @param tt Numeric time vector as a torch tensor.
#' @param a Numeric vector of real parts of complex eigenvalues.
#' @param b Numeric vector of imaginary parts of complex eigenvalues.
#' @param cc Numeric vector of real eigenvalues.
#' @param a_wald Optional mask tensor for complex modes (default: ones).
#' @param cc_wald Optional mask tensor for real modes (default: ones).
#'
#' @return A torch tensor containing the stacked basis matrix with rows
#' corresponding to basis functions and columns corresponding to time points.
#'
#' @details
#' The function builds the following basis components:
#' \itemize{
#'   \item For each complex eigenvalue pair \((a_i, b_i)\):
#'     \deqn{e^{a_i t} \sin(b_i t), \quad e^{a_i t} \cos(b_i t)}
#'   \item For each real eigenvalue \(c_i\):
#'     \deqn{e^{c_i t}}
#' }
#'
#' @examples
#' \dontrun{
#' tt <- torch_linspace(0, 1, 100)
#' a <- c(-0.5)
#' b <- c(2)
#' cc <- c(-0.1)
#' S <- ode_basis(tt, a, b, cc)
#' }
#'
#' @importFrom torch torch_exp torch_sin torch_cos torch_stack torch_ones
#' @export
ode_basis <- function(tt, a, b, cc, a_wald = NULL, cc_wald = NULL) {
  # Default to tensors of ones if a_wald or cc_wald are NULL
  if (is.null(a_wald)) a_wald <- torch_ones(length(a))
  if (is.null(cc_wald)) cc_wald <- torch_ones(length(cc))

  num_params <- length(a) + length(cc)
  num_pairs <- length(a)
  X <- list()

  for (i in 1:num_params) {
    if (i <= num_pairs) {
      ai <- a[i] * a_wald[i]
      bi <- b[i]
      x_pred_sin <- torch_exp(ai * tt) * torch_sin(bi * tt)
      x_pred_cos <- torch_exp(ai * tt) * torch_cos(bi * tt)
      X[[2 * i - 1]] <- x_pred_sin
      X[[2 * i]] <- x_pred_cos
    } else {
      cci <- cc[i - num_pairs] * cc_wald[i - num_pairs]
      x_pred <- torch_exp(tt * cci)
      X[[2 * num_pairs + i - num_pairs]] <- x_pred
    }
  }

  torch_stack(X)
}
