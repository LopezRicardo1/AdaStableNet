#' Construct Jordan Matrix from Complex and Real Eigenvalues
#'
#' Builds a block-diagonal matrix with 2×2 blocks for complex conjugate pairs
#' and 1×1 entries for real eigenvalues.
#'
#' @param re Vector of real parts of complex eigenvalues
#' @param im Vector of imaginary parts of complex eigenvalues
#' @param real Optional vector of real eigenvalues
#'
#' @return A torch tensor representing the full Jordan matrix
#' @importFrom torch torch_stack torch_reshape torch_block_diag torch_tensor
#' @export
Jordan <- function(re, im, real = NULL) {
  # Create a single 2×2 Jordan block for complex conjugate pair
  JordanBlock <- function(re, im) {
    torch_tensor(matrix(c(re, im, -im, re), nrow = 2, byrow = TRUE))
  }

  # Initialize list of blocks
  blocks <- list()

  # Add complex 2x2 blocks
  for (i in seq_along(re)) {
    blocks[[length(blocks) + 1]] <- JordanBlock(re[i], im[i])
  }

  # Add real eigenvalues as 1x1 blocks
  if (!is.null(real)) {
    for (j in seq_along(real)) {
      blocks[[length(blocks) + 1]] <- torch_tensor(matrix(real[j], nrow = 1))
    }
  }

  # Combine all blocks into a block-diagonal matrix
  torch_block_diag(blocks)
}
