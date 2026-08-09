#' Construct a Real Modal Matrix
#'
#' Constructs the real block-diagonal modal matrix associated with complex
#' conjugate eigenvalue pairs and real eigenvalues. Despite the historical
#' function name, this represents diagonalizable modes and does not create
#' nontrivial Jordan chains.
#'
#' @param re Real parts of complex-conjugate eigenvalue pairs.
#' @param im Positive imaginary parts of complex-conjugate eigenvalue pairs.
#' @param real Optional real eigenvalues.
#'
#' @return A numeric block-diagonal matrix.
#' @export
Jordan <- function(re, im, real = NULL) {
  re <- as.numeric(re)
  im <- as.numeric(im)
  real <- if (is.null(real)) numeric() else as.numeric(real)
  if (length(re) != length(im)) {
    stop("`re` and `im` must have equal lengths.", call. = FALSE)
  }
  if (!length(re) && !length(real)) {
    stop("At least one modal eigenvalue is required.", call. = FALSE)
  }
  if (any(!is.finite(c(re, im, real)))) {
    stop("Modal eigenvalues must be finite.", call. = FALSE)
  }
  if (any(im < 0)) {
    stop("`im` must contain nonnegative imaginary parts.", call. = FALSE)
  }

  blocks <- vector("list", length(re) + length(real))
  if (length(re)) {
    for (i in seq_along(re)) {
      blocks[[i]] <- matrix(c(re[i], -im[i], im[i], re[i]),
                            nrow = 2L, byrow = FALSE)
    }
  }
  if (length(real)) {
    offset <- length(re)
    for (i in seq_along(real)) {
      blocks[[offset + i]] <- matrix(real[i], nrow = 1L)
    }
  }
  sizes <- vapply(blocks, nrow, integer(1))
  ans <- matrix(0, sum(sizes), sum(sizes))
  starts <- cumsum(c(1L, utils::head(sizes, -1L)))
  for (i in seq_along(blocks)) {
    idx <- starts[i]:(starts[i] + sizes[i] - 1L)
    ans[idx, idx] <- blocks[[i]]
  }
  ans
}

#' Construct an Exponential-Sinusoidal ODE Basis
#'
#' Builds a real modal basis for a diagonalizable linear ODE. Each complex pair
#' contributes sine and cosine rows; each real eigenvalue contributes one
#' exponential row.
#'
#' @param tt Numeric observation times.
#' @param a Real parts of complex eigenvalue pairs.
#' @param b Nonnegative imaginary parts of complex eigenvalue pairs.
#' @param cc Optional real eigenvalues.
#' @param a_wald Optional zero-one mask for `a`.
#' @param cc_wald Optional zero-one mask for `cc`.
#'
#' @return A numeric matrix with modes in rows and times in columns.
#' @examples
#' S <- ode_basis(seq(0, 1, length.out = 11), -0.2, 2 * pi, -0.1)
#' dim(S)
#' @export
ode_basis <- function(tt, a, b, cc = NULL, a_wald = NULL, cc_wald = NULL) {
  tt <- as.numeric(tt)
  a <- as.numeric(a)
  b <- as.numeric(b)
  cc <- if (is.null(cc)) numeric() else as.numeric(cc)
  if (length(a) != length(b)) {
    stop("`a` and `b` must have equal lengths.", call. = FALSE)
  }
  if (any(!is.finite(c(tt, a, b, cc))) || any(b < 0)) {
    stop("Times and modal parameters must be finite; `b` must be nonnegative.",
         call. = FALSE)
  }
  if (!length(a) && !length(cc)) {
    stop("At least one mode is required.", call. = FALSE)
  }
  a_wald <- a_wald %||% rep(1, length(a))
  cc_wald <- cc_wald %||% rep(1, length(cc))
  if (length(a_wald) != length(a) || length(cc_wald) != length(cc)) {
    stop("Wald masks must match their modal parameter vectors.", call. = FALSE)
  }
  a <- a * as.numeric(a_wald)
  cc <- cc * as.numeric(cc_wald)

  S <- matrix(NA_real_, nrow = 2L * length(a) + length(cc), ncol = length(tt))
  row <- 1L
  if (length(a)) {
    for (i in seq_along(a)) {
      envelope <- exp(a[i] * tt)
      S[row, ] <- envelope * sin(b[i] * tt)
      S[row + 1L, ] <- envelope * cos(b[i] * tt)
      row <- row + 2L
    }
  }
  if (length(cc)) {
    for (i in seq_along(cc)) {
      S[row, ] <- exp(cc[i] * tt)
      row <- row + 1L
    }
  }
  S
}

.modal_eigenvalues <- function(a, b, cc = NULL) {
  cc <- if (is.null(cc)) numeric() else cc
  c(complex(real = a, imaginary = b),
    complex(real = a, imaginary = -b),
    as.complex(cc))
}

.classify_eigenvalues <- function(A, tol = NULL) {
  z <- eigen(A, only.values = TRUE)$values
  tol <- tol %||% (sqrt(.Machine$double.eps) * max(1, max(Mod(z))))
  positive <- which(Im(z) > tol)
  real_idx <- which(abs(Im(z)) <= tol)
  if (length(positive)) {
    positive <- positive[order(Im(z[positive]), Re(z[positive]))]
  }
  real_values <- if (length(real_idx)) sort(Re(z[real_idx])) else numeric()
  list(
    re = Re(z[positive]),
    im = Im(z[positive]),
    real = real_values,
    values = z,
    tolerance = tol
  )
}
