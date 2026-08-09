sort_complex <- function(z) {
  z[order(Re(z), Im(z))]
}

test_that("real modal blocks and basis obey the ODE", {
  a <- -0.2
  b <- 2.5
  cc <- -0.1
  tt <- seq(0, 1, length.out = 101)
  J <- Jordan(a, b, cc)
  S <- ode_basis(tt, a, b, cc)

  expect_equal(dim(J), c(3, 3))
  expect_equal(dim(S), c(3, length(tt)))
  expect_equal(sort_complex(eigen(J)$values),
               sort_complex(c(a + 1i * b, a - 1i * b, cc)),
               tolerance = 1e-10)

  middle <- 2:(length(tt) - 1)
  derivative <- (S[, middle + 1] - S[, middle - 1]) /
    rep(tt[middle + 1] - tt[middle - 1], each = nrow(S))
  expect_equal(derivative, J %*% S[, middle], tolerance = 2e-3)
})

test_that("modal input errors are informative", {
  expect_error(Jordan(1, c(1, 2)), "equal lengths")
  expect_error(ode_basis(1:3, 1, -1), "nonnegative")
  expect_error(ode_basis(1:3, numeric(), numeric(), NULL), "one mode")
})
