test_that("stability diagnostics distinguish spectral and transient behavior", {
  A <- matrix(c(-1, 8, 0, -2), 2, 2, byrow = TRUE)
  stage <- structure(list(A_hat = A), class = "adastablenet_stage")
  result <- stability_diagnostics(stage, horizon = 2, n_grid = 21)

  expect_true(result$spectrally_stable)
  expect_false(result$euclidean_dissipative)
  expect_gt(result$transient$maximum, 1)
  expect_equal(length(result$transient$time), 21)
  expect_equal(result$tolerance, sqrt(.Machine$double.eps))

  marginal <- stability_diagnostics(
    structure(list(A_hat = diag(c(1e-15, -0.1))),
              class = "adastablenet_stage"),
    horizon = 0, n_grid = 2
  )
  expect_true(marginal$spectrally_stable)
})

test_that("ode.ident diagnostics expose ICIS when available", {
  skip_if_not_installed("ode.ident")
  A <- diag(c(-1, -2))
  result <- identifiability_diagnostics(A, c(1, 1), n.digits = 6)

  expect_true(result$identifiable)
  expect_true(result$distinct_eigenvalues)
  expect_true(result$all_invariant_subspaces_excited)
  expect_gt(result$ICIS, 0)
  expect_equal(length(result$invariant_subspace_excitation[[1L]]), 2)
})
