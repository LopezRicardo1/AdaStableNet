test_that("low-level modal fitting returns stable full-rank outputs", {
  sim <- simulate_adastablenet(p = 2, n_time = 31, sigma = 0.01, seed = 5)
  modes <- sim$modal_parameters
  fit <- AdaStableNet(
    Y = t(sim$Y), tt = sim$time,
    initial_a = modes$a + 0.02,
    initial_b = modes$b * 0.98,
    initial_cc = modes$cc,
    eigen_real_wald = FALSE,
    num_iter = 80,
    verbose = FALSE
  )

  expect_s3_class(fit$Unbounded, "adastablenet_stage")
  expect_s3_class(fit$Eigen_Bound, "adastablenet_stage")
  expect_equal(dim(fit$Eigen_Bound$A_hat), c(2, 2))
  expect_equal(dim(fit$Eigen_Bound$X_hat), c(2, 31))
  expect_true(fit$Eigen_Bound$diagnostics$full_modal_rank)
  expect_lte(fit$Eigen_Bound$diagnostics$spectral_abscissa, 1e-7)
})

test_that("high-level fit supports standard model methods", {
  sim <- simulate_adastablenet(p = 3, n_time = 25, sigma = 0.02, seed = 10)
  fit <- FitAdaStableNet(
    sim$Y, sim$time, nbasis = 10, lambda_range = c(-12, 0),
    twoSE = FALSE, eigen_real_wald = FALSE,
    num_iter = 60, verbose = FALSE
  )

  expect_s3_class(fit, "adastablenet_fit")
  expect_equal(dim(coef(fit)), c(3, 3))
  expect_equal(dim(fitted(fit)), dim(sim$Y))
  expect_equal(dim(residuals(fit)), dim(sim$Y))
  expect_equal(dim(predict(fit, seq(0, 2, length.out = 8))), c(8, 3))
  expect_s3_class(summary(fit), "summary.adastablenet_fit")
})

test_that("fit validates dimensions and modal completeness", {
  expect_error(
    AdaStableNet(matrix(1, 3, 10), 1:10, 0, 1, NULL,
                 eigen_real_wald = FALSE, verbose = FALSE),
    "exactly one mode"
  )
  expect_error(
    FitAdaStableNet(matrix(1, 3, 2), 1:4, verbose = FALSE),
    "time dimension"
  )
})
