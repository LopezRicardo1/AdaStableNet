test_that("Wald branches and coefficient tests have expected shapes", {
  sim <- simulate_adastablenet(
    p = 3, n_time = 17, spectrum = "mixed", sigma = 0.04, seed = 20
  )
  fit <- FitAdaStableNet(
    sim$Y, sim$time, nbasis = 8, twoSE = FALSE,
    num_iter = 40, wald_nsteps = 4, verbose = FALSE
  )

  expect_s3_class(fit$AdaEigenStableNet$Wald_Real, "adastablenet_stage")
  expect_true(is.data.frame(fit$AdaEigenStableNet$Wald_Real$wald$table))
  pvals <- AdaStableNet_WaldTest(
    fit$AdaEigenStableNet$Eigen_Bound,
    sim$Y, sim$time, return = "pvals", nsteps = 4
  )
  expect_equal(dim(pvals), c(3, 3))
  expect_true(all(pvals >= 0 & pvals <= 1))
})
