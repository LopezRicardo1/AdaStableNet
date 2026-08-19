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

  network <- AdaStableNet_WaldNetwork(
    fit, sim$Y, sim$time, branch = "stable", nsteps = 4,
    stability_horizon = 0.5, stability_n_grid = 5
  )
  expect_s3_class(network, "adastablenet_wald_network")
  expect_equal(dim(network$adjacency), c(3, 3))
  expect_equal(nrow(network$edge_table), 6)
  expect_equal(network$n_possible_edges, 6)
  expect_true(all(diag(network$adjacency) == 0))
  expect_true(all(is.na(diag(network$p_adjusted))))
  expect_equal(unname(diag(network$A_sparse)),
               unname(diag(network$A_estimate)))
  off_diagonal <- row(network$selected) != col(network$selected)
  expect_true(all(network$A_sparse[off_diagonal & !network$selected] == 0))
  expect_identical(
    network$selected[off_diagonal],
    is.finite(network$p_adjusted[off_diagonal]) &
      network$p_adjusted[off_diagonal] <= 0.05
  )
  expect_true(is.logical(network$stability$after$spectrally_stable))

  two_stage_network <- AdaStableNet_WaldNetwork(
    fit, sim$Y, sim$time, branch = "two_stage", diagonal = "zero",
    nsteps = 4, stability_horizon = 0.5, stability_n_grid = 5
  )
  expect_identical(two_stage_network$branch, "two_stage")
  expect_equal(two_stage_network$x0_hat, fit$Ode2Stage$Yhat_fd[1L, ],
               tolerance = 1e-10)
  expect_true(all(diag(two_stage_network$A_sparse) == 0))
  expect_true(all(diag(two_stage_network$adjacency) == 0))

  consensus <- summarize_wald_networks(
    list(network, two_stage_network), min_frequency = 0.5
  )
  expect_s3_class(consensus, "adastablenet_wald_consensus")
  expect_equal(nrow(consensus$edge_table), 6)
  expect_true(all(consensus$edge_table$selection_frequency %in%
                    c(0, 0.5, 1)))
  expect_true(all(diag(consensus$adjacency) == 0))
})
