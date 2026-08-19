test_that("simulation is reproducible and has controlled dimensions", {
  x <- simulate_adastablenet(
    p = 5, n_time = 15, matrix_structure = "dense", seed = 99
  )
  y <- simulate_adastablenet(
    p = 5, n_time = 15, matrix_structure = "dense", seed = 99
  )

  expect_s3_class(x, "adastablenet_simulation")
  expect_equal(x$Y, y$Y)
  expect_equal(dim(x$Y), c(15, 5))
  expect_equal(dim(x$A), c(5, 5))
  expect_lte(max(Re(eigen(x$A)$values)), 1e-10)
  expect_equal(kappa(x$Q, exact = TRUE), 5, tolerance = 1e-8)
})

test_that("default sparse construction reproduces the eigen-bound design", {
  x <- simulate_adastablenet(
    p = 16,
    n_time = 15,
    spectrum = "marginal",
    sigma = 0,
    seed = 777
  )

  expect_identical(x$matrix_structure, "sparse")
  expect_equal(x$Q_sparsity, 0.90625, tolerance = 1e-12)
  expect_equal(x$A_sparsity, 0.8515625, tolerance = 1e-12)
  expect_equal(x$condition_number, 4.7754823917, tolerance = 1e-7)
  expect_true(x$condition_target_met)
  expect_equal(sum(abs(x$A) >= 1e-12), 38)
  expect_equal(max(abs(Re(x$eigenvalues))), 0, tolerance = 1e-10)
})

test_that("odd sparse construction reproduces the second eigen-bound design", {
  x <- simulate_adastablenet(
    p = 15,
    n_time = 15,
    spectrum = "mixed",
    sigma = 0,
    seed = 888
  )

  expect_equal(x$Q_sparsity, 0.8977777778, tolerance = 1e-10)
  expect_equal(x$A_sparsity, 0.8044444444, tolerance = 1e-10)
  expect_equal(x$condition_number, 3.196671649, tolerance = 1e-7)
  expect_equal(x$modal_parameters$a, rep(0, 7))
  expect_equal(x$modal_parameters$cc, -0.1)
  expect_equal(sum(abs(x$A) >= 1e-12), 44)
})

test_that("sparse and dense generators expose structure diagnostics", {
  sparse <- simulate_adastablenet(p = 8, n_time = 10, seed = 12)
  dense <- simulate_adastablenet(
    p = 8, n_time = 10, matrix_structure = "dense", seed = 12
  )

  expect_gt(sparse$Q_sparsity, 0)
  expect_gt(sparse$A_sparsity, dense$A_sparsity)
  expect_equal(dense$Q_sparsity, 0)
  expect_equal(dense$A_sparsity, 0)
})

test_that("multiple trajectories use an array", {
  x <- simulate_adastablenet(p = 4, n_time = 10, n_trajectories = 3, seed = 3)
  expect_equal(dim(x$Y), c(10, 4, 3))
  expect_equal(dim(x$x0), c(4, 3))
})
