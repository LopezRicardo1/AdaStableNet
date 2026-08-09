test_that("simulation is reproducible and has controlled dimensions", {
  x <- simulate_adastablenet(p = 5, n_time = 15, seed = 99)
  y <- simulate_adastablenet(p = 5, n_time = 15, seed = 99)

  expect_s3_class(x, "adastablenet_simulation")
  expect_equal(x$Y, y$Y)
  expect_equal(dim(x$Y), c(15, 5))
  expect_equal(dim(x$A), c(5, 5))
  expect_lte(max(Re(eigen(x$A)$values)), 1e-10)
  expect_equal(kappa(x$Q, exact = TRUE), 5, tolerance = 1e-8)
})

test_that("multiple trajectories use an array", {
  x <- simulate_adastablenet(p = 4, n_time = 10, n_trajectories = 3, seed = 3)
  expect_equal(dim(x$Y), c(10, 4, 3))
  expect_equal(dim(x$x0), c(4, 3))
})
