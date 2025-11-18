#' AdaStableNet: Adaptive Eigenvalue-Constrained ODE Solver
#'
#' Fits a dynamic system dX/dt = AX using adaptive eigenvalue-constrained regression.
#' Includes optional real eigenvalue Wald test and eigenvalue bounding for stability.
#'
#' @param Y Matrix of observations (features × timepoints)
#' @param tt Vector of timepoints
#' @param initial_a Vector of initial real parts for complex eigenvalues
#' @param initial_b Vector of initial imaginary parts for complex eigenvalues
#' @param initial_cc Optional vector of initial real eigenvalues
#' @param eigen_real_wald Logical; if TRUE, performs Wald test on real eigenvalues
#' @param wald_critical Wald z-score threshold
#' @param eigen_bound Logical; if TRUE, re-optimizes under stability constraints
#' @param lr Learning rate
#' @param num_iter Maximum number of iterations
#' @param tol Relative tolerance for convergence
#' @param ridge.pen Ridge penalty for inversion
#' @param verbose Logical; if TRUE, prints progress
#'
#' @return A list with components:
#'   \item{Unbounded}{Model fit without constraints}
#'   \item{Wald_Real}{Fit after masking unstable real eigenvalues (if enabled)}
#'   \item{Eigen_Bound}{Fit under exponential-stability constraints (if enabled)}
#'
#' @importFrom torch torch_tensor nn_module nn_parameter optim_adam lr_reduce_on_plateau
#' @importFrom torch linalg_eig nnf_mse_loss torch_matmul
#'
#' @export
AdaStableNet <- function(Y, tt, initial_a, initial_b, initial_cc = NULL,
                         eigen_real_wald = TRUE, wald_critical = 2, eigen_bound = TRUE,
                         lr = 0.01, num_iter = 1000, tol = 1e-3, ridge.pen = 1e-2, verbose = TRUE) {

  Y <- torch_tensor(Y)
  tt <- torch_tensor(tt)
  num_single <- length(initial_cc)
  num_pairs <- length(initial_a)

  if (verbose) cat("Starting initial training...\n")
  model_initial <- nn_module(
    initialize = function() {
      self$a <- nn_parameter(torch_tensor(initial_a, requires_grad = TRUE))
      self$b <- nn_parameter(torch_tensor(initial_b, requires_grad = TRUE))
      if (!is.null(initial_cc)) {
        self$cc <- nn_parameter(torch_tensor(initial_cc, requires_grad = TRUE))
      }
    },
    forward = function(tt, Y, ridge.pen) {
      S <- ode_basis(tt, self$a, self$b$abs(), self$cc)
      SS <- torch_matmul(S, S$t())
      SS_eig <- linalg_eig(SS)
      values <- SS_eig[[1]]$real
      vectors <- SS_eig[[2]]$real
      SS_inv <- vectors$mm((1 / torch_maximum(values, ridge.pen * values$sum()))$diag())$mm(vectors$t())
      xi <- Y$mm(S$t())$mm(SS_inv)
      X_hat <- xi$mm(S)
      nnf_mse_loss(Y, X_hat)
    }
  )

  model_instance <- model_initial()
  optimizer <- optim_adam(model_instance$parameters, lr = lr)
  scheduler <- lr_reduce_on_plateau(optimizer)

  prev_loss <- Inf
  for (iteration in seq_len(num_iter)) {
    optimizer$zero_grad()
    loss <- model_instance(tt, Y, ridge.pen)
    loss$backward()
    optimizer$step()
    scheduler$step(loss)
    if (iteration > 1) {
      rel_change <- abs((loss$item() - prev_loss) / prev_loss)
      if (rel_change < tol / 100) {
        if (verbose) cat(sprintf("Convergence reached. Relative change: %.6f\n", rel_change))
        break
      }
    }
    prev_loss <- loss$item()
    if (verbose && iteration %% 500 == 0) {
      cat(sprintf("Initial Iter [%d/%d] Loss = %.6f\n", iteration, num_iter, loss$item()))
    }
  }

  a <- as.numeric(model_instance$a)
  b <- as.numeric(model_instance$b$abs())
  cc <- if (!is.null(initial_cc)) as.numeric(model_instance$cc) else NULL

  if (verbose) {
    cat("Initial training complete.\n")
    cat("Estimated Eigenvalues - Initial Training:\n")
    for (i in seq_len(num_pairs)) {
      cat(sprintf("  Complex Eigenvalue Pair %d: a = %.4f, b = %.4f\n", i, a[i], b[i]))
    }
    if (!is.null(cc)) {
      for (i in seq_along(cc)) {
        cat(sprintf("  Single Real Eigenvalue %d: cc = %.4f\n", i, cc[i]))
      }
    }
  }

  Unbounded <- ada_output(a, b, cc, tt, Y, ridge.pen)

  # -- Wald test --
  Wald_Real <- NULL
  if (eigen_real_wald) {
    if (verbose) cat("Wald test on real eigenvalues...\n")
    res <- as.numeric(Y - Unbounded$X_hat)
    sigma <- sqrt(mean(res^2))
    A_var <- TheoVar(sigma, Unbounded$x0_hat, A = Unbounded$A_hat, tt = as.numeric(tt))
    zs_a <- sqrt(a^2 / A_var$re.eigs.sigma2[1:num_pairs * 2])
    zs_cc <- if (!is.null(cc)) sqrt(cc^2 / A_var$re.eigs.sigma2[-(1:(num_pairs * 2))]) else NULL
    a_wald <- as.numeric(zs_a > wald_critical)
    cc_wald <- if (!is.null(cc)) as.numeric(zs_cc > wald_critical) else NULL

    model_wald <- nn_module(
      initialize = function() {
        self$a <- nn_parameter(torch_tensor(a, requires_grad = TRUE))
        self$b <- nn_parameter(torch_tensor(b, requires_grad = TRUE))
        if (!is.null(cc)) {
          self$cc <- nn_parameter(torch_tensor(cc, requires_grad = TRUE))
        }
      },
      forward = function(tt, Y, ridge.pen, a_wald, cc_wald) {
        S <- ode_basis(tt, self$a * a_wald, self$b$abs(),
                       if (!is.null(self$cc)) self$cc * cc_wald else NULL,
                       a_wald, cc_wald)
        SS <- torch_matmul(S, S$t())
        SS_eig <- linalg_eig(SS)
        values <- SS_eig[[1]]$real
        vectors <- SS_eig[[2]]$real
        SS_inv <- vectors$mm((1 / torch_maximum(values, ridge.pen * values$sum()))$diag())$mm(vectors$t())
        xi <- Y$mm(S$t())$mm(SS_inv)
        X_hat <- xi$mm(S)
        nnf_mse_loss(Y, X_hat)
      }
    )

    model_instance <- model_wald()
    optimizer <- optim_adam(model_instance$parameters, lr = lr)
    prev_loss <- Inf
    for (iteration in seq_len(num_iter)) {
      optimizer$zero_grad()
      loss <- model_instance(tt, Y, ridge.pen, a_wald, cc_wald)
      loss$backward()
      optimizer$step()
      if (iteration > 1) {
        rel_change <- abs((loss$item() - prev_loss) / prev_loss)
        if (rel_change < tol / 100) break
      }
      prev_loss <- loss$item()
    }

    a <- as.numeric(model_instance$a) * a_wald
    b <- as.numeric(model_instance$b$abs())
    cc <- if (!is.null(initial_cc)) as.numeric(model_instance$cc) * cc_wald else NULL

    if (verbose) {
      cat("Estimated Eigenvalues - Wald Real:\n")
      for (i in seq_len(num_pairs)) {
        cat(sprintf("  Complex Eigenvalue Pair %d: a = %.4f, b = %.4f\n", i, a[i], b[i]))
      }
      if (!is.null(cc)) {
        for (i in seq_along(cc)) {
          cat(sprintf("  Single Real Eigenvalue %d: cc = %.4f\n", i, cc[i]))
        }
      }
    }

    Wald_Real <- ada_output(a, b, cc, tt, Y, ridge.pen, a_wald, cc_wald)
  }

  # -- Eigenvalue bounding --
  Eigen_Bound <- NULL
  if (eigen_bound) {
    if (!eigen_real_wald) {
      a_wald <- rep(1, length(a))
      cc_wald <- if (is.null(initial_cc)) NULL else rep(1, length(cc))
    }
    if (is.null(initial_cc)) cc_wald <- NULL

    a <- as.numeric(model_instance$a$abs()$log())
    b <- as.numeric(model_instance$b$abs()$log())
    cc <- if (!is.null(initial_cc)) as.numeric(model_instance$cc$abs()$log()) else NULL

    if (verbose) cat("Performing eigen-bound optimization...\n")

    model_bound <- nn_module(
      initialize = function() {
        self$a <- nn_parameter(torch_tensor(a, requires_grad = TRUE))
        self$b <- nn_parameter(torch_tensor(b, requires_grad = TRUE))
        if (!is.null(cc)) {
          self$cc <- nn_parameter(torch_tensor(cc, requires_grad = TRUE))
        }
      },
      forward = function(tt, Y, ridge.pen, a_wald, cc_wald) {
        S <- if (!is.null(cc)) {
          ode_basis(tt, -self$a$exp(), self$b$exp(), -self$cc$exp(), a_wald, cc_wald)
        } else {
          ode_basis(tt, -self$a$exp(), self$b$exp(), NULL, a_wald, cc_wald)
        }
        SS <- torch_matmul(S, S$t())
        SS_eig <- linalg_eig(SS)
        values <- SS_eig[[1]]$real
        vectors <- SS_eig[[2]]$real
        SS_inv <- vectors$mm((1 / torch_maximum(values, ridge.pen * values$sum()))$diag())$mm(vectors$t())
        xi <- Y$mm(S$t())$mm(SS_inv)
        X_hat <- xi$mm(S)
        nnf_mse_loss(Y, X_hat)
      }
    )

    model_instance <- model_bound()
    optimizer <- optim_adam(model_instance$parameters, lr = lr)
    prev_loss <- Inf
    for (iteration in seq_len(num_iter)) {
      optimizer$zero_grad()
      loss <- model_instance(tt, Y, ridge.pen, a_wald, cc_wald)
      loss$backward()
      optimizer$step()
      if (iteration > 1) {
        rel_change <- abs((loss$item() - prev_loss) / prev_loss)
        if (rel_change < tol / 100) break
      }
      prev_loss <- loss$item()
    }

    a <- as.numeric(-model_instance$a$exp()) * a_wald
    b <- as.numeric(model_instance$b$exp())
    cc <- if (is.null(initial_cc)) NULL else as.numeric(-model_instance$cc$exp()) * cc_wald

    if (verbose) {
      cat("Estimated Eigenvalues - Eigen Bound:\n")
      for (i in seq_len(num_pairs)) {
        cat(sprintf("  Complex Eigenvalue Pair %d: a = %.4f, b = %.4f\n", i, a[i], b[i]))
      }
      if (!is.null(cc)) {
        for (i in seq_along(cc)) {
          cat(sprintf("  Single Real Eigenvalue %d: cc = %.4f\n", i, cc[i]))
        }
      }
    }

    Eigen_Bound <- ada_output(a, b, cc, tt, Y, ridge.pen, a_wald, cc_wald)
  }

  list(Unbounded = Unbounded, Wald_Real = Wald_Real, Eigen_Bound = Eigen_Bound)
}
