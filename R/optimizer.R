.decode_modal_parameters <- function(theta, q, r, a_mask, cc_mask,
                                     stable, stability_margin) {
  active_a <- which(a_mask != 0)
  active_cc <- which(cc_mask != 0)
  cursor <- 0L
  a <- numeric(q)
  if (length(active_a)) {
    idx <- cursor + seq_along(active_a)
    a[active_a] <- if (stable) {
      stability_margin - exp(theta[idx])
    } else {
      theta[idx]
    }
    cursor <- max(idx)
  }
  b_idx <- cursor + seq_len(q)
  b <- if (q) exp(theta[b_idx]) else numeric()
  if (q) cursor <- max(b_idx)
  cc <- numeric(r)
  if (length(active_cc)) {
    idx <- cursor + seq_along(active_cc)
    cc[active_cc] <- if (stable) {
      stability_margin - exp(theta[idx])
    } else {
      theta[idx]
    }
  }
  list(a = a, b = b, cc = cc)
}

.initial_theta <- function(a, b, cc, a_mask, cc_mask, stable,
                           stability_margin, epsilon) {
  active_a <- which(a_mask != 0)
  active_cc <- which(cc_mask != 0)
  theta_a <- if (stable) {
    log(pmax(abs(a[active_a] - stability_margin), 1e-3, epsilon))
  } else {
    a[active_a]
  }
  theta_b <- log(pmax(b, epsilon))
  theta_cc <- if (stable) {
    log(pmax(abs(cc[active_cc] - stability_margin), 1e-3, epsilon))
  } else {
    cc[active_cc]
  }
  c(theta_a, theta_b, theta_cc)
}

.fit_modal_stage <- function(Y, tt, a, b, cc = NULL,
                             a_mask = NULL, cc_mask = NULL,
                             stable = FALSE, stability_margin = 0,
                             ridge.pen = 1e-3, optimizer = "BFGS",
                             num_iter = 1000L, tol = 1e-8,
                             n_starts = 1L, start_jitter = 0.05,
                             seed = NULL, verbose = FALSE) {
  cc <- if (is.null(cc)) numeric() else as.numeric(cc)
  q <- length(a)
  r <- length(cc)
  a_mask <- as.numeric(a_mask %||% rep(1, q))
  cc_mask <- as.numeric(cc_mask %||% rep(1, r))
  epsilon <- sqrt(.Machine$double.eps)
  theta0 <- .initial_theta(a, b, cc, a_mask, cc_mask, stable,
                           stability_margin, epsilon)

  evaluate_start <- function(start, start_id) {
    history <- numeric()
    objective <- function(theta) {
      pars <- .decode_modal_parameters(theta, q, r, a_mask, cc_mask,
                                       stable, stability_margin)
      value <- tryCatch({
        S <- ode_basis(tt, pars$a, pars$b, pars$cc)
        .profile_projection(Y, S, ridge.pen)$loss
      }, error = function(e) 1e100)
      if (!is.finite(value)) value <- 1e100
      history <<- c(history, value)
      value
    }
    if (!length(start)) {
      value <- objective(start)
      fit <- list(par = start, value = value, counts = c("function" = 1L),
                  convergence = 0L, message = NULL)
    } else {
      fit <- stats::optim(
        par = start,
        fn = objective,
        method = optimizer,
        control = list(maxit = as.integer(num_iter), reltol = tol)
      )
    }
    fit$loss_history <- history
    fit$start_id <- start_id
    fit
  }

  fits <- .with_seed(seed, {
    lapply(seq_len(n_starts), function(i) {
      start <- theta0
      if (i > 1L && length(start)) {
        start <- start + stats::rnorm(length(start), sd = start_jitter)
      }
      evaluate_start(start, i)
    })
  })
  values <- vapply(fits, function(x) x$value, numeric(1))
  best <- fits[[which.min(values)]]
  pars <- .decode_modal_parameters(best$par, q, r, a_mask, cc_mask,
                                   stable, stability_margin)
  if (verbose) {
    message(
      if (stable) "Stability-constrained" else "Unconstrained",
      " optimization completed with loss ", format(best$value, digits = 6),
      " (code ", best$convergence, ")."
    )
  }
  list(
    a = pars$a,
    b = pars$b,
    cc = if (r) pars$cc else NULL,
    a_mask = a_mask,
    cc_mask = if (r) cc_mask else NULL,
    diagnostics = list(
      optimizer = optimizer,
      convergence = best$convergence,
      message = best$message %||% "",
      counts = best$counts,
      loss = best$value,
      loss_history = best$loss_history,
      selected_start = best$start_id,
      start_losses = values,
      stable = stable,
      stability_margin = stability_margin
    )
  )
}
