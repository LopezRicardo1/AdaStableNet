.torch_namespace_available <- function() {
  old_install <- Sys.getenv("TORCH_INSTALL", unset = NA_character_)
  on.exit({
    if (is.na(old_install)) {
      Sys.unsetenv("TORCH_INSTALL")
    } else {
      Sys.setenv(TORCH_INSTALL = old_install)
    }
  }, add = TRUE)
  Sys.setenv(TORCH_INSTALL = "0")
  isTRUE(tryCatch(
    requireNamespace("torch", quietly = TRUE),
    error = function(e) FALSE
  ))
}

.torch_backend_available <- function() {
  if (!.torch_namespace_available()) return(FALSE)
  isTRUE(tryCatch(torch::torch_is_installed(), error = function(e) FALSE))
}

.resolve_optimizer_backend <- function(backend) {
  backend <- match.arg(backend, c("base", "torch", "auto"))
  if (backend == "auto") {
    return(if (.torch_backend_available()) "torch" else "base")
  }
  if (backend == "torch" && !.torch_backend_available()) {
    stop(
      "The Torch backend was requested, but R torch and its LibTorch runtime ",
      "are unavailable. Install `torch`, run `torch::install_torch()`, restart ",
      "R, or use `backend = \"base\"`.", call. = FALSE
    )
  }
  backend
}

.resolve_torch_device <- function(torch_device) {
  torch_device <- match.arg(torch_device, c("auto", "cpu", "cuda"))
  cuda_available <- isTRUE(tryCatch(
    torch::cuda_is_available(), error = function(e) FALSE
  ))
  resolved <- if (torch_device == "auto") {
    if (cuda_available) "cuda" else "cpu"
  } else {
    torch_device
  }
  if (resolved == "cuda" && !cuda_available) {
    stop("`torch_device = \"cuda\"` was requested, but CUDA is unavailable.",
         call. = FALSE)
  }
  list(name = resolved, object = torch::torch_device(resolved))
}

.decode_modal_parameters_torch <- function(theta, q, r, a_mask, cc_mask,
                                           stable, stability_margin) {
  zero <- (theta[1] * 0)$squeeze()
  cursor <- 0L
  a_values <- vector("list", q)
  for (i in seq_len(q)) {
    if (a_mask[i] == 0) {
      a_values[[i]] <- zero
    } else {
      cursor <- cursor + 1L
      a_values[[i]] <- (if (stable) {
        stability_margin - theta[cursor]$exp()
      } else {
        theta[cursor]
      })$squeeze()
    }
  }
  a <- if (q) torch::torch_stack(a_values) else NULL

  b_values <- vector("list", q)
  for (i in seq_len(q)) {
    cursor <- cursor + 1L
    b_values[[i]] <- theta[cursor]$exp()$squeeze()
  }
  b <- if (q) torch::torch_stack(b_values) else NULL

  cc_values <- vector("list", r)
  for (i in seq_len(r)) {
    if (cc_mask[i] == 0) {
      cc_values[[i]] <- zero
    } else {
      cursor <- cursor + 1L
      cc_values[[i]] <- (if (stable) {
        stability_margin - theta[cursor]$exp()
      } else {
        theta[cursor]
      })$squeeze()
    }
  }
  cc <- if (r) torch::torch_stack(cc_values) else NULL
  list(a = a, b = b, cc = cc)
}

.ode_basis_torch <- function(tt, a, b, cc, q, r) {
  rows <- vector("list", 2L * q + r)
  row <- 0L
  if (q) {
    for (i in seq_len(q)) {
      envelope <- (a[i] * tt)$exp()
      row <- row + 1L
      rows[[row]] <- envelope * (b[i] * tt)$sin()
      row <- row + 1L
      rows[[row]] <- envelope * (b[i] * tt)$cos()
    }
  }
  if (r) {
    for (i in seq_len(r)) {
      row <- row + 1L
      rows[[row]] <- (cc[i] * tt)$exp()
    }
  }
  torch::torch_stack(rows, dim = 1L)
}

.torch_profile_loss <- function(theta, Y, tt, q, r, a_mask, cc_mask,
                                stable, stability_margin, ridge.pen,
                                identity) {
  pars <- .decode_modal_parameters_torch(
    theta, q, r, a_mask, cc_mask, stable, stability_margin
  )
  S <- .ode_basis_torch(tt, pars$a, pars$b, pars$cc, q, r)
  gram <- S$mm(S$t())
  scale <- gram$diagonal()$mean()$clamp(min = .Machine$double.eps)
  ridge <- ridge.pen * scale + .Machine$double.eps * scale
  regularized <- gram + ridge * identity
  gram_inv <- torch::linalg_solve(regularized, identity)
  P <- Y$mm(S$t())$mm(gram_inv)
  X_hat <- P$mm(S)
  (Y - X_hat)$pow(2)$mean()
}

.fit_modal_stage_torch <- function(Y, tt, a, b, cc = NULL,
                                   a_mask = NULL, cc_mask = NULL,
                                   stable = FALSE, stability_margin = 0,
                                   ridge.pen = 1e-3, optimizer = "BFGS",
                                   num_iter = 1000L, tol = 1e-8,
                                   n_starts = 1L, start_jitter = 0.05,
                                   seed = NULL, verbose = FALSE,
                                   lr = 0.01, torch_device = "auto",
                                   torch_refine = TRUE,
                                   torch_refine_iter = 20L,
                                   torch_patience = 5L) {
  cc <- if (is.null(cc)) numeric() else as.numeric(cc)
  q <- length(a)
  r <- length(cc)
  a_mask <- as.numeric(a_mask %||% rep(1, q))
  cc_mask <- as.numeric(cc_mask %||% rep(1, r))
  epsilon <- sqrt(.Machine$double.eps)
  theta0 <- .initial_theta(a, b, cc, a_mask, cc_mask, stable,
                           stability_margin, epsilon)
  device <- .resolve_torch_device(torch_device)
  dtype <- torch::torch_float64()
  Y_tensor <- torch::torch_tensor(Y, dtype = dtype, device = device$object)
  tt_tensor <- torch::torch_tensor(tt, dtype = dtype, device = device$object)
  identity <- torch::torch_eye(
    nrow(Y), dtype = dtype, device = device$object
  )

  starts <- .with_seed(seed, {
    lapply(seq_len(n_starts), function(i) {
      if (i == 1L) theta0 else theta0 + stats::rnorm(
        length(theta0), sd = start_jitter
      )
    })
  })

  evaluate_start <- function(start, start_id) {
    if (!length(start)) {
      stop("The Torch backend requires at least one active modal parameter.",
           call. = FALSE)
    }
    if (!is.null(seed)) torch::torch_manual_seed(as.integer(seed + start_id - 1L))
    theta <- torch::torch_tensor(
      start, dtype = dtype, device = device$object, requires_grad = TRUE
    )
    adam <- torch::optim_adam(list(theta = theta), lr = lr)
    history <- numeric()
    gradient_history <- numeric()
    best_value <- Inf
    best_theta <- start
    stable_steps <- 0L
    convergence <- 1L
    message <- "maximum Adam iterations reached"
    evaluations <- 0L

    objective <- function() {
      .torch_profile_loss(
        theta, Y_tensor, tt_tensor, q, r, a_mask, cc_mask,
        stable, stability_margin, ridge.pen, identity
      )
    }

    for (iteration in seq_len(num_iter)) {
      adam$zero_grad()
      loss <- tryCatch(objective(), error = function(e) NULL)
      if (is.null(loss)) {
        convergence <- 52L
        message <- "Torch objective evaluation failed"
        break
      }
      value <- loss$item()
      evaluations <- evaluations + 1L
      if (!is.finite(value)) {
        convergence <- 52L
        message <- "non-finite Torch loss"
        break
      }
      loss$backward()
      gradient_norm <- tryCatch(theta$grad$norm()$item(), error = function(e) Inf)
      history <- c(history, value)
      gradient_history <- c(gradient_history, gradient_norm)
      if (value < best_value) {
        best_value <- value
        best_theta <- as.numeric(theta$detach()$cpu())
      }
      adam$step()

      if (length(history) > 1L) {
        relative_change <- abs(history[length(history)] -
                                 history[length(history) - 1L]) /
          max(abs(history[length(history) - 1L]), .Machine$double.eps)
        stable_steps <- if (relative_change <= tol) stable_steps + 1L else 0L
        if (stable_steps >= torch_patience) {
          convergence <- 0L
          message <- "Adam relative-loss tolerance reached"
          break
        }
      }
    }

    refinement_message <- "not requested"
    if (torch_refine && torch_refine_iter > 0L && is.finite(best_value)) {
      torch::with_no_grad({
        theta$copy_(torch::torch_tensor(
          best_theta, dtype = dtype, device = device$object
        ))
      })
      lbfgs <- torch::optim_lbfgs(
        list(theta = theta), max_iter = as.integer(torch_refine_iter),
        tolerance_grad = max(tol, 1e-10),
        tolerance_change = max(tol, 1e-12), line_search_fn = "strong_wolfe"
      )
      closure <- function() {
        lbfgs$zero_grad()
        loss <- objective()
        loss$backward()
        value <- loss$item()
        evaluations <<- evaluations + 1L
        history <<- c(history, value)
        gradient_history <<- c(
          gradient_history,
          tryCatch(theta$grad$norm()$item(), error = function(e) Inf)
        )
        if (is.finite(value) && value < best_value) {
          best_value <<- value
          best_theta <<- as.numeric(theta$detach()$cpu())
        }
        loss
      }
      refinement_error <- tryCatch({
        lbfgs$step(closure)
        NULL
      }, error = function(e) conditionMessage(e))
      refinement_message <- if (is.null(refinement_error)) {
        convergence <- 0L
        message <- "Torch L-BFGS refinement completed"
        "Torch L-BFGS completed"
      } else {
        paste("Torch L-BFGS skipped after error:", refinement_error)
      }
    }

    list(
      par = best_theta,
      value = best_value,
      convergence = convergence,
      message = message,
      refinement_message = refinement_message,
      counts = c("function" = evaluations, "gradient" = evaluations),
      loss_history = history,
      gradient_history = gradient_history,
      start_id = start_id
    )
  }

  fits <- lapply(seq_along(starts), function(i) evaluate_start(starts[[i]], i))
  values <- vapply(fits, function(x) x$value, numeric(1))
  if (!any(is.finite(values))) {
    stop(
      "Torch optimization failed for every start. Try `torch_device = \"cpu\"`, ",
      "a smaller `lr`, a larger ridge penalty, or `backend = \"base\"`.",
      call. = FALSE
    )
  }
  best <- fits[[which.min(values)]]
  pars <- .decode_modal_parameters(
    best$par, q, r, a_mask, cc_mask, stable, stability_margin
  )
  if (verbose) {
    message(
      if (stable) "Stability-constrained" else "Unconstrained",
      " Torch optimization completed with loss ",
      format(best$value, digits = 6), " on ", device$name,
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
      backend = "torch",
      optimizer = if (torch_refine) "Adam + L-BFGS" else "Adam",
      device = device$name,
      dtype = "float64",
      gradient = "autograd",
      torch_version = as.character(utils::packageVersion("torch")),
      learning_rate = lr,
      convergence = best$convergence,
      message = best$message,
      refinement = best$refinement_message,
      counts = best$counts,
      optimizer_loss = best$value,
      loss = best$value,
      loss_history = best$loss_history,
      gradient_history = best$gradient_history,
      selected_start = best$start_id,
      start_losses = values,
      stable = stable,
      stability_margin = stability_margin
    )
  )
}
