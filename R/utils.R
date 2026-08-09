`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

.validate_time <- function(tt, min_length = 4L) {
  if (!is.numeric(tt) || is.matrix(tt) || length(tt) < min_length) {
    stop("`tt` must be a numeric vector with at least ", min_length,
         " time points.", call. = FALSE)
  }
  if (any(!is.finite(tt))) {
    stop("`tt` must contain only finite values.", call. = FALSE)
  }
  if (is.unsorted(tt, strictly = TRUE)) {
    stop("`tt` must be strictly increasing with no duplicate values.",
         call. = FALSE)
  }
  as.numeric(tt)
}

.validate_data <- function(Y, tt, orientation = c("time_by_state", "state_by_time")) {
  orientation <- match.arg(orientation)
  if (is.data.frame(Y)) {
    Y <- as.matrix(Y)
  }
  if (!is.matrix(Y) || !is.numeric(Y)) {
    stop("`Y` must be a numeric matrix.", call. = FALSE)
  }
  if (any(!is.finite(Y))) {
    stop("`Y` must contain only finite values.", call. = FALSE)
  }
  expected <- if (orientation == "time_by_state") nrow(Y) else ncol(Y)
  if (expected != length(tt)) {
    stop(
      "The time dimension of `Y` must equal `length(tt)`; expected ",
      orientation, ".", call. = FALSE
    )
  }
  if (min(dim(Y)) < 1L) {
    stop("`Y` must contain at least one state and one observation.", call. = FALSE)
  }
  unname(Y)
}

.validate_scalar <- function(x, name, lower = -Inf, upper = Inf,
                             lower_open = FALSE, upper_open = FALSE,
                             integer = FALSE) {
  if (length(x) != 1L || !is.numeric(x) || !is.finite(x)) {
    stop("`", name, "` must be one finite numeric value.", call. = FALSE)
  }
  lower_bad <- if (lower_open) x <= lower else x < lower
  upper_bad <- if (upper_open) x >= upper else x > upper
  if (lower_bad || upper_bad) {
    interval <- paste0(if (lower_open) "(" else "[", lower, ", ", upper,
                       if (upper_open) ")" else "]")
    stop("`", name, "` must lie in ", interval, ".", call. = FALSE)
  }
  if (integer && x != as.integer(x)) {
    stop("`", name, "` must be an integer.", call. = FALSE)
  }
  invisible(x)
}

.with_seed <- function(seed, code) {
  if (is.null(seed)) {
    return(force(code))
  }
  .validate_scalar(seed, "seed", lower = 0, integer = TRUE)
  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had_seed) {
    old_seed <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  }
  on.exit({
    if (had_seed) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)
  set.seed(as.integer(seed))
  force(code)
}

.safe_kappa <- function(x) {
  out <- tryCatch(kappa(x, exact = TRUE), error = function(e) Inf)
  if (!is.finite(out)) Inf else unname(out)
}

.spectral_abscissa <- function(A) {
  max(Re(eigen(A, only.values = TRUE)$values))
}

.relative_frobenius <- function(estimate, truth) {
  denominator <- sqrt(sum(truth^2))
  if (denominator == 0) {
    return(sqrt(sum((estimate - truth)^2)))
  }
  sqrt(sum((estimate - truth)^2)) / denominator
}

.resolve_branch <- function(object, branch = c("stable", "wald", "unbounded")) {
  branch <- match.arg(branch)
  stages <- if (inherits(object, "adastablenet_fit")) {
    object$AdaEigenStableNet
  } else {
    object
  }
  key <- switch(
    branch,
    stable = "Eigen_Bound",
    wald = "Wald_Real",
    unbounded = "Unbounded"
  )
  stage <- stages[[key]]
  if (is.null(stage)) {
    stop("The `", branch, "` branch was not fitted.", call. = FALSE)
  }
  stage
}
