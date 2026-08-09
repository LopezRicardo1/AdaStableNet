# AdaStableNet sourceable simulation-study runner
#
# Quick run:
#   source(system.file(
#     "scripts", "run-simulation-study.R", package = "AdaStableNet"
#   ))
#
# Paper-scale run:
#   Sys.setenv(
#     ADASTABLENET_N_REP = 500,
#     ADASTABLENET_NOISE_SD = "0.05,0.15,0.30",
#     ADASTABLENET_OUTPUT_DIR = "AdaStableNet-paper-simulation"
#   )
#   source(system.file(
#     "scripts", "run-simulation-study.R", package = "AdaStableNet"
#   ))
#
# The two data-generating systems reproduce the sparse p = 16, seed = 777 and
# p = 15, seed = 888 constructions in ODE.eigen.bound.Rmd. Each system matrix is
# held fixed within a design cell while initial conditions and errors vary.

.simulation_env_integer <- function(name, default, lower = 1L) {
  value <- Sys.getenv(name, unset = as.character(default))
  parsed <- suppressWarnings(as.integer(value))
  if (length(parsed) != 1L || is.na(parsed) || parsed < lower) {
    stop("Environment variable `", name, "` must be an integer >= ", lower,
         ".", call. = FALSE)
  }
  parsed
}

.simulation_env_numeric_vector <- function(name, default) {
  value <- Sys.getenv(name, unset = paste(default, collapse = ","))
  parsed <- suppressWarnings(as.numeric(strsplit(value, ",", fixed = TRUE)[[1L]]))
  if (!length(parsed) || any(!is.finite(parsed)) || any(parsed < 0)) {
    stop("Environment variable `", name,
         "` must contain comma-separated nonnegative numbers.", call. = FALSE)
  }
  unique(parsed)
}

.simulation_env_flag <- function(name, default = TRUE) {
  value <- tolower(Sys.getenv(name, unset = if (default) "true" else "false"))
  if (!value %in% c("true", "false", "1", "0", "yes", "no")) {
    stop("Environment variable `", name, "` must be true or false.",
         call. = FALSE)
  }
  value %in% c("true", "1", "yes")
}

.simulation_predict_matrix <- function(A, time, x0) {
  t(vapply(
    time,
    function(ti) as.numeric(expm::expm(ti * A) %*% x0),
    numeric(nrow(A))
  ))
}

.simulation_matched_eigen_rmse <- function(estimate, truth) {
  remaining <- estimate
  truth <- truth[order(Im(truth), Re(truth))]
  differences <- numeric(length(truth))
  for (i in seq_along(truth)) {
    nearest <- which.min(Mod(remaining - truth[i]))
    differences[i] <- Mod(remaining[nearest] - truth[i])
    remaining <- remaining[-nearest]
  }
  sqrt(mean(differences^2))
}

.simulation_support_metrics <- function(estimate, truth, tolerance) {
  estimated_edge <- abs(estimate) >= tolerance
  true_edge <- abs(truth) >= tolerance
  tp <- sum(estimated_edge & true_edge)
  fp <- sum(estimated_edge & !true_edge)
  tn <- sum(!estimated_edge & !true_edge)
  fn <- sum(!estimated_edge & true_edge)
  precision <- if (tp + fp) tp / (tp + fp) else NA_real_
  recall <- if (tp + fn) tp / (tp + fn) else NA_real_
  f1 <- if (is.finite(precision) && is.finite(recall) &&
            precision + recall > 0) {
    2 * precision * recall / (precision + recall)
  } else {
    NA_real_
  }
  c(
    true_positive_rate = recall,
    false_positive_rate = if (fp + tn) fp / (fp + tn) else NA_real_,
    precision = precision,
    F1 = f1
  )
}

.simulation_summarize <- function(results) {
  group_names <- c("scenario", "p", "spectrum", "sigma", "method")
  metric_names <- c(
    "A_relative_error", "matched_eigen_RMSE",
    "spectral_abscissa_error", "training_RMSE", "future_RMSE",
    "future_RMSE_oracle_x0", "true_positive_rate", "false_positive_rate",
    "precision", "F1"
  )
  groups <- split(
    results,
    interaction(results[group_names], drop = TRUE, lex.order = TRUE)
  )
  rows <- lapply(groups, function(piece) {
    row <- piece[1L, group_names, drop = FALSE]
    row$n_total <- nrow(piece)
    row$n_success <- sum(piece$status == "ok")
    row$failure_rate <- mean(piece$status != "ok")
    for (metric in metric_names) {
      values <- piece[[metric]][piece$status == "ok"]
      values <- values[is.finite(values)]
      row[[paste0(metric, "_mean")]] <- if (length(values)) mean(values) else NA_real_
      row[[paste0(metric, "_median")]] <- if (length(values)) {
        stats::median(values)
      } else {
        NA_real_
      }
      row[[paste0(metric, "_sd")]] <- if (length(values) > 1L) {
        stats::sd(values)
      } else {
        NA_real_
      }
      row[[paste0(metric, "_MCSE")]] <- if (length(values) > 1L) {
        stats::sd(values) / sqrt(length(values))
      } else {
        NA_real_
      }
    }
    row
  })
  summary <- do.call(rbind, rows)
  rownames(summary) <- NULL
  summary[do.call(order, unname(summary[group_names])), , drop = FALSE]
}

.simulation_result_row <- function(scenario, sigma, replicate, method,
                                   simulation_seed, fit_seed, status,
                                   error_message = "", warnings = "") {
  data.frame(
    scenario = scenario$id, p = scenario$p, spectrum = scenario$spectrum,
    sigma = sigma, replicate = replicate, method = method,
    simulation_seed = simulation_seed, fit_seed = fit_seed,
    status = status, error_message = error_message, warnings = warnings,
    A_relative_error = NA_real_, matched_eigen_RMSE = NA_real_,
    spectral_abscissa_truth = NA_real_,
    spectral_abscissa_estimate = NA_real_,
    spectral_abscissa_error = NA_real_, stable_truth = NA,
    stable_estimate = NA, stability_correct = NA,
    training_RMSE = NA_real_, future_RMSE = NA_real_,
    future_RMSE_oracle_x0 = NA_real_, Ahat_sparsity = NA_real_,
    true_positive_rate = NA_real_, false_positive_rate = NA_real_,
    precision = NA_real_, F1 = NA_real_, convergence = NA_integer_,
    modal_rank = NA_integer_, loading_condition = NA_real_,
    fit_elapsed_seconds = NA_real_, stringsAsFactors = FALSE
  )
}

# Run the sparse p = 15 and p = 16 AdaStableNet simulation study.
run_adastablenet_simulation <- function(
    n_rep = 2L,
    noise_sd = 0.30,
    output_dir = file.path(getwd(), "adastablenet-simulation-results"),
    n_time = 201L,
    nbasis = 20L,
    num_iter = 120L,
    wald_nsteps = 6L,
    n_starts = 1L,
    support_tolerance = 1e-6,
    resume = TRUE,
    verbose = TRUE) {
  if (!requireNamespace("AdaStableNet", quietly = TRUE)) {
    stop("Install AdaStableNet before running the study.", call. = FALSE)
  }
  if (utils::packageVersion("AdaStableNet") < numeric_version("0.2.1")) {
    stop("AdaStableNet >= 0.2.1 is required for sparse simulation.",
         call. = FALSE)
  }
  if (!requireNamespace("expm", quietly = TRUE)) {
    stop("Package `expm` is required.", call. = FALSE)
  }

  n_rep <- as.integer(n_rep)
  n_time <- as.integer(n_time)
  nbasis <- as.integer(nbasis)
  num_iter <- as.integer(num_iter)
  wald_nsteps <- as.integer(wald_nsteps)
  n_starts <- as.integer(n_starts)
  if (n_rep < 1L || n_time < 21L || nbasis < 4L || num_iter < 1L ||
      wald_nsteps < 1L || n_starts < 1L) {
    stop("Invalid simulation or fitting control value.", call. = FALSE)
  }
  if (!length(noise_sd) || any(!is.finite(noise_sd)) || any(noise_sd < 0)) {
    stop("`noise_sd` must contain nonnegative finite values.", call. = FALSE)
  }
  if (length(support_tolerance) != 1L || !is.finite(support_tolerance) ||
      support_tolerance <= 0) {
    stop("`support_tolerance` must be positive and finite.", call. = FALSE)
  }

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  output_dir <- normalizePath(output_dir, mustWork = TRUE)
  checkpoint_path <- file.path(output_dir, "checkpoint.rds")
  results_path <- file.path(output_dir, "simulation-results.csv")
  summary_path <- file.path(output_dir, "simulation-summary.csv")
  diagnostics_path <- file.path(output_dir, "system-diagnostics.csv")

  scenarios <- data.frame(
    id = c("eigen_bound_p16_marginal", "eigen_bound_p15_mixed"),
    p = c(16L, 15L), spectrum = c("marginal", "mixed"),
    seed = c(777L, 888L), stringsAsFactors = FALSE
  )
  configuration <- list(
    package_version = as.character(utils::packageVersion("AdaStableNet")),
    scenarios = scenarios, n_rep = n_rep, noise_sd = noise_sd,
    n_time = n_time, time_range = c(0, 2), training_range = c(0, 1),
    nbasis = nbasis, num_iter = num_iter, wald_nsteps = wald_nsteps,
    n_starts = n_starts, support_tolerance = support_tolerance
  )
  saveRDS(configuration, file.path(output_dir, "configuration.rds"))

  completed <- if (resume && file.exists(checkpoint_path)) {
    readRDS(checkpoint_path)
  } else {
    list()
  }
  if (!is.list(completed)) {
    stop("The checkpoint is not a valid result list.", call. = FALSE)
  }

  system_diagnostics <- list()
  total <- nrow(scenarios) * length(noise_sd) * n_rep
  progress <- 0L
  for (scenario_index in seq_len(nrow(scenarios))) {
    scenario <- scenarios[scenario_index, , drop = FALSE]
    for (noise_index in seq_along(noise_sd)) {
      sigma <- noise_sd[noise_index]
      sim <- AdaStableNet::simulate_adastablenet(
        p = scenario$p, n_time = n_time, time_range = c(0, 2),
        spectrum = scenario$spectrum, sigma = sigma,
        matrix_structure = "sparse", sparse_threshold = 0.50,
        condition_number = 5, n_trajectories = n_rep,
        seed = scenario$seed
      )
      if (sim$A_sparsity < 0.50) {
        stop("The generated matrix for `", scenario$id,
             "` is not sufficiently sparse.", call. = FALSE)
      }

      truth_path <- file.path(output_dir, paste0("truth-", scenario$id, ".rds"))
      if (!file.exists(truth_path)) {
        saveRDS(list(A = sim$A, J = sim$J, Q = sim$Q,
                     eigenvalues = sim$eigenvalues,
                     modal_parameters = sim$modal_parameters), truth_path)
        utils::write.csv(
          sim$A, file.path(output_dir, paste0("A-", scenario$id, ".csv")),
          row.names = FALSE
        )
        grDevices::png(
          file.path(output_dir, paste0("A-pattern-", scenario$id, ".png")),
          width = 1200, height = 1100, res = 160
        )
        graphics::image(
          abs(sim$A) >= 1e-12, col = c("white", "#0072B2"), axes = FALSE,
          xlab = "Row", ylab = "Column",
          main = paste("Nonzero pattern:", scenario$id)
        )
        graphics::axis(1, at = seq(0, 1, length.out = scenario$p),
                       labels = seq_len(scenario$p))
        graphics::axis(2, at = seq(0, 1, length.out = scenario$p),
                       labels = seq_len(scenario$p))
        graphics::box()
        grDevices::dev.off()
      }

      diagnostic_key <- paste(scenario$id, format(sigma), sep = "|")
      system_diagnostics[[diagnostic_key]] <- data.frame(
        scenario = scenario$id, p = scenario$p,
        spectrum = scenario$spectrum, sigma = sigma,
        simulation_seed = scenario$seed, A_sparsity = sim$A_sparsity,
        A_nonzeros = sum(abs(sim$A) >= 1e-12),
        Q_sparsity = sim$Q_sparsity, Q_condition = sim$condition_number,
        spectral_abscissa = max(Re(sim$eigenvalues)),
        stringsAsFactors = FALSE
      )

      X <- sim$X
      Y <- sim$Y
      if (n_rep == 1L) {
        dim(X) <- c(n_time, scenario$p, 1L)
        dim(Y) <- c(n_time, scenario$p, 1L)
      }
      train <- sim$time <= 1
      future <- sim$time > 1

      for (replicate in seq_len(n_rep)) {
        progress <- progress + 1L
        key <- sprintf("%s|sigma=%.8g|rep=%06d",
                       scenario$id, sigma, replicate)
        if (!is.null(completed[[key]])) {
          if (verbose) message("[", progress, "/", total, "] resumed ", key)
          next
        }
        if (verbose) message("[", progress, "/", total, "] fitting ", key)

        observed <- Y[, , replicate, drop = TRUE]
        truth <- X[, , replicate, drop = TRUE]
        fit_seed <- as.integer(
          scenario$seed + 100000L + noise_index * 10000L + replicate
        )
        fit_warnings <- character()
        fit_error <- ""
        start_time <- proc.time()[["elapsed"]]
        fit <- tryCatch(
          withCallingHandlers(
            AdaStableNet::FitAdaStableNet(
              observed[train, , drop = FALSE], sim$time[train],
              nbasis = min(nbasis, sum(train) - 2L),
              lambda_range = c(-14, 1), twoSE = FALSE,
              eigen_real_wald = TRUE, eigen_bound = TRUE,
              num_iter = num_iter, n_starts = n_starts,
              wald_nsteps = wald_nsteps, seed = fit_seed, verbose = FALSE
            ),
            warning = function(warning_condition) {
              fit_warnings <<- c(fit_warnings,
                                 conditionMessage(warning_condition))
              invokeRestart("muffleWarning")
            }
          ),
          error = function(error_condition) {
            fit_error <<- conditionMessage(error_condition)
            NULL
          }
        )
        elapsed <- proc.time()[["elapsed"]] - start_time
        warning_text <- paste(unique(fit_warnings), collapse = " | ")
        methods <- c("two_stage", "unbounded", "wald", "stable")

        if (is.null(fit)) {
          completed[[key]] <- do.call(rbind, lapply(methods, function(method) {
            .simulation_result_row(
              scenario, sigma, replicate, method, scenario$seed, fit_seed,
              status = "fit_error", error_message = fit_error,
              warnings = warning_text
            )
          }))
          saveRDS(completed, checkpoint_path)
          next
        }

        estimates <- list(
          two_stage = fit$Ode2Stage$Ahat,
          unbounded = stats::coef(fit, branch = "unbounded"),
          wald = stats::coef(fit, branch = "wald"),
          stable = stats::coef(fit, branch = "stable")
        )
        rows <- vector("list", length(methods))
        names(rows) <- methods
        truth_abscissa <- max(Re(eigen(sim$A, only.values = TRUE)$values))

        for (method in methods) {
          A_hat <- estimates[[method]]
          row <- .simulation_result_row(
            scenario, sigma, replicate, method, scenario$seed, fit_seed,
            status = "ok", warnings = warning_text
          )
          prediction_error <- ""
          predicted <- tryCatch(
            if (method == "two_stage") {
              .simulation_predict_matrix(A_hat, sim$time, observed[1L, ])
            } else {
              stats::predict(fit, sim$time, branch = method)
            },
            error = function(error_condition) {
              prediction_error <<- conditionMessage(error_condition)
              NULL
            }
          )
          oracle_prediction <- tryCatch(
            .simulation_predict_matrix(A_hat, sim$time, truth[1L, ]),
            error = function(error_condition) NULL
          )
          if (is.null(predicted) || any(!is.finite(A_hat))) {
            row$status <- "method_error"
            row$error_message <- prediction_error
            rows[[method]] <- row
            next
          }

          estimate_abscissa <- max(Re(eigen(A_hat, only.values = TRUE)$values))
          support <- .simulation_support_metrics(A_hat, sim$A,
                                                 support_tolerance)
          row$A_relative_error <- sqrt(
            sum((A_hat - sim$A)^2) / sum(sim$A^2)
          )
          row$matched_eigen_RMSE <- .simulation_matched_eigen_rmse(
            eigen(A_hat, only.values = TRUE)$values, sim$eigenvalues
          )
          row$spectral_abscissa_truth <- truth_abscissa
          row$spectral_abscissa_estimate <- estimate_abscissa
          row$spectral_abscissa_error <- abs(estimate_abscissa - truth_abscissa)
          row$stable_truth <- truth_abscissa <= 1e-7
          row$stable_estimate <- estimate_abscissa <= 1e-7
          row$stability_correct <- row$stable_truth == row$stable_estimate
          row$training_RMSE <- sqrt(mean(
            (predicted[train, , drop = FALSE] - truth[train, , drop = FALSE])^2
          ))
          row$future_RMSE <- sqrt(mean(
            (predicted[future, , drop = FALSE] -
               truth[future, , drop = FALSE])^2
          ))
          row$future_RMSE_oracle_x0 <- if (is.null(oracle_prediction)) {
            NA_real_
          } else {
            sqrt(mean((oracle_prediction[future, , drop = FALSE] -
                         truth[future, , drop = FALSE])^2))
          }
          row$Ahat_sparsity <- mean(abs(A_hat) < support_tolerance)
          row$true_positive_rate <- support[["true_positive_rate"]]
          row$false_positive_rate <- support[["false_positive_rate"]]
          row$precision <- support[["precision"]]
          row$F1 <- support[["F1"]]
          row$fit_elapsed_seconds <- elapsed

          if (method != "two_stage") {
            stage_name <- switch(
              method, unbounded = "Unbounded", wald = "Wald_Real",
              stable = "Eigen_Bound"
            )
            diagnostics <- fit$AdaEigenStableNet[[stage_name]]$diagnostics
            row$convergence <- diagnostics$convergence
            row$modal_rank <- diagnostics$modal_rank
            row$loading_condition <- diagnostics$loading_condition
          }
          rows[[method]] <- row
        }
        completed[[key]] <- do.call(rbind, rows)
        rownames(completed[[key]]) <- NULL
        saveRDS(completed, checkpoint_path)
      }
    }
  }

  results <- do.call(rbind, unname(completed))
  rownames(results) <- NULL
  results <- results[order(results$scenario, results$sigma,
                           results$replicate, results$method), , drop = FALSE]
  summary <- .simulation_summarize(results)
  system_diagnostics <- do.call(rbind, system_diagnostics)
  rownames(system_diagnostics) <- NULL
  utils::write.csv(results, results_path, row.names = FALSE)
  utils::write.csv(summary, summary_path, row.names = FALSE)
  utils::write.csv(system_diagnostics, diagnostics_path, row.names = FALSE)

  successful <- results[results$status == "ok" &
                          is.finite(results$future_RMSE), , drop = FALSE]
  plot_path <- file.path(output_dir, "future-RMSE-boxplot.png")
  if (nrow(successful)) {
    grDevices::png(plot_path, width = 2200, height = 1300, res = 170)
    labels <- interaction(
      successful$scenario, paste0("sigma=", successful$sigma),
      successful$method, sep = "\n", drop = TRUE
    )
    graphics::boxplot(
      log10(successful$future_RMSE) ~ labels,
      las = 2, ylab = expression(log[10]("future RMSE")), xlab = "",
      col = "grey90", main = "AdaStableNet sparse simulation study"
    )
    grDevices::dev.off()
  }
  if (verbose) {
    message("Simulation complete: ", nrow(results), " method results.")
    message("Output directory: ", output_dir)
  }
  list(
    results = results, summary = summary,
    system_diagnostics = system_diagnostics, configuration = configuration,
    files = list(checkpoint = checkpoint_path, results = results_path,
                 summary = summary_path, diagnostics = diagnostics_path,
                 plot = if (file.exists(plot_path)) plot_path else NULL)
  )
}

if (.simulation_env_flag("ADASTABLENET_AUTORUN", TRUE)) {
  adastablenet_study <- run_adastablenet_simulation(
    n_rep = .simulation_env_integer("ADASTABLENET_N_REP", 2L),
    noise_sd = .simulation_env_numeric_vector("ADASTABLENET_NOISE_SD", 0.30),
    output_dir = Sys.getenv(
      "ADASTABLENET_OUTPUT_DIR",
      unset = file.path(getwd(), "adastablenet-simulation-results")
    ),
    n_time = .simulation_env_integer("ADASTABLENET_N_TIME", 201L, 21L),
    nbasis = .simulation_env_integer("ADASTABLENET_NBASIS", 20L, 4L),
    num_iter = .simulation_env_integer("ADASTABLENET_NUM_ITER", 120L),
    wald_nsteps = .simulation_env_integer("ADASTABLENET_WALD_STEPS", 6L),
    n_starts = .simulation_env_integer("ADASTABLENET_N_STARTS", 1L),
    support_tolerance = as.numeric(Sys.getenv(
      "ADASTABLENET_EDGE_TOLERANCE", unset = "1e-6"
    )),
    resume = .simulation_env_flag("ADASTABLENET_RESUME", TRUE),
    verbose = TRUE
  )
  simulation_results <- adastablenet_study$results
  simulation_summary <- adastablenet_study$summary
  simulation_system_diagnostics <- adastablenet_study$system_diagnostics
}
