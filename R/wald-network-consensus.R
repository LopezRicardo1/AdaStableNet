#' Reproducibility Summary for Wald-Sparsified Networks
#'
#' Combines networks fitted to replicate trajectories or bootstrap samples.
#' Because a real-data analysis has no known edge-support truth, selection
#' frequency and sign consistency provide reproducibility summaries without
#' making an unsupported ROC claim.
#'
#' @param networks Nonempty list of `adastablenet_wald_network` objects with
#'   the same states and edge ordering.
#' @param min_frequency Minimum selection frequency used to form the consensus
#'   adjacency matrix.
#'
#' @return An object of class `adastablenet_wald_consensus` with an edge table,
#'   binary consensus adjacency matrix, and median signed consensus weights.
#' @examples
#' sim <- simulate_adastablenet(p = 3, n_time = 21, sigma = 0.03, seed = 8)
#' fit <- FitAdaStableNet(
#'   sim$Y, sim$time, nbasis = 8, twoSE = FALSE,
#'   num_iter = 30, wald_nsteps = 4, verbose = FALSE
#' )
#' networks <- lapply(c("unbounded", "stable"), function(branch) {
#'   AdaStableNet_WaldNetwork(
#'     fit, sim$Y, sim$time, branch = branch, nsteps = 4,
#'     stability_n_grid = 5
#'   )
#' })
#' consensus <- summarize_wald_networks(networks, min_frequency = 0.5)
#' consensus$edge_table
#' @export
summarize_wald_networks <- function(networks, min_frequency = 0.5) {
  if (!is.list(networks) || !length(networks) ||
      any(!vapply(networks, inherits, logical(1L),
                  what = "adastablenet_wald_network"))) {
    stop("`networks` must be a nonempty list of Wald-network objects.",
         call. = FALSE)
  }
  .validate_scalar(min_frequency, "min_frequency", lower = 0, upper = 1)
  reference <- networks[[1L]]
  state_names <- reference$state_names
  reference_edges <- reference$edge_table
  edge_key <- paste(reference_edges$source_index,
                    reference_edges$target_index, sep = "->")
  compatible <- vapply(networks, function(network) {
    identical(network$state_names, state_names) &&
      identical(
        paste(network$edge_table$source_index,
              network$edge_table$target_index, sep = "->"),
        edge_key
      )
  }, logical(1L))
  if (!all(compatible)) {
    stop("All networks must use the same states and directed-edge ordering.",
         call. = FALSE)
  }

  estimates <- vapply(
    networks, function(network) network$edge_table$estimate,
    numeric(nrow(reference_edges))
  )
  selected <- vapply(
    networks, function(network) network$edge_table$selected,
    logical(nrow(reference_edges))
  )
  if (is.null(dim(estimates))) estimates <- matrix(estimates, ncol = 1L)
  if (is.null(dim(selected))) selected <- matrix(selected, ncol = 1L)
  positive <- selected & estimates > 0
  negative <- selected & estimates < 0
  selection_frequency <- rowMeans(selected)
  positive_frequency <- rowMeans(positive)
  negative_frequency <- rowMeans(negative)
  sign_consistency <- ifelse(
    selection_frequency > 0,
    pmax(positive_frequency, negative_frequency) / selection_frequency,
    NA_real_
  )
  median_estimate <- apply(estimates, 1L, stats::median, na.rm = TRUE)
  consensus_selected <- selection_frequency >= min_frequency

  edges <- reference_edges[c(
    "source_index", "target_index", "source", "target"
  )]
  edges$selection_frequency <- selection_frequency
  edges$positive_selection_frequency <- positive_frequency
  edges$negative_selection_frequency <- negative_frequency
  edges$sign_consistency <- sign_consistency
  edges$median_estimate <- median_estimate
  edges$consensus_selected <- consensus_selected

  p <- length(state_names)
  adjacency <- matrix(0L, p, p,
                      dimnames = list(target = state_names,
                                      source = state_names))
  A_network <- matrix(0, p, p, dimnames = dimnames(adjacency))
  edge_positions <- cbind(edges$target_index, edges$source_index)
  adjacency[edge_positions] <- as.integer(consensus_selected)
  A_network[edge_positions] <- ifelse(
    consensus_selected, median_estimate, 0
  )
  diag(adjacency) <- 0L
  diag(A_network) <- 0

  structure(list(
    n_networks = length(networks),
    min_frequency = min_frequency,
    state_names = state_names,
    edge_table = edges,
    adjacency = adjacency,
    A_network = A_network
  ), class = "adastablenet_wald_consensus")
}

#' @export
print.adastablenet_wald_consensus <- function(x, ...) {
  cat("AdaStableNet Wald-network reproducibility summary\n")
  cat("  Networks: ", x$n_networks, "\n", sep = "")
  cat("  Consensus edges: ", sum(x$adjacency), "/",
      length(x$state_names) * (length(x$state_names) - 1L),
      " (selection frequency >= ", format(x$min_frequency), ")\n",
      sep = "")
  invisible(x)
}
