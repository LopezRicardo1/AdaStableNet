#' AdaStableNet: Stable Estimation of Linear ODE Systems
#'
#' AdaStableNet estimates a constant matrix `A` in the autonomous linear system
#' `dX(t) / dt = A X(t)` from a noisy multivariate trajectory. It combines a
#' functional-data initializer with a profiled real-modal nonlinear
#' least-squares fit. See [FitAdaStableNet()] for the main interface,
#' [predict.adastablenet_fit()] for forecasting, and
#' [simulate_adastablenet()] for reproducible experiments.
#'
#' @references
#' Package methodology and citation information are available through
#' `citation("AdaStableNet")`.
#'
#' @keywords internal
"_PACKAGE"
