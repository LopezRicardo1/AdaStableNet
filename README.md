
<!-- README.md is generated from README.Rmd. Please edit that file. -->

# AdaStableNet

AdaStableNet estimates a constant dynamic matrix in the autonomous
linear ODE

$$
\frac{dX(t)}{dt} = A X(t)
$$

from a noisy multivariate trajectory. It combines functional-data
smoothing, gradient-matching initialization, and profiled nonlinear
least squares in a real modal basis. By default, the profiled ODE
branches are fitted to the B-spline-reconstructed trajectory rather than
directly to the noisy values. The package returns unconstrained,
Wald-screened, and stability-constrained fits together with diagnostics
and forecasting methods.

The data are conventional time-domain trajectories: each row of the main
interface is an ordered observation time and each column is a system
state. AdaStableNet does not treat observations as biological cells and
does not infer or use pseudotime.

## Installation

``` r
# install.packages("devtools")
devtools::install_github(
  "LopezRicardo1/AdaStableNet",
  build_vignettes = TRUE
)
```

## Quick start

``` r
library(AdaStableNet)
```

The simulator defaults to the sparse construction used in the original
eigen-bound study:

``` r
study_system <- simulate_adastablenet(
  p = 16,
  n_time = 31,
  spectrum = "marginal",
  sigma = 0.03,
  seed = 777
)
c(Q_sparsity = study_system$Q_sparsity,
  A_sparsity = study_system$A_sparsity)
```

    ## Q_sparsity A_sparsity
    ##  0.9062500  0.8515625

``` r
sim <- simulate_adastablenet(
  p = 3,
  n_time = 31,
  spectrum = "stable",
  sigma = 0.03,
  seed = 2026
)

fit <- FitAdaStableNet(
  sim$Y,
  sim$time,
  nbasis = 12,
  twoSE = FALSE,
  fit_ode2fd = TRUE,
  eigen_real_wald = FALSE,
  num_iter = 100,
  verbose = FALSE
)

fit
```

    ## AdaStableNet fit
    ##   States: 3
    ##   Time points: 31
    ##   Backend: base
    ##   Selected branch: stable
    ##   Training MSE: 0.00018312
    ##   Spectral abscissa: -0.047677
    ##   Modal loading rank: 3/3

``` r
summary(fit)
```

    ## AdaStableNet summary
    ##   Dimensions: 31 time points x 3 states
    ##   Smoothing: Minimum GCV (lambda = 5.584e-05)
    ##
    ##     branch                           backend        mse spectral_abscissa
    ##  two_stage functional-data gradient matching 0.00068689         -0.047543
    ##  unbounded                              base 0.00018312         -0.047677
    ##     stable                              base 0.00018312         -0.047677
    ##  numerical_abscissa modal_rank loading_condition convergence evaluations
    ##             0.35261         NA                NA          NA          NA
    ##             0.34745          3            2.2526           0           9
    ##             0.34745          3            2.2526           0          10

The fitted dynamic matrix and future trajectory use standard R methods:

``` r
A_hat <- coef(fit, branch = "stable")
future <- predict(fit, new_time = seq(0, 2, length.out = 101))
head(future)
```

    ##           [,1]      [,2]      [,3]
    ## [1,] 0.9280274 0.8627282 -1.835413
    ## [2,] 0.8145696 0.8856514 -1.870516
    ## [3,] 0.6994422 0.8944568 -1.906190
    ## [4,] 0.5844161 0.8890336 -1.941763
    ## [5,] 0.4712574 0.8694946 -1.976563
    ## [6,] 0.3616997 0.8361743 -2.009935

`predict()` propagates the fitted ODE with a matrix exponential. The
profiled branches start from their fitted modal state;
`branch = "two_stage"` starts from the B-spline-reconstructed state at
the first training time. Thus the forecast is a smooth ODE solution, not
a spline extrapolation and not a curve anchored to the first noisy
measurement.

## Model branches

- `unbounded` estimates modal real parts without a sign constraint.
- `wald` applies a modal Wald screen and sets real parts that are not
  distinguishable from zero to zero.
- `stable` constrains retained real parts to lie below
  `stability_margin`, which is zero by default. This is the
  stability-constrained optimization branch.

The three ideas should not be conflated. `AdaStableNet_WaldTest()`
performs the Wu et al. (2019) trajectory-sensitivity Wald test for
individual entries of the coefficient matrix. The `wald` branch instead
screens modal real parts, and the `stable` branch imposes the spectral
constraint.

## Wald-sparsified interaction network

Use `AdaStableNet_WaldNetwork()` to convert coefficient tests into a
directed network. In `X'(t) = A X(t)`, entry `A[i, j]` is the edge from
state `j` to state `i`. BH adjustment is applied only to off-diagonal
edges. Diagonal self-dynamics are retained in `A_sparse` by default but
are zero in the adjacency and weighted network.

``` r
network <- AdaStableNet_WaldNetwork(
  fit, sim$Y, sim$time,
  branch = "stable", method = "BH", alpha = 0.05,
  nsteps = 4, stability_horizon = 1, stability_n_grid = 11
)

network$adjacency       # binary directed graph, diagonal is zero
```

    ##       source
    ## target X1 X2 X3
    ##     X1  0  1  0
    ##     X2  1  0  1
    ##     X3  0  1  0

``` r
network$A_network       # selected signed off-diagonal weights
```

    ##       source
    ## target       X1        X2       X3
    ##     X1 0.000000 -6.231668 0.000000
    ##     X2 5.550812  0.000000 2.015903
    ##     X3 0.000000 -2.368519 0.000000

``` r
network$A_sparse        # ODE matrix: selected edges plus self-dynamics
```

    ##       source
    ## target         X1          X2         X3
    ##     X1 -0.1499917 -6.23166847  0.0000000
    ##     X2  5.5508115  0.05004857  2.0159026
    ##     X3  0.0000000 -2.36851931 -0.1823235

``` r
head(network$edge_table)
```

    ##   source_index target_index source target    estimate standard_error
    ## 1            1            2     X1     X2  5.55081155     0.08252907
    ## 2            1            3     X1     X3 -0.02258135     0.05795123
    ## 3            2            1     X2     X1 -6.23166847     0.08235155
    ## 4            2            3     X2     X3 -2.36851931     0.06248368
    ## 5            3            1     X3     X1  0.04708496     0.02872289
    ## 6            3            2     X3     X2  2.01590262     0.02765958
    ##       z_score   p_value p_adjusted selected sign
    ## 1  67.2588684 0.0000000  0.0000000     TRUE    1
    ## 2  -0.3896612 0.6967871  0.6967871    FALSE   -1
    ## 3 -75.6715417 0.0000000  0.0000000     TRUE   -1
    ## 4 -37.9062050 0.0000000  0.0000000     TRUE   -1
    ## 5   1.6392839 0.1011541  0.1213850    FALSE    1
    ## 6  72.8826290 0.0000000  0.0000000     TRUE    1

``` r
network$stability       # stability is recomputed after thresholding
```

    ## $before
    ## $branch
    ## [1] NA
    ##
    ## $spectral_abscissa
    ## [1] -0.04767653
    ##
    ## $numerical_abscissa
    ## [1] 0.3474495
    ##
    ## $tolerance
    ## [1] 1.490116e-08
    ##
    ## $spectrally_stable
    ## [1] TRUE
    ##
    ## $euclidean_dissipative
    ## [1] FALSE
    ##
    ## $transient
    ## $transient$horizon
    ## [1] 1
    ##
    ## $transient$time
    ##  [1] 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
    ##
    ## $transient$norm
    ##  [1] 1.0000000 1.0327367 1.0517157 1.0471111 1.0192436 0.9793269 1.0086630
    ##  [8] 1.0267831 1.0219794 0.9944173 0.9540227
    ##
    ## $transient$maximum
    ## [1] 1.051716
    ##
    ## $transient$time_of_maximum
    ## [1] 0.2
    ##
    ##
    ## attr(,"class")
    ## [1] "adastablenet_stability_diagnostics"
    ##
    ## $after
    ## $branch
    ## [1] NA
    ##
    ## $spectral_abscissa
    ## [1] -0.05193204
    ##
    ## $numerical_abscissa
    ## [1] 0.3438177
    ##
    ## $tolerance
    ## [1] 1.490116e-08
    ##
    ## $spectrally_stable
    ## [1] TRUE
    ##
    ## $euclidean_dissipative
    ## [1] FALSE
    ##
    ## $transient
    ## $transient$horizon
    ## [1] 1
    ##
    ## $transient$time
    ##  [1] 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
    ##
    ## $transient$norm
    ##  [1] 1.0000000 1.0323550 1.0509011 1.0458170 1.0174620 0.9777357 1.0065717
    ##  [8] 1.0240964 1.0187660 0.9907462 0.9499824
    ##
    ## $transient$maximum
    ## [1] 1.050901
    ##
    ## $transient$time_of_maximum
    ## [1] 0.2
    ##
    ##
    ## attr(,"class")
    ## [1] "adastablenet_stability_diagnostics"

Elementwise thresholding can change the eigenvalues, so a stable fitted
branch does not automatically make `A_sparse` stable. Use the full
fitted matrix for the primary trajectory estimate and treat
sparse-matrix forecasting as a sensitivity analysis unless the
post-sparsification check remains acceptable.

With the default margin, `stable` guarantees a nonpositive realized
spectral abscissa when the reconstructed modal loading is full rank. A
nonnormal matrix can still have transient growth. Inspect all three
stability measures:

``` r
stability_diagnostics(fit, branch = "stable", horizon = 2)
```

Single-trajectory identifiability is a property of `(A, x0)`. Install
`ode.ident` from `qiuxing/ode.ident` and inspect ICIS with:

``` r
identifiability_diagnostics(sim$A, sim$x0)
```

## Optional Torch backend

The default `backend = "base"` has no Torch dependency and uses
`optim()`. For automatic differentiation, install R Torch and its
LibTorch runtime once:

``` r
install.packages("torch")
torch::install_torch()
```

Then select the backend when fitting:

``` r
fit_torch <- FitAdaStableNet(
  sim$Y,
  sim$time,
  nbasis = 12,
  backend = "torch",
  torch_device = "auto",
  lr = 0.01,
  torch_refine = TRUE,
  num_iter = 100,
  verbose = FALSE
)

fit_torch$control$backend
fit_torch$AdaEigenStableNet$Eigen_Bound$diagnostics[
  c("device", "dtype", "gradient")
]
```

The Torch path uses float64 tensors, Adam optimization, and optional
L-BFGS refinement. `torch_device = "auto"` uses CUDA when it is
available and otherwise uses the CPU. Set `backend = "auto"` to use
Torch only when both the R package and its runtime are available; the
resolved backend is recorded in the fit.

The package checks the rank and condition number of the fitted modal
loading matrix and reports the actual spectral abscissa of every
reconstructed system matrix. The simulator defaults to the sparse
block-embedding construction from the original eigen-bound study. Use
`matrix_structure = "dense"` when an exact loading-matrix condition
number is the intended design. See
`vignette("simulation-study", package = "AdaStableNet")` for the
reproducible simulation design and compact benchmark.

## Package documentation and paper separation

The installed vignette contains a compact, reproducible simulation and
network- recovery example built from exported package functions:

``` r
vignette("simulation-study", package = "AdaStableNet")
```

Large Monte Carlo drivers, real-data preparation, checkpoints,
manuscript figures, tables, and journal source files are intentionally
maintained in a separate paper-analysis repository. They are not
installed with AdaStableNet.

## Scope

The current model assumes a constant, autonomous, homogeneous linear ODE
with all states observed and a complete diagonalizable real modal
representation. The modal and coefficient Wald calculations condition on
the fitted initial state. The coefficient calculation follows the
trajectory-sensitivity Fisher information of Wu et al. (2019); after
stability-constrained boundary fitting, its normal reference
distribution should be checked by simulation.
