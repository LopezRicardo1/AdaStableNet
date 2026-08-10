
<!-- README.md is generated from README.Rmd. Please edit that file. -->

# AdaStableNet

AdaStableNet estimates a constant dynamic matrix in the autonomous
linear ODE

$$
\frac{dX(t)}{dt} = A X(t)
$$

from a noisy multivariate trajectory. It combines functional-data
smoothing, gradient-matching initialization, and profiled nonlinear
least squares in a real modal basis. The package returns unconstrained,
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
    ##     branch backend        mse spectral_abscissa modal_rank loading_condition
    ##  unbounded    base 0.00018312         -0.047677          3            2.2526
    ##     stable    base 0.00018312         -0.047677          3            2.2526
    ##  convergence evaluations
    ##            0           9
    ##            0          10

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

## Model branches

- `unbounded` estimates modal real parts without a sign constraint.
- `wald` sets modal real parts that are not distinguishable from zero to
  zero.
- `stable` constrains retained real parts to lie below
  `stability_margin`, which is zero by default.

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

## Source the simulation study

The installed runner uses the original sparse (p=16), seed-777 marginal
design and (p=15), seed-888 mixed design. This quick run uses two
replications per design:

``` r
Sys.setenv(
  ADASTABLENET_N_REP = 2,
  ADASTABLENET_NOISE_SD = "0.30",
  ADASTABLENET_BACKEND = "base",
  ADASTABLENET_OUTPUT_DIR = file.path(
    getwd(), "adastablenet-simulation-results"
  )
)

source(system.file(
  "scripts", "run-simulation-study.R",
  package = "AdaStableNet"
))

simulation_system_diagnostics
head(simulation_results)
simulation_summary
```

For the paper run, set `ADASTABLENET_N_REP = 500`,
`ADASTABLENET_NOISE_SD = "0.05,0.15,0.30"`, and choose
`ADASTABLENET_BACKEND = "base"`, `"torch"`, or `"auto"` before sourcing
the same file. The runner checkpoints every completed replication, so
sourcing it again with the same output directory resumes the study.

## Scope

The current model assumes a constant, autonomous, homogeneous linear ODE
with all states observed and a complete diagonalizable real modal
representation. The approximate Wald calculations condition on the
fitted initial state.
