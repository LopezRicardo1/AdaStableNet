
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

## Installation

``` r
# install.packages("pak")
pak::pak("LopezRicardo1/AdaStableNet")
```

## Quick start

``` r
library(AdaStableNet)
```

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
    ##   Selected branch: stable
    ##   Training MSE: 0.00017412
    ##   Spectral abscissa: -0.078533
    ##   Modal loading rank: 3/3

``` r
summary(fit)
```

    ## AdaStableNet summary
    ##   Dimensions: 31 time points x 3 states
    ##   Smoothing: Minimum GCV (lambda = 1.776e-06)
    ##
    ##     branch        mse spectral_abscissa modal_rank loading_condition
    ##  unbounded 0.00017412         -0.078533          3            13.807
    ##     stable 0.00017412         -0.078533          3            13.807
    ##  convergence evaluations
    ##            0          39
    ##            0          12

The fitted dynamic matrix and future trajectory use standard R methods:

``` r
A_hat <- coef(fit, branch = "stable")
future <- predict(fit, new_time = seq(0, 2, length.out = 101))
head(future)
```

    ##           [,1]      [,2]      [,3]
    ## [1,] -4.857326 -1.856237 0.8588049
    ## [2,] -4.849353 -1.920652 1.0964955
    ## [3,] -4.766955 -1.950483 1.3229905
    ## [4,] -4.611650 -1.945384 1.5347597
    ## [5,] -4.386096 -1.905556 1.7285147
    ## [6,] -4.094040 -1.831743 1.9012597

## Model branches

- `unbounded` estimates modal real parts without a sign constraint.
- `wald` sets modal real parts that are not distinguishable from zero to
  zero.
- `stable` constrains retained real parts to lie below
  `stability_margin`, which is zero by default.

The package checks the rank and condition number of the fitted modal
loading matrix and reports the actual spectral abscissa of every
reconstructed system matrix. See
`vignette("simulation-study", package = "AdaStableNet")` for the
reproducible simulation design and compact benchmark.

## Scope

The current model assumes a constant, autonomous, homogeneous linear ODE
with all states observed and a complete diagonalizable real modal
representation. The approximate Wald calculations condition on the
fitted initial state.
