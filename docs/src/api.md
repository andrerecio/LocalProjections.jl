# API Reference

## Estimation

```@docs
lp
lpiv
coefpath
```

## Covariance and summaries

```@docs
vcov(::LocalProjections.CovarianceMatrices.AbstractAsymptoticVarianceEstimator, ::LocalProjections.LPResult)
vcov(::LocalProjections.CovarianceMatrices.AbstractAsymptoticVarianceEstimator, ::LocalProjections.BiasCorrectedLP)
stderror(::LocalProjections.LocalProjectionCovariance)
summarize
ewc_bandwidth
```

## State dependence

`lp` and `lpiv` take `state`, `statelag` and `regimes` keywords; see their
docstrings and the [Ramey–Zubairy tutorial](tutorials/ramey_zubairy.md).

```@docs
statetest
StateTestResult
LPState
```

## Bias correction

```@docs
biascorrect
BiasCorrectedLP
```

## Lag selection

```@docs
lagselect
VARLagSelection
nlags
```

## Bootstrap inference

```@docs
varbootstrap
LPBootstrap
```

## Forecast-error variance decomposition

Gorodnichenko and Lee (2020) local-projection estimators of the share of the
forecast-error variance due to an observed shock, with their bias corrections;
see the [FEVD tutorial](tutorials/fevd.md). Bands come from `summarize` (an
`FEVDSummary`).

```@docs
fevd
LPFEVD
fevdbootstrap
FEVDBootstrap
FEVDSummary
```

## Instrumental variables

```@docs
first_stage(::LocalProjections.LocalProjectionIV, ::Int)
first_stage(::LocalProjections.LocalProjectionIV)
weakivtest(::LocalProjections.LocalProjectionIV, ::Int)
weakivtest(::LocalProjections.LocalProjectionIV)
```

## Result types

```@docs
LocalProjection
LocalProjectionIV
LocalProjectionCovariance
IRFSummary
LocalProjections.LPResult
LocalProjections.LPEstimate
```

## Transformation terms

The left-hand side of an `lp`/`lpiv` formula must be one of `leads`, `cumul`,
`anchor` (`leads(y)|z` is sugar for `anchor(y, z)`), `ldiff`, `hbr`, or
`cumul` of the last two. On the right-hand side, `lags(x, n)` comes from
Regress.jl, `lags(firstdiff(x), n)` gives lagged first differences, and a
bare `cumul(x)`, `leads(x)`, `ldiff(x)` or `hbr(x, y)` tracks the projection
horizon.

```@docs
leads
cumul
anchor
ldiff
hbr
firstdiff
LongDiffTerm
DiffTerm
```

## Plotting and conversion

Plots.jl recipes are available for every result type: `plot(lp, cov_or_estimator; term, levels)`,
`plot(b::LPBootstrap; levels, method)`, `plot(f::LPFEVD; levels, prewhiten)` and
`plot(b::FEVDBootstrap; levels, band, biascorrected, prewhiten)`. The Makie functions `irfplot` and
`irfplot!` are re-exported from MacroEconometricTools.jl and enabled by the
package extension once Makie is loaded.

```@docs
as_irf_result
irfplot_axis
```

## Index

```@index
```
