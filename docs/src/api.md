# API Reference

## Estimation

```@docs
lp
lpiv
horizonmodel
coefpath
nobs(::AbstractLocalProjection)
```

## Covariance and summaries

The covariance estimator is stored in the result (HC1 by default, `CR1(unit)` for
panels); `lp_result + vcov(estimator)` returns a copy that uses `estimator` at every
horizon. Bands built on `EWC(B)` use Student-``t_B`` critical values.

```@docs
stderror(::AbstractLocalProjection)
summarize
IRFSummary
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
first_stage(::LocalProjectionIV, ::Int)
first_stage(::LocalProjectionIV)
weakivtest(::LocalProjectionIV, ::Int)
weakivtest(::LocalProjectionIV)
```

## Result types

```@docs
AbstractLocalProjection
LocalProjection
LocalProjectionIV
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
LocalProjections.LongDiffTerm
LocalProjections.DiffTerm
```

## Plotting and conversion

Every plot draws the `LocalProjectionIRFResult` that `as_irf_result` returns, with
the plotting methods of MacroEconometricTools.jl. With Plots.jl:
`plot(lp; term, levels)`, `plot(lp, estimator; term, levels)`,
`plot(b::LPBootstrap; levels, method)`, `plot(f::LPFEVD; levels, prewhiten)` and
`plot(b::FEVDBootstrap; levels, band, biascorrected, prewhiten)`. The Makie
functions `irfplot` and `irfplot!` are re-exported from MacroEconometricTools.jl and
need a Makie backend.

```@docs
as_irf_result(::AbstractLocalProjection)
as_irf_result(::LPBootstrap)
as_irf_result(::LPFEVD)
as_irf_result(::FEVDBootstrap)
irfplot(::AbstractLocalProjection)
irfplot!(::Any, ::AbstractLocalProjection)
irfplot(::LPBootstrap)
irfplot(::LPFEVD)
```

## Index

```@index
```
