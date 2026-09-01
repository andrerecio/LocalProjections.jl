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

The left-hand side of an `lp`/`lpiv` formula must be one of `leads`, `cumul`
or `anchor` (`leads(y)|z` is sugar for `anchor(y, z)`). On the right-hand side,
`lags(x, n)` comes from Regress.jl, and a bare `cumul(x)` or `leads(x)`
tracks the projection horizon.

```@docs
leads
cumul
anchor
```

## Plotting and conversion

Plots.jl recipes are available for every result type: `plot(lp, cov_or_estimator; term, levels)`
and `plot(b::LPBootstrap; levels, method)`. The Makie functions `irfplot` and
`irfplot!` are re-exported from MacroEconometricTools.jl and enabled by the
package extension once Makie is loaded.

```@docs
as_irf_result
irfplot_axis
```

## Index

```@index
```
