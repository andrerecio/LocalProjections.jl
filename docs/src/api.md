# API Reference

## Estimation

```@docs
lp
lpiv
```

## Results

```@docs
coefpath
stderror(::AbstractLocalProjection)
nobs(::AbstractLocalProjection)
horizonmodel
summarize
shockstd
LocalProjections.critical_value
as_irf_result
```

## IV diagnostics

```@docs
first_stage(::LocalProjectionIV)
first_stage(::LocalProjectionIV, ::Int)
weakivtest(::LocalProjectionIV)
weakivtest(::LocalProjectionIV, ::Int)
```

`normalization` takes MacroEconometricTools' `UnitEffect()` (the default: a shock that
moves the shock variable by one unit) or `UnitStd()` (a one-standard-deviation shock,
[`shockstd`](@ref)); LocalProjections re-exports both.

## Plotting

```@docs
irfplot(::AbstractLocalProjection)
irfplot!(::Any, ::AbstractLocalProjection)
```

## Types

```@docs
AbstractLocalProjection
LocalProjection
LocalProjectionIV
IRFSummary
```

## Transformation Terms

```@docs
leads
cumul
anchor
```

`lags(x, p)` (from Regress.jl) expands to the `p` columns ``x_{t-1}, \ldots, x_{t-p}``.

## Index

```@index
```
