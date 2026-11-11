# Tutorial 3: Inference and Plotting

This tutorial covers robust variance estimation and IRF visualization.

## Setup

```@example infer
using LocalProjections, DataFrames, Random
using StatsModels: @formula
using CovarianceMatrices: HC1, HC3, Bartlett, Parzen, NeweyWest
Random.seed!(456)

n = 200
df = DataFrame(
    y = cumsum(randn(n)),
    x = randn(n)
)

lp_result = lp(@formula(leads(y) ~ x), df; horizon=12)
nothing # hide
```

## Robust Standard Errors

LocalProjections.jl integrates with CovarianceMatrices.jl for robust inference. Each
result stores a covariance estimator (HC1 by default, `CR1(unit)` for panels fitted with
`unit`); `lp_result + vcov(estimator)` returns a copy that uses `estimator` at every
horizon. `stderror`, `summarize`, the plots and `as_irf_result` all use the stored
estimator. Cluster estimators name their clusters by column, e.g. `vcov(CR1(:id))`, and
each horizon is clustered on the rows it uses. Driscoll-Kraay standard errors for panels
name the time and unit columns the same way:
`vcov(DriscollKraay(Bartlett(4); tis = :t, iis = :id))`.

### Heteroskedasticity-Robust (HC)

```@example infer
# HC1 (default)
se_hc1 = stderror(lp_result; term=:x)
println("HC1 standard errors: ", round.(se_hc1, digits=4))

# HC3 (more conservative)
lp_hc3 = lp_result + vcov(HC3())
se_hc3 = stderror(lp_hc3; term=:x)
println("HC3 standard errors: ", round.(se_hc3, digits=4))
```

### HAC (Newey-West style)

For time series with autocorrelated errors:

```@example infer
# Bartlett kernel with Newey-West bandwidth
lp_nw = lp_result + vcov(Bartlett{NeweyWest}())
se_nw = stderror(lp_nw; term=:x)
println("Newey-West SEs: ", round.(se_nw, digits=4))

# Parzen kernel
se_parzen = stderror(lp_result + vcov(Parzen{NeweyWest}()); term=:x)
println("Parzen SEs: ", round.(se_parzen, digits=4))
```

### Fixed Bandwidth HAC

```@example infer
# Fixed bandwidth of 5
se_fixed = stderror(lp_result + vcov(Bartlett(5)); term=:x)
println("Fixed bandwidth SEs: ", round.(se_fixed, digits=4))
```

### Equal-weighted cosine (EWC)

`EWC(B)` projects the regression moments on `B` cosine functions. With `B` fixed, its
t-statistic is Student-t with `B` degrees of freedom in the limit (Lazarus, Lewis, Stock
and Watson 2018), so the bands of an EWC result use Student-t(`B`) critical values,
which are wider than the normal ones; every other estimator uses the normal quantile.
`critical_value` gives the value used:

```@example infer
using CovarianceMatrices: EWC
lp_ewc = lp_result + vcov(EWC(8))
s_ewc = summarize(lp_ewc; level=0.95)
println("Critical value: ", round(LocalProjections.critical_value(EWC(8), 0.95), digits=3))
println("Half-widths: ", round.((s_ewc.upper .- s_ewc.lower) ./ 2, digits=3))
```

## Summary Tables with `summarize()`

Get an `IRFSummary` table with coefficients, standard errors, and confidence intervals:

```@example infer
summary_irf = summarize(lp_nw)
println(summary_irf)
```

### Overriding the estimator

Passing an estimator is the same as `summarize(lp_result + vcov(estimator))`:

```@example infer
summary_irf = summarize(lp_result, HC1())
println(summary_irf)
```

### Scaling for Percentages

```@example infer
summary_pct = summarize(lp_result, HC1(); scale=100, level=0.90)
println(summary_pct)
```

### One-standard-deviation shocks

By default the response is to a shock that moves `x` by one unit (`UnitEffect()`).
With `normalization = UnitStd()` it is the response to a one-standard-deviation shock:
`σ̂ β_h`, where `σ̂ = shockstd(lp_result)` is the standard deviation of the residual of
`x` on the other regressors over the rows of horizon 0. The standard errors come from
the delta method, which includes the sampling error of `σ̂` and its covariance with
`β_h`, estimated with the stored covariance estimator:

```math
\operatorname{Var}(\hat\sigma \hat\beta_h) \approx
\sigma^2 \operatorname{Var}(\hat\beta_h) + \beta_h^2 \operatorname{Var}(\hat\sigma)
+ 2 \sigma \beta_h \operatorname{Cov}(\hat\beta_h, \hat\sigma).
```

Multiplying the unit-effect bands by `σ̂` instead treats `σ̂` as known and gives bands
that are too narrow, most of all at short horizons, where `β_h` is large relative to
its standard error.

```@example infer
lp_std = lp(@formula(leads(y) ~ x + lags(y, 2) + lags(x, 2)), df; horizon = 12)
shockstd(lp_std)
```

```@example infer
println(summarize(lp_std; normalization = UnitStd()))
```

`as_irf_result` and the plots take the same keyword. IV projections (`lpiv`) identify
the response to a unit change in the instrumented variable but not the size of the
structural shock, so `UnitStd()` is an error for them.

### Different Confidence Levels

```@example infer
# 68% CI (approx. 1 SE)
summary_68 = summarize(lp_result, HC1(); level=0.68)
println("68% CI width: ", round.(summary_68.upper .- summary_68.lower, digits=3))

# 99% CI
summary_99 = summarize(lp_result, HC1(); level=0.99)
println("99% CI width: ", round.(summary_99.upper .- summary_99.lower, digits=3))
```

## Plotting IRFs

Using Plots.jl with our plot recipes:

```@example infer
using Plots

# Basic plot with 95% CI
plot(lp_result, HC1())
```

### Customizing Plots

```@example infer
# Multiple confidence levels, scaled
plot(lp_result, HC1();
     levels=[0.68, 0.95],
     title="Impulse Response Function",
     xlabel="Horizon",
     ylabel="Response",
     legend=:topright)
```

### Comparing Variance Estimators

```@example infer
p1 = plot(lp_result, HC1(); title="HC1", legend=false)
p2 = plot(lp_result, HC3(); title="HC3", legend=false)
p3 = plot(lp_result, Bartlett{NeweyWest}(); title="Newey-West", legend=false)

plot(p1, p2, p3, layout=(1,3), size=(900, 300))
```

## Available Variance Estimators

| Estimator | Description |
|-----------|-------------|
| `HC0()` - `HC5()` | Heteroskedasticity-consistent (White) |
| `Bartlett{NeweyWest}()` | HAC with Newey-West bandwidth |
| `Bartlett(bw)` | HAC with fixed bandwidth |
| `Parzen{NeweyWest}()` | HAC with Parzen kernel |
| `QuadraticSpectral{Andrews}()` | HAC with QS kernel |
| `EWC(B)` | Equal-weighted cosine with `B` basis functions; Student-t(`B`) bands |

See [CovarianceMatrices.jl documentation](https://gragusa.github.io/CovarianceMatrices.jl/stable/) for more options.
