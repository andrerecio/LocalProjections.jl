# Tutorial 3: Inference and Plotting

This tutorial covers robust variance estimation and IRF visualization.

## Setup

```@example infer
using LocalProjections, DataFrames, Random
using StatsModels: @formula
using CovarianceMatrices: HC1, HC3, Bartlett, Parzen, NeweyWest, EWC
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

### Equal-Weighted Cosine (EWC)

The EWC estimator of Lazarus, Lewis, Stock and Watson (2018) averages `B` cosine-weighted
periodogram terms. `ewc_bandwidth` gives their rule ``B = \lfloor 0.41\,T_0^{2/3} \rfloor``
from the horizon-0 sample, to be held fixed across horizons. Bands on an EWC covariance
use Student-``t_B`` critical values instead of normal ones (fixed-smoothing asymptotics),
so they are wider than a normal band with the same standard errors.

```@example infer
B = ewc_bandwidth(lp_result)
lp_ewc = lp_result + vcov(EWC(B))
summarize(lp_ewc; term=:x, level=0.90)
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

### Different Confidence Levels

```@example infer
# 68% CI (approx. 1 SE)
summary_68 = summarize(lp_result, HC1(); level=0.68)
println("68% CI width: ", round.(summary_68.upper .- summary_68.lower, digits=3))

# 99% CI
summary_99 = summarize(lp_result, HC1(); level=0.99)
println("99% CI width: ", round.(summary_99.upper .- summary_99.lower, digits=3))
```

## Joint Inference Across Horizons

The bands above are *pointwise*: each covers its own horizon with probability
`level`, so they cannot test hypotheses that involve several horizons (is the
response zero everywhere? is it monotone?). The estimates of different horizons
come from regressions on overlapping samples and are correlated. `jointvcov`
estimates their covariance by stacking the scores of every horizon on one time axis
(Stata's `suest`), under the stored estimator, with the pointwise standard errors on
its diagonal:

```@example infer
V = jointvcov(lp_result; term=:x)
round.(V[1:4, 1:4], digits=5)
```

`jointtest` is the Wald test that the response equals `value` (0 by default) at all
`horizons` (all of them by default). Its reference distribution follows the
estimator: ``\chi^2_m`` for HC and kernel HAC, and the fixed-smoothing
``F_{m,B-m+1}`` for `EWC(B)`, which needs ``B \ge m``.

```@example infer
jointtest(lp_result; term=:x)
```

```@example infer
jointtest(lp_ewc; term=:x, horizons=0:6)
```

A *simultaneous* band covers the whole response with probability `level`.
The sup-t band of Montiel Olea and Plagborg-Møller (2019) widens the pointwise one
by a common critical value ``c``, the `level` quantile of
``\max_h |\hat\theta_h - \theta_h| / \mathrm{se}_h``, which is simulated from the
correlation in `jointvcov` (for `EWC(B)`, from its fixed-smoothing limit).
`bands = :supt` builds it in `summarize`, `plot`, `irfplot` and `as_irf_result`:

```@example infer
supt_critical_value(lp_result; term=:x, level=0.90)   # vs 1.645 pointwise
```

```@example infer
summarize(lp_result; term=:x, level=0.90, bands=:supt)
```

Any shape restriction can be read off the simultaneous band: if no function with
the hypothesized shape fits inside it, the hypothesis is rejected at `1 - level`.
The critical value is simulated with a fixed default seed (100,000 draws, Monte
Carlo error up to about 0.01 at 95%); pass `rng` and `ndraws` to `supt_critical_value` to
change them. Joint inference covers OLS and IV projections on a single time series
with HC0–HC5, kernel HAC or EWC estimators; panels, `fe()` and cluster or
Driscoll–Kraay estimators throw.

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

### Simultaneous Bands

```@example infer
plot(lp_result, HC1(); levels=[0.90], bands=:supt, title="90% sup-t band")
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
| `EWC(B)` | Equal-weighted cosine, Student-``t_B`` bands (`B = ewc_bandwidth(lp_result)`) |

See [CovarianceMatrices.jl documentation](https://gragusa.github.io/CovarianceMatrices.jl/stable/) for more options.
