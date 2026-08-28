# LocalProjections.jl

[![CI](https://github.com/gragusa/LocalProjections.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/gragusa/LocalProjections.jl/actions/workflows/CI.yml) [![codecov.io](http://codecov.io/github/gragusa/LocalProjections.jl/coverage.svg?branch=master)](http://codecov.io/github/gragusa/LocalProjections.jl?branch=master) [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) ![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826) ![lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)

Impulse response functions by local projections (Jordà, 2005): horizon-specific linear regressions with a formula interface, robust and HAR inference, and instrumental-variable support.

## Installation

This fork of [gragusa/LocalProjections.jl](https://github.com/gragusa/LocalProjections.jl)
extends the inference procedures (EWC HAR inference, with bias correction and
bootstrap bands in progress):

```julia
using Pkg
Pkg.add(url = "https://github.com/andrerecio/LocalProjections.jl", rev = "dev")
```

## Quick start

```julia
using LocalProjections, DataFrames, StatsModels, CovarianceMatrices, Plots

# Simulated data: y responds to cumulated shocks
n = 200
shock = randn(n)
y = cumsum(shock) + 0.5 * randn(n)
df = DataFrame(y = y, shock = shock)

# y_{t+h} = α_h + θ_h·shock_t + controls,  h = 0, …, 20
lp_result = lp(@formula(leads(y) ~ shock + lags(y, 2)), df; horizon = 20)

irf = coefpath(lp_result; term = :shock)                       # θ_0, …, θ_20
plot(lp_result, Bartlett{NeweyWest}(); term = :shock, levels = [0.68, 0.95])
```

### Response types

The formula LHS selects how the response is transformed at each horizon:

| LHS                             | Response at horizon `h`   |
| ------------------------------- | ------------------------- |
| `leads(y)`                      | `y[t+h]`                  |
| `cumul(y)`                      | `y[t] + ⋯ + y[t+h]`       |
| `anchor(y, z)` (or `leads(y)\|z`) | `y[t+h] − z[t]`         |

The RHS accepts `lags(x, n)` and standard StatsModels transformations.

## Inference

Any covariance estimator from
[CovarianceMatrices.jl](https://github.com/gragusa/CovarianceMatrices.jl) can
be applied horizon-by-horizon. `summarize` returns coefficients, standard
errors, and confidence bands as a table:

```julia
# Heteroskedasticity-robust (Eicker–Huber–White, HC1)
summarize(lp_result, HC1(); level = 0.90)

# HAC (Newey–West with a Bartlett kernel)
summarize(lp_result, Bartlett{NeweyWest}(); level = 0.90)

# Equal-weighted cosine (EWC) HAR inference, Lazarus–Lewis–Stock–Watson (2018)
B = ewc_bandwidth(lp_result)        # B = ⌊0.41·T₀^(2/3)⌋ from the h = 0 sample
summarize(lp_result, EWC(B); level = 0.90)
```

Confidence bands use the critical values canonically paired with the
estimator: Student-`t_B` for `EWC(B)` (fixed-smoothing asymptotics), standard
normal otherwise. The same rule applies in `summarize`, `plot`, and
`as_irf_result`.

For lower-level access, compute the covariance once and reuse it:

```julia
cov_ewc = vcov(EWC(B), lp_result)
se = stderror(cov_ewc; term = :shock)
plot(lp_result, cov_ewc; term = :shock, levels = [0.90])
```

## Instrumental variables

`lpiv` estimates local projections with endogenous regressors using
`(endogenous ~ instruments)` syntax, with first-stage diagnostics and the
Montiel Olea–Pflueger weak-instrument test:

```julia
# x is endogenous, z is its instrument
result = lpiv(@formula(leads(y) ~ (x ~ z) + lags(y, 2)), df; horizon = 10)

first_stage(result, 0)   # first-stage regression at horizon 0
weakivtest(result, 0)    # effective F and critical values
```

## License

MIT License — see [LICENSE](LICENSE) for details.
