# LocalProjections.jl

[![CI](https://github.com/andrerecio/LocalProjections.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/andrerecio/LocalProjections.jl/actions/workflows/CI.yml) [![codecov.io](http://codecov.io/github/andrerecio/LocalProjections.jl/coverage.svg?branch=main)](http://codecov.io/github/andrerecio/LocalProjections.jl?branch=main) [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) ![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826) ![lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)

Impulse response functions by local projections (Jordà, 2005): horizon-specific linear regressions with a formula interface, robust and HAR inference, small-sample bias correction, bootstrap confidence bands, and instrumental-variable support.

## Installation

This fork of [gragusa/LocalProjections.jl](https://github.com/gragusa/LocalProjections.jl)
extends the inference procedures: EWC HAR inference, the Herbst–Johannsen bias
correction, and VAR residual moving-block bootstrap bands with VAR lag
selection.

```julia
using Pkg
Pkg.add(url = "https://github.com/andrerecio/LocalProjections.jl")
```

## Quick start

```julia
using LocalProjections, DataFrames, StatsModels, CovarianceMatrices, Plots, Random

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

### Bias correction

`biascorrect` applies the Herbst–Johannsen (2024) small-sample correction to
the impulse-response path (the "BCC" estimator). The persistence of the
controls is estimated from the horizon-zero regression, and each horizon is
corrected recursively through the already-corrected lower horizons:

```julia
bc = biascorrect(lp_result)         # OLS local projections only
coefpath(bc)                        # corrected θ̂ᶜ path (θ̂₀ unchanged)
summarize(bc, HC1(); level = 0.90)  # bands centered on θ̂ᶜ
```

Following the reference implementation (Montiel Olea, Plagborg-Møller, Qian &
Wolf, 2025), the confidence bands are centered on the corrected path but use
the standard errors of the *uncorrected* coefficients. `plot` and
`as_irf_result` accept the corrected result anywhere a plain one is accepted.

## Lag selection

`lagselect` chooses the lag order of a reduced-form VAR by an information
criterion, following `ic_var.m` in the Montiel Olea, Plagborg-Møller, Qian &
Wolf (2025) replication suite. A single order is selected for the whole system
and used both for the local-projection controls and for the bootstrap VAR:

```julia
sel = lagselect(df, [:shock, :y]; maxlags = 10, criterion = :aic)
nlags(sel)          # selected p
DataFrame(sel)      # the full AIC/BIC path over p = 1:maxlags
```

`@formula` takes a literal lag count, so the selected `p` cannot be spliced in
programmatically — read it off `nlags(sel)` and write it into the formula
yourself, then pass the same value as `nlags` to [`varbootstrap`](#bootstrap-inference)
so the controls and the bootstrap VAR agree.

With `n` variables and `T` observations,

```
AIC(p) = logdet Σ̂ₚ + 2(n²p + n) / (T − p_max)
BIC(p) = logdet Σ̂ₚ + (n²p + n)·log(T − p_max) / (T − p_max)
```

Both criteria are always computed; `criterion` only decides which one is
minimized. Two conventions are inherited deliberately from the reference: the
penalty denominator is the fixed `T − p_max` for every `p`, so the criteria are
comparable across the grid, while `Σ̂ₚ` is estimated on the full sample. The
grid starts at `p = 1`.

## Bootstrap inference

`varbootstrap` implements the VAR residual moving-block bootstrap with Hall
percentile-`t` bands. A reduced-form VAR is fitted and used as the bootstrap
DGP; in each draw its residuals are resampled in overlapping blocks with
position-specific recentering (Brüggemann–Jentsch–Trenkler), the VAR is
iterated from randomly drawn contiguous initial conditions, and **the complete
local projection is re-estimated** on the artificial sample:

```julia
# `nlags(sel)` was 4 here, so the formula uses lags(·, 4) to match.
m = lp(@formula(leads(y) ~ shock + lags(y, 4) + lags(shock, 4)), df; horizon = 20)

b = varbootstrap(m, df; vars = [:shock, :y], nlags = 4,
                 nboot = 1000, rng = Xoshiro(20260831))

summarize(b; level = 0.90)              # Hall percentile-t bands
plot(b; levels = [0.68, 0.90])
```

`vars` is the VAR data vector in identification order: it must contain the
shock and every variable the LP formula refers to, since the LP is rebuilt in
each draw. The block length defaults to `ceil(5.03·T^(1/4))`, the reference
rule.

By default both analytical corrections are applied, which together give the
procedure the reference recommends:

  - `popecorrect = true` — Pope (1990) bias-corrected VAR slopes as the
    bootstrap DGP (the intercept and residuals stay the OLS ones);
  - `biascorrect = true` — the Herbst–Johannsen correction applied to the
    real-data path *and* to every bootstrap path.

Setting both to `false` reproduces the simpler uncorrected variant.

Three centers are involved and should not be confused:

| Object | Center used |
| --- | --- |
| Reported response | bias-corrected LP, `θ̂ᶜ` |
| Bootstrap responses | bias-corrected bootstrap LP |
| Center of the bootstrap `t`-statistic | Pope-corrected VAR pseudo-truth |

Centering the statistic at the VAR-implied response, not at the real-data LP
estimate, is essential: the fitted VAR is what generates the bootstrap samples.
`summarize` also accepts `method = :hall` or `method = :efron` for the other
two constructions in `boot_ci.m`.

Bands are **pointwise across horizons, not simultaneous**, and are generally
asymmetric around the point estimate. They may also fail to *contain* it: the
percentile-`t` and Hall constructions both correct for bias, so a shifted
bootstrap `t`-distribution can push the whole interval to one side of `θ̂`. In
the Monte Carlo study under `benchmark/coverage_study.jl` this happens in at
most 0.3% of horizon-replications — rare, but not impossible.

Draws in which the LP cannot be estimated are counted in `b.nfail` and excluded
from the quantiles. Because the Pope correction carries stability safeguards it
may be attenuated or skipped entirely — inspect `b.pope_delta` (`1.0` full,
`0.0` skipped). On near-unit-root data the full correction applies in only
about three quarters of samples.

Per-draw seeds are drawn from `rng` before the loop, so results are
reproducible and identical whether or not `threaded = true` is passed.

## Performance

The bootstrap re-estimates the whole local projection in every draw, which on a
typical macro sample (T ≈ 240, horizon 20) costs roughly 0.9 ms per draw — about
a second for 1000 draws. The cost is dominated by rebuilding the StatsModels
design each draw, not by linear algebra: the BLAS content of a draw is only a
few percent of its runtime on matrices this small. Swapping the BLAS backend
(for example with AppleAccelerate.jl on macOS) therefore does not help, and was
measured to be marginally slower here; it is available in the benchmark
environment behind `LP_ACCELERATE=1` for anyone who wants to re-check on
different hardware.

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
