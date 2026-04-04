# LocalProjections.jl

[![CI](https://github.com/gragusa/LocalProjections.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/gragusa/LocalProjections.jl/actions/workflows/CI.yml) [![codecov.io](http://codecov.io/github/gragusa/LocalProjections.jl/coverage.svg?branch=master)](http://codecov.io/github/gragusa/LocalProjections.jl?branch=master) [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) ![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826) ![lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)

Estimate local projection impulse response functions using horizon-specific linear regressions.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/gragusa/LocalProjections.jl")
```

## Example

```julia
using LocalProjections, DataFrames, StatsModels, CovarianceMatrices, Plots

# Generate sample data
n = 200
shock = randn(n)
y = cumsum(shock) + 0.5 * randn(n)  # y responds to cumulative shocks
df = DataFrame(y = y, shock = shock)

# Estimate local projections: y_{t+h} = α + β * shock_t + ε
# The coefficient β at each horizon h gives the impulse response
lp_result = lp(@formula(leads(y) ~ shock), df; horizon=20)

# Compute HAC-robust standard errors (Newey-West)
cov_result = vcov(Bartlett{NeweyWest}(), lp_result)

# Extract the impulse response function
irf = coefpath(lp_result; term=:shock)
se = stderror(cov_result; term=:shock)

# Plot IRF with 95% confidence bands
plot(lp_result, cov_result; term=:shock, levels=[0.95])
```

## Features

### Standard Local Projections
Forward-looking response at each horizon h:
```julia
lp(@formula(leads(y) ~ x + lag(z, 4)), df; horizon=12)
```

### Cumulative Responses
Accumulated effect from t to t+h:
```julia
lp(@formula(cumul(y) ~ x + lag(x, 4)), df; horizon=12)
```

### Anchored Responses
Deviation from a baseline variable (z stays fixed at time t):
```julia
# Pipe syntax
lp(@formula(leads(y)|z ~ x), df; horizon=12)

# Function syntax (equivalent)
lp(@formula(anchor(y, z) ~ x), df; horizon=12)
```

### Nested Transformations
Combine transformations as needed:
```julia
lp(@formula(cumul(log(y)) ~ x), df; horizon=12)
lp(@formula(leads(log(y))|log(baseline) ~ shock), df; horizon=12)
```

### Robust Standard Errors
Use any estimator from CovarianceMatrices.jl:
```julia
vcov(HC1(), lp_result)                    # Heteroskedasticity-robust
vcov(Bartlett{NeweyWest}(), lp_result)    # HAC with Newey-West bandwidth
vcov(Parzen{Andrews}(), lp_result)        # HAC with Andrews bandwidth
```

## API Reference

| Function | Description |
|----------|-------------|
| `lp(formula, data; horizon, shock)` | Estimate local projections |
| `coefpath(lp; term)` | Extract coefficient path across horizons |
| `vcov(estimator, lp)` | Compute robust variance-covariance |
| `stderror(cov; term)` | Extract standard errors for a term |
| `plot(lp, cov; term, levels)` | Plot IRF with confidence bands |

## Testing

Run tests from the dedicated test environment:

```julia
julia --project=test_pkg -e 'using TestItemRunner; @run_package_tests'
```

When developing `LocalProjections.jl` together with a local checkout of `Regress.jl`,
pin both packages into the test environment first:

```julia
julia --project=test_pkg -e '
using Pkg
Pkg.develop(path="/path/to/Regress.jl")
Pkg.develop(path="/path/to/LocalProjections.jl")
using TestItemRunner
@run_package_tests
'
```

The root `Pkg.test()` path can still be used, but `test_pkg` is the authoritative
workflow because it provides the full test dependency set.

## License

MIT License - see [LICENSE](LICENSE) for details.
