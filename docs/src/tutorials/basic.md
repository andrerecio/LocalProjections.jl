# Tutorial 1: Getting Started with Local Projections

This tutorial covers basic local projection estimation without transformations.

## What are Local Projections?

Local projections estimate impulse response functions by running separate
regressions at each forecast horizon ``h``:

```math
y_{t+h} = \alpha_h + \beta_h x_t + \gamma_h \text{controls}_t + \varepsilon_{t+h}
```

The sequence ``\{\beta_0, \beta_1, \ldots, \beta_H\}`` traces out the response of ``y`` to a shock in ``x``.

## Basic Example

```@example basic
using LocalProjections
using DataFrames
using StatsModels: @formula, nobs
using Random

Random.seed!(42)

# Simulate AR(1) process with shock
n = 200
shock = randn(n)
y = zeros(n)
for t in 2:n
    y[t] = 0.7 * y[t-1] + shock[t]
end

df = DataFrame(y = y, shock = shock)
nothing # hide
```

### Estimate Local Projections

```@example basic
lp_result = lp(@formula(leads(y) ~ shock), df; horizon=10)
```

### Extract the Impulse Response

```@example basic
irf = coefpath(lp_result)
println("IRF coefficients: ", round.(irf, digits=3))
```

## With Control Variables

You can include lagged controls to improve efficiency:

```@example basic
lp_controlled = lp(@formula(leads(y) ~ shock + lag(y, 1) + lag(y, 2)), df; horizon=10)
irf_controlled = coefpath(lp_controlled)
println("Controlled IRF: ", round.(irf_controlled, digits=3))
```

## Understanding the Output

The `LocalProjection` object contains:
- `models`: Vector of fitted OLS models (one per horizon)
- `horizon`: Maximum horizon
- `response`: Response variable name
- `shock`: Shock variable name
- `coef_names`: Coefficient names for each horizon
- `sample`: `:horizon` or `:common` (see [Estimation Sample](#Estimation-Sample))

```@example basic
println("Response: ", lp_result.response)
println("Shock: ", lp_result.shock)
println("Horizons: 0 to ", lp_result.horizon)
println("Coefficients: ", lp_result.coef_names[1])
```

## Specifying the Shock Variable

By default, `lp()` uses the first RHS coefficient (after intercept) as the shock.
You can specify a different variable:

```@example basic
lp_explicit = lp(@formula(leads(y) ~ shock + lag(y, 1)), df;
                 horizon=5, shock=:shock)
nothing # hide
```

!!! note
    The `shock` parameter must match the coefficient name, not the variable name.
    For `lag(x)`, the coefficient name is `"x_lag1"`.

## Estimation Sample

Each horizon is a separate regression, and the response ``y_{t+h}`` is not
observed for the last ``h`` periods. By default (`sample = :horizon`) every
horizon uses all the observations available to it, so the estimation sample
shrinks by one observation per horizon. This uses all the data and is the more
efficient choice.

With `sample = :common` every horizon is estimated on the same observations:
those complete at every horizon ``0, \dots, H``. Without internal gaps this is
the sample of the maximum horizon, so horizon 0 loses ``H`` observations; an
internal `NaN` in the response removes every row whose lead hits it at some
horizon, so the common sample is then smaller than the horizon-``H`` one. The
impulse response is not affected by a sample that changes along the path.

```@example basic
lp_byh = lp(@formula(leads(y) ~ shock + lag(y, 1)), df; horizon = 10)
lp_com = lp(@formula(leads(y) ~ shock + lag(y, 1)), df; horizon = 10, sample = :common)
(nobs.(lp_byh.models), nobs.(lp_com.models))
```

`lpiv` takes the same keyword, and the choice is stored in the result
(`lp_com.sample`). `varbootstrap` re-estimates every draw with it, while
`biascorrect` and `fevd` require `sample = :horizon`.
