# ============================================================================
# Critical values (fixed-smoothing HAR inference)
# Every confidence bound goes through `_irf_result`, which asks `_critical_value`
# for the critical value paired with the stored covariance estimator.
# ============================================================================

"""
    _critical_distribution(estimator)

Reference distribution used to build confidence-interval critical values for
a given covariance estimator.

Defaults to `Normal()`. For the equal-weighted cosine estimator `EWC(B)`,
returns `TDist(B)`: under fixed-smoothing asymptotics the EWC long-run
variance is paired with Student-``t_B`` critical values for a single
restriction (Lazarus, Lewis, Stock & Watson, 2018). Pairing EWC with normal
critical values would understate the band width in finite samples.
"""
_critical_distribution(::Any) = Normal()
_critical_distribution(estimator::EWC) = TDist(estimator.B)

"""
    _critical_value(estimator, level::Real)

Two-sided critical value at confidence `level` from the reference
distribution paired with `estimator` (see `_critical_distribution`).
"""
function _critical_value(estimator, level::Real)
    return quantile(_critical_distribution(estimator), 0.5 + level / 2)
end

"""
    ewc_bandwidth(T₀::Integer) -> Int
    ewc_bandwidth(lp::AbstractLocalProjection) -> Int

Number of cosine terms ``B`` for the equal-weighted cosine (EWC) long-run
variance estimator, using the Lazarus–Lewis–Stock–Watson (2018) rule for a
single restriction:

```math
B = \\lfloor 0.41\\,T_0^{2/3} \\rfloor
```

For a `LocalProjection`/`LocalProjectionIV`, ``T_0`` is the number of observations
of the horizon-zero regression (with `sample = :fixed`, the common sample); the
resulting integer is meant to be held fixed across horizons.

Bands built on `lp + vcov(EWC(B))` use Student-``t_B`` critical values. The EWC
covariance carries the finite-sample factor ``n/(n-k)`` that Regress.jl applies to
every long-run variance estimator, so standard errors are ``\\sqrt{n/(n-k)}`` times
those of Stata's `harreg`, which applies none; the critical values are the same.

# Example
```julia
r = lp(@formula(leads(y) ~ x + lags(y, 4)), df; horizon = 20)
B = ewc_bandwidth(r)
summarize(r + vcov(EWC(B)); level = 0.90)  # Student-t_B critical values
```
"""
ewc_bandwidth(T0::Integer) = max(1, floor(Int, 0.41 * T0^(2 / 3)))
ewc_bandwidth(lp::AbstractLocalProjection) = ewc_bandwidth(first(nobs(lp)))
