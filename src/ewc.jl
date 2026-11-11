# ============================================================================
# EWC bandwidth (fixed-smoothing HAR inference)
# The critical value paired with a covariance estimator, Student-t_B for `EWC(B)`,
# is upstream's `critical_value` in the main file; every confidence bound goes
# through `_irf_result`, which asks it.
# ============================================================================

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
