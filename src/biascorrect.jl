# ============================================================================
# Herbst–Johannsen bias correction (BCC)
# ============================================================================

"""
    _persistence_adjustments(w, H) -> Vector{Float64}

Persistence adjustments ``\\hat c_j = 1 + \\mathrm{tr}(\\hat\\Sigma_0^{-1}
\\hat\\Sigma_j)`` for ``j = 1, \\dots, H`` from the ``T \\times k`` control
matrix `w` (guide §3.1), with ``\\hat\\Sigma_0 = \\sum_t \\tilde w_t
\\tilde w_t' / (T-1)`` and ``\\hat\\Sigma_j = \\sum_{t \\le T-j} \\tilde
w_t \\tilde w_{t+j}' / (T-j-1)`` computed from the demeaned controls.
With no controls (`k == 0`) every trace term vanishes and ``\\hat c_j = 1``.
"""
function _persistence_adjustments(w::AbstractMatrix{<:Real}, H::Int)
    T, k = size(w)
    c = ones(H)
    k == 0 && return c
    T > H + 1 || throw(ArgumentError(
        "estimating the lag-$H control autocovariance needs more than " *
        "H + 1 = $(H + 1) horizon-zero observations, got $T"))
    wt = w .- mean(w; dims = 1)
    Sigma0 = Symmetric(wt' * wt / (T - 1))
    kappa = cond(Sigma0)
    (isfinite(kappa) && kappa < 1e12) || throw(ArgumentError(
        "the control covariance Σ̂₀ is (near-)singular (cond ≈ $(round(kappa, sigdigits = 3))); " *
        "drop collinear controls before bias-correcting"))
    F = factorize(Sigma0)
    for j in 1:H
        Sigmaj = view(wt, 1:(T - j), :)' * view(wt, (j + 1):T, :) / (T - j - 1)
        c[j] = 1 + tr(F \ Sigmaj)
    end
    return c
end

"""
    biascorrect(lp::LocalProjection) -> BiasCorrectedLP

Herbst–Johannsen small-sample bias correction of the impulse-response path
(guide §3; "BCC" in Herbst & Johannsen, 2024).

The control vector ``w_t`` is taken from the horizon-zero model matrix,
selecting columns by coefficient name and excluding only the intercept and
the contemporaneous shock. From its variance and lag autocovariances the
persistence adjustments ``\\hat c_j = 1 +
\\mathrm{tr}(\\hat\\Sigma_0^{-1}\\hat\\Sigma_j)`` are formed, and the
corrected path is built recursively through already-corrected lower
horizons:

```math
\\hat\\theta_0^{\\,c} = \\hat\\theta_0, \\qquad
\\hat\\theta_h^{\\,c} = \\hat\\theta_h + \\frac{1}{T_0 - h}
\\sum_{j=1}^{h} \\hat c_j \\, \\hat\\theta_{h-j}^{\\,c}.
```

The ``1/(T_0 - h)`` weights (and the lag-``j`` pairing in
``\\hat\\Sigma_j``) assume the only horizon-related sample loss is the
``h`` future observations. This is verified structurally: the horizon-``h``
design matrix must equal the horizon-zero design truncated by its last
``h`` rows, otherwise an error is thrown. One case is not detectable from
the fitted object alone: an internal `NaN` gap in an RHS-only variable
drops the same rows at every horizon, leaving the truncation pattern
intact while the retained rows are no longer contiguous in time — the
lag-``j`` autocovariance pairing is then misaligned at the gap. The
correction therefore additionally assumes the horizon-zero estimation
sample is a contiguous block of the original time axis.

A projection estimated with `sample = :common` is rejected: its horizons share
one sample instead of losing ``h`` observations each, which is not the setting
the ``1/(T_0 - h)`` correction is derived for.

Only defined for OLS local projections — the correction is derived for
least-squares LP coefficients, not for IV estimators.

# Example
```julia
lp_result = lp(@formula(leads(y) ~ x + lags(y, 4)), df; horizon = 20)
bc = biascorrect(lp_result)
coefpath(bc)                       # corrected θ̂ᶜ path
summarize(bc, HC1(); level = 0.90) # bands centered on θ̂ᶜ, SEs of θ̂
```
"""
function biascorrect(lp_result::LocalProjection)
    lp_result.state === nothing || throw(ArgumentError(
        "the Herbst–Johannsen correction is derived for a linear projection and " *
        "does not apply to a state-dependent one"))
    lp_result.sample === :common && throw(ArgumentError(
        "the Herbst–Johannsen correction is derived for horizon-specific samples, " *
        "where the horizon-h regression uses T₀ − h observations; the projection " *
        "was estimated with sample = :common, re-estimate it with sample = :horizon"))
    _rhs_tracks_horizon(lp_result.base_formula) && throw(ArgumentError(
        "the Herbst–Johannsen correction assumes the same regressors at every " *
        "horizon, but the formula has a horizon-tracking right-hand-side term " *
        "(a bare `cumul(x)` or `leads(x)`), so the design changes with h"))
    _lhs_kind(lp_result.base_formula) === :longdiff && throw(ArgumentError(
        "the Herbst–Johannsen correction is derived for a response in levels, " *
        "y_{t+h} or its cumulative sum; it does not cover the long-difference " *
        "responses ldiff() and hbr()"))
    H = lp_result.horizon
    m0 = lp_result.models[1]
    T0 = Int(nobs(m0))
    X0 = modelmatrix(m0)
    for h in 1:H
        mh = lp_result.models[h + 1]
        Th = Int(nobs(mh))
        (Th == T0 - h && modelmatrix(mh) == view(X0, 1:Th, :)) ||
            throw(ArgumentError(
                "the horizon-$h estimation sample is not the horizon-0 sample " *
                "truncated by its last $h rows (horizon-specific missingness), " *
                "so the Herbst–Johannsen 1/(T₀ − h) weights and lag-j " *
                "autocovariance pairing do not apply"))
    end
    keep = [!(n == "(Intercept)" || n == String(lp_result.shock))
            for n in lp_result.coef_names]
    w = modelmatrix(m0)[:, keep]
    c = _persistence_adjustments(w, H)
    return BiasCorrectedLP(lp_result, c, T0, lp_result.coef_names[keep])
end

function biascorrect(::LocalProjectionIV)
    throw(ArgumentError(
        "the Herbst–Johannsen bias correction is derived for OLS local " *
        "projections and is not available for LocalProjectionIV"))
end

"""
    coefpath(bc::BiasCorrectedLP; term=bc.shock) -> Vector{Float64}

Bias-corrected impulse-response path ``\\hat\\theta_h^{\\,c}`` (see
[`biascorrect`](@ref)). The correction is defined for the impulse response
of the shock only; request other terms from the uncorrected result via
`coefpath(bc.lp; term = ...)`.
"""
function coefpath(bc::BiasCorrectedLP; term::Symbol = bc.lp.shock)
    term === bc.lp.shock || throw(ArgumentError(
        "the Herbst–Johannsen correction applies to the impulse response " *
        "of the shock ($(bc.lp.shock)); use coefpath(bc.lp; term = :$term) " *
        "for the uncorrected path of another term"))
    theta = coefpath(bc.lp; term = term)
    corrected = copy(theta)
    for h in 1:bc.lp.horizon
        acc = 0.0
        for j in 1:h
            acc += bc.c[j] * corrected[h - j + 1]
        end
        corrected[h + 1] = theta[h + 1] + acc / (bc.T0 - h)
    end
    return corrected
end

"""
    vcov(estimator, bc::BiasCorrectedLP; kwargs...)

Covariance of the *uncorrected* per-horizon OLS coefficients (the
reference convention pairs the corrected path with the ordinary LP
standard errors; see [`BiasCorrectedLP`](@ref)).
"""
function vcov(estimator::CovarianceMatrices.AbstractAsymptoticVarianceEstimator,
        bc::BiasCorrectedLP; kwargs...)
    return vcov(estimator, bc.lp; kwargs...)
end

"""
    bc + vcov(estimator)

Apply a covariance estimator to the wrapped models, keeping the bias
correction (consistent with `lp + vcov(estimator)`).
"""
function Base.:+(bc::BiasCorrectedLP, v::VcovSpec)
    return BiasCorrectedLP(bc.lp + v, bc.c, bc.T0, bc.controls)
end

_maybe_biascorrect(m::LocalProjection, apply::Bool) = apply ? biascorrect(m) : m
