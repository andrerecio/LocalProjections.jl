# ============================================================================
# The implicit shock of a local projection and its variance decomposition
# (lecture 1 of Plagborg-Møller's local-projection course)
# ============================================================================

function _check_implicit(m::LocalProjection, what::String)
    m.unit === nothing || throw(
        ArgumentError("$what reads one time series; panels (`unit`) are not supported")
    )
    m.state === nothing || throw(
        ArgumentError("$what does not support state-dependent local projections")
    )
    m.sample === :fixed && throw(
        ArgumentError(
            "$what uses the rows available at each horizon, as the course's Stata code " *
                "does; re-estimate the projection with sample = :varying"
        )
    )
    _rhs_tracks_horizon(m.base_formula) && throw(
        ArgumentError(
            "$what needs the same controls at every horizon; the projection has a " *
                "horizon-tracking right-hand-side term"
        )
    )
    any(mm -> !isempty(mm.fe_groups), m.models) && throw(
        ArgumentError("$what does not support absorbed fixed effects `fe()`")
    )
    return nothing
end

# Design of the projection on every row of its data: the shock column `x`, the
# controls `C` (every other column, the intercept included), the left-hand side and
# the table.
function _implicit_design(m::LocalProjection)
    d = _lp_design(m.base_formula, m.data, 0, m.shock, false, nothing, nothing)
    X = d.Xs[1]
    j = _term_index(m, m.shock)
    return (; x = X[:, j], C = X[:, setdiff(axes(X, 2), j)], d.lhs, d.data, d.groups)
end

# Residuals of `y` on `C` over the rows where both are available; NaN elsewhere.
function _residualize(y::AbstractVector{Float64}, C::Matrix{Float64})
    rows = isfinite.(y) .& vec(all(isfinite, C; dims = 2))
    r = fill(NaN, length(y))
    any(rows) || return r
    Cr = C[rows, :]
    r[rows] = y[rows] .- Cr * (qr(Cr, ColumnNorm()) \ y[rows])
    return r
end

"""
    implicitshock(lp_result::LocalProjection) -> Vector{Float64}

The shock a local projection measures responses to (Frisch–Waugh): the residual
``\\tilde x_t`` of the regression of the impulse variable `lp_result.shock` on all
the other regressors, contemporaneous and lagged controls and the intercept,
estimated on every row where they are available, whether or not the response is.
The coefficient on ``x_t`` at horizon ``h`` is the slope of ``y_{t+h}`` on
``\\tilde x_t``.

The vector is aligned with the rows of the data passed to [`lp`](@ref) (sorted by
`time` if given), with `NaN` where the regressors are unavailable. Its standard
deviation is the typical size of a shock, which tells whether a unit impulse is
large. This is the `predict ..., resid` step of lecture 1 of Plagborg-Møller's
local-projection course; [`implicitfevd`](@ref) uses it.

Single time series, OLS, linear, `sample = :varying`, and the same regressors at
every horizon.
"""
function implicitshock(m::LocalProjection)
    _check_implicit(m, "implicitshock")
    D = _implicit_design(m)
    return _residualize(D.x, D.C)
end

function implicitshock(::LocalProjectionIV)
    throw(
        ArgumentError(
            "implicitshock is defined for OLS projections; in an IV projection the " *
                "instrument is a noisy proxy of the shock"
        )
    )
end

"""
    ImplicitFEVD

Forecast-error variance decomposition of a local projection with respect to its
implicit shock, built by [`implicitfevd`](@ref).

# Fields
- `response`, `shock`, `horizon`, `formula`;
- `share[h+1]`: the share at horizon ``h`` (an ``R^2``);
- `nobs[h+1]`: rows of the ``R^2`` regression at horizon ``h``;
- `implicit_shock`: ``\\tilde x_t`` on the rows of the data (see
  [`implicitshock`](@ref)), `shock_nobs` the rows it is estimated on;
- `forecast_errors`: ``T \\times (H+1)``, the residuals ``\\hat f_{h,t+h}`` of the
  response on the controls, `NaN` where unavailable.
"""
struct ImplicitFEVD
    response::Symbol
    shock::Symbol
    horizon::Int
    formula::FormulaTerm
    share::Vector{Float64}
    nobs::Vector{Int}
    implicit_shock::Vector{Float64}
    shock_nobs::Int
    forecast_errors::Matrix{Float64}
end

"""
    implicitfevd(lp_result::LocalProjection) -> ImplicitFEVD

Share of the forecast-error variance of the response due to the projection's
implicit shock ``\\tilde x_t`` ([`implicitshock`](@ref)), the variance
decomposition of lecture 1 of Plagborg-Møller's local-projection course, a
version of the Gorodnichenko and Lee (2020) ``R^2`` estimator. At each horizon
``h``:

1. regress the response at ``h`` (``y_{t+h}`` for `leads(y)`, or whatever the
   left-hand side defines) on the controls, every regressor but the shock, and keep
   the residual ``\\hat f_{h,t+h}``;
2. the share is the ``R^2`` of the regression of ``\\hat f_{h,t+h}`` on a
   constant and ``\\tilde x_t, \\tilde x_{t+1}, \\dots, \\tilde x_{t+h}``.

Each regression uses the rows where its variables are available. The share
answers how much the forecast error of the response, given the controls, would
shrink if the current and future shocks were known; it depends on the controls,
since they define both the shock and the information set.

Compared with [`fevd`](@ref)`(lp_result, data; estimator = :r2)`, the port of
Gorodnichenko and Lee's code: the shock is the residual ``\\tilde x_t`` rather
than the observed shock demeaned, the forecast error conditions on the
contemporaneous controls as well as the lags, and the ``R^2`` is the centred one
of a regression with a constant. With an observed shock and lagged controls only
the two estimate the same share. `implicitfevd` has no standard errors; `fevd`
and `fevdbootstrap` provide inference and bias correction for the ``R^2``
estimator.

Single time series, OLS, linear, `sample = :varying`, and the same regressors at
every horizon. An IV projection throws: with a proxy the decomposition is not
identified, and treating the proxy as the shock understates the share.

# Example
```julia
r = lp(@formula(leads(y) ~ x + rec + lags(rec, 2) + lags(y, 2) + lags(x, 2)), df;
       horizon = 10)
f = implicitfevd(r)
f.share            # h = 0, …, 10
```
"""
function implicitfevd(m::LocalProjection)
    _check_implicit(m, "implicitfevd")
    D = _implicit_design(m)
    xt = _residualize(D.x, D.C)
    T = length(xt)
    H = m.horizon
    share = fill(NaN, H + 1)
    n = zeros(Int, H + 1)
    fe = fill(NaN, T, H + 1)
    for h in 0:H
        y = _model_columns(_lhs_at(D.lhs, h), D.data, D.groups)[:, 1]
        f = _residualize(y, D.C)
        fe[:, h + 1] = f
        Xf = [t + j <= T ? xt[t + j] : NaN for t in 1:T, j in 0:h]
        rows = isfinite.(f) .& vec(all(isfinite, Xf; dims = 2))
        k = count(rows)
        k > h + 2 || throw(
            ArgumentError(
                "horizon $h leaves $k rows for the R² regression on $(h + 2) parameters"
            )
        )
        Z = hcat(ones(k), Xf[rows, :])
        fr = f[rows]
        e = fr .- Z * (qr(Z, ColumnNorm()) \ fr)
        share[h + 1] = 1 - sum(abs2, e) / sum(abs2, fr .- mean(fr))
        n[h + 1] = k
    end
    return ImplicitFEVD(
        m.response, m.shock, H, m.base_formula, share, n, xt, count(isfinite, xt), fe
    )
end

function implicitfevd(::LocalProjectionIV)
    throw(
        ArgumentError(
            "implicitfevd is defined for OLS projections: with a proxy (IV) the variance " *
                "decomposition is not identified, and treating the proxy as the shock " *
                "understates the share"
        )
    )
end

function Base.show(io::IO, f::ImplicitFEVD)
    return print(
        io, "ImplicitFEVD(horizon=0:", f.horizon, ", response=", f.response,
        ", shock=", f.shock, ")"
    )
end

function Base.show(io::IO, ::MIME"text/plain", f::ImplicitFEVD)
    println(io, "ImplicitFEVD (R² of the forecast error on the implicit shock)")
    println(io, "  Response:   ", f.response)
    println(io, "  Shock:      ", f.shock, " (residual on the controls, ", f.shock_nobs, " rows)")
    println(io, "  Formula:    ", f.formula)
    pretty_table(
        io, (; h = collect(0:(f.horizon)), s = f.share, n = f.nobs);
        column_labels = ["Horizon", "Share", "Obs"],
        title = "Forecast-error variance share of $(f.shock)",
        formatters = [(v, i, j) -> j == 2 ? round(v; digits = 4) : v],
        alignment = fill(:r, 3),
        table_format = TextTableFormat(borders = text_table_borders__unicode_rounded)
    )
    return nothing
end

@recipe function f(fv::ImplicitFEVD)
    xguide --> "horizon"
    yguide --> "share of forecast-error variance"
    label --> "$(fv.shock) → $(fv.response)"
    ylims --> (0, 1)
    seriestype --> :path
    markershape --> :circle
    return collect(0:(fv.horizon)), fv.share
end
