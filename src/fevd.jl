"""
    _GLSpec

A `fevd` formula compiled against the data schema: the response transform
(`lhs`, `:cumul` or `:ldiff`) and its variable (`response`), the applied
left-hand-side term, the contemporaneous regressors (shock first) with their
applied terms, the applied information-set terms (intercept and `lags()`), the
polynomial trend order, and every raw column the specification reads. Building
the design from it (`_gl_design`) needs only a column table, so the
same specification can be re-estimated on simulated data.
"""
struct _GLSpec
    lhs::Symbol
    response::Symbol
    lhs_term::AbstractTerm
    contemporaneous::Vector{Symbol}
    cterms::Vector{AbstractTerm}
    info::Vector{AbstractTerm}
    trend::Int
    variables::Vector{Symbol}
end

"""
    _GLDesign

Matrices of a `fevd` specification on the original time axis (``T`` rows):
`dhy` (``T \\times (H+1)``, ``y_{t+h} - y_{t-1}``), `C` (contemporaneous
regressors, demeaned shock first), `W` (information set ``W_{t-1}``: the
intercept, the `lags()` columns and the trend ``t, t^2, t^3`` on the absolute
row index), `q` (separate impulse-response regressors, or `nothing`) and `x`
(the shock demeaned with its full-sample mean). Rows before the effective
start hold `NaN` in the lagged columns.
"""
struct _GLDesign
    dhy::Matrix{Float64}
    C::Matrix{Float64}
    W::Matrix{Float64}
    q::Union{Nothing, Matrix{Float64}}
    x::Vector{Float64}
end

const _FEVD_GRAMMAR = "fevd formulas have the form `cumul(v) ~ x + ... + lags(w, p) + " *
                      "lags(firstdiff(z), p)`: plain variables enter at date t, `lags()` of a " *
                      "variable or of its `firstdiff()` form the information set at t - 1"

function _gl_parse(formula::FormulaTerm, data::AbstractDataFrame,
        shock::Union{Symbol, Nothing}, trend::Int, what::String)
    vars = unique(StatsModels.termvars(formula))
    for v in vars
        hasproperty(data, v) ||
            throw(ArgumentError("variable $v in `$what` is not a column of the data"))
        eltype(data[!, v]) <: Union{Missing, Real} || throw(ArgumentError(
            "column $v used in `$what` must be numeric (got eltype $(eltype(data[!, v])))"))
    end
    hints = Dict{Symbol, Any}(v => StatsModels.ContinuousTerm for v in vars)
    sch = StatsModels.schema(formula, data, hints)
    f = StatsModels.apply_schema(formula, sch, StatisticalModel)

    lhs_term = f.lhs
    lhs, inner = if lhs_term isa CumulTerm && !(lhs_term isa CumulTerm{<:LongDiffTerm})
        :cumul, lhs_term.term
    elseif lhs_term isa LongDiffTerm && lhs_term.scale === nothing
        :ldiff, lhs_term.term
    else
        throw(ArgumentError(
            "the left-hand side of `$what` must be cumul(v) (v a first difference, " *
            "the response is its level) or ldiff(y) (y in levels); got $(formula.lhs)"))
    end
    inner isa StatsModels.ContinuousTerm || throw(ArgumentError(
        "the left-hand side of `$what` must transform a plain variable; got $(formula.lhs)"))

    rhs = f.rhs isa StatsModels.MatrixTerm ? f.rhs.terms : (f.rhs,)
    intercept = false
    cterms = AbstractTerm[]
    info = AbstractTerm[]
    for t in rhs
        if t isa StatsModels.InterceptTerm{true}
            intercept = true
            push!(info, t)
        elseif t isa StatsModels.InterceptTerm{false}
            continue
        elseif t isa StatsModels.ContinuousTerm
            push!(cterms, t)
        elseif t isa LagTerm && t.nsteps >= 1 &&
               (t.term isa StatsModels.ContinuousTerm ||
                (t.term isa DiffTerm && t.term.term isa StatsModels.ContinuousTerm))
            push!(info, t)
        else
            throw(ArgumentError("unsupported term $t in `$what`; " * _FEVD_GRAMMAR))
        end
    end
    intercept || throw(ArgumentError(
        "`$what` must include an intercept: the forecast-error regressions have a constant"))
    csyms = Symbol[t.sym for t in cterms]
    allunique(csyms) ||
        throw(ArgumentError("`$what` lists a contemporaneous variable twice"))
    isempty(csyms) && throw(ArgumentError(
        "`$what` has no contemporaneous variable: the shock must enter as a plain " *
        "right-hand-side variable (its value at date t)"))
    s = shock === nothing ? csyms[1] : shock
    k = findfirst(isequal(s), csyms)
    k === nothing && throw(ArgumentError(
        "the shock $s is not a contemporaneous (plain) variable of `$what`; " *
        "contemporaneous variables: $(join(csyms, ", "))"))
    order = vcat(k, setdiff(eachindex(csyms), k))
    return _GLSpec(lhs, inner.sym, lhs_term, csyms[order], cterms[order], info, trend,
        Symbol.(vars))
end

# The raw columns of `vars` as a column table of Float64 vectors. `missing` and
# NaN are rejected: lags and leads are built on the original time axis and the
# reference has no notion of a gap (this also sidesteps issue LP-1).
function _gl_columns(data::AbstractDataFrame, vars::AbstractVector{Symbol})
    cols = map(vars) do v
        col = data[!, v]
        eltype(col) <: Union{Missing, Real} ||
            throw(ArgumentError("column $v must be numeric"))
        any(ismissing, col) && throw(ArgumentError(
            "column $v contains `missing`; fevd requires a gap-free sample"))
        c = Vector{Float64}(col)
        all(isfinite, c) || throw(ArgumentError(
            "column $v contains NaN or Inf; fevd requires a gap-free sample"))
        c
    end
    return NamedTuple{Tuple(vars)}(Tuple(cols))
end

# [C, W] (or [C, W] of the impulse-response set) on the original time axis.
# The shock column of C is replaced by the demeaned shock `x`.
function _gl_regressors(spec::_GLSpec, d::NamedTuple, x::Vector{Float64}, T::Int)
    C = Matrix{Float64}(undef, T, length(spec.cterms))
    C[:, 1] = x
    for (j, t) in enumerate(spec.cterms)
        j == 1 && continue
        C[:, j] = _float_column(t, d, "contemporaneous regressor")
    end
    blocks = Matrix{Float64}[_as_float_matrix(StatsModels.modelcols(t, d))
                             for t in spec.info]
    for k in 1:(spec.trend)
        push!(blocks, reshape(Float64.(1:T) .^ k, T, 1))
    end
    return C, reduce(hcat, blocks)
end

"""
    _gl_design(spec, irf_spec, d, H) -> _GLDesign

Build the matrices of a `fevd` specification from the column table `d`
(see `_GLDesign`).
"""
function _gl_design(spec::_GLSpec, irf_spec::Union{Nothing, _GLSpec}, d::NamedTuple,
        H::Int)
    xraw = d[spec.contemporaneous[1]]
    T = length(xraw)
    x = xraw .- mean(xraw)
    C, W = _gl_regressors(spec, d, x, T)
    q = irf_spec === nothing ? nothing : hcat(_gl_regressors(irf_spec, d, x, T)...)
    dhy = Matrix{Float64}(undef, T, H + 1)
    for h in 0:H
        dhy[:, h + 1] = Vector{Float64}(StatsModels.modelcols(_lhs_at(spec.lhs_term, h), d))
    end
    return _GLDesign(dhy, C, W, q, x)
end

# First row at which every regressor and the h = 0 response exist (at least 2,
# so that y_{t-1} exists); every horizon then uses the contiguous rows t0..T-h.
function _gl_start(R::Matrix{Float64}, dhy::Matrix{Float64}, what::String)
    T = size(R, 1)
    ok(t) = isfinite(dhy[t, 1]) && all(isfinite, view(R, t, :))
    t0 = findfirst(ok, 1:T)
    t0 === nothing && throw(ArgumentError("`$what` has no complete observation"))
    t0 = max(t0, 2)
    all(ok, t0:T) || throw(ArgumentError(
        "the regressors of `$what` are not complete on the contiguous rows $t0:$T"))
    return t0
end

"""
    LPFEVD

Forecast-error variance decomposition of Gorodnichenko and Lee (2020), built
by [`fevd`](@ref). `share[h+1]` is the estimated share ``\\hat s_h`` of the
variance of the forecast error of ``y_{t+h} - y_{t-1}`` (given information at
``t-1``) due to the shock, ``h = 0, \\dots,`` `horizon`.

# Fields
- `estimator` (`:r2`, `:lpa`, `:lpb`), `horizon`, `response` (the variable in
  `cumul(v)`/`ldiff(y)`), `shock`, `formula`, `irf_formula` (`nothing` unless a
  separate impulse-response regression was given), `contemporaneous` (the
  date-``t`` regressors, shock first);
- sample: `T` (rows of the data), `t0` (first regression row), `t0lp` (first
  row of the impulse-response regression; `t0` without `irf_formula`),
  `nobs[h+1]` ``= T_h = T - h - t_0 + 1``, `window[h+1, :]` ``= [t_0, T-h]``,
  `bandwidth[h+1]` (Newey–West lag of the asymptotic variance, computed on its
  window of ``n_h`` rows, see [`fevd`](@ref)), `dofadjust`, `trend`;
- estimates: `share`, `irf` (``\\hat\\psi_h``, response to a unit shock;
  `nothing` for `:r2`), `irf_se` (its Newey–West standard error), `shock_mean`
  and `shock_var` (full-sample mean and variance, ``T-1`` denominator, of the
  shock), `mse` (``\\hat\\theta_3`` for R², ``\\hat\\sigma^2_{f,h}`` for LP-A,
  the denominator ``m_h`` for LP-B), `sigma2v` (``\\hat\\sigma^2_{v,h}``, LP-B
  only), `forecast_errors` (``T \\times (H+1)``, `NaN` outside the window);
- asymptotic inference: `se_pw` and `se`, the delta-method standard errors of
  ``\\hat s_h`` with and without VAR(1) prewhitening of the moments, `theta[h+1]`
  (the moment estimates ``\\hat\\theta_h``) and `avar_pw[h+1]`, `avar[h+1]`
  (their covariance matrices ``G^{-1}\\hat\\Omega G^{-\\prime}/n_h``, with
  ``n_h = T_h`` except for LP-A/B with a later-starting `irf_formula`).

The remaining fields (`lhs`, `variables`, `spec`, `irf_spec`, `design`) hold the
compiled specification and its matrices, so that the bias corrections of
`fevdbootstrap` need not rebuild them.
"""
struct LPFEVD
    estimator::Symbol
    horizon::Int
    response::Symbol
    shock::Symbol
    formula::FormulaTerm
    irf_formula::Union{Nothing, FormulaTerm}
    contemporaneous::Vector{Symbol}
    T::Int
    t0::Int
    t0lp::Int
    nobs::Vector{Int}
    window::Matrix{Int}
    bandwidth::Vector{Int}
    dofadjust::Bool
    trend::Int
    share::Vector{Float64}
    irf::Union{Nothing, Vector{Float64}}
    irf_se::Union{Nothing, Vector{Float64}}
    shock_mean::Float64
    shock_var::Float64
    mse::Vector{Float64}
    sigma2v::Union{Nothing, Vector{Float64}}
    forecast_errors::Matrix{Float64}
    se_pw::Vector{Float64}
    se::Vector{Float64}
    theta::Vector{Vector{Float64}}
    avar_pw::Vector{Matrix{Float64}}
    avar::Vector{Matrix{Float64}}
    lhs::Symbol
    variables::Vector{Symbol}
    spec::_GLSpec
    irf_spec::Union{Nothing, _GLSpec}
    design::_GLDesign
end

"""
    fevd(formula, data; horizon, estimator = :r2, shock = nothing,
         irf_formula = nothing, dofadjust = true, trend = 0) -> LPFEVD
    fevd(lp_result::LocalProjection, data; estimator = :r2, irf_formula = nothing,
         dofadjust = true, trend = 0) -> LPFEVD

Share of the forecast-error variance of a variable due to an observed shock,
by the local-projection estimators of Gorodnichenko and Lee (2020): `:r2` (the
``R^2`` estimator of the paper's main text), `:lpa` and `:lpb`. A port of
`SubR2.m`, `SubLPA.m` and `SubLPB.m` with their asymptotic variances
(`SimulationAVar*.m`, horizon by horizon).

The target is the forecast error of ``y_{t+h} - y_{t-1}`` given information at
``t-1``. It is estimated by regressing ``y_{t+h} - y_{t-1}`` on the information
set ``W_{t-1}`` on the rows ``t = t_0, \\dots, T-h`` (``T_h = T-h-t_0+1``).

- **R²**: ``\\hat s_h`` is the uncentred ``R^2`` of the forecast error
  ``\\hat f_{t+h|t-1}`` on the shocks ``x_t, \\dots, x_{t+h}``, demeaned with their
  full-sample mean. It lies in ``[0, 1]``.
- **LP-A**: ``\\hat s_h = \\hat\\sigma^2_x \\sum_{i \\le h}\\hat\\psi_i^2 /
  \\hat\\sigma^2_{f,h}``, with ``\\hat\\psi_i`` the local-projection response to a
  unit shock and ``\\hat\\sigma^2_{f,h}`` the forecast-error variance. Not bounded
  by 1.
- **LP-B**: the denominator is replaced by
  ``\\hat\\sigma^2_x \\sum_{i\\le h}\\hat\\psi_i^2 + \\hat\\sigma^2_{v,h}``, with
  ``\\hat v`` the forecast error net of ``\\sum_{i\\le h}\\hat\\psi_i x_{t+h-i}``.
  It lies in ``[0, 1]``.

# Formula
The formula describes both the local projection and the information set:

```julia
fevd(@formula(cumul(dy) ~ tfp + lags(tfp, 4) + lags(dy, 4) + lags(infl, 4)), df;
     horizon = 20, estimator = :lpb)
```

- left-hand side: `cumul(v)` (with ``v`` a first difference, the response is its
  level: ``\\sum_{j\\le h} v_{t+j} = y_{t+h} - y_{t-1}``) or `ldiff(y)` (``y`` in
  levels);
- plain variables enter at date ``t`` (the reference's `XtIndicator`); the shock
  is the first of them, or the `shock` keyword. For LP-A/B the others are
  controls whose innovations, purged of ``W_{t-1}``, stay in the forecast error;
  the R² estimator ignores them, as `SubR2.m` does;
- `lags(v, p)` and `lags(firstdiff(v), p)` form ``W_{t-1}``, together with the
  required intercept and, with `trend = k ∈ 0:3`, the powers ``t, \\dots, t^k``
  of the absolute row index ``t = 1, \\dots, T``;
- anything else (`leads`, `anchor`, `hbr`, functions, interactions) throws.

Lags and leads are built on the original rows of `data`, which must have no
`missing` or `NaN` in the columns used; ``t_0 \\ge 2`` is the first row at which
every regressor exists.

`fevd(lp_result, data; ...)` reuses the projection's formula
(`lp_result.base_formula`), `horizon` and `shock`, so `horizon` and `shock` are
not accepted as keywords. The projection must be linear (no `state`) and OLS
(not `lpiv`), with a `cumul(v)` or `ldiff(y)` left-hand side and a right-hand
side of the form above.

# Keywords
- `horizon`: maximum horizon ``H``.
- `irf_formula` (LP-A/B): a separate, typically parsimonious, regression for
  ``\\hat\\psi_h`` with the same left-hand side and shock, e.g.
  `@formula(cumul(dy) ~ tfp + lags(tfp, 1) + lags(firstdiff(y), 1))` (the
  paper's set: the lagged response is formed from the level ``y``, so it starts
  at ``t_0^{LP} = 3`` as in the reference); the forecast errors
  still come from `formula`. The LP-A/B asymptotic variance stacks the
  impulse-response coefficients of horizons ``0, \\dots, h``, so it needs such a
  parsimonious regression: with the full `formula` the prewhitened variance is
  not computable beyond short horizons (see below).
- `dofadjust = true`: LP-A/B variances divided by ``T_h - k_W``, ``k_W`` the
  number of columns of ``W_{t-1}`` (contemporaneous regressors excluded), as in
  the paper (`dfIndicator = 0`); `false` divides by ``T_h``.
- `trend = 0`: polynomial trend order.

# Conventions
The shock mean and variance (``T-1`` denominator) are computed over all ``T``
rows. `irf_se` is a Newey–West standard error with lag
``\\mathrm{round}(0.75\\,T_h^{1/3} - 1)`` and no degrees-of-freedom factor.
`se_pw` / `se` are delta-method standard errors at ``\\hat s_h``, from the
moment conditions of Appendices A/B with Bartlett long-run variances (same lag)
of the moments with / without VAR(1) prewhitening; the R² moments treat the
forecast error as data (valid when the shock is unpredictable from
``W_{t-1}``). The prewhitened variance is `NaN`, with a warning, when the
moment vector (``2h+3`` for R², ``N_q(h+1)+2`` for LP-A/B, ``N_q`` the
impulse-response regressors) has at least ``n_h - 1`` elements, ``n_h`` the rows
of the asymptotic-variance window: ``n_h = T_h`` for R² and for LP-A/B without
`irf_formula`, ``n_h = T - h - \\max(t_0, t_0^{LP}) + 1`` (the rows shared by
the forecast-error and impulse-response regressions) for LP-A/B with an
`irf_formula` that starts later. For LP-A/B the bandwidth and the scaling of
`avar` use the same ``n_h``. `se_pw`, `se`, `avar_pw` and `avar` of a horizon
are `NaN`, with a warning, when its impulse-response regression fits exactly
(e.g. the response is a contemporaneous regressor); `irf` and `irf_se` keep
their computed values there (numerically zero, as in the reference).
Regressions use QR rather than the reference's normal equations. When a
horizon fits exactly, this changes the prewhitened variance at all later
horizons materially, not just at rounding level: the exact-fit moments stay in
the stacked moment vector as rounding noise, which the reference's
normal-equation prewhitening amplifies while QR does not (see the tutorial's
decision table).

Bias corrections and confidence bands: `fevdbootstrap`.

# Reference
Gorodnichenko, Y. and B. Lee (2020), "Forecast Error Variance Decompositions
with Local Projections", *Journal of Business & Economic Statistics* 38(4),
921–933.
"""
function fevd(formula::FormulaTerm, data::AbstractDataFrame;
        horizon::Integer, estimator::Symbol = :r2,
        shock::Union{Symbol, Nothing} = nothing,
        irf_formula::Union{FormulaTerm, Nothing} = nothing,
        dofadjust::Bool = true, trend::Integer = 0)
    estimator in _FEVD_ESTIMATORS || throw(ArgumentError(
        "`estimator` must be :r2, :lpa or :lpb (got :$estimator)"))
    horizon >= 0 || throw(ArgumentError("horizon must be non-negative"))
    0 <= trend <= 3 || throw(ArgumentError("`trend` must be 0, 1, 2 or 3 (got $trend)"))
    estimator === :r2 && irf_formula !== nothing &&
        throw(ArgumentError(
            "`irf_formula` sets the impulse-response regression of LP-A/LP-B; the R² " *
            "estimator has none"))
    H = Int(horizon)
    trend = Int(trend)

    spec = _gl_parse(formula, data, shock, trend, "formula")
    irf_spec = irf_formula === nothing ? nothing :
               _gl_parse(irf_formula, data, spec.contemporaneous[1], trend, "irf_formula")
    if irf_spec !== nothing
        (irf_spec.lhs === spec.lhs && irf_spec.response === spec.response) ||
            throw(ArgumentError(
                "`irf_formula` must have the same left-hand side as `formula` " *
                "($(formula.lhs)); got $(irf_formula.lhs)"))
    end
    variables = irf_spec === nothing ? spec.variables :
                unique(vcat(spec.variables, irf_spec.variables))
    d = _gl_columns(data, variables)
    T = nrow(data)
    design = _gl_design(spec, irf_spec, d, H)
    dhy, C, W, x = design.dhy, design.C, design.W, design.x
    xraw = d[spec.contemporaneous[1]]
    shock_mean = mean(xraw)
    shock_var = var(xraw)

    t0 = _gl_start(estimator === :r2 ? W : hcat(C, W), dhy, "formula")
    t0lp = irf_spec === nothing ? t0 : _gl_start(design.q, dhy, "irf_formula")
    width = estimator === :r2 ? max(size(W, 2), H + 1) : size(C, 2) + size(W, 2)
    width_lp = design.q === nothing ? width : size(design.q, 2)
    (T - H - t0 + 1 > width && T - H - t0lp + 1 > width_lp) || throw(ArgumentError(
        "horizon $H leaves too few observations: the horizon-$H regressions use " *
        "$(T - H - max(t0, t0lp) + 1) rows for $(max(width, width_lp)) parameters"))
    for h in 0:H
        all(isfinite, view(dhy, t0lp:(T - h), h + 1)) &&
        all(isfinite, view(dhy, t0:(T - h), h + 1)) || throw(ArgumentError(
            "the response is not available on the contiguous rows $t0:$(T - h)"))
    end

    nobs = [T - h - t0 + 1 for h in 0:H]
    window = hcat(fill(t0, H + 1), [T - h for h in 0:H])
    share = zeros(H + 1)
    se_pw = fill(NaN, H + 1)
    se = fill(NaN, H + 1)
    bandwidth = zeros(Int, H + 1)
    theta = Vector{Vector{Float64}}(undef, H + 1)
    avar_pw = Vector{Matrix{Float64}}(undef, H + 1)
    avar = Vector{Matrix{Float64}}(undef, H + 1)
    forecast_errors = fill(NaN, T, H + 1)
    infeasible = Int[]
    exact = Int[]

    if estimator === :r2
        mse = zeros(H + 1)
        for h in 0:H
            rows = t0:(T - h)
            xf = [x[t + j] for t in rows, j in 0:h]
            r = _gl_r2(dhy[rows, h + 1], W[rows, :], xf)
            share[h + 1] = r.s
            mse[h + 1] = r.theta[end]
            forecast_errors[rows, h + 1] = r.fe
            theta[h + 1] = r.theta
            bandwidth[h + 1] = r.bandwidth
            r.feasible || push!(infeasible, h)
            if sum(abs2, r.fe) <= 1e-20 * sum(abs2, view(dhy, rows, h + 1))
                push!(exact, h)
                m = length(r.theta)
                avar_pw[h + 1] = fill(NaN, m, m)
                avar[h + 1] = fill(NaN, m, m)
            else
                avar_pw[h + 1] = r.avar_pw
                avar[h + 1] = r.avar
                se_pw[h + 1] = r.se_pw
                se[h + 1] = r.se
            end
        end
        irf = irf_se = sigma2v = nothing
    else
        r = _gl_lp(dhy, C, W, design.q, x, shock_var, t0, t0lp, H;
            estimator = estimator, dofadjust = dofadjust)
        share .= r.share
        mse = r.mse
        irf = r.psi
        irf_se = r.irf_se
        sigma2v = estimator === :lpb ? r.sigma2v : nothing
        forecast_errors .= r.fe
        Q = r.Q
        Nq = size(Q, 2)
        rq = t0lp:T
        Σq = (Q[rq, :]' * Q[rq, :]) ./ length(rq)
        for h in 0:H
            err = estimator === :lpb ? view(r.v, :, h + 1) : view(r.fe, :, h + 1)
            σ²h = estimator === :lpb ? r.sigma2v[h + 1] : r.mse[h + 1]
            a = _gl_lp_avar(Q, r.resid_lp, err, x, shock_var, σ²h, r.beta_lp, Σq,
                t0, t0lp, h)
            theta[h + 1] = a.theta
            bandwidth[h + 1] = a.bandwidth
            a.feasible || push!(infeasible, h)
            if r.perfect[h + 1]
                push!(exact, h)
                m = length(a.theta)
                avar_pw[h + 1] = fill(NaN, m, m)
                avar[h + 1] = fill(NaN, m, m)
            else
                avar_pw[h + 1] = a.avar_pw
                avar[h + 1] = a.avar
                g = _gl_lp_gradient(estimator, view(r.psi, 1:(h + 1)), shock_var, σ²h,
                    r.mse[h + 1], r.share[h + 1], Nq)
                se_pw[h + 1] = _gl_delta_se(g, a.avar_pw)
                se[h + 1] = _gl_delta_se(g, a.avar)
            end
        end
    end

    if !isempty(infeasible)
        remedy = estimator === :r2 ?
                 "The R² moment vector has 2h + 3 elements whatever the regressors, so " *
                 "only a shorter `horizon` restores prewhitening; `se` and `avar` (no " *
                 "prewhitening) remain available." :
                 "Use a more parsimonious `irf_formula` (Gorodnichenko and Lee use the " *
                 "shock, one lag of it and of the response, and a constant)."
        size_rule = estimator === :r2 || t0lp <= t0 ? "T_h - 1 elements" :
                    "n - 1 elements, n = T - h - max(t0, t0lp) + 1 the rows shared by " *
                    "the forecast-error and impulse-response regressions"
        @warn "fevd: at horizons $(infeasible) the moment vector has at least " *
              size_rule * ", so it cannot be prewhitened; `se_pw` and `avar_pw` are " *
              "NaN there. " * remedy
    end
    isempty(exact) || @warn "fevd: the regression fits exactly at horizons $(exact) " *
          "(e.g. the response is a contemporaneous regressor), so the asymptotic " *
          "variance is degenerate; `se_pw`, `se`, `avar_pw` and `avar` are NaN there."

    return LPFEVD(estimator, H, spec.response, spec.contemporaneous[1], formula,
        irf_formula, spec.contemporaneous, T, t0, t0lp, nobs, window, bandwidth,
        dofadjust, trend, share, irf, irf_se, shock_mean, shock_var, mse, sigma2v,
        forecast_errors, se_pw, se, theta, avar_pw, avar, spec.lhs, variables, spec,
        irf_spec, design)
end

function fevd(m::LocalProjection, data::AbstractDataFrame; estimator::Symbol = :r2,
        irf_formula::Union{FormulaTerm, Nothing} = nothing, dofadjust::Bool = true,
        trend::Integer = 0)
    m.state === nothing || throw(ArgumentError(
        "fevd does not support state-dependent local projections"))
    _rhs_tracks_horizon(m.base_formula) && throw(ArgumentError(
        "fevd needs a horizon-invariant information set; the projection has a " *
        "horizon-tracking right-hand-side term"))
    try
        return fevd(m.base_formula, data; horizon = m.horizon, estimator = estimator,
            shock = m.shock, irf_formula = irf_formula, dofadjust = dofadjust,
            trend = trend)
    catch e
        e isa ArgumentError || rethrow()
        throw(ArgumentError("fevd(lp_result, data): `formula` is the projection's " *
                            "formula $(m.base_formula); " * e.msg))
    end
end

function fevd(::LocalProjectionIV, ::AbstractDataFrame; kwargs...)
    throw(ArgumentError("fevd estimates variance shares of an observed shock; " *
                        "it does not support instrumental-variable projections"))
end

function Base.show(io::IO, f::LPFEVD)
    print(io, "LPFEVD(estimator=:", f.estimator, ", horizon=0:", f.horizon,
        ", response=", f.response, ", shock=", f.shock, ")")
end

function Base.show(io::IO, ::MIME"text/plain", f::LPFEVD)
    println(io, "LPFEVD (Gorodnichenko–Lee ", _fevd_label(f.estimator), ")")
    println(io, "  Response:   ", f.spec.lhs_term)
    println(io, "  Shock:      ", f.shock)
    println(io, "  Formula:    ", f.formula)
    f.irf_formula === nothing || println(io, "  IRF formula: ", f.irf_formula)
    println(io, "  Sample:     T = ", f.T, ", t0 = ", f.t0,
        f.t0lp == f.t0 ? "" : ", t0 (IRF) = $(f.t0lp)",
        ", T_h = T - h - t0 + 1")
    f.estimator === :r2 ||
        println(io, "  Variances:  ", f.dofadjust ? "T_h - k_W" : "T_h",
            " denominator")
    hz = collect(0:(f.horizon))
    if f.irf === nothing
        data = (; h = hz, s = f.share, sp = f.se_pw, se = f.se, n = f.nobs)
        labels = ["Horizon", "Share", "s.e. (PW)", "s.e.", "T_h"]
    else
        data = (; h = hz, s = f.share, sp = f.se_pw, se = f.se, irf = f.irf,
            irf_se = f.irf_se, n = f.nobs)
        labels = ["Horizon", "Share", "s.e. (PW)", "s.e.", "IRF", "IRF s.e.", "T_h"]
    end
    nc = length(labels)
    pretty_table(io, data;
        column_labels = labels,
        title = "Forecast-error variance share of $(f.shock)",
        formatters = [fmt__round(4, collect(2:(nc - 1)))],
        alignment = fill(:r, nc),
        table_format = TextTableFormat(borders = text_table_borders__unicode_rounded))
    return nothing
end
