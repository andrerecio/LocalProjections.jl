# ============================================================================
# Joint inference across horizons: the covariance of the impulse-response path,
# Wald tests of several horizons and sup-t simultaneous bands (lecture 3 of
# Plagborg-Møller's local-projection course: `suest`, `test`, and
# `estimate_supt_critical_value` of Montiel Olea & Plagborg-Møller 2019).
# ============================================================================

# Horizons with a sampling variance: all of 0:H, except h = 0 when the response
# is the shock (coefficient 1, standard error 0 by construction).
_joint_default_horizons(lp::AbstractLocalProjection) =
    collect(lp.tautological_h0 ? (1:(lp.horizon)) : (0:(lp.horizon)))

function _joint_horizons(lp::AbstractLocalProjection, horizons)
    hs = horizons === nothing ? _joint_default_horizons(lp) : collect(Int, horizons)
    isempty(hs) && throw(ArgumentError("`horizons` is empty"))
    allunique(hs) || throw(ArgumentError("`horizons` lists a horizon twice"))
    for h in hs
        0 <= h <= lp.horizon || throw(
            ArgumentError("horizon $h is outside the estimated horizons 0:$(lp.horizon)")
        )
        h == 0 && lp.tautological_h0 && throw(
            ArgumentError(
                "the response is the shock, so its h = 0 coefficient is 1 with no " *
                    "sampling variance; leave h = 0 out of `horizons`"
            )
        )
    end
    return hs
end

# The checks shared by every joint procedure.
function _check_joint(lp::AbstractLocalProjection)
    lp.unit === nothing || throw(
        ArgumentError(
            "joint inference across horizons stacks the scores of every horizon on one " *
                "time axis; panels (`unit`) are not supported"
        )
    )
    est = _vcov_estimator(lp)
    est isa Union{CR, DriscollKraay} && throw(
        ArgumentError(
            "joint inference across horizons supports heteroskedasticity-robust (HC0–HC5), " *
                "kernel HAC and EWC estimators; $(nameof(typeof(est))) is not supported"
        )
    )
    est isa Union{HR, HAC, EWC} || throw(
        ArgumentError(
            "joint inference across horizons supports heteroskedasticity-robust (HC0–HC5), " *
                "kernel HAC and EWC estimators; got $(nameof(typeof(est)))"
        )
    )
    for m in lp.models
        isempty(m.fe_groups) || throw(
            ArgumentError(
                "joint inference across horizons does not support absorbed fixed effects `fe()`"
            )
        )
    end
    return est
end

# Contribution of each estimation row of `m` to the coefficient in column `j`:
# the `j`-th column of `momentmatrix(m) * bread(m)`, with the residual adjustment of
# a heteroskedasticity-robust `est` (HC2–HC5 leverage; HC0 and HC1 need none here,
# since HC1's adjustment is a constant per horizon). These are the moments behind
# Regress's `vcov(est, m)`.
function _influence(m, j::Int, est, h::Int, term::Symbol)
    keep = mask(m)
    keep[j] || throw(
        ArgumentError(
            "$term is collinear with the other regressors at horizon $h and was dropped"
        )
    )
    M = momentmatrix(m)[:, keep]
    if est isa HR && !(est isa Union{HR0, HR1})
        M = M .* residual_adjustment(est, m)
    end
    return M * bread(m)[:, count(view(keep, 1:j))]
end

# The influence of every horizon in `hs` on one time axis, the rows of `lp.data`
# from the first to the last row any of them uses, with zeros on the rows a horizon
# does not use. Lags between rows are then distances in time, as a kernel needs.
function _stacked_influence(lp::AbstractLocalProjection, term::Symbol, hs, est)
    j = _term_index(lp, term)
    used = reduce(.|, (lp.rows[h + 1] for h in hs))
    t1, t2 = findfirst(used), findlast(used)
    Ψ = zeros(t2 - t1 + 1, length(hs))
    for (c, h) in enumerate(hs)
        idx = findall(lp.rows[h + 1]) .- (t1 - 1)
        Ψ[idx, c] = _influence(lp.models[h + 1], j, est, h, term)
    end
    return Ψ
end

# Long-run covariance of the stacked influence under `est`, on the unit-free scale
# of a correlation: only its correlation is used. A heteroskedasticity-robust
# estimator becomes HC0 on these moments, as in Regress; a kernel or EWC estimator is
# copied, because an automatic bandwidth is stored in the estimator when selected.
function _stacked_avar(Ψ::Matrix{Float64}, est)
    k = est isa HR ? HR0() : deepcopy(est)
    return Matrix{Float64}(parent(aVar(k, Ψ; demean = false)))
end

function _joint_correlation(lp::AbstractLocalProjection, term::Symbol, hs)
    est = _check_joint(lp)
    Ω = _stacked_avar(_stacked_influence(lp, term, hs, est), est)
    d = sqrt.(diag(Ω))
    all(>(0), d) || throw(
        ArgumentError("a horizon in `horizons` has a zero long-run variance for $term")
    )
    R = Ω ./ (d * d')
    R[diagind(R)] .= 1.0
    return Symmetric(R), est
end

"""
    jointvcov(lp; term = lp.shock, horizons = nothing) -> Matrix{Float64}
    jointvcov(lp, estimator; kwargs...)

Covariance matrix of the coefficients of `term` across `horizons` (by default
`0:lp.horizon`, without ``h = 0`` when the response is the shock), the
`suest` step of joint inference on an impulse-response path.

The regressions of different horizons share their observations, so their
estimates are correlated. The correlation is estimated from the scores of all
horizons stacked on one time axis, with zeros on the rows a horizon does not use,
under the covariance estimator stored in `lp`: their outer products for HC0–HC5,
and the kernel or the cosine weights of a HAC or `EWC` estimator, applied across
horizons as well as within them. The diagonal is the squared [`stderror`](@ref)
of each horizon, so the matrix is consistent with [`summarize`](@ref):

```math
\\hat V_{hg} = \\mathrm{se}_h\\,\\mathrm{se}_g\\,\\hat\\rho_{hg}.
```

For HC0 and HC1 this is the stacked sandwich with each horizon's own
finite-sample factor, ``\\sqrt{c_h c_g}`` with ``c_h = n_h/(n_h-k)`` for HC1.
Stata's `suest, vce(robust)` applies one factor ``N/(N-1)`` to every block, with
``N`` the number of rows used at any horizon: with `HC0()` stored, its matrix is
`jointvcov(lp) .* N / (N - 1)`. A HAC estimator with an automatic bandwidth
selects one bandwidth for the stacked scores, so the correlation may use a
different bandwidth from the standard errors.

Supported: OLS and IV projections on a single time series, with an HC0–HC5,
kernel HAC or `EWC` estimator. Panels, absorbed fixed effects and cluster or
Driscoll–Kraay estimators throw. A `BiasCorrectedLP` uses the standard errors of
the uncorrected coefficients, as its bands do.

See also [`jointtest`](@ref) and [`supt_critical_value`](@ref).
"""
function jointvcov(lp::AbstractLocalProjection; term::Symbol = lp.shock, horizons = nothing)
    hs = _joint_horizons(lp, horizons)
    R, _ = _joint_correlation(lp, term, hs)
    se = stderror(lp; term)[hs .+ 1]
    return Matrix(Symmetric(se .* R .* se'))
end

function jointvcov(
        lp::AbstractLocalProjection,
        estimator::CovarianceMatrices.AbstractAsymptoticVarianceEstimator; kwargs...
    )
    return jointvcov(lp + vcov(estimator); kwargs...)
end

"""
    JointTest

Wald test of the impulse responses of one term at several horizons, built by
[`jointtest`](@ref).

# Fields
- `term`, `horizons`, `value` (the hypothesized responses), `estimate` (the
  estimated ones), `vcov` (their [`jointvcov`](@ref)), `estimator`;
- `wald`: the Wald statistic ``W = (\\hat\\theta - \\theta_0)'\\hat V^{-1}(\\hat\\theta - \\theta_0)``;
- `distribution`: `:chisq` or `:F`; `statistic`: ``W`` for ``\\chi^2``, ``F`` for
  `EWC(B)`; `df` (and `dfden` for ``F``), `pvalue`.
"""
struct JointTest
    term::Symbol
    horizons::Vector{Int}
    value::Vector{Float64}
    estimate::Vector{Float64}
    vcov::Matrix{Float64}
    estimator::Any
    wald::Float64
    distribution::Symbol
    statistic::Float64
    df::Int
    dfden::Union{Nothing, Int}
    pvalue::Float64
end

"""
    jointtest(lp; term = lp.shock, horizons = nothing, value = 0.0) -> JointTest
    jointtest(lp, estimator; kwargs...)

Wald test that the response to `term` equals `value` (a number, or one value per
horizon) at every horizon in `horizons` jointly; by default, the null of no
effect at any horizon `0:lp.horizon` (without ``h = 0`` when the response is the
shock). This is the `suest` and `test` step of lecture 3 of Plagborg-Møller's
local-projection course. The statistic uses [`jointvcov`](@ref).

The reference distribution follows the stored estimator, as the pointwise bands
do: ``\\chi^2_m`` with ``m`` = `length(horizons)`, and for `EWC(B)` the
fixed-smoothing distribution of Lazarus, Lewis, Stock and Watson (2018),

```math
F = \\frac{B - m + 1}{B\\,m}\\,W \\sim F_{m,\\,B-m+1},
```

which needs ``B \\ge m``.

With `HC0()` stored, `W * (N - 1) / N` is the ``\\chi^2`` statistic of Stata's
`suest, vce(robust)` followed by `test`, ``N`` the number of rows used at any
horizon (see [`jointvcov`](@ref)).

# Example
```julia
r = lp(@formula(leads(y) ~ x + lags(y, 2) + lags(x, 2)), df; horizon = 10)
jointtest(r)                          # H₀: θ₀ = ⋯ = θ₁₀ = 0, HC1
jointtest(r, EWC(ewc_bandwidth(r)))   # fixed-b F test
jointtest(r; horizons = 4:10)
```
"""
function jointtest(
        lp::AbstractLocalProjection; term::Symbol = lp.shock, horizons = nothing,
        value = 0.0
    )
    hs = _joint_horizons(lp, horizons)
    m = length(hs)
    θ0 = value isa Real ? fill(Float64(value), m) : collect(Float64, value)
    length(θ0) == m || throw(
        ArgumentError("`value` has $(length(θ0)) elements for $m horizons")
    )
    est = _vcov_estimator(lp)
    # An EWC(B) long-run variance has rank at most B
    est isa EWC && est.B < m && throw(
        ArgumentError(
            "the fixed-smoothing F test of $m restrictions needs EWC(B) with B ≥ $m " *
                "(got B = $(est.B)); test fewer horizons or use a larger B"
        )
    )
    θ = coefpath(lp; term)[hs .+ 1]
    V = jointvcov(lp; term, horizons = hs)
    F = cholesky(Symmetric(V); check = false)
    issuccess(F) || throw(
        ArgumentError(
            "the covariance of the responses at horizons $hs is singular; test fewer horizons"
        )
    )
    d = θ .- θ0
    W = dot(d, F \ d)
    if est isa EWC
        B = est.B
        stat = (B - m + 1) / (B * m) * W
        dist, dfden = :F, B - m + 1
        p = ccdf(FDist(m, dfden), stat)
    else
        stat, dist, dfden = W, :chisq, nothing
        p = ccdf(Chisq(m), W)
    end
    return JointTest(term, hs, θ0, θ, V, est, W, dist, stat, m, dfden, p)
end

function jointtest(
        lp::AbstractLocalProjection,
        estimator::CovarianceMatrices.AbstractAsymptoticVarianceEstimator; kwargs...
    )
    return jointtest(lp + vcov(estimator); kwargs...)
end

function _horizon_range_string(hs)
    return length(hs) > 1 && hs == collect(first(hs):last(hs)) ?
        "$(first(hs)):$(last(hs))" : join(hs, ", ")
end

function Base.show(io::IO, t::JointTest)
    return print(
        io, "JointTest(term=", t.term, ", horizons=", _horizon_range_string(t.horizons),
        ", ", t.distribution === :F ? "F" : "chi2", "=", round(t.statistic; digits = 4),
        ", p=", round(t.pvalue; sigdigits = 4), ")"
    )
end

function Base.show(io::IO, ::MIME"text/plain", t::JointTest)
    null = all(iszero, t.value) ? "0" : "θ₀"
    println(io, "Joint Wald test across horizons")
    println(io, "  Term:       ", t.term)
    println(io, "  H₀:         θ_h = ", null, " for h ∈ {", _horizon_range_string(t.horizons), "}")
    println(io, "  Vcov:       ", repr(t.estimator))
    if t.distribution === :F
        println(
            io, "  F(", t.df, ", ", t.dfden, ") = ", round(t.statistic; digits = 4),
            "   (Wald = ", round(t.wald; digits = 4), ", fixed-b EWC)"
        )
    else
        println(io, "  chi2(", t.df, ") = ", round(t.statistic; digits = 4))
    end
    return print(io, "  p-value:    ", round(t.pvalue; sigdigits = 4))
end

# --------------------------------------------------------------------------
# sup-t critical values
# --------------------------------------------------------------------------

# Seed of the default generator, so that bands built without an explicit `rng`
# are reproducible.
const _SUPT_SEED = 20261031

# The empirical quantile rule of Kessler's `estimate_supt_critical_value`: with
# `t` sorted and k = level·n, the average of the k-th and (k+1)-th values when k is
# an integer, otherwise the ⌊k⌋+1-th.
function _supt_quantile(t::AbstractVector{Float64}, level::Real)
    n = length(t)
    k = level * n
    return isinteger(k) ? (t[Int(k)] + t[Int(k) + 1]) / 2 : t[floor(Int, k) + 1]
end

"""
    _supt_from_draws(draws, sd, level) -> Float64

sup-t critical value from Gaussian `draws` (one draw per row) of a vector with
standard deviations `sd`: the `level` quantile of ``\\max_h |z_h| / sd_h``, as
`estimate_supt_critical_value` computes it from its `drawnorm` draws.
"""
function _supt_from_draws(draws::AbstractMatrix{<:Real}, sd::AbstractVector{<:Real}, level::Real)
    t = sort!(vec(maximum(abs.(draws ./ sd'), dims = 2)))
    return _supt_quantile(t, level)
end

# A square root L (R = L L') of the correlation matrix `R`. The Cholesky factor is
# unique, so a seeded generator gives the same draws on every machine; a
# rank-deficient `R` (e.g. EWC with B below the number of horizons) falls back to the
# eigendecomposition, with each eigenvector's sign fixed by its largest entry.
function _corr_factor(R::Symmetric)
    F = cholesky(R; check = false)
    issuccess(F) && return Matrix(F.L)
    λ, Q = eigen(R)
    for k in axes(Q, 2)
        i = argmax(abs.(view(Q, :, k)))
        Q[i, k] < 0 && (Q[:, k] .*= -1)
    end
    return Q * Diagonal(sqrt.(max.(λ, 0.0)))
end

# Sorted draws of max_h |t_h|: t = Z under a normal reference distribution, and the
# fixed-smoothing limit of EWC(B), t_h = Z_h / √(S_hh / B) with S = Σ_{b ≤ B} ξ_b ξ_b',
# Z and ξ_b independent N(0, R), otherwise.
function _supt_maxabs(R::Symmetric, est, ndraws::Int, rng::AbstractRNG)
    L = _corr_factor(R)
    m = size(R, 1)
    T = L * randn(rng, m, ndraws)
    if est isa EWC
        S = zeros(m, ndraws)
        G = Matrix{Float64}(undef, m, ndraws)
        Ξ = similar(G)
        for _ in 1:(est.B)
            mul!(Ξ, L, randn!(rng, G))
            S .+= Ξ .^ 2
        end
        T ./= sqrt.(S ./ est.B)
    end
    return sort!(vec(maximum(abs, T; dims = 1)))
end

"""
    supt_critical_value(lp; term = lp.shock, level = 0.95, horizons = nothing,
                        ndraws = 100_000, rng = Xoshiro($(_SUPT_SEED))) -> Float64
    supt_critical_value(lp, estimator; kwargs...)

Critical value ``c`` of the sup-t simultaneous confidence band
``\\hat\\theta_h \\pm c\\,\\mathrm{se}_h`` (Montiel Olea and Plagborg-Møller,
2019), which covers the whole impulse response of `term` over `horizons` with
probability `level`. ``c`` is the `level` quantile of
``\\max_h |\\hat\\theta_h - \\theta_h| / \\mathrm{se}_h``, simulated from the
correlation of the estimates across horizons ([`jointvcov`](@ref)): Gaussian
draws, or for `EWC(B)` the fixed-smoothing limit
``t_h = Z_h / \\sqrt{S_{hh}/B}``, with ``S`` a Wishart matrix with ``B`` degrees of
freedom, so that each margin is Student-``t_B`` as in the pointwise band.

The quantile rule is that of Kessler's Stata command
`estimate_supt_critical_value` (used in Plagborg-Møller's course), whose default is
1000 draws. With the default generator the value is reproducible; its Monte Carlo
error with 100,000 draws is up to about 0.01 at a 95% level; pass a larger
`ndraws` for more digits.

`summarize`, `plot`, `irfplot` and `as_irf_result` build these bands with
`bands = :supt`.
"""
function supt_critical_value(
        lp::AbstractLocalProjection; term::Symbol = lp.shock, level::Real = 0.95,
        horizons = nothing, ndraws::Integer = 100_000,
        rng::AbstractRNG = Xoshiro(_SUPT_SEED)
    )
    0 < level < 1 || throw(ArgumentError("`level` must lie strictly between 0 and 1"))
    ndraws >= 1 || throw(ArgumentError("`ndraws` must be positive"))
    hs = _joint_horizons(lp, horizons)
    R, est = _joint_correlation(lp, term, hs)
    length(hs) == 1 && return _critical_value(est, level)
    return _supt_quantile(_supt_maxabs(R, est, Int(ndraws), rng), level)
end

function supt_critical_value(
        lp::AbstractLocalProjection,
        estimator::CovarianceMatrices.AbstractAsymptoticVarianceEstimator; kwargs...
    )
    return supt_critical_value(lp + vcov(estimator); kwargs...)
end

# Critical values of the bands of `_irf_result` at each of `levels`. sup-t bands
# share one set of draws across the levels.
function _band_critical_values(lp::AbstractLocalProjection, term::Symbol, levels, bands::Symbol)
    est = _vcov_estimator(lp)
    bands === :pointwise && return [_critical_value(est, c) for c in levels]
    bands === :supt || throw(
        ArgumentError("`bands` must be :pointwise or :supt (got $(repr(bands)))")
    )
    hs = _joint_default_horizons(lp)
    R, est = _joint_correlation(lp, term, hs)
    length(hs) == 1 && return [_critical_value(est, c) for c in levels]
    t = _supt_maxabs(R, est, 100_000, Xoshiro(_SUPT_SEED))
    return [_supt_quantile(t, c) for c in levels]
end
