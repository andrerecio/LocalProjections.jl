# ============================================================================
# VAR residual moving-block bootstrap (Hall percentile-t)
# ============================================================================

# Run `run_draw(b)` for b = 1:nboot, on threads or serially. Each draw seeds
# its own RNG and writes only its own slots, so the two give identical results.
function _run_draws!(run_draw, nboot::Int, threaded::Bool)
    if threaded
        Threads.@threads for b in 1:nboot
            run_draw(b)
        end
    else
        for b in 1:nboot
            run_draw(b)
        end
    end
    return nothing
end

"""
    _position_means(U, ℓ) -> Matrix{Float64}

Position-specific residual means for the moving-block bootstrap, `ℓ × n`.

This is the **sliding-window** mean

```math
\\bar u_s = \\frac{1}{T_u-\\ell+1}\\sum_{r=0}^{T_u-\\ell}\\widehat u_{s+r},
\\qquad s = 1,\\dots,\\ell,
```

which is what the `filter` trick in `_estim/var_boot.m` computes, and what the
Brüggemann–Jentsch–Trenkler procedure requires: position `s` averages over
exactly the `Tu - ℓ + 1` values it can take across the equiprobable block
starts, so the resampled residuals have mean zero by construction.

Note this is *not* the stride-`ℓ` mean `mean(u[s:ℓ:end])` used by
`MacroEconometricTools.bootstrap_irf_block`; the two differ substantially
(on `Tu = 236, ℓ = 20` the sliding mean averages 217 terms, the stride mean 11).
Subtracting only the overall residual mean is not equivalent either.
"""
function _position_means(U::AbstractMatrix{Float64}, ℓ::Int)
    Tu, n = size(U)
    means = Matrix{Float64}(undef, ℓ, n)
    @inbounds for s in 1:ℓ, j in 1:n
        means[s, j] = mean(view(U, s:(Tu - ℓ + s), j))
    end
    return means
end

"""
    _mbb_residuals!(dest, U, means, starts)

Fill `dest` (`Tu × n`) with one moving-block bootstrap draw of the VAR
residuals. Port of the block-resampling branch of `_estim/var_boot.m`:
`length(starts)` blocks of length `ℓ` are laid down end to end, each taken from
offset `starts[b] ∈ {0,…,Tu-ℓ}`, recentered by the position-specific `means`,
and the concatenation is truncated to `Tu` rows.
"""
function _mbb_residuals!(
        dest::AbstractMatrix{Float64}, U::AbstractMatrix{Float64},
        means::AbstractMatrix{Float64}, starts::AbstractVector{Int}
    )
    Tu, n = size(U)
    ℓ = size(means, 1)
    @inbounds for (b, offset) in enumerate(starts)
        base = (b - 1) * ℓ
        for s in 1:ℓ
            row = base + s
            row > Tu && break
            for j in 1:n
                dest[row, j] = U[offset + s, j] - means[s, j]
            end
        end
    end
    return dest
end

"""
    LPBootstrap

Result of [`varbootstrap`](@ref): a local-projection impulse response together
with the VAR residual moving-block bootstrap distribution used to build
percentile-`t` bands.

Three centers are involved and must not be confused (guide §4.6, Step 7):

| Object | Center used |
|---|---|
| Reported real-data response (`theta`) | bias-corrected LP ``\\widehat\\theta_h^c`` |
| Bootstrap responses (`theta_boot`) | bias-corrected bootstrap LP |
| Center of the bootstrap `t`-statistic | Pope-corrected VAR pseudo-truth (`pseudo_truth`) |

`se` holds the real-data HC1 standard errors of the **uncorrected** OLS
coefficients: the reference never recomputes them after bias correction, and uses
HC1 whatever estimator is stored in the projection. `pope_delta` records the Pope
safeguard outcome (`1.0` full correction, `0.0` skipped because the OLS companion
was already explosive, in between attenuated); it is `NaN` when
`popecorrect = false`, since no Pope step ran.

Build bands with [`summarize`](@ref); plot with `irfplot(b; levels = [0.68, 0.90])`
(Makie) or `plot(b; levels)` (Plots.jl); convert with [`as_irf_result`](@ref).
Properties not stored on the result (`horizon`, `response`, `shock`, ...) are
forwarded to the projection it was run on.
"""
struct LPBootstrap{L <: AbstractLocalProjection}
    lp::L
    term::Symbol
    theta::Vector{Float64}
    se::Vector{Float64}
    pseudo_truth::Vector{Float64}
    theta_boot::Matrix{Float64}
    se_boot::Matrix{Float64}
    vars::Vector{Symbol}
    nlags::Int
    blocklength::Int
    nboot::Int
    nfail::Int
    biascorrected::Bool
    popecorrected::Bool
    pope_delta::Float64
end

function Base.getproperty(b::LPBootstrap, name::Symbol)
    name in fieldnames(LPBootstrap) && return getfield(b, name)
    return getproperty(getfield(b, :lp), name)
end

function Base.propertynames(b::LPBootstrap)
    return (fieldnames(LPBootstrap)..., propertynames(getfield(b, :lp))...)
end

function Base.show(io::IO, b::LPBootstrap)
    return print(
        io, "LPBootstrap(horizon=0:", length(b.theta) - 1, ", term=", b.term,
        ", nboot=", b.nboot, ")"
    )
end

function Base.show(io::IO, ::MIME"text/plain", b::LPBootstrap)
    println(io, "LPBootstrap (VAR residual moving-block bootstrap)")
    println(io, "  Term:          ", b.term)
    println(io, "  Horizons:      0:", length(b.theta) - 1)
    println(io, "  VAR variables: ", join(string.(b.vars), ", "), "   (p = ", b.nlags, ")")
    println(
        io, "  Draws:         ", b.nboot, " (block length ", b.blocklength,
        b.nfail > 0 ? ", $(b.nfail) failed)" : ")"
    )
    println(io, "  LP correction: ", b.biascorrected ? "Herbst–Johannsen" : "none (uncorrected OLS)")
    println(
        io, "  VAR DGP:       ",
        b.popecorrected ? "Pope-corrected slopes (δ = $(round(b.pope_delta, digits = 2)))" :
            "OLS slopes"
    )
    print(io, "  Bands:         pointwise, via summarize(b; level, method)")
    return nothing
end

"""
    coefpath(b::LPBootstrap; term=b.term) -> Vector{Float64}

The real-data impulse response path carried by the bootstrap result: the
bias-corrected path when `varbootstrap` was called with `biascorrect = true`.
"""
function coefpath(b::LPBootstrap; term::Symbol = b.term)
    term === b.term || throw(
        ArgumentError(
            "this bootstrap was run for term $(b.term); re-run `varbootstrap` " *
                "to bootstrap a different coefficient"
        )
    )
    return copy(b.theta)
end

"""
    varbootstrap(lp_result, data; vars, nlags, kwargs...) -> LPBootstrap

VAR residual moving-block bootstrap with Hall percentile-`t` bands for a local
projection, following Montiel Olea, Plagborg-Møller, Qian & Wolf (2025). See
`docs/src/inference_procedures_guide.md` §4.

A reduced-form VAR in `vars` (with the shock ordered so that a Cholesky
identification recovers it) is fitted on `data`, a Tables.jl table whose rows are
in time order, and used as the bootstrap DGP. In each draw the VAR residuals are
resampled in overlapping blocks with position-specific recentering, the VAR is
iterated from randomly drawn contiguous initial conditions, and **the complete
local projection is re-estimated** on the artificial sample.

# Required keywords
  - `vars`: the VAR data vector, in identification order. It must contain the
    shock and every variable the LP formula refers to, otherwise the LP cannot
    be rebuilt in each draw.
  - `nlags`: VAR lag length `p`. Choose it with [`lagselect`](@ref) or fix it.

# Optional keywords
  - `nboot = 1000`: bootstrap replications.
  - `blocklength`: block length ``\\ell``; defaults to
    ``\\lceil 5.03\\,T^{1/4}\\rceil`` on the VAR sample, the reference rule.
  - `biascorrect = true`: apply the Herbst–Johannsen correction to the
    real-data **and** every bootstrap LP path.
  - `popecorrect = true`: use Pope-corrected VAR slopes for the bootstrap DGP
    and the pseudo-truth.
  - `rng`, `threaded = false`: see below.

The two `true` defaults together give the complete procedure the reference
recommends. Setting both to `false` reproduces the simpler variant (OLS VAR
DGP, uncorrected LP) described in guide §4.5.

# Reproducibility
Per-draw seeds are drawn from `rng` sequentially *before* the loop, so output
is bit-identical for `threaded = true` and `threaded = false` and independent
of the thread count. Pass a seeded `rng` and record it; also record `nboot`,
`nlags` and `blocklength` when reporting results.

# Notes
Bands are **pointwise across horizons**, not simultaneous. Draws in which the
LP cannot be estimated are recorded in `nfail` and excluded from the quantiles.
Only single-series OLS local projections are supported: no IV, panels, `time`
reordering, `fe()`, state dependence or horizon-tracking regressors, and the
response transform must be `leads` or `cumul`. Each draw re-estimates the
projection with the same `sample` choice as `lp_result`; a `sample = :fixed`
projection needs `biascorrect = false` (see [`biascorrect`](@ref)). The
standard errors are HC1, as in the reference, whatever estimator `lp_result`
stores.

# Example
```julia
sel = lagselect(df, [:shock, :y]; maxlags = 10)
m = lp(@formula(leads(y) ~ shock + lags(y, 4) + lags(shock, 4)), df; horizon = 20)
b = varbootstrap(m, df; vars = [:shock, :y], nlags = nlags(sel), nboot = 1000)
summarize(b; level = 0.90)
```

See also [`summarize`](@ref), [`lagselect`](@ref), [`biascorrect`](@ref).
"""
function varbootstrap(
        lp_result::LocalProjection, data;
        vars::AbstractVector{Symbol},
        nlags::Integer,
        nboot::Integer = 1000,
        blocklength::Union{Integer, Nothing} = nothing,
        biascorrect::Bool = true,
        popecorrect::Bool = true,
        rng::AbstractRNG = default_rng(),
        threaded::Bool = false
    )
    nboot >= 1 || throw(ArgumentError("`nboot` must be at least 1 (got $nboot)"))
    p = Int(nlags)

    lp_result.state === nothing || throw(
        ArgumentError(
            "the VAR bootstrap simulates from a linear VAR, which is not the " *
                "data-generating process of a state-dependent local projection"
        )
    )
    (lp_result.unit === nothing && lp_result.time === nothing) || throw(
        ArgumentError(
            "the VAR bootstrap simulates one time series from the rows of `data` in " *
                "their order; estimate the projection without `unit` and `time`, with " *
                "the rows in time order"
        )
    )
    any(_is_fe, first(_split_rhs(lp_result.base_formula.rhs))) && throw(
        ArgumentError(
            "the VAR bootstrap does not support absorbed fixed effects `fe()`: a " *
                "simulated series has no groups to absorb"
        )
    )

    varlist = collect(vars)
    innov_index = findfirst(isequal(lp_result.shock), varlist)
    innov_index === nothing && throw(
        ArgumentError(
            "the shock $(lp_result.shock) is not in `vars` = $(varlist); the VAR " *
                "data vector must contain the shock, in identification order"
        )
    )
    resp_index = findfirst(isequal(lp_result.response), varlist)
    resp_index === nothing && throw(
        ArgumentError("the response $(lp_result.response) is not in `vars` = $(varlist)")
    )

    needed = _extract_all_vars_from_formula(lp_result.base_formula)
    missing_vars = setdiff(needed, varlist)
    isempty(missing_vars) || throw(
        ArgumentError(
            "the local projection formula refers to $(missing_vars), which are not " *
                "in `vars`; every variable the LP needs must be simulated by the VAR"
        )
    )

    kind = _lhs_kind(lp_result.base_formula)
    kind in (:leads, :cumul) || throw(
        ArgumentError(
            "the VAR bootstrap supports `leads` and `cumul` responses; got " *
                (
                kind === :anchor ? "an anchored response" :
                    kind === :longdiff ? "a long-difference (ldiff/hbr) response" :
                    "an unrecognized LHS"
            )
        )
    )
    biascorrect && lp_result.sample === :fixed && throw(
        ArgumentError(
            "the Herbst–Johannsen correction needs horizon-specific samples, but the " *
                "projection was estimated with sample = :fixed; pass biascorrect = false " *
                "or re-estimate it with sample = :varying"
        )
    )
    _rhs_tracks_horizon(lp_result.base_formula) && throw(
        ArgumentError(
            "the VAR bootstrap centers its t-statistic on the VAR-implied impulse " *
                "response to the shock, which is not the estimand of a local projection " *
                "with a horizon-tracking right-hand-side term (a bare `cumul(x)`, " *
                "`leads(x)`, `ldiff(x)` or `hbr(x, y)`); use a horizon-invariant regressor set"
        )
    )

    Y = _var_data(data, varlist)
    T = size(Y, 1)
    v = _var_ols(Y, p)
    ℓ = blocklength === nothing ? ceil(Int, 5.03 * T^(1 / 4)) : Int(blocklength)
    (1 <= ℓ <= v.Tu) || throw(
        ArgumentError(
            "`blocklength` = $ℓ must be between 1 and the number of VAR residuals " *
                "T - p = $(v.Tu)"
        )
    )

    A_dgp, δ = popecorrect ? _pope_biascorrect(v.A, v.Σu, T) : (v.A, NaN)

    H = lp_result.horizon
    ν = _var_impact(v.U, innov_index)
    irf_var = _var_irf(A_dgp, ν, H)
    pseudo = Vector{Float64}(irf_var[resp_index, :])
    kind === :cumul && (pseudo = cumsum(pseudo))

    real_model = _maybe_biascorrect(lp_result, biascorrect)
    term = lp_result.shock
    theta = coefpath(real_model; term = term)
    se = stderror(lp_result + vcov(HC1()); term = term)

    means = _position_means(v.U, ℓ)
    nblocks = cld(v.Tu, ℓ)
    n_init = T - v.Tu
    seeds = rand(rng, UInt64, nboot)
    names = Tuple(varlist)

    theta_boot = fill(NaN, nboot, H + 1)
    se_boot = fill(NaN, nboot, H + 1)
    failures = zeros(Int, nboot)

    function run_draw(b::Int)
        try
            drng = Xoshiro(seeds[b])
            Ustar = Matrix{Float64}(undef, v.Tu, length(varlist))
            starts = [rand(drng, 0:(v.Tu - ℓ)) for _ in 1:nblocks]
            _mbb_residuals!(Ustar, v.U, means, starts)
            init = rand(drng, 1:(v.Tu + 1))
            Yinit = Y[init:(init + n_init - 1), :]
            Yb = _var_simulate(v.c, A_dgp, Ustar, Yinit)
            datab = NamedTuple{names}(Tuple(Yb[:, j] for j in eachindex(varlist)))
            mb = lp(
                lp_result.base_formula, datab; horizon = H, shock = term,
                sample = lp_result.sample
            )
            θb = coefpath(_maybe_biascorrect(mb, biascorrect); term = term)
            sb = stderror(mb; term = term)   # HC1, the default of `lp`
            @inbounds for h in 1:(H + 1)
                theta_boot[b, h] = θb[h]
                se_boot[b, h] = sb[h]
            end
        catch err
            err isa InterruptException && rethrow()
            failures[b] = 1
        end
        return nothing
    end

    _run_draws!(run_draw, Int(nboot), threaded)

    nfail = sum(failures)
    if nfail > 0.05 * nboot
        @warn "VAR bootstrap: $nfail of $nboot draws failed and are excluded " *
            "from the quantiles; the bands may be unreliable."
    end
    minvalid = minimum(count(isfinite, view(theta_boot, :, h)) for h in 1:(H + 1))
    if minvalid < 100
        @warn "VAR bootstrap: some horizon has only $minvalid usable draws; " *
            "bootstrap quantiles are noisy below ~100 draws."
    end

    return LPBootstrap(
        real_model, term, theta, se, pseudo, theta_boot, se_boot,
        varlist, p, ℓ, Int(nboot), nfail, biascorrect, popecorrect, δ
    )
end

function varbootstrap(::LocalProjectionIV, data; kwargs...)
    throw(
        ArgumentError(
            "the VAR residual moving-block bootstrap is defined for OLS local " *
                "projections and is not available for LocalProjectionIV"
        )
    )
end

"""
    _boot_interval(b, h, α, method) -> (lower, upper)

One pointwise bootstrap interval at horizon index `h`. Port of
`_estim/boot_ci.m`, which returns all three constructions:

```math
t^*_{b,h} = \\frac{\\widehat\\theta^*_{b,h} - \\theta^{\\mathrm{pseudo}}_h}
                  {\\widehat{se}^*_{b,h}}
```

  - `:hall_t` (Hall percentile-`t`, recommended):
    ``[\\widehat\\theta_h - \\widehat{se}_h q^*_{h,1-\\alpha/2},\\;
       \\widehat\\theta_h - \\widehat{se}_h q^*_{h,\\alpha/2}]``; note the
    quantile reversal, which comes from inverting the studentized inequality;
  - `:hall`: ``\\widehat\\theta_h + \\theta^{\\mathrm{pseudo}}_h`` minus the
    reversed quantiles of the bootstrap responses;
  - `:efron`: the plain ``[\\alpha/2, 1-\\alpha/2]`` quantiles of the draws.

Intervals need not be symmetric around the point estimate. Quantiles use
`Statistics.quantile` (type 7), which is close to but not bit-identical with
MATLAB's `quantile`.
"""
function _boot_interval(b::LPBootstrap, h::Int, α::Float64, method::Symbol)
    θs = view(b.theta_boot, :, h)
    if b.se[h] == 0
        # Tautological horizon (response === shock at h = 0): degenerate band.
        return (b.theta[h], b.theta[h])
    end
    if method === :hall_t
        ses = view(b.se_boot, :, h)
        ts = Float64[]
        @inbounds for i in eachindex(θs)
            if isfinite(θs[i]) && isfinite(ses[i]) && ses[i] > 0
                push!(ts, (θs[i] - b.pseudo_truth[h]) / ses[i])
            end
        end
        isempty(ts) && throw(
            ErrorException(
                "no usable bootstrap draws at horizon $(h - 1); cannot form a " *
                    "percentile-t interval"
            )
        )
        return (
            b.theta[h] - b.se[h] * quantile(ts, 1 - α / 2),
            b.theta[h] - b.se[h] * quantile(ts, α / 2),
        )
    end
    vals = Float64[x for x in θs if isfinite(x)]
    isempty(vals) && throw(ErrorException("no usable bootstrap draws at horizon $(h - 1)"))
    ql = quantile(vals, α / 2)
    qu = quantile(vals, 1 - α / 2)
    method === :efron && return (ql, qu)
    # :hall
    return (b.theta[h] + b.pseudo_truth[h] - qu, b.theta[h] + b.pseudo_truth[h] - ql)
end

function _check_boot_method(method::Symbol)
    method in (:hall_t, :hall, :efron) ||
        throw(ArgumentError("`method` must be :hall_t, :hall or :efron (got :$method)"))
    return method
end

# Pointwise bootstrap bounds at every horizon for one coverage level
function _boot_bounds(b::LPBootstrap, level::Real, method::Symbol)
    α = 1 - Float64(level)
    H = length(b.theta) - 1
    bounds = [_boot_interval(b, h, α, method) for h in 1:(H + 1)]
    return first.(bounds), last.(bounds)
end

"""
    summarize(b::LPBootstrap; level=0.90, method=:hall_t, scale=1.0) -> IRFSummary

Bootstrap confidence bands for a local projection. `method` selects the
interval construction: `:hall_t` (Hall percentile-`t`, the recommended one),
`:hall`, or `:efron`; see `_boot_interval`.

The bands are **pointwise across horizons**, not simultaneous, and are
generally asymmetric around the point estimate. Note that a percentile-`t` or
Hall interval need not *contain* the point estimate: both correct for bias, so
when the bootstrap `t`-distribution is shifted the whole interval can sit to
one side of ``\\widehat\\theta_h``. That is a property of the method, not a
defect. The reported `coef` is the real-data path (bias-corrected when
`varbootstrap` was run with `biascorrect = true`) and `se` the HC1 standard
errors of the uncorrected coefficients. A negative `scale` flips the sign, and
the bounds swap so that `lower <= upper`.
"""
function summarize(
        b::LPBootstrap; level::Real = 0.9, method::Symbol = :hall_t, scale::Real = 1.0
    )
    _check_boot_method(method)
    _check_levels((level,))
    scale_f = Float64(scale)
    lower, upper = _boot_bounds(b, level, method)
    lower, upper = scale_f .* lower, scale_f .* upper
    if scale_f < 0
        lower, upper = upper, lower
    end
    H = length(b.theta) - 1
    return IRFSummary(
        b.response, b.term, Float64(level), scale_f, collect(0:H),
        scale_f .* b.theta, abs(scale_f) .* b.se, lower, upper
    )
end

# The IRF object of a bootstrap: the real-data path, its HC1 standard errors and the
# bootstrap bands at each level. Every plot and `as_irf_result` of an LPBootstrap
# draws from it.
function _irf_result(b::LPBootstrap, coverage, method::Symbol)
    _check_boot_method(method)
    levels = collect(Float64, coverage)
    _check_levels(levels)
    cube(v) = AxisArray(
        reshape(collect(Float64, v), 1, 1, length(v)),
        Axis{:response}([b.response]),
        Axis{:shock}([b.term]),
        Axis{:horizon}(0:(length(v) - 1))
    )
    bands = [_boot_bounds(b, c, method) for c in levels]
    meta = (
        horizon = length(b.theta) - 1,
        formula = b.base_formula,
        term = b.term,
        bootstrap = true,
        bootstrap_method = method,
        nboot = b.nboot,
        nfail = b.nfail,
        blocklength = b.blocklength,
        var_lags = b.nlags,
        var_variables = b.vars,
        bias_corrected = b.biascorrected,
        pope_corrected = b.popecorrected,
        pope_delta = b.pope_delta,
    )
    return LocalProjectionIRFResult(
        cube(b.theta), cube(b.se), [cube(first(s)) for s in bands],
        [cube(last(s)) for s in bands], levels, meta
    )
end

"""
    as_irf_result(b::LPBootstrap; term=b.term, coverage=[0.68, 0.90], method=:hall_t)

Convert an [`LPBootstrap`](@ref) to a `LocalProjectionIRFResult`, carrying the
bootstrap bands rather than normal-approximation ones.

The bands are those of [`summarize`](@ref) under `method`, so they are
asymmetric and **pointwise**. `stderr` holds the real-data HC1 standard errors
of the uncorrected coefficients (the reference convention), which are *not*
what the bands are built from. The metadata records `bootstrap = true` along
with the draw count, block length, VAR lag length, failed draws and which
corrections were applied.
"""
function as_irf_result(
        b::LPBootstrap; term::Symbol = b.term, coverage = [0.68, 0.9], method::Symbol = :hall_t
    )
    term === b.term ||
        throw(ArgumentError("this bootstrap was run for term $(b.term), not $term"))
    return _irf_result(b, coverage, method)
end

"""
    irfplot(b::LPBootstrap; levels = [0.90], method = :hall_t, kwargs...) -> Makie.Figure
    irfplot!(ax, b::LPBootstrap; levels = [0.90], method = :hall_t, kwargs...) -> ax

Plot the bootstrap impulse response with its pointwise `method` bands at each of
`levels` (requires a Makie backend). With Plots.jl, `plot(b; levels, method)` draws
the same object. The remaining keywords go to MacroEconometricTools' `irfplot`.
"""
function MacroEconometricTools.irfplot(
        b::LPBootstrap; levels = [0.9], method::Symbol = :hall_t, kwargs...
    )
    return irfplot(_irf_result(b, levels, method); kwargs...)
end

function MacroEconometricTools.irfplot!(
        ax, b::LPBootstrap; levels = [0.9], method::Symbol = :hall_t, kwargs...
    )
    return irfplot!(ax, _irf_result(b, levels, method); var = b.response, shock = b.term, kwargs...)
end

@recipe function f(b::LPBootstrap; levels = [0.9], method = :hall_t)
    return _irf_result(b, levels, method)
end
