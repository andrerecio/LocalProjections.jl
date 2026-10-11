# ============================================================================
# FEVD bias corrections (Gorodnichenko & Lee 2020)
# ============================================================================

const _FEVD_METHODS = (:var, :block, :simulation)

"""
    _gl_lagselect(Y, candidates; criterion = :hqic) -> NamedTuple

VAR lag order chosen as in the Gorodnichenko–Lee bootstrap drivers
(`BootstrapVARR2.m:144-173`). Every candidate ``p`` is fitted on the **common**
effective sample (the last ``T_{IC} = T - p_{\\max}`` rows, the preceding ``p``
rows as initial conditions), with the maximum-likelihood covariance
``\\hat\\Omega_p = \\hat U'\\hat U / T_{IC}`` and ``k = n^2 p``:

```math
\\mathrm{HQIC} = \\log\\det\\hat\\Omega_p + \\frac{2k\\log\\log T_{IC}}{T_{IC}},\\qquad
\\mathrm{BIC} = \\log\\det\\hat\\Omega_p + \\frac{k\\log T_{IC}}{T_{IC}},\\qquad
\\mathrm{AIC} = \\log\\det\\hat\\Omega_p + \\frac{2k}{T_{IC}}.
```

Ties go to the first candidate. Returns `(; p, values)`. These conventions
differ from [`lagselect`](@ref) (the `ic_var.m` port), which is left unchanged.
"""
function _gl_lagselect(Y::AbstractMatrix{Float64}, candidates::AbstractVector{Int};
        criterion::Symbol = :hqic)
    criterion in (:hqic, :bic, :aic) || throw(ArgumentError(
        "`criterion` must be :hqic, :bic or :aic (got :$criterion)"))
    isempty(candidates) && throw(ArgumentError("`nlags` has no candidate lag order"))
    all(>=(1), candidates) || throw(ArgumentError(
        "the candidate VAR lag orders must be at least 1 (got $(candidates))"))
    n = size(Y, 2)
    pmax = maximum(candidates)
    Tic = size(Y, 1) - pmax
    values = Vector{Float64}(undef, length(candidates))
    for (i, p) in enumerate(candidates)
        v = _var_ols(Y[(pmax - p + 1):end, :], p)
        Ω = (v.U' * v.U) ./ Tic
        _check_var_rank(Ω)
        k = n^2 * p
        penalty = criterion === :hqic ? 2k * log(log(Tic)) / Tic :
                  criterion === :bic ? k * log(Tic) / Tic : 2k / Tic
        values[i] = logdet(Symmetric(Ω)) + penalty
    end
    return (; p = candidates[argmin(values)], values)
end

# R² share at horizon h on the regression rows `rows` (`t` indexes the window
# t0..T-h of the original sample, possibly resampled) of the design `des`.
function _gl_r2_rows(des::_GLDesign, rows::AbstractVector{Int}, h::Int)
    xf = [des.x[t + j] for t in rows, j in 0:h]
    return _gl_r2(des.dhy[rows, h + 1], des.W[rows, :], xf; avar = false).s
end

# The specification of `f` re-estimated on the column table `d` (a bootstrap
# sample): the shares, the unit-shock responses (LP-A/B) and the shock variance.
# No asymptotic variances. On gap-free data t0 and t0lp depend only on the lag
# structure, so those of `f` apply.
function _gl_point(f::LPFEVD, d::NamedTuple)
    H = f.horizon
    des = _gl_design(f.spec, f.irf_spec, d, H)
    σ²x = var(d[f.shock])
    if f.estimator === :r2
        T = size(des.dhy, 1)
        share = [_gl_r2_rows(des, f.t0:(T - h), h) for h in 0:H]
        return (; share, psi = nothing, σ²x)
    end
    r = _gl_lp(des.dhy, des.C, des.W, des.q, des.x, σ²x, f.t0, f.t0lp, H;
        estimator = f.estimator, dofadjust = f.dofadjust, irf_se = false)
    return (; share = r.share, psi = r.psi, σ²x)
end

"""
    _gl_psd_factor(V) -> Matrix{Float64}

A factor ``M`` with ``MM' \\approx V`` for drawing from ``N(\\hat\\theta, V)``.
The Cholesky factor of the symmetrized `V` when it is positive definite;
otherwise the eigenvalues are floored at `eps()` (the diagonal of the eigenvalue
matrix only) and ``M = Q\\Lambda^{1/2}`` (`SimulationAVar*.m`, Q18). The
reference floors the whole eigenvalue matrix, off-diagonal zeros included, and
repeats the repair until `cholcov` succeeds, in an unbounded loop; drawing
through the eigenfactor makes one pass sufficient.
"""
function _gl_psd_factor(V::AbstractMatrix{Float64})
    S = Symmetric((V + V') ./ 2)
    C = cholesky(S; check = false)
    issuccess(C) && return Matrix{Float64}(C.L)
    E = eigen(S)
    return E.vectors .* transpose(sqrt.(max.(E.values, eps())))
end

# One simulated share at horizon h: θ* = θ̂ + M z, redrawn while a variance is
# nonpositive (R²: θ₃; LP-A/B: σ²x and σ²_h). NaN after `maxtries` rejections.
function _gl_sim_share(rng::AbstractRNG, θ̂::Vector{Float64}, M::Matrix{Float64},
        estimator::Symbol, h::Int; maxtries::Int = 10_000)
    m = length(θ̂)
    z = Vector{Float64}(undef, size(M, 2))
    for _ in 1:maxtries
        θ = θ̂ + M * Random.randn!(rng, z)
        if estimator === :r2
            θ[m] > 0 || continue
            k = h + 1
            return dot(view(θ, 1:k), view(θ, (k + 1):(2k))) / θ[m]
        end
        (θ[m - 1] > 0 && θ[m] > 0) || continue
        Nq = (m - 2) ÷ (h + 1)
        num = θ[m - 1] * sum(j -> abs2(θ[1 + j * Nq]), 0:h)
        return estimator === :lpa ? num / θ[m] : num / (num + θ[m])
    end
    return NaN
end

"""
    FEVDBootstrap

Result of [`fevdbootstrap`](@ref): a bias-corrected Gorodnichenko–Lee
forecast-error variance decomposition together with the draws it is built from.
Fields of the underlying [`LPFEVD`](@ref) are reachable directly (`b.share`,
`b.horizon`, …); `b.fevd` is the estimate itself.

# Fields
- `fevd`, `method` (`:var`, `:block` or `:simulation`);
- `draws` (``B_{\\mathrm{eff}} \\times (H+1)``, the shares ``s^*_{b,h}`` of the
  draws that succeeded; for `:simulation` with `bound`, clamped to ``[0, 1]``),
  `mean_draws` (``\\bar m_h``), `se_boot` (their standard deviation, ``B-1``
  denominator);
- `pseudo_truth`: the share implied by the bootstrap VAR, ``s^{VAR}_h``
  (`:var` only, else `nothing`);
- `biascorrected`: ``\\hat s_h - (\\bar m_h - s^{VAR}_h)`` for `:var`,
  ``2\\hat s_h - \\bar m_h`` for `:block` and `:simulation`; never clamped;
- `se_pw`, `se`: delta-method standard errors with and without prewhitening,
  at ``\\hat s_h`` (those of `fevd`) for `:var` and `:block`, at the
  bias-corrected share for `:simulation` (the authors' convention, so they
  depend on the draws);
- LP-A/B with `:var` (else `nothing`): `irf_draws` (unit-shock responses
  ``\\psi^*_{b,h}``), `irf_bc` ``= 2\\hat\\psi_h - \\mathrm{mean}_b\\,\\psi^*_{b,h}``
  (centred on ``\\hat\\psi``, not on the VAR response, as in the reference),
  `irf_se_boot`, and `irfstr_draws` (``\\psi^*_{b,h}`` times the standard
  deviation of the simulated shock, the one-standard-deviation responses);
- `:var` metadata (else `nothing`): `vars` (VAR variables, Cholesky order),
  `nlags` (VAR lag order), `ic` (`(; criterion, candidates, values)`, or
  `nothing` when `nlags` was fixed), `skipfirst`, `burnin`;
- `blocklength` (`:block`), `prewhiten` and `bound` (`:simulation`; `bound` is
  `false` for LP-A/B, which are never clamped);
- `nboot` (requested draws) and `nfail` (draws that failed and are excluded).
"""
struct FEVDBootstrap
    fevd::LPFEVD
    method::Symbol
    draws::Matrix{Float64}
    mean_draws::Vector{Float64}
    pseudo_truth::Union{Nothing, Vector{Float64}}
    biascorrected::Vector{Float64}
    se_boot::Vector{Float64}
    se_pw::Vector{Float64}
    se::Vector{Float64}
    irf_draws::Union{Nothing, Matrix{Float64}}
    irf_bc::Union{Nothing, Vector{Float64}}
    irf_se_boot::Union{Nothing, Vector{Float64}}
    irfstr_draws::Union{Nothing, Matrix{Float64}}
    vars::Union{Nothing, Vector{Symbol}}
    nlags::Union{Nothing, Int}
    ic::Union{Nothing,
        @NamedTuple{criterion::Symbol, candidates::Vector{Int}, values::Vector{Float64}}}
    skipfirst::Union{Nothing, Bool}
    burnin::Union{Nothing, Int}
    blocklength::Union{Nothing, Int}
    prewhiten::Union{Nothing, Bool}
    bound::Union{Nothing, Bool}
    nboot::Int
    nfail::Int
end

function Base.getproperty(b::FEVDBootstrap, name::Symbol)
    name in fieldnames(FEVDBootstrap) && return getfield(b, name)
    return getproperty(getfield(b, :fevd), name)
end

# The bootstrap's own fields first (they take precedence in `getproperty`), then
# those of the wrapped LPFEVD that it does not shadow (`se_pw`, `se`)
function Base.propertynames(b::FEVDBootstrap)
    Tuple(unique((fieldnames(FEVDBootstrap)..., propertynames(getfield(b, :fevd))...)))
end

# Rows of `draws` (and of the optional LP draws) whose draw did not fail.
# `errors[b]` holds the exception of draw `b` (or `nothing`); the first one is
# reported so that a systematic failure is not reduced to a count.
function _fevd_draw_moments(draws::Matrix{Float64}, failures::Vector{Int},
        errors::Vector{Any}, what::String)
    nboot = length(failures)
    nfail = sum(failures)
    i = findfirst(!isnothing, errors)
    cause = i === nothing ? "" : "; first error: " * sprint(showerror, errors[i])
    nfail == nboot && throw(ErrorException(
        "fevdbootstrap ($what): all $nboot draws failed" * cause))
    nfail > 0 && @warn "fevdbootstrap ($what): $nfail of $nboot draws failed and " *
          "are excluded from the bias correction and the bands" * cause
    ok = failures .== 0
    kept = draws[ok, :]
    return kept, ok, nfail
end

_col_mean(D::Matrix{Float64}) = vec(mean(D; dims = 1))
_col_std(D::Matrix{Float64}) = vec(std(D; dims = 1))

# Default VAR ordering: the date-t controls, then the shock, then every other
# variable in order of appearance on the right-hand side(s), then the response
# if it only appears on the left.
function _fevd_default_vars(f::LPFEVD)
    rhs = StatsModels.termvars(f.formula.rhs)
    f.irf_formula === nothing || append!(rhs, StatsModels.termvars(f.irf_formula.rhs))
    order = unique(vcat(Symbol.(rhs), f.variables))
    rest = [v for v in order if !(v in f.contemporaneous)]
    return vcat(f.contemporaneous[2:end], f.shock, rest)
end

# The VAR lag order: fixed, or chosen among candidates by `_gl_lagselect`.
function _fevd_nlags(Y::Matrix{Float64}, nlags, criterion::Symbol)
    if nlags isa Integer
        nlags >= 1 ||
            throw(ArgumentError("the VAR lag order must be at least 1 (got $nlags)"))
        return Int(nlags), nothing
    end
    candidates = collect(Int, nlags)
    sel = _gl_lagselect(Y, candidates; criterion = criterion)
    return sel.p, (; criterion, candidates, values = sel.values)
end

# Pairs `y => v` for the levels `y` that the specification uses only through
# `lags(firstdiff(y), p)` and whose first difference is the column `v` of
# `candidates` (diff(y) ≈ v[2:end]).
function _fevd_level_pairs(f::LPFEVD, data::AbstractDataFrame,
        candidates::AbstractVector{Symbol})
    plain = Symbol[]
    diffs = Symbol[]
    for spec in (f.spec, f.irf_spec)
        spec === nothing && continue
        push!(plain, spec.response)
        append!(plain, spec.contemporaneous)
        for t in spec.info
            t isa LagTerm || continue
            t.term isa DiffTerm ? push!(diffs, t.term.term.sym) : push!(plain, t.term.sym)
        end
    end
    pairs = Pair{Symbol, Symbol}[]
    for s in unique(diffs)
        s in plain && continue
        ds = diff(Vector{Float64}(data[!, s]))
        for v in candidates
            v === s && continue
            hasproperty(data, v) || continue
            col = data[!, v]
            (eltype(col) <: Real && all(isfinite, col)) || continue
            if isapprox(ds, Vector{Float64}(col[2:end]); rtol = 1e-8)
                push!(pairs, s => v)
                break
            end
        end
    end
    return pairs
end

function _fevd_var(f::LPFEVD, data::AbstractDataFrame; vars, nlags, criterion::Symbol,
        skipfirst::Bool, burnin::Integer, nboot::Int, rng::AbstractRNG, threaded::Bool)
    f.trend == 0 || throw(ArgumentError(
        "method = :var is not available with `trend` ≥ 1: the reference simulates the " *
        "trend on the burn-in clock but estimates it on the sample clock; use " *
        "method = :block or :simulation"))
    nrow(data) == f.T || throw(ArgumentError(
        "`data` has $(nrow(data)) rows but the FEVD was estimated on $(f.T); pass the " *
        "data frame given to `fevd`"))
    burnin >= 0 || throw(ArgumentError("`burnin` must be non-negative (got $burnin)"))
    criterion in (:hqic, :bic, :aic) || throw(ArgumentError(
        "`criterion` must be :hqic, :bic or :aic (got :$criterion)"))
    skipfirst && f.lhs !== :cumul &&
        throw(ArgumentError(
            "`skipfirst` reproduces the reference's VAR on rows 2:T, which only arises with " *
            "a cumul(v) response; with ldiff(y) the VAR always uses every row"))
    # Levels that enter only through firstdiff() and whose difference is another
    # variable (cumul(dy) with lags(firstdiff(y), p)): the VAR simulates the
    # difference and each draw rebuilds the level by cumulation, as the reference
    # does (yb = cumsum(Δy_b)); a VAR in both would be singular.
    default = _fevd_default_vars(f)
    levels = _fevd_level_pairs(f, data, vars === nothing ? default : collect(Symbol, vars))
    if vars === nothing
        lmap = Dict(levels)
        varlist = unique([get(lmap, s, s) for s in default])
    else
        varlist = collect(Symbol, vars)
        for (s, d) in levels
            s in varlist && throw(ArgumentError(
                "`vars` holds both $s and its first difference $d, so the VAR would be " *
                "singular; drop $s: each draw rebuilds it by cumulating $d"))
        end
    end
    absent = setdiff(f.variables, varlist, first.(levels))
    isempty(absent) || throw(ArgumentError(
        "the FEVD specification uses $(absent), which are not in `vars` = $(varlist); " *
        "every variable the estimator needs must be simulated by the VAR"))
    j = findfirst(isequal(f.shock), varlist)
    r = findfirst(isequal(f.response), varlist)

    Yall = _var_data(data, varlist)
    Y = skipfirst ? Yall[2:end, :] : Yall
    p, ic = _fevd_nlags(Y, nlags, criterion)
    v = _var_ols(Y, p)
    Ω = (v.U' * v.U) ./ v.Tu
    _check_var_rank(Ω, varlist)
    H = f.horizon
    pseudo = _var_fevd(v.A, Ω, H; shock = j, response = r, cumulate = f.lhs === :cumul)

    T = f.T
    lpest = f.estimator !== :r2
    simulated = setdiff(f.variables, first.(levels))
    names = Tuple(vcat(simulated, first.(levels)))
    cols = [findfirst(isequal(s), varlist) for s in simulated]
    lcols = [findfirst(isequal(d), varlist) for d in last.(levels)]
    keep = (p + burnin):(p + burnin + T - 1)
    seeds = rand(rng, UInt64, nboot)
    draws = fill(NaN, nboot, H + 1)
    irf_draws = lpest ? fill(NaN, nboot, H + 1) : nothing
    irfstr_draws = lpest ? fill(NaN, nboot, H + 1) : nothing
    failures = zeros(Int, nboot)
    errors = Vector{Any}(nothing, nboot)

    function run_draw(b::Int)
        try
            drng = Xoshiro(seeds[b])
            # var_bootstrap.m: iid residuals (the first of burnin + T draws is
            # unused there, hence burnin + T - 1), then a random block of p
            # actual observations as the initial condition
            Ustar = v.U[rand(drng, 1:(v.Tu), burnin + T - 1), :]
            init = rand(drng, 1:(v.Tu + 1))
            Yb = _var_simulate(v.c, v.A, Ustar, Y[init:(init + p - 1), :])
            # the origin of a rebuilt level is irrelevant: it enters through firstdiff
            d = NamedTuple{names}((Tuple(Yb[keep, k] for k in cols)...,
                Tuple(accumulate(+, Yb[keep, k]) for k in lcols)...))
            all(c -> all(isfinite, c), d) || error("non-finite simulated data")
            est = _gl_point(f, d)
            all(isfinite, est.share) || error("non-finite share")
            draws[b, :] = est.share
            if lpest
                irf_draws[b, :] = est.psi
                irfstr_draws[b, :] = est.psi .* sqrt(est.σ²x)
            end
        catch err
            err isa InterruptException && rethrow()
            failures[b] = 1
            errors[b] = err
        end
        return nothing
    end
    _run_draws!(run_draw, nboot, threaded)

    kept, ok, nfail = _fevd_draw_moments(draws, failures, errors, "VAR bootstrap")
    m = _col_mean(kept)
    if lpest
        irf_kept = irf_draws[ok, :]
        irf_bc = 2 .* f.irf .- _col_mean(irf_kept)
        irf_se_boot = _col_std(irf_kept)
        irfstr_kept = irfstr_draws[ok, :]
    else
        irf_kept = irf_bc = irf_se_boot = irfstr_kept = nothing
    end
    return FEVDBootstrap(f, :var, kept, m, pseudo, f.share .- (m .- pseudo),
        _col_std(kept), copy(f.se_pw), copy(f.se), irf_kept, irf_bc, irf_se_boot,
        irfstr_kept, varlist, p, ic, skipfirst, Int(burnin), nothing, nothing, nothing,
        nboot, nfail)
end

function _fevd_block(f::LPFEVD; blocklength::Integer, nboot::Int, rng::AbstractRNG,
        threaded::Bool)
    f.estimator === :r2 || throw(ArgumentError(
        "method = :block (Kilian–Kim) is defined for the R² estimator only; use " *
        ":var or :simulation for LP-A/LP-B"))
    ℓ = Int(blocklength)
    ℓ >= 1 || throw(ArgumentError("`blocklength` must be at least 1 (got $ℓ)"))
    H = f.horizon
    f.nobs[end] >= ℓ || throw(ArgumentError(
        "`blocklength` = $ℓ exceeds the $(f.nobs[end]) observations of horizon $H"))
    des = f.design
    T, t0 = f.T, f.t0
    seeds = rand(rng, UInt64, nboot)
    draws = fill(NaN, nboot, H + 1)
    failures = zeros(Int, nboot)
    errors = Vector{Any}(nothing, nboot)

    function run_draw(b::Int)
        try
            drng = Xoshiro(seeds[b])
            rows = Int[]
            for h in 0:H
                Th = f.nobs[h + 1]
                nb = cld(Th, ℓ)
                resize!(rows, nb * ℓ)
                for k in 1:nb
                    start = rand(drng, t0:(T - h - ℓ + 1))
                    for i in 1:ℓ
                        rows[(k - 1) * ℓ + i] = start + i - 1
                    end
                end
                draws[b, h + 1] = _gl_r2_rows(des, view(rows, 1:Th), h)
            end
            all(isfinite, view(draws, b, :)) || error("non-finite share")
        catch err
            err isa InterruptException && rethrow()
            failures[b] = 1
            errors[b] = err
        end
        return nothing
    end
    _run_draws!(run_draw, nboot, threaded)

    kept, _, nfail = _fevd_draw_moments(draws, failures, errors, "block bootstrap")
    m = _col_mean(kept)
    return FEVDBootstrap(f, :block, kept, m, nothing, 2 .* f.share .- m, _col_std(kept),
        copy(f.se_pw), copy(f.se), nothing, nothing, nothing, nothing, nothing, nothing,
        nothing, nothing, nothing, ℓ, nothing, nothing, nboot, nfail)
end

function _fevd_simulation(f::LPFEVD; prewhiten::Bool, bound::Bool, nboot::Int,
        rng::AbstractRNG, threaded::Bool)
    H = f.horizon
    est = f.estimator
    clamp01 = bound && est === :r2
    Vs = prewhiten ? f.avar_pw : f.avar
    factors = Union{Nothing, Matrix{Float64}}[all(isfinite, Vs[h + 1]) ?
                                              _gl_psd_factor(Vs[h + 1]) : nothing
                                              for h in 0:H]
    skipped = [h for h in 0:H if factors[h + 1] === nothing]
    hint = prewhiten && any(h -> all(isfinite, f.avar[h + 1]), skipped) ?
           " The covariance without prewhitening is available at some of them " *
           "(`prewhiten = false`)." : ""
    isempty(skipped) || @warn "fevdbootstrap (simulation): the " *
          "$(prewhiten ? "prewhitened " : "")asymptotic variance is not available at " *
          "horizons $(skipped) (see the warnings of `fevd`); the bias-corrected share " *
          "and its standard errors are NaN there." * hint

    seeds = rand(rng, UInt64, nboot)
    draws = fill(NaN, nboot, H + 1)
    failures = zeros(Int, nboot)
    function run_draw(b::Int)
        drng = Xoshiro(seeds[b])
        for h in 0:H
            M = factors[h + 1]
            M === nothing && continue
            s = _gl_sim_share(drng, f.theta[h + 1], M, est, h)
            isfinite(s) || (failures[b] = 1)
            draws[b, h + 1] = clamp01 ? clamp(s, 0.0, 1.0) : s
        end
        return nothing
    end
    _run_draws!(run_draw, nboot, threaded)

    kept, _, nfail = _fevd_draw_moments(draws, failures, Vector{Any}(nothing, nboot), "simulation")
    m = _col_mean(kept)
    bc = 2 .* f.share .- m
    # Delta-method standard errors at the bias-corrected share
    # (SimulationAVarR2.m:180-190, SimulationAVarLP*.m:319-327)
    se_pw = fill(NaN, H + 1)
    se = fill(NaN, H + 1)
    for h in 0:H
        θ = f.theta[h + 1]
        if est === :r2
            g = _gl_r2_gradient(θ, bc[h + 1])
        else
            Nq = (length(θ) - 2) ÷ (h + 1)
            g = _gl_lp_gradient(est, view(f.irf, 1:(h + 1)), f.shock_var, θ[end],
                f.mse[h + 1], bc[h + 1], Nq)
        end
        se_pw[h + 1] = _gl_delta_se(g, f.avar_pw[h + 1])
        se[h + 1] = _gl_delta_se(g, f.avar[h + 1])
    end
    # Only horizons with a bias-corrected share: where the covariance used for the
    # draws is missing, bc (and so the gradient) is NaN, already reported above
    negative = [h
                for h in 0:H
                if isfinite(bc[h + 1]) &&
        ((isnan(se_pw[h + 1]) && all(isfinite, f.avar_pw[h + 1])) ||
         (isnan(se[h + 1]) && all(isfinite, f.avar[h + 1])))]
    isempty(negative) || @warn "fevdbootstrap (simulation): the delta-method variance " *
          "at the bias-corrected share is negative at horizons $(negative) (the " *
          "covariance is not positive semidefinite); the standard errors are NaN there."
    return FEVDBootstrap(f, :simulation, kept, m, nothing, bc, _col_std(kept), se_pw, se,
        nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing,
        nothing, prewhiten, clamp01, nboot, nfail)
end

"""
    fevdbootstrap(f::LPFEVD, data = nothing; method = :var, nboot = 2000,
                  rng = Random.default_rng(), threaded = false, kwargs...) -> FEVDBootstrap

Bias correction of a Gorodnichenko–Lee forecast-error variance decomposition
(see [`fevd`](@ref)) by one of the three procedures of the paper. Each produces
draws ``s^*_{b,h}``, ``b = 1, \\dots, B``, of the estimator, with mean
``\\bar m_h``:

| `method` | estimators | draws | bias-corrected share |
|---|---|---|---|
| `:var` | R², LP-A, LP-B | the estimator re-run on samples simulated from a VAR fitted to `data` | ``\\hat s_h - (\\bar m_h - s^{VAR}_h)`` |
| `:block` | R² | the R² re-computed on moving blocks of regression rows | ``2\\hat s_h - \\bar m_h`` |
| `:simulation` | R², LP-A, LP-B | ``\\theta^* \\sim N(\\hat\\theta_h, \\hat V_h)`` mapped into the share | ``2\\hat s_h - \\bar m_h`` |

The bias-corrected share is never clamped to ``[0, 1]``.

# VAR bootstrap (`method = :var`, `BootstrapVAR{R2,LPA,LPB}.m`)
A reduced-form VAR is fitted by least squares to the variables `vars` of
`data` (the data frame given to `fevd`). In each draw its residuals are
resampled iid, the VAR is iterated from a randomly drawn block of ``p`` actual
observations for `burnin` periods plus the ``T`` rows of the data, and the
**same** FEVD specification (estimator, formulas, `dofadjust`) is re-estimated
on the last ``T`` rows, without re-selecting any lag. The pseudo-truth
``s^{VAR}_h`` is the FEVD of the fitted VAR for the Cholesky shock of the shock
variable, with one-standard-deviation shocks; for a `cumul(v)` response the
response row is cumulated, so that ``s^{VAR}_h`` refers to the same
``y_{t+h} - y_{t-1}`` as the estimator.

- `vars`: the VAR variables in Cholesky order. They must include every variable
  of the specification (the response is `v` itself for `cumul(v)` and `y` in
  levels for `ldiff(y)`), so that each draw can rebuild all regressors; further
  variables may be added. The exception is a level `y` that enters only through
  `lags(firstdiff(y), p)` while its first difference `v` (``\\Delta y_t = v_t``
  in `data`) is in the VAR, e.g. `cumul(dy)` with `lags(firstdiff(y), 1)`: `y` is
  then left out of the VAR (a VAR in both would be singular) and each draw
  rebuilds it by cumulating the simulated `v`, as the reference does. The default orders the date-``t`` controls first,
  then the shock, then the remaining variables of the formula in order of
  appearance (for the application of the paper this gives
  `[tfp, dy, infl, rr, effr]` for both shocks). ``s^{VAR}_h``, and hence the
  bias correction, depends on this order.
- `nlags = 1:10`: a fixed lag order, or candidates (each ``\\ge 1``) among which
  `criterion` (`:hqic` (default), `:bic` or `:aic`) chooses on a common sample
  with the maximum-likelihood residual covariance and ``n^2 p`` parameters.
- `skipfirst = false`: with a `cumul(v)` response, `true` drops the first row
  from the VAR, as the reference does when it forms ``\\Delta y`` from the level
  (its VAR then has ``T - 1`` rows while the draws still have ``T``); use it to
  reproduce the reference's lag choice and pseudo-truth exactly.
- `burnin = 100`.

For LP-A/B the result also holds the impulse-response draws and the
bias-corrected responses ``2\\hat\\psi_h - \\mathrm{mean}_b\\,\\psi^*_{b,h}``.
Not available with `trend ≥ 1`.

# Block bootstrap (`method = :block`, R² only, `BootstrapBlockR2.m`)
For each horizon separately, ``\\lceil T_h/\\ell \\rceil`` blocks of `blocklength
= 4` consecutive regression rows ``(y_{t+h} - y_{t-1}, W_{t-1}, x_t, \\dots,
x_{t+h})`` are drawn from the window ``t_0, \\dots, T-h`` (block starts uniform,
non-circular), concatenated and truncated to ``T_h`` rows; the data are not
re-simulated and the shock is not re-demeaned. The reference also draws and
discards 100 burn-in rows, which is equivalent when 100 is a multiple of
``\\ell``. Requires ``T_H \\ge \\ell``.

# Asymptotic simulation (`method = :simulation`, `SimulationAVar*.m`, Appendices A/B)
For each horizon separately, ``\\theta^*`` is drawn from the normal
approximation ``N(\\hat\\theta_h, \\hat V_h)`` stored by `fevd` (`theta`,
`avar_pw` with `prewhiten = true`, else `avar`), and mapped into the share:
``\\theta_1^{*\\prime}\\theta_2^*/\\theta_3^*`` for R² (``\\theta_1 =
\\Sigma_x^{-1}\\theta_2`` is not imposed), and the LP-A / LP-B formulas in the
drawn ``\\psi``, ``\\sigma^2_x`` and ``\\sigma^2_h`` for LP-A/B. Draws with a
nonpositive variance (``\\theta_3`` for R², ``\\sigma^2_x`` or ``\\sigma^2_h``
for LP-A/B) are redrawn. A covariance that is not positive definite is repaired
by flooring its eigenvalues at `eps()` (used for drawing only). With
`bound = true` (default) the R² draws are clamped to ``[0, 1]`` before their
mean is taken; LP-A/B draws are never clamped. Horizons whose covariance is
`NaN` in `fevd` give `NaN`. `se_pw` and `se` are the delta-method standard errors
evaluated at the **bias-corrected** share, the authors' choice
(`SimulationAVarR2.m:180-182`); `fevd` reports them at ``\\hat s_h``.

# Reproducibility
Per-draw seeds are drawn from `rng` before the loop (draw ``b`` uses
`Xoshiro(seed_b)` for all its horizons), so `threaded = true` and
`threaded = false` give bit-identical results. Draws that fail (an exception or
a non-finite share) are excluded, counted in `nfail`, and reported with a
warning; the reference has no such handling.

# Example
```julia
f = fevd(@formula(cumul(dy) ~ tfp + lags(tfp, 4) + lags(dy, 4) + lags(infl, 4) +
                              lags(rr, 4) + lags(effr, 4)), df; horizon = 20)
b = fevdbootstrap(f, df; method = :var, nboot = 2000, rng = Xoshiro(1))
b.biascorrected
fevdbootstrap(f; method = :block, rng = Xoshiro(1))
fevdbootstrap(f; method = :simulation, rng = Xoshiro(1))
```
"""
function fevdbootstrap(f::LPFEVD, data::Union{Nothing, AbstractDataFrame} = nothing;
        method::Symbol = :var,
        nboot::Integer = 2000,
        rng::AbstractRNG = Random.default_rng(),
        threaded::Bool = false,
        vars::Union{Nothing, AbstractVector{Symbol}} = nothing,
        nlags::Union{Integer, AbstractVector{<:Integer}} = 1:10,
        criterion::Symbol = :hqic,
        skipfirst::Bool = false,
        burnin::Integer = 100,
        blocklength::Integer = 4,
        prewhiten::Bool = true,
        bound::Bool = true)
    method in _FEVD_METHODS || throw(ArgumentError(
        "`method` must be :var, :block or :simulation (got :$method)"))
    nboot >= 2 || throw(ArgumentError(
        "`nboot` must be at least 2 (got $nboot): the draws' standard deviation " *
        "needs two of them"))
    B = Int(nboot)
    if method === :var
        data === nothing && throw(ArgumentError(
            "method = :var fits a VAR to the data; pass the data frame given to `fevd`"))
        return _fevd_var(f, data; vars, nlags, criterion, skipfirst, burnin,
            nboot = B, rng, threaded)
    elseif method === :block
        return _fevd_block(f; blocklength, nboot = B, rng, threaded)
    end
    return _fevd_simulation(f; prewhiten, bound, nboot = B, rng, threaded)
end

function fevdbootstrap(f::LPFEVD, data; kwargs...)
    fevdbootstrap(f, DataFrame(data); kwargs...)
end

const _FEVD_METHOD_LABELS = Dict(:var => "VAR residual bootstrap",
    :block => "moving-block bootstrap", :simulation => "asymptotic simulation")

function Base.show(io::IO, b::FEVDBootstrap)
    print(io, "FEVDBootstrap(estimator=:", b.estimator, ", method=:", b.method,
        ", horizon=0:", b.horizon, ", nboot=", b.nboot, ")")
end

function Base.show(io::IO, ::MIME"text/plain", b::FEVDBootstrap)
    f = b.fevd
    println(io, "FEVDBootstrap (Gorodnichenko–Lee ", _fevd_label(f.estimator), ", ",
        _FEVD_METHOD_LABELS[b.method], ")")
    println(io, "  Response:   ", f.spec.lhs_term)
    println(io, "  Shock:      ", f.shock)
    if b.method === :var
        println(io, "  VAR:        ", join(string.(b.vars), ", "), "   (p = ", b.nlags,
            b.ic === nothing ? "" : ", by :$(b.ic.criterion)", ")",
            b.skipfirst ? ", rows 2:T" : "", ", burn-in ", b.burnin)
    elseif b.method === :block
        println(io, "  Blocks:     length ", b.blocklength)
    else
        println(io, "  Covariance: ", b.prewhiten ? "prewhitened" : "not prewhitened",
            b.bound ? ", R² draws clamped to [0, 1]" : "")
    end
    println(io, "  Draws:      ", b.nboot, b.nfail > 0 ? " ($(b.nfail) failed)" : "")
    hz = collect(0:(f.horizon))
    if b.pseudo_truth === nothing
        data = (; h = hz, s = f.share, bc = b.biascorrected, m = b.mean_draws,
            sb = b.se_boot)
        labels = ["Horizon", "Share", "Bias-corr.", "Mean draws", "s.e. (draws)"]
    else
        data = (; h = hz, s = f.share, bc = b.biascorrected, m = b.mean_draws,
            pt = b.pseudo_truth, sb = b.se_boot)
        labels = ["Horizon", "Share", "Bias-corr.", "Mean draws", "VAR share",
            "s.e. (draws)"]
    end
    nc = length(labels)
    pretty_table(io, data;
        column_labels = labels,
        title = "Forecast-error variance share of $(f.shock)",
        formatters = [fmt__round(4, collect(2:nc))],
        alignment = fill(:r, nc),
        table_format = TextTableFormat(borders = text_table_borders__unicode_rounded))
    return nothing
end
