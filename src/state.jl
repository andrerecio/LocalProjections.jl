# ============================================================================
# State-dependent local projections (Ramey & Zubairy 2018)
# `lp`/`lpiv` with `state = :s` split every column of the design by regime,
# [B .* (1 - I_{t-lag}), B .* I_{t-lag}], block by block (exogenous incl. the
# intercept, endogenous, instruments); names get a `_label` suffix.
# ============================================================================

"""
    LPState

State-dependence specification attached to a local projection estimated with
the `state` keyword of [`lp`](@ref) / [`lpiv`](@ref).

- `var`: the state column, an indicator (or a weight in ``[0, 1]``)
- `lag`: the state enters the horizon-``h`` regression as ``I_{t-lag}``
- `labels`: coefficient-name suffixes of the regimes `1 - I` and `I`
- `shock`: the shock term before it was split by regime
"""
struct LPState
    var::Symbol
    lag::Int
    labels::NTuple{2, String}
    shock::Symbol
end

function _describe_state(st::LPState)
    return "$(st.var) lagged $(st.lag); regimes $(st.labels[1]) (1 - I) and $(st.labels[2]) (I)"
end

function _state_labels(state::Symbol, regimes)
    regimes === nothing && return ("$(state)0", "$(state)1")
    length(regimes) == 2 ||
        throw(ArgumentError("`regimes` must hold two labels, for the regimes 1 - I and I"))
    labels = (String(regimes[1]), String(regimes[2]))
    labels[1] == labels[2] && throw(ArgumentError("`regimes` labels must differ"))
    return labels
end

"""
    _state_weights(tbl, state, lag, groups) -> Vector{Float64}

The state column lagged `lag` periods within each block of rows in `groups` (the units
of a panel), as the weight ``I_{t-lag}`` of the second regime. Rows where the lag is
unavailable, or the state is `missing`/`NaN`, are `NaN` and drop out through the usual
complete-row mask.
"""
function _state_weights(tbl, state::Symbol, lag::Integer, groups)
    lag >= 0 || throw(ArgumentError("`statelag` must be non-negative"))
    haskey(tbl, state) || throw(ArgumentError("state column `$state` is not in the data"))
    raw = tbl[state]
    nonmissingtype(eltype(raw)) <: Real ||
        throw(ArgumentError("state column `$state` must be numeric or Bool"))
    s = fill(NaN, length(raw))
    for g in groups, i in (lag + 1):length(g)
        v = raw[g[i - lag]]
        s[g[i]] = ismissing(v) ? NaN : Float64(v)
    end
    all(v -> isnan(v) || 0.0 <= v <= 1.0, s) || throw(
        ArgumentError("state column `$state` must be an indicator or a weight in [0, 1]")
    )
    return s
end

# Every column once per regime: [X .* (1 - I), X .* I]
_split_by_state(X::Matrix{Float64}, s::Vector{Float64}) = hcat(X .* (1.0 .- s), X .* s)

function _state_names(names::Vector{String}, labels::NTuple{2, String})
    return vcat(names .* "_" .* labels[1], names .* "_" .* labels[2])
end

"""
    _resolve_state_shock(shock, coef_names, endo_names, labels) -> (Symbol, Symbol)

The stored shock (a regime-split name, so that `coefpath` and `summarize` keep a usable
default) and the shock before the split. `shock` may be the unsplit term (both regimes
are then the shock; the first is stored) or one of the split names. `coef_names` and
`endo_names` are the unsplit names.
"""
function _resolve_state_shock(shock, coef_names, endo_names, labels::NTuple{2, String})
    if shock !== nothing
        str = String(shock)
        for label in labels
            suffix = "_" * label
            base = chop(str; tail = length(suffix))
            endswith(str, suffix) && base in coef_names && return Symbol(shock), Symbol(base)
        end
    end
    base = _select_shock(shock, coef_names, endo_names)
    return Symbol(base, "_", labels[1]), base
end

# The terms whose h = 0 coefficient is 1 by construction when response == shock
function _shock_terms(lp::AbstractLocalProjection)
    st = lp.state
    st === nothing && return (lp.shock,)
    return (Symbol(st.shock, "_", st.labels[1]), Symbol(st.shock, "_", st.labels[2]))
end

# ============================================================================
# Test of equal responses across regimes
# ============================================================================

"""
    StateTestResult

Horizon-by-horizon Wald test that a coefficient is the same in the two regimes
of a state-dependent local projection; see [`statetest`](@ref). `diff` is the
second regime minus the first, `se` its standard error, `stat = (diff / se)^2`
and `pvalue` refers to ``\\chi^2_1`` (``F_{1,B}`` for an `EWC(B)` estimator).

A `StateTestResult` is a Tables.jl table with one row per horizon and columns
`horizon`, `diff`, `se`, `stat`, `pvalue`, so `DataFrame(result)` works.
"""
struct StateTestResult
    term::Symbol
    labels::NTuple{2, String}
    horizon::Vector{Int}
    diff::Vector{Float64}
    se::Vector{Float64}
    stat::Vector{Float64}
    pvalue::Vector{Float64}
end

Tables.istable(::Type{StateTestResult}) = true
Tables.columnaccess(::Type{StateTestResult}) = true

function Tables.columns(r::StateTestResult)
    return (horizon = r.horizon, diff = r.diff, se = r.se, stat = r.stat, pvalue = r.pvalue)
end

function Base.show(io::IO, r::StateTestResult)
    return print(
        io, "StateTestResult(term=$(r.term), $(r.labels[2]) - $(r.labels[1]), ",
        "horizon=$(first(r.horizon)):$(last(r.horizon)))"
    )
end

function Base.show(io::IO, ::MIME"text/plain", r::StateTestResult)
    return pretty_table(
        io, Any[r.horizon r.diff r.se r.stat r.pvalue];
        column_labels = ["Horizon", "Diff", "Std.Err.", "Wald", "p-value"],
        title = "H0: $(r.term)_$(r.labels[2]) = $(r.term)_$(r.labels[1])",
        formatters = [(v, i, j) -> v isa AbstractFloat ? round(v; digits = 4) : v],
        alignment = [:r, :r, :r, :r, :r],
        table_format = TextTableFormat(borders = text_table_borders__unicode_rounded)
    )
end

"""
    statetest(lp; term = <shock>) -> StateTestResult
    statetest(lp, estimator; term = <shock>) -> StateTestResult

Test, at every horizon, that `term` has the same coefficient in the two regimes of a
state-dependent [`lp`](@ref) or [`lpiv`](@ref) result. The Wald statistic uses the full
covariance matrix of each horizon model (the covariance between the two regime
coefficients included), from the estimator stored in `lp`; passing `estimator` is the
same as `statetest(lp + vcov(estimator))`. `term` is the name before the regime split
and defaults to the shock.

This is `test expf·cumulg = recf·cumulg` after the joint `ivreg2` regression of
`jordagk.do`. The p-value refers to ``\\chi^2_1``, or to ``F_{1,B}`` for `EWC(B)`. With a
tautological `h = 0` (response equal to the shock) the difference is identically zero
and the statistic is `NaN`.
"""
function statetest(lp::AbstractLocalProjection; term::Union{Symbol, Nothing} = nothing)
    st = lp.state
    st === nothing &&
        throw(ArgumentError("statetest needs a projection estimated with `state = ...`"))
    base = term === nothing ? st.shock : term
    i1 = findfirst(==(string(base, "_", st.labels[1])), lp.coef_names)
    i2 = findfirst(==(string(base, "_", st.labels[2])), lp.coef_names)
    (i1 === nothing || i2 === nothing) &&
        throw(ArgumentError("term $base not present in model"))
    n = lp.horizon + 1
    diff, se = Vector{Float64}(undef, n), Vector{Float64}(undef, n)
    for h in 0:lp.horizon
        if h == 0 && lp.tautological_h0
            diff[1], se[1] = 0.0, 0.0
            continue
        end
        model = horizonmodel(lp, h)
        b, V = coef(model), vcov(model)
        diff[h + 1] = b[i2] - b[i1]
        se[h + 1] = sqrt(V[i1, i1] + V[i2, i2] - 2 * V[i1, i2])
    end
    stat = (diff ./ se) .^ 2
    estimator = _vcov_estimator(lp)
    dist = estimator isa EWC ? FDist(1, estimator.B) : Chisq(1)
    pvalue = [isnan(w) ? NaN : ccdf(dist, w) for w in stat]
    return StateTestResult(base, st.labels, collect(0:lp.horizon), diff, se, stat, pvalue)
end

function statetest(
        lp::AbstractLocalProjection,
        estimator::CovarianceMatrices.AbstractAsymptoticVarianceEstimator;
        kwargs...
    )
    return statetest(lp + vcov(estimator); kwargs...)
end

# ============================================================================
# Weak-instrument test of one regime
# ============================================================================

# Regime of a state-dependent projection, given as 1/2 or as its label
function _regime_index(st::LPState, regime)
    regime === nothing && throw(
        ArgumentError(
            "state-dependent projection: pass `regime = 1` ($(st.labels[1])) or " *
                "`regime = 2` ($(st.labels[2])); the Montiel Olea–Pflueger test covers one " *
                "endogenous regressor at a time"
        )
    )
    regime isa Integer && regime in (1, 2) && return Int(regime)
    idx = findfirst(==(String(regime)), collect(st.labels))
    idx === nothing && throw(
        ArgumentError(
            "unknown regime $regime; use 1, 2, \"$(st.labels[1])\" or \"$(st.labels[2])\""
        )
    )
    return idx
end

# The model `weakivtest` runs on at one horizon: the horizon model itself, or for a
# state-dependent projection the one-regime regression of `regime`.
function _weakiv_model(lpiv, model, regime)
    if lpiv.state === nothing
        regime === nothing ||
            throw(ArgumentError("`regime` applies to state-dependent projections only"))
        return model
    end
    return _regime_iv_model(model, _regime_index(lpiv.state, regime))
end

"""
    _regime_iv_model(model, r)

The one-regime regression of Ramey and Zubairy: the endogenous regressors and
instruments of regime `r` only, with the exogenous controls of *both* regimes.
The regime blocks of a state-dependent design do not overlap, so this has the
same regime-`r` coefficients as the joint model; it is what the weak-instrument
test, defined for a single endogenous regressor, is run on. It carries the
covariance estimator of `model`.
"""
function _regime_iv_model(model::IVMatrixEstimator, r::Int)
    pe = model.postestimation
    n_endo = pe.n_endogenous
    n_exo = size(pe.X, 2) - n_endo
    n_instr = size(pe.Z, 2) - n_exo
    k, q = n_endo ÷ 2, n_instr ÷ 2
    endo_cols = n_exo .+ (((r - 1) * k + 1):(r * k))
    instr_cols = n_exo .+ (((r - 1) * q + 1):(r * q))
    X = hcat(pe.X[:, 1:n_exo], pe.X[:, endo_cols])
    Z = hcat(pe.Z[:, 1:n_exo], pe.Z[:, instr_cols])
    aux = iv(TSLS(), Z, X, pe.y; has_intercept = model.has_intercept, n_endogenous = k)
    return aux + vcov(model.vcov_estimator)
end
