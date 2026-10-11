# ============================================================================
# State-dependent local projections (Ramey & Zubairy 2018)
# ============================================================================

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
    _state_weights(df, state, lag) -> Vector{Float64}

The state column lagged `lag` periods on the rows of `df`, as the weight
``I_{t-lag}`` of the second regime. Rows where the lag is unavailable are `NaN`
and drop out through the usual complete-row mask.
"""
function _state_weights(df::AbstractDataFrame, state::Symbol, lag::Integer)
    lag >= 0 || throw(ArgumentError("`statelag` must be non-negative"))
    raw = df[!, state]
    eltype(raw) <: Union{Missing, Real} ||
        throw(ArgumentError("state column $state must be numeric or Bool"))
    n = length(raw)
    s = fill(NaN, n)
    for t in (lag + 1):n
        v = raw[t - lag]
        s[t] = ismissing(v) ? NaN : Float64(v)
    end
    all(v -> isnan(v) || 0.0 <= v <= 1.0, s) || throw(ArgumentError(
        "state column $state must be an indicator or a weight in [0, 1]"))
    return s
end

# Every column once per regime: [X .* (1 - I), X .* I]
_split_by_state(X::Matrix{Float64}, s::Vector{Float64}) = hcat(X .* (1.0 .- s), X .* s)

function _state_names(names::Vector{String}, labels::NTuple{2, String})
    return vcat(names .* "_" .* labels[1], names .* "_" .* labels[2])
end

# `shock` may be the term before the split (both regimes are then the shock) or
# one of the split names; the stored shock is a split name so that `coefpath`
# and `summarize` keep a usable default.
function _resolve_state_shock(shock::Union{Symbol, Nothing}, default::String,
        names::Vector{String}, labels::NTuple{2, String})
    shock === nothing && return Symbol(default, "_", labels[1]), Symbol(default)
    str = String(shock)
    for label in labels
        suffix = "_" * label
        if endswith(str, suffix) && chop(str; tail = length(suffix)) in names
            return shock, Symbol(chop(str; tail = length(suffix)))
        end
    end
    str in names || throw(ArgumentError("shock term $shock not present in model"))
    return Symbol(str, "_", labels[1]), shock
end

# The terms whose h = 0 coefficient is 1 by construction when response == shock
function _shock_terms(lp)
    st = lp.state
    st === nothing && return (lp.shock,)
    return (Symbol(st.shock, "_", st.labels[1]), Symbol(st.shock, "_", st.labels[2]))
end

# Regime of a state-dependent projection, given as 1/2 or as its label
function _regime_index(st::LPState, regime)
    regime === nothing && throw(ArgumentError(
        "state-dependent projection: pass `regime = 1` ($(st.labels[1])) or " *
        "`regime = 2` ($(st.labels[2])); the Montiel Olea–Pflueger test covers one " *
        "endogenous regressor at a time"))
    regime isa Integer && regime in (1, 2) && return Int(regime)
    idx = findfirst(==(String(regime)), collect(st.labels))
    idx === nothing && throw(ArgumentError(
        "unknown regime $regime; use 1, 2, \"$(st.labels[1])\" or \"$(st.labels[2])\""))
    return idx
end

# ============================================================================
# Test of equal responses across regimes (state-dependent projections)
# ============================================================================

"""
    StateTestResult

Horizon-by-horizon Wald test that a coefficient is the same in the two regimes
of a state-dependent local projection; see [`statetest`](@ref). `diff` is the
second regime minus the first, `se` its standard error, `stat = (diff / se)^2`
and `pvalue` refers to ``\\chi^2_1`` (``F_{1,B}`` for an `EWC(B)` estimator).
`DataFrame(result)` gives the table.
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

function DataFrames.DataFrame(r::StateTestResult)
    return DataFrame(horizon = r.horizon, diff = r.diff, se = r.se, stat = r.stat,
        pvalue = r.pvalue)
end

function Base.show(io::IO, r::StateTestResult)
    print(io, "StateTestResult(term=$(r.term), $(r.labels[2]) - $(r.labels[1]), ",
        "horizon=$(first(r.horizon)):$(last(r.horizon)))")
end

function Base.show(io::IO, ::MIME"text/plain", r::StateTestResult)
    pretty_table(io, hcat(r.horizon, r.diff, r.se, r.stat, r.pvalue);
        column_labels = ["Horizon", "Diff", "Std.Err.", "Wald", "p-value"],
        title = "H0: $(r.term)_$(r.labels[2]) = $(r.term)_$(r.labels[1])",
        formatters = [fmt__round(4)], alignment = [:r, :r, :r, :r, :r],
        table_format = TextTableFormat(borders = text_table_borders__unicode_rounded))
end

"""
    statetest(lp, estimator; term = <shock>)

Test, at every horizon, that `term` has the same coefficient in the two regimes
of a state-dependent [`lp`](@ref) or [`lpiv`](@ref) result, using the full
covariance matrix from `estimator` (the covariance between the two regime
coefficients is needed, which [`LocalProjectionCovariance`](@ref) does not
keep). `term` is the name before the regime split and defaults to the shock.

This is `test expf·cumulg = recf·cumulg` after the joint `ivreg2` regression of
`jordagk.do`. With a tautological `h = 0` (response equal to the shock) the
difference is identically zero and the statistic is `NaN`.
"""
function statetest(lp::LPResult,
        estimator::CovarianceMatrices.AbstractAsymptoticVarianceEstimator;
        term::Union{Symbol, Nothing} = nothing)
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
    for (i, model) in enumerate(lp.models)
        if i == 1 && lp.tautological_h0
            diff[i], se[i] = 0.0, 0.0
            continue
        end
        b = coef(model)
        V = CovarianceMatrices.vcov(estimator, model)
        diff[i] = b[i2] - b[i1]
        se[i] = sqrt(V[i1, i1] + V[i2, i2] - 2 * V[i1, i2])
    end
    stat = (diff ./ se) .^ 2
    dist = estimator isa CovarianceMatrices.EWC ? FDist(1, estimator.B) : Chisq(1)
    pvalue = [isnan(w) ? NaN : ccdf(dist, w) for w in stat]
    return StateTestResult(base, st.labels, collect(0:lp.horizon), diff, se, stat, pvalue)
end
