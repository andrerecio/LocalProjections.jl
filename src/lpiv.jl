# ============================================================================
# Instrumental Variables Local Projections
# ============================================================================

"""
    _parse_lpiv_formula(formula::FormulaTerm, sch)

Parse an IV formula to extract endogenous, instrument, and exogenous terms.
Handles `(x ~ z)` syntax for IV specification within the formula.

Returns:
- `lhs_term`: Applied schema LHS term
- `endo_terms`: Vector of endogenous variable terms
- `instr_terms`: Vector of instrument terms
- `exo_terms`: Vector of exogenous control terms
"""
function _parse_lpiv_formula(formula::FormulaTerm, sch)
    lhs_term = StatsModels.apply_schema(formula.lhs, sch, StatisticalModel)

    endo_terms = AbstractTerm[]
    instr_terms = AbstractTerm[]
    exo_terms = AbstractTerm[]

    # Handle RHS - can be single term, tuple of terms, or InteractionTerm
    rhs = formula.rhs isa Tuple ? formula.rhs : (formula.rhs,)

    for term in rhs
        if term isa FormulaTerm
            # IV specification: (endo ~ instruments)
            endo = StatsModels.apply_schema(term.lhs, sch, StatisticalModel)
            push!(endo_terms, endo)

            instrs = term.rhs isa Tuple ? term.rhs : (term.rhs,)
            for instr in instrs
                instr_applied = StatsModels.apply_schema(instr, sch, StatisticalModel)
                push!(instr_terms, instr_applied)
            end
        else
            # Regular exogenous term
            term_applied = StatsModels.apply_schema(term, sch, StatisticalModel)
            push!(exo_terms, term_applied)
        end
    end

    return lhs_term, endo_terms, instr_terms, exo_terms
end

"""
    _extract_all_vars_from_formula(formula::FormulaTerm) -> Vector{Symbol}

Extract all variable symbols from formula, including those nested in IV specs.
"""
function _extract_all_vars_from_formula(formula::FormulaTerm)
    all_vars = Symbol[]

    # LHS variables
    append!(all_vars, StatsModels.termvars(formula.lhs))

    # RHS - handle IV specs and regular terms
    rhs = formula.rhs isa Tuple ? formula.rhs : (formula.rhs,)
    for term in rhs
        if term isa FormulaTerm
            # IV specification: extract from both sides
            append!(all_vars, StatsModels.termvars(term.lhs))
            append!(all_vars, StatsModels.termvars(term.rhs))
        else
            append!(all_vars, StatsModels.termvars(term))
        end
    end

    return unique(all_vars)
end

"""
    lpiv(formula, data; horizon, shock=nothing)

Estimate local projections with instrumental variables from horizon 0 to `horizon`.

# Formula Syntax
```julia
lpiv(@formula(leads(y) ~ (x ~ z1) + lags(r, 5) + w), df; horizon=10)
```

Where:
- `leads(y)` - response with horizon transformation (also supports `cumul(y)`, `anchor(y, baseline)`)
- `(x ~ z1)` - x is endogenous, instrumented by z1
- `lags(r, 5)` and `w` - exogenous controls

# Horizon-tracking right-hand-side terms

A bare `cumul(x)` or `leads(x)` anywhere on the right-hand side — endogenous
regressor, instrument or exogenous control — tracks the projection horizon:
at horizon `h` its column is rebuilt as `x_t + ... + x_{t+h}` (or `x_{t+h}`)
instead of being computed once. `cumul(x, 3)` pins an explicit horizon and
stays constant. Coefficient names come from the unpinned terms, so they are
the same at every horizon.

This is what makes the Ramey-Zubairy (2018) cumulative fiscal multiplier a
single call: the endogenous regressor is cumulative spending through the same
horizon as the cumulative response.

```julia
res = lpiv(@formula(cumul(y) ~ (cumul(g) ~ bp) + lags(y, 4) + lags(g, 4)),
           rz; horizon = 20)

summarize(res, vcov(Bartlett{NeweyWest}(), res))   # multiplier path with s.e. and bands
```

The multiplier is the coefficient on `cumul(g)`, which `shock` picks up by
default. On how the automatic Newey-West bandwidth selected here compares with
Stata `ivreg2, bw(auto)` output, see section 8 of
`docs/src/inference_procedures_guide.md`.

# State-dependent projections

`state = :s` splits the whole regression by the lagged state ``I_{t-1}``, as in
[`lp`](@ref): exogenous controls (intercept included), endogenous regressors and
instruments each appear once per regime, so both regimes are estimated jointly
— the `ivreg2 … (expf·cumulg recf·cumulg = exp0shock rec0shock)` regression of
`jordagk.do`. Because the two blocks never overlap, the estimates and standard
errors equal those of Ramey and Zubairy's one-regime-at-a-time regressions.
`statelag` and `regimes` work as in `lp`.

```julia
m = lpiv(@formula(cumul(y) ~ (cumul(g) ~ newsy) + lags(newsy, 4) + lags(y, 4) + lags(g, 4)),
         rz; horizon = 20, state = :slack, regimes = ("exp", "rec"))
summarize(m, Bartlett(6); term = Symbol("cumul(g)_rec"))   # multiplier in slack
statetest(m, Bartlett(6))                                   # H0: equal multipliers
weakivtest(m + vcov(Bartlett(6)), 8; regime = 2)            # first stage, slack regime
```

# Arguments
- `formula::FormulaTerm`: A formula with IV specification using `(endo ~ instruments)` syntax
- `data::AbstractDataFrame`: DataFrame containing the variables
- `horizon::Integer`: Maximum forecast horizon (must be non-negative)
- `shock::Union{Symbol, Nothing}`: Optional coefficient to track (defaults to first endogenous)

# Returns
`LocalProjectionIV` struct containing IV models for each horizon.

# Example
```julia
using LocalProjections, DataFrames, StatsModels, CovarianceMatrices

# Generate data with endogeneity
n = 200
z = randn(n)
u = randn(n)
x = 0.5*z + 0.5*u      # Endogenous (correlated with u)
y = 1.0 .+ 2.0*x .+ u  # True effect = 2.0

df = DataFrame(y=y, x=x, z=z)

# Fit IV local projection
result = lpiv(@formula(leads(y) ~ (x ~ z)), df; horizon=5)

# Extract IRF (should be ~2.0, OLS would be biased)
irf = coefpath(result; term=:x)

# Apply HAC standard errors
result_hac = result + vcov(Bartlett{NeweyWest}())

# First-stage OLS regression (one per endogenous variable)
fs = first_stage(result, 0)
println("First-stage R²: ", r2(fs[1]))
println("First-stage coefs: ", coef(fs[1]))
```
"""
function lpiv(formula::FormulaTerm, data::AbstractDataFrame;
        horizon::Integer, shock::Union{Symbol, Nothing} = nothing,
        state::Union{Symbol, Nothing} = nothing, statelag::Integer = 1,
        regimes = nothing)
    horizon < 0 && throw(ArgumentError("horizon must be non-negative"))
    df_base = DataFrame(data)

    # Extract ALL variable names from formula (including those inside IV specs)
    all_vars = _extract_all_vars_from_formula(formula)

    # Create hints to treat all variables as continuous
    hints = Dict{Symbol, Any}(var => StatsModels.ContinuousTerm for var in all_vars)

    sch = StatsModels.schema(formula, df_base, hints)

    # Parse the IV formula
    lhs_term, endo_terms, instr_terms,
    exo_terms = _parse_lpiv_formula(formula, sch)

    # Validate IV specification
    isempty(endo_terms) &&
        throw(ArgumentError("No endogenous variables found. Use (endo ~ instruments) syntax."))
    isempty(instr_terms) &&
        throw(ArgumentError("No instruments found. Use (endo ~ instruments) syntax."))

    # Extract base variable names for filtering
    base_vars_lhs = _extract_base_variables(formula.lhs)
    base_vars_rhs = Symbol[]
    rhs = formula.rhs isa Tuple ? formula.rhs : (formula.rhs,)
    for term in rhs
        if term isa FormulaTerm
            append!(base_vars_rhs, _extract_base_variables(term.lhs))
            append!(base_vars_rhs, _extract_base_variables(term.rhs))
        else
            append!(base_vars_rhs, _extract_base_variables(term))
        end
    end
    base_vars = unique(vcat(base_vars_lhs, base_vars_rhs))

    # Response variable named by the LHS (leads, cumul, anchor, ldiff, hbr)
    response = _lhs_response(lhs_term)

    # Stage 1: Remove rows with missing base variables
    df_base_complete = dropmissing(df_base, base_vars, disallowmissing = true)

    # Stage 2: Build X (exogenous + endogenous) and Z (exogenous + instruments).
    # Built once unless some term tracks the horizon (a bare `cumul(x)` or
    # `leads(x)` anywhere on the RHS), in which case they are rebuilt per h.

    n_obs = nrow(df_base_complete)

    # Exogenous block (includes the intercept supplied by StatsModels)
    if !isempty(exo_terms)
        exo_tuple = length(exo_terms) == 1 ? exo_terms[1] : Tuple(exo_terms)
        dummy_formula_exo = StatsModels.FormulaTerm(StatsModels.Term(response), exo_tuple)
        mf_exo = StatsModels.ModelFrame(dummy_formula_exo, df_base_complete)
        exo_applied = mf_exo.f.rhs
        exo_data = mf_exo.data
        coef_names_exo = Vector{String}(coefnames(mf_exo))
    else
        # No exogenous terms - just intercept
        exo_applied = nothing
        exo_data = nothing
        coef_names_exo = ["(Intercept)"]
    end

    # Coefficient names come from the unspecialised terms, so they are the same
    # at every horizon even when the underlying columns are not.
    _term_names(t) = (n = StatsModels.coefnames(t); n isa Vector ? n : [n])
    endo_names = String[]
    for t in endo_terms
        append!(endo_names, _term_names(t))
    end
    instr_names = String[]
    for t in instr_terms
        append!(instr_names, _term_names(t))
    end
    coef_names_base = Vector{String}(vcat(coef_names_exo, endo_names))

    dynamic = _has_dynamic_horizon(endo_terms) || _has_dynamic_horizon(instr_terms) ||
              (exo_applied !== nothing && _has_dynamic_horizon(exo_applied))

    # State dependence: every block is split by regime, [B .* (1 - I), B .* I]
    state_spec = nothing
    state_weights = nothing
    if state !== nothing
        labels = _state_labels(state, regimes)
        shock_split, base_shock = _resolve_state_shock(
            shock, endo_names[1], coef_names_base, labels)
        shock = shock_split
        state_spec = LPState(state, statelag, labels, base_shock)
        state_weights = _state_weights(df_base_complete, state, statelag)
        coef_names_exo = _state_names(coef_names_exo, labels)
        endo_names = _state_names(endo_names, labels)
        instr_names = _state_names(instr_names, labels)
        coef_names_base = Vector{String}(vcat(coef_names_exo, endo_names))
    end

    # `h === nothing` builds the (horizon-invariant) design directly; an Int
    # pins every horizon-tracking term to that horizon first.
    function _build_design(h::Union{Int, Nothing})
        spec(t) = h === nothing ? t : _specialize_horizon(t, h)
        X_exo = exo_applied === nothing ? ones(n_obs, 1) :
                _as_float_matrix(StatsModels.modelcols(spec(exo_applied), exo_data))
        X_endo = reduce(hcat,
            (_as_float_matrix(StatsModels.modelcols(spec(t), df_base_complete))
            for t in endo_terms))::Matrix{Float64}
        Z_instr = reduce(hcat,
            (_as_float_matrix(StatsModels.modelcols(spec(t), df_base_complete))
            for t in instr_terms))::Matrix{Float64}
        if state_weights !== nothing
            X_exo = _split_by_state(X_exo, state_weights)
            X_endo = _split_by_state(X_endo, state_weights)
            Z_instr = _split_by_state(Z_instr, state_weights)
        end
        return (hcat(X_exo, X_endo)::Matrix{Float64},
            hcat(X_exo, Z_instr)::Matrix{Float64})
    end

    XZof = if dynamic
        h -> _build_design(h)
    else
        let XZ = _build_design(nothing)
            h -> XZ
        end
    end

    # Widths are horizon-invariant, so probing horizon 0 is enough.
    X_probe, Z_probe = XZof(0)
    n_exo = length(coef_names_exo)
    n_endogenous = size(X_probe, 2) - n_exo
    n_instruments = size(Z_probe, 2) - n_exo

    # Check order condition
    n_instruments >= n_endogenous ||
        throw(ArgumentError("Not enough instruments: $n_instruments < $n_endogenous (order condition violated)"))

    # Set shock variable
    if shock === nothing
        shock_symbol = Symbol(endo_names[1])
    else
        shock_symbol = shock
    end

    # Function barrier: coef_names_base is concretely typed and `XZof` returns a
    # pair of Matrix{Float64}, so _lpiv_estimate_horizons can be fully inferred.
    return _lpiv_estimate_horizons(XZof, coef_names_base, df_base_complete,
        horizon, n_endogenous, response, shock_symbol, formula,
        endo_names, instr_names, lhs_term, state_spec)
end

"""
    _lpiv_estimate_horizons(XZof, coef_names_base, df, ...)

Function barrier for type-stable per-horizon IV estimation.
`XZof(h)` returns the `(X_full, Z_full)` pair at horizon `h` — the same pair
every time unless some RHS term tracks the horizon.
"""
function _lpiv_estimate_horizons(XZof::F,
        coef_names_base::Vector{String}, df_base_complete,
        horizon, n_endogenous, response, shock_symbol, formula,
        endo_names, instr_names, lhs_term,
        state::Union{Nothing, LPState} = nothing) where {F}
    # Helper to estimate one horizon
    function _estimate_iv_horizon(h)
        X_full, Z_full = XZof(h)::Tuple{Matrix{Float64}, Matrix{Float64}}
        # Identify complete rows
        XZ_complete = vec(all(!isnan, X_full, dims = 2)) .&
                      vec(all(!isnan, Z_full, dims = 2))
        # Convert to Vector{Float64} for type stability (modelcols may return ShiftedArray)
        y_h = Vector{Float64}(StatsModels.modelcols(_lhs_at(lhs_term, h), df_base_complete))
        y_complete_rows = .!isnan.(y_h)
        complete_rows = XZ_complete .& y_complete_rows
        sum(complete_rows) == 0 &&
            throw(ArgumentError("No complete observations available for horizon $h"))
        y = view(y_h, complete_rows)
        X = view(X_full, complete_rows, :)
        Z = view(Z_full, complete_rows, :)
        return iv(TSLS(), collect(Z), collect(X), collect(y);
            has_intercept = false, n_endogenous = n_endogenous)
    end

    # Estimate first model to get concrete type
    first_model = _estimate_iv_horizon(0)

    # Verify shock variable is present
    available = Symbol.(coef_names_base)
    shock_symbol in available ||
        throw(ArgumentError("shock term $(shock_symbol) not present in model"))

    models = Vector{typeof(first_model)}(undef, horizon + 1)
    models[1] = first_model

    for (i, h) in enumerate(1:horizon)
        models[i + 1] = _estimate_iv_horizon(h)
    end

    return LocalProjectionIV{eltype(models)}(
        models, horizon, response, shock_symbol, formula, coef_names_base,
        endo_names, instr_names,
        _lhs_tautological(lhs_term) &&
            response === (state === nothing ? shock_symbol : state.shock), state)
end

"""
    first_stage(lpiv::LocalProjectionIV)

Return first-stage regressions and diagnostics for all horizons.

Returns a vector of `FirstStageIV`, one per horizon (0 to `lpiv.horizon`).
Each contains full OLS models plus non-robust and robust first-stage F-statistics.
"""
function Regress.first_stage(lpiv::LocalProjectionIV)
    return [first_stage(lpiv, h) for h in 0:lpiv.horizon]
end

"""
    first_stage(lpiv::LocalProjectionIV, h::Int) -> FirstStageIV

Return first-stage regressions and diagnostics for horizon `h` (0-indexed).

# Example
```julia
result = lpiv(@formula(leads(y) ~ (x ~ z1 + z2)), df; horizon=10)
fs = first_stage(result, 0)
coef(fs.models[1])    # first-stage coefficients
fs.F_nonrobust        # non-robust F
fs.F_robust           # robust F using the model's current vcov
```
"""
function Regress.first_stage(lpiv::LocalProjectionIV, h::Int)
    0 <= h <= lpiv.horizon ||
        throw(BoundsError("horizon $h out of range 0:$(lpiv.horizon)"))
    return Regress.first_stage(lpiv.models[h + 1];
        endogenous_names = lpiv.endogenous_names,
        instrument_names = lpiv.instrument_names)
end

# ============================================================================
# Weak IV Test for LocalProjectionIV
# ============================================================================

"""
    _warn_unsupported_weakiv_estimator(estimator)

The Montiel-Olea-Pflueger test in `Regress.jl` builds its weight matrix from
the covariance estimator attached to the model, but only kernel HAC
estimators (`CovarianceMatrices.HAC`, e.g. `Bartlett{NeweyWest}`) and
CR0/CR1 cluster estimators are actually supported: any other estimator —
notably `EWC`, and CR2/CR3 — silently falls back to an HC0-style
(heteroskedasticity-robust only) weight matrix upstream. Emit a warning so
that fallback is visible instead of silent.

Note that for `EWC` this is not merely a missing feature: the MOP limiting
distribution and critical values are derived under a consistently estimated
weight matrix, whereas the fixed-``B`` EWC covariance converges to a random
(Wishart-type) limit — the very non-degeneracy that motivates Student-``t_B``
critical values in HAR inference. EWC is therefore unsuitable for this test
by construction; use a kernel HAC estimator instead.
"""
function _warn_unsupported_weakiv_estimator(estimator)
    supported_cluster = estimator isa
                        Union{CovarianceMatrices.CR0, CovarianceMatrices.CR1}
    unsupported = estimator isa CovarianceMatrices.Correlated &&
                  !(estimator isa CovarianceMatrices.HAC) &&
                  !supported_cluster
    if unsupported
        @warn "weakivtest does not support $(typeof(estimator)): the test statistics " *
              "are computed with an HC0-style (heteroskedasticity-robust only) weight " *
              "matrix instead. Note that the Montiel-Olea-Pflueger critical values " *
              "assume a consistently estimated weight matrix, so fixed-smoothing " *
              "estimators such as EWC are unsuitable for this test by construction. " *
              "For serial-correlation-robust weak-IV inference attach a kernel HAC " *
              "estimator, e.g. `lpiv_result + vcov(Bartlett{NeweyWest}())`."
    end
    return nothing
end

"""
    weakivtest(lpiv::LocalProjectionIV, h::Int; kwargs...) -> WeakIVTestResult

Montiel-Olea-Pflueger robust weak instrument test for horizon `h` (0-indexed).

The test uses the covariance estimator attached to the horizon-`h` model
(`HR1` by default; set it with `lpiv_result + vcov(estimator)`). Kernel HAC
estimators are supported. Unsupported estimators trigger a warning and fall
back to heteroskedasticity-robust weighting; in particular `EWC` is
incompatible with this test by construction — its fixed-``B`` covariance is
not a consistent estimate of the weight matrix the MOP critical values
assume (see `_warn_unsupported_weakiv_estimator`).

# Keyword Arguments
- `level::Real=0.05`: Confidence level alpha
- `eps::Real=0.001`: Convergence tolerance for bias optimization
- `benchmark::Symbol=:nagar`: Bias benchmark (`:nagar` or `:ols`)
- `regime`: required for a state-dependent projection (`lpiv(...; state = ...)`), where
  the test covers one regime at a time — `1`/`2` or the regime label. It runs on
  Ramey and Zubairy's one-regime regression: that regime's endogenous regressor
  and instrument, with the exogenous controls of both regimes.

# Returns
A `WeakIVTestResult` containing effective F, robust F, critical values, etc.

See also: [`lpiv`](@ref), [`first_stage`](@ref)
"""
function Regress.weakivtest(lpiv::LocalProjectionIV, h::Int; regime = nothing, kwargs...)
    0 <= h <= lpiv.horizon ||
        throw(BoundsError("horizon $h out of range 0:$(lpiv.horizon)"))
    if lpiv.state !== nothing
        r = _regime_index(lpiv.state, regime)
        if h == 0 && lpiv.tautological_h0
            return _tautological_weakivtest(_regime_iv_model(lpiv.models[1], r); kwargs...)
        end
        _warn_unsupported_weakiv_estimator(lpiv.models[h + 1].vcov_estimator)
        return Regress.weakivtest(_regime_iv_model(lpiv.models[h + 1], r); kwargs...)
    end
    regime === nothing ||
        throw(ArgumentError("`regime` applies to state-dependent projections only"))
    if h == 0 && lpiv.tautological_h0
        @info "weakivtest at h=0 skipped: response == shock (tautological)"
        return _tautological_weakivtest(lpiv.models[1]; kwargs...)
    end
    _warn_unsupported_weakiv_estimator(lpiv.models[h + 1].vcov_estimator)
    return Regress.weakivtest(lpiv.models[h + 1]; kwargs...)
end

"""
    weakivtest(lpiv::LocalProjectionIV; kwargs...) -> Vector{WeakIVTestResult}

Montiel-Olea-Pflueger robust weak instrument test for all horizons.

Returns a vector of `WeakIVTestResult`, one per horizon (0 to `lpiv.horizon`).
The covariance estimator attached to the models is used for the weight
matrix; see [`weakivtest(::LocalProjectionIV, ::Int)`](@ref) for details.
"""
function Regress.weakivtest(lpiv::LocalProjectionIV; regime = nothing, kwargs...)
    if lpiv.state !== nothing
        r = _regime_index(lpiv.state, regime)
        _warn_unsupported_weakiv_estimator(lpiv.models[1].vcov_estimator)
        return map(enumerate(lpiv.models)) do (i, m)
            aux = _regime_iv_model(m, r)
            i == 1 && lpiv.tautological_h0 ?
            _tautological_weakivtest(aux; kwargs...) : Regress.weakivtest(aux; kwargs...)
        end
    end
    regime === nothing ||
        throw(ArgumentError("`regime` applies to state-dependent projections only"))
    _warn_unsupported_weakiv_estimator(lpiv.models[1].vcov_estimator)
    results = Vector{WeakIVTestResult{Float64}}(undef, lpiv.horizon + 1)
    for (i, m) in enumerate(lpiv.models)
        if i == 1 && lpiv.tautological_h0
            results[1] = _tautological_weakivtest(m; kwargs...)
        else
            results[i] = Regress.weakivtest(m; kwargs...)
        end
    end
    return results
end

"""
    _regime_iv_model(model, r)

The one-regime regression of Ramey and Zubairy: the endogenous regressors and
instruments of regime `r` only, with the exogenous controls of *both* regimes.
The regime blocks of a state-dependent design do not overlap, so this has the
same regime-`r` coefficients as the joint model; it is what the weak-instrument
test, defined for a single endogenous regressor, is run on.
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
    aux = iv(TSLS(), Z, X, pe.y; has_intercept = false, n_endogenous = k)
    return aux + vcov(model.vcov_estimator)
end

# Sentinel WeakIVTestResult for tautological h=0 (response == shock)
function _tautological_weakivtest(model; level = 0.05, kwargs...)
    T = Float64
    nan4 = (NaN, NaN, NaN, NaN)
    N = nobs(model)
    n_endo = model.postestimation.n_endogenous
    k_z = size(model.postestimation.Z, 2)
    k_x = size(model.postestimation.X, 2)
    K = k_z - k_x + n_endo  # number of excluded instruments
    return WeakIVTestResult{T}(
        Inf, Inf, Inf,          # F_eff, F_nonrobust, F_robust
        1.0, 0.0,               # btsls, sebtsls
        1.0, 0.0,               # bliml, sebliml
        1.0, 0.0,               # bgmmf, sebgmmf
        1.0,                    # kappa
        nan4, nan4, nan4,       # cv_TSLS, cv_LIML, cv_GMMf
        T(level), K, N
    )
end
