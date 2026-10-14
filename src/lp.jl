"""
    lp(formula, data; horizon, shock=nothing, sample=:horizon)

Estimate local projections implied by `formula` up to the supplied `horizon`.
`shock` selects the coefficient path of interest (defaults to the first RHS term).

# Estimation sample

With the default `sample = :horizon` every horizon is estimated on all the
observations available to it, so the sample shrinks as ``h`` grows (the
horizon-``h`` regression loses the last ``h`` rows to the lead of the
response). This uses all the data and is the more efficient choice.

`sample = :common` estimates every horizon on one sample: the observations that
are complete at *every* horizon ``0, \\dots, H``. Without internal gaps this is
the sample of the maximum horizon, and horizon 0 loses ``H`` observations; an
internal `NaN` in the response removes, at every horizon, each row whose lead
hits it at some horizon (up to ``H + 1`` rows per gap), so the common sample is
then smaller than the horizon-``H`` one. Responses at different horizons are
comparable observation by observation (the IRF path is not affected by a
changing sample). The
choice is stored in the result's `sample` field and reused by `varbootstrap`;
`biascorrect` and `fevd` require `sample = :horizon`.

```julia
r = lp(@formula(leads(y) ~ x + lags(y, 4)), df; horizon = 20, sample = :common)
nobs.(r.models)   # the same number of observations at every horizon
```

# Horizon-tracking right-hand-side terms

`cumul(x)` and `leads(x)` are normally left-hand-side transforms, but they are
also accepted on the right-hand side, where they *track the projection
horizon*: at horizon `h` the column is rebuilt as the cumulative sum
`x_t + ... + x_{t+h}` (or the lead `x_{t+h}`) rather than being computed once.
Writing an explicit horizon — `cumul(x, 3)` — pins the column instead, and it
is then constant across horizons like any other regressor.

The coefficient keeps the horizon-free name (`"cumul(x)"`, not `"cumul(x, 0)"`,
`"cumul(x, 1)"`, ...), so `coefpath`, `summarize` and the plot recipes work
unchanged. When no horizon-tracking term appears on the RHS the design matrix
is still built exactly once, as before.

# State-dependent projections

`state = :s` estimates the state-dependent local projection of Ramey and
Zubairy (2018),

```math
y_{t+h} = (1 - I_{t-1})\\,[\\alpha_{0,h} + \\beta_{0,h} x_t + \\gamma_{0,h}' w_t]
        + I_{t-1}\\,[\\alpha_{1,h} + \\beta_{1,h} x_t + \\gamma_{1,h}' w_t] + u_{t+h},
```

where `s` is an indicator column of `data` (a weight in ``[0, 1]``, such as the
Auerbach–Gorodnichenko transition function, is also accepted). *Every*
regressor, the intercept included, is split by regime, which spans the same
column space as the hand-built `L.state rec* exp*` lists of `jordagk.do`.
`statelag` (default 1) is the lag at which the state enters; pass 0 when the
column is already timed.

Coefficient names get a regime suffix: `x_s0` for the regime ``1 - I`` and
`x_s1` for the regime ``I``, or `regimes = ("exp", "rec")` for `x_exp` / `x_rec`.
`shock = :x` designates both; the default term of `coefpath` and `summarize` is
then the first regime, and the second is `term = :x_s1`. Everything downstream
(`vcov`, `summarize`, the plot recipes) works per term, and
[`statetest`](@ref) tests equality of the two regimes horizon by horizon.
`biascorrect` and `varbootstrap` reject state-dependent projections: both rest
on a linear VAR.

```julia
r = lp(@formula(leads(y) ~ newsy + lags(newsy, 4) + lags(y, 4) + lags(g, 4)), rz;
       horizon = 20, state = :slack, regimes = ("exp", "rec"))
summarize(r, Bartlett(6); term = :newsy_rec)
statetest(r, Bartlett(6))
```

See also [`lpiv`](@ref), whose endogenous and instrument blocks accept the same
terms.
"""
function lp(formula::FormulaTerm, data::AbstractDataFrame;
        horizon::Integer, shock::Union{Symbol, Nothing} = nothing,
        state::Union{Symbol, Nothing} = nothing, statelag::Integer = 1,
        regimes = nothing, sample::Symbol = :horizon)
    horizon < 0 && throw(ArgumentError("horizon must be non-negative"))
    _check_sample(sample)
    df_base = DataFrame(data)  # avoid mutating caller's data

    # Apply schema to formula to convert FunctionTerms to proper terms (CumulTerm, LagTerm, etc.)
    # Collect all variable names from the formula
    all_vars = StatsModels.termvars(formula)

    # Create hints to treat all variables as continuous (not categorical)
    # This prevents StatsModels from treating numeric columns with many unique values as categorical
    hints = Dict{Symbol, Any}(var => StatsModels.ContinuousTerm for var in all_vars)

    sch = StatsModels.schema(formula, df_base, hints)
    lhs_term = StatsModels.apply_schema(formula.lhs, sch, StatisticalModel)
    rhs_term = StatsModels.apply_schema(formula.rhs, sch, StatisticalModel)

    # The response variable named by the LHS; the per-horizon response itself
    # is rebuilt from `lhs_term` by `_lhs_at` (leads, cumul, anchor, ldiff, hbr)
    response = _lhs_response(lhs_term)

    # Extract all variable names from RHS (including from function terms)
    rhs_terms = StatsModels.termvars(rhs_term)
    isempty(rhs_terms) &&
        throw(ArgumentError("formula must contain at least one regressor"))

    # ========================================================================
    # Stage 1: Remove rows with missing values in base variables
    # Extract all base variables (without transformations) from formula
    # ========================================================================
    base_vars_lhs = _extract_base_variables(formula.lhs)
    base_vars_rhs = _extract_base_variables(formula.rhs)
    base_vars = unique(vcat(base_vars_lhs, base_vars_rhs))

    # Keep only complete cases (remove rows with missing base variables). The
    # state column is masked through its NaN weights instead: it is typically
    # undefined over a few leading rows that no regression reaches.
    df_base_complete = dropmissing(df_base, base_vars, disallowmissing = true)

    # ========================================================================
    # Stage 2: Compute X matrix ONCE (RHS is constant across all horizons)
    # ========================================================================

    # Create a dummy formula with the response on LHS to compute RHS ModelFrame
    # We use the original response variable since it exists in df_base
    dummy_formula = StatsModels.FormulaTerm(StatsModels.Term(response), formula.rhs)

    # Create ModelFrame for the RHS computation (only once!)
    mf_base = StatsModels.ModelFrame(dummy_formula, df_base_complete)

    # Store coefficient names (constant across all horizons, taken from the
    # unspecialised terms so a horizon-tracking cumul(x) stays named "cumul(x)")
    coef_names_base = Vector{String}(coefnames(mf_base))

    # Build the X matrix. A bare cumul(x)/leads(x) on the RHS tracks the
    # projection horizon, so the design has to be rebuilt at every h; otherwise
    # it is constant and is built exactly once.
    rhs_applied = mf_base.f.rhs
    Xof = if _has_dynamic_horizon(rhs_applied)
        let rhs = rhs_applied, dat = mf_base.data
            h -> _as_float_matrix(StatsModels.modelcols(_specialize_horizon(rhs, h), dat))
        end
    else
        let X = _as_float_matrix(StatsModels.modelcols(rhs_applied, mf_base.data))
            h -> X
        end
    end

    if state !== nothing
        length(coef_names_base) >= 2 || throw(ArgumentError(
            "Cannot select a shock term: only an intercept found in the model."))
        labels = _state_labels(state, regimes)
        shock_symbol, base_shock = _resolve_state_shock(
            shock, coef_names_base[2], coef_names_base, labels)
        weights = _state_weights(df_base_complete, state, statelag)
        Xstate = let Xof = Xof, weights = weights
            h -> _split_by_state(Xof(h)::Matrix{Float64}, weights)
        end
        if !_has_dynamic_horizon(rhs_applied)
            Xstate = let X = Xstate(0)
                h -> X
            end
        end
        return _lp_estimate_horizons(Xstate, _state_names(coef_names_base, labels),
            df_base_complete, horizon, response, shock_symbol, formula, lhs_term,
            LPState(state, statelag, labels, base_shock), sample)
    end

    # Set shock variable: use provided shock or default to first RHS coefficient
    # Note: coef_names_base includes intercept, so we need the second element (first RHS term)
    if shock === nothing
        # Default to first RHS coefficient (skip intercept which is first)
        if length(coef_names_base) >= 2
            shock_symbol = Symbol(coef_names_base[2])
        else
            # Only intercept exists - cannot select a meaningful shock term
            throw(ArgumentError(
                "Cannot automatically select shock term: only intercept found in model. " *
                "Please provide a non-intercept regressor or specify `shock` explicitly."))
        end
    else
        shock_symbol = shock
    end

    # Function barrier: coef_names_base is concretely typed and `Xof` is a
    # closure returning Matrix{Float64}, so _lp_estimate_horizons can be fully
    # inferred by the compiler.
    return _lp_estimate_horizons(Xof, coef_names_base, df_base_complete, horizon,
        response, shock_symbol, formula, lhs_term, nothing, sample)
end

"""
    _lp_estimate_horizons(Xof, coef_names_base, df, horizon, response, shock, formula, ...)

Function barrier for type-stable per-horizon OLS estimation.
`Xof(h)` returns the `Matrix{Float64}` design at horizon `h` — the same matrix
every time unless the RHS carries a horizon-tracking `cumul`/`leads` term.
With `sample = :common` every horizon uses the rows complete at all horizons.
"""
function _lp_estimate_horizons(Xof::F, coef_names_base::Vector{String},
        df_base_complete, horizon, response, shock_symbol, formula, lhs_term,
        state::Union{Nothing, LPState} = nothing, sample::Symbol = :horizon) where {F}
    # Design, response and complete-case rows (no NaN in X or y) at horizon h
    function _horizon_data(h)
        X = Xof(h)::Matrix{Float64}
        # Convert to Vector{Float64} for type stability (modelcols may return ShiftedArray)
        y_h = Vector{Float64}(StatsModels.modelcols(_lhs_at(lhs_term, h), df_base_complete))
        return X, y_h, vec(all(!isnan, X, dims = 2)) .& .!isnan.(y_h)
    end
    common = sample === :common ? _common_rows(h -> _horizon_data(h)[3], horizon) :
             nothing

    # Helper to estimate one horizon
    function _estimate_horizon(h)
        X, y_h, complete_rows = _horizon_data(h)
        common === nothing || (complete_rows = common)
        sum(complete_rows) == 0 &&
            throw(ArgumentError("No complete observations available for horizon $h"))
        y = view(y_h, complete_rows)
        x = view(X, complete_rows, :)
        return ols(x, y; has_intercept = false)
    end

    # Estimate first model to get concrete type for typed vector
    first_model = _estimate_horizon(0)

    # Verify shock variable is present
    available = Symbol.(coef_names_base)
    shock_symbol in available ||
        throw(ArgumentError("shock term $(shock_symbol) not present in model"))

    models = Vector{typeof(first_model)}(undef, horizon + 1)
    models[1] = first_model

    for (i, h) in enumerate(1:horizon)
        models[i + 1] = _estimate_horizon(h)
    end

    return LocalProjection{eltype(models)}(
        models, horizon, response, shock_symbol, formula, coef_names_base,
        _lhs_tautological(lhs_term) &&
            response === (state === nothing ? shock_symbol : state.shock), state, sample)
end
