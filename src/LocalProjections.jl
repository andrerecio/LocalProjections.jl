module LocalProjections

export LocalProjection, LocalProjectionIV, LocalProjectionCovariance, IRFSummary
export lp, lpiv, coefpath, stderror, vcov, summarize, first_stage, weakivtest
export ewc_bandwidth
export biascorrect, BiasCorrectedLP
export lagselect, VARLagSelection, nlags
export varbootstrap, LPBootstrap
export WeakIVTestResult, FirstStageIV
export lag, lead, cumul, CumulTerm, lags, leads, LeadTerm, anchor, AnchorTerm
export as_irf_result, LocalProjectionIRFResult
export irfplot, irfplot!, irfplot_axis

using DataFrames
using PrettyTables: pretty_table, TextHighlighter, TextTableFormat,
                    text_table_borders__unicode_rounded, fmt__round, @crayon_str
using Tables
using StatsModels
using StatsModels: AbstractTerm, Term, FunctionTerm, ConstantTerm, FormulaTerm,
                   ContinuousTerm, coefnames
using Regress
using Regress: OLSMatrixEstimator, IVMatrixEstimator, ols, iv, TSLS, VcovSpec,
               WeakIVTestResult, FirstStageIV, lags, LagTerm,
               first_stage, weakivtest
using CovarianceMatrices
using Random: AbstractRNG, Xoshiro
import Random
using Statistics
using LinearAlgebra: I, Symmetric, cond, eigvals, factorize, kron, logdet, tr
using Distributions
using RecipesBase
using ShiftedArrays: lag, lead
using StatsBase
# Extend (and re-export) the shared StatsAPI generics rather than defining our
# own functions: this keeps `vcov`/`stderror` the same binding as the one
# exported by StatsModels, CovarianceMatrices, and Regress, so unqualified
# calls work when those packages are loaded together with LocalProjections.
import StatsBase: vcov, stderror
using AxisArrays: AxisArrays, AxisArray, Axis
using MacroEconometricTools: LocalProjectionIRFResult, irfplot, irfplot!

# ============================================================================
# Helper Functions for Common Patterns
# ============================================================================

"""
    _parse_unary_binary_args(t::FunctionTerm, func_name::String, default_value)

Parse arguments from a FunctionTerm that accepts 1 or 2 arguments.
Returns (term, param_value) where param_value is either the provided value or default_value.
"""
function _parse_unary_binary_args(t::FunctionTerm, func_name::String, default_value)
    if length(t.args) == 1
        return (first(t.args), default_value)
    elseif length(t.args) == 2
        term, param_arg = t.args
        (param_arg isa ConstantTerm) ||
            throw(ArgumentError("$func_name parameter must be a number (got $param_arg)"))
        return (term, param_arg.n)
    else
        throw(ArgumentError("$func_name() requires 1 or 2 arguments"))
    end
end

"""
    _extract_single_column(cols, term_name::String="term")

Extract a single column from a matrix or vector, throwing an error if multiple columns.
Returns a Vector (using vec() for matrices).
"""
function _extract_single_column(cols, term_name::String = "term")
    if cols isa AbstractMatrix
        size(cols, 2) == 1 ||
            throw(ArgumentError("$term_name must be a single variable, got $(size(cols, 2)) columns"))
        return vec(cols)
    end
    return cols
end

"""
    _check_horizon_provided(horizon::Union{Int,Nothing}, func_name::String)

Check that horizon is not nothing (for standalone formula usage).
Throws error if horizon is nothing.
"""
function _check_horizon_provided(horizon::Union{Int, Nothing}, func_name::String)
    horizon === nothing &&
        throw(ArgumentError("$func_name() without explicit horizon can only be used in lp() context"))
    return horizon
end

"""
    _termvars_unary(t::FunctionTerm)

Extract termvars from a unary function term (extracts from first argument).
"""
function _termvars_unary(t::FunctionTerm)
    length(t.args) >= 1 && return StatsModels.termvars(t.args[1])
    return Symbol[]
end

# Add termvars support for lag/lead from ShiftedArrays (used in RHS formulas)
StatsModels.termvars(t::FunctionTerm{typeof(lag)}) = _termvars_unary(t)
StatsModels.termvars(t::FunctionTerm{typeof(lead)}) = _termvars_unary(t)

# Add termvars support for StatsModels.LeadLagTerm (created after apply_schema)
StatsModels.termvars(t::StatsModels.LeadLagTerm) = StatsModels.termvars(t.term)

"""
    _unwrap_lhs(lhs_term)

Unwrap LHS term to determine if it's anchored, cumulative, or leads.
Returns (is_anchor, is_cumul, is_leads, anchor_term, cumul_term, leads_term).
"""
function _unwrap_lhs(lhs_term)
    if lhs_term isa AnchorTerm
        inner = lhs_term.response
        is_cumul = inner isa CumulTerm
        is_leads = inner isa LeadTerm || !is_cumul  # Default to leads
        return (true, is_cumul, is_leads, lhs_term,
            is_cumul ? inner : nothing,
            is_leads && inner isa LeadTerm ? inner : nothing)
    else
        return (false, lhs_term isa CumulTerm, lhs_term isa LeadTerm,
            nothing, lhs_term isa CumulTerm ? lhs_term : nothing,
            lhs_term isa LeadTerm ? lhs_term : nothing)
    end
end

"""
    _extract_single_response(term, context::String)

Extract a single response variable from a term, throwing an error if multiple variables.
Returns the Symbol of the single base variable.
"""
function _extract_single_response(term, context::String)::Symbol
    vars = _extract_base_variables(term)
    length(vars) == 1 ||
        throw(ArgumentError("$context must reference a single base variable"))
    return vars[1]
end

"""
    _build_lhs_for_horizon(h::Int, is_anchor, is_cumul, is_leads, anchor_term, cumul_term, leads_term)

Build the LHS term for a specific horizon h, handling anchor/cumul/leads combinations.
"""
function _build_lhs_for_horizon(h::Int, is_anchor, is_cumul, is_leads,
        anchor_term, cumul_term, leads_term)
    if is_anchor
        inner = if is_cumul && cumul_term !== nothing
            CumulTerm{typeof(cumul_term.term)}(cumul_term.term, h)
        elseif is_leads && leads_term !== nothing
            LeadTerm{typeof(leads_term.term)}(leads_term.term, h)
        else
            LeadTerm{typeof(anchor_term.response)}(anchor_term.response, h)
        end
        return AnchorTerm{typeof(inner), typeof(anchor_term.anchor)}(
            inner, anchor_term.anchor, 0)
    elseif is_cumul
        return CumulTerm{typeof(cumul_term.term)}(cumul_term.term, h)
    elseif is_leads
        return LeadTerm{typeof(leads_term.term)}(leads_term.term, h)
    else
        throw(ArgumentError("Invalid LHS term type"))
    end
end

# ============================================================================
# Cumulative Sum Term (cumul)
# This implements cumul(y) for cumulative impulse responses
# Supports nested transforms like cumul(log(y))
# ============================================================================

"""
    cumul(term)
    cumul(term, horizon)

Create cumulative sum term. Used in formulas like:
- `@formula(cumul(y) ~ x)` - horizon determined by lp() context
- `@formula(cumul(y, 5) ~ x)` - explicit horizon for standalone use
- `@formula(cumul(log(y)) ~ x)` - supports nested transformations
"""
cumul(t::T) where {T <: AbstractTerm} = CumulTerm{T}(t, nothing)
cumul(t::T, h::Int) where {T <: AbstractTerm} = CumulTerm{T}(t, h)

# termvars: Extract variables from cumul() for schema creation
StatsModels.termvars(t::FunctionTerm{typeof(cumul)}) = _termvars_unary(t)

# Struct for cumulative sum term
struct CumulTerm{T <: AbstractTerm} <: AbstractTerm
    term::T                      # The term to cumulate (can be nested like log(y))
    horizon::Union{Int, Nothing}  # nothing in lp() context, Int for standalone
end

StatsModels.terms(t::CumulTerm) = (t.term,)
# termvars: needed once the schema has replaced the FunctionTerm, so that a
# horizon-tracking cumul(x) standing alone on the RHS still counts as a regressor.
StatsModels.termvars(t::CumulTerm) = StatsModels.termvars(t.term)

function StatsModels.apply_schema(
        t::FunctionTerm{typeof(cumul)}, sch::StatsModels.Schema, ctx::Type)
    term, horizon = _parse_unary_binary_args(t, "cumul", nothing)
    term = StatsModels.apply_schema(term, sch, ctx)
    return CumulTerm{typeof(term)}(term, horizon)
end

function StatsModels.apply_schema(t::CumulTerm, sch::StatsModels.Schema, ctx::Type)
    term = StatsModels.apply_schema(t.term, sch, ctx)
    CumulTerm{typeof(term)}(term, t.horizon)
end

# modelcols: Apply cumulative sum transformation
# Note: In lp() context, horizon is nothing and will be handled specially
function StatsModels.modelcols(ct::CumulTerm, d::Tables.ColumnTable)
    _check_horizon_provided(ct.horizon, "cumul")
    original_cols = StatsModels.modelcols(ct.term, d)
    original_cols = _extract_single_column(original_cols, "cumul() response")
    return _create_cumulative(original_cols, ct.horizon)
end

# width: Return number of columns (always 1 for cumul)
StatsModels.width(ct::CumulTerm) = 1

# show: Display the term
function Base.show(io::IO, ct::CumulTerm)
    if ct.horizon === nothing
        print(io, "cumul($(ct.term))")
    else
        print(io, "cumul($(ct.term), $(ct.horizon))")
    end
end

# coefnames: Return coefficient name
function StatsModels.coefnames(ct::CumulTerm)
    base_names = StatsModels.coefnames(ct.term)
    if ct.horizon === nothing
        return ["cumul(" * base_names[1] * ")"]
    else
        return ["cumul(" * base_names[1] * ", $(ct.horizon))"]
    end
end

# ============================================================================
# Lead Term (leads)
# This implements leads(y) for forward-looking regressions with NaN handling
# Named 'leads' to avoid conflict with ShiftedArrays.lead
# Uses our _lead_to_float64() for type stability (NaN instead of missing)
# ============================================================================

"""
    leads(term)
    leads(term, horizon)

Create lead term with NaN handling. Used in formulas like:
- `@formula(leads(y) ~ x)` - horizon determined by lp() context
- `@formula(leads(y, 3) ~ x)` - explicit horizon for standalone use
- `@formula(leads(log(y)) ~ x)` - supports nested transformations

Note: Named 'leads' to distinguish from ShiftedArrays.lead (which returns missing).
This version returns Float64 with NaN for type stability.
"""
leads(t::T) where {T <: AbstractTerm} = LeadTerm{T}(t, nothing)
leads(t::T, h::Int) where {T <: AbstractTerm} = LeadTerm{T}(t, h)

# termvars: Extract variables from leads() for schema creation
StatsModels.termvars(t::FunctionTerm{typeof(leads)}) = _termvars_unary(t)

# Struct for lead term
struct LeadTerm{T <: AbstractTerm} <: AbstractTerm
    term::T                      # The term to lead (can be nested like log(y))
    horizon::Union{Int, Nothing}  # nothing in lp() context, Int for standalone
end

StatsModels.terms(t::LeadTerm) = (t.term,)
# termvars: needed once the schema has replaced the FunctionTerm, so that a
# horizon-tracking leads(x) standing alone on the RHS still counts as a regressor.
StatsModels.termvars(t::LeadTerm) = StatsModels.termvars(t.term)

function StatsModels.apply_schema(
        t::FunctionTerm{typeof(leads)}, sch::StatsModels.Schema, ctx::Type)
    term, horizon = _parse_unary_binary_args(t, "leads", nothing)
    term = StatsModels.apply_schema(term, sch, ctx)
    return LeadTerm{typeof(term)}(term, horizon)
end

function StatsModels.apply_schema(t::LeadTerm, sch::StatsModels.Schema, ctx::Type)
    term = StatsModels.apply_schema(t.term, sch, ctx)
    LeadTerm{typeof(term)}(term, t.horizon)
end

# modelcols: Apply lead transformation with NaN handling
function StatsModels.modelcols(lt::LeadTerm, d::Tables.ColumnTable)
    _check_horizon_provided(lt.horizon, "leads")
    original_cols = StatsModels.modelcols(lt.term, d)
    original_cols = _extract_single_column(original_cols, "leads() response")
    return lead(original_cols, lt.horizon, default = NaN)
end

# width: Return number of columns (always 1 for leads)
StatsModels.width(lt::LeadTerm) = 1

# show: Display the term
function Base.show(io::IO, lt::LeadTerm)
    if lt.horizon === nothing
        print(io, "leads($(lt.term))")
    else
        print(io, "leads($(lt.term), $(lt.horizon))")
    end
end

# coefnames: Return coefficient name
function StatsModels.coefnames(lt::LeadTerm)
    base_names = StatsModels.coefnames(lt.term)
    if lt.horizon === nothing
        return ["leads(" * base_names[1] * ")"]
    else
        return ["leads(" * base_names[1] * ", $(lt.horizon))"]
    end
end

# ============================================================================
# Anchor Term (anchor)
# This implements anchor(y, z) for anchored responses: y_{t+h} - z_t
# The anchor variable z stays fixed at time t while y shifts forward to t+h
# ============================================================================

"""
    anchor(response_term, anchor_term)
    anchor(response_term, anchor_term, horizon)

Create anchored response term for local projections. Used in formulas like:
- `@formula(anchor(y, z) ~ x)` - horizon determined by lp() context
- `@formula(anchor(y, z, 5) ~ x)` - explicit horizon for standalone use
- `@formula(anchor(log(y), z) ~ x)` - supports nested transformations on response

Computes y_{t+h} - z_t where:
- y is the response variable (can be transformed)
- z is the anchor variable (stays at time t)
- h is the horizon (0, 1, 2, ...)

At h=0: returns y_t - z_t
At h=1: returns y_{t+1} - z_t
At h=2: returns y_{t+2} - z_t, etc.
"""
function anchor(response::T, anchor_var::S) where {T <: AbstractTerm, S <: AbstractTerm}
    AnchorTerm{T, S}(response, anchor_var, nothing)
end
function anchor(response::T, anchor_var::S, h::Int) where {
        T <: AbstractTerm, S <: AbstractTerm}
    AnchorTerm{T, S}(response, anchor_var, h)
end

# termvars: Extract variables from anchor() for schema creation
function StatsModels.termvars(t::FunctionTerm{typeof(anchor)})
    length(t.args) >= 2 || return Symbol[]
    return unique(vcat(StatsModels.termvars(t.args[1]), StatsModels.termvars(t.args[2])))
end

# Struct for anchored response term
struct AnchorTerm{T <: AbstractTerm, S <: AbstractTerm} <: AbstractTerm
    response::T                  # The response term (can be nested like log(y))
    anchor::S                    # The anchor term (stays at time t)
    horizon::Union{Int, Nothing}  # nothing in lp() context, Int for standalone
end

StatsModels.terms(t::AnchorTerm) = (t.response, t.anchor)

function StatsModels.apply_schema(
        t::FunctionTerm{typeof(anchor)}, sch::StatsModels.Schema, ctx::Type)
    if length(t.args) == 2  # anchor(response, anchor_var) - horizon from context
        response, anchor_var = t.args
        horizon = nothing
    elseif length(t.args) == 3  # anchor(response, anchor_var, horizon)
        response, anchor_var, h_arg = t.args
        (h_arg isa ConstantTerm) ||
            throw(ArgumentError("anchor horizon must be a number (got $h_arg)"))
        horizon = h_arg.n
    else
        throw(ArgumentError("anchor() requires 2 or 3 arguments"))
    end

    response = StatsModels.apply_schema(response, sch, ctx)
    anchor_var = StatsModels.apply_schema(anchor_var, sch, ctx)
    return AnchorTerm{typeof(response), typeof(anchor_var)}(response, anchor_var, horizon)
end

function StatsModels.apply_schema(t::AnchorTerm, sch::StatsModels.Schema, ctx::Type)
    response = StatsModels.apply_schema(t.response, sch, ctx)
    anchor_var = StatsModels.apply_schema(t.anchor, sch, ctx)
    AnchorTerm{typeof(response), typeof(anchor_var)}(response, anchor_var, t.horizon)
end

# modelcols: Apply anchored transformation (y_{t+h} - z_t)
function StatsModels.modelcols(at::AnchorTerm, d::Tables.ColumnTable)
    _check_horizon_provided(at.horizon, "anchor")
    response_cols = StatsModels.modelcols(at.response, d)
    anchor_cols = StatsModels.modelcols(at.anchor, d)
    response_cols = _extract_single_column(response_cols, "anchor() response")
    anchor_cols = _extract_single_column(anchor_cols, "anchor() anchor variable")
    return _create_anchored(response_cols, anchor_cols, at.horizon)
end

# width: Return number of columns (always 1 for anchor)
StatsModels.width(at::AnchorTerm) = 1

# show: Display the term
function Base.show(io::IO, at::AnchorTerm)
    if at.horizon === nothing
        print(io, "anchor($(at.response), $(at.anchor))")
    else
        print(io, "anchor($(at.response), $(at.anchor), $(at.horizon))")
    end
end

# coefnames: Return coefficient name
function StatsModels.coefnames(at::AnchorTerm)
    response_names = StatsModels.coefnames(at.response)
    anchor_names = StatsModels.coefnames(at.anchor)
    if at.horizon === nothing
        return ["anchor(" * response_names[1] * ", " * anchor_names[1] * ")"]
    else
        return ["anchor(" * response_names[1] * ", " * anchor_names[1] * ", $(at.horizon))"]
    end
end

# termvars: Return variables used in anchor term
function StatsModels.termvars(at::AnchorTerm)
    unique(vcat(StatsModels.termvars(at.response), StatsModels.termvars(at.anchor)))
end

# ============================================================================
# Horizon-tracking terms on the right-hand side
# ============================================================================

"""
    _has_dynamic_horizon(term) -> Bool

`true` when `term` contains a `CumulTerm` or `LeadTerm` whose horizon is
`nothing` — a right-hand-side term that must be rebuilt at every projection
horizon rather than once. `cumul(x, 3)` and `leads(x, 3)` pin an explicit
horizon and are therefore *not* dynamic.
"""
_has_dynamic_horizon(::Any) = false
_has_dynamic_horizon(t::CumulTerm) = t.horizon === nothing || _has_dynamic_horizon(t.term)
_has_dynamic_horizon(t::LeadTerm) = t.horizon === nothing || _has_dynamic_horizon(t.term)
function _has_dynamic_horizon(t::AnchorTerm)
    _has_dynamic_horizon(t.response) || _has_dynamic_horizon(t.anchor)
end
_has_dynamic_horizon(t::StatsModels.MatrixTerm) = any(_has_dynamic_horizon, t.terms)
_has_dynamic_horizon(t::StatsModels.InteractionTerm) = any(_has_dynamic_horizon, t.terms)
_has_dynamic_horizon(t::Tuple) = any(_has_dynamic_horizon, t)
_has_dynamic_horizon(t::AbstractVector) = any(_has_dynamic_horizon, t)

"""
    _specialize_horizon(term, h::Int)

Return `term` with every horizon-tracking `CumulTerm`/`LeadTerm` (horizon
`nothing`) pinned to horizon `h`. Terms carrying an explicit horizon, and every
other term, are returned unchanged.

Coefficient names are deliberately taken from the *unspecialised* terms, so
`cumul(g)` names one coefficient across all horizons rather than `cumul(g, 0)`,
`cumul(g, 1)`, ... — `coefpath`, `summarize` and the plot recipes all assume
horizon-invariant coefficient names.
"""
_specialize_horizon(t, ::Int) = t
function _specialize_horizon(t::CumulTerm, h::Int)
    inner = _specialize_horizon(t.term, h)
    return CumulTerm{typeof(inner)}(inner, t.horizon === nothing ? h : t.horizon)
end
function _specialize_horizon(t::LeadTerm, h::Int)
    inner = _specialize_horizon(t.term, h)
    return LeadTerm{typeof(inner)}(inner, t.horizon === nothing ? h : t.horizon)
end
function _specialize_horizon(t::StatsModels.MatrixTerm, h::Int)
    return StatsModels.MatrixTerm(map(x -> _specialize_horizon(x, h), t.terms))
end
function _specialize_horizon(t::StatsModels.InteractionTerm, h::Int)
    return StatsModels.InteractionTerm(map(x -> _specialize_horizon(x, h), t.terms))
end
_specialize_horizon(t::Tuple, h::Int) = map(x -> _specialize_horizon(x, h), t)

"""
    _as_float_matrix(cols)

Materialise `modelcols` output as a `Matrix{Float64}`, mapping `missing` to the
`NaN` sentinel the per-horizon row masks use.
"""
function _as_float_matrix(cols::AbstractMatrix)
    Matrix{Float64}(map(v -> ismissing(v) ? NaN : Float64(v), cols))
end
function _as_float_matrix(cols::AbstractVector)
    reshape(Vector{Float64}(map(v -> ismissing(v) ? NaN : Float64(v), cols)), :, 1)
end

"""
    _rhs_tracks_horizon(formula::FormulaTerm) -> Bool

`true` when the *unapplied* right-hand side of `formula` contains a bare
`cumul(x)` or `leads(x)` — a regressor that is rebuilt at every projection
horizon — anywhere, including inside an IV block `(endo ~ instruments)` or
an interaction. Explicit horizons (`cumul(x, 3)`) do not count.

This is the raw-formula counterpart of `_has_dynamic_horizon`, used
by procedures that only hold the fitted object and its `base_formula`.
"""
_rhs_tracks_horizon(formula::FormulaTerm) = _tracks_horizon(formula.rhs)

_tracks_horizon(::Any) = false
_tracks_horizon(t::Tuple) = any(_tracks_horizon, t)
_tracks_horizon(t::FormulaTerm) = _tracks_horizon(t.lhs) || _tracks_horizon(t.rhs)
_tracks_horizon(t::StatsModels.InteractionTerm) = any(_tracks_horizon, t.terms)
function _tracks_horizon(t::FunctionTerm)
    nm = nameof(t.f)
    (nm === :cumul || nm === :leads) && length(t.args) == 1 && return true
    return any(_tracks_horizon, t.args)
end

# ============================================================================
# Pipe Operator (|) for Anchored Response Syntax
# Intercept | during schema application to create AnchorTerm
# ============================================================================

"""
    termvars for FunctionTerm{typeof(|)}

Extract variable names from pipe operator for schema creation.
This tells StatsModels which variables are referenced so it can
properly determine their types (continuous vs categorical).
"""
function StatsModels.termvars(t::FunctionTerm{typeof(|)})
    if length(t.args) != 2
        return Symbol[]
    end
    lhs, rhs = t.args
    return unique(vcat(StatsModels.termvars(lhs), StatsModels.termvars(rhs)))
end

"""
    apply_schema for FunctionTerm{typeof(|)}

Intercepts pipe operator in formulas to create AnchorTerm for anchored responses.
Enables syntax like:
- `@formula(leads(y)|z ~ x)` - equivalent to `@formula(anchor(y, z) ~ x)`
- `@formula(cumul(y)|z ~ x)` - cumulative response anchored to z

The pipe creates an AnchorTerm where:
- lhs is the response term (can be transformed: leads(y), cumul(y), log(y), etc.)
- rhs is the anchor term (stays fixed at time t)

# Examples
```julia
# Standard: y_{t+h}
lp(@formula(leads(y) ~ x), df; horizon=12)

# Anchored: y_{t+h} - z_t (pipe syntax)
lp(@formula(leads(y)|z ~ x), df; horizon=12)

```
"""
function StatsModels.apply_schema(t::FunctionTerm{typeof(|)}, sch::StatsModels.Schema, ctx::Type)
    # The pipe operator in formulas: lhs | rhs
    # Convert to AnchorTerm(lhs, rhs, nothing)
    if length(t.args) != 2
        throw(ArgumentError("Pipe operator | requires exactly 2 arguments (got $(length(t.args)))"))
    end

    lhs, rhs = t.args

    # Apply schema to both sides
    lhs_term = StatsModels.apply_schema(lhs, sch, ctx)
    rhs_term = StatsModels.apply_schema(rhs, sch, ctx)

    # Return AnchorTerm
    return AnchorTerm{typeof(lhs_term), typeof(rhs_term)}(lhs_term, rhs_term, nothing)
end

"""
    LocalProjection

Stack of horizon-specific OLS models produced by [`lp`](@ref).
"""
struct LocalProjection{M <: OLSMatrixEstimator}
    models::Vector{M}
    horizon::Int
    response::Symbol
    shock::Symbol
    base_formula::FormulaTerm
    coef_names::Vector{String}  # Coefficient names (constant across all horizons)
    tautological_h0::Bool       # true when response == shock (h=0 is trivial: coef=1, SE=0)
end

"""
    coefnames(lp::LocalProjection)

Return the coefficient names for the local projection models.
Since all horizons share the same RHS, returns a single set of coefficient names.
"""
StatsModels.coefnames(lp::LocalProjection) = lp.coef_names

"""
    coefnames(lp::LocalProjection, h::Int)

Return the coefficient names for horizon `h` (0-indexed).
Since all horizons share the same RHS, returns the same names regardless of `h`.
"""
StatsModels.coefnames(lp::LocalProjection, h::Int) = lp.coef_names

function Base.show(io::IO, lp::LocalProjection)
    print(io, "LocalProjection(horizon=0:$(lp.horizon), response=$(lp.response), shock=$(lp.shock))")
end

function Base.show(io::IO, ::MIME"text/plain", lp::LocalProjection)
    println(io, "LocalProjection")
    println(io, "  Response:   $(lp.response)")
    println(io, "  Shock:      $(lp.shock)")
    println(io, "  Horizon:    0:$(lp.horizon)")
    println(io, "  Formula:    $(lp.base_formula)")
    println(io, "  Coef names: $(lp.coef_names)")
end

"""
    lp + vcov(estimator)

Apply a covariance estimator to all models in a LocalProjection using the `+` operator.
Returns a new LocalProjection with updated variance-covariance specification for each model.

This is consistent with Regress.jl's `model + vcov(estimator)` pattern and allows chainable
operations like `(lp + vcov(A)) + vcov(B)`.

# Arguments
- `lp::LocalProjection`: The local projection result
- `v::VcovSpec{V}`: A variance-covariance specification from `vcov(estimator)`

# Returns
A new `LocalProjection` with the vcov estimator applied to each underlying model.

# Example
```julia
lp_result = lp(@formula(leads(y) ~ x), df; horizon=12)

# Apply robust standard errors
lp_robust = lp_result + vcov(Bartlett{NeweyWest}())

# Coefficients unchanged, but models now have HAC vcov
@assert coefpath(lp_result) == coefpath(lp_robust)
```
"""
function Base.:+(lp::LocalProjection{M}, v::VcovSpec{V}) where {M <: OLSMatrixEstimator, V}
    # Apply vcov to each model using Regress.jl's + operator
    new_models = [m + v for m in lp.models]
    M_new = eltype(new_models)
    return LocalProjection{M_new}(
        convert(Vector{M_new}, new_models),
        lp.horizon, lp.response, lp.shock, lp.base_formula, lp.coef_names,
        lp.tautological_h0
    )
end

"""
    LocalProjectionCovariance

Diagonal covariance entries term-by-term across horizons.
"""
struct LocalProjectionCovariance{E}
    estimator::E
    variances::Dict{Symbol, Vector{Float64}}
    horizon::Int
end

"""
    _create_cumulative(y::AbstractVector, h::Int) -> Vector{Float64}

Helper function to compute cumulative sum from t to t+h for each observation.
Returns sum_{j=0}^{h} y_{t+j}.

At h=0, returns y itself (converted to Float64).
At h=1, returns y_t + y_{t+1}.
At h=2, returns y_t + y_{t+1} + y_{t+2}, etc.

NaN values are returned when:
- The sum cannot be computed (at the end of the series)
- Any component y_{t+j} is missing or NaN

# Returns
- `Vector{Float64}`: Type-stable Float64 vector with NaN at boundaries
"""
function _create_cumulative(y::AbstractVector, h::Int)::Vector{Float64}
    n = length(y)

    # Convert input to Float64, replacing missing with NaN
    # This handles both pure Float64 vectors and Union{Missing, Float64} vectors
    y_float = map(v -> ismissing(v) ? NaN : Float64(v), y)

    # Fast path for h=0
    h == 0 && return y_float

    result = Vector{Float64}(undef, n)
    @inbounds for t in 1:n
        if t + h > n
            result[t] = NaN
        else
            # Sum from t to t+h (inclusive, so h+1 values total)
            cumsum_val = 0.0
            all_valid = true
            for j in 0:h
                val = y_float[t + j]
                if isnan(val)
                    all_valid = false
                    break
                end
                cumsum_val += val
            end
            result[t] = all_valid ? cumsum_val : NaN
        end
    end

    return result
end

"""
    _create_anchored(y::AbstractVector, z::AbstractVector, h::Int) -> Vector{Float64}

Helper function to compute anchored response: y_{t+h} - z_t for each observation.

At h=0, returns y_t - z_t (both at time t).
At h=1, returns y_{t+1} - z_t (y shifted forward, z stays at t).
At h=2, returns y_{t+2} - z_t, etc.

The key difference from standard lead:
- Standard lead: y_{t+h} evolves freely
- Anchored: y_{t+h} - z_t measures deviation from anchor z_t

NaN values are returned when:
- Cannot compute lead (at the end of the series)
- Either y_{t+h} or z_t is missing or NaN

# Returns
- `Vector{Float64}`: Type-stable Float64 vector with NaN at boundaries
"""
function _create_anchored(y::AbstractVector, z::AbstractVector, h::Int)::Vector{Float64}
    n = length(y)
    length(z) == n || throw(ArgumentError("y and z must have same length"))
    h > n && throw(ArgumentError("horizon h=$h is too large for series of length $n"))
    return lead(y, h, default = NaN) .- z
end

"""

(term::AbstractTerm)

Recursively extract base variable names from a term, stripping away
function transformations like lag(), lead(), cumul().

Returns a vector of Symbol representing the raw variables referenced.

# Examples
```julia
_extract_base_variables(Term(:x))                    # [:x]
_extract_base_variables(FunctionTerm(lag, [:x, 4])) # [:x]  (strips lag)
```
"""
# Specialized methods for specific term types
function _extract_base_variables(t::Union{
        Term, StatsModels.ContinuousTerm, StatsModels.CategoricalTerm})
    [t.sym]
end
_extract_base_variables(t::CumulTerm) = _extract_base_variables(t.term)
_extract_base_variables(t::LeadTerm) = _extract_base_variables(t.term)
function _extract_base_variables(t::AnchorTerm)
    unique(vcat(_extract_base_variables(t.response), _extract_base_variables(t.anchor)))
end

function _extract_base_variables(ft::FunctionTerm)
    # For lag/lead/cumul/leads, extract the base variable (first argument)
    if ft.f in (lag, lead, cumul, leads)
        return _extract_base_variables(ft.args[1])
    elseif ft.f === anchor
        # For anchor, extract both response and anchor variables
        length(ft.args) >= 2 || return StatsModels.termvars(ft)
        return unique(vcat(_extract_base_variables(ft.args[1]),
            _extract_base_variables(ft.args[2])))
    else
        # For other functions, use termvars
        return StatsModels.termvars(ft)
    end
end

function _extract_base_variables(terms::Tuple)
    # For RHS tuple of terms
    all_vars = Symbol[]
    for t in terms
        append!(all_vars, _extract_base_variables(t))
    end
    return unique(all_vars)
end

function _extract_base_variables(ft::FormulaTerm)
    unique(vcat(
        _extract_base_variables(ft.lhs),
        _extract_base_variables(ft.rhs)
    ))
end

# InteractionTerm and other composite terms
function _extract_base_variables(t::StatsModels.InteractionTerm)
    all_vars = Symbol[]
    for component in t.terms
        append!(all_vars, _extract_base_variables(component))
    end
    return unique(all_vars)
end

"""
    lp(formula, data; horizon, shock=nothing)

Estimate local projections implied by `formula` up to the supplied `horizon`.
`shock` selects the coefficient path of interest (defaults to the first RHS term).

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

See also [`lpiv`](@ref), whose endogenous and instrument blocks accept the same
terms.
"""
function lp(formula::FormulaTerm, data::AbstractDataFrame;
        horizon::Integer, shock::Union{Symbol, Nothing} = nothing)
    horizon < 0 && throw(ArgumentError("horizon must be non-negative"))
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

    # Check if LHS is a CumulTerm (cumulative impulse response), LeadTerm (forward-looking), or AnchorTerm (anchored)
    # Note: AnchorTerm can contain LeadTerm or CumulTerm inside (from pipe syntax like leads(y)|z or cumul(y)|z)
    # If AnchorTerm contains plain term (y|z), default to leads behavior
    is_anchor, is_cumulative, is_leads, anchor_term, cumul_term,
    leads_term = _unwrap_lhs(lhs_term)

    # Extract response variable/data
    # For cumulative/leads/anchor cases, we need to extract the base variable names for Stage 1 filtering
    # but we'll evaluate the transformed term for actual calculation
    response = if is_anchor
        _extract_single_response(anchor_term.response, "anchor() response term")
    elseif is_cumulative
        _extract_single_response(cumul_term, "cumul() term")
    elseif is_leads
        _extract_single_response(leads_term, "leads() term")
    else
        throw(ArgumentError("A local projection without leads and cumulated variables does not make much sense"))
    end

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

    # Keep only complete cases (remove rows with missing base variables)
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
        response, shock_symbol, formula,
        is_anchor, is_cumulative, is_leads, anchor_term, cumul_term, leads_term)
end

"""
    _lp_estimate_horizons(Xof, coef_names_base, df, horizon, response, shock, formula, ...)

Function barrier for type-stable per-horizon OLS estimation.
`Xof(h)` returns the `Matrix{Float64}` design at horizon `h` — the same matrix
every time unless the RHS carries a horizon-tracking `cumul`/`leads` term.
"""
function _lp_estimate_horizons(Xof::F, coef_names_base::Vector{String},
        df_base_complete, horizon, response, shock_symbol, formula,
        is_anchor, is_cumulative, is_leads, anchor_term, cumul_term,
        leads_term) where {F}
    # Helper to estimate one horizon
    function _estimate_horizon(h)
        X = Xof(h)::Matrix{Float64}
        # Identify rows where X is complete (no NaN values)
        X_missing_ind = vec(all(!isnan, X, dims = 2))
        lhs_h = _build_lhs_for_horizon(h, is_anchor, is_cumulative, is_leads,
            anchor_term, cumul_term, leads_term)
        # Convert to Vector{Float64} for type stability (modelcols may return ShiftedArray)
        y_h = Vector{Float64}(StatsModels.modelcols(lhs_h, df_base_complete))
        y_complete_rows = .!isnan.(y_h)
        complete_rows = X_missing_ind .& y_complete_rows
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

    return LocalProjection(
        models, horizon, response, shock_symbol, formula, coef_names_base,
        response === shock_symbol)
end

"""
    stderror(cov; term)

Standard errors across horizons for `term`.
"""
function stderror(cov::LocalProjectionCovariance; term::Symbol)
    variances = get(cov.variances, term) do
        throw(ArgumentError("term $term not found in covariance object"))
    end
    return sqrt.(variances)
end

"""
    IRFSummary

Summary of impulse response function with coefficients, standard errors,
and confidence intervals. Displays with PrettyTables, highlighting
statistically significant coefficients in bold.
"""
struct IRFSummary
    term::Symbol
    level::Float64
    scale::Float64
    horizon::Vector{Int}
    coef::Vector{Float64}
    se::Vector{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
end

# Convert to DataFrame for data access
function DataFrames.DataFrame(s::IRFSummary)
    DataFrame(
        horizon = s.horizon,
        coef = s.coef,
        se = s.se,
        lower = s.lower,
        upper = s.upper
    )
end

function Base.show(io::IO, s::IRFSummary)
    println(io, "IRFSummary(term=$(s.term), level=$(s.level), scale=$(s.scale))")
end

function Base.show(io::IO, ::MIME"text/plain", s::IRFSummary)
    # Determine significance: CI doesn't include zero
    significant = (s.lower .> 0) .| (s.upper .< 0)

    # Create data matrix for PrettyTables
    data = hcat(s.horizon, s.coef, s.se, s.lower, s.upper)

    # Format the level as percentage
    level_pct = round(Int, s.level * 100)

    # Column labels (PrettyTables v3 API)
    labels = ["Horizon", "Coef", "Std.Err.", "Lower $level_pct%", "Upper $level_pct%"]

    # Highlighters for bold significant values (PrettyTables v3 API)
    # Make coef, lower, upper bold when significant
    hl_coef = TextHighlighter(
        (d, i, j) -> j == 2 && significant[i],
        crayon"bold"
    )
    hl_lower = TextHighlighter(
        (d, i, j) -> j == 4 && significant[i],
        crayon"bold"
    )
    hl_upper = TextHighlighter(
        (d, i, j) -> j == 5 && significant[i],
        crayon"bold"
    )

    # Title
    title = "Impulse Response: $(s.term)"
    if s.scale != 1.0
        title *= " (scale=$(s.scale))"
    end

    # Table format with rounded borders
    table_fmt = TextTableFormat(borders = text_table_borders__unicode_rounded)

    # Enable color output for ANSI bold codes
    ioc = IOContext(io, :color => true)

    pretty_table(ioc, data;
        column_labels = labels,
        title = title,
        highlighters = [hl_coef, hl_lower, hl_upper],
        formatters = [fmt__round(4)],
        alignment = [:r, :r, :r, :r, :r],
        table_format = table_fmt
    )
end

# NOTE: Plot recipes, coefpath, vcov, and summarize are defined after LPResult below

# ============================================================================
# Instrumental Variables Local Projections
# ============================================================================

"""
    LocalProjectionIV

Stack of horizon-specific IV models produced by [`lpiv`](@ref).
"""
struct LocalProjectionIV{M <: IVMatrixEstimator}
    models::Vector{M}
    horizon::Int
    response::Symbol
    shock::Symbol
    base_formula::FormulaTerm
    coef_names::Vector{String}  # Coefficient names (constant across all horizons)
    endogenous_names::Vector{String}
    instrument_names::Vector{String}
    tautological_h0::Bool       # true when response == shock (h=0 is trivial: coef=1, SE=0)
end

"""
    coefnames(lpiv::LocalProjectionIV)

Return the coefficient names for the local projection IV models.
Since all horizons share the same RHS, returns a single set of coefficient names.
"""
StatsModels.coefnames(lpiv::LocalProjectionIV) = lpiv.coef_names

"""
    coefnames(lpiv::LocalProjectionIV, h::Int)

Return the coefficient names for horizon `h` (0-indexed).
Since all horizons share the same RHS, returns the same names regardless of `h`.
"""
StatsModels.coefnames(lpiv::LocalProjectionIV, h::Int) = lpiv.coef_names

"""Type alias for dispatching shared methods on both OLS and IV local projections."""
const LPResult = Union{LocalProjection, LocalProjectionIV}

function Base.show(io::IO, lpiv::LocalProjectionIV)
    print(io,
        "LocalProjectionIV(horizon=0:$(lpiv.horizon), response=$(lpiv.response), shock=$(lpiv.shock))")
end

function Base.show(io::IO, ::MIME"text/plain", lpiv::LocalProjectionIV)
    println(io, "LocalProjectionIV")
    println(io, "  Response:     $(lpiv.response)")
    println(io, "  Shock:        $(lpiv.shock)")
    println(io, "  Horizon:      0:$(lpiv.horizon)")
    println(io, "  Formula:      $(lpiv.base_formula)")
    println(io, "  Endogenous:   $(lpiv.endogenous_names)")
    println(io, "  Instruments:  $(lpiv.instrument_names)")
    println(io, "  Coef names:   $(lpiv.coef_names)")
end

"""
    lpiv + vcov(estimator)

Apply a covariance estimator to all models in a LocalProjectionIV using the `+` operator.
Returns a new LocalProjectionIV with updated variance-covariance specification for each model.

# Example
```julia
result = lpiv(@formula(leads(y) ~ (x ~ z)), df; horizon=10)
result_hac = result + vcov(Bartlett{NeweyWest}())
```
"""
function Base.:+(lpiv::LocalProjectionIV{M}, v::VcovSpec{V}) where {
        M <: IVMatrixEstimator, V}
    new_models = [m + v for m in lpiv.models]
    M_new = eltype(new_models)
    return LocalProjectionIV{M_new}(
        convert(Vector{M_new}, new_models),
        lpiv.horizon, lpiv.response, lpiv.shock, lpiv.base_formula,
        lpiv.coef_names, lpiv.endogenous_names, lpiv.instrument_names,
        lpiv.tautological_h0
    )
end

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

summarize(res, vcov(Bartlett(30.0), res))   # multiplier path with s.e. and bands
```

The multiplier is the coefficient on `cumul(g)`, which `shock` picks up by
default. On the choice of HAC bandwidth here — and why the automatic
Newey-West rule does *not* reproduce Stata `ivreg2, bw(auto)` output — see
section 8 of `docs/src/inference_procedures_guide.md`.

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
        horizon::Integer, shock::Union{Symbol, Nothing} = nothing)
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

    # Check LHS structure (leads, cumul, anchor)
    is_anchor, is_cumulative, is_leads, anchor_term, cumul_term,
    leads_term = _unwrap_lhs(lhs_term)

    # Extract response variable name
    response = if is_anchor
        _extract_single_response(anchor_term.response, "anchor() response term")
    elseif is_cumulative
        _extract_single_response(cumul_term, "cumul() term")
    elseif is_leads
        _extract_single_response(leads_term, "leads() term")
    else
        throw(ArgumentError("LHS must use leads(), cumul(), or anchor()"))
    end

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
        endo_names, instr_names,
        is_anchor, is_cumulative, is_leads, anchor_term, cumul_term, leads_term)
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
        endo_names, instr_names,
        is_anchor, is_cumulative, is_leads, anchor_term, cumul_term,
        leads_term) where {F}
    # Helper to estimate one horizon
    function _estimate_iv_horizon(h)
        X_full, Z_full = XZof(h)::Tuple{Matrix{Float64}, Matrix{Float64}}
        # Identify complete rows
        XZ_complete = vec(all(!isnan, X_full, dims = 2)) .&
                      vec(all(!isnan, Z_full, dims = 2))
        lhs_h = _build_lhs_for_horizon(h, is_anchor, is_cumulative, is_leads,
            anchor_term, cumul_term, leads_term)
        # Convert to Vector{Float64} for type stability (modelcols may return ShiftedArray)
        y_h = Vector{Float64}(StatsModels.modelcols(lhs_h, df_base_complete))
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

    return LocalProjectionIV(
        models, horizon, response, shock_symbol, formula, coef_names_base,
        endo_names, instr_names, response === shock_symbol)
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

# Returns
A `WeakIVTestResult` containing effective F, robust F, critical values, etc.

See also: [`lpiv`](@ref), [`first_stage`](@ref)
"""
function Regress.weakivtest(lpiv::LocalProjectionIV, h::Int; kwargs...)
    0 <= h <= lpiv.horizon ||
        throw(BoundsError("horizon $h out of range 0:$(lpiv.horizon)"))
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
function Regress.weakivtest(lpiv::LocalProjectionIV; kwargs...)
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

# ============================================================================
# Shared methods for LocalProjection and LocalProjectionIV (via LPResult)
# ============================================================================

"""
    coefpath(lp; term = lp.shock)

Collect the coefficient path across horizons for `term`.
Works for both `LocalProjection` and `LocalProjectionIV`.
"""
function coefpath(lp::LPResult; term::Symbol = lp.shock)
    n = lp.horizon + 1
    names = lp.coef_names
    idx = findfirst(==(String(term)), names)
    idx === nothing &&
        throw(ArgumentError("term $term not present in model"))
    coefficients = Vector{Float64}(undef, n)
    for (i, model) in enumerate(lp.models)
        coefficients[i] = coef(model)[idx]
    end
    # At h=0 when response == shock, the coefficient is known exactly:
    # 1.0 for the shock term, 0.0 for all others
    if lp.tautological_h0
        coefficients[1] = term === lp.shock ? 1.0 : 0.0
    end
    return coefficients
end

# ============================================================================
# Herbst–Johannsen bias correction (BCC)
# ============================================================================

"""
    BiasCorrectedLP

Herbst–Johannsen bias-corrected local projection (the "BCC" estimator of
Herbst & Johannsen, 2024). Wraps an OLS [`LocalProjection`](@ref) together
with the persistence adjustments ``\\hat c_j`` estimated from the
horizon-zero control matrix. Construct with [`biascorrect`](@ref).

`coefpath` returns the corrected impulse-response path
``\\hat\\theta_h^{\\,c}``. `summarize`, the plot recipes, and
`as_irf_result` center the confidence bands on the corrected path while
using the standard errors of the *uncorrected* OLS coefficients: following
the reference implementation (`lp_biascorr.m` in Montiel Olea,
Plagborg-Møller, Qian & Wolf, 2025), neither the sampling uncertainty of
``\\hat c_j`` nor the covariance across lower-horizon responses entering
the recursion is propagated into the bands.

Properties not stored on the wrapper (`horizon`, `response`, `shock`,
`models`, ...) are forwarded to the wrapped `LocalProjection`.
"""
struct BiasCorrectedLP{L <: LocalProjection}
    lp::L
    c::Vector{Float64}       # persistence adjustments ĉ_j, j = 1:horizon
    T0::Int                  # horizon-zero estimation sample size
    controls::Vector{String} # coefficient names spanning w_t
end

function Base.getproperty(bc::BiasCorrectedLP, name::Symbol)
    name in fieldnames(BiasCorrectedLP) && return getfield(bc, name)
    return getproperty(getfield(bc, :lp), name)
end

function Base.propertynames(bc::BiasCorrectedLP)
    (fieldnames(BiasCorrectedLP)..., propertynames(getfield(bc, :lp))...)
end

"""
    LPEstimate

Union of the plain LP results ([`LPResult`](@ref)) and
[`BiasCorrectedLP`](@ref), for the shared `summarize`/plotting/
`as_irf_result` pipeline.
"""
const LPEstimate = Union{LPResult, BiasCorrectedLP}

function Base.show(io::IO, bc::BiasCorrectedLP)
    lp = bc.lp
    print(io,
        "BiasCorrectedLP(horizon=0:$(lp.horizon), response=$(lp.response), shock=$(lp.shock))")
end

function Base.show(io::IO, mime::MIME"text/plain", bc::BiasCorrectedLP)
    println(io, "BiasCorrectedLP (Herbst–Johannsen)")
    println(io, "  Controls:   ",
        isempty(bc.controls) ? "(none)" : join(bc.controls, ", "))
    println(io, "  T₀:         $(bc.T0)")
    show(io, mime, bc.lp)
end

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
    _rhs_tracks_horizon(lp_result.base_formula) && throw(ArgumentError(
        "the Herbst–Johannsen correction assumes the same regressors at every " *
        "horizon, but the formula has a horizon-tracking right-hand-side term " *
        "(a bare `cumul(x)` or `leads(x)`), so the design changes with h"))
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
    vcov(estimator, bc::BiasCorrectedLP)

Covariance of the *uncorrected* per-horizon OLS coefficients (the
reference convention pairs the corrected path with the ordinary LP
standard errors; see [`BiasCorrectedLP`](@ref)).
"""
function vcov(estimator::CovarianceMatrices.AbstractAsymptoticVarianceEstimator,
        bc::BiasCorrectedLP)
    return vcov(estimator, bc.lp)
end

"""
    bc + vcov(estimator)

Apply a covariance estimator to the wrapped models, keeping the bias
correction (consistent with `lp + vcov(estimator)`).
"""
function Base.:+(bc::BiasCorrectedLP, v::VcovSpec)
    return BiasCorrectedLP(bc.lp + v, bc.c, bc.T0, bc.controls)
end

"""
    vcov(estimator, lp)

Compute diagonal covariance entries horizon-by-horizon using `estimator`
from `CovarianceMatrices.jl`. Works for both `LocalProjection` and `LocalProjectionIV`.
"""
function vcov(estimator::CovarianceMatrices.AbstractAsymptoticVarianceEstimator,
        lp::LPResult)
    n = lp.horizon + 1
    variances = Dict{Symbol, Vector{Float64}}()
    names = Symbol.(lp.coef_names)

    for (i, model) in enumerate(lp.models)
        # At h=0 when response == shock, all coefficients are known exactly (variance = 0)
        if i == 1 && lp.tautological_h0
            for name in names
                vec = get!(variances, name) do
                    fill(NaN, n)
                end
                vec[1] = 0.0
            end
            continue
        end
        cov = CovarianceMatrices.vcov(estimator, model)
        for (j, name) in enumerate(names)
            vec = get!(variances, name) do
                fill(NaN, n)
            end
            vec[i] = cov[j, j]
        end
    end

    return LocalProjectionCovariance(estimator, variances, lp.horizon)
end

# ============================================================================
# Reduced-form VAR and lag-order selection
# ============================================================================

"""
    _var_data(data, vars) -> Matrix{Float64}

Materialize the VAR data matrix `Y` (`T × n`) from `vars`, in the given order.

`missing` and `NaN` are rejected: the moving-block bootstrap simulates a
complete series and the VAR recursion has no notion of a gap. This is also the
safe side of issue LP-1 (`lp` drops missing rows *before* lags are built, so an
internal gap would silently collapse the time axis).
"""
function _var_data(data::AbstractDataFrame, vars::AbstractVector{Symbol})
    isempty(vars) && throw(ArgumentError("`vars` must name at least one variable"))
    length(unique(vars)) == length(vars) ||
        throw(ArgumentError("`vars` contains duplicate names"))
    T = nrow(data)
    Y = Matrix{Float64}(undef, T, length(vars))
    for (j, v) in enumerate(vars)
        hasproperty(data, v) ||
            throw(ArgumentError("variable $v named in `vars` is not a column of the data"))
        col = data[!, v]
        any(ismissing, col) &&
            throw(ArgumentError("column $v contains `missing`; the VAR bootstrap " *
                                "requires a gap-free sample"))
        @inbounds for t in 1:T
            Y[t, j] = Float64(col[t])
        end
        any(isnan, view(Y, :, j)) &&
            throw(ArgumentError("column $v contains `NaN`; the VAR bootstrap " *
                                "requires a gap-free sample"))
    end
    return Y
end

"""
    _var_ols(Y::AbstractMatrix{Float64}, p::Int)

Least-squares reduced-form VAR(`p`) with intercept. Port of
`_estim/var_estim.m` in `lp_var_nberma`.

Returns a named tuple `(; c, A, U, Σu, T, Tu)` where `c` is the `n`-vector of
intercepts, `A` is `n × n × p` with `A[:, :, l]` the coefficient matrix on
``Y_{t-l}``, `U` is the `Tu × n` residual matrix, and

```math
\\widehat\\Sigma_u = \\frac{U'U}{T_u - (np + 1)},
\\qquad T_u = T - p,
```

the denominator used by `var_estim.m`: effective observations minus the number
of regressors per equation, intercept included.
"""
function _var_ols(Y::AbstractMatrix{Float64}, p::Int)
    p >= 1 || throw(ArgumentError("the VAR lag length must be at least 1 (got $p)"))
    T, n = size(Y)
    Tu = T - p
    k = n * p + 1
    Tu > k || throw(ArgumentError(
        "a VAR($p) on $n variables needs more than $k usable observations, " *
        "but T - p = $Tu are available (T = $T)"))
    X = Matrix{Float64}(undef, Tu, k)
    @inbounds for l in 1:p, j in 1:n, t in 1:Tu
        X[t, (l - 1) * n + j] = Y[p + t - l, j]
    end
    @inbounds X[:, end] .= 1.0
    Yd = Y[(p + 1):T, :]
    B = X \ Yd                       # k × n
    U = Yd - X * B
    Σu = (U' * U) ./ (Tu - k)
    A = Array{Float64}(undef, n, n, p)
    @inbounds for l in 1:p
        A[:, :, l] = transpose(B[((l - 1) * n + 1):(l * n), :])
    end
    return (; c = Vector{Float64}(B[end, :]), A = A, U = Matrix{Float64}(U),
        Σu = Matrix{Float64}(Σu), T = T, Tu = Tu)
end

"""
    VARLagSelection

Result of [`lagselect`](@ref): the AIC and BIC paths over `p = 1:maxlags` for a
reduced-form VAR, and the selected lag order.

Fields: `vars`, `maxlags`, `criterion`, `lags`, `aic`, `bic`, `selected`,
`nobs`. Use [`nlags`](@ref) to get the selected order as an `Int`.
"""
struct VARLagSelection
    vars::Vector{Symbol}
    maxlags::Int
    criterion::Symbol
    lags::Vector{Int}
    aic::Vector{Float64}
    bic::Vector{Float64}
    selected::Int
    nobs::Int
end

"""
    nlags(s::VARLagSelection) -> Int

The lag order selected by [`lagselect`](@ref) under its `criterion`.
"""
nlags(s::VARLagSelection) = s.selected

"""
    lagselect(data, vars; maxlags=10, criterion=:aic) -> VARLagSelection

Select the lag order of a reduced-form VAR in `vars` by an information
criterion. Port of `_estim/ic_var.m` in `lp_var_nberma`, which is what the
Montiel Olea, Plagborg-Møller, Qian & Wolf (2025) simulations use to set the
lag length of both the local-projection controls and the bootstrap VAR.

With ``n`` variables, ``T`` observations and ``\\widehat\\Sigma_p`` the
residual covariance of a VAR(``p``),

```math
\\mathrm{AIC}(p) = \\log\\det\\widehat\\Sigma_p
                 + \\frac{2\\,(n^2p + n)}{T - p_{\\max}},
\\qquad
\\mathrm{BIC}(p) = \\log\\det\\widehat\\Sigma_p
                 + \\frac{(n^2p + n)\\log(T - p_{\\max})}{T - p_{\\max}}.
```

Two conventions are inherited deliberately from the reference and are *not*
bugs: the penalty denominator is the fixed ``T - p_{\\max}`` for every ``p``, so
the criteria are comparable across the grid, but ``\\widehat\\Sigma_p`` is
estimated by `_var_ols` on the full sample (``T - p`` rows, with the
``(T-p) - (np+1)`` degrees-of-freedom denominator) rather than on a common
subsample. The grid starts at ``p = 1``; ``p = 0`` is not considered.

Both criteria are always computed; `criterion` (`:aic` or `:bic`) only decides
which one `selected` minimizes. A single order is chosen for the whole system —
there is no per-horizon or per-equation selection.

# Example
```julia
sel = lagselect(df, [:shock, :y]; maxlags = 10, criterion = :aic)
nlags(sel)     # selected p, to be used in the LP formula and in `varbootstrap`
```

See also [`nlags`](@ref), [`varbootstrap`](@ref).
"""
function lagselect(data::AbstractDataFrame, vars::AbstractVector{Symbol};
        maxlags::Integer = 10, criterion::Symbol = :aic)
    criterion in (:aic, :bic) ||
        throw(ArgumentError("`criterion` must be :aic or :bic (got :$criterion)"))
    maxlags >= 1 || throw(ArgumentError("`maxlags` must be at least 1 (got $maxlags)"))
    Y = _var_data(data, vars)
    T, n = size(Y)
    pmax = Int(maxlags)
    denom = T - pmax
    denom > 0 || throw(ArgumentError(
        "`maxlags` = $pmax is not smaller than the sample size T = $T"))
    aics = Vector{Float64}(undef, pmax)
    bics = Vector{Float64}(undef, pmax)
    for p in 1:pmax
        v = _var_ols(Y, p)
        ld = logdet(Symmetric(v.Σu))
        k = n^2 * p + n
        aics[p] = ld + 2 * k / denom
        bics[p] = ld + k * log(denom) / denom
    end
    sel = argmin(criterion === :aic ? aics : bics)
    return VARLagSelection(collect(vars), pmax, criterion, collect(1:pmax),
        aics, bics, sel, T)
end

function lagselect(data, vars::AbstractVector{Symbol}; kwargs...)
    lagselect(DataFrame(data), vars; kwargs...)
end

function DataFrames.DataFrame(s::VARLagSelection)
    DataFrame(lags = s.lags, aic = s.aic, bic = s.bic)
end

function Base.show(io::IO, s::VARLagSelection)
    print(io, "VARLagSelection(criterion=:", s.criterion, ", selected=", s.selected, ")")
end

function Base.show(io::IO, ::MIME"text/plain", s::VARLagSelection)
    data = (; Lags = s.lags, AIC = s.aic, BIC = s.bic)
    labels = ["Lags", "AIC", "BIC"]
    chosen = s.criterion === :aic ? 2 : 3
    hl = TextHighlighter(
        (d, i, j) -> s.lags[i] == s.selected && j == chosen,
        crayon"bold"
    )
    title = "VAR lag-order selection (ic_var): " *
            join(string.(s.vars), ", ") *
            "  |  T = $(s.nobs), p = 1:$(s.maxlags)"
    ioc = IOContext(io, :color => true)
    pretty_table(ioc, data;
        column_labels = labels,
        title = title,
        highlighters = [hl],
        formatters = [fmt__round(4, [2, 3])],
        alignment = [:r, :r, :r],
        table_format = TextTableFormat(
            borders = text_table_borders__unicode_rounded))
    println(io, "Selected by :", s.criterion, ": p = ", s.selected)
    return nothing
end

# ============================================================================
# VAR residual moving-block bootstrap (Hall percentile-t)
# ============================================================================

"""
    _pope_biascorrect(A, Σu, T) -> (A_corrected, δ)

Pope (1990, eq. 9) analytical bias correction of the VAR slope matrices. Port
of `_estim/var_biascorr.m` in `lp_var_nberma`.

With companion matrix ``\\mathcal A``, ``G = \\mathrm{blkdiag}(\\Sigma_u, 0)``
and ``\\Gamma_0`` solving the discrete Lyapunov equation
``\\Gamma_0 = \\mathcal A\\Gamma_0\\mathcal A' + G``,

```math
M = (I-\\mathcal A')^{-1}
  + \\mathcal A'(I-\\mathcal A'\\mathcal A')^{-1}
  + \\sum_{\\lambda\\in\\mathrm{eig}(\\mathcal A)}\\lambda(I-\\lambda\\mathcal A')^{-1},
\\qquad b = G\\,M\\,\\Gamma_0^{-1},
\\qquad \\mathcal A^{c} = \\mathcal A + b/T.
```

Two details are easy to get wrong and are deliberate here:

  - the middle term is ``\\mathcal A'(I - \\mathcal A'\\mathcal A')^{-1}``, i.e.
    the square of the *transpose*, not ``\\mathcal A'\\mathcal A``;
  - `T` is the **full** row count of the VAR data, not the number `Tu = T - p`
    of fitted residuals.

Two stability safeguards, both from the reference: if the OLS companion matrix
already has an eigenvalue outside the unit circle the correction is skipped
entirely (returning `δ = 0`); otherwise `δ` is lowered from 1 in steps of 0.01
until ``\\mathcal A + \\delta b/T`` is stable. Requesting the correction
therefore does not guarantee a full correction — inspect the returned `δ`.

Validated against Pope's closed form for the AR(1) case, where `b` must equal
``1 + 3\\rho`` exactly.
"""
function _pope_biascorrect(A::AbstractMatrix{Float64}, Σu::AbstractMatrix{Float64},
        T::Real)
    n, np = size(A)
    Acomp = np == n ? Matrix{Float64}(A) :
            [Matrix{Float64}(A); Matrix{Float64}(I, np - n, np - n) zeros(np - n, n)]
    maximum(abs, eigvals(Acomp)) > 1 && return (Matrix{Float64}(A), 0.0)
    G = zeros(Float64, np, np)
    G[1:n, 1:n] = Σu
    # Discrete Lyapunov solve: vec(𝒜 Γ 𝒜') = (𝒜 ⊗ 𝒜) vec(Γ).
    Γ0 = reshape((I - kron(Acomp, Acomp)) \ vec(G), np, np)
    At = Matrix{Float64}(transpose(Acomp))
    M = inv(I - At) + At * inv(I - At * At)
    for λ in eigvals(Acomp)
        M = M + λ * inv(I - λ * At)
    end
    b = real.(G * (M / Γ0))
    Acorr = Acomp + b ./ T
    δ = 1.0
    while maximum(abs, eigvals(Acorr)) > 1 && δ > 0
        δ -= 0.01
        Acorr = Acomp + δ .* b ./ T
    end
    return (Matrix{Float64}(Acorr[1:n, :]), δ)
end

"""
    _pope_biascorrect(A::Array{Float64,3}, Σu, T)

Method taking the `n × n × p` slope array produced by `_var_ols` and
returning a corrected array of the same shape.
"""
function _pope_biascorrect(A::Array{Float64, 3}, Σu::AbstractMatrix{Float64}, T::Real)
    n, _, p = size(A)
    flat = Matrix{Float64}(undef, n, n * p)
    @inbounds for l in 1:p
        flat[:, ((l - 1) * n + 1):(l * n)] = A[:, :, l]
    end
    corrected, δ = _pope_biascorrect(flat, Σu, T)
    out = Array{Float64}(undef, n, n, p)
    @inbounds for l in 1:p
        out[:, :, l] = corrected[:, ((l - 1) * n + 1):(l * n)]
    end
    return (out, δ)
end

"""
    _var_impact(U, innov_index) -> Vector{Float64}

Unit-impact structural shock vector. Port of the Cholesky step in
`_estim/var_ir_estim.m`, which obtains it as the numerically equivalent
regression of all residual columns on columns `1:innov_index` without a
constant, keeping the coefficient on column `innov_index`.

For `innov_index == 1` this reduces to ``\\Sigma_u[:,1]/\\Sigma_u[1,1]``, i.e.
``L_{:,1}/L_{11}`` for the lower-triangular Cholesky factor `L` — the impact
vector normalized to a unit change in the shock. It is always computed from the
**original OLS** residuals, never from Pope-corrected quantities.
"""
function _var_impact(U::AbstractMatrix{Float64}, innov_index::Int)
    Z = U[:, 1:innov_index]
    B = Z \ U                       # innov_index × n
    return Vector{Float64}(B[innov_index, :])
end

"""
    _var_irf(A, ν, H) -> Matrix{Float64}

VAR-implied structural impulse responses, `n × (H+1)`. Port of
`_estim/var_ir.m`: the recursion
``\\Theta_h = \\sum_{l=1}^{\\min(h,p)} A_l \\Theta_{h-l}`` with
``\\Theta_0 = I``, returning ``\\Theta_h \\nu`` at each horizon. Equivalent to
``e_r' C^h b`` on the companion form but cheaper.
"""
function _var_irf(A::Array{Float64, 3}, ν::AbstractVector{Float64}, H::Int)
    n, _, p = size(A)
    Θ = Vector{Matrix{Float64}}(undef, H + 1)
    Θ[1] = Matrix{Float64}(I, n, n)
    out = Matrix{Float64}(undef, n, H + 1)
    out[:, 1] = ν
    for h in 1:H
        M = zeros(Float64, n, n)
        for l in 1:min(h, p)
            M += A[:, :, l] * Θ[h - l + 1]
        end
        Θ[h + 1] = M
        out[:, h + 1] = M * ν
    end
    return out
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
function _mbb_residuals!(dest::AbstractMatrix{Float64}, U::AbstractMatrix{Float64},
        means::AbstractMatrix{Float64}, starts::AbstractVector{Int})
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
    _var_simulate(c, A, Ustar, Yinit) -> Matrix{Float64}

Iterate the fitted VAR forward from the drawn initial conditions. Port of
`_estim/var_sim.m`. `Yinit` supplies the first `p` rows verbatim and the
remaining `size(Ustar, 1)` rows follow
``Y_t^* = c + \\sum_{l=1}^{p} A_l Y_{t-l}^* + u_t^*``, so the returned matrix
has `p + size(Ustar, 1)` rows.
"""
function _var_simulate(c::AbstractVector{Float64}, A::Array{Float64, 3},
        Ustar::AbstractMatrix{Float64}, Yinit::AbstractMatrix{Float64})
    n, _, p = size(A)
    Tu = size(Ustar, 1)
    T = size(Yinit, 1) + Tu
    Y = Matrix{Float64}(undef, T, n)
    @inbounds Y[1:size(Yinit, 1), :] = Yinit
    @inbounds for t in (p + 1):T
        for j in 1:n
            Y[t, j] = c[j] + Ustar[t - p, j]
        end
        for l in 1:p, i in 1:n, j in 1:n
            Y[t, j] += A[j, i, l] * Y[t - l, i]
        end
    end
    return Y
end

"""
    _lhs_kind(formula) -> Symbol

Which response transform the local-projection formula uses: `:leads`, `:cumul`,
`:anchor`, or `:unknown`. Determines how the VAR-implied pseudo-truth must be
accumulated across horizons.
"""
function _lhs_kind(formula::FormulaTerm)
    lhs = formula.lhs
    if lhs isa FunctionTerm
        nm = nameof(lhs.f)
        nm === :leads && return :leads
        nm === :cumul && return :cumul
        (nm === :anchor || nm === :|) && return :anchor
    end
    return :unknown
end

_maybe_biascorrect(m::LocalProjection, apply::Bool) = apply ? biascorrect(m) : m

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
coefficients — the reference never recomputes them after bias correction.
`pope_delta` records the Pope safeguard outcome (`1.0` full correction, `0.0`
skipped because the OLS companion was already explosive, in between
attenuated); it is `NaN` when `popecorrect = false`, since no Pope step ran.

Build bands with [`summarize`](@ref); plot with `plot(b; levels = [0.68, 0.90])`.
"""
struct LPBootstrap{L <: LPEstimate}
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
    (fieldnames(LPBootstrap)..., propertynames(getfield(b, :lp))...)
end

function Base.show(io::IO, b::LPBootstrap)
    print(io, "LPBootstrap(horizon=0:", length(b.theta) - 1, ", term=", b.term,
        ", nboot=", b.nboot, ")")
end

function Base.show(io::IO, ::MIME"text/plain", b::LPBootstrap)
    println(io, "LPBootstrap (VAR residual moving-block bootstrap)")
    println(io, "  Term:          ", b.term)
    println(io, "  Horizons:      0:", length(b.theta) - 1)
    println(io, "  VAR variables: ", join(string.(b.vars), ", "),
        "   (p = ", b.nlags, ")")
    println(io, "  Draws:         ", b.nboot, " (block length ", b.blocklength,
        b.nfail > 0 ? ", $(b.nfail) failed)" : ")")
    println(io, "  LP correction: ",
        b.biascorrected ? "Herbst–Johannsen" : "none (uncorrected OLS)")
    println(io, "  VAR DGP:       ",
        b.popecorrected ?
        "Pope-corrected slopes (δ = $(round(b.pope_delta, digits = 2)))" :
        "OLS slopes")
    print(io, "  Bands:         pointwise, via summarize(b; level, method)")
    return nothing
end

"""
    coefpath(b::LPBootstrap; term=b.term) -> Vector{Float64}

The real-data impulse response path carried by the bootstrap result — the
bias-corrected path when `varbootstrap` was called with `biascorrect = true`.
"""
function coefpath(b::LPBootstrap; term::Symbol = b.term)
    term === b.term || throw(ArgumentError(
        "this bootstrap was run for term $(b.term); re-run `varbootstrap` " *
        "to bootstrap a different coefficient"))
    return copy(b.theta)
end

"""
    varbootstrap(lp_result, data; vars, nlags, kwargs...) -> LPBootstrap

VAR residual moving-block bootstrap with Hall percentile-`t` bands for a local
projection, following Montiel Olea, Plagborg-Møller, Qian & Wolf (2025). See
`docs/src/inference_procedures_guide.md` §4.

A reduced-form VAR in `vars` (with the shock ordered so that a Cholesky
identification recovers it) is fitted and used as the bootstrap DGP. In each
draw the VAR residuals are resampled in overlapping blocks with
position-specific recentering, the VAR is iterated from randomly drawn
contiguous initial conditions, and **the complete local projection is
re-estimated** on the artificial sample.

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
Only OLS local projections are supported, and the response transform must be
`leads` or `cumul`.

# Example
```julia
sel = lagselect(df, [:shock, :y]; maxlags = 10)
m = lp(@formula(leads(y) ~ shock + lags(y, 4) + lags(shock, 4)), df; horizon = 20)
b = varbootstrap(m, df; vars = [:shock, :y], nlags = nlags(sel), nboot = 1000)
summarize(b; level = 0.90)
```

See also [`summarize`](@ref), [`lagselect`](@ref), [`biascorrect`](@ref).
"""
function varbootstrap(lp_result::LocalProjection, data::AbstractDataFrame;
        vars::AbstractVector{Symbol},
        nlags::Integer,
        nboot::Integer = 1000,
        blocklength::Union{Integer, Nothing} = nothing,
        biascorrect::Bool = true,
        popecorrect::Bool = true,
        rng::AbstractRNG = Random.default_rng(),
        threaded::Bool = false)
    nboot >= 1 || throw(ArgumentError("`nboot` must be at least 1 (got $nboot)"))
    p = Int(nlags)

    varlist = collect(vars)
    innov_index = findfirst(isequal(lp_result.shock), varlist)
    innov_index === nothing && throw(ArgumentError(
        "the shock $(lp_result.shock) is not in `vars` = $(varlist); the VAR " *
        "data vector must contain the shock, in identification order"))
    resp_index = findfirst(isequal(lp_result.response), varlist)
    resp_index === nothing && throw(ArgumentError(
        "the response $(lp_result.response) is not in `vars` = $(varlist)"))

    needed = StatsModels.termvars(lp_result.base_formula)
    missing_vars = setdiff(unique(needed), varlist)
    isempty(missing_vars) || throw(ArgumentError(
        "the local projection formula refers to $(missing_vars), which are not " *
        "in `vars`; every variable the LP needs must be simulated by the VAR"))

    kind = _lhs_kind(lp_result.base_formula)
    kind in (:leads, :cumul) || throw(ArgumentError(
        "the VAR bootstrap supports `leads` and `cumul` responses; got " *
        "$(kind === :anchor ? "an anchored response" : "an unrecognized LHS")"))
    _rhs_tracks_horizon(lp_result.base_formula) && throw(ArgumentError(
        "the VAR bootstrap centers its t-statistic on the VAR-implied impulse " *
        "response to the shock, which is not the estimand of a local projection " *
        "with a horizon-tracking right-hand-side term (a bare `cumul(x)` or " *
        "`leads(x)`); use a horizon-invariant regressor set"))

    Y = _var_data(data, varlist)
    T = size(Y, 1)
    v = _var_ols(Y, p)
    ℓ = blocklength === nothing ? ceil(Int, 5.03 * T^(1 / 4)) : Int(blocklength)
    (1 <= ℓ <= v.Tu) || throw(ArgumentError(
        "`blocklength` = $ℓ must be between 1 and the number of VAR residuals " *
        "T - p = $(v.Tu)"))

    A_dgp, δ = popecorrect ? _pope_biascorrect(v.A, v.Σu, T) : (v.A, NaN)

    H = lp_result.horizon
    ν = _var_impact(v.U, innov_index)
    irf_var = _var_irf(A_dgp, ν, H)
    pseudo = Vector{Float64}(irf_var[resp_index, :])
    kind === :cumul && (pseudo = cumsum(pseudo))

    real_model = _maybe_biascorrect(lp_result, biascorrect)
    term = lp_result.shock
    theta = coefpath(real_model; term = term)
    se = stderror(vcov(HC1(), lp_result); term = term)

    means = _position_means(v.U, ℓ)
    nblocks = cld(v.Tu, ℓ)
    n_init = T - v.Tu
    seeds = rand(rng, UInt64, nboot)

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
            dfb = DataFrame(Yb, varlist)
            mb = lp(lp_result.base_formula, dfb; horizon = H, shock = term)
            cb = vcov(HC1(), mb)
            θb = coefpath(_maybe_biascorrect(mb, biascorrect); term = term)
            sb = stderror(cb; term = term)
            @inbounds for h in 1:(H + 1)
                theta_boot[b, h] = θb[h]
                se_boot[b, h] = sb[h]
            end
        catch
            failures[b] = 1
        end
        return nothing
    end

    if threaded
        Threads.@threads for b in 1:nboot
            run_draw(b)
        end
    else
        for b in 1:nboot
            run_draw(b)
        end
    end

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

    return LPBootstrap(real_model, term, theta, se, pseudo, theta_boot, se_boot,
        varlist, p, ℓ, Int(nboot), nfail, biascorrect, popecorrect, δ)
end

function varbootstrap(::LocalProjectionIV, ::AbstractDataFrame; kwargs...)
    throw(ArgumentError(
        "the VAR residual moving-block bootstrap is defined for OLS local " *
        "projections and is not available for LocalProjectionIV"))
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
       \\widehat\\theta_h - \\widehat{se}_h q^*_{h,\\alpha/2}]`` — note the
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
        isempty(ts) && throw(ErrorException(
            "no usable bootstrap draws at horizon $(h - 1); cannot form a " *
            "percentile-t interval"))
        return (b.theta[h] - b.se[h] * quantile(ts, 1 - α / 2),
            b.theta[h] - b.se[h] * quantile(ts, α / 2))
    end
    vals = Float64[x for x in θs if isfinite(x)]
    isempty(vals) && throw(ErrorException(
        "no usable bootstrap draws at horizon $(h - 1)"))
    ql = quantile(vals, α / 2)
    qu = quantile(vals, 1 - α / 2)
    method === :efron && return (ql, qu)
    # :hall
    return (b.theta[h] + b.pseudo_truth[h] - qu,
        b.theta[h] + b.pseudo_truth[h] - ql)
end

"""
    summarize(b::LPBootstrap; level=0.90, method=:hall_t, scale=1.0) -> IRFSummary

Bootstrap confidence bands for a local projection. `method` selects the
interval construction: `:hall_t` (Hall percentile-`t`, the recommended one),
`:hall`, or `:efron` — see `_boot_interval`.

The bands are **pointwise across horizons**, not simultaneous, and are
generally asymmetric around the point estimate. Note that a percentile-`t` or
Hall interval need not *contain* the point estimate: both correct for bias, so
when the bootstrap `t`-distribution is shifted the whole interval can sit to
one side of ``\\widehat\\theta_h``. That is a property of the method, not a
defect. The reported `coef` is the
real-data path (bias-corrected when `varbootstrap` was run with
`biascorrect = true`) and `se` the HC1 standard errors of the uncorrected
coefficients.
"""
function summarize(b::LPBootstrap; level::Real = 0.90, method::Symbol = :hall_t,
        scale::Real = 1.0)
    method in (:hall_t, :hall, :efron) || throw(ArgumentError(
        "`method` must be :hall_t, :hall or :efron (got :$method)"))
    level_f = Float64(level)
    (0 < level_f < 1) ||
        throw(ArgumentError("`level` must be in (0, 1) (got $level_f)"))
    scale_f = Float64(scale)
    α = 1 - level_f
    H = length(b.theta) - 1
    lower = Vector{Float64}(undef, H + 1)
    upper = Vector{Float64}(undef, H + 1)
    for h in 1:(H + 1)
        lo, hi = _boot_interval(b, h, α, method)
        lower[h] = lo * scale_f
        upper[h] = hi * scale_f
    end
    return IRFSummary(b.term, level_f, scale_f, collect(0:H),
        b.theta .* scale_f, b.se .* scale_f, lower, upper)
end

# ============================================================================
# Critical values (fixed-smoothing HAR inference)
# ============================================================================

"""
    _critical_distribution(estimator)

Reference distribution used to build confidence-interval critical values for
a given covariance estimator.

Defaults to `Normal()`. For the equal-weighted cosine estimator `EWC(B)`,
returns `TDist(B)`: under fixed-smoothing asymptotics the EWC long-run
variance is paired with Student-``t_B`` critical values for a single
restriction (Lazarus, Lewis, Stock & Watson, 2018). Pairing EWC with normal
critical values would understate the band width in finite samples.
"""
_critical_distribution(::Any) = Normal()
_critical_distribution(estimator::CovarianceMatrices.EWC) = TDist(estimator.B)

"""
    _critical_value(estimator, level::Real)

Two-sided critical value at confidence `level` from the reference
distribution paired with `estimator` (see `_critical_distribution`).
"""
function _critical_value(estimator, level::Real)
    quantile(_critical_distribution(estimator), 0.5 + level / 2)
end

"""
    ewc_bandwidth(T₀::Integer) -> Int
    ewc_bandwidth(lp::LPResult) -> Int

Number of cosine terms ``B`` for the equal-weighted cosine (EWC) long-run
variance estimator, using the Lazarus--Lewis--Stock--Watson (2018) rule for a
single restriction:

```math
B = \\lfloor 0.41\\,T_0^{2/3} \\rfloor
```

For a `LocalProjection`/`LocalProjectionIV`, ``T_0`` is the effective sample
size of the horizon-zero regression; the resulting integer is meant to be
held fixed across horizons.

# Example
```julia
lp_result = lp(@formula(leads(y) ~ x + lags(y, 4)), df; horizon = 20)
B = ewc_bandwidth(lp_result)
cov = vcov(EWC(B), lp_result)
summarize(lp_result, cov; level = 0.90)  # Student-t_B critical values
```
"""
ewc_bandwidth(T0::Integer) = max(1, floor(Int, 0.41 * T0^(2 / 3)))
ewc_bandwidth(lp::LPEstimate) = ewc_bandwidth(Int(nobs(lp.models[1])))

"""
    summarize(lp, cov; term=lp.shock, level=0.95, scale=1.0) -> IRFSummary

Create a summary table of impulse response coefficients with standard errors
and confidence intervals. Works for both `LocalProjection` and `LocalProjectionIV`.

Critical values are taken from the reference distribution paired with the
covariance estimator stored in `cov`: Student-``t_B`` for `EWC(B)`
(fixed-smoothing HAR inference, Lazarus et al. 2018), standard normal
otherwise.

For a [`BiasCorrectedLP`](@ref), the bands are centered on the
bias-corrected path but use the standard errors of the uncorrected OLS
coefficients (the reference convention; see [`biascorrect`](@ref)).
"""
function summarize(lp::LPEstimate, cov::LocalProjectionCovariance;
        term::Symbol = lp.shock, level::Real = 0.95, scale::Real = 1.0)
    level_f = Float64(level)
    scale_f = Float64(scale)
    beta = coefpath(lp; term = term) .* scale_f
    se = stderror(cov; term = term) .* scale_f
    z = _critical_value(cov.estimator, level_f)
    lower = beta .- z .* se
    upper = beta .+ z .* se

    IRFSummary(term, level_f, scale_f,
        collect(0:lp.horizon), beta, se, lower, upper)
end

"""
    summarize(lp, estimator; term=lp.shock, level=0.95, scale=1.0) -> IRFSummary

Convenience method that computes vcov internally before creating summary table.
"""
function summarize(lp::LPEstimate,
        estimator::CovarianceMatrices.AbstractAsymptoticVarianceEstimator;
        term::Symbol = lp.shock, level::Real = 0.95, scale::Real = 1.0)
    cov = vcov(estimator, lp)
    summarize(lp, cov; term = term, level = level, scale = scale)
end

# ============================================================================
# Makie Extension Stubs
# ============================================================================

"""
    irfplot(lp, estimator_or_cov; term=lp.shock, levels=[0.95])

Plot impulse response function from a local projection. Requires Makie.

Creates a line plot with confidence bands.

# Arguments
- `lp::Union{LocalProjection, LocalProjectionIV}`: estimated local projection
- `estimator_or_cov`: a `CovarianceMatrices` estimator or `LocalProjectionCovariance`
- `term::Symbol`: which coefficient to plot (default: shock variable)
- `levels::Vector{Float64}`: confidence levels for bands (default: `[0.95]`)
"""
# irfplot and irfplot! are imported from MacroEconometricTools (hub) — methods added by Makie extension

"""
    irfplot_axis(subfig, lp, estimator_or_cov; kwargs...)

Create an `Axis` with an IRF plot in the given sub-figure position. Requires Makie.

Returns `(subfig, ax, plot)`.
"""
function irfplot_axis end

# ============================================================================
# Plot Recipes using RecipesBase
# ============================================================================

"""
    IRFPlot

Internal wrapper type for dispatching plot recipes on LocalProjection/LocalProjectionIV
with covariance. Users should call `plot(lp, cov; ...)` or `plot(lp, estimator; ...)` directly.
"""
struct IRFPlot{L <: LPEstimate, E}
    lp::L
    cov::LocalProjectionCovariance{E}
    term::Symbol
    levels::Vector{Float64}
end

@recipe function f(wrapper::IRFPlot)
    lp = wrapper.lp
    cov = wrapper.cov
    term = wrapper.term
    levels = wrapper.levels

    beta = coefpath(lp; term = term)
    se = stderror(cov; term = term)
    horizons = collect(0:lp.horizon)

    sorted_levels = sort(levels; rev = true)
    for level in sorted_levels
        (level <= 0 || level >= 1) && throw(ArgumentError("levels must be in (0, 1)"))
    end

    z_max = _critical_value(cov.estimator, sorted_levels[1])
    ribbon_max = z_max .* se

    xlabel --> "Horizon"
    ylabel --> String(term)
    label --> "IRF"
    linewidth --> 2
    fillalpha --> 0.3
    legend --> :best

    if length(sorted_levels) > 1
        for (idx, level) in enumerate(sorted_levels[2:end])
            @series begin
                z = _critical_value(cov.estimator, level)
                band = z .* se
                ribbon := band
                fillalpha := 0.3 + 0.15 * idx
                label := ""
                linewidth := 0
                linecolor := :transparent
                horizons, beta
            end
        end
    end

    ribbon --> ribbon_max
    return horizons, beta
end

@recipe function f(lp::LPEstimate, cov::LocalProjectionCovariance;
        term = lp.shock, levels = [0.95])
    IRFPlot(lp, cov, term, Float64.(levels))
end

@recipe function f(lp::LPEstimate,
        estimator::CovarianceMatrices.AbstractAsymptoticVarianceEstimator;
        term = lp.shock, levels = [0.95])
    cov = vcov(estimator, lp)
    IRFPlot(lp, cov, term, Float64.(levels))
end

"""
    BootstrapIRFPlot

Internal wrapper for dispatching the plot recipe on [`LPBootstrap`](@ref).
Users should call `plot(b; levels, method)` directly.
"""
struct BootstrapIRFPlot{B <: LPBootstrap}
    boot::B
    levels::Vector{Float64}
    method::Symbol
end

@recipe function f(wrapper::BootstrapIRFPlot)
    b = wrapper.boot
    beta = b.theta
    horizons = collect(0:(length(beta) - 1))

    sorted_levels = sort(wrapper.levels; rev = true)
    for level in sorted_levels
        (level <= 0 || level >= 1) && throw(ArgumentError("levels must be in (0, 1)"))
    end
    bands = [summarize(b; level = lv, method = wrapper.method) for lv in sorted_levels]

    xlabel --> "Horizon"
    ylabel --> String(b.term)
    label --> "IRF"
    linewidth --> 2
    fillalpha --> 0.3
    legend --> :best

    # Bootstrap bands are asymmetric, so the ribbon takes a (below, above) pair.
    if length(sorted_levels) > 1
        for (idx, s) in enumerate(bands[2:end])
            @series begin
                ribbon := (beta .- s.lower, s.upper .- beta)
                fillalpha := 0.3 + 0.15 * idx
                label := ""
                linewidth := 0
                linecolor := :transparent
                horizons, beta
            end
        end
    end

    widest = bands[1]
    ribbon --> (beta .- widest.lower, widest.upper .- beta)
    return horizons, beta
end

@recipe function f(b::LPBootstrap; levels = [0.90], method = :hall_t)
    BootstrapIRFPlot(b, Float64.(levels), method)
end

# ============================================================================
# MacroEconometricTools Integration - LocalProjectionIRFResult Conversion
# ============================================================================

"""
    as_irf_result(lp; term=lp.shock, vcov_estimator=nothing, coverage=[0.68, 0.90, 0.95])

Convert a `LocalProjection` or `LocalProjectionIV` result to MET's `LocalProjectionIRFResult` type.

This enables using MET's unified plotting infrastructure with AxisArray support.
For IV results, first-stage diagnostics are included in metadata.

# Keyword Arguments
- `term::Symbol=lp.shock`: Which coefficient to extract (defaults to shock variable)
- `vcov_estimator=nothing`: Covariance estimator for confidence bands (e.g., `HR1()`)
- `coverage::Vector{Float64}=[0.68, 0.90, 0.95]`: Confidence levels for bands

# Returns
- `LocalProjectionIRFResult`: MET-compatible IRF result with AxisArray data

# Example
```julia
using LocalProjections, MacroEconometricTools, CovarianceMatrices

lp_result = lp(@formula(leads(y) ~ x), df; horizon=12)
irf = as_irf_result(lp_result; vcov_estimator=HR1())

# AxisArray indexing
irf.data[response=:y, shock=:x, horizon=0:6]

# Use with MET's plotting
using Plots
plot(irf)  # Uses MET's recipe
```
"""
function as_irf_result(lp::LPEstimate;
        term::Symbol = lp.shock,
        vcov_estimator = nothing,
        coverage::Vector{Float64} = [0.68, 0.90, 0.95])
    T = Float64

    # Extract coefficient path
    beta = coefpath(lp; term = term)
    H = length(beta)
    horizons = 0:(H - 1)

    # Build AxisArray with named dimensions
    data = AxisArray(
        reshape(beta, 1, 1, H),
        Axis{:response}([lp.response]),
        Axis{:shock}([term]),
        Axis{:horizon}(horizons)
    )

    if vcov_estimator !== nothing
        cov = vcov(vcov_estimator, lp)
        se = stderror(cov; term = term)

        stderr_arr = AxisArray(
            reshape(se, 1, 1, H),
            Axis{:response}([lp.response]),
            Axis{:shock}([term]),
            Axis{:horizon}(horizons)
        )

        # Compute confidence bands as AxisArrays
        # Critical values follow the estimator pairing (Student-t_B for EWC)
        lower = [AxisArray(
                     reshape(beta .- _critical_value(vcov_estimator, c) .* se, 1, 1, H),
                     Axis{:response}([lp.response]),
                     Axis{:shock}([term]),
                     Axis{:horizon}(horizons)
                 ) for c in coverage]

        upper = [AxisArray(
                     reshape(beta .+ _critical_value(vcov_estimator, c) .* se, 1, 1, H),
                     Axis{:response}([lp.response]),
                     Axis{:shock}([term]),
                     Axis{:horizon}(horizons)
                 ) for c in coverage]
    else
        stderr_arr = AxisArray(zeros(T, 1, 1, H),
            Axis{:response}([lp.response]),
            Axis{:shock}([term]),
            Axis{:horizon}(horizons))
        lower = [copy(stderr_arr) for _ in coverage]
        upper = [copy(stderr_arr) for _ in coverage]
    end

    # Build metadata, adding IV-specific fields when applicable
    meta = (horizon = lp.horizon, formula = lp.base_formula, term = term)
    if lp isa BiasCorrectedLP
        meta = (meta..., bias_corrected = true)
    end
    if lp isa LocalProjectionIV
        fs_diagnostics = [first_stage(lp, h) for h in 0:(H - 1)]
        meta = (meta..., is_iv = true, first_stage = fs_diagnostics)
    end

    return LocalProjectionIRFResult{T, typeof(data)}(
        data, stderr_arr, lower, upper, coverage, meta
    )
end

"""
    as_irf_result(b::LPBootstrap; term=b.term, coverage=[0.68, 0.90], method=:hall_t)

Convert a [`LPBootstrap`](@ref) to a `LocalProjectionIRFResult`, carrying the
bootstrap bands rather than normal-approximation ones.

The bands are those of [`summarize`](@ref) under `method`, so they are
asymmetric and **pointwise**. `stderr` holds the real-data HC1 standard errors
of the uncorrected coefficients — the reference convention — which are *not*
what the bands are built from. The metadata records `bootstrap = true` along
with the draw count, block length, VAR lag length, failed draws and which
corrections were applied.
"""
function as_irf_result(b::LPBootstrap;
        term::Symbol = b.term,
        coverage::Vector{Float64} = [0.68, 0.90],
        method::Symbol = :hall_t)
    term === b.term || throw(ArgumentError(
        "this bootstrap was run for term $(b.term), not $term"))
    T = Float64
    beta = coefpath(b; term = term)
    H = length(beta)
    horizons = 0:(H - 1)
    resp = b.response

    _ax(v) = AxisArray(
        reshape(collect(T, v), 1, 1, H),
        Axis{:response}([resp]),
        Axis{:shock}([term]),
        Axis{:horizon}(horizons)
    )

    data = _ax(beta)
    stderr_arr = _ax(b.se)
    bands = [summarize(b; level = c, method = method) for c in coverage]
    lower = [_ax(s.lower) for s in bands]
    upper = [_ax(s.upper) for s in bands]

    meta = (horizon = H - 1,
        formula = b.base_formula,
        term = term,
        bootstrap = true,
        bootstrap_method = method,
        nboot = b.nboot,
        nfail = b.nfail,
        blocklength = b.blocklength,
        var_lags = b.nlags,
        var_variables = b.vars,
        bias_corrected = b.biascorrected,
        pope_corrected = b.popecorrected,
        pope_delta = b.pope_delta)

    return LocalProjectionIRFResult{T, typeof(data)}(
        data, stderr_arr, lower, upper, coverage, meta
    )
end

end # module LocalProjections
