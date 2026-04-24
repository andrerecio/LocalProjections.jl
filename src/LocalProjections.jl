module LocalProjections

export LocalProjection, LocalProjectionIV, LocalProjectionCovariance, IRFSummary
export lp, lpiv, coefpath, stderror, vcov, summarize, first_stage, weakivtest
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
using Statistics
using Distributions
using RecipesBase
using ShiftedArrays: lag, lead
using StatsBase
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

    # Extract X matrix (this handles all lag/lead transformations on RHS)
    X_raw = StatsModels.modelcols(mf_base.f.rhs, mf_base.data)

    # Convert missing to NaN for type stability (handles lag(w) which returns missing)
    # This ensures X is always Float64, matching our leads/cumul which use default=NaN
    X = Matrix{Float64}(map(v -> ismissing(v) ? NaN : Float64(v), X_raw))

    # Store coefficient names (constant across all horizons)
    coef_names_base = Vector{String}(coefnames(mf_base))

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

    # Function barrier: X and coef_names_base are now concretely typed (Matrix{Float64},
    # Vector{String}), so _lp_estimate_horizons can be fully inferred by the compiler.
    return _lp_estimate_horizons(X, coef_names_base, df_base_complete, horizon,
        response, shock_symbol, formula,
        is_anchor, is_cumulative, is_leads, anchor_term, cumul_term, leads_term)
end

"""
    _lp_estimate_horizons(X, coef_names_base, df, horizon, response, shock, formula, ...)

Function barrier for type-stable per-horizon OLS estimation.
Called after formula parsing to ensure X::Matrix{Float64} is concretely typed,
breaking the type instability cascade from StatsModels' abstract return types.
"""
function _lp_estimate_horizons(X::Matrix{Float64}, coef_names_base::Vector{String},
        df_base_complete, horizon, response, shock_symbol, formula,
        is_anchor, is_cumulative, is_leads, anchor_term, cumul_term, leads_term)
    # Identify rows where X is complete (no NaN values)
    X_missing_ind = vec(all(!isnan, X, dims = 2))

    # Helper to estimate one horizon
    function _estimate_horizon(h)
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

    # Stage 2: Build X (endogenous + exogenous) and Z (instruments + exogenous) matrices ONCE

    # Get exogenous matrix (includes intercept from StatsModels)
    # Create a dummy formula for exogenous terms
    if !isempty(exo_terms)
        exo_tuple = length(exo_terms) == 1 ? exo_terms[1] : Tuple(exo_terms)
        dummy_formula_exo = StatsModels.FormulaTerm(StatsModels.Term(response), exo_tuple)
        mf_exo = StatsModels.ModelFrame(dummy_formula_exo, df_base_complete)
        X_exo_raw = StatsModels.modelcols(mf_exo.f.rhs, mf_exo.data)
        X_exo = Matrix{Float64}(map(v -> ismissing(v) ? NaN : Float64(v), X_exo_raw))
        coef_names_exo = Vector{String}(coefnames(mf_exo))
    else
        # No exogenous terms - just intercept
        n_obs = nrow(df_base_complete)
        X_exo = ones(n_obs, 1)
        coef_names_exo = ["(Intercept)"]
    end

    # Get endogenous matrix - extract columns directly from terms (no formula intercept)
    endo_names = String[]
    X_endo_cols = Matrix{Float64}[]
    for endo_term in endo_terms
        # Use modelcols directly on the term to avoid intercept
        endo_col_raw = StatsModels.modelcols(endo_term, df_base_complete)
        endo_col = map(v -> ismissing(v) ? NaN : Float64(v), endo_col_raw)
        push!(X_endo_cols,
            endo_col isa AbstractMatrix ? Matrix{Float64}(endo_col) :
            reshape(Vector{Float64}(endo_col), :, 1))
        term_names = StatsModels.coefnames(endo_term)
        append!(endo_names, term_names isa Vector ? term_names : [term_names])
    end
    X_endo = hcat(X_endo_cols...)::Matrix{Float64}

    # Get instrument matrix - extract columns directly from terms (no formula intercept)
    instr_names = String[]
    Z_instr_cols = Matrix{Float64}[]
    for instr_term in instr_terms
        # Use modelcols directly on the term to avoid intercept
        instr_col_raw = StatsModels.modelcols(instr_term, df_base_complete)
        instr_col = map(v -> ismissing(v) ? NaN : Float64(v), instr_col_raw)
        push!(Z_instr_cols,
            instr_col isa AbstractMatrix ? Matrix{Float64}(instr_col) :
            reshape(Vector{Float64}(instr_col), :, 1))
        term_names = StatsModels.coefnames(instr_term)
        append!(instr_names, term_names isa Vector ? term_names : [term_names])
    end
    Z_instr = hcat(Z_instr_cols...)::Matrix{Float64}

    # Build full X = [exogenous, endogenous] and Z = [exogenous, instruments]
    X_full = hcat(X_exo, X_endo)::Matrix{Float64}
    Z_full = hcat(X_exo, Z_instr)::Matrix{Float64}

    n_endogenous = size(X_endo, 2)
    coef_names_base = Vector{String}(vcat(coef_names_exo, endo_names))

    # Check order condition
    n_instruments = size(Z_instr, 2)
    n_instruments >= n_endogenous ||
        throw(ArgumentError("Not enough instruments: $n_instruments < $n_endogenous (order condition violated)"))

    # Set shock variable
    if shock === nothing
        shock_symbol = Symbol(endo_names[1])
    else
        shock_symbol = shock
    end

    # Function barrier: X_full, Z_full, coef_names_base are now concretely typed,
    # so _lpiv_estimate_horizons can be fully inferred by the compiler.
    return _lpiv_estimate_horizons(X_full, Z_full, coef_names_base, df_base_complete,
        horizon, n_endogenous, response, shock_symbol, formula,
        endo_names, instr_names,
        is_anchor, is_cumulative, is_leads, anchor_term, cumul_term, leads_term)
end

"""
    _lpiv_estimate_horizons(X_full, Z_full, coef_names_base, df, ...)

Function barrier for type-stable per-horizon IV estimation.
Called after formula parsing and matrix assembly to ensure all matrix arguments
are concretely typed as Matrix{Float64}.
"""
function _lpiv_estimate_horizons(X_full::Matrix{Float64}, Z_full::Matrix{Float64},
        coef_names_base::Vector{String}, df_base_complete,
        horizon, n_endogenous, response, shock_symbol, formula,
        endo_names, instr_names,
        is_anchor, is_cumulative, is_leads, anchor_term, cumul_term, leads_term)
    # Identify complete rows
    X_missing_ind = vec(all(!isnan, X_full, dims = 2))
    Z_missing_ind = vec(all(!isnan, Z_full, dims = 2))
    XZ_complete = X_missing_ind .& Z_missing_ind

    # Helper to estimate one horizon
    function _estimate_iv_horizon(h)
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
    weakivtest(lpiv::LocalProjectionIV, h::Int; kwargs...) -> WeakIVTestResult

Montiel-Olea-Pflueger robust weak instrument test for horizon `h` (0-indexed).

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
    return Regress.weakivtest(lpiv.models[h + 1]; kwargs...)
end

"""
    weakivtest(lpiv::LocalProjectionIV; kwargs...) -> Vector{WeakIVTestResult}

Montiel-Olea-Pflueger robust weak instrument test for all horizons.

Returns a vector of `WeakIVTestResult`, one per horizon (0 to `lpiv.horizon`).
"""
function Regress.weakivtest(lpiv::LocalProjectionIV; kwargs...)
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

"""
    vcov(estimator, lp)

Compute diagonal covariance entries horizon-by-horizon using `estimator`
from `CovarianceMatrices.jl`. Works for both `LocalProjection` and `LocalProjectionIV`.
"""
function vcov(estimator, lp::LPResult)
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

"""
    summarize(lp, cov; term=lp.shock, level=0.95, scale=1.0) -> IRFSummary

Create a summary table of impulse response coefficients with standard errors
and confidence intervals. Works for both `LocalProjection` and `LocalProjectionIV`.
"""
function summarize(lp::LPResult, cov::LocalProjectionCovariance;
        term::Symbol = lp.shock, level::Real = 0.95, scale::Real = 1.0)
    level_f = Float64(level)
    scale_f = Float64(scale)
    beta = coefpath(lp; term = term) .* scale_f
    se = stderror(cov; term = term) .* scale_f
    z = quantile(Normal(), 0.5 + level_f / 2)
    lower = beta .- z .* se
    upper = beta .+ z .* se

    IRFSummary(term, level_f, scale_f,
        collect(0:lp.horizon), beta, se, lower, upper)
end

"""
    summarize(lp, estimator; term=lp.shock, level=0.95, scale=1.0) -> IRFSummary

Convenience method that computes vcov internally before creating summary table.
"""
function summarize(lp::LPResult,
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
struct IRFPlot{L <: LPResult, E}
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

    z_max = quantile(Normal(), 0.5 + sorted_levels[1] / 2)
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
                z = quantile(Normal(), 0.5 + level / 2)
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

@recipe function f(lp::LPResult, cov::LocalProjectionCovariance;
        term = lp.shock, levels = [0.95])
    IRFPlot(lp, cov, term, Float64.(levels))
end

@recipe function f(lp::LPResult,
        estimator::CovarianceMatrices.AbstractAsymptoticVarianceEstimator;
        term = lp.shock, levels = [0.95])
    cov = vcov(estimator, lp)
    IRFPlot(lp, cov, term, Float64.(levels))
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
function as_irf_result(lp::LPResult;
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
        lower = [AxisArray(
                     reshape(beta .- quantile(Normal(), 0.5 + c/2) .* se, 1, 1, H),
                     Axis{:response}([lp.response]),
                     Axis{:shock}([term]),
                     Axis{:horizon}(horizons)
                 ) for c in coverage]

        upper = [AxisArray(
                     reshape(beta .+ quantile(Normal(), 0.5 + c/2) .* se, 1, 1, H),
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
    if lp isa LocalProjectionIV
        fs_diagnostics = [first_stage(lp, h) for h in 0:(H - 1)]
        meta = (meta..., is_iv = true, first_stage = fs_diagnostics)
    end

    return LocalProjectionIRFResult{T, typeof(data)}(
        data, stderr_arr, lower, upper, coverage, meta
    )
end

end # module LocalProjections
