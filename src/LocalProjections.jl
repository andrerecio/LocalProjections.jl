module LocalProjections

export LocalProjection, LocalProjectionIV, LocalProjectionCovariance, IRFSummary
export lp, lpiv, coefpath, stderror, vcov, summarize, first_stage, weakivtest
export LPState, statetest, StateTestResult
export ewc_bandwidth
export biascorrect, BiasCorrectedLP
export lagselect, VARLagSelection, nlags
export varbootstrap, LPBootstrap
export fevd, LPFEVD, fevdbootstrap, FEVDBootstrap, FEVDSummary
export WeakIVTestResult, FirstStageIV
export lag, lead, cumul, CumulTerm, lags, leads, LeadTerm, anchor, AnchorTerm
export ldiff, hbr, LongDiffTerm, firstdiff, DiffTerm
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
using LinearAlgebra: I, Symmetric, cholesky, cond, dot, eigen, eigmin, eigvals,
                     factorize, Factorization, issuccess, kron, logdet, lu, tr
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
    # cumul(ldiff(y)) sums the changes at horizons 0, ..., h, so the horizon
    # belongs to cumul: cumul(ldiff(y), 3), not cumul(ldiff(y, 3)).
    term isa LongDiffTerm && term.horizon !== nothing &&
        throw(ArgumentError(
            "inside cumul(), write $(_longdiff_name(term))($(term.term)) without a horizon; " *
            "pin the horizon on cumul instead, e.g. cumul($(_longdiff_name(term))(...), 3)"))
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
# Long differences (ldiff, hbr) and first differences (firstdiff)
# ldiff(y):   y_{t+h} - y_{t-1}                  (Piger & Stockwell 2025)
# hbr(x, y):  (x_{t+h} - x_{t-1}) / y_{t-1}      (Hall 2009; Barro & Redlick 2011)
# firstdiff(x): x_t - x_{t-1}, for lagged differences on the right-hand side
# ============================================================================

"""
    ldiff(term)
    ldiff(term, horizon)

Long difference ``y_{t+h} - y_{t-1}``, the response of the long-difference local
projections of Piger and Stockwell (2025):

- `@formula(ldiff(y) ~ shock + lags(firstdiff(y), 12))` — horizon from `lp()`
- `@formula(ldiff(y, 3) ~ x)` — explicit horizon for standalone use
- `@formula(cumul(ldiff(y)) ~ x)` — ``\\sum_{j=0}^{h} (y_{t+j} - y_{t-1})``

The baseline ``y_{t-1}`` is fixed while the response moves to ``t+h``; at
``h = 0`` the long difference is the first difference ``\\Delta y_t``. On the
right-hand side a bare `ldiff(x)` tracks the projection horizon, like
`cumul(x)`. See also [`hbr`](@ref) and [`firstdiff`](@ref).
"""
ldiff(t::AbstractTerm) = LongDiffTerm(t, nothing, nothing)
ldiff(t::AbstractTerm, h::Int) = LongDiffTerm(t, nothing, h)

"""
    hbr(x)
    hbr(x, y)
    hbr(x, horizon)
    hbr(x, y, horizon)

Hall–Barro–Redlick transformation ``(x_{t+h} - x_{t-1}) / y_{t-1}``: the change
in `x` since ``t-1`` in units of ``y_{t-1}``. `hbr(x)` scales by ``x_{t-1}``
itself (a growth rate). With `x` and `y` in levels, e.g. real government
spending and real GDP per capita, the responses of GDP and spending share one
unit, so their ratio is a dollar multiplier (Hall 2009; Barro and Redlick
2011; Ramey and Zubairy 2018):

```julia
# Impulse responses in HBR units
lp(@formula(hbr(gdp) ~ news + lags(news, 4) + lags(log(gdp), 4) + lags(log(gov), 4)),
   df; horizon = 20)

# One-step cumulative multiplier: sum_j (Y_{t+j} - Y_{t-1}) / Y_{t-1} on
# sum_j (G_{t+j} - G_{t-1}) / Y_{t-1}, spending instrumented by the news shock
lpiv(@formula(cumul(hbr(gdp)) ~ (cumul(hbr(gov, gdp)) ~ news) + lags(news, 4) +
              lags(log(gdp), 4) + lags(log(gov), 4)), df; horizon = 20)
```

A numeric second argument is a horizon, so `hbr(x, 3)` is `hbr(x, x, 3)`. As
with [`ldiff`](@ref), `cumul(hbr(x, y))` sums the transformed changes over
horizons ``0, \\dots, h`` and a bare `hbr` on the right-hand side tracks the
projection horizon.
"""
hbr(t::AbstractTerm) = LongDiffTerm(t, t, nothing)
hbr(t::AbstractTerm, h::Int) = LongDiffTerm(t, t, h)
hbr(t::AbstractTerm, s::AbstractTerm) = LongDiffTerm(t, s, nothing)
hbr(t::AbstractTerm, s::AbstractTerm, h::Int) = LongDiffTerm(t, s, h)

"""
    firstdiff(term)

First difference ``x_t - x_{t-1}`` (`NaN` in the first row). Its main use is on
the right-hand side of a long-difference projection, where the controls are
lagged differences: `lags(firstdiff(y), 12)` gives ``\\Delta y_{t-1}, \\dots,
\\Delta y_{t-12}``.
"""
firstdiff(t::AbstractTerm) = DiffTerm(t)

"""
    LongDiffTerm

Term behind [`ldiff`](@ref) and [`hbr`](@ref): the change
``(x_{t+h} - x_{t-1}) / s_{t-1}``, with `scale` ``s`` equal to `nothing` (no
scaling) for `ldiff`. `horizon` is `nothing` until the projection pins it.
"""
struct LongDiffTerm{T <: AbstractTerm, S <: Union{Nothing, AbstractTerm}} <: AbstractTerm
    term::T
    scale::S
    horizon::Union{Int, Nothing}
end

"""
    DiffTerm

Term behind [`firstdiff`](@ref): ``x_t - x_{t-1}``.
"""
struct DiffTerm{T <: AbstractTerm} <: AbstractTerm
    term::T
end

_longdiff_name(t::LongDiffTerm) = t.scale === nothing ? "ldiff" : "hbr"

StatsModels.termvars(t::FunctionTerm{typeof(ldiff)}) = _termvars_unary(t)
StatsModels.termvars(t::FunctionTerm{typeof(firstdiff)}) = _termvars_unary(t)
function StatsModels.termvars(t::FunctionTerm{typeof(hbr)})
    unique(reduce(vcat, (StatsModels.termvars(a) for a in t.args); init = Symbol[]))
end
function StatsModels.termvars(t::LongDiffTerm)
    t.scale === nothing && return StatsModels.termvars(t.term)
    unique(vcat(StatsModels.termvars(t.term), StatsModels.termvars(t.scale)))
end
StatsModels.termvars(t::DiffTerm) = StatsModels.termvars(t.term)

StatsModels.terms(t::LongDiffTerm) = t.scale === nothing ? (t.term,) : (t.term, t.scale)
StatsModels.terms(t::DiffTerm) = (t.term,)

function StatsModels.apply_schema(
        t::FunctionTerm{typeof(ldiff)}, sch::StatsModels.Schema, ctx::Type)
    term, horizon = _parse_unary_binary_args(t, "ldiff", nothing)
    return LongDiffTerm(StatsModels.apply_schema(term, sch, ctx), nothing, horizon)
end

function StatsModels.apply_schema(
        t::FunctionTerm{typeof(hbr)}, sch::StatsModels.Schema, ctx::Type)
    args = collect(t.args)
    horizon = nothing
    if length(args) >= 2 && last(args) isa ConstantTerm
        horizon = pop!(args).n
    end
    (1 <= length(args) <= 2 && !any(a -> a isa ConstantTerm, args)) ||
        throw(ArgumentError("hbr() takes a variable, an optional scale variable " *
                            "and an optional horizon, e.g. hbr(g, y) or hbr(g, y, 3)"))
    term = StatsModels.apply_schema(args[1], sch, ctx)
    scale = length(args) == 2 ? StatsModels.apply_schema(args[2], sch, ctx) : term
    return LongDiffTerm(term, scale, horizon)
end

function StatsModels.apply_schema(t::LongDiffTerm, sch::StatsModels.Schema, ctx::Type)
    scale = t.scale === nothing ? nothing : StatsModels.apply_schema(t.scale, sch, ctx)
    LongDiffTerm(StatsModels.apply_schema(t.term, sch, ctx), scale, t.horizon)
end

function StatsModels.apply_schema(
        t::FunctionTerm{typeof(firstdiff)}, sch::StatsModels.Schema, ctx::Type)
    length(t.args) == 1 || throw(ArgumentError("firstdiff() takes one variable"))
    return DiffTerm(StatsModels.apply_schema(only(t.args), sch, ctx))
end

function StatsModels.apply_schema(t::DiffTerm, sch::StatsModels.Schema, ctx::Type)
    DiffTerm(StatsModels.apply_schema(t.term, sch, ctx))
end

"`modelcols` of a single-column term as a `Vector{Float64}` with `NaN` for `missing`."
function _float_column(t::AbstractTerm, d::Tables.ColumnTable, context::String)
    col = _extract_single_column(StatsModels.modelcols(t, d), context)
    return Vector{Float64}(map(v -> ismissing(v) ? NaN : Float64(v), col))
end

function _longdiff_scale(t::LongDiffTerm, d::Tables.ColumnTable)
    t.scale === nothing && return nothing
    return _float_column(t.scale, d, "hbr() scale variable")
end

"""
    _create_long_difference(x, s, h, cumulative) -> Vector{Float64}

``(x_{t+h} - x_{t-1}) / s_{t-1}``, or with `cumulative = true`
``\\sum_{j=0}^{h} (x_{t+j} - x_{t-1}) / s_{t-1}``; no scaling when `s` is
`nothing`. `NaN` where a lead, the baseline or the scale is unavailable.
"""
function _create_long_difference(x::Vector{Float64}, s::Union{Nothing, Vector{Float64}},
        h::Int, cumulative::Bool)
    n = length(x)
    ahead = cumulative ? _create_cumulative(x, h) : collect(lead(x, h, default = NaN))
    k = cumulative ? h + 1 : 1
    out = fill(NaN, n)
    @inbounds for t in 2:n
        v = ahead[t] - k * x[t - 1]
        out[t] = s === nothing ? v : v / s[t - 1]
    end
    any(isinf, out) && throw(ArgumentError(
        "hbr(): the scale variable is zero at a period t - 1 used by the projection"))
    return out
end

function StatsModels.modelcols(t::LongDiffTerm, d::Tables.ColumnTable)
    _check_horizon_provided(t.horizon, _longdiff_name(t))
    x = _float_column(t.term, d, "$(_longdiff_name(t))() response")
    return _create_long_difference(x, _longdiff_scale(t, d), t.horizon, false)
end

# cumul(ldiff(y)) and cumul(hbr(x, y)): the changes relative to t - 1 are summed
# over horizons 0, ..., h, so the baseline enters h + 1 times, not once.
function StatsModels.modelcols(ct::CumulTerm{<:LongDiffTerm}, d::Tables.ColumnTable)
    _check_horizon_provided(ct.horizon, "cumul")
    t = ct.term
    x = _float_column(t.term, d, "$(_longdiff_name(t))() response")
    return _create_long_difference(x, _longdiff_scale(t, d), ct.horizon, true)
end

function StatsModels.modelcols(t::DiffTerm, d::Tables.ColumnTable)
    x = _float_column(t.term, d, "firstdiff() variable")
    out = fill(NaN, length(x))
    for i in 2:length(x)
        out[i] = x[i] - x[i - 1]
    end
    return out
end

StatsModels.width(::LongDiffTerm) = 1
StatsModels.width(::DiffTerm) = 1

function _longdiff_label(t::LongDiffTerm)
    x = StatsModels.coefnames(t.term)
    x = x isa AbstractVector ? x[1] : x
    args = if t.scale === nothing || t.scale == t.term
        x
    else
        s = StatsModels.coefnames(t.scale)
        x * ", " * (s isa AbstractVector ? s[1] : s)
    end
    t.horizon === nothing || (args *= ", $(t.horizon)")
    return _longdiff_name(t) * "(" * args * ")"
end

StatsModels.coefnames(t::LongDiffTerm) = [_longdiff_label(t)]
Base.show(io::IO, t::LongDiffTerm) = print(io, _longdiff_label(t))

function StatsModels.coefnames(t::DiffTerm)
    x = StatsModels.coefnames(t.term)
    return ["firstdiff(" * (x isa AbstractVector ? x[1] : x) * ")"]
end
Base.show(io::IO, t::DiffTerm) = print(io, "firstdiff(", t.term, ")")

_extract_base_variables(t::LongDiffTerm) = StatsModels.termvars(t)
_extract_base_variables(t::DiffTerm) = _extract_base_variables(t.term)

# ============================================================================
# The response at horizon h
# `lp`/`lpiv` build the right-hand side once and rebuild only the response at
# every horizon. Which response depends on the left-hand-side term alone, so
# the per-horizon loop asks `_lhs_at(lhs, h)` and each term type answers.
# ============================================================================

const _LHS_HELP = "the left-hand side of a local projection must be leads(y), " *
                  "cumul(y), anchor(y, z), ldiff(y), hbr(x, y) or cumul() of the last two"

"""
    _lhs_at(lhs, h) -> AbstractTerm

The response term of horizon `h` for the left-hand side `lhs`: `leads(y)` →
``y_{t+h}``, `cumul(y)` → ``\\sum_{j\\le h} y_{t+j}``, `anchor(y, z)` →
``y_{t+h} - z_t``, `ldiff(y)` → ``y_{t+h} - y_{t-1}``, `hbr(x, y)` →
``(x_{t+h} - x_{t-1})/y_{t-1}``, `cumul(ldiff(y))` / `cumul(hbr(x, y))` → the
sums over horizons ``0, \\dots, h``. A horizon written on the left-hand side
is overridden: the projection sets it.
"""
_lhs_at(::Any, ::Int) = throw(ArgumentError(_LHS_HELP))
_lhs_at(t::LeadTerm, h::Int) = LeadTerm(t.term, h)
_lhs_at(t::CumulTerm, h::Int) = CumulTerm(t.term, h)
_lhs_at(t::LongDiffTerm, h::Int) = LongDiffTerm(t.term, t.scale, h)
function _lhs_at(t::AnchorTerm, h::Int)
    # A plain response inside anchor(y, z) (or y | z) is read as leads(y)
    inner = t.response isa Union{LeadTerm, CumulTerm, LongDiffTerm} ?
            _lhs_at(t.response, h) : LeadTerm(t.response, h)
    return AnchorTerm(inner, t.anchor, 0)
end

"""
    _lhs_response(lhs) -> Symbol

The response variable named by the left-hand side (for `hbr(x, y)` it is `x`;
`y` only scales it).
"""
_lhs_response(::Any) = throw(ArgumentError(_LHS_HELP))
_lhs_response(t::LeadTerm) = _extract_single_response(t, "leads() term")
_lhs_response(t::CumulTerm) = _extract_single_response(t, "cumul() term")
_lhs_response(t::CumulTerm{<:LongDiffTerm}) = _lhs_response(t.term)
function _lhs_response(t::LongDiffTerm)
    _extract_single_response(t.term, "$(_longdiff_name(t))() response")
end
function _lhs_response(t::AnchorTerm)
    _extract_single_response(t.response, "anchor() response term")
end

"""
    _lhs_tautological(lhs) -> Bool

Whether regressing the horizon-0 response on the response variable itself is an
identity (coefficient 1, zero variance), which `lp`/`lpiv` then impose exactly.
Not so for long differences: ``y_t - y_{t-1}`` on ``y_t`` has coefficient 1 only
when ``y_{t-1}`` is among the controls.
"""
_lhs_tautological(::Any) = true
_lhs_tautological(::LongDiffTerm) = false
_lhs_tautological(::CumulTerm{<:LongDiffTerm}) = false
_lhs_tautological(t::AnchorTerm) = _lhs_tautological(t.response)

# ============================================================================
# Horizon-tracking terms on the right-hand side
# ============================================================================

"""
    _has_dynamic_horizon(term) -> Bool

`true` when `term` contains a `CumulTerm`, `LeadTerm` or `LongDiffTerm`
(`ldiff`/`hbr`) whose horizon is `nothing` — a right-hand-side term that must be rebuilt at every projection
horizon rather than once. `cumul(x, 3)` and `leads(x, 3)` pin an explicit
horizon and are therefore *not* dynamic.
"""
_has_dynamic_horizon(::Any) = false
_has_dynamic_horizon(t::CumulTerm) = t.horizon === nothing || _has_dynamic_horizon(t.term)
_has_dynamic_horizon(t::LeadTerm) = t.horizon === nothing || _has_dynamic_horizon(t.term)
function _has_dynamic_horizon(t::AnchorTerm)
    _has_dynamic_horizon(t.response) || _has_dynamic_horizon(t.anchor)
end
_has_dynamic_horizon(t::LongDiffTerm) = t.horizon === nothing
# cumul(ldiff(y), 3) is pinned as a whole: its inner change has no horizon of its own
_has_dynamic_horizon(t::CumulTerm{<:LongDiffTerm}) = t.horizon === nothing
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
function _specialize_horizon(t::LongDiffTerm, h::Int)
    return t.horizon === nothing ? LongDiffTerm(t.term, t.scale, h) : t
end
function _specialize_horizon(t::CumulTerm{<:LongDiffTerm}, h::Int)
    return t.horizon === nothing ? CumulTerm(t.term, h) : t
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
`cumul(x)`, `leads(x)`, `ldiff(x)` or `hbr(x, y)` — a regressor that is rebuilt at every projection
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
    # cumul(ldiff(y), 3): the pinned cumul horizon also fixes the inner change
    nm === :cumul && _is_longdiff_call(first(t.args)) && return false
    _is_longdiff_call(t) && return !any(a -> a isa ConstantTerm, t.args)
    return any(_tracks_horizon, t.args)
end

_is_longdiff_call(t) = t isa FunctionTerm && nameof(t.f) in (:ldiff, :hbr)

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
    state::Union{Nothing, LPState}  # `nothing` for a linear projection
end

function LocalProjection(models::Vector{M}, horizon, response, shock, base_formula,
        coef_names, tautological_h0) where {M <: OLSMatrixEstimator}
    return LocalProjection{M}(models, horizon, response, shock, base_formula,
        coef_names, tautological_h0, nothing)
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
    lp.state === nothing || println(io, "  State:      ", _describe_state(lp.state))
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
        lp.tautological_h0, lp.state
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
    # An anchor such as lag(y) is `missing` in its first rows
    tofloat(v) = ismissing(v) ? NaN : Float64(v)
    return tofloat.(lead(y, h, default = NaN)) .- tofloat.(z)
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

# Any other term, e.g. `term(1)` or `lags(term(:x), 4)` in a formula built
# programmatically rather than with @formula
_extract_base_variables(t::AbstractTerm) = StatsModels.termvars(t)

# InteractionTerm and other composite terms
function _extract_base_variables(t::StatsModels.InteractionTerm)
    all_vars = Symbol[]
    for component in t.terms
        append!(all_vars, _extract_base_variables(component))
    end
    return unique(all_vars)
end

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
        regimes = nothing)
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
            LPState(state, statelag, labels, base_shock))
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
        response, shock_symbol, formula, lhs_term)
end

"""
    _lp_estimate_horizons(Xof, coef_names_base, df, horizon, response, shock, formula, ...)

Function barrier for type-stable per-horizon OLS estimation.
`Xof(h)` returns the `Matrix{Float64}` design at horizon `h` — the same matrix
every time unless the RHS carries a horizon-tracking `cumul`/`leads` term.
"""
function _lp_estimate_horizons(Xof::F, coef_names_base::Vector{String},
        df_base_complete, horizon, response, shock_symbol, formula, lhs_term,
        state::Union{Nothing, LPState} = nothing) where {F}
    # Helper to estimate one horizon
    function _estimate_horizon(h)
        X = Xof(h)::Matrix{Float64}
        # Identify rows where X is complete (no NaN values)
        X_missing_ind = vec(all(!isnan, X, dims = 2))
        # Convert to Vector{Float64} for type stability (modelcols may return ShiftedArray)
        y_h = Vector{Float64}(StatsModels.modelcols(_lhs_at(lhs_term, h), df_base_complete))
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

    return LocalProjection{eltype(models)}(
        models, horizon, response, shock_symbol, formula, coef_names_base,
        _lhs_tautological(lhs_term) &&
            response === (state === nothing ? shock_symbol : state.shock), state)
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
    state::Union{Nothing, LPState}  # `nothing` for a linear projection
end

function LocalProjectionIV(models::Vector{M}, horizon, response, shock, base_formula,
        coef_names, endogenous_names, instrument_names,
        tautological_h0) where {M <: IVMatrixEstimator}
    return LocalProjectionIV{M}(models, horizon, response, shock, base_formula,
        coef_names, endogenous_names, instrument_names, tautological_h0, nothing)
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
    lpiv.state === nothing || println(io, "  State:        ", _describe_state(lpiv.state))
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
        lpiv.tautological_h0, lpiv.state
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
        coefficients[1] = term in _shock_terms(lp) ? 1.0 : 0.0
    end
    return coefficients
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
    lp_result.state === nothing || throw(ArgumentError(
        "the Herbst–Johannsen correction is derived for a linear projection and " *
        "does not apply to a state-dependent one"))
    _rhs_tracks_horizon(lp_result.base_formula) && throw(ArgumentError(
        "the Herbst–Johannsen correction assumes the same regressors at every " *
        "horizon, but the formula has a horizon-tracking right-hand-side term " *
        "(a bare `cumul(x)` or `leads(x)`), so the design changes with h"))
    _lhs_kind(lp_result.base_formula) === :longdiff && throw(ArgumentError(
        "the Herbst–Johannsen correction is derived for a response in levels, " *
        "y_{t+h} or its cumulative sum; it does not cover the long-difference " *
        "responses ldiff() and hbr()"))
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
    vcov(estimator, bc::BiasCorrectedLP; kwargs...)

Covariance of the *uncorrected* per-horizon OLS coefficients (the
reference convention pairs the corrected path with the ordinary LP
standard errors; see [`BiasCorrectedLP`](@ref)).
"""
function vcov(estimator::CovarianceMatrices.AbstractAsymptoticVarianceEstimator,
        bc::BiasCorrectedLP; kwargs...)
    return vcov(estimator, bc.lp; kwargs...)
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
    vcov(estimator, lp; kwargs...)

Compute diagonal covariance entries horizon-by-horizon using `estimator`
from `CovarianceMatrices.jl`. Works for both `LocalProjection` and `LocalProjectionIV`.

Keyword arguments are forwarded to `CovarianceMatrices.vcov` on each horizon's
model. The one that matters is `dofadjust`: on `lpiv` results kernel HAC and
EWC omit the `n/(n-k)` factor that `lp` applies, and `dofadjust = true` adds
it, so `vcov(Bartlett(6), lpiv_result; dofadjust = true)` is on the same
convention as `vcov(Bartlett(6), lp_result)`. On `lp` results Regress.jl
ignores `dofadjust = false` for HAC/EWC, so the factor cannot be removed there.
"""
function vcov(estimator::CovarianceMatrices.AbstractAsymptoticVarianceEstimator,
        lp::LPResult; kwargs...)
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
        cov = CovarianceMatrices.vcov(estimator, model; kwargs...)
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
    _var_ma(A, H) -> Array{Float64,3}

Reduced-form moving-average coefficients of a VAR, `n × n × (H+1)`: the
recursion ``\\Theta_h = \\sum_{l=1}^{\\min(h,p)} A_l \\Theta_{h-l}`` with
``\\Theta_0 = I``, so that `Θ[:, :, h+1]` is ``[F^h]_{1:n,1:n}`` for the
companion matrix ``F`` (`var_ir.m`, `var_estimation.m`).
"""
function _var_ma(A::Array{Float64, 3}, H::Int)
    n, _, p = size(A)
    Θ = Array{Float64}(undef, n, n, H + 1)
    Θ[:, :, 1] = Matrix{Float64}(I, n, n)
    for h in 1:H
        M = zeros(Float64, n, n)
        for l in 1:min(h, p)
            M += A[:, :, l] * Θ[:, :, h - l + 1]
        end
        Θ[:, :, h + 1] = M
    end
    return Θ
end

"""
    _var_irf(A, ν, H) -> Matrix{Float64}

VAR-implied structural impulse responses, `n × (H+1)`. Port of
`_estim/var_ir.m`: ``\\Theta_h \\nu`` at each horizon, with ``\\Theta_h`` the
moving-average coefficients of `_var_ma`. Equivalent to ``e_r' C^h b`` on the
companion form but cheaper.
"""
function _var_irf(A::Array{Float64, 3}, ν::AbstractVector{Float64}, H::Int)
    Θ = _var_ma(A, H)
    out = Matrix{Float64}(undef, size(A, 1), H + 1)
    out[:, 1] = ν
    for h in 1:H
        out[:, h + 1] = Θ[:, :, h + 1] * ν
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
`:anchor`, `:longdiff` (`ldiff`, `hbr`, or `cumul` of either), or `:unknown`.
Determines how the VAR-implied pseudo-truth must be accumulated across horizons.
"""
function _lhs_kind(formula::FormulaTerm)
    lhs = formula.lhs
    if lhs isa FunctionTerm
        nm = nameof(lhs.f)
        _is_longdiff_call(lhs) && return :longdiff
        nm === :cumul && _is_longdiff_call(first(lhs.args)) && return :longdiff
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
        "$(kind === :anchor ? "an anchored response" :
           kind === :longdiff ? "a long-difference (ldiff/hbr) response" :
           "an unrecognized LHS")"))
    lp_result.state === nothing || throw(ArgumentError(
        "the VAR bootstrap simulates from a linear VAR, which is not the " *
        "data-generating process of a state-dependent local projection"))
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
        catch err
            err isa InterruptException && rethrow()
            failures[b] = 1
        end
        return nothing
    end

    _run_draws!(run_draw, nboot, threaded)

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

function _check_level(level::Real)
    level_f = Float64(level)
    (0 < level_f < 1) ||
        throw(ArgumentError("`level` must be in (0, 1) (got $level_f)"))
    return level_f
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
    level_f = _check_level(level)
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
# Forecast-error variance decomposition (Gorodnichenko & Lee 2020)
# ============================================================================

const _FEVD_ESTIMATORS = (:r2, :lpa, :lpb)

function _fevd_label(estimator::Symbol)
    estimator === :r2 ? "R²" :
    estimator === :lpa ? "LP-A" : "LP-B"
end

"""
    _gl_bandwidth(n) -> Int

Newey–West lag of Gorodnichenko and Lee (2020) for a sample of `n` rows,
``L = \\mathrm{round}(0.75\\, n^{1/3} - 1)`` (`SubR2.m`, `SubLPA.m`), floored
at zero as `lrv_nw.m` does. MATLAB rounds half away from zero, and the cube
root is taken with `^` rather than `cbrt`, which rounds differently at
`n = 1000`. `lrv_nw(⋅, L)` corresponds to CovarianceMatrices' `Bartlett(L + 1)`.
"""
_gl_bandwidth(n::Integer) = max(round(Int, 0.75 * n^(1 / 3) - 1, RoundNearestTiesAway), 0)

"""
    _gl_lrv_nw(Z, L) -> Matrix{Float64}

Bartlett long-run variance of the rows of `Z`, a port of `lrv_nw.m`:
``\\Gamma_0 + \\sum_{i=1}^{L} (1 - i/(L+1)) (\\Gamma_i + \\Gamma_i')`` with
``\\Gamma_i = \\sum_{t=i+1}^{n} z_t z_{t-i}' / n``. No demeaning and no
degrees-of-freedom factor; `L ≤ 0` returns ``\\Gamma_0``.
"""
function _gl_lrv_nw(Z::AbstractMatrix{Float64}, L::Integer)
    n = size(Z, 1)
    V = (Z' * Z) ./ n
    for i in 1:min(L, n - 1)
        Γ = (view(Z, (i + 1):n, :)' * view(Z, 1:(n - i), :)) ./ n
        w = 1 - i / (L + 1)
        V .+= w .* (Γ .+ Γ')
    end
    return V
end

"""
    _gl_prewhitened_lrv(Z, L) -> Matrix{Float64}

VAR(1)-prewhitened long-run variance (`SubR2.m:213-220`,
`SimulationAVarLPA.m:249-256`): ``z_t = A z_{t-1} + u_t`` by least squares
without intercept, then ``(I - A)^{-1}\\, \\mathrm{lrv}(U, L)\\, (I - A')^{-1}``.
`lrv(U, L)` divides by the ``n - 1`` residual rows while `L` comes from `n`, and
``A`` is not capped (no Andrews–Monahan eigenvalue adjustment), as in the
reference. The VAR is solved by QR rather than the normal equations.
"""
function _gl_prewhitened_lrv(Z::AbstractMatrix{Float64}, L::Integer)
    n = size(Z, 1)
    Z0 = Z[1:(n - 1), :]
    Z1 = Z[2:n, :]
    A = Matrix(transpose(Z0 \ Z1))
    U = Z1 - Z0 * transpose(A)
    IA = I - A
    return (IA \ _gl_lrv_nw(U, L)) / transpose(IA)
end

# Covariance of the moment estimator, G⁻¹ Ω G⁻ᵀ / n (the reference's `GOGT`),
# from a factorization of G that the two calls of a horizon (with and without
# prewhitening) share
function _gl_gogt(F::Factorization, Ω::AbstractMatrix, n::Integer)
    X = F \ Ω
    return Matrix(transpose(F \ Matrix(transpose(X)))) ./ n
end
_gl_gogt(G::AbstractMatrix, Ω::AbstractMatrix, n::Integer) = _gl_gogt(lu(G), Ω, n)

# Delta-method standard error; NaN when the quadratic form is negative or NaN
function _gl_delta_se(g::AbstractVector{Float64}, V::AbstractMatrix{Float64})
    v = dot(g, V * g)
    return v >= 0 ? sqrt(v) : NaN
end

"""
    _gl_r2_gradient(θ, s) -> Vector{Float64}

Gradient of the R² share ``\\theta_2'\\theta_1/\\theta_3`` with respect to
``\\theta = (\\theta_1, \\theta_2, \\theta_3)`` (`SubR2.m`,
`SimulationAVarR2.m:180-190`): ``[\\theta_2', \\theta_1', -s] / \\theta_3``.
Exact at ``s = \\hat s``; the simulation method plugs in the bias-corrected share.
"""
function _gl_r2_gradient(θ::AbstractVector{Float64}, s::Float64)
    k = (length(θ) - 1) ÷ 2
    return vcat(θ[(k + 1):(2k)], θ[1:k], -s) ./ θ[end]
end

"""
    _gl_r2(y, W, xf; avar = true) -> NamedTuple

The R² estimator at one horizon, a port of `SubR2.m:157-230`. Rows are the
horizon-``h`` window ``t_0, \\dots, T-h``: `y` holds ``y_{t+h} - y_{t-1}``,
`W` the information set ``W_{t-1}`` (intercept included) and `xf` the demeaned
shocks ``x_t, \\dots, x_{t+h}``. With ``\\hat f`` the residual of `y` on `W`
and ``n`` the number of rows,

```math
\\hat\\Sigma_x = \\frac{X_f'X_f}{n},\\quad \\hat\\theta_2 = \\frac{X_f'\\hat f}{n},\\quad
\\hat\\theta_1 = \\hat\\Sigma_x^{-1}\\hat\\theta_2,\\quad \\hat\\theta_3 = \\frac{\\hat f'\\hat f}{n},\\quad
\\hat s = \\frac{\\hat\\theta_2'\\hat\\theta_1}{\\hat\\theta_3},
```

the uncentred ``R^2`` of ``\\hat f`` on ``X_f``. With `avar = true` the result
also holds the covariance of ``\\hat\\theta = (\\hat\\theta_1, \\hat\\theta_2,
\\hat\\theta_3)`` with and without prewhitening (`avar_pw`, `avar`, i.e.
``G^{-1}\\hat\\Omega G^{-\\prime}/n``) and the delta-method standard errors of
``\\hat s`` (`se_pw`, `se`). ``\\hat f`` is treated as data: the estimation
error of the forecast-error regression is ignored, which is valid when the
shock is unpredictable from ``W_{t-1}``. The prewhitened quantities are `NaN`
when the ``2h+3`` moments cannot be prewhitened on ``n - 1`` rows
(`feasible = false`).
"""
function _gl_r2(y::AbstractVector{Float64}, W::AbstractMatrix{Float64},
        xf::AbstractMatrix{Float64}; avar::Bool = true)
    n = length(y)
    k = size(xf, 2)
    fe = y - W * (W \ y)
    Σx = (xf' * xf) ./ n
    θ2 = (xf' * fe) ./ n
    θ1 = Σx \ θ2
    θ3 = dot(fe, fe) / n
    s = dot(θ2, θ1) / θ3
    θ = vcat(θ1, θ2, θ3)
    avar || return (; s, theta = θ, fe)

    m = 2k + 1
    e = fe - xf * θ1
    exf = e .* xf
    Z = hcat(exf .- mean(exf; dims = 1), fe .* xf .- transpose(θ2), fe .^ 2 .- θ3)
    G = lu(-cat(Σx, Matrix{Float64}(I, k + 1, k + 1); dims = (1, 2)))
    ξ = _gl_r2_gradient(θ, s)
    L = _gl_bandwidth(n)
    V = _gl_gogt(G, _gl_lrv_nw(Z, L), n)
    feasible = m < n - 1
    V_pw = feasible ? _gl_gogt(G, _gl_prewhitened_lrv(Z, L), n) : fill(NaN, m, m)
    return (; s, theta = θ, fe, avar_pw = V_pw, avar = V,
        se_pw = _gl_delta_se(ξ, V_pw), se = _gl_delta_se(ξ, V), bandwidth = L, feasible)
end

"""
    _gl_lp(dhy, C, W, q, x, σ²x, t0, t0lp, H; estimator, dofadjust, irf_se = true)

LP-A / LP-B estimators for all horizons, a port of `SubLPA.m` / `SubLPB.m`
(`semIndicator = 0`). `dhy` is ``T \\times (H+1)`` with column ``h+1`` equal to
``y_{t+h} - y_{t-1}``; `C` the contemporaneous regressors (the demeaned shock
first), `W` the information set ``W_{t-1}``, `x` the demeaned shock and `σ²x` its
full-sample variance. `q` is the regressor matrix of a separate impulse-response
regression (shock first, sample from `t0lp`), or `nothing` to read ``\\hat\\psi_h``
off the forecast-error regression ``[C, W]``.

- ``X^\\perp = C - W\\hat\\Gamma`` from **one** regression on rows ``t_0, \\dots, T``;
- per horizon, on rows ``t_0, \\dots, T-h``: ``\\hat b_h`` from the regression of
  `dhy` on ``[C, W]`` and ``\\hat f_t = \\hat u_t + X^{\\perp\\prime}_t \\hat b_{h,C}``;
- ``D_h = T_h - k_W`` with `dofadjust` (the reference's `dfIndicator = 0`), else ``T_h``;
- LP-A: ``\\hat\\sigma^2_{f,h} = \\sum \\hat f^2 / D_h`` and
  ``\\hat s_h = \\hat\\sigma^2_x \\sum_{i \\le h} \\hat\\psi_i^2 / \\hat\\sigma^2_{f,h}``;
- LP-B: ``\\hat v_t = \\hat f_t - \\sum_{i \\le h}\\hat\\psi_i x_{t+h-i}``,
  ``\\hat\\sigma^2_{v,h} = \\sum \\hat v^2 / D_h``,
  ``m_h = \\hat\\sigma^2_x\\sum_{i\\le h}\\hat\\psi_i^2 + \\hat\\sigma^2_{v,h}`` and
  ``\\hat s_h = \\hat\\sigma^2_x\\sum_{i\\le h}\\hat\\psi_i^2 / m_h``.

`irf_se` is the Newey–West standard error of ``\\hat\\psi_h`` with lag
`_gl_bandwidth` of the impulse-response sample and no
degrees-of-freedom factor. Returns a named tuple with the per-horizon
estimates and the matrices the asymptotic variance needs (`resid_lp`,
`beta_lp`, `fe`, `v`).
"""
function _gl_lp(dhy::Matrix{Float64}, C::Matrix{Float64}, W::Matrix{Float64},
        q::Union{Nothing, Matrix{Float64}}, x::Vector{Float64}, σ²x::Float64,
        t0::Int, t0lp::Int, H::Int; estimator::Symbol, dofadjust::Bool,
        irf_se::Bool = true)
    T = size(dhy, 1)
    nc = size(C, 2)
    kW = size(W, 2)
    R = hcat(C, W)
    shared = q === nothing
    Q = shared ? R : q
    Nq = size(Q, 2)
    lpb = estimator === :lpb

    # X⊥: one regression on rows t0..T, not t0..T-h (SubLPA.m:311-315)
    r0 = t0:T
    Cperp = fill(NaN, T, nc)
    Cperp[r0, :] = C[r0, :] - W[r0, :] * (W[r0, :] \ C[r0, :])

    psi = zeros(H + 1)
    se_psi = irf_se ? zeros(H + 1) : Float64[]
    mse = zeros(H + 1)
    sigma2v = lpb ? zeros(H + 1) : Float64[]
    fe = fill(NaN, T, H + 1)
    v = lpb ? fill(NaN, T, H + 1) : Matrix{Float64}(undef, 0, 0)
    resid_lp = fill(NaN, T, H + 1)
    beta_lp = zeros(Nq, H + 1)
    perfect = falses(H + 1)

    for h in 0:H
        rows = t0:(T - h)
        Th = length(rows)
        y = dhy[rows, h + 1]
        Rh = R[rows, :]
        b = Rh \ y
        u = y - Rh * b
        if shared
            rlp, ylp, Qh, blp, ulp = rows, y, Rh, b, u
        else
            rlp = t0lp:(T - h)
            ylp = dhy[rlp, h + 1]
            Qh = Q[rlp, :]
            blp = Qh \ ylp
            ulp = ylp - Qh * blp
        end
        psi[h + 1] = blp[1]
        beta_lp[:, h + 1] = blp
        resid_lp[rlp, h + 1] = ulp
        perfect[h + 1] = sum(abs2, ulp) <= 1e-20 * sum(abs2, ylp)

        f = u + Cperp[rows, :] * b[1:nc]
        fe[rows, h + 1] = f
        D = Th - (dofadjust ? kW : 0)
        if lpb
            vh = copy(f)
            @inbounds for i in 0:h, (j, t) in enumerate(rows)

                vh[j] -= psi[i + 1] * x[t + h - i]
            end
            v[rows, h + 1] = vh
            sigma2v[h + 1] = sum(abs2, vh) / D
            mse[h + 1] = σ²x * sum(abs2, view(psi, 1:(h + 1))) + sigma2v[h + 1]
        else
            mse[h + 1] = sum(abs2, f) / D
        end

        if irf_se
            nlp = length(rlp)
            SX = (Qh' * Qh) ./ nlp
            Ω = _gl_lrv_nw(Qh .* ulp, _gl_bandwidth(nlp))
            Vψ = ((SX \ Ω) / SX) ./ nlp
            se_psi[h + 1] = sqrt(max(Vψ[1, 1], 0.0))
        end
    end
    share = σ²x .* cumsum(psi .^ 2) ./ mse
    return (; share, psi, irf_se = se_psi, mse, sigma2v, fe, v, resid_lp, beta_lp,
        Q, perfect)
end

"""
    _gl_lp_gradient(estimator, ψ, σ²x, σ²h, m, s, Nq) -> Vector{Float64}

Gradient of the LP-A / LP-B share with respect to
``\\theta_h = (\\mathrm{vec}(b_0, \\dots, b_h), \\sigma^2_x, \\sigma^2_h)``
(`SimulationAVarLP*.m:319-323`), with ``\\psi_j`` the first of the `Nq`
coefficients of ``b_j``:

- LP-A: ``\\frac{1}{\\sigma^2_{f}}\\,[2\\sigma^2_x\\psi' \\otimes e_1',\\ \\sum\\psi^2,\\ -s]``;
- LP-B: ``\\frac{1-s}{m}\\,[2\\sigma^2_x\\psi' \\otimes e_1',\\ \\sum\\psi^2,\\ -\\frac{s}{1-s}]``,

where `m` is the LP-B denominator (pass `m = σ²h` for LP-A). Both are exact at
``s = \\hat s``; the reference's simulation method plugs in the bias-corrected
share instead. The LP-B last entry is computed as ``-s/m``, which is the same
number without dividing by ``1 - s``.
"""
function _gl_lp_gradient(estimator::Symbol, ψ::AbstractVector{Float64}, σ²x::Float64,
        σ²h::Float64, m::Float64, s::Float64, Nq::Int)
    np = length(ψ)
    g = zeros(Nq * np + 2)
    c = estimator === :lpa ? 1 / σ²h : (1 - s) / m
    for j in 1:np
        g[1 + (j - 1) * Nq] = c * 2 * ψ[j] * σ²x
    end
    g[end - 1] = c * sum(abs2, ψ)
    g[end] = estimator === :lpa ? -s / σ²h : -s / m
    return g
end

"""
    _gl_lp_avar(Q, resid_lp, err, x, σ²x, σ²h, beta_lp, Σq, t0, t0lp, h) -> NamedTuple

Horizon-``h`` asymptotic covariance of
``\\hat\\theta_h = (\\mathrm{vec}(\\hat b_0, \\dots, \\hat b_h), \\hat\\sigma^2_x, \\hat\\sigma^2_h)``
for LP-A / LP-B (`SimulationAVarLP*.m:212-262`, `jointIndicator = 0`). On rows
``\\max(t_0, t_0^{LP}), \\dots, T-h`` the moments are
``[q_t\\hat u_{t,0}, \\dots, q_t\\hat u_{t,h},\\ x_t^2 - \\hat\\sigma^2_x,\\ e_t^2 - \\hat\\sigma^2_h]``
with `err` ``= \\hat f`` (LP-A) or ``\\hat v`` (LP-B), demeaned column by column;
``G = -\\mathrm{blkdiag}(I_{h+1} \\otimes \\hat\\Sigma_q, I_2)`` with `Σq` the
second-moment matrix of ``q_t`` over ``t_0^{LP}, \\dots, T``.
"""
function _gl_lp_avar(Q::Matrix{Float64}, resid_lp::Matrix{Float64},
        err::AbstractVector{Float64}, x::Vector{Float64}, σ²x::Float64, σ²h::Float64,
        beta_lp::Matrix{Float64}, Σq::Matrix{Float64}, t0::Int, t0lp::Int, h::Int)
    T, Nq = size(Q)
    rows = max(t0, t0lp):(T - h)
    n = length(rows)
    m = Nq * (h + 1) + 2
    Z = Matrix{Float64}(undef, n, m)
    for j in 0:h
        Z[:, (j * Nq + 1):((j + 1) * Nq)] = Q[rows, :] .* view(resid_lp, rows, j + 1)
    end
    Z[:, m - 1] = view(x, rows) .^ 2 .- σ²x
    Z[:, m] = view(err, rows) .^ 2 .- σ²h
    Z .-= mean(Z; dims = 1)
    G = lu(-cat(kron(Matrix{Float64}(I, h + 1, h + 1), Σq), Matrix{Float64}(I, 2, 2);
        dims = (1, 2)))
    L = _gl_bandwidth(n)
    V = _gl_gogt(G, _gl_lrv_nw(Z, L), n)
    feasible = m < n - 1
    V_pw = feasible ? _gl_gogt(G, _gl_prewhitened_lrv(Z, L), n) : fill(NaN, m, m)
    θ = vcat(vec(beta_lp[:, 1:(h + 1)]), σ²x, σ²h)
    return (; theta = θ, avar_pw = V_pw, avar = V, bandwidth = L, nobs = n, feasible)
end

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

# ============================================================================
# FEVD bias corrections (Gorodnichenko & Lee 2020)
# ============================================================================

const _FEVD_METHODS = (:var, :block, :simulation)

"""
    _var_fevd(A, Ω, H; shock, response, cumulate = false) -> Vector{Float64}

Share of the forecast-error variance of variable `response` due to the Cholesky
shock of variable `shock` in a VAR with slopes `A` (`n × n × p`) and residual
covariance `Ω`, horizons ``0, \\dots, H`` (`var_chol.m:54-81`). With
``\\Psi_h = \\Theta_h L`` the one-standard-deviation responses
(``\\Theta_h`` from `_var_ma`, ``L`` the lower Cholesky factor of ``\\Omega``),

```math
s_h = \\frac{\\sum_{l \\le h} \\Psi_l[r, j]^2}{\\sum_{l \\le h} \\sum_m \\Psi_l[r, m]^2}.
```

With `cumulate = true` the response row is first cumulated over horizons, for
all shocks (`var_chol.m:67`): the share then refers to ``y_{t+h} - y_{t-1}``
when the VAR holds the first difference of ``y``. The share is invariant to a
scalar rescaling of ``\\Omega`` but depends on the Cholesky order.
"""
function _var_fevd(A::Array{Float64, 3}, Ω::AbstractMatrix{Float64}, H::Int;
        shock::Int, response::Int, cumulate::Bool = false)
    C = cholesky(Symmetric(Matrix{Float64}(Ω)); check = false)
    issuccess(C) || throw(ArgumentError(
        "the VAR residual covariance is not positive definite: the VAR variables " *
        "are linearly dependent (e.g. a level together with its first difference)"))
    L = C.L
    Θ = _var_ma(A, H)
    n = size(A, 1)
    ψ = zeros(n)
    num = 0.0
    den = 0.0
    s = Vector{Float64}(undef, H + 1)
    for h in 0:H
        row = vec(transpose(view(Θ, response, :, h + 1)) * L)
        ψ = cumulate ? ψ .+ row : row
        num += ψ[shock]^2
        den += sum(abs2, ψ)
        s[h + 1] = num / den
    end
    return s
end

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

# A VAR residual covariance that is singular up to rounding (linearly dependent
# variables) gives meaningless shares even when its Cholesky factorization
# happens to succeed: test the smallest eigenvalue of the correlation matrix.
function _check_var_rank(Ω::AbstractMatrix{Float64}, varlist = nothing)
    d = [Ω[i, i] for i in axes(Ω, 1)]
    singular = !all(>(0), d) ||
               eigmin(Symmetric(Ω ./ sqrt.(d .* transpose(d)))) <= sqrt(eps()) * size(Ω, 1)
    singular && throw(ArgumentError(
        "the VAR residual covariance is singular: the VAR variables" *
        (varlist === nothing ? "" : " $(varlist)") * " are linearly dependent (e.g. " *
        "a level together with its first difference)"))
    return nothing
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

"""
    FEVDSummary

Pointwise confidence band for a Gorodnichenko–Lee forecast-error variance
share, built by [`summarize`](@ref) from an [`LPFEVD`](@ref) or a
[`FEVDBootstrap`](@ref). Displays with PrettyTables; `DataFrame(s)` gives the
columns `horizon, share, estimate, se, lower, upper`.

# Fields
- `response`, `shock`, `estimator` (`:r2`, `:lpa`, `:lpb`);
- `method`: `:asymptotic` for an `LPFEVD`, else the bias-correction procedure
  of the bootstrap (`:var`, `:block`, `:simulation`);
- `band`: `:normal` (``\\text{centre} \\pm z\\cdot\\text{s.e.}``) or `:shift`
  (bootstrap quantiles translated to the centre);
- `level`, `biascorrected` (whether the band is centred on the bias-corrected
  share), `prewhiten` (whether a `:normal` band uses the prewhitened standard
  error; `nothing` for `:shift`);
- `horizon`, `share` (the estimate ``\\hat s_h``), `estimate` (the centre of
  the band: the bias-corrected share or ``\\hat s_h``), `se` (the standard
  error of the band: the delta-method one for `:normal`, the standard
  deviation of the draws for `:shift`), `lower`, `upper`.
"""
struct FEVDSummary
    response::Symbol
    shock::Symbol
    estimator::Symbol
    method::Symbol
    band::Symbol
    level::Float64
    biascorrected::Bool
    prewhiten::Union{Nothing, Bool}
    horizon::Vector{Int}
    share::Vector{Float64}
    estimate::Vector{Float64}
    se::Vector{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
end

function DataFrames.DataFrame(s::FEVDSummary)
    DataFrame(horizon = s.horizon, share = s.share, estimate = s.estimate, se = s.se,
        lower = s.lower, upper = s.upper)
end

function Base.show(io::IO, s::FEVDSummary)
    print(io, "FEVDSummary(estimator=:", s.estimator, ", method=:", s.method,
        ", band=:", s.band, ", level=", s.level, ")")
end

function Base.show(io::IO, ::MIME"text/plain", s::FEVDSummary)
    what = s.method === :asymptotic ? "delta method" : _FEVD_METHOD_LABELS[s.method]
    println(
        io, "FEVDSummary (Gorodnichenko–Lee ", _fevd_label(s.estimator), ", ", what, ")")
    println(io, "  Response:   ", s.response)
    println(io, "  Shock:      ", s.shock)
    se_label = s.band === :shift ? "s.e. (draws)" :
               s.prewhiten ? "s.e. (PW)" : "s.e."
    print(io, "  Band:       ",
        s.band === :shift ? "bootstrap quantiles shifted to the centre" :
        "normal, centre ± z·$(se_label)",
        s.method === :asymptotic ? "" :
        s.biascorrected ? ", bias-corrected" : ", uncorrected")
    pct = round(s.level * 100; digits = 1)
    pct_s = isinteger(pct) ? string(Int(pct)) : string(pct)
    if s.biascorrected
        data = (; h = s.horizon, s = s.share, bc = s.estimate, se = s.se, lo = s.lower,
            hi = s.upper)
        labels = ["Horizon", "Share", "Bias-corr.", se_label, "Lower $pct_s%",
            "Upper $pct_s%"]
    else
        data = (; h = s.horizon, s = s.share, se = s.se, lo = s.lower, hi = s.upper)
        labels = ["Horizon", "Share", se_label, "Lower $pct_s%", "Upper $pct_s%"]
    end
    println(io)
    nc = length(labels)
    pretty_table(io, data;
        column_labels = labels,
        title = "Forecast-error variance share of $(s.shock)",
        formatters = [fmt__round(4, collect(2:nc))],
        alignment = fill(:r, nc),
        table_format = TextTableFormat(borders = text_table_borders__unicode_rounded))
    return nothing
end

# MATLAB's `quantile`: Hyndman–Fan type 5, plotting positions (i - 0.5)/n with
# linear interpolation, clamped at the extremes (spec Q37). `_boot_interval`
# uses type 7.
_gl_quantile5(v::AbstractVector{<:Real}, p::Real) = quantile(v, p; alpha = 0.5, beta = 0.5)

"""
    summarize(f::LPFEVD; level=0.90, prewhiten=true) -> FEVDSummary

Normal confidence band ``\\hat s_h \\pm z_{(1+\\text{level})/2}\\,
\\widehat{\\mathrm{se}}_h`` for a Gorodnichenko–Lee FEVD, with the delta-method
standard error at ``\\hat s_h`` (`se_pw` with `prewhiten = true`, else `se`).
Deterministic; the band is not clamped to ``[0, 1]`` and is `NaN` at horizons
whose standard error is `NaN`. Bias-corrected bands come from
[`fevdbootstrap`](@ref).
"""
function summarize(f::LPFEVD; level::Real = 0.90, prewhiten::Bool = true)
    level_f = _check_level(level)
    se = copy(prewhiten ? f.se_pw : f.se)
    z = quantile(Normal(), 0.5 + level_f / 2)
    return FEVDSummary(f.response, f.shock, f.estimator, :asymptotic, :normal, level_f,
        false, prewhiten, collect(0:(f.horizon)), copy(f.share), copy(f.share), se,
        f.share .- z .* se, f.share .+ z .* se)
end

"""
    summarize(b::FEVDBootstrap; level=0.90, band=:shift, biascorrected=true,
              prewhiten=true) -> FEVDSummary

Pointwise confidence band for a bias-corrected Gorodnichenko–Lee FEVD. With
``c_h`` the centre — the bias-corrected share ``\\tilde s_h`` (`b.biascorrected`)
when `biascorrected = true`, the estimate ``\\hat s_h`` otherwise — and
``q \\in \\{\\alpha/2, 1-\\alpha/2\\}``, ``\\alpha = 1 - `` `level`:

- `band = :shift` (the paper's band, `cbIndicator = 0`):
  ``Q_q(s^*_{\\cdot,h}) + c_h - \\bar m_h``, the distribution of the draws
  translated so that its mean is the centre. ``Q_q`` is MATLAB's `quantile`
  (Hyndman–Fan type 5). With `biascorrected = false` this is the uncorrected band
  ``Q_q(s^*) + \\hat s_h - \\bar m_h`` drawn in the paper's figures
  (`AppSimulation.m:325-329`). The reported `se` is the standard deviation of the
  draws.
- `band = :normal` (`cbIndicator = 1` / `2`): ``c_h + \\Phi^{-1}(q)\\,
  \\mathrm{se}_h`` with `b.se_pw` (`prewhiten = true`) or `b.se`. These are the
  delta-method standard errors at ``\\hat s_h`` for `:var` and `:block` and at
  ``\\tilde s_h`` for `:simulation` (see [`FEVDBootstrap`](@ref)); with
  `biascorrected = false` the same band is translated to ``\\hat s_h``.

The keyword is `band` rather than `method` (the name `summarize(::LPBootstrap)`
uses for its interval construction) because `method` already names the
bias-correction procedure of [`fevdbootstrap`](@ref) and of `b.method`. Bands
are pointwise and never clamped to ``[0, 1]``; they are `NaN` at horizons
without draws (`:simulation` horizons whose covariance is `NaN`).
"""
function summarize(b::FEVDBootstrap; level::Real = 0.90, band::Symbol = :shift,
        biascorrected::Bool = true, prewhiten::Bool = true)
    band in (:shift, :normal) || throw(ArgumentError(
        "`band` must be :shift or :normal (got :$band)"))
    level_f = _check_level(level)
    α = 1 - level_f
    f = b.fevd
    H = f.horizon
    centre = copy(biascorrected ? b.biascorrected : f.share)
    lower = fill(NaN, H + 1)
    upper = fill(NaN, H + 1)
    if band === :shift
        se = copy(b.se_boot)
        for h in 1:(H + 1)
            col = view(b.draws, :, h)
            (!isempty(col) && all(isfinite, col)) || continue
            shift = centre[h] - b.mean_draws[h]
            lower[h] = _gl_quantile5(col, α / 2) + shift
            upper[h] = _gl_quantile5(col, 1 - α / 2) + shift
        end
        pw = nothing
    else
        se = copy(prewhiten ? b.se_pw : b.se)
        z = quantile(Normal(), 1 - α / 2)
        lower .= centre .- z .* se
        upper .= centre .+ z .* se
        pw = prewhiten
    end
    return FEVDSummary(f.response, f.shock, f.estimator, b.method, band, level_f,
        biascorrected, pw, collect(0:H), copy(f.share), centre, se, lower, upper)
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

"""
    FEVDPlot

Internal wrapper for dispatching the plot recipe on [`LPFEVD`](@ref) and
[`FEVDBootstrap`](@ref). Users should call `plot(f; levels, prewhiten)` or
`plot(b; levels, band, biascorrected, prewhiten)` directly.
"""
struct FEVDPlot{F <: Union{LPFEVD, FEVDBootstrap}}
    fevd::F
    levels::Vector{Float64}
    band::Symbol
    biascorrected::Bool
    prewhiten::Bool
end

function _fevd_plot_summary(f::LPFEVD, w::FEVDPlot, level)
    summarize(f; level,
        prewhiten = w.prewhiten)
end
function _fevd_plot_summary(b::FEVDBootstrap, w::FEVDPlot, level)
    summarize(b; level,
        band = w.band, biascorrected = w.biascorrected, prewhiten = w.prewhiten)
end

@recipe function f(wrapper::FEVDPlot)
    obj = wrapper.fevd
    sorted_levels = sort(wrapper.levels; rev = true)
    for level in sorted_levels
        (level <= 0 || level >= 1) && throw(ArgumentError("levels must be in (0, 1)"))
    end
    bands = [_fevd_plot_summary(obj, wrapper, lv) for lv in sorted_levels]
    centre = bands[1].estimate
    horizons = bands[1].horizon
    corrected = bands[1].biascorrected

    # Shares live in [0, 1], but neither the bias-corrected share nor the bands
    # are clamped (and LP-A is unbounded), so the default range covers both.
    finite = filter(isfinite, vcat(centre, bands[1].lower, bands[1].upper))
    lo = isempty(finite) ? 0.0 : min(0.0, minimum(finite))
    hi = isempty(finite) ? 1.0 : max(1.0, maximum(finite))

    xlabel --> "Horizon"
    ylabel --> "FEVD share of $(obj.shock)"
    ylims --> (lo, hi)
    linewidth --> 2
    fillalpha --> 0.3
    legend --> :best

    # Bands are asymmetric for the shifted bootstrap quantiles, so the ribbon
    # takes a (below, above) pair.
    if length(sorted_levels) > 1
        for (idx, s) in enumerate(bands[2:end])
            @series begin
                ribbon := (centre .- s.lower, s.upper .- centre)
                fillalpha := 0.3 + 0.15 * idx
                label := ""
                linewidth := 0
                linecolor := :transparent
                horizons, centre
            end
        end
    end

    # The uncorrected estimate next to a bias-corrected centre
    if corrected
        @series begin
            ribbon := nothing
            fillrange := nothing
            label := "Estimate"
            linestyle := :dash
            linewidth := 1.5
            horizons, bands[1].share
        end
    end

    widest = bands[1]
    label --> (corrected ? "Bias-corrected" : "FEVD")
    ribbon --> (centre .- widest.lower, widest.upper .- centre)
    return horizons, centre
end

@recipe function f(fv::LPFEVD; levels = [0.90], prewhiten = true)
    FEVDPlot(fv, Float64.(levels), :normal, false, prewhiten)
end

@recipe function f(b::FEVDBootstrap; levels = [0.90], band = :shift,
        biascorrected = true, prewhiten = true)
    FEVDPlot(b, Float64.(levels), band, biascorrected, prewhiten)
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

# Shared body of the two FEVD conversions: `summaries` are FEVDSummary bands,
# one per coverage level, with a common centre and standard error.
function _fevd_irf_result(summaries::Vector{FEVDSummary}, coverage::Vector{Float64},
        meta::NamedTuple)
    T = Float64
    s1 = summaries[1]
    H = length(s1.horizon)
    _ax(v) = AxisArray(
        reshape(collect(T, v), 1, 1, H),
        Axis{:response}([s1.response]),
        Axis{:shock}([s1.shock]),
        Axis{:horizon}(0:(H - 1))
    )
    data = _ax(s1.estimate)
    lower = [_ax(s.lower) for s in summaries]
    upper = [_ax(s.upper) for s in summaries]
    return LocalProjectionIRFResult{T, typeof(data)}(
        data, _ax(s1.se), lower, upper, coverage, meta
    )
end

"""
    as_irf_result(f::LPFEVD; coverage=[0.68, 0.90], prewhiten=true)

Convert a Gorodnichenko–Lee FEVD ([`LPFEVD`](@ref)) to a
`LocalProjectionIRFResult`. The `data` array holds the variance **shares**
``\\hat s_h`` (not impulse responses), on the `response × shock × horizon`
axes; `stderr` the delta-method standard errors (`se_pw` with `prewhiten =
true`, else `se`); the bands are the normal ones of [`summarize`](@ref). The
metadata records `fevd = true` and `quantity = :fevd_share`, with the
estimator and the sample.
"""
function as_irf_result(f::LPFEVD;
        coverage::Vector{Float64} = [0.68, 0.90],
        prewhiten::Bool = true)
    summaries = [summarize(f; level = c, prewhiten = prewhiten) for c in coverage]
    meta = (horizon = f.horizon,
        formula = f.formula,
        irf_formula = f.irf_formula,
        term = f.shock,
        fevd = true,
        quantity = :fevd_share,
        estimator = f.estimator,
        prewhiten = prewhiten,
        T = f.T,
        t0 = f.t0,
        nobs = f.nobs,
        bias_corrected = false)
    return _fevd_irf_result(summaries, coverage, meta)
end

"""
    as_irf_result(b::FEVDBootstrap; coverage=[0.68, 0.90], band=:shift,
                  biascorrected=true, prewhiten=true)

Convert a bias-corrected Gorodnichenko–Lee FEVD ([`FEVDBootstrap`](@ref)) to a
`LocalProjectionIRFResult`. `data` holds the centre of the bands of
[`summarize`](@ref) under the same keywords — the bias-corrected shares by
default — and `stderr` their standard error (the standard deviation of the
draws for `band = :shift`). The metadata records `fevd = true`,
`quantity = :fevd_share`, `bootstrap = true`, the procedure
(`bootstrap_method`), the band, the uncorrected shares (`share`), the VAR
pseudo-truth for `:var`, and the draw counts.
"""
function as_irf_result(b::FEVDBootstrap;
        coverage::Vector{Float64} = [0.68, 0.90],
        band::Symbol = :shift,
        biascorrected::Bool = true,
        prewhiten::Bool = true)
    summaries = [summarize(b; level = c, band, biascorrected, prewhiten)
                 for c in coverage]
    f = b.fevd
    meta = (horizon = f.horizon,
        formula = f.formula,
        irf_formula = f.irf_formula,
        term = f.shock,
        fevd = true,
        quantity = :fevd_share,
        estimator = f.estimator,
        bootstrap = true,
        bootstrap_method = b.method,
        band = band,
        bias_corrected = biascorrected,
        share = copy(f.share),
        pseudo_truth = b.pseudo_truth,
        nboot = b.nboot,
        nfail = b.nfail,
        var_lags = b.nlags,
        var_variables = b.vars)
    return _fevd_irf_result(summaries, coverage, meta)
end

end # module LocalProjections
