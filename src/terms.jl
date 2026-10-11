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
