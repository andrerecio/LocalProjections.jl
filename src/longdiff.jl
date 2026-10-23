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
lpiv(TSLS(), @formula(cumul(hbr(gdp)) ~ (cumul(hbr(gov, gdp)) ~ news) + lags(news, 4) +
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
    return unique(reduce(vcat, (StatsModels.termvars(a) for a in t.args); init = Symbol[]))
end
function StatsModels.termvars(t::LongDiffTerm)
    t.scale === nothing && return StatsModels.termvars(t.term)
    return unique(vcat(StatsModels.termvars(t.term), StatsModels.termvars(t.scale)))
end
StatsModels.termvars(t::DiffTerm) = StatsModels.termvars(t.term)

StatsModels.terms(t::LongDiffTerm) = t.scale === nothing ? (t.term,) : (t.term, t.scale)
StatsModels.terms(t::DiffTerm) = (t.term,)

function StatsModels.apply_schema(
        t::FunctionTerm{typeof(ldiff)}, sch::StatsModels.Schema, ctx::Type
    )
    term, horizon = _parse_unary_binary_args(t, "ldiff", nothing)
    return LongDiffTerm(StatsModels.apply_schema(term, sch, ctx), nothing, horizon)
end

function StatsModels.apply_schema(
        t::FunctionTerm{typeof(hbr)}, sch::StatsModels.Schema, ctx::Type
    )
    args = collect(t.args)
    horizon = nothing
    if length(args) >= 2 && last(args) isa ConstantTerm
        horizon = pop!(args).n
    end
    (1 <= length(args) <= 2 && !any(a -> a isa ConstantTerm, args)) || throw(
        ArgumentError(
            "hbr() takes a variable, an optional scale variable " *
                "and an optional horizon, e.g. hbr(g, y) or hbr(g, y, 3)"
        )
    )
    term = StatsModels.apply_schema(args[1], sch, ctx)
    scale = length(args) == 2 ? StatsModels.apply_schema(args[2], sch, ctx) : term
    return LongDiffTerm(term, scale, horizon)
end

function StatsModels.apply_schema(t::LongDiffTerm, sch::StatsModels.Schema, ctx::Type)
    scale = t.scale === nothing ? nothing : StatsModels.apply_schema(t.scale, sch, ctx)
    return LongDiffTerm(StatsModels.apply_schema(t.term, sch, ctx), scale, t.horizon)
end

function StatsModels.apply_schema(
        t::FunctionTerm{typeof(firstdiff)}, sch::StatsModels.Schema, ctx::Type
    )
    length(t.args) == 1 || throw(ArgumentError("firstdiff() takes one variable"))
    return DiffTerm(StatsModels.apply_schema(only(t.args), sch, ctx))
end

function StatsModels.apply_schema(t::DiffTerm, sch::StatsModels.Schema, ctx::Type)
    return DiffTerm(StatsModels.apply_schema(t.term, sch, ctx))
end

# cumul(ldiff(y)) sums the changes at horizons 0, ..., h, so the horizon belongs to
# cumul: cumul(ldiff(y), 3), not cumul(ldiff(y, 3)).
_check_cumul_inner(::AbstractTerm) = nothing
function _check_cumul_inner(t::LongDiffTerm)
    t.horizon === nothing && return nothing
    throw(
        ArgumentError(
            "inside cumul(), write $(_longdiff_name(t))($(t.term)) without a horizon; " *
                "pin the horizon on cumul instead, e.g. cumul($(_longdiff_name(t))(...), 3)"
        )
    )
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
function _create_long_difference(
        x::Vector{Float64}, s::Union{Nothing, Vector{Float64}}, h::Int, cumulative::Bool
    )
    n = length(x)
    ahead = cumulative ? _create_cumulative(x, h) : collect(lead(x, h, default = NaN))
    k = cumulative ? h + 1 : 1
    out = fill(NaN, n)
    @inbounds for t in 2:n
        v = ahead[t] - k * x[t - 1]
        out[t] = s === nothing ? v : v / s[t - 1]
    end
    any(isinf, out) && throw(
        ArgumentError(
            "hbr(): the scale variable is zero at a period t - 1 used by the projection"
        )
    )
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
