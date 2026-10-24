# ============================================================================
# The response at horizon h
# `lp`/`lpiv` rebuild the response at every horizon. Which response depends on
# the left-hand-side term alone, so the per-horizon loop asks `_lhs_at(lhs, h)`
# and each term type answers.
# ============================================================================

const _LHS_HELP = "the response must use leads(), cumul(), anchor(), or the y|z syntax, " *
    "e.g. `leads(y) ~ x`; long differences use ldiff(y) or hbr(x, y), alone or " *
    "inside cumul()"

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
    return _extract_single_response(t.term, "$(_longdiff_name(t))() response")
end
function _lhs_response(t::AnchorTerm)
    return _extract_single_response(t.response, "anchor() response term")
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

"""
    _lhs_kind(formula) -> Symbol

Which response transform the local-projection formula uses: `:leads`, `:cumul`,
`:anchor`, `:longdiff` (`ldiff`, `hbr`, or `cumul` of either), or `:unknown`.
Determines how the VAR-implied pseudo-truth must be accumulated across horizons,
and which responses the bias correction and the VAR bootstrap cover.
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

# ============================================================================
# Horizon-tracking terms on the right-hand side
# A bare cumul(x), leads(x), ldiff(x) or hbr(x, y) among the regressors (exogenous,
# endogenous or instrument) is rebuilt at every projection horizon; an explicit
# horizon such as cumul(x, 3) pins the column.
# ============================================================================

"""
    _has_dynamic_horizon(term) -> Bool

`true` when `term` contains a `CumulTerm`, `LeadTerm` or `LongDiffTerm`
(`ldiff`/`hbr`) whose horizon is `nothing`: a right-hand-side term that must be
rebuilt at every projection horizon rather than once. `cumul(x, 3)` and
`leads(x, 3)` pin an explicit horizon and are therefore *not* dynamic.
"""
_has_dynamic_horizon(::Any) = false
_has_dynamic_horizon(t::CumulTerm) = t.horizon === nothing || _has_dynamic_horizon(t.term)
_has_dynamic_horizon(t::LeadTerm) = t.horizon === nothing || _has_dynamic_horizon(t.term)
function _has_dynamic_horizon(t::AnchorTerm)
    return _has_dynamic_horizon(t.response) || _has_dynamic_horizon(t.anchor)
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

Return `term` with every horizon-tracking `CumulTerm`/`LeadTerm`/`LongDiffTerm`
(horizon `nothing`) pinned to horizon `h`. Terms carrying an explicit horizon, and
every other term, are returned unchanged.

Coefficient names are taken from the *unspecialized* terms, so `cumul(g)` names one
coefficient across all horizons rather than `cumul(g, 0)`, `cumul(g, 1)`, ...:
`coefpath`, `summarize` and the plots all assume horizon-invariant coefficient names.
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
    _rhs_tracks_horizon(formula::FormulaTerm) -> Bool

`true` when the *unapplied* right-hand side of `formula` contains a bare
`cumul(x)`, `leads(x)`, `ldiff(x)` or `hbr(x, y)` (a regressor that is rebuilt at
every projection horizon) anywhere, including inside an IV block
`(endo ~ instruments)` or an interaction. Explicit horizons (`cumul(x, 3)`) do not
count.

This is the raw-formula counterpart of `_has_dynamic_horizon`, for procedures that
only hold the fitted object and its `base_formula`.
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
