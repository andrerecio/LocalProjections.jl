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
    sample::Symbol              # :horizon (per-horizon samples) or :common
end

function LocalProjection(models::Vector{M}, horizon, response, shock, base_formula,
        coef_names, tautological_h0) where {M <: OLSMatrixEstimator}
    return LocalProjection{M}(models, horizon, response, shock, base_formula,
        coef_names, tautological_h0, nothing, :horizon)
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
    println(io, "  Sample:     ", _describe_sample(lp))
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
        lp.tautological_h0, lp.state, lp.sample
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
    sample::Symbol              # :horizon (per-horizon samples) or :common
end

function LocalProjectionIV(models::Vector{M}, horizon, response, shock, base_formula,
        coef_names, endogenous_names, instrument_names,
        tautological_h0) where {M <: IVMatrixEstimator}
    return LocalProjectionIV{M}(models, horizon, response, shock, base_formula,
        coef_names, endogenous_names, instrument_names, tautological_h0, nothing,
        :horizon)
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

function _describe_sample(r::LPResult)
    n = [Int(nobs(m)) for m in r.models]
    r.sample === :common && return "common, T = $(n[1]) at every horizon"
    return "per horizon, T = $(n[1]) at h = 0 to $(n[end]) at h = $(r.horizon)"
end

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
    println(io, "  Sample:       ", _describe_sample(lpiv))
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
        lpiv.tautological_h0, lpiv.state, lpiv.sample
    )
end

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
