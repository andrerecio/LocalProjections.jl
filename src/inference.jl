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
