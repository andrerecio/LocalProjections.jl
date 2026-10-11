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
