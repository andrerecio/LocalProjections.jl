module LocalProjectionsMakieExt

using Makie
using LocalProjections
using LocalProjections: LPResult, LocalProjectionCovariance,
                        coefpath, stderror, vcov
using CovarianceMatrices
using Distributions: Normal, quantile

import LocalProjections: irfplot, irfplot!, irfplot_axis

# ============================================================================
# IRFPlotMakie Recipe
# ============================================================================

Makie.@recipe(IRFPlotMakie, lp, estimator_or_cov) do scene
    Makie.Theme(;
        term = Makie.automatic,
        levels = [0.95],
        irf_scale = 1.0,
        drawzero = true,
        zerolinecolor = :gray70,
        bandcolor = :blue,
        bandalpha = 0.25,
        linecolor = :black,
        linewidth = 2.0
    )
end

function Makie.plot!(plot::IRFPlotMakie)
    lp = plot[1][]
    est = plot[2][]

    term_val = plot.term[]
    term = term_val === Makie.automatic ? lp.shock : term_val
    levels = plot.levels[]
    irf_scale = plot.irf_scale[]
    drawzero = plot.drawzero[]
    zerolinecolor = plot.zerolinecolor[]
    bandcolor = plot.bandcolor[]
    bandalpha = plot.bandalpha[]
    linecolor = plot.linecolor[]
    linewidth = plot.linewidth[]

    # Get covariance (convert estimator if needed)
    cov = if est isa LocalProjectionCovariance
        est
    else
        vcov(est, lp)
    end

    beta = coefpath(lp; term = term) .* irf_scale
    se = stderror(cov; term = term) .* irf_scale
    horizons = collect(0:lp.horizon)

    sorted_levels = sort(Float64.(levels); rev = true)
    for level in sorted_levels
        (level <= 0 || level >= 1) &&
            throw(ArgumentError("levels must be in (0, 1), got $level"))
    end

    # Draw confidence bands (widest first, fading alpha)
    for (idx, level) in enumerate(sorted_levels)
        z = quantile(Normal(), 0.5 + level / 2)
        lower = beta .- z .* se
        upper = beta .+ z .* se
        alpha = clamp(bandalpha * 0.8^(idx - 1), 0.0f0, 1.0f0)
        Makie.band!(plot, horizons, lower, upper; color = (bandcolor, alpha))
    end

    # Point estimate line
    Makie.lines!(plot, horizons, beta; color = linecolor, linewidth = linewidth)

    # Zero line
    if drawzero
        Makie.hlines!(plot, [0.0]; color = zerolinecolor, linewidth = 1, linestyle = :dash)
    end

    return plot
end

# ============================================================================
# Convenience wrappers mapping irfplot/irfplot! to irfplotmakie/irfplotmakie!
# ============================================================================

irfplot(args...; kwargs...) = irfplotmakie(args...; kwargs...)
irfplot!(args...; kwargs...) = irfplotmakie!(args...; kwargs...)

function irfplot_axis(subfig, lp, estimator_or_cov; kwargs...)
    kw = Dict{Symbol, Any}(kwargs)
    term = get(kw, :term, lp.shock)
    title = pop!(kw, :title, "")
    ax = Makie.Axis(subfig[1, 1];
        xlabel = "Horizon",
        ylabel = string(term),
        title = title
    )
    p = irfplotmakie!(ax, lp, estimator_or_cov; kw...)
    return (subfig, ax, p)
end

end # module
