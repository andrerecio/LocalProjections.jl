# ============================================================================
# FEVD conversion to MacroEconometricTools and plots
# ============================================================================

# Shared body of the two FEVD conversions: `summaries` are FEVDSummary bands,
# one per coverage level, with a common centre and standard error.
function _fevd_irf_result(summaries::Vector{FEVDSummary}, coverage, meta::NamedTuple)
    s1 = summaries[1]
    H = length(s1.horizon)
    cube(v) = AxisArray(
        reshape(collect(Float64, v), 1, 1, H),
        Axis{:response}([s1.response]),
        Axis{:shock}([s1.shock]),
        Axis{:horizon}(0:(H - 1))
    )
    return LocalProjectionIRFResult(
        cube(s1.estimate), cube(s1.se), [cube(s.lower) for s in summaries],
        [cube(s.upper) for s in summaries], collect(Float64, coverage), meta
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
function as_irf_result(f::LPFEVD; coverage = [0.68, 0.9], prewhiten::Bool = true)
    summaries = [summarize(f; level = c, prewhiten = prewhiten) for c in coverage]
    meta = (
        horizon = f.horizon,
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
        bias_corrected = false,
    )
    return _fevd_irf_result(summaries, coverage, meta)
end

"""
    as_irf_result(b::FEVDBootstrap; coverage=[0.68, 0.90], band=:shift,
                  biascorrected=true, prewhiten=true)

Convert a bias-corrected Gorodnichenko–Lee FEVD ([`FEVDBootstrap`](@ref)) to a
`LocalProjectionIRFResult`. `data` holds the centre of the bands of
[`summarize`](@ref) under the same keywords (the bias-corrected shares by
default) and `stderr` their standard error (the standard deviation of the
draws for `band = :shift`). The metadata records `fevd = true`,
`quantity = :fevd_share`, `bootstrap = true`, the procedure
(`bootstrap_method`), the band, the uncorrected shares (`share`), the VAR
pseudo-truth for `:var`, and the draw counts.
"""
function as_irf_result(
        b::FEVDBootstrap; coverage = [0.68, 0.9], band::Symbol = :shift,
        biascorrected::Bool = true, prewhiten::Bool = true
    )
    summaries = [summarize(b; level = c, band, biascorrected, prewhiten) for c in coverage]
    f = b.fevd
    meta = (
        horizon = f.horizon,
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
        var_variables = b.vars,
    )
    return _fevd_irf_result(summaries, coverage, meta)
end

"""
    irfplot(f::LPFEVD; levels = [0.90], prewhiten = true, kwargs...) -> Makie.Figure
    irfplot(b::FEVDBootstrap; levels = [0.90], band = :shift, biascorrected = true,
            prewhiten = true, kwargs...) -> Makie.Figure

Plot the forecast-error variance shares with the bands of [`summarize`](@ref) at each
of `levels`, from [`as_irf_result`](@ref) (requires a Makie backend); the remaining
keywords go to MacroEconometricTools' `irfplot`. With Plots.jl, `plot(f; levels,
prewhiten)` and `plot(b; levels, band, biascorrected, prewhiten)` draw the same bands,
with the y-axis covering ``[0, 1]`` and, for a bias-corrected centre, the uncorrected
estimate as a dashed line.
"""
function MacroEconometricTools.irfplot(
        f::LPFEVD; levels = [0.9], prewhiten::Bool = true, kwargs...
    )
    return irfplot(as_irf_result(f; coverage = levels, prewhiten); kwargs...)
end

function MacroEconometricTools.irfplot(
        b::FEVDBootstrap; levels = [0.9], band::Symbol = :shift, biascorrected::Bool = true,
        prewhiten::Bool = true, kwargs...
    )
    return irfplot(as_irf_result(b; coverage = levels, band, biascorrected, prewhiten); kwargs...)
end

"""
    FEVDPlot

Internal wrapper for dispatching the Plots.jl recipe on [`LPFEVD`](@ref) and
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

_fevd_plot_summary(f::LPFEVD, w::FEVDPlot, level) = summarize(f; level, prewhiten = w.prewhiten)
function _fevd_plot_summary(b::FEVDBootstrap, w::FEVDPlot, level)
    return summarize(
        b; level, band = w.band, biascorrected = w.biascorrected, prewhiten = w.prewhiten
    )
end

@recipe function f(wrapper::FEVDPlot)
    obj = wrapper.fevd
    sorted_levels = sort(wrapper.levels; rev = true)
    _check_levels(sorted_levels)
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

@recipe function f(fv::LPFEVD; levels = [0.9], prewhiten = true)
    return FEVDPlot(fv, Float64.(levels), :normal, false, prewhiten)
end

@recipe function f(
        b::FEVDBootstrap; levels = [0.9], band = :shift, biascorrected = true, prewhiten = true
    )
    return FEVDPlot(b, Float64.(levels), band, biascorrected, prewhiten)
end
