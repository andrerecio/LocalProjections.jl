# Regenerate the two bias-corrected bootstrap figures of the Ramey–Zubairy
# (2018) worked example in the README:
#
#     docs/src/assets/ramey_zubairy_bootstrap_lags4.png   # RZ specification, 4 lags
#     docs/src/assets/ramey_zubairy_bootstrap_lags9.png   # AIC-selected VAR order, 9 lags
#
#     julia --project=docs -e 'using Pkg; Pkg.instantiate()'
#     julia --project=docs docs/make_rz_bootstrap_figure.jl
#
# Each figure shows the impulse responses of government spending and GDP to the
# military-news shock: the Herbst–Johannsen bias-corrected local projection with
# 68% and 90% Hall percentile-t bands from the VAR moving-block bootstrap (Pope-
# corrected data-generating process), and the uncorrected OLS path as a dashed
# line. The README states the numbers these figures are built from.

using LocalProjections, DataFrames, Plots, Random
using StatsModels: @formula

const HERE = @__DIR__
const DATA = joinpath(HERE, "src", "data", "ramey_zubairy.csv")
const ASSETS = joinpath(HERE, "src", "assets")

const HMAX = 20
const NBOOT = 1000
const SEED = 20260916
const VARS = [:newsy, :y, :g]      # VAR data vector, shock first

"Minimal reader for the cleaned RZ file (docs env has no CSV.jl)."
function readclean(path)
    ls = readlines(path)
    hdr = Symbol.(strip.(split(ls[1], ',')))
    rows = [split(l, ',') for l in ls[2:end] if !isempty(strip(l))]
    df = DataFrame()
    for (j, h) in enumerate(hdr)
        v = Vector{Union{Missing, Float64}}(undef, length(rows))
        for (i, r) in enumerate(rows)
            s = strip(r[j])
            v[i] = isempty(s) ? missing : parse(Float64, s)
        end
        df[!, h] = v
    end
    df
end

rz = readclean(DATA)
# The VAR columns must be free of missing values; newsy starts in 1890Q1.
rzc = dropmissing(rz, VARS)

# ---------------------------------------------------------------- estimation

# `@formula` takes a literal lag count, so the two specifications are spelled out.
function formulas(::Val{4})
    (
        @formula(leads(y) ~ newsy + lags(newsy, 4) + lags(y, 4) + lags(g, 4)),
        @formula(leads(g) ~ newsy + lags(newsy, 4) + lags(y, 4) + lags(g, 4)))
end
function formulas(::Val{9})
    (
        @formula(leads(y) ~ newsy + lags(newsy, 9) + lags(y, 9) + lags(g, 9)),
        @formula(leads(g) ~ newsy + lags(newsy, 9) + lags(y, 9) + lags(g, 9)))
end

# The AIC-selected order the README quotes; fail loudly if the data or the
# selection rule ever change.
sel = lagselect(rzc, VARS; maxlags = 10, criterion = :aic)
nlags(sel) == 9 || error("lagselect picked $(nlags(sel)) lags, expected 9")

function estimate(p::Int)
    fy, fg = formulas(Val(p))
    irf_y = lp(fy, rzc; horizon = HMAX)
    irf_g = lp(fg, rzc; horizon = HMAX)
    # Same seed for both responses: identical artificial samples.
    boot_y = varbootstrap(irf_y, rzc; vars = VARS, nlags = p, nboot = NBOOT,
        rng = Xoshiro(SEED))
    boot_g = varbootstrap(irf_g, rzc; vars = VARS, nlags = p, nboot = NBOOT,
        rng = Xoshiro(SEED))
    (; irf_y, irf_g, boot_y, boot_g)
end

# ---------------------------------------------------------------- plotting

const H = 0:HMAX
const INK = RGB(0.106, 0.180, 0.310)          # response line and band fill
const RULE = RGB(0.62, 0.63, 0.66)            # axes and reference lines
const TEXT = RGB(0.20, 0.22, 0.26)

function base()
    (titlelocation = :left, titlefontsize = 9, titlefontcolor = TEXT,
        framestyle = :axes, grid = :y, gridalpha = 0.10, gridlinewidth = 0.6,
        foreground_color_axis = RULE, foreground_color_border = RULE,
        tickfontsize = 7, tickfontcolor = RULE, guidefontsize = 7,
        guidefontcolor = RULE, legend = false, ylabel = "")
end

function bootpanel(irf, boot; title, xlab = "quarters")
    s90 = summarize(boot; level = 0.90)
    s68 = summarize(boot; level = 0.68)
    p = plot(H, s90.lower; fillrange = s90.upper, linealpha = 0, fillcolor = INK,
        fillalpha = 0.14, label = "", title = title, xlabel = xlab, base()...)
    plot!(p, H, s68.lower; fillrange = s68.upper, linealpha = 0, fillcolor = INK,
        fillalpha = 0.20, label = "")
    hline!(p, [0.0]; c = RULE, ls = :dot, lw = 0.8, label = "")
    plot!(p, H, coefpath(irf; term = :newsy); c = INK, ls = :dash, lw = 1.0,
        label = "")                                   # uncorrected OLS path
    plot!(p, H, s90.coef; c = INK, lw = 1.8, label = "")   # bias-corrected path
    return p
end

function figure(p::Int, subtitle::AbstractString)
    est = estimate(p)
    @info "lags = $p" nfail=(est.boot_y.nfail, est.boot_g.nfail) pope_delta=(
        est.boot_y.pope_delta, est.boot_g.pope_delta)
    panels = [
        bootpanel(est.irf_g, est.boot_g; title = "Government spending · news"),
        bootpanel(est.irf_y, est.boot_y; title = "GDP · news")
    ]
    plot(panels...; layout = (1, 2), size = (1000, 340), dpi = 150,
        background_color = :white, left_margin = 6Plots.mm, right_margin = 4Plots.mm,
        top_margin = 1Plots.mm, bottom_margin = 3Plots.mm,
        plot_title = "Ramey–Zubairy news shock · $subtitle · bias-corrected LP, " *
                     "68/90% bootstrap bands, dashed = uncorrected",
        plot_titlefontsize = 9, plot_titlefontcolor = TEXT,
        plot_titlelocation = :left)
end

mkpath(ASSETS)
for (p, subtitle) in ((4, "4 lags (RZ)"), (9, "9 lags (AIC)"))
    out = joinpath(ASSETS, "ramey_zubairy_bootstrap_lags$(p).png")
    savefig(figure(p, subtitle), out)
    @info "wrote $out"
end
