# Regenerate docs/src/assets/ramey_zubairy_state.png — the state-dependence
# figure of the Ramey–Zubairy (2018) worked example in the README.
#
#     julia --project=docs -e 'using Pkg; Pkg.instantiate()'
#     julia --project=docs docs/make_rz_state_figure.jl
#
# The figure shows the one-step cumulative multiplier by state of the economy,
# one row per state — expansion (unemployment below 6.5% in the previous
# quarter) and slack — for the military-news and the Blanchard–Perotti shock,
# with 68% and 90% bands from the automatic Newey–West bandwidth. The linear
# (state-independent) multiplier is dashed for reference. Horizons 0 and 1 are left out: the news
# instrument is very weak there (effective F of about 2) and the bands dwarf
# the rest of the path. The README states the numbers this figure is built from.

using LocalProjections, DataFrames, Plots
using StatsModels: @formula
using CovarianceMatrices: Bartlett, NeweyWest

const HERE = @__DIR__
const DATA = joinpath(HERE, "src", "data", "ramey_zubairy.csv")
const OUT = joinpath(HERE, "src", "assets", "ramey_zubairy_state.png")

const HMAX = 20
const HSHOW = 2:HMAX

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
rz.bp = rz.g

# ---------------------------------------------------------------- estimation

const REGIMES = ("expansion", "slack")

const F_NEWS = @formula(cumul(y) ~
                        (cumul(g) ~ newsy) +
                        lags(newsy, 4) + lags(y, 4) + lags(g, 4))
const F_BP = @formula(cumul(y) ~ (cumul(g) ~ bp) + lags(y, 4) + lags(g, 4))

function fit(f)
    (linear = lpiv(f, rz; horizon = HMAX),
        state = lpiv(f, rz; horizon = HMAX, state = :slack, regimes = REGIMES))
end

news = fit(F_NEWS)
bp = fit(F_BP)

# ---------------------------------------------------------------- plotting

const INK = RGB(0.106, 0.180, 0.310)          # slack
const ACCENT = RGB(0.760, 0.290, 0.235)       # expansion
const RULE = RGB(0.62, 0.63, 0.66)            # axes and reference lines
const TEXT = RGB(0.20, 0.22, 0.26)

function base()
    (titlelocation = :left, titlefontsize = 9, titlefontcolor = TEXT,
        framestyle = :axes, grid = :y, gridalpha = 0.10, gridlinewidth = 0.6,
        foreground_color_axis = RULE, foreground_color_border = RULE,
        tickfontsize = 7, tickfontcolor = RULE, guidefontsize = 7,
        guidefontcolor = RULE, legend = false, ylabel = "")
end

function statepanel(m, regime, color; title, xlab = "")
    # A `Bartlett{NeweyWest}` instance is stateful: build a fresh one per model.
    cov = vcov(Bartlett{NeweyWest}(), m.state)
    term = Symbol("cumul(g)_", regime)
    s90 = summarize(m.state, cov; term, level = 0.90)
    s68 = summarize(m.state, cov; term, level = 0.68)
    pval = statetest(m.state, Bartlett{NeweyWest}()).pvalue
    @info title two_year=round(s90.coef[9]; digits = 3) four_year=round(
        s90.coef[17]; digits = 3) p_equal_two_year=round(pval[9]; digits = 3)
    idx = collect(HSHOW) .+ 1
    p = plot(HSHOW, s90.lower[idx]; fillrange = s90.upper[idx], linealpha = 0,
        fillcolor = color, fillalpha = 0.14, label = "", title = title,
        xlabel = xlab, ylims = (-0.1, 1.25), base()...)
    plot!(p, HSHOW, s68.lower[idx]; fillrange = s68.upper[idx], linealpha = 0,
        fillcolor = color, fillalpha = 0.20, label = "")
    hline!(p, [0.0, 1.0]; c = RULE, ls = :dot, lw = 0.8, label = "")
    plot!(p, HSHOW, coefpath(m.linear)[idx]; c = TEXT, ls = :dash, lw = 1.0, label = "")
    plot!(p, HSHOW, s90.coef[idx]; c = color, lw = 1.8, label = "")
    return p
end

panels = [
    statepanel(news, "expansion", ACCENT; title = "Expansion · news"),
    statepanel(bp, "expansion", ACCENT; title = "Expansion · Blanchard–Perotti"),
    statepanel(news, "slack", INK; title = "Slack · news", xlab = "quarters"),
    statepanel(bp, "slack", INK; title = "Slack · Blanchard–Perotti", xlab = "quarters")
]

fig = plot(panels...; layout = (2, 2), size = (1000, 620), dpi = 150,
    background_color = :white, left_margin = 6Plots.mm, right_margin = 4Plots.mm,
    top_margin = 1Plots.mm, bottom_margin = 3Plots.mm,
    plot_title = "Ramey–Zubairy cumulative multipliers by state · 68/90% " *
                 "Newey–West bands, dashed = linear model",
    plot_titlefontsize = 9, plot_titlefontcolor = TEXT,
    plot_titlelocation = :left)

mkpath(dirname(OUT))
savefig(fig, OUT)
@info "wrote $OUT"
