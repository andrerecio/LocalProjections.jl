# Regenerate docs/src/assets/piger_stockwell_application.png — the long-difference
# example in the README, a replication of Figure 20 of Piger & Stockwell (2025).
#
#     julia --project=docs -e 'using Pkg; Pkg.instantiate()'
#     julia --project=docs docs/make_ps_figure.jl
#
# Impulse responses of 100·log industrial production and 100·log CPI to the
# standardized Jarocinski–Karadi (2020) monetary-policy and central-bank
# information shocks, 1990:2–2019:12, h = 0..36, estimated in levels
# (y_{t+h} on 12 lags of the five variables) and in long differences
# (y_{t+h} − y_{t−1} on 12 lags of their first differences). 90% bands from
# Newey–West with 37 lags, as in their application code (Bartlett(38) here).
# The README states the numbers this figure is built from.

using LocalProjections, DataFrames, Plots, Statistics
using StatsModels: @formula
using CovarianceMatrices: Bartlett

const HERE = @__DIR__
const DATA = joinpath(HERE, "src", "data", "piger_stockwell.csv")
const OUT = joinpath(HERE, "src", "assets", "piger_stockwell_application.png")

const HMAX = 36

"Minimal reader for the cleaned data file (docs env has no CSV.jl)."
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

ps = readclean(DATA)
ps = ps[1:(nrow(ps) - 57), :]          # sample ends in 2019:12, as in their code
for s in (:mp, :cbi)
    # Standardized shock. NaN rather than missing before 1990:2, so that those
    # months still supply the lagged controls (see "Scope and known limitations").
    ps[!, s] = coalesce.(ps[!, s] ./ std(skipmissing(ps[!, s])), NaN)
end

# ---------------------------------------------------------------- estimation

level = Dict(
    :mp => @formula(leads(y) ~
             mp + lags(lip, 12) + lags(lcpi, 12) + lags(lsp500, 12) +
             lags(ebp, 12) + lags(gs1, 12)),
    :cbi => @formula(leads(y) ~
             cbi + lags(lip, 12) + lags(lcpi, 12) + lags(lsp500, 12) +
             lags(ebp, 12) + lags(gs1, 12)))
longdiff = Dict(
    :mp => @formula(ldiff(y) ~
             mp + lags(firstdiff(lip), 12) + lags(firstdiff(lcpi), 12) +
             lags(firstdiff(lsp500), 12) + lags(firstdiff(ebp), 12) +
             lags(firstdiff(gs1), 12)),
    :cbi => @formula(ldiff(y) ~
             cbi + lags(firstdiff(lip), 12) + lags(firstdiff(lcpi), 12) +
             lags(firstdiff(lsp500), 12) + lags(firstdiff(ebp), 12) +
             lags(firstdiff(gs1), 12)))

function paths(shock, resp)
    d = copy(ps)
    d.y = d[!, resp]                   # the formulas name the response `y`
    out = map((level, longdiff)) do f
        m = lp(f[shock], d; horizon = HMAX)
        summarize(m, Bartlett(38); term = shock, level = 0.90)
    end
    return (level = out[1], longdiff = out[2])
end

# ---------------------------------------------------------------- plotting

const H = 0:HMAX
const INK = RGB(0.106, 0.180, 0.310)          # long differences
const ACCENT = RGB(0.760, 0.290, 0.235)       # levels
const RULE = RGB(0.62, 0.63, 0.66)
const TEXT = RGB(0.20, 0.22, 0.26)

function base()
    (titlelocation = :left, titlefontsize = 9, titlefontcolor = TEXT,
        framestyle = :axes, grid = :y, gridalpha = 0.10, gridlinewidth = 0.6,
        foreground_color_axis = RULE, foreground_color_border = RULE,
        tickfontsize = 7, tickfontcolor = RULE, guidefontsize = 7,
        guidefontcolor = RULE, legend = false, ylabel = "", xticks = 0:6:HMAX)
end

function panel(shock, resp; title, xlab = "")
    s = paths(shock, resp)
    for (k, v) in pairs(s)
        @info title k coef_12_24_36=round.(v.coef[[13, 25, 37]]; digits = 3) se_ratio_36=round(
            s.longdiff.se[37]/s.level.se[37]; digits = 2)
    end
    p = plot(H, s.level.lower; fillrange = s.level.upper, linealpha = 0,
        fillcolor = ACCENT, fillalpha = 0.14, title = title, xlabel = xlab, base()...)
    plot!(p, H, s.longdiff.lower; fillrange = s.longdiff.upper, linealpha = 0,
        fillcolor = INK, fillalpha = 0.16)
    hline!(p, [0.0]; c = RULE, ls = :dot, lw = 0.8)
    plot!(p, H, s.level.coef; c = ACCENT, ls = :dash, lw = 1.6)
    plot!(p, H, s.longdiff.coef; c = INK, lw = 1.8)
    return p
end

panels = [
    panel(:mp, :lip; title = "Industrial production · monetary policy"),
    panel(:mp, :lcpi; title = "CPI · monetary policy"),
    panel(:cbi, :lip; title = "Industrial production · CB information", xlab = "months"),
    panel(:cbi, :lcpi; title = "CPI · CB information", xlab = "months")
]

fig = plot(panels...; layout = (2, 2), size = (1000, 620), dpi = 150,
    background_color = :white, left_margin = 6Plots.mm, right_margin = 4Plots.mm,
    top_margin = 1Plots.mm, bottom_margin = 3Plots.mm,
    plot_title = "Piger–Stockwell application · solid = long differences, " *
                 "dashed = levels · 90% Newey–West bands (37 lags)",
    plot_titlefontsize = 9, plot_titlefontcolor = TEXT,
    plot_titlelocation = :left)

mkpath(dirname(OUT))
savefig(fig, OUT)
@info "wrote $OUT"
