# Regenerate docs/src/assets/ramey_zubairy_ewc.png — the HAR-inference figure of
# the Ramey–Zubairy (2018) worked example in the README.
#
#     julia --project=docs -e 'using Pkg; Pkg.instantiate()'
#     julia --project=docs docs/make_rz_ewc_figure.jl
#
# The figure shows the impulse responses of government spending and GDP to the
# military-news shock with 68% and 90% bands from the equal-weighted-cosine
# (EWC) variance estimator and Student-t_B critical values, as recommended by
# Lazarus, Lewis, Stock & Watson (2018). The 90% band from the automatic
# Newey–West bandwidth with normal critical values (the one in
# ramey_zubairy_replication.png) is dashed for comparison. The README states the
# numbers this figure is built from.

using LocalProjections, DataFrames, Plots
using StatsModels: @formula
using CovarianceMatrices: Bartlett, NeweyWest, EWC

const HERE = @__DIR__
const DATA = joinpath(HERE, "src", "data", "ramey_zubairy.csv")
const OUT = joinpath(HERE, "src", "assets", "ramey_zubairy_ewc.png")

const HMAX = 20

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

# ---------------------------------------------------------------- estimation

irf_g = lp(@formula(leads(g) ~ newsy + lags(newsy, 4) + lags(y, 4) + lags(g, 4)),
    rz; horizon = HMAX)
irf_y = lp(@formula(leads(y) ~ newsy + lags(newsy, 4) + lags(y, 4) + lags(g, 4)),
    rz; horizon = HMAX)

# One B for every horizon, from the h = 0 sample: B = ⌊0.41·T₀^(2/3)⌋.
const B = ewc_bandwidth(irf_g)
B == ewc_bandwidth(irf_y) || error("the two responses have different h = 0 samples")

function bands(m)
    cov_ewc = vcov(EWC(B), m)
    # A `Bartlett{NeweyWest}` instance is stateful: build a fresh one per model.
    cov_nw = vcov(Bartlett{NeweyWest}(), m)
    (ewc90 = summarize(m, cov_ewc; term = :newsy, level = 0.90),
        ewc68 = summarize(m, cov_ewc; term = :newsy, level = 0.68),
        nw90 = summarize(m, cov_nw; term = :newsy, level = 0.90))
end

# ---------------------------------------------------------------- plotting

const H = 0:HMAX
const INK = RGB(0.106, 0.180, 0.310)          # response line and band fill
const ACCENT = RGB(0.760, 0.290, 0.235)       # Newey–West comparison band
const RULE = RGB(0.62, 0.63, 0.66)            # axes and reference lines
const TEXT = RGB(0.20, 0.22, 0.26)

function base()
    (titlelocation = :left, titlefontsize = 9, titlefontcolor = TEXT,
        framestyle = :axes, grid = :y, gridalpha = 0.10, gridlinewidth = 0.6,
        foreground_color_axis = RULE, foreground_color_border = RULE,
        tickfontsize = 7, tickfontcolor = RULE, guidefontsize = 7,
        guidefontcolor = RULE, legend = false, ylabel = "")
end

function ewcpanel(m; title, xlab = "quarters")
    s = bands(m)
    # EWC / Newey–West standard-error ratio by horizon, quoted in the README
    @info title B se_ratio=round.(s.ewc90.se ./ s.nw90.se; digits = 2)
    p = plot(H, s.ewc90.lower; fillrange = s.ewc90.upper, linealpha = 0,
        fillcolor = INK, fillalpha = 0.14, label = "", title = title, xlabel = xlab,
        base()...)
    plot!(p, H, s.ewc68.lower; fillrange = s.ewc68.upper, linealpha = 0,
        fillcolor = INK, fillalpha = 0.20, label = "")
    hline!(p, [0.0]; c = RULE, ls = :dot, lw = 0.8, label = "")
    plot!(p, H, s.nw90.lower; c = ACCENT, ls = :dash, lw = 1.0, label = "")
    plot!(p, H, s.nw90.upper; c = ACCENT, ls = :dash, lw = 1.0, label = "")
    plot!(p, H, s.ewc90.coef; c = INK, lw = 1.8, label = "")
    return p
end

panels = [
    ewcpanel(irf_g; title = "Government spending · news"),
    ewcpanel(irf_y; title = "GDP · news")
]

fig = plot(panels...; layout = (1, 2), size = (1000, 340), dpi = 150,
    background_color = :white, left_margin = 6Plots.mm, right_margin = 4Plots.mm,
    top_margin = 1Plots.mm, bottom_margin = 3Plots.mm,
    plot_title = "Ramey–Zubairy news shock · EWC (B = $B), 68/90% Student-t bands, " *
                 "dashed = 90% Newey–West",
    plot_titlefontsize = 9, plot_titlefontcolor = TEXT,
    plot_titlelocation = :left)

mkpath(dirname(OUT))
savefig(fig, OUT)
@info "wrote $OUT"
