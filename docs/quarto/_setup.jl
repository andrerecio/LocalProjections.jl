# Shared by every page of the site: packages, data, and small formatting helpers.
using LocalProjections, CSV, DataFrames, Plots, Printf, Statistics
using StatsModels: @formula, term
using CovarianceMatrices: Bartlett, NeweyWest

gr()
default(dpi = 150)

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
datafile(name) = joinpath(ROOT, "docs", "src", "data", name)

f2(x) = @sprintf("%.2f", x)
f3(x) = @sprintf("%.3f", x)
f4(x) = @sprintf("%.4f", x)

# a Markdown table, printed from a cell with `output: asis`
function mdtable(header, rows; align = fill("---:", length(header)))
    println("| ", join(header, " | "), " |")
    println("|", join(align, "|"), "|")
    for r in rows
        println("| ", join(r, " | "), " |")
    end
    println()
end

# ── Ramey & Zubairy (2018): US quarterly, 1889Q1–2015Q4 ─────────────────────
function rz_data()
    rz = CSV.read(datafile("ramey_zubairy.csv"), DataFrame; missingstring = "")
    rz.bp = rz.g          # Blanchard–Perotti shock: current spending
    return rz
end

# Published one-step linear multipliers and standard errors (`multlin1`/`seylin`
# in `Multiplier-Standard-Errors.xlsx` of the replication package), at the
# horizons reported in docs/src/tutorials/ramey_zubairy.md.
const RZ_PUBLISHED = (
    h = [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    news = [1.255, 1.035, 0.820, 0.695, 0.674, 0.673, 0.667, 0.705, 0.717, 0.716,
        0.708, 0.713, 0.727],
    news_se = [0.329, 0.235, 0.155, 0.123, 0.097, 0.074, 0.060, 0.053, 0.052, 0.046,
        0.046, 0.053, 0.063],
    bp = [0.208, 0.235, 0.257, 0.251, 0.271, 0.353, 0.411, 0.439, 0.458, 0.471,
        0.467, 0.453, 0.440],
    bp_se = [0.155, 0.143, 0.133, 0.133, 0.132, 0.119, 0.104, 0.102, 0.102, 0.105,
        0.119, 0.133, 0.139])

# ── Piger & Stockwell (2025): US monthly, 1989:1–2024:9 ─────────────────────
# Sample through 2019:12 as in their application code; shocks standardized. The
# shocks are `missing` before 1990:2 and recoded as NaN, so that those months
# still supply the lagged controls (`lp` drops rows with `missing` before it
# builds lags).
function ps_data()
    ps = CSV.read(datafile("piger_stockwell.csv"), DataFrame; missingstring = "")
    ps = ps[1:(nrow(ps) - 57), :]
    for s in (:mp, :cbi)
        ps[!, s] = coalesce.(ps[!, s] ./ std(skipmissing(ps[!, s])), NaN)
    end
    return ps
end

# ── figure style (shared with docs/make_rz_*.jl) ─────────────────────────────
const INK = RGB(0.106, 0.180, 0.310)          # response line and band fill
const BANDALPHA = 0.18
const ACCENT = RGB(0.760, 0.290, 0.235)       # thresholds and comparisons
const RULE = RGB(0.62, 0.63, 0.66)            # axes and reference lines
const TEXT = RGB(0.20, 0.22, 0.26)

# left/bottom spines only, faint horizontal grid; the titles carry the meaning
function base()
    (titlelocation = :left, titlefontsize = 9, titlefontcolor = TEXT,
        framestyle = :axes, grid = :y, gridalpha = 0.10, gridlinewidth = 0.6,
        foreground_color_axis = RULE, foreground_color_border = RULE,
        tickfontsize = 7, tickfontcolor = RULE, guidefontsize = 7,
        guidefontcolor = RULE, legend = false, ylabel = "")
end

function irfpanel(s::IRFSummary; title, reference = 0.0, xlab = "")
    H = s.horizon
    p = plot(H, s.lower; fillrange = s.upper, linealpha = 0, fillcolor = INK,
        fillalpha = BANDALPHA, title, xlabel = xlab, base()...)
    hline!(p, [reference]; c = RULE, ls = :dot, lw = 0.8)
    plot!(p, H, s.coef; c = INK, lw = 1.8)
    return p
end

# two estimates of one response: `a` shaded navy and solid, `b` brick and dashed
function comparepanel(a::IRFSummary, b::IRFSummary; title, xlab = "", xticks = :auto)
    H = a.horizon
    p = plot(H, b.lower; fillrange = b.upper, linealpha = 0, fillcolor = ACCENT,
        fillalpha = 0.12, title, xlabel = xlab, xticks, base()...)
    plot!(p, H, a.lower; fillrange = a.upper, linealpha = 0, fillcolor = INK,
        fillalpha = BANDALPHA)
    hline!(p, [0.0]; c = RULE, ls = :dot, lw = 0.8)
    plot!(p, H, b.coef; c = ACCENT, ls = :dash, lw = 1.6)
    plot!(p, H, a.coef; c = INK, lw = 1.8)
    return p
end

# effective F on a log scale against the Montiel Olea–Pflueger critical value
function fpanel(H, F, cv; title, labelpos = :top, xlab = "")
    # h = 0 under Blanchard–Perotti has instrument == regressor, so the first
    # stage is degenerate and F is numerically infinite: drop that point
    ok = isfinite.(F) .& (F .< 1e4)
    lo = min(minimum(F[ok]), cv) / 1.7
    hi = max(maximum(F[ok]), cv) * 1.7
    ticks = filter(t -> lo <= t <= hi, [1, 2, 5, 10, 20, 50, 100, 200, 500])
    p = plot(H[ok], F[ok]; c = INK, lw = 1.8, yscale = :log10, ylims = (lo, hi),
        yticks = (ticks, string.(ticks)), title, xlabel = xlab, base()...)
    hline!(p, [cv]; c = ACCENT, ls = :dash, lw = 1.2)
    annotate!(p, first(H[ok]) + 0.4, labelpos === :top ? hi / 1.3 : lo * 1.3,
        text("5% critical value = $(round(cv, digits = 2))", 6, ACCENT, :left))
    return p
end

const FIGKW = (background_color = :white, left_margin = 5Plots.mm,
    right_margin = 3Plots.mm, bottom_margin = 3Plots.mm)
