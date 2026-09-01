# Regenerate docs/src/assets/ramey_zubairy_replication.png — the Ramey–Zubairy
# (2018) baseline replication figure used in the README.
#
#     julia --project=docs -e 'using Pkg; Pkg.instantiate()'
#     julia --project=docs docs/make_rz_figure.jl
#
# Layout mirrors the paper's linear-model figure: the impulse responses of
# government spending and GDP, the one-step cumulative multiplier, and the
# Montiel Olea–Pflueger weak-instrument diagnostic, for both identification
# schemes. See docs/src/tutorials/ramey_zubairy.md for the estimator, and
# section 8 of docs/src/inference_procedures_guide.md for how this automatic
# Newey–West bandwidth differs from Stata's `ivreg2, bw(auto)`.

using LocalProjections, DataFrames, Plots
using StatsModels: @formula
using CovarianceMatrices: Bartlett, NeweyWest

const HERE = @__DIR__
const DATA = joinpath(HERE, "src", "data", "ramey_zubairy.csv")
const OUT = joinpath(HERE, "src", "assets", "ramey_zubairy_replication.png")

const HMAX = 20
const LEVEL = 0.90

# Automatic (Newey–West 1994) bandwidth selection. NOTE: a `Bartlett{NeweyWest}`
# object caches one kernel weight per moment column on first use, so a single
# instance cannot be reused across models with different regressor counts —
# build a fresh one for every call.
kernel() = Bartlett{NeweyWest}()

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
rz.bp = rz.g          # Blanchard–Perotti shock: current spending

# ---------------------------------------------------------------- estimation

# Impulse responses (the two-step block of jordagk.do)
irf_g_news = lp(@formula(leads(g) ~ newsy + lags(newsy, 4) + lags(y, 4) + lags(g, 4)),
    rz; horizon = HMAX)
irf_y_news = lp(@formula(leads(y) ~ newsy + lags(newsy, 4) + lags(y, 4) + lags(g, 4)),
    rz; horizon = HMAX)
irf_g_bp = lp(@formula(leads(g) ~ bp + lags(y, 4) + lags(g, 4)), rz; horizon = HMAX)
irf_y_bp = lp(@formula(leads(y) ~ bp + lags(y, 4) + lags(g, 4)), rz; horizon = HMAX)

# One-step cumulative multipliers: cumul() tracks the horizon on both sides
mult_news = lpiv(
    @formula(cumul(y) ~ (cumul(g) ~ newsy) +
                        lags(newsy, 4) + lags(y, 4) + lags(g, 4)),
    rz; horizon = HMAX)
mult_bp = lpiv(
    @formula(cumul(y) ~ (cumul(g) ~ bp) +
                        lags(y, 4) + lags(g, 4)), rz; horizon = HMAX)

summ(m, term) = summarize(m, vcov(kernel(), m); term = term, level = LEVEL)
summ(m) = summarize(m, vcov(kernel(), m); level = LEVEL)

# Montiel Olea–Pflueger effective F, computed off the same HAC covariance
function mop(m)
    hac = m + vcov(kernel())
    ws = [weakivtest(hac, h) for h in 0:HMAX]
    (F = [w.F_eff for w in ws], cv = ws[1].cv_TSLS[2])   # cv[2] = 23.1085
end

mop_news = mop(mult_news)
mop_bp = mop(mult_bp)

# ---------------------------------------------------------------- plotting

const H = 0:HMAX
const INK = RGB(0.106, 0.180, 0.310)          # response line and band fill
const BANDALPHA = 0.18                        # confidence band opacity
const ACCENT = RGB(0.760, 0.290, 0.235)       # critical value
const RULE = RGB(0.62, 0.63, 0.66)            # axes and reference lines
const TEXT = RGB(0.20, 0.22, 0.26)

# Shared minimalist frame: left/bottom spines only, faint horizontal grid, no
# axis labels — the titles carry the meaning.
function base()
    (titlelocation = :left, titlefontsize = 9, titlefontcolor = TEXT,
        framestyle = :axes, grid = :y, gridalpha = 0.10, gridlinewidth = 0.6,
        foreground_color_axis = RULE, foreground_color_border = RULE,
        tickfontsize = 7, tickfontcolor = RULE, guidefontsize = 7,
        guidefontcolor = RULE, legend = false, ylabel = "")
end

function irfpanel(s; title, reference = 0.0, xlab = "")
    p = plot(H, s.lower; fillrange = s.upper, linealpha = 0, fillcolor = INK,
        fillalpha = BANDALPHA, label = "", title = title, xlabel = xlab, base()...)
    hline!(p, [reference]; c = RULE, ls = :dot, lw = 0.8, label = "")
    plot!(p, H, s.coef; c = INK, lw = 1.8, label = "")
    return p
end

function fpanel(m; title, labelpos = :top, xlab = "")
    # h = 0 under Blanchard–Perotti has instrument == regressor, so the first
    # stage is degenerate and F is numerically infinite: drop that point.
    ok = isfinite.(m.F) .& (m.F .< 1e4)
    F = m.F[ok]
    lo = min(minimum(F), m.cv) / 1.7
    hi = max(maximum(F), m.cv) * 1.7
    ladder = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    ticks = filter(t -> lo <= t <= hi, ladder)
    p = plot(H[ok], F; c = INK, lw = 1.8, label = "", yscale = :log10,
        ylims = (lo, hi), yticks = (ticks, string.(ticks)),
        title = title, xlabel = xlab, base()...)
    hline!(p, [m.cv]; c = ACCENT, ls = :dash, lw = 1.2, label = "")
    # Label the threshold in place; a legend box would collide with the curve.
    annotate!(p, first(H[ok]) + 0.4, labelpos === :top ? hi / 1.3 : lo * 1.3,
        text("Montiel Olea-Pflueger 5% cv = $(round(m.cv, digits = 2))",
            6, ACCENT, :left))
    return p
end

panels = [
    irfpanel(summ(irf_g_news, :newsy); title = "Government spending · news"),
    irfpanel(summ(irf_g_bp, :bp); title = "Government spending · Blanchard–Perotti"),
    irfpanel(summ(irf_y_news, :newsy); title = "GDP · news"),
    irfpanel(summ(irf_y_bp, :bp); title = "GDP · Blanchard–Perotti"),
    irfpanel(summ(mult_news); title = "Cumulative multiplier · news", reference = 1.0),
    irfpanel(summ(mult_bp); title = "Cumulative multiplier · Blanchard–Perotti",
        reference = 1.0),
    fpanel(mop_news; title = "Effective F, log scale · news", xlab = "quarters"),
    fpanel(mop_bp; title = "Effective F, log scale · Blanchard–Perotti",
        labelpos = :bottom, xlab = "quarters")
]

fig = plot(panels...; layout = (4, 2), size = (940, 1000), dpi = 150,
    background_color = :white, left_margin = 6Plots.mm, right_margin = 4Plots.mm,
    top_margin = 1Plots.mm, bottom_margin = 3Plots.mm,
    plot_title = "Ramey–Zubairy (2018) linear model · US 1889Q1–2015Q4 · " *
                 "90% bands, Newey–West automatic bandwidth",
    plot_titlefontsize = 10, plot_titlefontcolor = TEXT,
    plot_titlelocation = :left)

mkpath(dirname(OUT))
savefig(fig, OUT)
@info "wrote $OUT"
