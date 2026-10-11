"""
    FEVDSummary

Pointwise confidence band for a Gorodnichenko–Lee forecast-error variance
share, built by [`summarize`](@ref) from an [`LPFEVD`](@ref) or a
[`FEVDBootstrap`](@ref). Displays with PrettyTables; `DataFrame(s)` gives the
columns `horizon, share, estimate, se, lower, upper`.

# Fields
- `response`, `shock`, `estimator` (`:r2`, `:lpa`, `:lpb`);
- `method`: `:asymptotic` for an `LPFEVD`, else the bias-correction procedure
  of the bootstrap (`:var`, `:block`, `:simulation`);
- `band`: `:normal` (``\\text{centre} \\pm z\\cdot\\text{s.e.}``) or `:shift`
  (bootstrap quantiles translated to the centre);
- `level`, `biascorrected` (whether the band is centred on the bias-corrected
  share), `prewhiten` (whether a `:normal` band uses the prewhitened standard
  error; `nothing` for `:shift`);
- `horizon`, `share` (the estimate ``\\hat s_h``), `estimate` (the centre of
  the band: the bias-corrected share or ``\\hat s_h``), `se` (the standard
  error of the band: the delta-method one for `:normal`, the standard
  deviation of the draws for `:shift`), `lower`, `upper`.
"""
struct FEVDSummary
    response::Symbol
    shock::Symbol
    estimator::Symbol
    method::Symbol
    band::Symbol
    level::Float64
    biascorrected::Bool
    prewhiten::Union{Nothing, Bool}
    horizon::Vector{Int}
    share::Vector{Float64}
    estimate::Vector{Float64}
    se::Vector{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
end

function DataFrames.DataFrame(s::FEVDSummary)
    DataFrame(horizon = s.horizon, share = s.share, estimate = s.estimate, se = s.se,
        lower = s.lower, upper = s.upper)
end

function Base.show(io::IO, s::FEVDSummary)
    print(io, "FEVDSummary(estimator=:", s.estimator, ", method=:", s.method,
        ", band=:", s.band, ", level=", s.level, ")")
end

function Base.show(io::IO, ::MIME"text/plain", s::FEVDSummary)
    what = s.method === :asymptotic ? "delta method" : _FEVD_METHOD_LABELS[s.method]
    println(
        io, "FEVDSummary (Gorodnichenko–Lee ", _fevd_label(s.estimator), ", ", what, ")")
    println(io, "  Response:   ", s.response)
    println(io, "  Shock:      ", s.shock)
    se_label = s.band === :shift ? "s.e. (draws)" :
               s.prewhiten ? "s.e. (PW)" : "s.e."
    print(io, "  Band:       ",
        s.band === :shift ? "bootstrap quantiles shifted to the centre" :
        "normal, centre ± z·$(se_label)",
        s.method === :asymptotic ? "" :
        s.biascorrected ? ", bias-corrected" : ", uncorrected")
    pct = round(s.level * 100; digits = 1)
    pct_s = isinteger(pct) ? string(Int(pct)) : string(pct)
    if s.biascorrected
        data = (; h = s.horizon, s = s.share, bc = s.estimate, se = s.se, lo = s.lower,
            hi = s.upper)
        labels = ["Horizon", "Share", "Bias-corr.", se_label, "Lower $pct_s%",
            "Upper $pct_s%"]
    else
        data = (; h = s.horizon, s = s.share, se = s.se, lo = s.lower, hi = s.upper)
        labels = ["Horizon", "Share", se_label, "Lower $pct_s%", "Upper $pct_s%"]
    end
    println(io)
    nc = length(labels)
    pretty_table(io, data;
        column_labels = labels,
        title = "Forecast-error variance share of $(s.shock)",
        formatters = [fmt__round(4, collect(2:nc))],
        alignment = fill(:r, nc),
        table_format = TextTableFormat(borders = text_table_borders__unicode_rounded))
    return nothing
end

# MATLAB's `quantile`: Hyndman–Fan type 5, plotting positions (i - 0.5)/n with
# linear interpolation, clamped at the extremes (spec Q37). `_boot_interval`
# uses type 7.
_gl_quantile5(v::AbstractVector{<:Real}, p::Real) = quantile(v, p; alpha = 0.5, beta = 0.5)

"""
    summarize(f::LPFEVD; level=0.90, prewhiten=true) -> FEVDSummary

Normal confidence band ``\\hat s_h \\pm z_{(1+\\text{level})/2}\\,
\\widehat{\\mathrm{se}}_h`` for a Gorodnichenko–Lee FEVD, with the delta-method
standard error at ``\\hat s_h`` (`se_pw` with `prewhiten = true`, else `se`).
Deterministic; the band is not clamped to ``[0, 1]`` and is `NaN` at horizons
whose standard error is `NaN`. Bias-corrected bands come from
[`fevdbootstrap`](@ref).
"""
function summarize(f::LPFEVD; level::Real = 0.90, prewhiten::Bool = true)
    level_f = _check_level(level)
    se = copy(prewhiten ? f.se_pw : f.se)
    z = quantile(Normal(), 0.5 + level_f / 2)
    return FEVDSummary(f.response, f.shock, f.estimator, :asymptotic, :normal, level_f,
        false, prewhiten, collect(0:(f.horizon)), copy(f.share), copy(f.share), se,
        f.share .- z .* se, f.share .+ z .* se)
end

"""
    summarize(b::FEVDBootstrap; level=0.90, band=:shift, biascorrected=true,
              prewhiten=true) -> FEVDSummary

Pointwise confidence band for a bias-corrected Gorodnichenko–Lee FEVD. With
``c_h`` the centre — the bias-corrected share ``\\tilde s_h`` (`b.biascorrected`)
when `biascorrected = true`, the estimate ``\\hat s_h`` otherwise — and
``q \\in \\{\\alpha/2, 1-\\alpha/2\\}``, ``\\alpha = 1 - `` `level`:

- `band = :shift` (the paper's band, `cbIndicator = 0`):
  ``Q_q(s^*_{\\cdot,h}) + c_h - \\bar m_h``, the distribution of the draws
  translated so that its mean is the centre. ``Q_q`` is MATLAB's `quantile`
  (Hyndman–Fan type 5). With `biascorrected = false` this is the uncorrected band
  ``Q_q(s^*) + \\hat s_h - \\bar m_h`` drawn in the paper's figures
  (`AppSimulation.m:325-329`). The reported `se` is the standard deviation of the
  draws.
- `band = :normal` (`cbIndicator = 1` / `2`): ``c_h + \\Phi^{-1}(q)\\,
  \\mathrm{se}_h`` with `b.se_pw` (`prewhiten = true`) or `b.se`. These are the
  delta-method standard errors at ``\\hat s_h`` for `:var` and `:block` and at
  ``\\tilde s_h`` for `:simulation` (see [`FEVDBootstrap`](@ref)); with
  `biascorrected = false` the same band is translated to ``\\hat s_h``.

The keyword is `band` rather than `method` (the name `summarize(::LPBootstrap)`
uses for its interval construction) because `method` already names the
bias-correction procedure of [`fevdbootstrap`](@ref) and of `b.method`. Bands
are pointwise and never clamped to ``[0, 1]``; they are `NaN` at horizons
without draws (`:simulation` horizons whose covariance is `NaN`).
"""
function summarize(b::FEVDBootstrap; level::Real = 0.90, band::Symbol = :shift,
        biascorrected::Bool = true, prewhiten::Bool = true)
    band in (:shift, :normal) || throw(ArgumentError(
        "`band` must be :shift or :normal (got :$band)"))
    level_f = _check_level(level)
    α = 1 - level_f
    f = b.fevd
    H = f.horizon
    centre = copy(biascorrected ? b.biascorrected : f.share)
    lower = fill(NaN, H + 1)
    upper = fill(NaN, H + 1)
    if band === :shift
        se = copy(b.se_boot)
        for h in 1:(H + 1)
            col = view(b.draws, :, h)
            (!isempty(col) && all(isfinite, col)) || continue
            shift = centre[h] - b.mean_draws[h]
            lower[h] = _gl_quantile5(col, α / 2) + shift
            upper[h] = _gl_quantile5(col, 1 - α / 2) + shift
        end
        pw = nothing
    else
        se = copy(prewhiten ? b.se_pw : b.se)
        z = quantile(Normal(), 1 - α / 2)
        lower .= centre .- z .* se
        upper .= centre .+ z .* se
        pw = prewhiten
    end
    return FEVDSummary(f.response, f.shock, f.estimator, b.method, band, level_f,
        biascorrected, pw, collect(0:H), copy(f.share), centre, se, lower, upper)
end
