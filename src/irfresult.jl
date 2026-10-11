# ============================================================================
# MacroEconometricTools Integration - LocalProjectionIRFResult Conversion
# ============================================================================

"""
    as_irf_result(lp; term=lp.shock, vcov_estimator=nothing, coverage=[0.68, 0.90, 0.95])

Convert a `LocalProjection` or `LocalProjectionIV` result to MET's `LocalProjectionIRFResult` type.

This enables using MET's unified plotting infrastructure with AxisArray support.
For IV results, first-stage diagnostics are included in metadata.

# Keyword Arguments
- `term::Symbol=lp.shock`: Which coefficient to extract (defaults to shock variable)
- `vcov_estimator=nothing`: Covariance estimator for confidence bands (e.g., `HR1()`)
- `coverage::Vector{Float64}=[0.68, 0.90, 0.95]`: Confidence levels for bands

# Returns
- `LocalProjectionIRFResult`: MET-compatible IRF result with AxisArray data

# Example
```julia
using LocalProjections, MacroEconometricTools, CovarianceMatrices

lp_result = lp(@formula(leads(y) ~ x), df; horizon=12)
irf = as_irf_result(lp_result; vcov_estimator=HR1())

# AxisArray indexing
irf.data[response=:y, shock=:x, horizon=0:6]

# Use with MET's plotting
using Plots
plot(irf)  # Uses MET's recipe
```
"""
function as_irf_result(lp::LPEstimate;
        term::Symbol = lp.shock,
        vcov_estimator = nothing,
        coverage::Vector{Float64} = [0.68, 0.90, 0.95])
    T = Float64

    # Extract coefficient path
    beta = coefpath(lp; term = term)
    H = length(beta)
    horizons = 0:(H - 1)

    # Build AxisArray with named dimensions
    data = AxisArray(
        reshape(beta, 1, 1, H),
        Axis{:response}([lp.response]),
        Axis{:shock}([term]),
        Axis{:horizon}(horizons)
    )

    if vcov_estimator !== nothing
        cov = vcov(vcov_estimator, lp)
        se = stderror(cov; term = term)

        stderr_arr = AxisArray(
            reshape(se, 1, 1, H),
            Axis{:response}([lp.response]),
            Axis{:shock}([term]),
            Axis{:horizon}(horizons)
        )

        # Compute confidence bands as AxisArrays
        # Critical values follow the estimator pairing (Student-t_B for EWC)
        lower = [AxisArray(
                     reshape(beta .- _critical_value(vcov_estimator, c) .* se, 1, 1, H),
                     Axis{:response}([lp.response]),
                     Axis{:shock}([term]),
                     Axis{:horizon}(horizons)
                 ) for c in coverage]

        upper = [AxisArray(
                     reshape(beta .+ _critical_value(vcov_estimator, c) .* se, 1, 1, H),
                     Axis{:response}([lp.response]),
                     Axis{:shock}([term]),
                     Axis{:horizon}(horizons)
                 ) for c in coverage]
    else
        stderr_arr = AxisArray(zeros(T, 1, 1, H),
            Axis{:response}([lp.response]),
            Axis{:shock}([term]),
            Axis{:horizon}(horizons))
        lower = [copy(stderr_arr) for _ in coverage]
        upper = [copy(stderr_arr) for _ in coverage]
    end

    # Build metadata, adding IV-specific fields when applicable
    meta = (horizon = lp.horizon, formula = lp.base_formula, term = term)
    if lp isa BiasCorrectedLP
        meta = (meta..., bias_corrected = true)
    end
    if lp isa LocalProjectionIV
        fs_diagnostics = [first_stage(lp, h) for h in 0:(H - 1)]
        meta = (meta..., is_iv = true, first_stage = fs_diagnostics)
    end

    return LocalProjectionIRFResult{T, typeof(data)}(
        data, stderr_arr, lower, upper, coverage, meta
    )
end

"""
    as_irf_result(b::LPBootstrap; term=b.term, coverage=[0.68, 0.90], method=:hall_t)

Convert a [`LPBootstrap`](@ref) to a `LocalProjectionIRFResult`, carrying the
bootstrap bands rather than normal-approximation ones.

The bands are those of [`summarize`](@ref) under `method`, so they are
asymmetric and **pointwise**. `stderr` holds the real-data HC1 standard errors
of the uncorrected coefficients — the reference convention — which are *not*
what the bands are built from. The metadata records `bootstrap = true` along
with the draw count, block length, VAR lag length, failed draws and which
corrections were applied.
"""
function as_irf_result(b::LPBootstrap;
        term::Symbol = b.term,
        coverage::Vector{Float64} = [0.68, 0.90],
        method::Symbol = :hall_t)
    term === b.term || throw(ArgumentError(
        "this bootstrap was run for term $(b.term), not $term"))
    T = Float64
    beta = coefpath(b; term = term)
    H = length(beta)
    horizons = 0:(H - 1)
    resp = b.response

    _ax(v) = AxisArray(
        reshape(collect(T, v), 1, 1, H),
        Axis{:response}([resp]),
        Axis{:shock}([term]),
        Axis{:horizon}(horizons)
    )

    data = _ax(beta)
    stderr_arr = _ax(b.se)
    bands = [summarize(b; level = c, method = method) for c in coverage]
    lower = [_ax(s.lower) for s in bands]
    upper = [_ax(s.upper) for s in bands]

    meta = (horizon = H - 1,
        formula = b.base_formula,
        term = term,
        bootstrap = true,
        bootstrap_method = method,
        nboot = b.nboot,
        nfail = b.nfail,
        blocklength = b.blocklength,
        var_lags = b.nlags,
        var_variables = b.vars,
        bias_corrected = b.biascorrected,
        pope_corrected = b.popecorrected,
        pope_delta = b.pope_delta)

    return LocalProjectionIRFResult{T, typeof(data)}(
        data, stderr_arr, lower, upper, coverage, meta
    )
end

# Shared body of the two FEVD conversions: `summaries` are FEVDSummary bands,
# one per coverage level, with a common centre and standard error.
function _fevd_irf_result(summaries::Vector{FEVDSummary}, coverage::Vector{Float64},
        meta::NamedTuple)
    T = Float64
    s1 = summaries[1]
    H = length(s1.horizon)
    _ax(v) = AxisArray(
        reshape(collect(T, v), 1, 1, H),
        Axis{:response}([s1.response]),
        Axis{:shock}([s1.shock]),
        Axis{:horizon}(0:(H - 1))
    )
    data = _ax(s1.estimate)
    lower = [_ax(s.lower) for s in summaries]
    upper = [_ax(s.upper) for s in summaries]
    return LocalProjectionIRFResult{T, typeof(data)}(
        data, _ax(s1.se), lower, upper, coverage, meta
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
function as_irf_result(f::LPFEVD;
        coverage::Vector{Float64} = [0.68, 0.90],
        prewhiten::Bool = true)
    summaries = [summarize(f; level = c, prewhiten = prewhiten) for c in coverage]
    meta = (horizon = f.horizon,
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
        bias_corrected = false)
    return _fevd_irf_result(summaries, coverage, meta)
end

"""
    as_irf_result(b::FEVDBootstrap; coverage=[0.68, 0.90], band=:shift,
                  biascorrected=true, prewhiten=true)

Convert a bias-corrected Gorodnichenko–Lee FEVD ([`FEVDBootstrap`](@ref)) to a
`LocalProjectionIRFResult`. `data` holds the centre of the bands of
[`summarize`](@ref) under the same keywords — the bias-corrected shares by
default — and `stderr` their standard error (the standard deviation of the
draws for `band = :shift`). The metadata records `fevd = true`,
`quantity = :fevd_share`, `bootstrap = true`, the procedure
(`bootstrap_method`), the band, the uncorrected shares (`share`), the VAR
pseudo-truth for `:var`, and the draw counts.
"""
function as_irf_result(b::FEVDBootstrap;
        coverage::Vector{Float64} = [0.68, 0.90],
        band::Symbol = :shift,
        biascorrected::Bool = true,
        prewhiten::Bool = true)
    summaries = [summarize(b; level = c, band, biascorrected, prewhiten)
                 for c in coverage]
    f = b.fevd
    meta = (horizon = f.horizon,
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
        var_variables = b.vars)
    return _fevd_irf_result(summaries, coverage, meta)
end
