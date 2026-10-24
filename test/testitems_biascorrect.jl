# Herbst–Johannsen bias correction (guide §3), ported from the andrerecio fork
# onto the 0.2 stored-estimator API.

@testitem "Herbst–Johannsen correction matches direct lp_biascorr.m port" tags = [
    :biascorr, :verification,
] begin
    using LocalProjections
    using DataFrames, LinearAlgebra, Statistics, Test, StableRNGs
    using StatsBase: modelmatrix

    rng = StableRNG(20260830)
    n = 200
    x = randn(rng, n)
    y = zeros(n)
    for t in 2:n
        y[t] = 0.6 * y[t - 1] + 0.5 * x[t] + randn(rng)
    end
    df = DataFrame(y = y, x = x)
    lp_result = lp(@formula(leads(y) ~ x + lags(y, 2)), df; horizon = 8)
    bc = biascorrect(lp_result)

    # Direct port of lp_biascorr.m (guide §3) on the same inputs
    keep = [!(nm in ("(Intercept)", "x")) for nm in lp_result.coef_names]
    w = modelmatrix(horizonmodel(lp_result, 0))[:, keep]
    irs = coefpath(lp_result; term = :x)
    T = size(w, 1)
    H = lp_result.horizon
    wd = w .- mean(w; dims = 1)
    S0 = (wd' * wd) / (T - 1)
    acf = [1 + tr(S0 \ ((wd[1:(end - j), :]' * wd[(j + 1):end, :]) / (T - j - 1))) for j in 1:H]
    irs_c = copy(irs)
    for h in 1:H
        irs_c[h + 1] = irs[h + 1] + (1 / (T - h)) * sum(acf[j] * irs_c[h - j + 1] for j in 1:h)
    end

    @test coefpath(bc) ≈ irs_c
    @test bc.c ≈ acf
    @test bc.T0 == T
    @test bc.controls == ["y_lag1", "y_lag2"]
    # Impact response is untouched; later horizons genuinely move
    @test coefpath(bc)[1] == irs[1]
    @test maximum(abs.(coefpath(bc)[2:end] .- irs[2:end])) > 0
end

@testitem "biascorrect conventions in summarize/as_irf_result" tags = [
    :biascorr, :summarize, :api,
] begin
    using LocalProjections
    using DataFrames, StatsBase, Test, StableRNGs
    using CovarianceMatrices: EWC, HC1, Bartlett
    using Distributions: TDist, quantile
    using MacroEconometricTools: point_estimate

    rng = StableRNG(20260830)
    n = 200
    x = randn(rng, n)
    y = zeros(n)
    for t in 2:n
        y[t] = 0.6 * y[t - 1] + 0.5 * x[t] + randn(rng)
    end
    df = DataFrame(y = y, x = x)
    lp_result = lp(@formula(leads(y) ~ x + lags(y, 2)), df; horizon = 8)
    bc = biascorrect(lp_result)

    # Wrapped-field forwarding and the shared AbstractLocalProjection methods
    @test bc isa AbstractLocalProjection
    @test bc.horizon == lp_result.horizon
    @test bc.shock === :x && bc.response === :y
    @test nobs(bc) == nobs(lp_result)
    @test coefnames(bc) == coefnames(lp_result)
    @test stderror(bc) == stderror(lp_result)

    # Script-22 convention: bands centered on θ̂ᶜ with SEs of the
    # uncorrected OLS coefficients
    s_bc = summarize(bc, HC1(); level = 0.9)
    s_lp = summarize(lp_result, HC1(); level = 0.9)
    @test s_bc.coef ≈ coefpath(bc)
    @test s_bc.se == s_lp.se
    @test (s_bc.upper .- s_bc.coef) ≈ (s_lp.upper .- s_lp.coef)
    # the stored estimator of the wrapped projection is used (HC1 by default)
    @test summarize(bc; level = 0.9).se == s_lp.se

    # Estimator/critical-value pairing flows through (t_B for EWC)
    B = ewc_bandwidth(bc)
    s_ewc = summarize(bc, EWC(B); level = 0.9)
    tcrit = quantile(TDist(B), 0.95)
    @test s_ewc.upper ≈ s_ewc.coef .+ tcrit .* s_ewc.se

    # bc + vcov keeps the correction, changes only the attached estimator
    bc2 = bc + vcov(Bartlett(4))
    @test bc2 isa BiasCorrectedLP
    @test bc2.c == bc.c && coefpath(bc2) ≈ coefpath(bc)
    @test stderror(bc2) == stderror(lp_result + vcov(Bartlett(4)))

    # as_irf_result centers on the corrected path and flags the metadata
    irf = as_irf_result(bc; vcov_estimator = HC1(), coverage = [0.9])
    @test vec(Array(point_estimate(irf))) ≈ coefpath(bc)
    @test irf.metadata.bias_corrected === true
    @test !haskey(as_irf_result(lp_result).metadata, :bias_corrected)

    show_txt = sprint(show, MIME("text/plain"), bc)
    @test occursin("Herbst–Johannsen", show_txt) && occursin("y_lag1", show_txt)
end

@testitem "biascorrect guards and edge cases" tags = [:biascorr, :api] begin
    using LocalProjections
    using DataFrames, Test, StableRNGs

    rng = StableRNG(20260830)
    n = 200
    x = randn(rng, n)
    y = zeros(n)
    for t in 2:n
        y[t] = 0.6 * y[t - 1] + 0.5 * x[t] + randn(rng)
    end
    df = DataFrame(y = y, x = x, s = Float64.(x .> 0), id = repeat(1:2, inner = 100), t = repeat(1:100, 2))
    f = @formula(leads(y) ~ x + lags(y, 2))

    # Derived for OLS LP only
    iv_result = lpiv(TSLS(), @formula(leads(y) ~ (x ~ x) + lags(y, 1)), df; horizon = 2)
    @test_throws "not available for LocalProjectionIV" biascorrect(iv_result)

    # Correction is defined for the shock's impulse response only
    bc = biascorrect(lp(f, df; horizon = 4))
    @test_throws "use coefpath(bc.lp; term = :y_lag1)" coefpath(bc; term = :y_lag1)

    # Internal NaN gap → horizon-h sample is no longer the truncated
    # horizon-0 sample (guide §3.2: T − h must be checked, not copied)
    df_gap = copy(df)
    df_gap.y[100] = NaN
    @test_throws "not the horizon-0 sample" biascorrect(lp(f, df_gap; horizon = 4))
    # ... and the same for a `missing`, which 0.2 keeps in place as a gap
    df_mis = copy(df)
    df_mis.y = Vector{Union{Missing, Float64}}(df_mis.y)
    df_mis.y[100] = missing
    @test_throws "not the horizon-0 sample" biascorrect(lp(f, df_mis; horizon = 4))

    # Collinear controls → Σ̂₀ (near-)singular
    df_col = copy(df)
    df_col.z = 2 .* df.y
    lp_col = lp(@formula(leads(y) ~ x + lags(y, 1) + lags(z, 1)), df_col; horizon = 4)
    @test_throws "(near-)singular" biascorrect(lp_col)

    # Tautological h=0 (response == shock): θ̂₀ᶜ stays pinned at 1
    lp_taut = lp(@formula(leads(x) ~ x + lags(y, 2)), df; horizon = 4)
    @test lp_taut.tautological_h0
    @test coefpath(biascorrect(lp_taut))[1] == 1.0

    # No controls beyond intercept + shock: ĉ_j ≡ 1, recursion still runs
    lp_min = lp(@formula(leads(y) ~ x), df; horizon = 4)
    bc_min = biascorrect(lp_min)
    @test all(bc_min.c .== 1.0)
    @test coefpath(bc_min)[1] == coefpath(lp_min)[1]

    # Settings the correction is not derived for
    @test_throws "sample = :fixed" biascorrect(lp(f, df; horizon = 4, sample = :fixed))
    @test_throws "state-dependent" biascorrect(lp(f, df; horizon = 2, state = :s))
    @test_throws "panel local projection" biascorrect(lp(f, df; horizon = 2, unit = :id, time = :t))
    @test_throws "fixed effects" biascorrect(lp(@formula(leads(y) ~ x + fe(id)), df; horizon = 2))
    @test_throws "horizon-tracking" biascorrect(lp(@formula(leads(y) ~ x + cumul(x) + lags(y, 2)), df; horizon = 4, shock = :x))
    @test_throws "long-difference" biascorrect(lp(@formula(ldiff(y) ~ x + lags(firstdiff(y), 2)), df; horizon = 3))
    @test_throws "long-difference" biascorrect(lp(@formula(cumul(hbr(y, x)) ~ x), DataFrame(y = 10 .+ y, x = 5 .+ x); horizon = 2))
end
