# VAR residual moving-block bootstrap with Hall percentile-t bands (guide §4),
# ported from the andrerecio fork onto 0.2.

@testitem "moving-block resampling matches var_boot.m" tags = [:bootstrap, :verification] begin
    using LocalProjections
    using LinearAlgebra, StableRNGs, Statistics, Test

    rng = StableRNG(202)
    Tu, n = 97, 3
    U = randn(rng, Tu, n)
    for t in 2:Tu
        U[t, :] .+= 0.4 .* U[t - 1, :]
    end
    ℓ = 12

    # var_boot.m computes the position means with
    #   filter(ones(1, Tu-l+1)/(Tu-l+1), 1, res), rows end-l+1:end
    # which is the SLIDING-WINDOW mean over the Tu-l+1 possible block starts.
    means = LocalProjections._position_means(U, ℓ)
    @test size(means) == (ℓ, n)
    for s in 1:ℓ, j in 1:n
        @test means[s, j] ≈ mean(U[s:(Tu - ℓ + s), j]) rtol = 1.0e-12
    end

    # It is NOT the stride-l mean used by MacroEconometricTools.bootstrap_irf_block.
    stride_means = [mean(U[s:ℓ:(Tu - ℓ + s), j]) for s in 1:ℓ, j in 1:n]
    @test !isapprox(means, stride_means; rtol = 1.0e-3)

    # Block layout with fixed starts is deterministic and matches the reference.
    starts = [0, 5, 40, 60, 17, 33, 80, 2, 51]   # cld(97, 12) == 9 blocks
    @test length(starts) == cld(Tu, ℓ)
    dest = Matrix{Float64}(undef, Tu, n)
    LocalProjections._mbb_residuals!(dest, U, means, starts)
    for (b, off) in enumerate(starts), s in 1:ℓ
        row = (b - 1) * ℓ + s
        row > Tu && continue
        for j in 1:n
            @test dest[row, j] ≈ U[off + s, j] - means[s, j] rtol = 1.0e-12
        end
    end

    # Recentering makes the resampled residuals mean-zero in expectation: averaging
    # over every admissible start at a fixed position gives exactly zero.
    for s in 1:ℓ, j in 1:n
        @test abs(mean(U[(0:(Tu - ℓ)) .+ s, j] .- means[s, j])) < 1.0e-10
    end
end

@testitem "Hall percentile-t intervals match boot_ci.m" tags = [:bootstrap, :verification] begin
    using LocalProjections
    using DataFrames, StableRNGs, Statistics, Test

    rng = StableRNG(20260901)
    T = 220
    sh = zeros(T)
    y = zeros(T)
    for t in 2:T
        sh[t] = 0.3 * sh[t - 1] + randn(rng)
        y[t] = 0.6 * y[t - 1] + 0.8 * sh[t] + randn(rng)
    end
    df = DataFrame(shock = sh, y = y)
    m = lp(@formula(leads(y) ~ shock + lags(y, 2) + lags(shock, 2)), df; horizon = 6)
    b = varbootstrap(m, df; vars = [:shock, :y], nlags = 2, nboot = 300, rng = StableRNG(77))

    level = 0.9
    α = 1 - level

    # Direct port of _estim/boot_ci.m for all three constructions.
    for h in 1:(b.horizon + 1)
        θs = [x for x in b.theta_boot[:, h] if isfinite(x)]
        ts = [
            (b.theta_boot[i, h] - b.pseudo_truth[h]) / b.se_boot[i, h]
                for i in 1:b.nboot
                if isfinite(b.theta_boot[i, h]) && isfinite(b.se_boot[i, h]) && b.se_boot[i, h] > 0
        ]
        ql, qu = quantile(θs, α / 2), quantile(θs, 1 - α / 2)
        tl, tu = quantile(ts, α / 2), quantile(ts, 1 - α / 2)

        efron = (ql, qu)
        hall = (b.theta[h] + b.pseudo_truth[h] - qu, b.theta[h] + b.pseudo_truth[h] - ql)
        # Note the quantile reversal on the studentized statistic.
        hall_t = (b.theta[h] - b.se[h] * tu, b.theta[h] - b.se[h] * tl)

        se_ = summarize(b; level = level, method = :efron)
        sh_ = summarize(b; level = level, method = :hall)
        st_ = summarize(b; level = level, method = :hall_t)
        @test (se_.lower[h], se_.upper[h]) == efron
        @test (sh_.lower[h], sh_.upper[h]) == hall
        @test (st_.lower[h], st_.upper[h]) == hall_t
    end

    # The three constructions genuinely differ.
    st = summarize(b; level = level, method = :hall_t)
    sh_ = summarize(b; level = level, method = :hall)
    se_ = summarize(b; level = level, method = :efron)
    @test st.lower != sh_.lower
    @test st.lower != se_.lower
    @test sh_.lower != se_.lower

    # Percentile-t bands are asymmetric around the point estimate.
    @test !isapprox(st.upper .- st.coef, st.coef .- st.lower; rtol = 1.0e-6)

    # A negative scale flips the sign and keeps lower <= upper
    sneg = summarize(b; level = level, scale = -1)
    @test sneg.coef == -st.coef && sneg.lower == -st.upper && sneg.upper == -st.lower
end

@testitem "varbootstrap end-to-end conventions" tags = [:bootstrap, :api] begin
    using LocalProjections
    using DataFrames, StableRNGs, Statistics, StatsBase, Test
    using CovarianceMatrices: HC1, Bartlett

    rng = StableRNG(20260902)
    T = 240
    sh = zeros(T)
    y = zeros(T)
    for t in 2:T
        sh[t] = 0.3 * sh[t - 1] + randn(rng)
        y[t] = 0.6 * y[t - 1] + 0.8 * sh[t] + randn(rng)
    end
    df = DataFrame(shock = sh, y = y)
    m = lp(@formula(leads(y) ~ shock + lags(y, 4) + lags(shock, 4)), df; horizon = 10)

    b = varbootstrap(m, df; vars = [:shock, :y], nlags = 4, nboot = 200, rng = StableRNG(1))
    @test b isa LPBootstrap
    @test b.nboot == 200
    @test b.nfail == 0
    @test b.biascorrected && b.popecorrected
    @test b.blocklength == ceil(Int, 5.03 * T^(1 / 4))
    @test size(b.theta_boot) == (200, 11)
    @test size(b.se_boot) == (200, 11)

    # Reported path is the bias-corrected one; SEs are those of the UNCORRECTED
    # coefficients (the reference never recomputes them), always HC1.
    @test b.theta ≈ coefpath(biascorrect(m); term = :shock) rtol = 1.0e-12
    @test b.se ≈ stderror(m + vcov(HC1()); term = :shock) rtol = 1.0e-12
    @test !isapprox(b.theta, coefpath(m; term = :shock); rtol = 1.0e-8)
    bk = varbootstrap(m + vcov(Bartlett(4)), df; vars = [:shock, :y], nlags = 4, nboot = 100, rng = StableRNG(1))
    @test bk.se == b.se

    # Property forwarding to the inner LP.
    @test b.horizon == 10
    @test b.shock == :shock
    @test b.response == :y
    @test coefpath(b) == b.theta

    # Reproducibility: identical output threaded and unthreaded.
    b_serial = varbootstrap(
        m, df; vars = [:shock, :y], nlags = 4, nboot = 100, rng = StableRNG(42), threaded = false
    )
    b_thread = varbootstrap(
        m, df; vars = [:shock, :y], nlags = 4, nboot = 100, rng = StableRNG(42), threaded = true
    )
    @test b_serial.theta_boot == b_thread.theta_boot
    @test b_serial.se_boot == b_thread.se_boot

    # Any Tables.jl table gives the same draws
    b_nt = varbootstrap(
        m, (shock = sh, y = y); vars = [:shock, :y], nlags = 4, nboot = 100, rng = StableRNG(42)
    )
    @test b_nt.theta_boot == b_serial.theta_boot

    # Script-23 variant: no Pope, no Herbst-Johannsen.
    b23 = varbootstrap(
        m, df; vars = [:shock, :y], nlags = 4, nboot = 100,
        rng = StableRNG(1), biascorrect = false, popecorrect = false
    )
    @test !b23.biascorrected && !b23.popecorrected
    @test isnan(b23.pope_delta)
    @test b23.theta ≈ coefpath(m; term = :shock) rtol = 1.0e-12
    @test !isapprox(b23.theta, b.theta; rtol = 1.0e-8)

    # sample = :fixed is re-estimated per draw, with biascorrect = false
    mf = lp(@formula(leads(y) ~ shock + lags(y, 4) + lags(shock, 4)), df; horizon = 10, sample = :fixed)
    bf = varbootstrap(mf, df; vars = [:shock, :y], nlags = 4, nboot = 100, rng = StableRNG(1), biascorrect = false)
    @test bf.theta ≈ coefpath(mf) rtol = 1.0e-12
    @test all(isfinite, bf.theta_boot)

    # summarize returns a usable IRFSummary with pointwise bands.
    s = summarize(b; level = 0.9)
    @test s isa IRFSummary
    @test s.level == 0.9
    @test s.response === :y && s.term === :shock
    @test length(s.lower) == 11
    # Ordering is guaranteed (q_{1-a/2} >= q_{a/2}); containment of the point
    # estimate is NOT: a Hall percentile-t interval corrects for bias, so it can
    # sit entirely to one side of theta-hat when the bootstrap t-distribution is
    # shifted. Do not assert containment here.
    @test all(s.lower .<= s.upper)
    @test DataFrame(s) isa DataFrame

    # Wider level gives wider bands.
    s68 = summarize(b; level = 0.68)
    @test all(s68.upper .- s68.lower .<= s.upper .- s.lower .+ 1.0e-12)

    # Pseudo-truth is the VAR-implied response, not the LP path.
    @test !isapprox(b.pseudo_truth, b.theta; rtol = 1.0e-6)
    @test length(b.pseudo_truth) == 11

    io = IOBuffer()
    show(io, b)
    @test occursin("LPBootstrap", String(take!(io)))
    show(io, MIME"text/plain"(), b)
    out = String(take!(io))
    @test occursin("moving-block", out)
    @test occursin("Herbst", out)
    @test occursin("Pope", out)
end

@testitem "varbootstrap plotting and as_irf_result" tags = [:bootstrap, :api] begin
    using LocalProjections
    using DataFrames, StableRNGs, Test
    import Plots, CairoMakie
    using MacroEconometricTools: LocalProjectionIRFResult

    rng = StableRNG(20260903)
    T = 200
    sh = randn(rng, T)
    y = zeros(T)
    for t in 2:T
        y[t] = 0.5 * y[t - 1] + 0.9 * sh[t] + randn(rng)
    end
    df = DataFrame(shock = sh, y = y)
    m = lp(@formula(leads(y) ~ shock + lags(y, 2)), df; horizon = 8)
    b = varbootstrap(m, df; vars = [:shock, :y], nlags = 2, nboot = 150, rng = StableRNG(3))

    # Plots recipe: single and multiple levels, and each interval method.
    @test Plots.plot(b) isa Plots.Plot
    @test Plots.plot(b; levels = [0.68, 0.9]) isa Plots.Plot
    @test Plots.plot(b; levels = [0.9], method = :efron) isa Plots.Plot
    @test Plots.plot(b; levels = [0.9], method = :hall) isa Plots.Plot
    @test_throws ArgumentError Plots.plot(b; levels = [1.5])
    @test_throws ArgumentError Plots.plot(b; method = :bogus)

    # Makie
    @test irfplot(b; levels = [0.68, 0.9]) isa CairoMakie.Figure
    fig = CairoMakie.Figure()
    ax = CairoMakie.Axis(fig[1, 1])
    @test (irfplot!(ax, b; levels = [0.9], method = :efron); true)

    r = as_irf_result(b; coverage = [0.68, 0.9])
    @test r isa LocalProjectionIRFResult
    @test r.coverage == [0.68, 0.9]
    @test r.metadata.bootstrap == true
    @test r.metadata.bootstrap_method == :hall_t
    @test r.metadata.nboot == 150
    @test r.metadata.bias_corrected == true
    @test r.metadata.pope_corrected == true
    @test r.metadata.var_lags == 2
    @test vec(r.data[1, 1, :]) ≈ b.theta rtol = 1.0e-12

    # Bands carried over are the bootstrap ones, and asymmetric.
    s90 = summarize(b; level = 0.9)
    @test vec(r.lower[2][1, 1, :]) ≈ s90.lower rtol = 1.0e-12
    @test vec(r.upper[2][1, 1, :]) ≈ s90.upper rtol = 1.0e-12

    @test_throws ArgumentError as_irf_result(b; term = :y)
end

@testitem "varbootstrap guards and edge cases" tags = [:bootstrap, :api] begin
    using LocalProjections
    using DataFrames, StableRNGs, Test

    rng = StableRNG(20260904)
    T = 180
    sh = randn(rng, T)
    y = zeros(T)
    z = randn(rng, T)
    for t in 2:T
        y[t] = 0.5 * y[t - 1] + 0.9 * sh[t] + randn(rng)
    end
    df = DataFrame(
        shock = sh, y = y, z = z, s = Float64.(z .> 0),
        id = repeat(1:2, inner = 90), t = repeat(1:90, 2)
    )
    f = @formula(leads(y) ~ shock + lags(y, 2))
    m = lp(f, df; horizon = 5)
    kw = (; vars = [:shock, :y], nlags = 2, nboot = 5)

    # The shock must be in the VAR data vector.
    @test_throws "is not in `vars`" varbootstrap(m, df; vars = [:y], nlags = 2, nboot = 5)
    # Every variable the LP formula needs must be simulated.
    @test_throws "is not in `vars`" varbootstrap(m, df; vars = [:shock], nlags = 2, nboot = 5)
    mz = lp(@formula(leads(y) ~ shock + z), df; horizon = 2)
    @test_throws "must be simulated by the VAR" varbootstrap(mz, df; kw...)
    # Block length must fit the residual sample.
    @test_throws "`blocklength`" varbootstrap(m, df; kw..., blocklength = 10_000)
    @test_throws "`nboot` must be at least 1" varbootstrap(m, df; vars = [:shock, :y], nlags = 2, nboot = 0)
    # The low-draw warning is part of the contract (guide 5 checklist).
    b = @test_logs (:warn, r"usable draws") match_mode = :any varbootstrap(
        m, df; vars = [:shock, :y], nlags = 2, nboot = 50, rng = StableRNG(8)
    )
    @test_throws ArgumentError summarize(b; method = :bogus)
    @test_throws ArgumentError summarize(b; level = 1.5)
    @test_throws ArgumentError coefpath(b; term = :y)

    # missing / NaN in the VAR columns are rejected: the VAR needs a gap-free series.
    dfm = copy(df)
    dfm.y = Vector{Union{Missing, Float64}}(dfm.y)
    dfm.y[20] = missing
    @test_throws "contains `missing`" varbootstrap(m, dfm; kw...)
    dfn = copy(df)
    dfn.y[20] = NaN
    @test_throws "contains `NaN`" varbootstrap(m, dfn; kw...)

    # Unsupported responses and settings
    ma = lp(@formula(leads(y) | shock ~ shock + lags(y, 2)), df; horizon = 4)
    @test_throws "anchored response" varbootstrap(ma, df; kw...)
    ml = lp(@formula(ldiff(y) ~ shock + lags(firstdiff(y), 2)), df; horizon = 3)
    @test_throws "long-difference" varbootstrap(ml, df; kw...)
    mt = lp(@formula(leads(y) ~ shock + cumul(shock) + lags(y, 2)), df; horizon = 4, shock = :shock)
    @test_throws "horizon-tracking" varbootstrap(mt, df; kw...)
    @test_throws "state-dependent" varbootstrap(lp(f, df; horizon = 2, state = :s), df; kw...)
    @test_throws "without `unit` and `time`" varbootstrap(lp(f, df; horizon = 2, unit = :id, time = :t), df; kw...)
    @test_throws "fixed effects" varbootstrap(lp(@formula(leads(y) ~ shock + fe(id)), df; horizon = 2), df; vars = [:shock, :y, :id], nlags = 2, nboot = 5)
    @test_throws "sample = :fixed" varbootstrap(lp(f, df; horizon = 4, sample = :fixed), df; kw...)

    # IV local projections are rejected, as for biascorrect.
    dfiv = copy(df)
    dfiv.x = 0.7 .* z .+ randn(StableRNG(9), T)
    miv = lpiv(TSLS(), @formula(leads(y) ~ (x ~ z) + lags(y, 2)), dfiv; horizon = 4)
    @test_throws "not available for LocalProjectionIV" varbootstrap(miv, dfiv; vars = [:x, :y], nlags = 2, nboot = 5)

    # cumul responses are supported and the pseudo-truth is cumulated.
    mc = lp(@formula(cumul(y) ~ shock + lags(y, 2)), df; horizon = 6)
    bc = varbootstrap(mc, df; vars = [:shock, :y], nlags = 2, nboot = 120, rng = StableRNG(10))
    bl = varbootstrap(
        lp(@formula(leads(y) ~ shock + lags(y, 2)), df; horizon = 6),
        df; vars = [:shock, :y], nlags = 2, nboot = 120, rng = StableRNG(10)
    )
    @test bc.pseudo_truth ≈ cumsum(bl.pseudo_truth) rtol = 1.0e-12
    @test all(isfinite, summarize(bc).lower)

    # Tautological h = 0 (response === shock) has a degenerate band, not a NaN.
    mtt = lp(@formula(leads(shock) ~ shock + lags(y, 2)), df; horizon = 5)
    bt = varbootstrap(mtt, df; vars = [:shock, :y], nlags = 2, nboot = 120, rng = StableRNG(12))
    st = summarize(bt; level = 0.9)
    @test st.coef[1] == 1.0
    @test st.se[1] == 0.0
    @test st.lower[1] == 1.0 && st.upper[1] == 1.0
    @test all(isfinite, st.lower) && all(isfinite, st.upper)
end
