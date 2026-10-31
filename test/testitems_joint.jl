# Joint inference across horizons and the implicit-shock variance decomposition,
# from lectures 1 and 3 of Plagborg-Møller's local-projection course (2026).

@testmodule CourseData begin
    using LocalProjections
    using StatsModels: @formula

    function readtable(path)
        lines = readlines(path)
        hdr = Symbol.(split(lines[1], ","))
        cols = [Float64[] for _ in hdr]
        for l in lines[2:end], (j, v) in enumerate(split(l, ","; keepempty = true))
            push!(cols[j], isempty(v) ? NaN : parse(Float64, v))
        end
        return NamedTuple{Tuple(hdr)}(Tuple(cols))
    end

    const bk = readtable(joinpath(@__DIR__, "..", "docs", "src", "data", "bilal_kanzig.csv"))

    # settings.do: L(0/2).recessiondates L(1/2).(y x oil treasury), horizons 0-10
    const formula = @formula(
        leads(lnrgdppc_world_pwt) ~ gtmp_bkly_aw_dtfe2s + recessiondates +
            lags(recessiondates, 2) + lags(lnrgdppc_world_pwt, 2) +
            lags(gtmp_bkly_aw_dtfe2s, 2) + lags(lnpoil_wti, 2) + lags(treasury1y, 2)
    )
    model() = lp(formula, bk; horizon = 10)
end

@testitem "Course FEVD and suest reproduce Stata" tags = [:joint, :verification] setup = [
    CourseData,
] begin
    using LocalProjections, LinearAlgebra, Test
    using CovarianceMatrices: HC0, HC1
    m = CourseData.model()
    @test nobs(m) == collect(58:-1:48)

    # Reference: fork stata_test/lp_course/course_check.do (StataNow 19.5), the
    # commands of lecture1.do and lecture3.do with residuals stored as double.
    # Variance decomposition: reg x controls; predict resid; per h, reg F`h'.y
    # controls; predict res; reg res F(0/`h').resid; e(r2)
    r2 = [
        0.008201821727452985, 0.026707978736787874, 0.08512454437910177,
        0.10523099348585685, 0.18455619400706835, 0.3068270800430123,
        0.44917978906123746, 0.49159722060512523, 0.46031884005625956,
        0.45985882322669525, 0.43832824462791553,
    ]
    f = implicitfevd(m)
    @test f.share ≈ r2 rtol = 1.0e-10
    @test f.nobs == collect(58:-1:48)
    @test f.shock_nobs == 58
    x̃ = implicitshock(m)
    @test all(isnan, x̃[1:2])
    @test x̃[[3, 31, 60]] ≈ [-0.01990895331311761, 0.03894602677006376, -0.044749169551179024] rtol = 1.0e-9
    @test isequal(x̃, f.implicit_shock)

    # regress, robust (HC1) standard errors at h = 0, 5, 10
    @test stderror(m)[[1, 6, 11]] ≈ [1.300066290611318, 3.815385862188192, 5.206397380628676] rtol = 1.0e-9

    # suest lp*, vce(robust); test [lp0_mean]x ... [lp10_mean]x, matvlc(V):
    # HC0 scores of every horizon stacked, one factor N/(N-1), N = 58
    N = 58
    V = jointvcov(m, HC0()) .* (N / (N - 1))
    @test diag(V) ≈ [
        1.3343466000124997, 2.654783226969278, 3.6150183222343717, 4.702019765889592,
        7.3861600066555955, 11.179289216716256, 16.1943908689193, 18.349529050120985,
        19.152143190546084, 27.036588550715358, 20.11196804845917,
    ] rtol = 1.0e-9
    # The design is ill-conditioned (cond(X'X) ≈ 2e10): compare on the matrix scale
    @test V[1, 2] ≈ 1.090121073398204 atol = 1.0e-8
    @test V[5, 9] ≈ 4.705847537141558 atol = 1.0e-8
    @test V[1, 11] ≈ -0.5272783699447325 atol = 1.0e-8
    @test V[10, 11] ≈ 21.23680685269526 atol = 1.0e-8

    t = jointtest(m, HC0())
    @test t.df == 11
    @test t.distribution === :chisq
    @test t.wald * (N - 1) / N ≈ 30.31615522231833 rtol = 1.0e-9   # chi2(11) = 30.32
    @test t.horizons == collect(0:10)

    # The default (HC1) keeps each horizon's own n/(n-k), so W is smaller
    t1 = jointtest(m)
    @test t1.estimator == HC1()
    @test t1.wald < t.wald
end

@testitem "sup-t critical values" tags = [:joint, :supt] setup = [CourseData] begin
    using LocalProjections, LinearAlgebra, Random, Test
    using LocalProjections: _supt_from_draws, _supt_quantile
    using CovarianceMatrices: HC0, EWC, Bartlett
    using Distributions: Normal, TDist, quantile
    m = CourseData.model()

    # Kessler's quantile rule: average at an integer k = level·n, else ⌊k⌋+1
    @test _supt_quantile(collect(1.0:10.0), 0.5) == 5.5
    @test _supt_quantile(collect(1.0:10.0), 0.55) == 6.0
    # max |z_h| / sd_h per row
    draws = [1.0 -4.0; -3.0 2.0; 0.5 0.5]
    @test _supt_from_draws(draws, [1.0, 2.0], 0.5) == 2.0

    # estimate_supt_critical_value with 1e6 drawnorm draws (seed 20261031) on the
    # suest matrix: 1.876955 (68%), 2.424262 (90%), 2.689406 (95%). The critical
    # value depends only on the correlation, which every HC estimator shares.
    @test supt_critical_value(m; level = 0.68) ≈ 1.876955 atol = 0.01
    @test supt_critical_value(m; level = 0.9) ≈ 2.424262 atol = 0.01
    @test supt_critical_value(m, HC0(); level = 0.95) ≈ 2.689406 atol = 0.015
    # reproducible with the default generator, and an explicit one is honoured
    @test supt_critical_value(m) == supt_critical_value(m)
    @test supt_critical_value(m; rng = Xoshiro(1), ndraws = 2000) ==
        supt_critical_value(m; rng = Xoshiro(1), ndraws = 2000)

    # between the pointwise and the Bonferroni critical values
    for level in (0.68, 0.9)
        c = supt_critical_value(m; level)
        @test c > quantile(Normal(), 0.5 + level / 2)
        @test c < quantile(Normal(), 1 - (1 - level) / (2 * 11))
    end
    # one horizon: the pointwise critical value of the estimator
    @test supt_critical_value(m; horizons = [4], level = 0.9) ≈ quantile(Normal(), 0.95)
    @test supt_critical_value(m, EWC(20); horizons = [4], level = 0.9) ≈ quantile(TDist(20), 0.95)
    # EWC: fixed-b draws, wider than the Gaussian sup-t on the same correlation
    @test supt_critical_value(m, EWC(10); level = 0.9) > supt_critical_value(m, HC0(); level = 0.9)

    # summarize / as_irf_result with bands = :supt
    c = supt_critical_value(m; level = 0.68)
    s = summarize(m; level = 0.68, bands = :supt)
    @test s.lower ≈ coefpath(m) .- c .* stderror(m)
    @test s.upper ≈ coefpath(m) .+ c .* stderror(m)
    sp = summarize(m; level = 0.68)
    @test all(s.upper .- s.lower .> sp.upper .- sp.lower)
    irf = as_irf_result(m; coverage = [0.68], bands = :supt)
    @test irf.metadata.bands === :supt
    @test summarize(m, Bartlett(3); level = 0.9, bands = :supt).lower ≈
        summarize(m + vcov(Bartlett(3)); level = 0.9, bands = :supt).lower
    @test_throws ArgumentError summarize(m; bands = :bonferroni)
end

@testitem "jointvcov conventions" tags = [:joint] setup = [CourseData] begin
    using LocalProjections, LinearAlgebra, Test
    using CovarianceMatrices: HC0, HC1, HC3, EWC, Bartlett, CR1
    using StatsModels: @formula, coefnames
    m = CourseData.model()
    j = findfirst(==("gtmp_bkly_aw_dtfe2s"), coefnames(m))

    # the diagonal is the stored estimator's squared standard errors
    for est in (HC0(), HC1(), HC3(), Bartlett(3), EWC(12))
        V = jointvcov(m, est)
        @test diag(V) ≈ stderror(m + vcov(est)) .^ 2 rtol = 1.0e-10
        @test issymmetric(V)
        @test isposdef(Symmetric(V))
    end
    # HC0: the stacked sandwich itself, so a block is the per-horizon vcov
    V0 = jointvcov(m, HC0())
    m0 = m + vcov(HC0())
    @test V0[4, 4] ≈ vcov(horizonmodel(m0, 3))[j, j] rtol = 1.0e-12
    # HC1 and HC0 share the correlation; the scales are each horizon's n/(n-k)
    V1 = jointvcov(m)
    c = [nobs(m)[h] / (nobs(m)[h] - length(coefnames(m))) for h in 1:11]
    @test V1 ≈ V0 .* sqrt.(c * c') rtol = 1.0e-10
    # a subset of horizons is the corresponding block
    @test jointvcov(m; horizons = [2, 5, 7]) ≈ V1[[3, 6, 8], [3, 6, 8]] rtol = 1.0e-12
    # another term
    @test size(jointvcov(m; term = :recessiondates)) == (11, 11)

    # Errors
    @test_throws ArgumentError jointvcov(m; horizons = 0:11)
    @test_throws ArgumentError jointvcov(m; horizons = Int[])
    @test_throws ArgumentError jointvcov(m; horizons = [1, 1])
    @test_throws ArgumentError jointvcov(m + vcov(CR1(:year)))
    @test_throws ArgumentError jointtest(m; value = [0.0, 1.0])
    # EWC F test needs B ≥ m
    @test_throws ArgumentError jointtest(m, EWC(8))
    t = jointtest(m, EWC(12))
    @test t.distribution === :F
    @test (t.df, t.dfden) == (11, 2)
    @test t.statistic ≈ (12 - 11 + 1) / (12 * 11) * t.wald
    # value: a nonzero null
    θ = coefpath(m)
    @test jointtest(m; value = θ).wald ≈ 0 atol = 1.0e-20
    @test occursin("chi2(11)", sprint(show, MIME"text/plain"(), jointtest(m)))
end

@testitem "joint inference: response = shock, IV, panels, sample = :fixed" tags = [
    :joint,
] begin
    using LocalProjections, LinearAlgebra, StableRNGs, Test
    using CovarianceMatrices: HC0, HC1
    using StatsModels: @formula
    rng = StableRNG(7)
    T = 200
    x = randn(rng, T)
    z = x .+ randn(rng, T)
    y = zeros(T)
    for t in 2:T
        y[t] = 0.6y[t - 1] + x[t] + 0.5randn(rng)
    end
    d = (; y, x, z)

    # The response is the shock: h = 0 has no variance and is left out
    r = lp(@formula(leads(x) ~ x + lags(y, 2)), d; horizon = 4)
    @test jointtest(r).horizons == 1:4
    @test size(jointvcov(r)) == (4, 4)
    @test_throws ArgumentError jointvcov(r; horizons = 0:4)
    @test summarize(r; bands = :supt).lower[1] == 1.0

    # sample = :fixed: one sample, so HC0 is the plain stacked sandwich
    rf = lp(@formula(leads(y) ~ x + lags(y, 2)), d; horizon = 4, sample = :fixed)
    @test diag(jointvcov(rf, HC0())) ≈ stderror(rf + vcov(HC0())) .^ 2

    # IV: the scores of the projected regressors
    ri = lpiv(TSLS(), @formula(leads(y) ~ (z ~ x) + lags(y, 2)), d; horizon = 4)
    V = jointvcov(ri)
    @test diag(V) ≈ stderror(ri) .^ 2
    @test isposdef(Symmetric(V))
    @test jointtest(ri).pvalue < 0.01

    # panels are rejected
    dp = (;
        y = vcat(y, y), x = vcat(x, x), id = vcat(fill(1, T), fill(2, T)),
        t = vcat(1:T, 1:T),
    )
    rp = lp(@formula(leads(y) ~ x + lags(y, 1)), dp; horizon = 2, unit = :id, time = :t)
    @test_throws ArgumentError jointvcov(rp)
    @test_throws ArgumentError summarize(rp; bands = :supt)
end

@testitem "implicitfevd" tags = [:joint, :fevd] begin
    using LocalProjections, StableRNGs, Test
    using StatsModels: @formula
    rng = StableRNG(11)
    T = 400
    ε = randn(rng, T)
    r = Float64.(rand(rng, T) .< 0.2)
    x = 0.5 .* r .+ ε                      # impulse depends on a contemporaneous control
    y = zeros(T)
    for t in 2:T
        y[t] = 0.7y[t - 1] + ε[t] + 0.3r[t] + randn(rng)
    end
    d = (; y, x, r)
    m = lp(@formula(leads(y) ~ x + r + lags(y, 2) + lags(x, 2)), d; horizon = 6)
    f = implicitfevd(m)
    @test all(0 .<= f.share .<= 1)
    @test length(f.share) == 7
    # x̃ is orthogonal to the controls and recovers ε up to estimation error
    x̃ = implicitshock(m)
    ok = isfinite.(x̃)
    @test count(ok) == T - 2
    @test sum(x̃[ok] .* r[ok]) ≈ 0 atol = 1.0e-8
    a, b = x̃[ok], ε[ok] .- sum(ε[ok]) / count(ok)
    @test sum(a .* b) / sqrt(sum(abs2, a) * sum(abs2, b)) > 0.99
    # population share at h = 0: Var(ε) / (Var(ε) + Var(noise)) = 1/2
    @test f.share[1] ≈ 0.5 atol = 0.1
    # cumul responses work as well
    @test implicitfevd(lp(@formula(cumul(y) ~ x + r + lags(y, 2)), d; horizon = 3)).share[1] ≈
        f.share[1] atol = 0.05
    @test occursin("Share", sprint(show, MIME"text/plain"(), f))

    # Unsupported projections
    s = (; y, x, r, st = Float64.(rand(rng, T) .< 0.5))
    @test_throws ArgumentError implicitfevd(
        lp(@formula(leads(y) ~ x + lags(y, 1)), s; horizon = 2, state = :st)
    )
    @test_throws ArgumentError implicitfevd(
        lp(@formula(leads(y) ~ x + lags(y, 1)), d; horizon = 2, sample = :fixed)
    )
    @test_throws ArgumentError implicitfevd(
        lpiv(TSLS(), @formula(leads(y) ~ (r ~ x) + lags(y, 1)), d; horizon = 2)
    )
    @test_throws ArgumentError implicitshock(
        lpiv(TSLS(), @formula(leads(y) ~ (r ~ x) + lags(y, 1)), d; horizon = 2)
    )
end

@testitem "integer columns inside lags()" tags = [:data] setup = [CourseData] begin
    # A 0/1 dummy read from CSV is an Int column; Regress's `lags` pads it with NaN
    using LocalProjections, Test
    bk = CourseData.bk
    bki = merge(bk, (; recessiondates = Int.(bk.recessiondates)))
    @test eltype(bki.recessiondates) == Int
    @test coefpath(lp(CourseData.formula, bki; horizon = 3)) ==
        coefpath(lp(CourseData.formula, bk; horizon = 3))
end
