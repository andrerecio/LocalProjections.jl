# State-dependent local projections (Ramey & Zubairy 2018), ported from the
# andrerecio fork onto the 0.2 stored-estimator API.

@testitem "state-dependent lp equals the hand-built regime interactions" tags = [
    :state, :core, :verification,
] begin
    using LocalProjections
    using DataFrames, StatsBase, Test, Random
    using CovarianceMatrices: HR0, Bartlett
    import Regress

    rng = Random.Xoshiro(20260921)
    n = 240
    x = randn(rng, n)
    s = Float64.(rand(rng, n) .< 0.4)
    y = zeros(n)
    for t in 2:n
        y[t] = 0.5y[t - 1] + (s[t - 1] == 1 ? 1.5 : 0.5) * x[t] + randn(rng)
    end
    df = DataFrame(y = y, x = x, s = s)

    r = lp(@formula(leads(y) ~ x + lags(y, 2)), df; horizon = 4, state = :s)
    @test coefnames(r) == [
        "(Intercept)_s0", "x_s0", "y_lag1_s0", "y_lag2_s0",
        "(Intercept)_s1", "x_s1", "y_lag1_s1", "y_lag2_s1",
    ]
    @test r.shock === :x_s0
    @test r.state == LPState(:s, 1, ("s0", "s1"), :x)
    @test nobs(r) == [n - 2 - h for h in 0:4]
    @test coefnames(horizonmodel(r, 2)) == coefnames(r)

    # Same column space as the Ramey-Zubairy lists: L.state, a constant and the
    # regressors interacted with L.state and 1 - L.state
    se_s1 = stderror(r + vcov(HR0()); term = :x_s1)
    for h in (0, 3)
        ts = 3:(n - h)
        I1 = s[ts .- 1]
        Xh = hcat(
            ones(length(ts)), I1, x[ts] .* (1 .- I1), x[ts] .* I1,
            y[ts .- 1] .* (1 .- I1), y[ts .- 1] .* I1, y[ts .- 2] .* (1 .- I1), y[ts .- 2] .* I1
        )
        mh = Regress.ols(Xh, y[ts .+ h]; has_intercept = true)
        @test coefpath(r; term = :x_s0)[h + 1] ≈ coef(mh)[3] rtol = 1.0e-10
        @test coefpath(r; term = :x_s1)[h + 1] ≈ coef(mh)[4] rtol = 1.0e-10
        @test se_s1[h + 1] ≈ sqrt(vcov(HR0(), mh)[4, 4]) rtol = 1.0e-8
        # the split intercepts span the constant, so R² is the centered one
        @test r2(horizonmodel(r, h)) ≈ r2(mh) rtol = 1.0e-10
    end

    # ... and as two separate regressions, one per regime
    ts = 3:n
    for (k, term) in ((0.0, :x_s0), (1.0, :x_s1))
        keep = ts[s[ts .- 1] .== k]
        mk = Regress.ols(hcat(ones(length(keep)), x[keep], y[keep .- 1], y[keep .- 2]), y[keep])
        @test coefpath(r; term)[1] ≈ coef(mk)[2] rtol = 1.0e-10
    end

    # labels, shock designation, state already timed (a leading missing state)
    df.sl = [missing; s[1:(end - 1)]]
    r2_ = lp(
        @formula(leads(y) ~ lags(y, 2) + x), df; horizon = 4, state = :sl,
        statelag = 0, regimes = ("lo", "hi"), shock = :x
    )
    @test r2_.shock === :x_lo
    @test coefpath(r2_; term = :x_hi) ≈ coefpath(r; term = :x_s1) rtol = 1.0e-10
    r3 = lp(@formula(leads(y) ~ lags(y, 2) + x), df; horizon = 2, state = :s, shock = :x_s1)
    @test r3.shock === :x_s1 && r3.state.shock === :x

    # statetest uses the covariance between the regimes
    st = statetest(r, Bartlett(4))
    @test st.diff ≈ coefpath(r; term = :x_s1) .- coefpath(r; term = :x_s0)
    @test all(0 .<= st.pvalue .<= 1)
    @test st.pvalue[1] < 0.01                     # 1.5 vs 0.5 on impact
    @test size(DataFrame(st)) == (5, 5)
    @test statetest(r + vcov(Bartlett(4))).stat == st.stat
    @test (r + vcov(Bartlett(4))).state == r.state

    # tautological h = 0: the shock is the response, in both regimes
    rt = lp(@formula(leads(x) ~ x + lags(y, 1)), df; horizon = 2, state = :s)
    @test rt.tautological_h0
    @test coefpath(rt; term = :x_s0)[1] == 1.0 && coefpath(rt; term = :x_s1)[1] == 1.0
    @test coefpath(rt; term = :y_lag1_s1)[1] == 0.0
    @test stderror(rt + vcov(HR0()); term = :x_s1)[1] == 0.0
    @test isnan(statetest(rt, HR0()).stat[1])
end

@testitem "state-dependent lp/lpiv reproduce Ramey-Zubairy (ivreg2, Stata 19.5)" tags = [
    :state, :iv, :weakiv, :verification,
] begin
    using LocalProjections
    using DataFrames, StatsBase, Test
    using CovarianceMatrices: Bartlett
    using Distributions: Chisq, ccdf

    path = joinpath(@__DIR__, "..", "docs", "src", "data", "ramey_zubairy.csv")
    lines = readlines(path)
    hdr = Symbol.(split(lines[1], ","))
    cols = [Union{Missing, Float64}[] for _ in hdr]
    for l in lines[2:end], (j, v) in enumerate(split(l, ","; keepempty = true))
        push!(cols[j], isempty(v) ? missing : parse(Float64, v))
    end
    rz = DataFrame(cols, hdr)

    # Reference: fork stata_test/rz_compare.do, -ivreg2 ..., robust bw(6)-, h = 8,
    # news shock. ivreg2 applies no n/(n-k) to kernel HAC; Regress 0.3 does, on OLS
    # and IV models alike, so the Stata variances are ours times (n-k)/n.
    r = lp(
        @formula(leads(y) ~ newsy + lags(newsy, 4) + lags(y, 4) + lags(g, 4)), rz;
        horizon = 8, state = :slack, regimes = ("exp", "rec")
    )
    n, k = nobs(r)[9], length(coefnames(r))
    @test n == 492 && k == 28
    @test coefpath(r; term = :newsy_exp)[9] ≈ 8.98901816161449e-2 rtol = 1.0e-9
    @test coefpath(r; term = :newsy_rec)[9] ≈ 3.70392317221167e-1 rtol = 1.0e-9
    se = stderror(r + vcov(Bartlett(6)); term = :newsy_rec) .* sqrt((n - k) / n)
    @test se[9] ≈ 8.10147008674164e-2 rtol = 1.0e-8

    m = lpiv(
        TSLS(),
        @formula(cumul(y) ~ (cumul(g) ~ newsy) + lags(newsy, 4) + lags(y, 4) + lags(g, 4)),
        rz; horizon = 8, state = :slack, regimes = ("exp", "rec")
    )
    gexp, grec = Symbol("cumul(g)_exp"), Symbol("cumul(g)_rec")
    @test m.shock === gexp
    @test m.endogenous_names == ["cumul(g)_exp", "cumul(g)_rec"]
    @test m.instrument_names == ["newsy_exp", "newsy_rec"]
    @test coefpath(m; term = gexp)[9] ≈ 5.90610510995029e-1 rtol = 1.0e-9
    @test coefpath(m; term = grec)[9] ≈ 6.20128665175688e-1 rtol = 1.0e-9
    mh = m + vcov(Bartlett(6))
    n, k = nobs(m)[9], length(coefnames(m))
    adj = sqrt((n - k) / n)
    @test stderror(mh; term = gexp)[9] * adj ≈ 1.00411664695866e-1 rtol = 1.0e-8
    @test stderror(mh; term = grec)[9] * adj ≈ 1.14414760866032e-1 rtol = 1.0e-8

    # the Wald statistic scales with the inverse of the variance factor
    st = statetest(mh)
    @test st.stat[9] * n / (n - k) ≈ 3.49868427322225e-2 rtol = 1.0e-7
    @test ccdf(Chisq(1), st.stat[9] * n / (n - k)) ≈ 8.51623232302201e-1 rtol = 1.0e-7

    # Montiel Olea-Pflueger effective F of the one-regime regression (-weakivtest-);
    # the finite-sample factor cancels in it
    @test weakivtest(mh, 8; regime = "rec").F_eff ≈ 1.86280736822122e+2 rtol = 1.0e-8
    @test weakivtest(mh, 8; regime = 2).F_eff == weakivtest(mh, 8; regime = :rec).F_eff
    @test length(weakivtest(mh; regime = 1)) == 9
    @test_throws "pass `regime = 1`" weakivtest(mh, 8)
    @test_throws "unknown regime boom" weakivtest(mh, 8; regime = "boom")
end

@testitem "state-dependent HBR multipliers reproduce Stata (ivreg2)" tags = [
    :state, :hbr, :verification,
] begin
    using LocalProjections
    using DataFrames, StatsBase, Test
    using CovarianceMatrices: Bartlett

    path = joinpath(@__DIR__, "..", "docs", "src", "data", "ramey_zubairy.csv")
    lines = readlines(path)
    hdr = Symbol.(split(lines[1], ","))
    cols = [Union{Missing, Float64}[] for _ in hdr]
    for l in lines[2:end], (j, v) in enumerate(split(l, ","; keepempty = true))
        push!(cols[j], isempty(v) ? missing : parse(Float64, v))
    end
    rz = DataFrame(cols, hdr)

    # Reference: fork stata_test/rz_hbr_compare.do, -ivreg2, robust bw(6)-, h = 8
    f = @formula(
        cumul(hbr(ypc)) ~ (cumul(hbr(gpc, ypc)) ~ newsy_hbr) + lags(newsy_hbr, 4) +
            lags(log(ypc), 4) + lags(log(gpc), 4)
    )
    ms = lpiv(TSLS(), f, rz; horizon = 8, state = :slack, regimes = ("exp", "rec"))
    gexp, grec = Symbol("cumul(hbr(gpc, ypc))_exp"), Symbol("cumul(hbr(gpc, ypc))_rec")
    @test coefpath(ms; term = gexp)[9] ≈ 7.68340603825077e-1 rtol = 1.0e-8
    @test coefpath(ms; term = grec)[9] ≈ 6.26191624567483e-1 rtol = 1.0e-8
    mh = ms + vcov(Bartlett(6))
    n, k = nobs(ms)[9], length(coefnames(ms))
    adj = sqrt((n - k) / n)
    @test stderror(mh; term = gexp)[9] * adj ≈ 1.12658838507871e-1 rtol = 1.0e-6
    @test stderror(mh; term = grec)[9] * adj ≈ 1.32589668223685e-1 rtol = 1.0e-6
    @test statetest(mh).stat[9] * n / (n - k) ≈ 6.86856330533422e-1 rtol = 1.0e-6
end

@testitem "state-dependent projections: panels, sample = :fixed and EWC" tags = [
    :state, :panel, :sample, :ewc,
] begin
    using LocalProjections
    using DataFrames, StableRNGs, StatsBase, Test
    using CovarianceMatrices: EWC, HC1
    using Distributions: FDist, ccdf

    rng = StableRNG(99)
    T = 220
    z = randn(rng, T)
    u = randn(rng, T)
    x = 0.8 .* z .+ 0.5 .* u
    y = zeros(T)
    for t in 2:T
        y[t] = 0.5 * y[t - 1] + x[t] + u[t]
    end
    s = Float64.(randn(rng, T) .> 0)
    df = DataFrame(y = y, x = x, z = z, s = s)
    H = 5

    # sample = :fixed: the split design of the maximum horizon at every h
    f = @formula(leads(y) ~ x + lags(y, 1))
    sth = lp(f, df; horizon = H, state = :s)
    stc = lp(f, df; horizon = H, state = :s, sample = :fixed)
    @test all(==(nobs(sth)[end]), nobs(stc))
    for h in 0:H
        @test modelmatrix(horizonmodel(stc, h)) == modelmatrix(horizonmodel(sth, H))
    end
    @test coef(horizonmodel(stc, H)) == coef(horizonmodel(sth, H))
    @test length(statetest(stc, HC1()).pvalue) == H + 1

    # EWC: t_B bands per regime, F(1, B) p-values in statetest
    B = ewc_bandwidth(sth)
    se = stderror(sth + vcov(EWC(B)); term = :x_s1)
    @test summarize(sth, EWC(B); term = :x_s1, level = 0.9).upper ≈
        coefpath(sth; term = :x_s1) .+ LocalProjections._critical_value(EWC(B), 0.9) .* se
    st = statetest(sth, EWC(B))
    @test st.pvalue ≈ ccdf.(FDist(1, B), st.stat)

    # Panel: the state is lagged within each unit, so the first row of every unit
    # has no state and drops out
    N = 3
    dp = DataFrame(
        id = repeat(1:N, inner = 60), t = repeat(1:60, N),
        y = randn(rng, 3 * 60), x = randn(rng, 3 * 60), s = Float64.(rand(rng, 3 * 60) .< 0.5)
    )
    rp = lp(@formula(leads(y) ~ x), dp; horizon = 2, unit = :id, time = :t, state = :s)
    @test nobs(rp) == [N * (60 - 1 - h) for h in 0:2]
    # same coefficients as the stacked hand-built interactions
    rows = [(u, t) for u in 1:N for t in 2:60]
    sel(v, u, t) = v[(u - 1) * 60 + t]
    I1 = [sel(dp.s, u, t - 1) for (u, t) in rows]
    xs = [sel(dp.x, u, t) for (u, t) in rows]
    Xh = hcat(1 .- I1, xs .* (1 .- I1), I1, xs .* I1)
    yh = [sel(dp.y, u, t) for (u, t) in rows]
    @test coef(horizonmodel(rp, 0)) ≈ Xh \ yh rtol = 1.0e-10
end

@testitem "state-dependent projections: guards" tags = [:state, :api] begin
    using LocalProjections
    using DataFrames, StatsBase, Test, Random
    using CovarianceMatrices: HR0

    rng = Random.Xoshiro(7)
    n = 120
    df = DataFrame(
        y = randn(rng, n), x = randn(rng, n), z = randn(rng, n),
        s = Float64.(rand(rng, n) .< 0.5), bad = 2 .* rand(rng, n), w = rand(rng, n),
        id = repeat(1:4, inner = 30)
    )
    f = @formula(leads(y) ~ x + lags(y, 1))

    @test_throws "indicator or a weight in [0, 1]" lp(f, df; horizon = 2, state = :bad)
    @test_throws "`statelag` must be non-negative" lp(f, df; horizon = 2, state = :s, statelag = -1)
    @test_throws "two labels" lp(f, df; horizon = 2, state = :s, regimes = ("a",))
    @test_throws "must differ" lp(f, df; horizon = 2, state = :s, regimes = ("a", "a"))
    @test_throws "shock `nope`" lp(f, df; horizon = 2, state = :s, shock = :nope)
    @test_throws "not in the data" lp(f, df; horizon = 2, state = :nostate)
    @test_throws "fixed effects" lp(@formula(leads(y) ~ x + fe(id)), df; horizon = 2, state = :s)

    # a smooth weight in [0, 1] (Auerbach-Gorodnichenko) is accepted
    rw = lp(f, df; horizon = 2, state = :w)
    @test length(coefnames(rw)) == 6

    r = lp(f, df; horizon = 2, state = :s)
    @test_throws "needs a projection estimated with `state" statetest(lp(f, df; horizon = 2), HR0())
    @test_throws "term nope not present" statetest(r, HR0(); term = :nope)

    # linear results are untouched, and `regime` is rejected there
    @test lp(f, df; horizon = 2).state === nothing
    miv = lpiv(TSLS(), @formula(leads(y) ~ (x ~ z) + lags(y, 1)), df; horizon = 2)
    @test miv.state === nothing
    @test_throws "`regime` applies to state-dependent projections only" weakivtest(miv, 1; regime = 1)

    # instrument equal to the regressor at h = 0 (Blanchard-Perotti): Regress.jl treats
    # the regressor as exogenous there, so there is no regime first stage to test
    dbp = transform(df, :x => identity => :xz)
    mbp = lpiv(TSLS(), @formula(cumul(y) ~ (cumul(x) ~ xz) + lags(y, 1)), dbp; horizon = 2, state = :s)
    @test_throws "cannot be split into the two regimes" weakivtest(mbp, 0; regime = 2)
    @test isfinite(weakivtest(mbp, 1; regime = 2).F_eff)

    @test occursin("State:", sprint(show, MIME("text/plain"), r))
    @test !occursin("State:", sprint(show, MIME("text/plain"), miv))
end
