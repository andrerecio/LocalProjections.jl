# Equal-weighted cosine (EWC) inference with fixed-smoothing Student-t_B critical
# values (ported from the andrerecio fork onto the 0.2 stored-estimator API).

@testitem "EWC bandwidth rule (LLSW 2018)" tags = [:ewc, :vcov, :api] begin
    using LocalProjections
    using DataFrames, StatsBase, Test

    # Integer rule: B = floor(0.41 * T0^(2/3))
    @test ewc_bandwidth(256) == 16   # baseline in the inference guide
    @test ewc_bandwidth(100) == 8
    @test ewc_bandwidth(1) == 1      # floor never below 1

    # LP method uses the horizon-zero effective sample size
    n = 150
    df = DataFrame(x = randn(n), y = randn(n))
    r = lp(@formula(leads(y) ~ x + lags(y, 2)), df; horizon = 6)
    @test nobs(r)[1] == n - 2  # two lags consumed
    @test ewc_bandwidth(r) == ewc_bandwidth(n - 2)

    # sample = :fixed: the common sample of all horizons
    rf = lp(@formula(leads(y) ~ x + lags(y, 2)), df; horizon = 6, sample = :fixed)
    @test ewc_bandwidth(rf) == ewc_bandwidth(n - 2 - 6)
end

@testitem "EWC inference uses Student-t_B critical values" tags = [
    :ewc, :vcov, :summarize, :api,
] begin
    using LocalProjections
    using DataFrames, Test
    using CovarianceMatrices: EWC, HC1, Bartlett
    using Distributions: Normal, TDist, quantile

    n = 200
    df = DataFrame(x = randn(n), y = randn(n))
    r = lp(@formula(leads(y) ~ x + lags(y, 2)), df; horizon = 6)
    B = ewc_bandwidth(r)
    beta = coefpath(r; term = :x)
    tcrit = quantile(TDist(B), 0.95)
    zcrit = quantile(Normal(), 0.95)

    # summarize pairs EWC with t_B, not the normal, through either idiom
    rewc = r + vcov(EWC(B))
    se_ewc = stderror(rewc; term = :x)
    for s in (summarize(rewc; term = :x, level = 0.9), summarize(r, EWC(B); term = :x, level = 0.9))
        @test s.lower ≈ beta .- tcrit .* se_ewc
        @test s.upper ≈ beta .+ tcrit .* se_ewc
        @test !(s.upper ≈ beta .+ zcrit .* se_ewc)  # t_B bands are wider
    end

    # Non-EWC estimators keep normal critical values
    for est in (HC1(), Bartlett(4))
        se = stderror(r + vcov(est); term = :x)
        @test summarize(r, est; term = :x, level = 0.9).upper ≈ beta .+ zcrit .* se
    end

    # as_irf_result, and therefore the plots, follow the same pairing
    irf = as_irf_result(r; vcov_estimator = EWC(B), coverage = [0.68, 0.9])
    @test vec(irf.upper[2].data) ≈ beta .+ tcrit .* se_ewc
    @test vec(irf.lower[2].data) ≈ beta .- tcrit .* se_ewc
    @test vec(irf.upper[1].data) ≈ beta .+ quantile(TDist(B), 0.84) .* se_ewc
    @test as_irf_result(rewc; coverage = [0.9]).upper[1].data ≈ irf.upper[2].data

    # IV local projections pair EWC with t_B too
    z = randn(n)
    dfi = DataFrame(y = randn(n), x = z .+ randn(n), z = z)
    ri = lpiv(TSLS(), @formula(leads(y) ~ (x ~ z)), dfi; horizon = 3) + vcov(EWC(B))
    s = summarize(ri; level = 0.9)
    @test s.upper ≈ coefpath(ri) .+ tcrit .* stderror(ri)
end

@testitem "EWC covariance matches the cosine-projection formula" tags = [
    :ewc, :vcov, :verification,
] begin
    using LocalProjections
    using DataFrames, LinearAlgebra, StableRNGs, Test
    using CovarianceMatrices
    using CovarianceMatrices: EWC
    using StatsBase: modelmatrix, residuals

    # AR(1) errors so the long-run correction actually matters
    rng = StableRNG(250)
    n = 250
    e = zeros(n)
    for t in 2:n
        e[t] = 0.5 * e[t - 1] + randn(rng)
    end
    x = randn(rng, n)
    y = 0.8 .* x .+ e
    df = DataFrame(x = x, y = y)
    r = lp(@formula(leads(y) ~ x + lags(y, 1)), df; horizon = 3)
    B = ewc_bandwidth(r)
    re = r + vcov(EWC(B))

    for h in 0:3
        m = horizonmodel(r, h)
        X = modelmatrix(m)
        u = residuals(m)
        T, k = size(X)
        g = X .* u
        # Guide §2.2: Type-II DCT projections of the score
        Lam = [sqrt(2 / T) * sum(cos(π * j * (t - 0.5) / T) * g[t, :] for t in 1:T) for j in 1:B]
        Omega = sum(l * l' for l in Lam) / B
        V_raw = T * ((X'X) \ Omega) / (X'X)
        # Regress applies n/(n-k) to the stored EWC covariance; dofadjust = false
        # gives the raw sandwich (the harreg convention)
        @test vcov(horizonmodel(re, h)) ≈ (T / (T - k)) * V_raw rtol = 1.0e-10
        @test CovarianceMatrices.vcov(EWC(B), m; dofadjust = false) ≈ V_raw rtol = 1.0e-10
    end
end

@testitem "EWC bands reproduce Stata harreg up to sqrt(n/(n-k))" tags = [
    :ewc, :verification,
] begin
    using LocalProjections
    using DataFrames, StatsBase, Test
    using CovarianceMatrices: EWC
    using Distributions: TDist, quantile

    path = joinpath(@__DIR__, "..", "docs", "src", "data", "ramey_zubairy.csv")
    lines = readlines(path)
    hdr = Symbol.(split(lines[1], ","))
    cols = [Union{Missing, Float64}[] for _ in hdr]
    for l in lines[2:end], (j, v) in enumerate(split(l, ","; keepempty = true))
        push!(cols[j], isempty(v) ? missing : parse(Float64, v))
    end
    rz = DataFrame(cols, hdr)

    # Reference (fork stata_test/harreg_compare.do), h = 8:
    #   harreg F8.y newsy L(1/4).newsy L(1/4).y L(1/4).g, df(25) level(90)
    r = lp(@formula(leads(y) ~ newsy + lags(newsy, 4) + lags(y, 4) + lags(g, 4)), rz; horizon = 8)
    n, k = nobs(r)[9], length(coefnames(r))
    @test n == 492
    @test ewc_bandwidth(r) == 25   # harreg's default df at h = 8 is also 25
    s = summarize(r + vcov(EWC(25)); term = :newsy, level = 0.9)
    @test s.coef[9] ≈ 2.29480275659605e-1 rtol = 1.0e-10
    @test quantile(TDist(25), 0.95) ≈ 1.7081407612519 rtol = 1.0e-12      # e(cv_fb)
    @test s.se[9] ≈ 6.23430524884572e-2 * sqrt(n / (n - k)) rtol = 1.0e-10
    # Without the factor the bounds are harreg's
    adj = sqrt((n - k) / n)
    cv = quantile(TDist(25), 0.95)
    @test s.coef[9] - cv * s.se[9] * adj ≈ 1.22989566523205e-1 rtol = 1.0e-9
    @test s.coef[9] + cv * s.se[9] * adj ≈ 3.35970984796006e-1 rtol = 1.0e-9
end
