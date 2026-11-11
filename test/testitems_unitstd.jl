# Upstream's one-standard-deviation shocks (`normalization = UnitStd()`, `shockstd`) on
# the fork's features: supported for bias-corrected projections, an `ArgumentError`
# where the horizon-0 residual does not measure the shock.

@testitem "one-standard-deviation shocks on the fork's features" tags = [
    :summarize, :biascorr, :state, :joint, :api,
] begin
    using LocalProjections
    using DataFrames, StableRNGs, Test
    using CovarianceMatrices: HC0

    rng = StableRNG(20261111)
    n = 240
    x = 2 .* randn(rng, n)
    y = zeros(n)
    for t in 2:n
        y[t] = 0.6 * y[t - 1] + x[t] + randn(rng)
    end
    df = DataFrame(; y, x, s = Float64.(rand(rng, n) .< 0.4))
    f = @formula(leads(y) ~ x + lags(y, 2) + lags(x, 2))
    H = 6
    r = lp(f, df; horizon = H) + vcov(HC0())
    sigma = shockstd(r)

    # Bias correction: σ̂ times the corrected path. The delta-method variance is that of
    # the uncorrected response with the corrected path in place of β̂_h:
    #     σ̂² se_h² + θ̂ᶜ_h² Var(σ̂) + 2 σ̂ θ̂ᶜ_h Cov(β̂_h, σ̂).
    # Var(σ̂) is the squared standard error at h = 0 of the projection of the shock on
    # itself, and Cov(β̂_h, σ̂) is backed out of the uncorrected standard errors.
    bc = biascorrect(r)
    @test shockstd(bc) == sigma
    s_bc = summarize(bc; normalization = UnitStd(), level = 0.9)
    @test s_bc.normalization === UnitStd()
    @test s_bc.shocksize == sigma
    @test s_bc.coef ≈ sigma .* coefpath(bc)
    beta, theta, se = coefpath(r), coefpath(bc), stderror(r)
    @test !(theta ≈ beta)
    rx = lp(@formula(leads(x) ~ x + lags(y, 2) + lags(x, 2)), df; horizon = 0) + vcov(HC0())
    var_sigma = only(summarize(rx; normalization = UnitStd()).se)^2
    s_r = summarize(r; normalization = UnitStd())
    cov_beta_sigma = (s_r.se .^ 2 .- sigma^2 .* se .^ 2 .- beta .^ 2 .* var_sigma) ./
        (2 .* sigma .* beta)
    v = sigma^2 .* se .^ 2 .+ theta .^ 2 .* var_sigma .+ 2 .* sigma .* theta .* cov_beta_sigma
    @test s_bc.se ≈ sqrt.(v)
    irf = as_irf_result(bc; normalization = UnitStd())
    @test irf.metadata.bias_corrected
    @test irf.metadata.normalization === UnitStd()
    @test irf.metadata.shocksize == sigma

    # State dependence: a regime column is zero outside its regime, so the horizon-0
    # residual is not the shock of a regime
    rs = lp(f, df; horizon = 2, state = :s)
    @test_throws "state-dependent" shockstd(rs)
    @test_throws "state-dependent" shockstd(rs; term = :x_s1)
    @test_throws "state-dependent" summarize(rs; normalization = UnitStd())
    @test_throws "state-dependent" as_irf_result(rs; normalization = UnitStd())
    @test summarize(rs).shocksize == 1.0

    # Horizon-tracking regressors change the design with the horizon
    rt = lp(@formula(cumul(y) ~ cumul(x) + lags(y, 1)), df; horizon = 2)
    @test_throws "horizon-tracking" shockstd(rt)
    @test_throws "horizon-tracking" summarize(rt; normalization = UnitStd())
    @test summarize(rt).normalization === UnitEffect()

    # Sup-t bands use the correlation of the coefficients across horizons
    @test_throws "sup-t bands" summarize(r; bands = :supt, normalization = UnitStd())
    @test_throws "sup-t bands" as_irf_result(r; bands = :supt, normalization = UnitStd())
    s_supt = summarize(r; bands = :supt)
    @test s_supt.normalization === UnitEffect()
    @test s_supt.upper .- s_supt.lower ≈
        2 .* supt_critical_value(r) .* s_supt.se
    meta = as_irf_result(r; bands = :supt).metadata
    @test meta.bands === :supt && meta.shocksize == 1.0

    # A bootstrap summary is of the unit-effect response
    df2 = DataFrame(; shock = x, y)
    m = lp(@formula(leads(y) ~ shock + lags(y, 2) + lags(shock, 2)), df2; horizon = 3)
    b = varbootstrap(m, df2; vars = [:shock, :y], nlags = 2, nboot = 20, rng = StableRNG(3))
    sb = summarize(b)
    @test sb.normalization === UnitEffect()
    @test sb.shocksize == 1.0
    @test_throws MethodError summarize(b; normalization = UnitStd())
end
