# Upstream's one-standard-deviation shocks (`normalization = UnitStd()`, `shockstd`) on
# the fork's features: supported for bias-corrected projections and joint inference
# across horizons, an `ArgumentError` where the horizon-0 residual does not measure
# the shock.

@testitem "one-standard-deviation shocks on the fork's features" tags = [
    :summarize, :biascorr, :state, :joint, :api,
] begin
    using LocalProjections
    using DataFrames, LinearAlgebra, StableRNGs, Test
    using CovarianceMatrices: HC0, HC1, EWC, Bartlett, NeweyWest

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

    # Joint covariance of σ̂ β̂_h across horizons. Under HC0 it is the delta method on
    # the joint covariance V of the coefficients, with c_h = Cov(β̂_h, σ̂):
    #     σ̂² V + Var(σ̂) β̂ β̂' + σ̂ (β̂ c' + c β̂').
    V_std = jointvcov(r; normalization = UnitStd())
    @test V_std ≈ sigma^2 .* jointvcov(r) .+ var_sigma .* (beta * beta') .+
        sigma .* (beta * cov_beta_sigma' .+ cov_beta_sigma * beta')
    @test jointvcov(r; normalization = UnitStd(), horizons = [1, 4]) ≈ V_std[[2, 5], [2, 5]]
    @test jointvcov(r, HC0(); normalization = UnitStd()) == V_std
    @test jointvcov(r; normalization = UnitEffect()) == jointvcov(r)

    # Its diagonal is the squared delta-method standard error under every estimator
    for est in (HC1(), Bartlett(4), Bartlett{NeweyWest}(), EWC(12))
        re = r + vcov(est)
        V = jointvcov(re; normalization = UnitStd())
        @test sqrt.(diag(V)) ≈ summarize(re; normalization = UnitStd()).se
        @test issymmetric(V)
    end
    @test sqrt.(diag(jointvcov(bc; normalization = UnitStd()))) ≈ s_bc.se

    # Sup-t bands use the correlation of the normalized responses
    c_std = supt_critical_value(r; normalization = UnitStd(), level = 0.9)
    @test c_std == supt_critical_value(r; normalization = UnitStd(), level = 0.9)
    @test c_std != supt_critical_value(r; level = 0.9)
    @test 1.645 < c_std < 3.0
    s_supt = summarize(r; bands = :supt, normalization = UnitStd(), level = 0.9)
    @test s_supt.normalization === UnitStd()
    @test s_supt.coef == s_r.coef && s_supt.se == s_r.se
    @test s_supt.upper .- s_supt.lower ≈ 2 .* c_std .* s_supt.se
    meta = as_irf_result(r; bands = :supt, normalization = UnitStd()).metadata
    @test meta.bands === :supt && meta.shocksize == sigma
    s_unit = summarize(r; bands = :supt)
    @test s_unit.normalization === UnitEffect()
    @test s_unit.upper .- s_unit.lower ≈ 2 .* supt_critical_value(r) .* s_unit.se
    @test supt_critical_value(r; normalization = UnitStd(), horizons = [3], level = 0.9) ≈
        LocalProjections.critical_value(HC0(), 0.9)

    # When the response is the shock, the unit-effect response at h = 0 is 1 with no
    # sampling error and is left out; the one-standard-deviation response there is σ̂
    rxx = lp(@formula(leads(x) ~ x + lags(y, 2) + lags(x, 2)), df; horizon = 3) + vcov(HC0())
    @test size(jointvcov(rxx)) == (3, 3)
    V_xx = jointvcov(rxx; normalization = UnitStd())
    @test size(V_xx) == (4, 4)
    @test V_xx[1, 1] ≈ var_sigma
    @test_throws "leave h = 0 out" jointvcov(rxx; horizons = 0:2)
    s_xx = summarize(rxx; bands = :supt, normalization = UnitStd())
    @test s_xx.coef[1] ≈ sigma && s_xx.lower[1] < sigma < s_xx.upper[1]

    # Joint inference inherits the guards of the shock size
    @test_throws "state-dependent" jointvcov(rs; normalization = UnitStd())
    @test_throws "state-dependent" summarize(rs; bands = :supt, normalization = UnitStd())
    riv = lpiv(TSLS(), @formula(leads(y) ~ (x ~ lag(x)) + lags(y, 2)), df; horizon = 2)
    @test_throws "not identified by an IV" jointvcov(riv; normalization = UnitStd())
    @test size(jointvcov(riv)) == (3, 3)

    # A bootstrap summary is of the unit-effect response
    df2 = DataFrame(; shock = x, y)
    m = lp(@formula(leads(y) ~ shock + lags(y, 2) + lags(shock, 2)), df2; horizon = 3)
    b = varbootstrap(m, df2; vars = [:shock, :y], nlags = 2, nboot = 20, rng = StableRNG(3))
    sb = summarize(b)
    @test sb.normalization === UnitEffect()
    @test sb.shocksize == 1.0
    @test_throws MethodError summarize(b; normalization = UnitStd())
end
