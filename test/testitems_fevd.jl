@testitem "fevd: R², LP-A and LP-B run and return well-formed results" tags = [
    :fevd, :smoke] begin
    using LocalProjections
    using DataFrames, StableRNGs, StatsModels, Statistics, Test

    rng = StableRNG(20261010)
    T = 200
    x = randn(rng, T)
    z = zeros(T)
    dy = zeros(T)
    for t in 2:T
        z[t] = 0.5 * z[t - 1] + randn(rng)
        dy[t] = 0.3 * dy[t - 1] + 0.8 * x[t] + 0.4 * x[t - 1] + 0.3 * z[t] + randn(rng)
    end
    df = DataFrame(x = x .+ 0.5, z = z, dy = dy, y = cumsum(dy))

    H = 8
    fm = @formula(cumul(dy) ~ x + z + lags(x, 2) + lags(z, 2) + lags(dy, 2))
    for est in (:r2, :lpa, :lpb)
        f = fevd(fm, df; horizon = H, estimator = est)
        @test f isa LPFEVD
        @test f.estimator === est
        @test f.response === :dy && f.shock === :x && f.lhs === :cumul
        @test f.contemporaneous == [:x, :z]
        @test f.t0 == 3 && f.t0lp == 3
        @test f.nobs == [T - h - 3 + 1 for h in 0:H]
        @test f.window == hcat(fill(3, H + 1), T .- (0:H))
        @test length(f.share) == H + 1 && all(isfinite, f.share)
        @test all(isfinite, f.se_pw) && all(isfinite, f.se)
        @test all(>(0), f.se) && all(>(0), f.se_pw)
        @test f.shock_mean ≈ mean(df.x) && f.shock_var ≈ var(df.x)
        @test size(f.forecast_errors) == (T, H + 1)
        for h in 0:H
            @test all(isfinite, f.forecast_errors[3:(T - h), h + 1])
            @test all(isnan, f.forecast_errors[(T - h + 1):T, h + 1])
            m = est === :r2 ? 2h + 3 : 9 * (h + 1) + 2
            @test length(f.theta[h + 1]) == m
            @test size(f.avar[h + 1]) == (m, m) && size(f.avar_pw[h + 1]) == (m, m)
        end
        if est === :r2
            @test f.irf === nothing && f.irf_se === nothing && f.sigma2v === nothing
            @test all(s -> 0 <= s <= 1, f.share)
        else
            @test length(f.irf) == H + 1 && all(isfinite, f.irf_se)
            @test f.share ≈ f.shock_var .* cumsum(f.irf .^ 2) ./ f.mse
        end
        est === :lpb && @test all(s -> 0 <= s <= 1, f.share)
        @test occursin("LPFEVD", sprint(show, f))
        @test occursin("Share", sprint(show, MIME"text/plain"(), f))
    end

    # ldiff(y) with lags of first differences, a separate impulse-response
    # regression and a trend
    f = fevd(@formula(ldiff(y) ~ x + lags(x, 2) + lags(firstdiff(y), 2)), df;
        horizon = H, estimator = :lpb, trend = 1,
        irf_formula = @formula(ldiff(y) ~ x + lags(x, 1) + lags(firstdiff(y), 1)))
    @test f.lhs === :ldiff && f.response === :y
    @test f.t0 == 4 && f.t0lp == 3
    @test all(isfinite, f.share) && all(isfinite, f.se_pw)
    @test length(f.theta[1]) == 5 + 2   # x, x_lag1, Δy_lag1, intercept, t; σ²x, σ²v

    # The LP-A/B estimates from an lp() result reuse its formula, horizon and shock
    m = lp(fm, df; horizon = H)
    @test fevd(m, df; estimator = :lpb).share ≈
          fevd(fm, df; horizon = H, estimator = :lpb).share
    @test fevd(m, df; estimator = :lpb).irf ≈ coefpath(m)

    # Guards
    @test_throws ArgumentError fevd(@formula(leads(y) ~ x + lags(x, 2)), df; horizon = 4)
    @test_throws ArgumentError fevd(@formula(cumul(dy) ~ x + log(abs(z))), df; horizon = 4)
    @test_throws ArgumentError fevd(@formula(cumul(dy) ~ 0 + x + lags(x, 2)), df; horizon = 4)
    @test_throws ArgumentError fevd(@formula(cumul(dy) ~ lags(x, 2)), df; horizon = 4)
    @test_throws ArgumentError fevd(fm, df; horizon = 4, shock = :dy)
    @test_throws ArgumentError fevd(fm, df; horizon = 4, estimator = :var)
    @test_throws ArgumentError fevd(fm, df; horizon = 4, trend = 4)
    @test_throws ArgumentError fevd(fm, df; horizon = 4, irf_formula = fm)
    bad = copy(df)
    bad.z = allowmissing(bad.z)
    bad.z[50] = missing
    @test_throws ArgumentError fevd(fm, bad; horizon = 4)
end

@testitem "fevd: VAR helpers (_var_ma, _var_fevd, _gl_lagselect)" tags = [
    :fevd, :smoke] begin
    using LocalProjections
    using LinearAlgebra, StableRNGs, Test
    const LPm = LocalProjections

    A = zeros(3, 3, 2)
    A[:, :, 1] = [0.5 0.1 0.0; 0.2 0.4 0.1; 0.0 0.3 0.3]
    A[:, :, 2] = [0.1 0.0 0.05; 0.0 0.1 0.0; 0.05 0.0 0.1]
    Ω = [1.0 0.3 0.1; 0.3 2.0 0.4; 0.1 0.4 1.5]
    H = 8

    # Θ_h = [F^h]_{1:n,1:n} for the companion matrix F
    F = [A[:, :, 1] A[:, :, 2]; Matrix(1.0I, 3, 3) zeros(3, 3)]
    Θ = LPm._var_ma(A, H)
    for h in 0:H
        @test Θ[:, :, h + 1] ≈ (F ^ h)[1:3, 1:3] rtol=1e-12
    end
    @test LPm._var_irf(A, [1.0, 0.5, 0.0], H) ≈
          reduce(hcat, [Θ[:, :, h + 1] * [1.0, 0.5, 0.0] for h in 0:H]) rtol=1e-14

    # FEVD by brute force from Ψ_h = Θ_h L, with and without cumulation
    L = cholesky(Symmetric(Ω)).L
    for cumulate in (false, true)
        Ψ = [Θ[:, :, h + 1] * L for h in 0:H]
        cumulate && (Ψ = cumsum(Ψ))
        for r in 1:3
            shares = [LPm._var_fevd(A, Ω, H; shock = j, response = r, cumulate)
                      for j in 1:3]
            @test sum(shares) ≈ ones(H + 1) rtol=1e-12
            for j in 1:3, h in 0:H

                num = sum(Ψ[l + 1][r, j]^2 for l in 0:h)
                den = sum(sum(abs2, Ψ[l + 1][r, :]) for l in 0:h)
                @test shares[j][h + 1] ≈ num / den rtol=1e-12
            end
            # invariant to the scale of Ω
            @test LPm._var_fevd(A, 3.7 .* Ω, H; shock = 2, response = r, cumulate) ≈
                  shares[2] rtol=1e-12
        end
    end
    @test_throws ArgumentError LPm._var_fevd(A, [1.0 1.0 0.0; 1.0 1.0 0.0; 0.0 0.0 1.0],
        H; shock = 1, response = 1)

    # IC on the common sample: hand computation of HQIC / BIC / AIC
    rng = StableRNG(7)
    T = 150
    Y = zeros(T, 2)
    for t in 3:T
        Y[t, :] = 0.5 * Y[t - 1, :] - 0.2 * Y[t - 2, :] + randn(rng, 2)
    end
    cand = [1, 2, 3, 4]
    Tic = T - 4
    for (crit, pen) in ((:hqic, k -> 2k * log(log(Tic)) / Tic),
        (:bic, k -> k * log(Tic) / Tic), (:aic, k -> 2k / Tic))
        sel = LPm._gl_lagselect(Y, cand; criterion = crit)
        for (i, p) in enumerate(cand)
            Yp = Y[(4 - p + 1):end, :]
            X = hcat([Yp[(p - l + 1):(end - l), :] for l in 1:p]..., ones(Tic))
            U = Yp[(p + 1):end, :] - X * (X \ Yp[(p + 1):end, :])
            @test sel.values[i] ≈ logdet(U' * U / Tic) + pen(4p) rtol=1e-10
        end
        @test sel.p == cand[argmin(sel.values)]
    end
    @test_throws ArgumentError LPm._gl_lagselect(Y, [0, 1, 2])
    @test_throws ArgumentError LPm._gl_lagselect(Y, [1, 2]; criterion = :fpe)
end

@testitem "fevdbootstrap: VAR, block and simulation bias corrections" tags = [
    :fevd, :smoke] begin
    using LocalProjections
    using DataFrames, Random, StableRNGs, StatsModels, Statistics, Test

    rng = StableRNG(20261011)
    T = 200
    x = randn(rng, T)
    z = zeros(T)
    dy = zeros(T)
    for t in 2:T
        z[t] = 0.5 * z[t - 1] + 0.2 * dy[t - 1] + randn(rng)
        dy[t] = 0.3 * dy[t - 1] + 0.8 * x[t] + 0.4 * x[t - 1] + 0.3 * z[t] + randn(rng)
    end
    df = DataFrame(x = x .+ 0.5, z = z, dy = dy, y = cumsum(dy))
    H = 6
    fm = @formula(cumul(dy) ~ x + lags(x, 2) + lags(z, 2) + lags(dy, 2))
    nb = 60

    # --- VAR residual bootstrap -------------------------------------------
    for est in (:r2, :lpa, :lpb)
        f = fevd(fm, df; horizon = H, estimator = est)
        b = fevdbootstrap(f, df; method = :var, nboot = nb, nlags = 1:4,
            rng = StableRNG(1))
        @test b isa FEVDBootstrap
        @test b.method === :var && b.fevd === f
        @test b.vars == [:x, :z, :dy]               # shock, then RHS order
        @test b.nlags in 1:4 && b.ic.criterion === :hqic
        @test b.ic.candidates == 1:4 && length(b.ic.values) == 4
        @test b.nlags == b.ic.candidates[argmin(b.ic.values)]
        @test b.burnin == 100 && b.skipfirst == false
        @test b.nboot == nb && b.nfail == 0
        @test size(b.draws) == (nb, H + 1) && all(isfinite, b.draws)
        @test b.mean_draws ≈ vec(mean(b.draws; dims = 1))
        @test b.se_boot ≈ vec(std(b.draws; dims = 1))
        @test length(b.pseudo_truth) == H + 1
        @test all(s -> 0 <= s <= 1, b.pseudo_truth)
        @test b.biascorrected ≈ f.share .- (b.mean_draws .- b.pseudo_truth)
        @test b.se_pw == f.se_pw && b.se == f.se
        @test b.blocklength === nothing && b.prewhiten === nothing
        # forwarding to the LPFEVD
        @test b.share == f.share && b.horizon == H && b.estimator === est
        if est === :r2
            @test b.irf_draws === nothing && b.irf_bc === nothing
        else
            @test size(b.irf_draws) == (nb, H + 1)
            @test b.irf_bc ≈ 2 .* f.irf .- vec(mean(b.irf_draws; dims = 1))
            @test b.irf_se_boot ≈ vec(std(b.irf_draws; dims = 1))
            @test size(b.irfstr_draws) == (nb, H + 1)
        end
        @test occursin("FEVDBootstrap", sprint(show, b))
        @test occursin("VAR share", sprint(show, MIME"text/plain"(), b))
    end

    f = fevd(fm, df; horizon = H, estimator = :r2)
    # fixed lag order, explicit order, AIC; the pseudo-truth depends on the order
    b1 = fevdbootstrap(f, df; nboot = 5, nlags = 2, rng = StableRNG(2))
    @test b1.nlags == 2 && b1.ic === nothing
    b2 = fevdbootstrap(f, df; nboot = 5, nlags = 2, vars = [:z, :dy, :x],
        rng = StableRNG(2))
    @test b2.vars == [:z, :dy, :x] && b2.pseudo_truth != b1.pseudo_truth
    b3 = fevdbootstrap(f, df; nboot = 5, nlags = 1:3, criterion = :aic,
        skipfirst = true, burnin = 0, rng = StableRNG(2))
    @test b3.ic.criterion === :aic && b3.skipfirst && b3.burnin == 0
    # the same seed gives the same draws
    @test fevdbootstrap(f, df; nboot = 5, nlags = 2, rng = StableRNG(2)).draws == b1.draws

    # ldiff(y): the VAR holds y in levels
    fl = fevd(@formula(ldiff(y) ~ x + lags(x, 2) + lags(firstdiff(y), 2)), df;
        horizon = H, estimator = :lpb)
    bl = fevdbootstrap(fl, df; nboot = 20, nlags = 2, rng = StableRNG(3))
    @test bl.vars == [:x, :y] && all(isfinite, bl.biascorrected)

    # --- block bootstrap (R² only) ----------------------------------------
    bb = fevdbootstrap(f; method = :block, nboot = nb, rng = StableRNG(4))
    @test bb.method === :block && bb.blocklength == 4 && bb.pseudo_truth === nothing
    @test size(bb.draws) == (nb, H + 1) && all(isfinite, bb.draws)
    @test bb.biascorrected ≈ 2 .* f.share .- bb.mean_draws
    @test bb.se_pw == f.se_pw && bb.vars === nothing
    # with blocklength = T_H the only block at h = H is the sample itself
    bd = fevdbootstrap(f; method = :block, nboot = 5, blocklength = f.nobs[end],
        rng = StableRNG(4))
    @test all(==(f.share[end]), bd.draws[:, end])
    @test bd.biascorrected[end] ≈ f.share[end] atol=1e-14

    # --- asymptotic simulation ----------------------------------------------
    for est in (:r2, :lpa, :lpb)
        fe = fevd(fm, df; horizon = H, estimator = est)
        bs = fevdbootstrap(fe; method = :simulation, nboot = 500, rng = StableRNG(5))
        @test bs.method === :simulation && bs.prewhiten
        @test size(bs.draws) == (500, H + 1) && all(isfinite, bs.draws)
        @test bs.biascorrected ≈ 2 .* fe.share .- bs.mean_draws
        @test bs.bound == (est === :r2)
        est === :r2 && @test all(s -> 0 <= s <= 1, bs.draws)
        est === :lpb && @test all(s -> 0 <= s <= 1, bs.draws)
        # delta-method standard errors at the bias-corrected share
        h = 3
        θ = fe.theta[h + 1]
        if est === :r2
            g = vcat(θ[(h + 2):(2h + 2)], θ[1:(h + 1)], -bs.biascorrected[h + 1]) ./ θ[end]
        else
            Nq = (length(θ) - 2) ÷ (h + 1)
            c = est === :lpa ? 1 / θ[end] : (1 - bs.biascorrected[h + 1]) / fe.mse[h + 1]
            g = zeros(length(θ))
            for j in 0:h
                g[1 + j * Nq] = c * 2 * fe.irf[j + 1] * fe.shock_var
            end
            g[end - 1] = c * sum(abs2, fe.irf[1:(h + 1)])
            g[end] = -bs.biascorrected[h + 1] / (est === :lpa ? θ[end] : fe.mse[h + 1])
        end
        @test bs.se_pw[h + 1] ≈ sqrt(g' * fe.avar_pw[h + 1] * g) rtol=1e-10
        @test bs.se[h + 1] ≈ sqrt(g' * fe.avar[h + 1] * g) rtol=1e-10
        @test fevdbootstrap(fe; method = :simulation, nboot = 50, prewhiten = false,
            rng = StableRNG(5)).prewhiten == false
    end
    bu = fevdbootstrap(f; method = :simulation, nboot = 500, bound = false,
        rng = StableRNG(5))
    @test bu.bound == false

    # --- threaded and serial runs are bit-identical --------------------------
    fb = fevd(fm, df; horizon = H, estimator = :lpb)
    for (ff, kw) in ((fb, (; method = :var, nlags = 2)), (f, (; method = :block)),
        (fb, (; method = :simulation)))
        data = kw.method === :var ? df : nothing
        s1 = fevdbootstrap(
            ff, data; nboot = 40, rng = StableRNG(9), threaded = false, kw...)
        s2 = fevdbootstrap(ff, data; nboot = 40, rng = StableRNG(9), threaded = true, kw...)
        @test s1.draws == s2.draws
        @test s1.biascorrected == s2.biascorrected
        kw.method === :var && @test s1.irf_draws == s2.irf_draws
    end

    # --- guards --------------------------------------------------------------
    flb = fevd(fm, df; horizon = H, estimator = :lpb)
    @test_throws ArgumentError fevdbootstrap(f; method = :bogus)
    @test_throws ArgumentError fevdbootstrap(f; method = :var)              # no data
    @test_throws ArgumentError fevdbootstrap(flb; method = :block)           # R² only
    @test_throws ArgumentError fevdbootstrap(f; method = :block, blocklength = T)
    @test_throws ArgumentError fevdbootstrap(f, df; vars = [:x, :dy])        # z missing
    @test_throws ArgumentError fevdbootstrap(f, df; nlags = 0:2)
    @test_throws ArgumentError fevdbootstrap(f, df; nlags = 0)
    @test_throws ArgumentError fevdbootstrap(f, df; criterion = :fpe)
    @test_throws ArgumentError fevdbootstrap(f, df[1:150, :])
    @test_throws ArgumentError fevdbootstrap(fl, df; skipfirst = true)       # ldiff
    @test_throws ArgumentError fevdbootstrap(f, df; nboot = 0)
    ft = fevd(fm, df; horizon = H, trend = 1)
    @test_throws ArgumentError fevdbootstrap(ft, df)
    @test fevdbootstrap(ft; method = :simulation, nboot = 20, rng = StableRNG(1)) isa
          FEVDBootstrap
end

@testitem "fevd: summarize, plot recipes and as_irf_result" tags = [:fevd, :smoke] begin
    using LocalProjections
    using DataFrames, Distributions, Plots, StableRNGs, StatsModels, Statistics, Test

    # MATLAB `quantile` is Hyndman–Fan type 5: (i - 0.5)/n plotting positions
    v = [4.0, 1.0, 3.0, 2.0]
    q5 = LocalProjections._gl_quantile5
    @test q5(v, 0.375) == 2.0                    # (2 - 0.5)/4
    @test q5(v, 0.3) ≈ 1.7                       # 1 + (0.3 - 0.125)/0.25
    @test q5(v, 0.05) == 1.0 && q5(v, 0.99) == 4.0   # clamped at the extremes

    rng = StableRNG(20261012)
    T = 200
    x = randn(rng, T)
    dy = zeros(T)
    for t in 2:T
        dy[t] = 0.3 * dy[t - 1] + 0.8 * x[t] + 0.4 * x[t - 1] + randn(rng)
    end
    df = DataFrame(x = x, dy = dy)
    H = 5
    fm = @formula(cumul(dy) ~ x + lags(x, 2) + lags(dy, 2))
    f = fevd(fm, df; horizon = H, estimator = :r2)

    # --- summarize(::LPFEVD): ŝ ± z·se ----------------------------------------
    z = quantile(Normal(), 0.95)
    s = summarize(f)
    @test s isa FEVDSummary
    @test s.method === :asymptotic && s.band === :normal && s.level == 0.90
    @test !s.biascorrected && s.prewhiten === true
    @test s.horizon == 0:H && s.share == f.share && s.estimate == f.share
    @test s.se == f.se_pw
    @test s.lower ≈ f.share .- z .* f.se_pw && s.upper ≈ f.share .+ z .* f.se_pw
    s0 = summarize(f; level = 0.68, prewhiten = false)
    @test s0.se == f.se && s0.prewhiten === false
    @test s0.upper ≈ f.share .+ quantile(Normal(), 0.84) .* f.se
    @test names(DataFrame(s)) == ["horizon", "share", "estimate", "se", "lower", "upper"]
    @test occursin("FEVDSummary", sprint(show, s))
    @test occursin("Lower 90%", sprint(show, MIME"text/plain"(), s))
    @test_throws ArgumentError summarize(f; level = 1.2)

    # --- summarize(::FEVDBootstrap): shifted quantiles and normal bands --------
    b = fevdbootstrap(f, df; method = :var, nboot = 99, nlags = 2, rng = StableRNG(1))
    sb = summarize(b)
    @test sb.method === :var && sb.band === :shift && sb.biascorrected
    @test sb.prewhiten === nothing
    @test sb.estimate == b.biascorrected && sb.se == b.se_boot
    for h in 1:(H + 1)
        d = sort(b.draws[:, h])
        n = length(d)
        # type 5 by hand: position n·p + 1/2 between order statistics
        tq(p) = (a = n * p + 0.5; j = clamp(floor(Int, a), 1, n - 1);
            d[j] + clamp(a - j, 0, 1) * (d[j + 1] - d[j]))
        shift = b.biascorrected[h] - b.mean_draws[h]
        @test sb.lower[h] ≈ tq(0.05) + shift
        @test sb.upper[h] ≈ tq(0.95) + shift
    end
    # uncorrected band: Q_q(draws) + ŝ - m̄ (AppSimulation.m:325-329)
    su = summarize(b; biascorrected = false)
    @test su.estimate == f.share
    @test su.lower ≈ sb.lower .- (b.biascorrected .- f.share)
    @test su.upper ≈ sb.upper .- (b.biascorrected .- f.share)
    sn = summarize(b; band = :normal, level = 0.80)
    @test sn.se == b.se_pw && sn.prewhiten === true
    @test sn.lower ≈ b.biascorrected .+ quantile(Normal(), 0.10) .* b.se_pw
    @test sn.upper ≈ b.biascorrected .+ quantile(Normal(), 0.90) .* b.se_pw
    @test summarize(b; band = :normal, prewhiten = false).se == b.se
    @test summarize(b; band = :normal, biascorrected = false).lower ≈
          f.share .- z .* b.se_pw
    @test occursin("Bias-corr.", sprint(show, MIME"text/plain"(), sb))
    @test !occursin("Bias-corr.", sprint(show, MIME"text/plain"(), su))
    @test_throws ArgumentError summarize(b; band = :hall)
    @test_throws ArgumentError summarize(b; level = 0)

    # a horizon without draws (a :simulation horizon whose covariance is NaN)
    # gives a NaN band there and leaves the others untouched
    bs = fevdbootstrap(f; method = :simulation, nboot = 200, rng = StableRNG(2))
    vals = Any[getfield(bs, k) for k in fieldnames(FEVDBootstrap)]
    D = copy(bs.draws)
    D[:, 2] .= NaN
    vals[findfirst(==(:draws), fieldnames(FEVDBootstrap))] = D
    bnan = FEVDBootstrap(vals...)
    snan = summarize(bnan)
    @test isnan(snan.lower[2]) && isnan(snan.upper[2])
    @test snan.lower[[1; 3:(H + 1)]] ≈ summarize(bs).lower[[1; 3:(H + 1)]]

    # --- plot recipes -----------------------------------------------------------
    p = plot(f)
    @test p isa Plots.Plot && length(p.series_list) == 1
    @test length(plot(f; levels = [0.68, 0.90], prewhiten = false).series_list) == 2
    pb = plot(b)                                   # bias-corrected + estimate
    @test length(pb.series_list) == 2
    lo, hi = Plots.ylims(pb)
    @test lo <= 0 && hi >= 1
    @test length(plot(b; biascorrected = false).series_list) == 1
    @test plot(b; levels = [0.68, 0.90], band = :normal) isa Plots.Plot
    @test plot(bs) isa Plots.Plot
    @test_throws ArgumentError plot(f; levels = [1.5])

    # --- as_irf_result ------------------------------------------------------------
    r = as_irf_result(f)
    @test r isa LocalProjectionIRFResult
    @test r.coverage == [0.68, 0.90]
    @test vec(r.data) == f.share && vec(r.stderr) == f.se_pw
    @test vec(r.upper[2]) ≈ summarize(f; level = 0.90).upper
    @test r.metadata.fevd && r.metadata.quantity === :fevd_share
    @test r.metadata.estimator === :r2 && !r.metadata.bias_corrected
    rb = as_irf_result(b; coverage = [0.90])
    @test vec(rb.data) == b.biascorrected && vec(rb.stderr) == b.se_boot
    @test vec(rb.lower[1]) == sb.lower && vec(rb.upper[1]) == sb.upper
    @test rb.metadata.bootstrap && rb.metadata.bootstrap_method === :var
    @test rb.metadata.share == f.share && rb.metadata.pseudo_truth == b.pseudo_truth
    rn = as_irf_result(b; band = :normal, biascorrected = false)
    @test vec(rn.data) == f.share && rn.metadata.band === :normal
end
