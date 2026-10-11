# Analytical, deterministic and API checks of the Gorodnichenko–Lee (2020) FEVD
# estimators (`fevd`, `fevdbootstrap`): spec §6.B and the argument guards.

@testitem "fevd checks: VAR FEVD by brute force (companion powers, level forecast error)" tags = [
    :fevd, :verification] begin
    using LocalProjections
    using LinearAlgebra, Random, StableRNGs, Statistics, Test
    const LPm = LocalProjections

    # A stable VAR(2) in (x, Δy, z); the response Δy is cumulated to the level y
    A = zeros(3, 3, 2)
    A[:, :, 1] = [0.4 0.1 -0.1; 0.3 0.2 0.15; 0.1 -0.2 0.5]
    A[:, :, 2] = [-0.1 0.0 0.05; 0.1 0.15 0.0; 0.0 0.1 -0.2]
    Ω = [1.0 0.4 -0.2; 0.4 1.5 0.3; -0.2 0.3 0.8]
    n, H = 3, 10
    L = cholesky(Symmetric(Ω)).L
    F = [A[:, :, 1] A[:, :, 2]; Matrix(1.0I, n, n) zeros(n, n)]
    @test maximum(abs, eigvals(F)) < 1

    # Brute force without _var_ma: Ψ_h = [F^h]_{1:n,1:n} L
    Ψ = [(F ^ h)[1:n, 1:n] * L for h in 0:H]
    for r in 1:n, j in 1:n

        s = LPm._var_fevd(A, Ω, H; shock = j, response = r)
        bf = [sum(Ψ[l + 1][r, j]^2 for l in 0:h) /
              sum(sum(abs2, Ψ[l + 1][r, :]) for l in 0:h) for h in 0:H]
        @test s ≈ bf atol=1e-12
    end
    # shares across all shocks sum to one; invariance to a scalar rescaling of Ω
    for r in 1:n, cumulate in (false, true)

        S = reduce(hcat,
            [LPm._var_fevd(A, Ω, H; shock = j, response = r, cumulate) for j in 1:n])
        @test vec(sum(S; dims = 2)) ≈ ones(H + 1) atol=1e-12
        @test all(s -> 0 <= s <= 1, S)
        for c in (1e-3, 0.25, 40.0)
            @test reduce(hcat,
                [LPm._var_fevd(A, c .* Ω, H; shock = j, response = r, cumulate)
                 for j in 1:n]) ≈ S atol=1e-12
        end
    end

    # Cumulated row: FEVD of the forecast error of y_{t+h} - y_{t-1}, y = Σ Δy, from
    # an augmented state (Y_t, Y_{t-1}, y_t) with y_t = y_{t-1} + Δy_t. The level
    # forecast error is Σ_{i ≤ h} G_i u_{t+h-i}, G_i the level row of F̃^i.
    r = 2
    Fa = zeros(2n + 1, 2n + 1)
    Fa[1:(2n), 1:(2n)] = F
    Fa[2n + 1, 1:(2n)] = F[r, :]           # y_t = y_{t-1} + Δy_t, Δy_t from the VAR
    Fa[2n + 1, 2n + 1] = 1.0
    Ba = zeros(2n + 1, n)
    Ba[1:n, :] = L
    Ba[2n + 1, :] = L[r, :]                # Δy_t's impact enters the level too
    G = [(Fa ^ i * Ba)[2n + 1, :] for i in 0:H]
    for j in 1:n
        bf = [sum(G[i + 1][j]^2 for i in 0:h) / sum(sum(abs2, G[i + 1]) for i in 0:h)
              for h in 0:H]
        @test LPm._var_fevd(A, Ω, H; shock = j, response = r, cumulate = true) ≈ bf atol=1e-12
        @test !(LPm._var_fevd(A, Ω, H; shock = j, response = r) ≈ bf)
    end

    # The same by simulation: propagate independent structural shocks dated
    # t, ..., t+h through the VAR from a zero history (the forecast error does not
    # depend on the history) and decompose the variance of y_{t+h} - y_{t-1}.
    rng = StableRNG(31)
    hs = 4
    N = 100_000
    contrib = zeros(N, n)
    for k in 1:N, j in 1:n

        Yp = zeros(n, hs + 3)               # columns: t-2, t-1, t, ..., t+hs
        ε = randn(rng, hs + 1)
        for s in 0:hs
            c = s + 3
            Yp[:, c] = A[:, :, 1] * Yp[:, c - 1] + A[:, :, 2] * Yp[:, c - 2] +
                       L[:, j] .* ε[s + 1]
        end
        contrib[k, j] = sum(Yp[r, 3:end])  # y_{t+hs} - y_{t-1}
    end
    v = vec(var(contrib; dims = 1))
    s_mc = v ./ sum(v)
    s_an = [LPm._var_fevd(A, Ω, hs; shock = j, response = r, cumulate = true)[end]
            for j in 1:n]
    @test s_mc ≈ s_an atol=0.01             # MC s.d. of each share ≲ 0.002
end

@testitem "fevd checks: delta-method gradients by finite differences" tags = [
    :fevd, :verification] begin
    using LocalProjections
    using DataFrames, LinearAlgebra, StableRNGs, StatsModels, Statistics, Test
    const LPm = LocalProjections

    rng = StableRNG(20261020)
    T = 220
    x = randn(rng, T)
    z = zeros(T)
    dy = zeros(T)
    for t in 2:T
        z[t] = 0.5 * z[t - 1] + 0.2 * dy[t - 1] + randn(rng)
        dy[t] = 0.3 * dy[t - 1] + 0.8 * x[t] + 0.4 * x[t - 1] + 0.3 * z[t] + randn(rng)
    end
    df = DataFrame(x = x .+ 0.3, z = z, dy = dy)
    H = 4
    fm = @formula(cumul(dy) ~ x + z + lags(x, 2) + lags(z, 2) + lags(dy, 2))
    fq = @formula(cumul(dy) ~ x + lags(x, 1) + lags(dy, 1))

    # Central finite-difference gradient of a scalar map
    function fdgrad(g, θ)
        map(eachindex(θ)) do i
            δ = 1e-6 * max(1.0, abs(θ[i]))
            θp = copy(θ)
            θm = copy(θ)
            θp[i] += δ
            θm[i] -= δ
            (g(θp) - g(θm)) / (2δ)
        end
    end
    quad(g, V) = sqrt(dot(g, V * g))

    # --- R²: s(θ) = θ₁'θ₂ / θ₃ with θ = (θ₁, θ₂, θ₃) ----------------------------
    f = fevd(fm, df; horizon = H, estimator = :r2)
    for h in 0:H
        k = h + 1
        sr2(θ) = dot(θ[1:k], θ[(k + 1):(2k)]) / θ[end]
        θ = f.theta[h + 1]
        @test sr2(θ) ≈ f.share[h + 1] rtol=1e-12
        g = fdgrad(sr2, θ)
        # analytic ξ = (θ₂, θ₁, -ŝ)/θ₃ against the finite difference
        @test g ≈ vcat(θ[(k + 1):(2k)], θ[1:k], -f.share[h + 1]) ./ θ[end] rtol=1e-6
        @test f.se[h + 1] ≈ quad(g, f.avar[h + 1]) rtol=1e-6
        @test f.se_pw[h + 1] ≈ quad(g, f.avar_pw[h + 1]) rtol=1e-6
    end

    # --- LP-A / LP-B: θ = (vec(b₀, …, b_h), σ²x, σ²_h), ψ_j = θ[1 + j·Nq] ------
    for est in (:lpa, :lpb), irf_formula in (nothing, fq)

        fl = fevd(fm, df; horizon = H, estimator = est, irf_formula)
        Nq = irf_formula === nothing ? 9 : 4
        for h in 0:H
            θ = fl.theta[h + 1]
            @test length(θ) == Nq * (h + 1) + 2
            @test θ[end - 1] == fl.shock_var
            @test θ[end] == (est === :lpa ? fl.mse[h + 1] : fl.sigma2v[h + 1])
            @test [θ[1 + j * Nq] for j in 0:h] == fl.irf[1:(h + 1)]
            function slp(θ)
                num = θ[end - 1] * sum(j -> abs2(θ[1 + j * Nq]), 0:h)
                est === :lpa ? num / θ[end] : num / (num + θ[end])
            end
            @test slp(θ) ≈ fl.share[h + 1] rtol=1e-12
            g = fdgrad(slp, θ)
            ga = LPm._gl_lp_gradient(est, fl.irf[1:(h + 1)], fl.shock_var, θ[end],
                fl.mse[h + 1], fl.share[h + 1], Nq)
            @test ga≈g rtol=1e-6 atol=1e-9
            @test fl.se[h + 1] ≈ quad(g, fl.avar[h + 1]) rtol=1e-6
            @test fl.se_pw[h + 1] ≈ quad(g, fl.avar_pw[h + 1]) rtol=1e-6
        end
    end
end

@testitem "fevd checks: block bootstrap degeneracy (blocklength = T_h)" tags = [
    :fevd] begin
    using LocalProjections
    using DataFrames, StableRNGs, StatsModels, Statistics, Test

    rng = StableRNG(20261021)
    T = 150
    x = randn(rng, T)
    dy = zeros(T)
    for t in 2:T
        dy[t] = 0.4 * dy[t - 1] + 0.7 * x[t] + 0.3 * x[t - 1] + randn(rng)
    end
    df = DataFrame(x = x, dy = dy)
    fm = @formula(cumul(dy) ~ x + lags(x, 2) + lags(dy, 2))

    # horizon 0: the only admissible block start is t0, so every draw is the sample
    f0 = fevd(fm, df; horizon = 0)
    b0 = fevdbootstrap(f0; method = :block, blocklength = f0.nobs[1], nboot = 25,
        rng = StableRNG(1))
    @test b0.nfail == 0 && size(b0.draws) == (25, 1)
    @test all(==(f0.share[1]), b0.draws)
    # the mean of identical draws may be off by an ulp (summation rounding)
    @test b0.mean_draws ≈ f0.share atol=1e-14
    @test b0.biascorrected ≈ f0.share atol=1e-14
    @test b0.se_boot[1] <= 1e-14
    s0 = summarize(b0)
    @test s0.lower ≈ f0.share atol=1e-14
    @test s0.upper ≈ f0.share atol=1e-14
    @test_throws ArgumentError fevdbootstrap(f0; method = :block,
        blocklength = f0.nobs[1] + 1)

    # horizon H: blocklength = T_H pins the draws at h = H only
    H = 6
    f = fevd(fm, df; horizon = H)
    b = fevdbootstrap(f; method = :block, blocklength = f.nobs[end], nboot = 25,
        rng = StableRNG(2))
    @test all(==(f.share[end]), b.draws[:, end])
    @test b.biascorrected[end] ≈ f.share[end] atol=1e-14
    @test b.se_boot[end] <= 1e-14
    @test all(>(0), b.se_boot[1:H])             # earlier horizons still resample
    # blocklength = 1 is the iid row bootstrap: the draws vary at every horizon
    b1 = fevdbootstrap(f; method = :block, blocklength = 1, nboot = 25, rng = StableRNG(3))
    @test all(>(0), b1.se_boot)
end

@testitem "fevd checks: population FEVD of the bivariate DGP2 (BiVarPopulation.m)" tags = [
    :fevd, :verification] begin
    using LocalProjections
    using DataFrames, LinearAlgebra, StableRNGs, StatsModels, Statistics, Test

    # Code defaults of BiVarPopulation.m / BiVarDataGen.m for dgpidx = 2 (the
    # headers disagree, spec Q32): σx = 3, gy = 0.5, ρp = 0.9, σp = 1.5, ρa = σa = 0.
    σx, gy, ρp, σp = 3.0, 0.5, 0.9, 1.5
    ρa, σa = 0.0, 0.0
    H = 10

    # BiVarPopulation.m §3: Kalman gain from the Riccati equation, iterated to
    # convergence (MATLAB's `dare`), then psie = cumsum([1; H'F^(l-1)K]) σe
    Fk = [ρp 0 0; 0 ρa -σa; 0 0 0]
    Bk = [σp 0; 0 σa; 0 1]
    Hk = [1.0, 1.0, 0.0]
    Q = Bk * Bk'
    riccati(P) = Fk * (P - (P * Hk) * (P * Hk)' ./ dot(Hk, P * Hk)) * Fk' + Q
    P = foldl((P, _) -> riccati(P), 1:500; init = Q)
    @test riccati(P) ≈ P atol=1e-12
    K = Fk * P * Hk ./ dot(Hk, P * Hk)
    σe = sqrt(dot(Hk, P * Hk))
    psie = cumsum(vcat(1.0, [dot(Hk, Fk^(l - 1) * K) for l in 1:H])) .* σe
    psix = σx .* ρp .^ (0:H)                     # DGP2: psix = 0.9^x σx (ρp happens to be 0.9 too)
    s_code = cumsum(psix .^ 2) ./ (cumsum(psix .^ 2) .+ cumsum(psie .^ 2))
    # closed form: with σa = 0 the filter is trivial (K = (ρp, 0, 0), σe = σp)
    ψx = [3 * 0.9^i for i in 0:H]
    ψe = [1.5 * sum(0.9^j for j in 0:i) for i in 0:H]
    s_pop = cumsum(ψx .^ 2) ./ (cumsum(ψx .^ 2) .+ cumsum(ψe .^ 2))
    @test σe ≈ 1.5 rtol=1e-12
    @test K ≈ [0.9, 0.0, 0.0] atol=1e-12
    @test s_code ≈ s_pop rtol=1e-12
    @test s_pop[1] ≈ 0.8                         # 9 / (9 + 2.25)

    # BiVarDataGen.m, dgpidx = 2: xpart AR(1) in σx·w₁, Δp AR(1) in σp·w₂ plus the
    # drift gy, y = xpart + p; x = σx·w₁. One long sample.
    rng = StableRNG(424242)
    T, burn = 20_000, 100
    N = T + burn + 1
    wn = randn(rng, N, 2)
    xpart = zeros(N)
    dp = zeros(N)
    for t in 2:N
        xpart[t] = 0.9 * xpart[t - 1] + σx * wn[t, 1]
        dp[t] = ρp * dp[t - 1] + σp * wn[t, 2]
    end
    p = cumsum(dp) .+ (0:(N - 1)) .* gy
    y = xpart[(end - T + 1):end] .+ p[(end - T + 1):end]
    x = σx .* wn[(end - T + 1):end, 1]
    df = DataFrame(x = x, y = y)

    # BiSimulation.m conventions: ylevel = 0 (lags of Δy), lagy = lagx = L,
    # XtIndicator = 1. Δy_t = 0.9Δy_{t-1} + x_t - x_{t-1} + σp w₂ + c, so one lag of
    # each spans the information set; L = 2.
    fm = @formula(ldiff(y) ~ x + lags(x, 2) + lags(firstdiff(y), 2))
    # Tolerance: in 30 replications at T = 20 000 (seeds 1001-1030, horizons 0-12)
    # the bias was below 0.002 and the s.d. below 0.008 (R²), 0.0075 (LP-A/B);
    # 0.03 is about 4 s.d.
    for est in (:r2, :lpa, :lpb)
        f = fevd(fm, df; horizon = H, estimator = est)
        @test f.share ≈ s_pop atol=0.03
        @test mean(abs.(f.share .- s_pop)) < 0.01
        # unit-shock response 0.9^h, within 4 Newey–West standard errors
        if est !== :r2
            @test all(<(0.1), f.irf_se)          # keeps the check below informative
            @test all(abs.(f.irf .- 0.9 .^ (0:H)) .<= 4 .* f.irf_se)
        end
    end
end

@testitem "fevd checks: exogenous-shock VAR, estimators converge to the VAR FEVD" tags = [
    :fevd, :verification] begin
    using LocalProjections
    using DataFrames, LinearAlgebra, StableRNGs, StatsModels, Statistics, Test
    const LPm = LocalProjections

    # VAR(1) in (x, Δy, z) with x iid (first row of A zero) and ordered first:
    # Y_t = A Y_{t-1} + B ε_t, B lower triangular, so the first Cholesky shock of
    # Ω = BB' is x/B₁₁ and the LP estimands equal the VAR FEVD of the level y.
    A = zeros(3, 3, 1)
    A[:, :, 1] = [0 0 0; 0.4 0.3 0.1; 0.2 0.1 0.5]
    B = [1.0 0 0; 0.8 1.0 0; 0.3 0.4 0.9]
    Ω = B * B'
    H = 8
    s_var = LPm._var_fevd(A, Ω, H; shock = 1, response = 2, cumulate = true)
    @test cholesky(Symmetric(Ω)).L ≈ B

    rng = StableRNG(777)
    T, burn = 20_000, 200
    Y = zeros(T + burn, 3)
    for t in 2:(T + burn)
        Y[t, :] = A[:, :, 1] * Y[t - 1, :] + B * randn(rng, 3)
    end
    Y = Y[(burn + 1):end, :]
    df = DataFrame(x = Y[:, 1] .+ 1.0, dy = Y[:, 2], z = Y[:, 3])
    fm = @formula(cumul(dy) ~ x + lags(x, 2) + lags(dy, 2) + lags(z, 2))

    # Tolerance: in 30 replications at T = 20 000 (seeds 2001-2030) the bias was
    # below 0.002 and the s.d. at most 0.009 (R², LP-B) and 0.012 (LP-A, h = 8);
    # 0.04 is above 3 s.d.
    for est in (:r2, :lpa, :lpb)
        f = fevd(fm, df; horizon = H, estimator = est)
        @test f.share ≈ s_var atol=0.04
        @test mean(abs.(f.share .- s_var)) < 0.015
    end
    # The fitted VAR's FEVD (the bootstrap pseudo-truth) converges as well
    v = LPm._var_ols(Y, 1)
    s_hat = LPm._var_fevd(v.A, (v.U' * v.U) ./ v.Tu, H; shock = 1, response = 2,
        cumulate = true)
    @test s_hat ≈ s_var atol=0.03
    # LP-B unit-shock responses: the level response to a unit x, which moves the
    # reduced-form residuals by B[:, 1]: Σ_{l ≤ h} (Θ_l B[:, 1])[2]
    Θ = LPm._var_ma(A, H)
    fb = fevd(fm, df; horizon = H, estimator = :lpb)
    ψ = cumsum([(Θ[:, :, h + 1] * B[:, 1])[2] for h in 0:H])
    @test all(<(0.05), fb.irf_se)
    @test all(abs.(fb.irf .- ψ) .<= 4 .* fb.irf_se)
end

@testitem "fevd checks: trend and dofadjust conventions" tags = [:fevd] begin
    using LocalProjections
    using DataFrames, StableRNGs, StatsModels, Statistics, Test

    rng = StableRNG(20261025)
    T = 160
    x = randn(rng, T)
    dy = zeros(T)
    for t in 2:T
        dy[t] = 0.4 * dy[t - 1] + 0.7 * x[t] + 0.3 * x[t - 1] + randn(rng)
    end
    df = DataFrame(x = x, dy = dy, tt = Float64.(1:T), tt2 = Float64.(1:T) .^ 2)
    H = 5
    fm = @formula(cumul(dy) ~ x + lags(x, 2) + lags(dy, 2))
    # `trend = k` adds t, …, t^k on the absolute row index: with the intercept,
    # lags(tt, 1) = t - 1 (and lags(tt2, 1) = (t - 1)²) span the same columns
    for est in (:r2, :lpa, :lpb)
        a1 = fevd(fm, df; horizon = H, estimator = est, trend = 1)
        b1 = fevd(@formula(cumul(dy) ~ x + lags(x, 2) + lags(dy, 2) + lags(tt, 1)), df;
            horizon = H, estimator = est)
        a2 = fevd(fm, df; horizon = H, estimator = est, trend = 2)
        b2 = fevd(
            @formula(cumul(dy) ~ x + lags(x, 2) + lags(dy, 2) + lags(tt, 1) +
                                 lags(tt2, 1)),
            df; horizon = H, estimator = est)
        @test a1.t0 == b1.t0 == 3
        @test a1.share ≈ b1.share atol=1e-10
        @test a1.se ≈ b1.se atol=1e-10
        @test a2.share ≈ b2.share atol=1e-10
        @test a2.se_pw ≈ b2.se_pw atol=1e-10
        @test a1.trend == 1 && a2.trend == 2
        @test !(a1.share ≈ fevd(fm, df; horizon = H, estimator = est).share)
    end
    # dofadjust: LP-A/B variances over T_h - k_W (k_W = 5 here: two lags of x and
    # of Δy plus the intercept; the date-t shock is not counted) instead of T_h
    for est in (:lpa, :lpb)
        d = fevd(fm, df; horizon = H, estimator = est)
        c = fevd(fm, df; horizon = H, estimator = est, dofadjust = false)
        ratio = est === :lpa ? c.mse ./ d.mse : c.sigma2v ./ d.sigma2v
        @test ratio ≈ (d.nobs .- 5) ./ d.nobs rtol=1e-12
        @test c.irf == d.irf && !c.dofadjust
    end
    # R² has no degrees-of-freedom correction
    @test fevd(fm, df; horizon = H, dofadjust = false).share ==
          fevd(fm, df; horizon = H).share
end

@testitem "fevd checks: formula and data guards" tags = [:fevd, :api] begin
    using LocalProjections
    using DataFrames, StableRNGs, StatsModels, Statistics, Test

    rng = StableRNG(20261022)
    T = 120
    x = randn(rng, T)
    z = randn(rng, T)
    dy = zeros(T)
    for t in 2:T
        dy[t] = 0.3 * dy[t - 1] + 0.6 * x[t] + 0.2 * z[t] + randn(rng)
    end
    df = DataFrame(x = x, z = z, dy = dy, y = cumsum(dy), s = Float64.(z .> 0))
    H = 4
    ok = @formula(cumul(dy) ~ x + lags(x, 2) + lags(dy, 2))
    @test fevd(ok, df; horizon = H) isa LPFEVD
    @test fevd(@formula(ldiff(y) ~ x + lags(x, 2) + lags(firstdiff(y), 2)), df;
        horizon = H) isa LPFEVD

    # Rejected left-hand sides
    for fm in (@formula(leads(y) ~ x + lags(x, 2)),
        @formula(anchor(y, z) ~ x + lags(x, 2)),
        @formula(leads(y) | z ~ x + lags(x, 2)),
        @formula(hbr(dy, y) ~ x + lags(x, 2)),
        @formula(cumul(ldiff(y)) ~ x + lags(x, 2)),
        @formula(cumul(hbr(dy, y)) ~ x + lags(x, 2)),
        @formula(y ~ x + lags(x, 2)))
        for est in (:r2, :lpa, :lpb)
            @test_throws ArgumentError fevd(fm, df; horizon = H, estimator = est)
        end
    end

    # Rejected right-hand sides: interactions, functions, leads, missing intercept
    for fm in (@formula(cumul(dy) ~ x + z + x & z + lags(x, 2)),
        @formula(cumul(dy) ~ x * z + lags(x, 2)),
        @formula(cumul(dy) ~ x + lags(x, 2) & z),
        @formula(cumul(dy) ~ x + log(abs(z) + 1) + lags(x, 2)),
        @formula(cumul(dy) ~ x + leads(z) + lags(x, 2)),
        @formula(cumul(dy) ~ x + firstdiff(z) + lags(x, 2)),
        @formula(cumul(dy) ~ 0 + x + lags(x, 2) + lags(dy, 2)),
        @formula(cumul(dy) ~ -1 + x + lags(x, 2) + lags(dy, 2)),
        @formula(cumul(dy) ~ x + lags(x, 0) + lags(dy, 2)),
        @formula(cumul(dy) ~ lags(x, 2) + lags(dy, 2)))      # no date-t variable
        for est in (:r2, :lpb)
            @test_throws ArgumentError fevd(fm, df; horizon = H, estimator = est)
        end
    end

    # Shock, estimator, horizon and keyword guards
    fz = @formula(cumul(dy) ~ x + z + lags(x, 2) + lags(dy, 2))
    @test fevd(fz, df; horizon = H, shock = :z).shock === :z
    @test fevd(fz, df; horizon = H, shock = :z, estimator = :lpb).contemporaneous ==
          [:z, :x]
    @test_throws ArgumentError fevd(fz, df; horizon = H, shock = :dy)  # only lagged
    @test_throws ArgumentError fevd(fz, df; horizon = H, shock = :w)   # not a column
    @test_throws ArgumentError fevd(@formula(cumul(dy) ~ x + lags(w, 2)), df; horizon = H)
    @test_throws ArgumentError fevd(ok, df; horizon = -1)
    @test_throws ArgumentError fevd(ok, df; horizon = T)               # too few rows
    @test_throws ArgumentError fevd(ok, df; horizon = H, estimator = :var)
    @test_throws ArgumentError fevd(ok, df; horizon = H, trend = -1)
    @test_throws ArgumentError fevd(ok, df; horizon = H, trend = 4)
    fq = @formula(cumul(dy) ~ x + lags(x, 1))
    @test_throws ArgumentError fevd(ok, df; horizon = H, irf_formula = fq)  # R²
    @test fevd(ok, df; horizon = H, estimator = :lpb, irf_formula = fq).t0lp == 2
    @test_throws ArgumentError fevd(ok, df; horizon = H, estimator = :lpb,
        irf_formula = @formula(ldiff(y) ~ x + lags(x, 1)))             # other LHS
    @test_throws ArgumentError fevd(ok, df; horizon = H, estimator = :lpb,
        irf_formula = @formula(cumul(dy) ~ z + lags(x, 1)))            # shock absent

    # Data with gaps: missing, NaN or Inf in any column used
    for col in (:x, :dy), bad in (missing, NaN, Inf)

        d = allowmissing(copy(df))
        d[60, col] = bad
        for est in (:r2, :lpb)
            @test_throws ArgumentError fevd(ok, d; horizon = H, estimator = est)
        end
    end
    d = allowmissing(copy(df))
    d[1, :z] = missing                       # a column the formula does not use
    @test fevd(ok, d; horizon = H).share == fevd(ok, df; horizon = H).share
    d = copy(df)
    d.dy = string.(d.dy)
    # a non-numeric column is rejected before StatsModels.schema sees it
    @test_throws ArgumentError fevd(ok, d; horizon = H)

    # LP results that fevd cannot use
    miv = lpiv(@formula(leads(y) ~ (x ~ z)), df; horizon = H)
    @test_throws ArgumentError fevd(miv, df)
    mst = lp(@formula(cumul(dy) ~ x + lags(dy, 2)), df; horizon = H, state = :s)
    @test mst.state !== nothing
    @test_throws ArgumentError fevd(mst, df)
end

@testitem "fevd checks: fevdbootstrap guards, reproducibility and skipfirst" tags = [
    :fevd, :api] begin
    using LocalProjections
    using DataFrames, Random, StableRNGs, StatsModels, Statistics, Test
    const LPm = LocalProjections

    rng = StableRNG(20261023)
    T = 160
    x = randn(rng, T)
    z = zeros(T)
    dy = zeros(T)
    for t in 2:T
        z[t] = 0.5 * z[t - 1] + 0.2 * dy[t - 1] + randn(rng)
        dy[t] = 0.3 * dy[t - 1] + 0.8 * x[t] + 0.4 * x[t - 1] + 0.3 * z[t] + randn(rng)
    end
    df = DataFrame(x = x, z = z, dy = dy)
    H = 5
    fm = @formula(cumul(dy) ~ x + lags(x, 2) + lags(z, 2) + lags(dy, 2))
    fq = @formula(cumul(dy) ~ x + lags(x, 1) + lags(dy, 1))
    fs = Dict(:r2 => fevd(fm, df; horizon = H, estimator = :r2),
        :lpa => fevd(fm, df; horizon = H, estimator = :lpa, irf_formula = fq),
        :lpb => fevd(fm, df; horizon = H, estimator = :lpb, irf_formula = fq))

    # --- guards ------------------------------------------------------------------
    for t in (1, 2)
        ft = fevd(fm, df; horizon = H, trend = t)
        @test_throws ArgumentError fevdbootstrap(ft, df; method = :var, nboot = 5)
        @test fevdbootstrap(ft; method = :block, nboot = 5, rng = StableRNG(1)) isa
              FEVDBootstrap
    end
    for est in (:lpa, :lpb)
        @test_throws ArgumentError fevdbootstrap(fs[est]; method = :block, nboot = 5)
    end
    for nb in (1, 0, -1)                    # the draws' std needs nboot ≥ 2
        @test_throws ArgumentError fevdbootstrap(fs[:r2], df; nboot = nb)
        @test_throws ArgumentError fevdbootstrap(fs[:r2]; method = :block, nboot = nb)
        @test_throws ArgumentError fevdbootstrap(fs[:r2]; method = :simulation, nboot = nb)
    end
    @test_throws ArgumentError fevdbootstrap(fs[:r2]; method = :block, blocklength = 0)
    @test_throws ArgumentError fevdbootstrap(fs[:r2], df; burnin = -1, nboot = 5)
    @test_throws ArgumentError fevdbootstrap(fs[:r2], df; vars = [:x, :dy, :w], nboot = 5)
    @test_throws ArgumentError fevdbootstrap(fs[:r2], df; vars = [:x, :x, :z, :dy],
        nboot = 5)
    @test_throws ArgumentError fevdbootstrap(fs[:r2], df; method = :simulation_var)
    # nboot = 2 works for every procedure
    for (est, f) in fs
        @test size(fevdbootstrap(f, df; nboot = 2, nlags = 2, rng = StableRNG(1)).draws) ==
              (2, H + 1)
        @test size(fevdbootstrap(f; method = :simulation, nboot = 2,
            rng = StableRNG(1)).draws) == (2, H + 1)
    end
    @test size(fevdbootstrap(fs[:r2]; method = :block, nboot = 2,
        rng = StableRNG(1)).draws) == (2, H + 1)
    # the wrapper's se_pw/se shadow the LPFEVD's: no duplicate property names
    @test allunique(propertynames(fevdbootstrap(fs[:lpb], df; nboot = 2, nlags = 2,
        rng = StableRNG(1))))

    # failed draws report the first exception, not just a count
    err = try
        LPm._fevd_draw_moments(fill(NaN, 2, H + 1), [1, 1],
            Any[ErrorException("boom"), nothing], "test")
        nothing
    catch e
        e
    end
    @test err isa ErrorException && occursin("all 2 draws failed", err.msg) &&
          occursin("first error: boom", err.msg)
    @test_logs (:warn, r"1 of 2 draws failed.*first error: boom") LPm._fevd_draw_moments(
        zeros(2, H + 1), [1, 0], Any[ErrorException("boom"), nothing], "test")

    # prewhitening infeasible at long horizons: the warning's remedy depends on the
    # estimator (R²: only a shorter horizon; LP-A/B: a parsimonious irf_formula)
    logs, _ = Test.collect_test_logs() do
        fevd(fm, df; horizon = 60)
    end
    @test length(logs) == 1 && occursin("cannot be prewhitened", logs[1].message)
    @test occursin("shorter `horizon`", logs[1].message) &&
          !occursin("irf_formula", logs[1].message)
    flong = @test_logs (:warn, r"cannot be prewhitened.*irf_formula") fevd(fm, df;
        horizon = 20, estimator = :lpb)
    infeasible = findall(isnan, flong.se_pw) .- 1
    @test !isempty(infeasible) && all(isfinite, flong.se[infeasible .+ 1])
    # :simulation skips those horizons with one warning, and does not misreport them
    # as a negative delta-method variance
    bl = @test_logs (:warn, r"not available at horizons") fevdbootstrap(flong;
        method = :simulation, nboot = 50, rng = StableRNG(3))
    @test all(isnan, bl.biascorrected[infeasible .+ 1])
    @test all(isnan, bl.se[infeasible .+ 1])

    # --- threaded == serial, same seed == same result, every method ----------------
    numeric = (:draws, :mean_draws, :pseudo_truth, :biascorrected, :se_boot, :se_pw,
        :se, :irf_draws, :irf_bc, :irf_se_boot, :irfstr_draws, :nlags, :nfail)
    same(a, b) = all(k -> isequal(getfield(a, k), getfield(b, k)), numeric)
    combos = [(:r2, :var), (:r2, :block), (:r2, :simulation), (:lpa, :var),
        (:lpa, :simulation), (:lpb, :var), (:lpb, :simulation)]
    for (est, method) in combos
        f = fs[est]
        data = method === :var ? df : nothing
        kw = (; method, nboot = 30, nlags = 1:3)
        a = fevdbootstrap(f, data; kw..., rng = StableRNG(11), threaded = false)
        b = fevdbootstrap(f, data; kw..., rng = StableRNG(11), threaded = true)
        c = fevdbootstrap(f, data; kw..., rng = StableRNG(11), threaded = false)
        d = fevdbootstrap(f, data; kw..., rng = StableRNG(12), threaded = false)
        @test same(a, b)
        @test same(a, c)
        @test a.draws != d.draws
        @test all(isfinite, a.draws) && a.nfail == 0
        # a Xoshiro master stream works as well and is reproducible
        @test fevdbootstrap(f, data; kw..., rng = Xoshiro(5)).draws ==
              fevdbootstrap(f, data; kw..., rng = Xoshiro(5), threaded = true).draws
    end

    # --- skipfirst drops exactly the first row from the VAR --------------------------
    f = fs[:r2]
    vars = [:x, :z, :dy]
    Y = Matrix{Float64}(df[:, vars])
    for (skip, Yv) in ((false, Y), (true, Y[2:end, :]))
        b = fevdbootstrap(f, df; nboot = 3, nlags = 2, skipfirst = skip, rng = StableRNG(1))
        @test b.vars == vars && b.skipfirst == skip
        v = LPm._var_ols(Yv, 2)
        @test b.pseudo_truth ≈ LPm._var_fevd(v.A, (v.U' * v.U) ./ v.Tu, H; shock = 1,
            response = 3, cumulate = true) rtol=1e-12
        bi = fevdbootstrap(f, df; nboot = 3, nlags = 1:4, skipfirst = skip,
            rng = StableRNG(1))
        @test bi.ic.values ≈ LPm._gl_lagselect(Yv, collect(1:4)).values rtol=1e-12
    end
    b0 = fevdbootstrap(f, df; nboot = 3, nlags = 2, rng = StableRNG(1))
    b1 = fevdbootstrap(f, df; nboot = 3, nlags = 2, skipfirst = true, rng = StableRNG(1))
    @test b0.pseudo_truth != b1.pseudo_truth

    # non-DataFrame tables are accepted
    @test fevdbootstrap(f, (; x = df.x, z = df.z, dy = df.dy); nboot = 3, nlags = 2,
        rng = StableRNG(1)).draws == b0.draws
end

@testitem "fevd checks: method = :var with lags(firstdiff(y), p) and a cumul(dy) response" tags = [
    :fevd, :api] begin
    using LocalProjections
    using DataFrames, Random, StableRNGs, StatsModels, Statistics, Test
    const LPm = LocalProjections

    rng = StableRNG(20261010)
    T = 200
    x = randn(rng, T)
    w = zeros(T)
    dy = zeros(T)
    for t in 2:T
        w[t] = 0.5 * w[t - 1] + 0.2 * dy[t - 1] + randn(rng)
        dy[t] = 0.3 * dy[t - 1] + 0.8 * x[t] + 0.4 * x[t - 1] + 0.3 * w[t] + randn(rng)
    end
    df = DataFrame(x = x, w = w, dy = dy, y = accumulate(+, dy))
    H = 6
    # the lagged response formed from the level (the reference's lagy/lagyLP) next
    # to the same specification with the growth column
    fdiff = @formula(cumul(dy) ~ x + lags(x, 2) + lags(firstdiff(y), 2) + lags(w, 2))
    fdy = @formula(cumul(dy) ~ x + lags(x, 2) + lags(dy, 2) + lags(w, 2))
    for est in (:r2, :lpb)
        f = fevd(fdiff, df; horizon = H, estimator = est)
        g = fevd(fdy, df; horizon = H, estimator = est)
        @test f.t0 == g.t0 + 1                   # firstdiff(y) starts one row later
        @test :y in f.variables
        for nlags in (1:4, 2)
            # y is not simulated: each draw rebuilds it by cumulating dy, so the VAR,
            # its lag choice and the pseudo-truth are those of the dy specification
            b = fevdbootstrap(f, df; method = :var, nboot = 20, nlags = nlags,
                rng = StableRNG(1))
            c = fevdbootstrap(g, df; method = :var, nboot = 20, nlags = nlags,
                rng = StableRNG(1))
            @test b.vars == c.vars == [:x, :dy, :w]
            @test b.nlags == c.nlags
            @test b.pseudo_truth ≈ c.pseudo_truth rtol=1e-12
            @test b.nfail == 0 && all(isfinite, b.draws)
            @test b.draws != c.draws               # different estimator rows (t0)
        end
        # an explicit order without y works; with y the VAR would be singular
        @test fevdbootstrap(f, df; method = :var, nboot = 3, nlags = 2,
            vars = [:x, :w, :dy], rng = StableRNG(1)).vars == [:x, :w, :dy]
        @test_throws ArgumentError fevdbootstrap(f, df; method = :var, nboot = 3,
            vars = [:x, :dy, :w, :y])
        # a level next to its difference is rejected with an ArgumentError (not a
        # DomainError from logdet), whether the lag order is fixed or selected
        for nlags in (2, 1:4)
            @test_throws ArgumentError fevdbootstrap(g, df; method = :var, nboot = 3,
                nlags = nlags, vars = [:x, :dy, :w, :y])
        end
    end
    @test_throws ArgumentError LPm._gl_lagselect(Matrix(df[:, [:x, :dy, :y]]), [1, 2])
end

@testitem "fevd checks: fevd(lp_result) and display" tags = [:fevd, :api] begin
    using LocalProjections
    using DataFrames, StableRNGs, StatsModels, Statistics, Test

    rng = StableRNG(20261024)
    T = 150
    x = randn(rng, T)
    dy = zeros(T)
    for t in 2:T
        dy[t] = 0.4 * dy[t - 1] + 0.7 * x[t] + 0.3 * x[t - 1] + randn(rng)
    end
    df = DataFrame(x = x, w = randn(rng, T), dy = dy, y = cumsum(dy))
    H = 5
    fq = @formula(cumul(dy) ~ x + lags(x, 1) + lags(dy, 1))
    fields = (:estimator, :horizon, :response, :shock, :contemporaneous, :T, :t0, :t0lp,
        :nobs, :window, :bandwidth, :dofadjust, :trend, :share, :irf, :irf_se,
        :shock_mean, :shock_var, :mse, :sigma2v, :forecast_errors, :se_pw, :se, :theta,
        :avar_pw, :avar, :lhs, :variables)
    for fm in (@formula(cumul(dy) ~ x + w + lags(x, 2) + lags(dy, 2)),
        @formula(ldiff(y) ~ x + lags(x, 2) + lags(firstdiff(y), 2)))
        m = lp(fm, df; horizon = H)
        for est in (:r2, :lpa, :lpb), dofadjust in (true, false)

            a = fevd(m, df; estimator = est, dofadjust)
            b = fevd(fm, df; horizon = H, estimator = est, dofadjust)
            @test all(k -> isequal(getfield(a, k), getfield(b, k)), fields)
            @test a.formula == b.formula
        end
        a = fevd(m, df; estimator = :lpb, trend = 1)
        @test a.share == fevd(fm, df; horizon = H, estimator = :lpb, trend = 1).share
    end
    # shock chosen in lp() carries over
    fm = @formula(cumul(dy) ~ x + w + lags(x, 2) + lags(dy, 2))
    m = lp(fm, df; horizon = H, shock = :w)
    @test fevd(m, df; estimator = :lpb).shock === :w
    @test fevd(m, df; estimator = :lpb).share ==
          fevd(fm, df; horizon = H, estimator = :lpb, shock = :w).share
    m = lp(fm, df; horizon = H)
    a = fevd(m, df; estimator = :lpa, irf_formula = fq)
    @test a.share == fevd(fm, df; horizon = H, estimator = :lpa, irf_formula = fq).share

    # show / summarize on every result type run and mention what they display
    plain(x) = sprint(show, MIME"text/plain"(), x)
    for est in (:r2, :lpa, :lpb)
        f = fevd(fm, df; horizon = H, estimator = est,
            irf_formula = est === :r2 ? nothing : fq)
        @test occursin("LPFEVD", sprint(show, f))
        @test occursin("Share", plain(f))
        s = summarize(f)
        @test occursin("FEVDSummary", sprint(show, s)) && occursin("Lower", plain(s))
        methods = est === :r2 ? (:var, :block, :simulation) : (:var, :simulation)
        for method in methods
            b = fevdbootstrap(f, method === :var ? df : nothing; method, nboot = 20,
                nlags = 2, rng = StableRNG(1))
            @test occursin("FEVDBootstrap", sprint(show, b))
            @test occursin("Bias-corr.", plain(b))
            for band in (:shift, :normal), bc in (true, false)

                sb = summarize(b; band, biascorrected = bc)
                @test sb isa FEVDSummary
                @test !isempty(plain(sb))
                @test size(DataFrame(sb)) == (H + 1, 6)
            end
        end
    end
end

@testitem "fevd checks: application values pinned from the authors' MATLAB code" tags = [
    :fevd, :verification] begin
    # Gorodnichenko & Lee (2020), Sec. 5 and App. H2. The numbers are outputs of
    # their MATLAB code (SubR2.m, SubLPA.m, SubLPB.m with the VAR-bootstrap
    # specification, and the IC / var_chol blocks of BootstrapVARR2.m) run under
    # GNU Octave 11.3, at h = 0, 4, ..., 20. The full fixture set behind them
    # (test/fixtures/goro_lee, local only) is compared in
    # testitems_fevd_octave.jl when present. MP cases: the LP rows start at
    # h = 4 because the h = 0 regression fits exactly (rounding noise).
    using LocalProjections
    using DataFrames, Random, StatsModels, Test

    function readdata(file)
        path = joinpath(@__DIR__, "..", "docs", "src", "data", file)
        lines = [l for l in eachline(path) if !isempty(strip(l))]
        hdr = Symbol.(split(lines[1], ','))
        cells = [split(l, ',') for l in lines[2:end]]
        cols = map(eachindex(hdr)) do j
            raw = [String(c[j]) for c in cells]
            v = tryparse.(Float64, raw)
            any(isnothing, v) ? raw : Vector{Float64}(v)
        end
        return DataFrame(cols, hdr)
    end
    data = Dict(:sec5 => readdata("gorodnichenko_lee.csv"),
        :rz => readdata("gorodnichenko_lee_rz.csv"))

    fm = Dict(
        :tfp_y => (:sec5,
            @formula(cumul(dy) ~
                     tfp + lags(tfp, 4) + lags(dy, 4) + lags(infl, 4) +
                     lags(rr, 4) + lags(effr, 4))),
        :tfp_pi => (:sec5,
            @formula(ldiff(infl) ~
                     tfp + lags(tfp, 4) + lags(dy, 4) + lags(infl, 4) +
                     lags(rr, 4) + lags(effr, 4))),
        :mp_y => (:sec5,
            @formula(cumul(dy) ~
                     rr + tfp + dy + infl + lags(rr, 4) + lags(tfp, 4) +
                     lags(dy, 4) + lags(infl, 4) + lags(effr, 4))),
        :mp_pi => (:sec5,
            @formula(ldiff(infl) ~
                     rr + tfp + dy + infl + lags(rr, 4) + lags(tfp, 4) +
                     lags(dy, 4) + lags(infl, 4) + lags(effr, 4))),
        :rz_y => (:rz,
            @formula(cumul(dy) ~
                     newsy + lags(newsy, 4) + lags(dy, 4) + lags(infl, 4) +
                     lags(tbill, 4))),
        :rz_pi => (:rz,
            @formula(ldiff(infl) ~
                     newsy + lags(newsy, 4) + lags(dy, 4) + lags(infl, 4) +
                     lags(tbill, 4))))

    r2 = Dict(
        :tfp_y => [0.04374653864229356, 0.05168354494090151, 0.10249163954445038,
            0.17901763795720427, 0.28032889335098954, 0.39058217459553957],
        :tfp_pi => [0.04312073521138736, 0.044368284913082784, 0.057351048051441346,
            0.11055504368623148, 0.2490751669840451, 0.29290816418403787],
        :mp_y => [0.0045551790708792785, 0.04148461610004755, 0.15452390038744437,
            0.22682150483613833, 0.1689712187778707, 0.2760595146904745],
        :mp_pi => [0.010483923182685165, 0.03496591612244138, 0.08411299116886288,
            0.3188984511074155, 0.4756447779742236, 0.5810981015245916],
        :rz_y => [0.02018954561668958, 0.1285357262706182, 0.2483243333383722,
            0.3343751340091258, 0.37319512854678855, 0.36483187508516535],
        :rz_pi => [0.005379763172305299, 0.06409279343362605, 0.06054769013227348,
            0.09048804751488292, 0.09701606325690744, 0.16357591963696458]
    )
    lpa = Dict(
        :tfp_y => [0.054862906130668423, 0.10010666774126921, 0.22459518209881454,
            0.3579534754092597, 0.4418754772068545, 0.45629454063992403],
        :tfp_pi => [0.054078080726149634, 0.10920226532396879, 0.1745180734059074,
            0.2546850186770864, 0.33286132352783115, 0.5578983897945862],
        :mp_y => [0.07381511954576923, 0.3261984762324546, 0.534192847044502,
            0.6519569667184074, 0.6814139232652479],
        :mp_pi => [0.023657048086282967, 0.12965913160254233,
            0.6510932558407346, 0.9475545126767448, 1.3521133001663193],
        :rz_y => [0.026419187818264075, 0.0778679026612578, 0.15609030372840796,
            0.2148300830371029, 0.227091116698903, 0.22548060243425105],
        :rz_pi => [0.007039731174010325, 0.045503397242230105, 0.05992274974749293,
            0.08763347729404433, 0.11521399076499161, 0.19021461106032744]
    )
    lpb = Dict(
        :tfp_y => [0.05415826533087517, 0.09390744799385915, 0.19464531366398746,
            0.28861624292336663, 0.3437448839839549, 0.3874038037642107],
        :tfp_pi => [0.05339332998869254, 0.10028183989998868, 0.14738728581436272,
            0.20572873904350805, 0.29567712515270067, 0.4135209645844347],
        :mp_y => [0.06719843567768029, 0.25315049144349305,
            0.35842924396725634, 0.3723022521609792, 0.4153675034101641],
        :mp_pi => [0.022871847414661754, 0.11136990067035656,
            0.43938059458301915, 0.5683467032710556, 0.6509968337805419],
        :rz_y => [0.02623951676808353, 0.079676734083178, 0.16024193179978705,
            0.22130311888149912, 0.24077795335614924, 0.24144894086955163],
        :rz_pi => [0.0070269101331091885, 0.045650383611939516, 0.05926854049905252,
            0.08692275403727052, 0.10978436865105574, 0.17849335696996615]
    )
    svar = Dict(
        :tfp_y => [0.04218167150558921, 0.025550708824615083, 0.043060274888680396,
            0.05767853537049154, 0.06969566505765504, 0.07911484398351479],
        :tfp_pi => [0.037943719724430454, 0.03251190923345107, 0.023690719082481035,
            0.019919524537610013, 0.018003296983733937, 0.016933585528766015],
        :mp_y => [0.0, 0.03035957077328295, 0.062031112596122796,
            0.08502387815644111, 0.10010923901653576, 0.108289634285993],
        :mp_pi => [0.0, 0.010529207546933766, 0.020679934489583357,
            0.02674922868019285, 0.030547661640901332, 0.03300268456034357],
        :rz_y => [0.024337418903309586, 0.07145350526300101, 0.09806661877485194,
            0.11250976517713984, 0.12137322141916765, 0.12737380039862373],
        :rz_pi => [0.005083127752757575, 0.022265480989980036, 0.023195050131966936,
            0.02326860773353832, 0.023235573308000277, 0.023197799891972192]
    )
    lag = Dict(:tfp_y => 2, :tfp_pi => 2, :mp_y => 2, :mp_pi => 2, :rz_y => 3, :rz_pi => 3)
    r2_se_pw_tfp_y = [0.029816110878584656, 0.04913411075390642, 0.09650437885003182,
        0.18227211202604307, 0.16532727722582868, 0.12522496883752232]
    lpb_irf_tfp_y = [0.21645138704537867, 0.48654339613852127, 0.7286477851028458,
        0.6615028251034278, 0.24265923900753883, -0.22594435637139504]
    lpb_irse_tfp_y = [0.07745426435516745, 0.19204005767699678, 0.24201449528558047,
        0.2653889304896865, 0.2602712783005131, 0.2703203555264047]
    lpb_lpset_tfp_y = [0.06388485515078499, 0.07371323781555877, 0.12003226861390084,
        0.14344264081057465, 0.1673136475187618, 0.22441851473683816]

    hs = 1:4:21
    quiet(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())
    for (name, (set, f0)) in fm
        df = data[set]
        lp_hs = startswith(String(name), "mp") ? hs[2:end] : hs
        f = fevd(f0, df; horizon = 20, estimator = :r2)
        @test f.share[hs] ≈ r2[name] rtol=1e-9
        fa = quiet(() -> fevd(f0, df; horizon = 20, estimator = :lpa))
        @test fa.share[lp_hs] ≈ lpa[name] rtol=1e-9
        fb = quiet(() -> fevd(f0, df; horizon = 20, estimator = :lpb))
        @test fb.share[lp_hs] ≈ lpb[name] rtol=1e-9
        # HQIC over 1:10 on G-L's VAR sample (rows 2:T when y enters differenced)
        b = fevdbootstrap(f, df; method = :var, skipfirst = f.lhs === :cumul,
            nboot = 2, rng = Xoshiro(1))
        @test b.nlags == lag[name]
        @test b.pseudo_truth[hs]≈svar[name] rtol=1e-9 atol=1e-14
        if name === :tfp_y
            @test f.se_pw[hs] ≈ r2_se_pw_tfp_y rtol=1e-9
            @test fb.irf[hs] ≈ lpb_irf_tfp_y rtol=1e-9
            @test fb.irf_se[hs] ≈ lpb_irse_tfp_y rtol=1e-9
            fl = fevd(f0, df; horizon = 20, estimator = :lpb,
                irf_formula = @formula(cumul(dy) ~
                                       tfp + lags(tfp, 1) +
                                       lags(firstdiff(y), 1)))
            @test fl.share[hs] ≈ lpb_lpset_tfp_y rtol=1e-9
        end
    end
end
