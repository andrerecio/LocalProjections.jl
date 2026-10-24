# Reduced-form VAR primitives and lag-order selection (ports of the lp_var_nberma
# MATLAB suite), ported from the andrerecio fork onto 0.2.

@testitem "lagselect matches direct ic_var.m port" tags = [:lagselect, :verification] begin
    using LocalProjections
    using DataFrames, LinearAlgebra, StableRNGs, Test

    rng = StableRNG(20260831)
    T = 300
    sh = zeros(T)
    y = zeros(T)
    for t in 3:T
        sh[t] = 0.5 * sh[t - 1] + randn(rng)
        y[t] = 0.7 * y[t - 1] - 0.35 * y[t - 2] + 0.8 * sh[t] + 0.4 * sh[t - 1] + randn(rng)
    end
    df = DataFrame(shock = sh, y = y)
    Y = Matrix(df[!, [:shock, :y]])

    # Direct port of _estim/ic_var.m: criteria over p = 1:p_max with the penalty
    # denominator fixed at T - p_max, but Sigma_p estimated on the full sample.
    n = 2
    pmax = 8
    denom = T - pmax
    aic_ref = Float64[]
    bic_ref = Float64[]
    for p in 1:pmax
        v = LocalProjections._var_ols(Y, p)
        ld = logdet(Symmetric(v.Σu))
        k = n^2 * p + n
        push!(aic_ref, ld + 2 * k / denom)
        push!(bic_ref, ld + k * log(denom) / denom)
    end

    sel = lagselect(df, [:shock, :y]; maxlags = pmax, criterion = :aic)
    @test sel.aic ≈ aic_ref rtol = 1.0e-12
    @test sel.bic ≈ bic_ref rtol = 1.0e-12
    @test sel.selected == argmin(aic_ref)
    @test nlags(sel) == sel.selected
    @test sel.lags == collect(1:pmax)
    @test sel.nobs == T

    selb = lagselect(df, [:shock, :y]; maxlags = pmax, criterion = :bic)
    @test selb.selected == argmin(bic_ref)
    # Both criteria are always computed; only the selection differs.
    @test selb.aic == sel.aic
    @test selb.bic == sel.bic
    # BIC never selects a longer lag order than AIC here (heavier penalty).
    @test selb.selected <= sel.selected

    # Any Tables.jl table works, e.g. a NamedTuple of vectors
    @test lagselect((shock = sh, y = y), [:shock, :y]; maxlags = pmax).aic == sel.aic
end

@testitem "_var_ols matches var_estim.m conventions and MacroEconometricTools" tags = [
    :lagselect, :verification,
] begin
    using LocalProjections
    using LinearAlgebra, StableRNGs, Test
    import MacroEconometricTools as MET

    rng = StableRNG(11)
    T = 300
    Y = zeros(T, 2)
    A1 = [0.5 0.0; 0.4 0.8]
    for t in 2:T
        Y[t, :] = A1 * Y[t - 1, :] + randn(rng, 2)
    end

    p = 2
    v = LocalProjections._var_ols(Y, p)
    @test v.T == T
    @test v.Tu == T - p
    @test size(v.A) == (2, 2, p)
    @test size(v.U) == (T - p, 2)

    # var_estim.m: Sigmahat = res'res / (T_u - (n*p + 1)), intercept included in
    # the regressor count.
    k = 2 * p + 1
    @test v.Σu ≈ (v.U' * v.U) ./ ((T - p) - k) rtol = 1.0e-12

    # Cross-check the hand-rolled VAR against the hub package.
    m = MET.VAR(Y, p)
    @test v.A ≈ m.coefficients.lags rtol = 1.0e-10
    @test v.c ≈ m.coefficients.intercept rtol = 1.0e-10
    @test v.Σu ≈ Matrix(m.Σ) rtol = 1.0e-10

    # Residuals are orthogonal to the regressors.
    @test maximum(abs, sum(v.U; dims = 1)) < 1.0e-9
end

@testitem "lagselect API and guards" tags = [:lagselect, :api] begin
    using LocalProjections
    using DataFrames, StableRNGs, Test

    rng = StableRNG(4)
    T = 120
    df = DataFrame(shock = randn(rng, T), y = randn(rng, T))

    sel = lagselect(df, [:shock, :y]; maxlags = 6)
    @test sel isa VARLagSelection
    @test nlags(sel) in 1:6

    tbl = DataFrame(sel)
    @test names(tbl) == ["lags", "aic", "bic"]
    @test nrow(tbl) == 6

    io = IOBuffer()
    show(io, sel)
    @test occursin("VARLagSelection", String(take!(io)))
    show(io, MIME"text/plain"(), sel)
    out = String(take!(io))
    @test occursin("ic_var", out)
    @test occursin("Selected", out)

    @test_throws "`criterion` must be :aic or :bic" lagselect(df, [:shock, :y]; criterion = :hq)
    @test_throws "`maxlags` must be at least 1" lagselect(df, [:shock, :y]; maxlags = 0)
    @test_throws "not smaller than the sample size" lagselect(df, [:shock, :y]; maxlags = T)
    @test_throws "not a column of the data" lagselect(df, [:shock, :nope])
    @test_throws "duplicate names" lagselect(df, [:shock, :shock])
    @test_throws "at least one variable" lagselect(df, Symbol[])
    @test_throws "Tables.jl table" lagselect(randn(10, 2), [:a, :b])

    dfm = copy(df)
    dfm.y = Vector{Union{Missing, Float64}}(dfm.y)
    dfm.y[10] = missing
    @test_throws "contains `missing`" lagselect(dfm, [:shock, :y])

    dfn = copy(df)
    dfn.y[10] = NaN
    @test_throws "contains `NaN`" lagselect(dfn, [:shock, :y])
end

@testitem "Pope correction matches var_biascorr.m and its closed form" tags = [
    :bootstrap, :verification,
] begin
    using LocalProjections
    using LinearAlgebra, Test

    # Pope (1990) for a univariate AR(1): the scaled negative bias b equals
    # exactly 1 + 3*rho. This is an external check, not a re-port.
    for ρ in (0.0, 0.3, 0.5, 0.7, 0.9, 0.95)
        T = 1000.0
        Ac, δ = LocalProjections._pope_biascorrect(fill(ρ, 1, 1), fill(2.7, 1, 1), T)
        @test δ == 1.0
        @test (Ac[1, 1] - ρ) * T ≈ 1 + 3ρ rtol = 1.0e-10
    end

    # Safeguard 1: an explosive OLS companion skips the correction entirely.
    Ac, δ = LocalProjections._pope_biascorrect(fill(1.05, 1, 1), fill(1.0, 1, 1), 100.0)
    @test δ == 0.0
    @test Ac[1, 1] == 1.05

    # Safeguard 2: delta backs off in steps of 0.01 until stability is restored.
    # rho = 0.99, T = 20 would give 0.99 + (1 + 3*0.99)/20 = 1.1885 (explosive).
    Ac, δ = LocalProjections._pope_biascorrect(fill(0.99, 1, 1), fill(1.0, 1, 1), 20.0)
    @test 0 < δ < 1
    @test abs(Ac[1, 1]) <= 1

    # Direct port of _estim/var_biascorr.m on a bivariate VAR(2) with complex
    # companion eigenvalues.
    A = hcat([0.5 -0.6; 0.6 0.5], [0.1 0.0; 0.0 0.1])
    Σ = [1.0 0.3; 0.3 2.0]
    Tn = 200.0
    n, np = size(A)
    Acomp = [A; Matrix{Float64}(I, np - n, np - n) zeros(np - n, n)]
    G = zeros(np, np)
    G[1:n, 1:n] = Σ
    Γ0 = reshape((I - kron(Acomp, Acomp)) \ vec(G), np, np)
    At = Matrix(Acomp')
    # NOTE: the middle term squares the TRANSPOSE, not A'A.
    aux = inv(I - At) + At * inv(I - At * At) + sum(λ * inv(I - λ * At) for λ in eigvals(Acomp))
    b = real.(G * (aux / Γ0))
    ref = (Acomp + b ./ Tn)[1:n, :]

    got, δ2 = LocalProjections._pope_biascorrect(A, Σ, Tn)
    @test δ2 == 1.0
    @test got ≈ ref rtol = 1.0e-10

    # The 3-D method is the same correction, reshaped.
    A3 = Array{Float64}(undef, 2, 2, 2)
    A3[:, :, 1] = A[:, 1:2]
    A3[:, :, 2] = A[:, 3:4]
    got3, δ3 = LocalProjections._pope_biascorrect(A3, Σ, Tn)
    @test δ3 == δ2
    @test got3[:, :, 1] ≈ ref[:, 1:2] rtol = 1.0e-12
    @test got3[:, :, 2] ≈ ref[:, 3:4] rtol = 1.0e-12
end

@testitem "VAR simulation, impact vector and IRF recursion" tags = [
    :bootstrap, :verification,
] begin
    using LocalProjections
    using LinearAlgebra, StableRNGs, Statistics, Test

    # _var_simulate is var_sim.m: initial rows verbatim, then the VAR recursion.
    c = [0.1, -0.2]
    A = Array{Float64}(undef, 2, 2, 2)
    A[:, :, 1] = [0.5 0.1; 0.0 0.6]
    A[:, :, 2] = [0.1 0.0; 0.2 -0.1]
    rng = StableRNG(5)
    Tu = 50
    Ustar = randn(rng, Tu, 2)
    Yinit = randn(rng, 2, 2)
    Y = LocalProjections._var_simulate(c, A, Ustar, Yinit)
    @test size(Y) == (2 + Tu, 2)
    @test Y[1:2, :] == Yinit
    for t in 3:(2 + Tu)
        expected = c .+ A[:, :, 1] * Y[t - 1, :] .+ A[:, :, 2] * Y[t - 2, :] .+ Ustar[t - 2, :]
        @test Y[t, :] ≈ expected rtol = 1.0e-12
    end

    # _var_impact with innov_index = 1 is Sigma[:,1]/Sigma[1,1] = L[:,1]/L11.
    rngu = StableRNG(6)
    U = randn(rngu, 500, 3) * [1.0 0.0 0.0; 0.4 1.0 0.0; -0.3 0.2 1.0]'
    ν = LocalProjections._var_impact(U, 1)
    S = (U' * U) ./ size(U, 1)
    @test ν ≈ S[:, 1] ./ S[1, 1] rtol = 1.0e-10
    L = cholesky(Symmetric(S)).L
    @test ν ≈ L[:, 1] ./ L[1, 1] rtol = 1.0e-10
    @test ν[1] ≈ 1.0 rtol = 1.0e-12   # unit-impact normalization

    # _var_irf is the var_ir.m recursion Theta_h = sum_l A_l Theta_{h-l}.
    H = 6
    irf = LocalProjections._var_irf(A, [1.0, 0.0], H)
    Θ = [Matrix{Float64}(I, 2, 2)]
    for h in 1:H
        M = zeros(2, 2)
        for l in 1:min(h, 2)
            M += A[:, :, l] * Θ[h - l + 1]
        end
        push!(Θ, M)
    end
    for h in 0:H
        @test irf[:, h + 1] ≈ Θ[h + 1] * [1.0, 0.0] rtol = 1.0e-12
    end
    @test LocalProjections._var_ma(A, H)[:, :, H + 1] ≈ Θ[H + 1] rtol = 1.0e-12
end
