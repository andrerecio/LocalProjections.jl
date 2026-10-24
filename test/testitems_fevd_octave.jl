# Gorodnichenko–Lee (2020) FEVD estimators against the Octave fixtures.
#
# The fixtures in test/fixtures/goro_lee/ are outputs of the authors' MATLAB
# code (docs/src/code/goro_lee_r2/lib, unmodified) run under GNU Octave 11.3;
# see test/fixtures/goro_lee/README.md and synthetic/README.md. Unlike the
# in-test transcriptions of testitems_fevd_parity.jl, these are numbers produced
# by the reference implementation itself.
#
# The fixture CSVs are generated locally and not committed (see the README).
# The items are tagged :octave, and test/runtests.jl skips that tag when the
# fixtures are absent; a custom filter (e.g. `:fevd in ti.tags`) must exclude
# it in that case. The key application values are pinned in
# testitems_fevd_checks.jl, which always runs.
#
# Tolerances. Deterministic quantities are compared at rtol = 1e-10 (Octave's
# BLAS/LAPACK, its normal-equation solves and libm `pow` agree with MATLAB, and
# with the package's QR solves, to rounding: ~1e-12 relative). Array
# comparisons use `isapprox`, i.e. the norm of the difference relative to the
# norm of the reference. The exceptions are documented where they occur (the
# MP h = 0 perfect fit, compared with an absolute tolerance). Stochastic
# quantities (the bootstrap and simulation bias corrections) are compared up to
# Monte Carlo error, with tolerances derived in the last test item.

# JuliaFormatter (sciml) is switched off for this file: with both `tags = ...` and
# `setup = ...` on a @testitem it formats the whole item body as macro keyword
# arguments (`a = 1` becomes `a=1`). The file is formatted by hand.
#! format: off

@testmodule GLOctave begin
    using DataFrames, Statistics

    const DIR = normpath(joinpath(@__DIR__, "fixtures", "goro_lee"))
    const SYN = joinpath(DIR, "synthetic")
    const DATA = normpath(joinpath(@__DIR__, "..", "docs", "src", "data"))

    # DelimitedFiles is not in the test target, so the CSVs are parsed by hand.
    # Values are written as %.17g (round-trips Float64), NaN literally.
    parsefloat(s::AbstractString) = parse(Float64, strip(s))

    # Headerless numeric matrix (the synthetic fixtures)
    function readmat(path::AbstractString)
        rows = [parsefloat.(split(l, ',')) for l in eachline(path) if !isempty(strip(l))]
        return permutedims(reduce(hcat, rows))
    end
    syn(parts...) = readmat(joinpath(SYN, parts...))
    synvec(parts...) = vec(syn(parts...))
    synscalar(parts...) = only(syn(parts...))
    # one file per horizon, <stem>_h<h>.csv
    syncell(dir, stem, H) = [syn(dir, "$(stem)_h$h.csv") for h in 0:H]

    # CSV with a header line; columns that do not parse as numbers stay strings
    function readtable(path::AbstractString)
        lines = [l for l in eachline(path) if !isempty(strip(l))]
        hdr = Symbol.(strip.(split(lines[1], ',')))
        cells = [split(l, ',') for l in lines[2:end]]
        cols = map(eachindex(hdr)) do j
            raw = [String(c[j]) for c in cells]
            vals = tryparse.(Float64, raw)
            any(isnothing, vals) ? raw : Vector{Float64}(vals)
        end
        return DataFrame(cols, hdr)
    end
    fixture(name) = readtable(joinpath(DIR, name))

    # The application data (Section 5 and Appendix H2), docs/src/data
    sec5() = readtable(joinpath(DATA, "gorodnichenko_lee.csv"))
    rz() = readtable(joinpath(DATA, "gorodnichenko_lee_rz.csv"))

    # The synthetic data: y in levels (y = 10 + cumsum(dy)), X = [x z2 z3].
    # `dy` is diff(y), with dy(1) = y(1) - 10 (any finite value: the first
    # effective row is 5, so dy(1) never enters a regression).
    function syndata()
        y = synvec("data_y.csv")
        X = syn("data_X.csv")
        return DataFrame(y = y, dy = [y[1] - 10; diff(y)], x = X[:, 1], z2 = X[:, 2],
            z3 = X[:, 3])
    end

    # NaN-aware comparison of the T × (H+1) arrays that are NaN outside the window
    function nanapprox(a, b; rtol = 1e-10)
        ok = .!isnan.(b)
        return isequal(isnan.(a), isnan.(b)) && isapprox(a[ok], b[ok]; rtol = rtol)
    end

    # Run `f` with log messages suppressed (the expected `fevd` warnings about
    # infeasible prewhitening and exact fits)
    quiet(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())

    # blkdiag
    function blkdiag(A::AbstractMatrix, B::AbstractMatrix)
        M = zeros(size(A, 1) + size(B, 1), size(A, 2) + size(B, 2))
        M[1:size(A, 1), 1:size(A, 2)] = A
        M[(size(A, 1) + 1):end, (size(A, 2) + 1):end] = B
        return M
    end

    # The package lags the raw shock column and orders the information set as
    # [1, lags...]; the reference lags the demeaned shock and orders
    # [lags..., y lags, 1]. On the regression rows both span the same space,
    # Q_pkg = Q_ref * M, so b_pkg = M⁻¹ b_ref and θ_h = (vec(b_0..b_h), σ²x, σ²h)
    # maps with J = blkdiag(I_{h+1} ⊗ M⁻¹, I_2). M[1, :] = e₁' (the shock column
    # is the same demeaned x_t in both), so ψ_h and the shares are invariant.
    function reparam(Qref::AbstractMatrix, Qpkg::AbstractMatrix, rows, h::Int)
        M = Qref[rows, :] \ Qpkg[rows, :]
        J = blkdiag(kron(Matrix{Float64}(I_(h + 1)), inv(M)), Matrix{Float64}(I_(2)))
        return J, M
    end
    I_(n) = [i == j ? 1.0 : 0.0 for i in 1:n, j in 1:n]

    # ------------------------------------------------------------------
    # Application cases (README "Case arguments"; data mapping in
    # docs/src/data/gorodnichenko_lee_README.md). `fm` is the forecast-error
    # (and shared LP) specification, `fmlp` the parsimonious LP set of the
    # SimulationAVar* runs. ylevel = 0 (GDP): cumul(dy); the LP set's lagged
    # response is lags(firstdiff(y), 1) so that it starts at t0LP = 3 as the
    # reference's yendo = [NaN; diff(y)] does. ylevel = 1 (inflation): ldiff(infl).
    # ------------------------------------------------------------------
    using StatsModels, LocalProjections
    function app_case(name::Symbol)
        if name === :tfp_y
            return (; data = :sec5, case = 1, cumul = true, shock = :tfp, vpos = 1,
                fm = @formula(cumul(dy) ~
                              tfp + lags(tfp, 4) + lags(dy, 4) + lags(infl, 4) +
                              lags(rr, 4) + lags(effr, 4)),
                fmlp = @formula(cumul(dy) ~ tfp + lags(tfp, 1) + lags(firstdiff(y), 1)),
                Xcols = [:tfp, :dy, :infl, :rr, :effr], mp = false)
        elseif name === :tfp_pi
            return (; data = :sec5, case = 2, cumul = false, shock = :tfp, vpos = 1,
                fm = @formula(ldiff(infl) ~
                              tfp + lags(tfp, 4) + lags(dy, 4) +
                              lags(infl, 4) + lags(rr, 4) + lags(effr, 4)),
                fmlp = @formula(ldiff(infl) ~ tfp + lags(tfp, 1) + lags(infl, 1)),
                Xcols = [:tfp, :dy, :infl, :rr, :effr], mp = false)
        elseif name === :mp_y
            return (; data = :sec5, case = 3, cumul = true, shock = :rr, vpos = 4,
                fm = @formula(cumul(dy) ~
                              rr + tfp + dy + infl + lags(rr, 4) + lags(tfp, 4) +
                              lags(dy, 4) + lags(infl, 4) + lags(effr, 4)),
                fmlp = @formula(cumul(dy) ~ rr + tfp + dy + infl),
                Xcols = [:rr, :tfp, :dy, :infl, :effr], mp = true)
        elseif name === :mp_pi
            return (; data = :sec5, case = 4, cumul = false, shock = :rr, vpos = 4,
                fm = @formula(ldiff(infl) ~
                              rr + tfp + dy + infl + lags(rr, 4) +
                              lags(tfp, 4) + lags(dy, 4) + lags(infl, 4) +
                              lags(effr, 4)),
                fmlp = @formula(ldiff(infl) ~ rr + tfp + dy + infl),
                Xcols = [:rr, :tfp, :dy, :infl, :effr], mp = true)
        elseif name === :rz_y
            return (; data = :rz, case = 5, cumul = true, shock = :newsy, vpos = 1,
                fm = @formula(cumul(dy) ~
                              newsy + lags(newsy, 4) + lags(dy, 4) +
                              lags(infl, 4) + lags(tbill, 4)),
                fmlp = @formula(cumul(dy) ~ newsy + lags(newsy, 1) + lags(firstdiff(y), 1)),
                Xcols = [:newsy, :dy, :infl, :tbill], mp = false)
        elseif name === :rz_pi
            return (; data = :rz, case = 6, cumul = false, shock = :newsy, vpos = 1,
                fm = @formula(ldiff(infl) ~
                              newsy + lags(newsy, 4) + lags(dy, 4) +
                              lags(infl, 4) + lags(tbill, 4)),
                fmlp = @formula(ldiff(infl) ~ newsy + lags(newsy, 1) + lags(infl, 1)),
                Xcols = [:newsy, :dy, :infl, :tbill], mp = false)
        end
        error("unknown case $name")
    end
    const APP_CASES = (:tfp_y, :tfp_pi, :mp_y, :mp_pi, :rz_y, :rz_pi)
    # BootstrapVARR2's XVAR = X(:, XVAROrder) with y (or diff(y)) inserted at yPlace
    const XVAR_LAYOUT = Dict(:tfp_y => ([1, 3, 4, 5], 2), :tfp_pi => ([1, 2, 4, 5], 3),
        :mp_y => ([2, 4, 1, 5], 2), :mp_pi => ([2, 3, 1, 5], 3),
        :rz_y => ([1, 3, 4], 2), :rz_pi => ([1, 2, 4], 3))

    # ------------------------------------------------------------------
    # Monte Carlo tolerances for the stochastic fixtures (see the last test item)
    # ------------------------------------------------------------------
    const NSIG = 4.0
    quantile5(v, p) = quantile(v, p; alpha = 0.5, beta = 0.5)     # MATLAB `quantile`
    kurtosis(v) = (m = mean(v); mean((v .- m) .^ 4) / mean((v .- m) .^ 2)^2)
    # Sparsity 1/f(Q_q) of the draws at quantile q, by the difference quotient of
    # type-5 sample quantiles (Siddiqui), and the resulting Monte Carlo standard
    # deviation sqrt(q(1-q)/B)/f(Q_q) of a sample quantile from B draws
    function quantile_sd(v, q; δ = 0.02)
        lo, hi = max(q - δ, 0.0), min(q + δ, 1.0)
        return sqrt(q * (1 - q) / length(v)) * (quantile5(v, hi) - quantile5(v, lo)) /
               (hi - lo)
    end
end

@testitem "fevd octave: lrv_nw.m, prewhitening and the bandwidth table" tags = [
    :fevd, :verification, :octave] setup = [GLOctave] begin
    using LocalProjections
    using LinearAlgebra, Test
    O = GLOctave
    LPj = LocalProjections

    Z = O.syn("lrv", "Z.csv")                       # 60 × 3
    for L in (0, 1, 3)
        @test LPj._gl_lrv_nw(Z, L) ≈ O.syn("lrv", "lrv_L$L.csv") rtol=1e-12
    end

    # SubR2.m:214-220 on ZFull = Z, L = round(0.75 * 60^(1/3) - 1)
    L = Int(O.synscalar("lrv", "pw_L.csv"))
    @test L == 2
    @test LPj._gl_bandwidth(size(Z, 1)) == L
    @test LPj._gl_prewhitened_lrv(Z, L) ≈ O.syn("lrv", "pw_Omega.csv") rtol=1e-10
    # the VAR(1) step it is built on (QR here, normal equations in the reference)
    Z0, Z1 = Z[1:(end - 1), :], Z[2:end, :]
    A = Matrix(transpose(Z0 \ Z1))
    @test A ≈ O.syn("lrv", "pw_A.csv") rtol=1e-10
    @test Z1 - Z0 * A' ≈ O.syn("lrv", "pw_U.csv") rtol=1e-10

    # round(0.75 * T^(1/3) - 1) for T = 1..2000 (MATLAB round: half away from
    # zero; ^ rather than cbrt), including the tie at T = 8 and the near-ties at
    # T = 216 and 1000 that cbrt would round up
    bw = O.syn("lrv", "bandwidth.csv")
    @test size(bw) == (2000, 2)
    @test bw[:, 1] == 1:2000
    @test [LPj._gl_bandwidth(T) for T in 1:2000] == Int.(bw[:, 2])
    @test Int.(bw[[8, 216, 1000], 2]) == [1, 3, 6]
end

@testitem "fevd octave: SubR2.m on synthetic data (3 cases)" tags = [
    :fevd, :verification, :octave] setup = [GLOctave] begin
    using LocalProjections
    using DataFrames, LinearAlgebra, Statistics, StatsModels, Test
    O = GLOctave
    LPj = LocalProjections

    df = O.syndata()
    T = nrow(df)
    H = 8
    fm0 = @formula(cumul(dy) ~ x + lags(x, 4) + lags(z2, 2) + lags(z3, 3) +
                               lags(firstdiff(y), 2))
    fm0_ldiff = @formula(ldiff(y) ~ x + lags(x, 4) + lags(z2, 2) + lags(z3, 3) +
                                    lags(firstdiff(y), 2))
    fm1 = @formula(ldiff(y) ~ x + lags(x, 4) + lags(z2, 2) + lags(z3, 3) + lags(y, 2))
    # (fixture directory, trend, formulas expressing it)
    cases = [("r2_yl0_tr0", 0, [fm0, fm0_ldiff]), ("r2_yl1_tr0", 0, [fm1]),
        ("r2_yl0_tr1", 1, [fm0, fm0_ldiff])]

    for (dir, trend, forms) in cases
        vd = O.synvec(dir, "vd.csv")
        sepw = O.synvec(dir, "seAVarWPW.csv")
        se = O.synvec(dir, "seAVarWoPW.csv")
        idxt = Int.(O.syn(dir, "idxt.csv"))
        effT = Int.(O.synvec(dir, "effectiveT.csv"))
        fe = O.syn(dir, "forecastError.csv")
        θ = [vec(m) for m in O.syncell(dir, "theta", H)]
        Vpw = O.syncell(dir, "GOGTWPW", H)
        V = O.syncell(dir, "GOGTWoPW", H)
        @test idxt[:, 1] == fill(5, H + 1)

        for fm in forms
            f = fevd(fm, df; horizon = H, estimator = :r2, trend = trend)
            @test f.t0 == 5
            @test f.window == idxt
            @test f.nobs == effT
            @test f.bandwidth == [LPj._gl_bandwidth(n) for n in effT]
            @test f.share ≈ vd rtol=1e-10
            @test f.se_pw ≈ sepw rtol=1e-10
            @test f.se ≈ se rtol=1e-10
            @test f.mse ≈ [t[end] for t in θ] rtol=1e-10
            @test O.nanapprox(f.forecast_errors, fe)
            # θ = [θ1; θ2; θ3] and its covariance do not depend on how W is
            # parametrized (the forecast error is a residual)
            for h in 0:H
                @test f.theta[h + 1] ≈ θ[h + 1] rtol=1e-10
                @test f.avar_pw[h + 1] ≈ Vpw[h + 1] rtol=1e-10
                @test f.avar[h + 1] ≈ V[h + 1] rtol=1e-10
            end
        end

        # private core on the reference's own matrices (trend = 0: W is the
        # forecast-error set of the shared LP-A fixture without its two
        # contemporaneous columns, dhy is SubLPA's)
        trend == 0 || continue
        lpdir = replace(dir, "r2_" => "lpa_", "_tr0" => "_df0_shared")
        dhy = O.syn(lpdir, "dhy.csv")
        W = O.syn(lpdir, "rhsAllLP.csv")[:, 3:end]
        x = df.x .- mean(df.x)
        for h in 0:H
            rows = 5:(T - h)
            xf = [x[t + j] for t in rows, j in 0:h]
            r = LPj._gl_r2(dhy[rows, h + 1], W[rows, :], xf)
            @test r.s ≈ vd[h + 1] rtol=1e-10
            @test r.theta ≈ θ[h + 1] rtol=1e-10
            @test r.fe ≈ fe[rows, h + 1] rtol=1e-10
            @test r.avar_pw ≈ Vpw[h + 1] rtol=1e-10
            @test r.avar ≈ V[h + 1] rtol=1e-10
            @test r.se_pw ≈ sepw[h + 1] rtol=1e-10
            @test r.se ≈ se[h + 1] rtol=1e-10
        end
    end

    # SimulationAVarR2.m, deterministic prefix (θ, V_h and the delta-method s.e.
    # at ŝ; ylevel = 0, trend = 0). The driver demeans x a second time, so these
    # equal r2_yl0_tr0 up to ~2e-16.
    f = fevd(fm0, df; horizon = H, estimator = :r2)
    dir = "simavar_r2_yl0"
    @test f.share ≈ O.synvec(dir, "vd.csv") rtol=1e-10
    @test f.se_pw ≈ O.synvec(dir, "seAVarWPW.csv") rtol=1e-10
    @test f.se ≈ O.synvec(dir, "seAVarWoPW.csv") rtol=1e-10
    for (h, (t, Vp, Vn)) in enumerate(zip(O.syncell(dir, "theta", H),
        O.syncell(dir, "GOGTWPW", H), O.syncell(dir, "GOGTWoPW", H)))
        @test f.theta[h] ≈ vec(t) rtol=1e-10
        @test f.avar_pw[h] ≈ Vp rtol=1e-10
        @test f.avar[h] ≈ Vn rtol=1e-10
    end
end

@testitem "fevd octave: SubLPA.m / SubLPB.m on synthetic data (16 cases)" tags = [
    :fevd, :verification, :octave] setup = [GLOctave] begin
    using LocalProjections
    using DataFrames, LinearAlgebra, Statistics, StatsModels, Test
    O = GLOctave
    LPj = LocalProjections

    df = O.syndata()
    T = nrow(df)
    H = 8
    x = df.x .- mean(df.x)
    # XtIndicator = [1 1 0]: x_t and z2_t enter at date t
    fe_form = Dict(
        0 => @formula(cumul(dy) ~ x + z2 + lags(x, 4) + lags(z2, 2) + lags(z3, 3) +
                                  lags(firstdiff(y), 2)),
        1 => @formula(ldiff(y) ~ x + z2 + lags(x, 4) + lags(z2, 2) + lags(z3, 3) +
                                 lags(y, 2)))
    # separate LP set: lagyLP = 1, lagXLP = [1 0 0], XtIndicatorLP = [1 0 0]
    lp_form = Dict(0 => @formula(cumul(dy) ~ x + lags(x, 1) + lags(firstdiff(y), 1)),
        1 => @formula(ldiff(y) ~ x + lags(x, 1) + lags(y, 1)))

    for est in (:lpa, :lpb), ylevel in (0, 1), dfi in (0, 1), set in ("shared", "sep")
        dir = "$(est)_yl$(ylevel)_df$(dfi)_$(set)"
        rd(name) = O.syn(dir, "$name.csv")
        lpb = est === :lpb
        dof = dfi == 0          # the reference's dfIndicator = 0 is the d.o.f. adjustment
        sep = set == "sep"
        idxt = Int.(rd("idxt"))
        t0lp = Int(only(rd("t0LP")))
        ir, vd = vec(rd("ir")), vec(rd("vd"))
        irse, mse = vec(rd("irSeAVar")), vec(rd("mse"))
        fe = rd("forecastError")
        resid = rd("residualLP")
        Qref = rd("rhsAllLP")
        β = rd("BetaHat")
        @test idxt[:, 1] == fill(5, H + 1)
        @test t0lp == (sep ? (ylevel == 0 ? 3 : 2) : 5)

        # public estimator
        f = O.quiet() do
            fevd(fe_form[ylevel], df; horizon = H, estimator = est, dofadjust = dof,
                irf_formula = sep ? lp_form[ylevel] : nothing)
        end
        @test f.t0 == 5
        @test f.t0lp == t0lp
        @test f.window == idxt
        @test f.share ≈ vd rtol=1e-10
        @test f.irf ≈ ir rtol=1e-10
        @test f.irf_se ≈ irse rtol=1e-10
        @test f.mse ≈ mse rtol=1e-10
        @test f.shock_mean ≈ only(rd("meanx")) rtol=1e-12
        @test f.shock_var ≈ only(rd("varx")) rtol=1e-12
        @test O.nanapprox(f.forecast_errors, fe)
        @test O.nanapprox(f.design.dhy[2:end, :], rd("dhy")[2:end, :])
        if lpb
            @test f.sigma2v ≈ vec(rd("sigma2vHat")) rtol=1e-10
        else
            @test f.sigma2v === nothing
        end
        # LP coefficients in the package's parametrization: b_pkg = M⁻¹ b_ref
        Qpkg = sep ? f.design.q : hcat(f.design.C, f.design.W)
        Nq = size(Qpkg, 2)
        @test Nq == size(Qref, 2) == (sep ? 4 : 14)
        rows = t0lp:T
        _, M = O.reparam(Qref, Qpkg, rows, 0)
        @test Qref[rows, :] * M ≈ Qpkg[rows, :] rtol=1e-12
        @test M[1, :] ≈ [1.0; zeros(Nq - 1)] atol=1e-12
        for h in 0:H
            @test f.theta[h + 1][1:(Nq * (h + 1))] ≈ vec(M \ β[:, 1:(h + 1)]) rtol=1e-10
        end

        # private core on the reference's own matrices: [C, W] is the shared
        # case's rhsAllLP (= rhsAllFE), the separate LP set is this case's rhsAllLP
        RFE = O.syn("$(est)_yl$(ylevel)_df$(dfi)_shared", "rhsAllLP.csv")
        r = LPj._gl_lp(rd("dhy"), RFE[:, 1:2], RFE[:, 3:end], sep ? Qref : nothing, x,
            only(rd("varx")), 5, t0lp, H; estimator = est, dofadjust = dof)
        @test r.share ≈ vd rtol=1e-10
        @test r.psi ≈ ir rtol=1e-10
        @test r.irf_se ≈ irse rtol=1e-10
        @test r.mse ≈ mse rtol=1e-10
        @test r.beta_lp ≈ β rtol=1e-10
        @test O.nanapprox(r.fe, fe)
        @test O.nanapprox(r.resid_lp, resid)
        if lpb
            @test r.sigma2v ≈ vec(rd("sigma2vHat")) rtol=1e-10
            @test O.nanapprox(r.v, rd("vFE"))
        end
    end
end

@testitem "fevd octave: SimulationAVarLPA.m / SimulationAVarLPB.m deterministic prefix" tags = [
    :fevd, :verification, :octave] setup = [GLOctave] begin
    using LocalProjections
    using DataFrames, LinearAlgebra, Statistics, StatsModels, Test
    O = GLOctave
    LPj = LocalProjections

    df = O.syndata()
    T = nrow(df)
    H = 8
    fe_form = Dict(
        0 => @formula(cumul(dy) ~ x + z2 + lags(x, 4) + lags(z2, 2) + lags(z3, 3) +
                                  lags(firstdiff(y), 2)),
        1 => @formula(ldiff(y) ~ x + z2 + lags(x, 4) + lags(z2, 2) + lags(z3, 3) +
                                 lags(y, 2)))
    lp_form = Dict(0 => @formula(cumul(dy) ~ x + lags(x, 1) + lags(firstdiff(y), 1)),
        1 => @formula(ldiff(y) ~ x + lags(x, 1) + lags(y, 1)))

    # jointIndicator = 0, dfIndicator = 0, the parsimonious LP set (Nq = 4)
    for est in (:lpa, :lpb), ylevel in (0, 1)
        dir = "simavar_$(est)_yl$(ylevel)"
        rd(name) = O.syn(dir, "$name.csv")
        Nq = Int(only(rd("Nq")))
        t0lp = Int(only(rd("t0LP")))
        window = Int.(rd("window"))
        θ = [vec(m) for m in O.syncell(dir, "theta", H)]
        Vpw = O.syncell(dir, "GOGTWPW", H)
        V = O.syncell(dir, "GOGTWoPW", H)
        Δ = [vec(m) for m in O.syncell(dir, "Deltah", H)]
        fev = vec(rd("forecastErrorVar"))       # σ̂²_f (LP-A), m_h (LP-B)

        f = fevd(fe_form[ylevel], df; horizon = H, estimator = est,
            irf_formula = lp_form[ylevel])
        @test Nq == 4 == size(f.design.q, 2)
        @test f.t0lp == t0lp
        @test f.share ≈ vec(rd("vd")) rtol=1e-10
        @test f.irf ≈ vec(rd("ir")) rtol=1e-10
        @test f.irf_se ≈ vec(rd("irSeAVar")) rtol=1e-10
        @test f.shock_var ≈ only(rd("varx")) rtol=1e-12
        @test f.mse ≈ fev rtol=1e-10
        # delta-method standard errors at ŝ: invariant to the parametrization of q
        @test f.se_pw ≈ vec(rd("seAVarWPW")) rtol=1e-10
        @test f.se ≈ vec(rd("seAVarWoPW")) rtol=1e-10
        # AVar window [max(t0, t0LP), T - h, T_h]
        tmin = max(f.t0, t0lp)
        @test window == hcat(fill(tmin, H + 1), T .- (0:H), T .- (0:H) .- tmin .+ 1)

        # Σ̂_q and θ̂_h, V_h in the package's parametrization (see GLOctave.reparam);
        # the reference's q is the separate-LP-set SubLP fixture's rhsAllLP
        sub = "$(est)_yl$(ylevel)_df0_sep"
        Qref = O.syn(sub, "rhsAllLP.csv")
        Qpkg = f.design.q
        rows = t0lp:T
        Σq = rd("Sigmaq")
        @test Qref[rows, :]' * Qref[rows, :] ./ length(rows) ≈ Σq rtol=1e-12
        for h in 0:H
            J, M = O.reparam(Qref, Qpkg, rows, h)
            @test M' * Σq * M ≈ Qpkg[rows, :]' * Qpkg[rows, :] ./ length(rows) rtol=1e-10
            @test f.theta[h + 1] ≈ J * θ[h + 1] rtol=1e-10
            @test f.avar_pw[h + 1] ≈ J * Vpw[h + 1] * J' rtol=1e-10
            @test f.avar[h + 1] ≈ J * V[h + 1] * J' rtol=1e-10
            # the gradient (the reference's Deltah at vdBc = ŝ): it only loads on
            # ψ_j (the shock coefficient, identical in both parametrizations),
            # σ²x and σ²_h, so it is the same vector in both
            σ²h = f.theta[h + 1][end]
            g = LPj._gl_lp_gradient(est, f.irf[1:(h + 1)], f.shock_var, σ²h, f.mse[h + 1],
                f.share[h + 1], Nq)
            @test g ≈ Δ[h + 1] rtol=1e-10
            @test sqrt(dot(g, Vpw[h + 1] * g)) ≈ f.se_pw[h + 1] rtol=1e-10
            # private core on the reference's matrices reproduces the reference's θ, V
            err = est === :lpa ? O.syn(sub, "forecastError.csv")[:, h + 1] :
                  O.syn(sub, "vFE.csv")[:, h + 1]
            a = LPj._gl_lp_avar(Qref, O.syn(sub, "residualLP.csv"), err,
                df.x .- mean(df.x), only(rd("varx")), θ[h + 1][end],
                O.syn(sub, "BetaHat.csv"), Σq, f.t0, t0lp, h)
            @test a.theta ≈ θ[h + 1] rtol=1e-10
            @test a.avar_pw ≈ Vpw[h + 1] rtol=1e-10
            @test a.avar ≈ V[h + 1] rtol=1e-10
            @test a.nobs == window[h + 1, 3]
        end
    end
end

@testitem "fevd octave: var_chol.m, the IC block of BootstrapVARR2.m and var_bootstrap.m" tags = [
    :fevd, :verification, :octave] setup = [GLOctave] begin
    using LocalProjections
    using DataFrames, LinearAlgebra, Random, StatsModels, Test
    O = GLOctave
    LPj = LocalProjections

    # var_chol(input, 2, 8, 0, 1, 0, dIdx, 0, 0): input = [x(2:T), diff(y), z2(2:T)]
    Y = O.syn("var", "input.csv")
    n, p, H = 3, 2, 8
    v = LPj._var_ols(Y, p)
    Ω = (v.U' * v.U) ./ v.Tu                 # var_estimation.m divides by T_eff
    @test [n^2 * p, v.Tu] == Int.(vec(O.syn("var", "k_T.csv")))
    @test v.c ≈ O.synvec("var", "c.csv") rtol=1e-10
    @test hcat(v.A[:, :, 1], v.A[:, :, 2]) ≈ O.syn("var", "phi.csv") rtol=1e-10
    @test Ω ≈ O.syn("var", "omega.csv") rtol=1e-10
    @test v.U ≈ permutedims(O.syn("var", "residual.csv")) rtol=1e-10
    Θ = LPj._var_ma(v.A, H)
    Θref = O.syn("var", "psiReduced.csv")
    @test reduce(hcat, [Θ[:, :, h + 1] for h in 0:H]) ≈ Θref rtol=1e-10
    # one-s.d. responses to shock 1 (row 2 cumulated with dIdx = 2) and the FEVD
    # of every variable
    L = cholesky(Symmetric(Ω)).L
    ψ = reduce(hcat, [Θ[:, :, h + 1] * L[:, 1] for h in 0:H])
    @test ψ ≈ O.syn("var", "psi_nodiff.csv") rtol=1e-10
    ψd = copy(ψ)
    ψd[2, :] = cumsum(ψ[2, :])
    @test ψd ≈ O.syn("var", "psi_diff.csv") rtol=1e-10
    vd_nodiff = O.syn("var", "vd_nodiff.csv")
    vd_diff = O.syn("var", "vd_diff.csv")
    for r in 1:n
        @test LPj._var_fevd(v.A, Ω, H; shock = 1, response = r) ≈
              vd_nodiff[r, :] rtol=1e-10
        @test LPj._var_fevd(v.A, Ω, H; shock = 1, response = r, cumulate = r == 2) ≈
              vd_diff[r, :] rtol=1e-10
    end

    # IC block: XVAR = [x, y or Δy, z2, z3] (rows 2:T for ylevel = 0), candidates
    # 1:6, common sample of rows(XVAR) - 6, MLE Ω, k = n²p
    df = O.syndata()
    # (the VAR must simulate every variable of the specification, so the ylevel = 0
    # case lags dy itself; rows 2:T of dy are diff(y) exactly)
    fm = Dict(
        0 => @formula(cumul(dy) ~ x + lags(x, 4) + lags(z2, 2) + lags(z3, 3) + lags(dy, 2)),
        1 => @formula(ldiff(y) ~ x + lags(x, 4) + lags(z2, 2) + lags(z3, 3) + lags(y, 2)))
    for ylevel in (0, 1), crit in (:bic, :hqic, :aic)
        XVAR = O.syn("ic", "XVAR_yl$ylevel.csv")
        ic = O.synvec("ic", "ic_yl$(ylevel)_$crit.csv")
        lag, Tic = Int.(vec(O.syn("ic", "lag_yl$(ylevel)_$crit.csv")))
        @test size(XVAR, 1) == (ylevel == 0 ? 199 : 200)
        @test Tic == size(XVAR, 1) - 6
        sel = LPj._gl_lagselect(XVAR, collect(1:6); criterion = crit)
        @test sel.values ≈ ic rtol=1e-10
        @test sel.p == lag
        # public route: the VAR of fevdbootstrap(method = :var), with
        # skipfirst = true for the cumul(dy) response (G-L's rows 2:T)
        f = fevd(fm[ylevel], df; horizon = 2, estimator = :r2)
        vars = [:x, ylevel == 0 ? :dy : :y, :z2, :z3]
        b = fevdbootstrap(f, df; method = :var, vars = vars, nlags = 1:6,
            criterion = crit, skipfirst = ylevel == 0, nboot = 2, rng = Xoshiro(1))
        @test b.ic.values ≈ ic rtol=1e-10
        @test b.nlags == lag
    end
    @test Int(O.syn("ic", "lag_yl1_aic.csv")[1]) == 2

    # var_bootstrap(c, phi, input, residual, 200, 100) replayed with the recorded
    # idx/init through _var_simulate (spec §3.4): MATLAB state s is row p + s - 1,
    # residual idx(s) feeds it (idx(1) unused), the output rows TBurnIn+1:Tend are
    # rows p+TBurnIn : p+TBurnIn+Tout-1, and Tout = 200 even when the system has
    # 199 rows (case 1, Q34).
    for case in ("case1", "case2")
        rd(name) = O.syn("varboot", "$(case)_$name.csv")
        pb, Tout, burnin, Teff = Int.(vec(rd("meta")))
        Yv = rd("input")
        nv = size(Yv, 2)
        A = reshape(rd("phi"), nv, nv, pb)
        U = permutedims(rd("residual"))
        idx = Int.(vec(rd("idx")))
        init = Int(only(rd("init")))
        @test length(idx) == burnin + Tout
        @test 1 <= init <= size(Yv, 1) - pb + 1 == Teff + 1
        # the package's VAR fit reproduces the bootstrap DGP
        vb = LPj._var_ols(Yv, pb)
        @test vb.c ≈ vec(rd("c")) rtol=1e-10
        @test vb.A ≈ A rtol=1e-10
        @test vb.U ≈ U rtol=1e-10
        Yb = LPj._var_simulate(vec(rd("c")), A, U[idx[2:end], :],
            Yv[init:(init + pb - 1), :])
        @test size(Yb, 1) == pb + burnin + Tout - 1
        @test Yb[(pb + burnin):(pb + burnin + Tout - 1), :] ≈ rd("sim") rtol=1e-12
    end
end

@testitem "fevd octave: application R² (Sec. 5 and App. H2)" tags = [
    :fevd, :verification, :octave] setup = [GLOctave] begin
    using LocalProjections
    using DataFrames, LinearAlgebra, Statistics, StatsModels, Test
    O = GLOctave

    data = Dict(:sec5 => O.sec5(), :rz => O.rz())
    # docs/src/data holds the AppData.m / AppDataRZ.m matrices bit for bit, and
    # y = cumsum(Δy) summed left to right as MATLAB does
    d5 = O.fixture("app_data_sec5.csv")
    cols5 = [:time, :dtfp_util, :rr_shock, :dgdp, :dcons, :dinv, :effr, :infl]
    @test Matrix(d5[:, cols5]) ==
          Matrix(data[:sec5][:, [:quarter, :tfp, :rr, :dy, :dc, :di, :effr, :infl]])
    drz = O.fixture("app_data_rz.csv")
    @test Matrix(drz[:, [:time, :news, :dgdp, :infl, :unemp, :tbill]]) ==
          Matrix(data[:rz][:, [:quarter, :newsy, :dy, :infl, :unemp, :tbill]])
    @test data[:sec5].y == accumulate(+, data[:sec5].dy)
    @test data[:rz].y == accumulate(+, data[:rz].dy)

    # the GDP cases written with ldiff(y) instead of cumul(dy)
    fm_ldiff = Dict(
        :tfp_y => @formula(ldiff(y) ~ tfp + lags(tfp, 4) + lags(dy, 4) + lags(infl, 4) +
                                      lags(rr, 4) + lags(effr, 4)),
        :mp_y => @formula(ldiff(y) ~ rr + tfp + dy + infl + lags(rr, 4) + lags(tfp, 4) +
                                     lags(dy, 4) + lags(infl, 4) + lags(effr, 4)),
        :rz_y => @formula(ldiff(y) ~ newsy + lags(newsy, 4) + lags(dy, 4) + lags(infl, 4) +
                                     lags(tbill, 4)))
    for name in O.APP_CASES
        c = O.app_case(name)
        df = data[c.data]
        # the inputs the reference received are the columns of docs/src/data
        inp = O.fixture("app_$(name)_inputs.csv")
        @test Matrix(inp[:, 2:end]) == Matrix(df[:, c.Xcols])
        @test inp.y == (c.cumul ? df.y : df.infl)

        ref = O.fixture("app_$(name)_r2.csv")
        f = fevd(c.fm, df; horizon = 20, estimator = :r2)
        @test f.shock == c.shock
        @test f.window == Int.(hcat(ref.t0, ref.tend))
        @test f.nobs == Int.(ref.Th)
        @test f.share ≈ ref.vd rtol=1e-10
        @test f.se_pw ≈ ref.se_pw rtol=1e-10
        @test f.se ≈ ref.se_nopw rtol=1e-10
        # ldiff(y) on the sequential cumsum gives the reference's LHS bit for bit
        if haskey(fm_ldiff, name)
            fl = fevd(fm_ldiff[name], df; horizon = 20, estimator = :r2)
            @test fl.share ≈ ref.vd rtol=1e-10
            @test fl.se_pw ≈ ref.se_pw rtol=1e-10
        end
    end

    # TFP → y: θ_h, the GOGT matrices and the forecast errors of SubR2
    df = data[:sec5]
    f = fevd(O.app_case(:tfp_y).fm, df; horizon = 20, estimator = :r2)
    th = O.fixture("app_tfp_y_r2_theta.csv")
    g = O.fixture("app_tfp_y_r2_gogt.csv")
    for h in 0:20
        sel = th.h .== h
        @test th.i[sel] == 1:(2h + 3)
        @test f.theta[h + 1] ≈ th.theta[sel] rtol=1e-10
        m = 2h + 3
        gs = g[g.h .== h, :]
        @test nrow(gs) == m^2
        @test Int.(gs.i) == repeat(1:m, m) && Int.(gs.j) == repeat(1:m; inner = m)
        @test f.avar_pw[h + 1] ≈ reshape(gs.gogt_pw, m, m) rtol=1e-10
        @test f.avar[h + 1] ≈ reshape(gs.gogt_nopw, m, m) rtol=1e-10
    end
    fe = Matrix(O.fixture("app_tfp_y_r2_fe.csv"))
    @test O.nanapprox(f.forecast_errors, fe)
end

@testitem "fevd octave: application LP-A / LP-B (VAR set and LP set)" tags = [
    :fevd, :verification, :octave] setup = [GLOctave] begin
    using LocalProjections
    using DataFrames, LinearAlgebra, Statistics, StatsModels, Test
    O = GLOctave

    data = Dict(:sec5 => O.sec5(), :rz => O.rz())
    for name in O.APP_CASES, est in (:lpa, :lpb), set in (:varset, :lpset)
        c = O.app_case(name)
        df = data[c.data]
        ref = O.fixture("app_$(name)_$(est)_$(set).csv")
        # varset: SubLP*(y, X, 0, lagX, XtInd, 20, ylevel, 0, 0, 1, 0), the LP set
        # is the forecast-error set; lpset: the parsimonious XLP set
        f = O.quiet() do
            fevd(c.fm, df; horizon = 20, estimator = est,
                irf_formula = set === :lpset ? c.fmlp : nothing)
        end
        @test f.shock == c.shock
        @test f.window == Int.(hcat(ref.t0, ref.tend))
        @test f.t0lp == Int(only(unique(ref.t0lp)))
        @test f.shock_mean ≈ only(unique(ref.meanx)) rtol=1e-10 atol=1e-14
        @test f.shock_var ≈ only(unique(ref.varx)) rtol=1e-12
        @test f.mse ≈ ref.mse rtol=1e-10
        est === :lpb && @test f.sigma2v ≈ ref.sigma2v rtol=1e-10

        # MP cases, h = 0 (README caveat, spec Q33): with Δy_t and π_t among the
        # date-t regressors the h = 0 impulse-response regression fits exactly
        # (all mp_y files; mp_pi with the forecast-error set, which also holds
        # π_{t-1}). Both implementations then return rounding noise for ψ̂_0, its
        # s.e. and ŝ_0 (~1e-15, ~1e-15, ~1e-30), compared with an absolute
        # tolerance; the package flags the exact fit and sets the delta-method
        # s.e. to NaN there.
        degenerate = name === :mp_y || (name === :mp_pi && set === :varset)
        hs = degenerate ? (2:21) : (1:21)
        @test f.irf[hs] ≈ ref.ir[hs] rtol=1e-10
        @test f.irf_se[hs] ≈ ref.irse[hs] rtol=1e-10
        @test f.share[hs] ≈ ref.vd[hs] rtol=1e-10
        if degenerate
            @test abs(ref.ir[1]) < 1e-12 && abs(f.irf[1]) < 1e-12
            @test abs(ref.irse[1]) < 1e-12 && abs(f.irf_se[1]) < 1e-12
            @test ref.vd[1] < 1e-24 && f.share[1] < 1e-24
            @test isnan(f.se_pw[1]) && isnan(f.se[1])
        end
        # not degenerate: the LP set of mp_pi has no π_{t-1}
        if name === :mp_pi && set === :lpset
            @test ref.ir[1] ≈ 0.018 atol=1e-3
            @test isfinite(f.se_pw[1])
        end
    end
    # LP-A is not bounded by 1 (README caveat)
    f = O.quiet(() -> fevd(O.app_case(:mp_pi).fm, data[:sec5]; horizon = 20,
        estimator = :lpa))
    @test f.share[end] ≈ 1.3521 atol=5e-5
end

@testitem "fevd octave: MP exact fit at h = 0 and the QR-prewhitened variance at h ≥ 1" tags = [
    :fevd, :verification, :octave] setup = [GLOctave] begin
    # In the MP specification the h = 0 impulse-response regression fits exactly,
    # so the first N_q moment columns q_t û_{t,0} are rounding noise at every
    # later h. The reference's normal-equation prewhitening is dominated by that
    # noise (se_pw differs from the package by up to ~2.5x; README caveat and the
    # tutorial's decision table); the package's QR prewhitening is not: its se_pw
    # equals the one computed with the degenerate block removed from Z, G and the
    # gradient. `se` (no prewhitening) is insensitive either way.
    using LocalProjections
    using LinearAlgebra, Statistics, Test
    O = GLOctave
    LPj = LocalProjections

    c = O.app_case(:mp_y)
    H = 20
    f = O.quiet(() -> fevd(c.fm, O.sec5(); horizon = H, estimator = :lpb,
        irf_formula = c.fmlp))
    d = f.design
    r = LPj._gl_lp(d.dhy, d.C, d.W, d.q, d.x, f.shock_var, f.t0, f.t0lp, H;
        estimator = :lpb, dofadjust = true)
    @test r.perfect[1] && !any(r.perfect[2:end])
    Q = r.Q
    Nq = size(Q, 2)
    T = f.T
    rq = f.t0lp:T
    Σq = (Q[rq, :]' * Q[rq, :]) ./ length(rq)
    for h in (1, 4, 8, 16, 20)
        rows = max(f.t0, f.t0lp):(T - h)
        n = length(rows)
        m = Nq * (h + 1) + 2
        Z = Matrix{Float64}(undef, n, m)
        for j in 0:h
            Z[:, (j * Nq + 1):((j + 1) * Nq)] = Q[rows, :] .* r.resid_lp[rows, j + 1]
        end
        Z[:, m - 1] = d.x[rows] .^ 2 .- f.shock_var
        Z[:, m] = r.v[rows, h + 1] .^ 2 .- r.sigma2v[h + 1]
        Z .-= mean(Z; dims = 1)
        @test maximum(abs, Z[:, 1:Nq]) < 1e-10           # the degenerate block
        keep = (Nq + 1):m
        G = -cat(kron(Matrix{Float64}(I, h, h), Σq), Matrix{Float64}(I, 2, 2);
            dims = (1, 2))
        L = LPj._gl_bandwidth(n)
        g = LPj._gl_lp_gradient(:lpb, r.psi[1:(h + 1)], f.shock_var, r.sigma2v[h + 1],
            r.mse[h + 1], r.share[h + 1], Nq)[keep]
        Vpw = LPj._gl_gogt(G, LPj._gl_prewhitened_lrv(Z[:, keep], L), n)
        V = LPj._gl_gogt(G, LPj._gl_lrv_nw(Z[:, keep], L), n)
        @test f.se_pw[h + 1] ≈ LPj._gl_delta_se(g, Vpw) rtol=1e-6
        @test f.se[h + 1] ≈ LPj._gl_delta_se(g, V) rtol=1e-8
    end
end

@testitem "fevd octave: application VAR(4) FEVD, HQIC lag and bootstrap pseudo-truth" tags = [
    :fevd, :verification, :octave] setup = [GLOctave] begin
    using LocalProjections
    using DataFrames, LinearAlgebra, Random, Statistics, StatsModels, Test
    O = GLOctave
    LPj = LocalProjections

    data = Dict(:sec5 => O.sec5(), :rz => O.rz())
    ic = O.fixture("app_ic.csv")
    for name in O.APP_CASES
        c = O.app_case(name)
        df = data[c.data]
        T = nrow(df)
        icc = ic[Int.(ic.case) .== c.case, :]
        sv = O.fixture("app_$(name)_svar.csv")
        lag = Int(only(icc.p[icc.selected .== 1]))
        @test lag == Int(only(unique(sv.lagvar))) == (c.data === :sec5 ? 2 : 3)
        @test Int(only(unique(icc.xvar_rows))) == T - c.cumul
        @test Int(only(unique(icc.T_IC))) == T - c.cumul - 10

        # private core on the reference's XVAR (diff(y) of the sequential cumsum)
        inp = O.fixture("app_$(name)_inputs.csv")
        X = Matrix(inp[:, 2:end])
        order, place = O.XVAR_LAYOUT[name]
        XV = X[:, order]
        XVAR = c.cumul ? hcat(XV[2:end, 1:(place - 1)], diff(inp.y), XV[2:end, place:end]) :
               hcat(XV[:, 1:(place - 1)], inp.y, XV[:, place:end])
        sel = LPj._gl_lagselect(XVAR, collect(1:10); criterion = :hqic)
        @test sel.values ≈ icc.hqic rtol=1e-10
        @test sel.p == lag
        v = LPj._var_ols(XVAR, lag)
        Ω = (v.U' * v.U) ./ v.Tu
        @test LPj._var_fevd(v.A, Ω, 20; shock = Int(only(unique(icc.x_location))),
            response = Int(only(unique(icc.y_location))), cumulate = c.cumul) ≈
              sv.svar rtol=1e-10

        # public route: fevdbootstrap(method = :var) with the default VAR ordering,
        # HQIC over 1:10 and, for the cumul(dy) responses, G-L's rows 2:T
        f = fevd(c.fm, df; horizon = 20, estimator = :r2)
        b = fevdbootstrap(f, df; method = :var, nlags = 1:10, criterion = :hqic,
            skipfirst = c.cumul, nboot = 2, rng = Xoshiro(1))
        @test b.vars == (c.data === :sec5 ? [:tfp, :dy, :infl, :rr, :effr] :
                [:newsy, :dy, :infl, :tbill])
        @test findfirst(==(c.shock), b.vars) == Int(only(unique(icc.x_location)))
        @test findfirst(==(c.cumul ? :dy : :infl), b.vars) ==
              Int(only(unique(icc.y_location)))
        @test b.ic.candidates == 1:10
        @test b.ic.values ≈ icc.hqic rtol=1e-10
        @test b.nlags == lag
        @test b.pseudo_truth ≈ sv.svar rtol=1e-10
    end

    # var_chol(timeseries, 4, 20, ., s, 0, 2, 0, 0): a VAR(4) on all rows, the
    # MLE covariance, Δy (variable 2) cumulated; one-s.d. responses and FEVDs
    for (dataset, file, vars, shocks) in (
        (:sec5, "app_sec5_var4_chol.csv", [:tfp, :dy, :infl, :rr, :effr], (1, 4)),
        (:rz, "app_rz_var4_chol.csv", [:newsy, :dy, :infl, :tbill], (1,)))
        ref = O.fixture(file)
        Y = Matrix{Float64}(data[dataset][:, vars])
        n = size(Y, 2)
        v = LPj._var_ols(Y, 4)
        Ω = (v.U' * v.U) ./ v.Tu
        Θ = LPj._var_ma(v.A, 20)
        L = cholesky(Symmetric(Ω)).L
        for s in shocks, r in 1:n
            ψ = [dot(Θ[r, :, h + 1], L[:, s]) for h in 0:20]
            r == 2 && (ψ = cumsum(ψ))
            @test ψ ≈ ref[!, Symbol("psi_s$(s)_v$(r)")] rtol=1e-10
            @test LPj._var_fevd(v.A, Ω, 20; shock = s, response = r, cumulate = r == 2) ≈
                  ref[!, Symbol("vd_s$(s)_v$(r)")] rtol=1e-10
        end
    end
end

@testitem "fevd octave: application bias corrections up to Monte Carlo error (TFP → y)" tags = [
    :fevd, :verification, :octave] setup = [GLOctave] begin
    # The Octave files hold one run of each procedure for each of two seeds (B =
    # 2000, the paper's settings); Julia's draws come from a different generator,
    # so agreement is up to Monte Carlo error. With σ_h the standard deviation of
    # the draws at horizon h (`se_boot`), κ_h their kurtosis and B = 2000, one
    # run's Monte Carlo standard deviations are
    #   mean of the draws m̄_h (and hence vdBc = ŝ - m̄ + s_VAR or 2ŝ - m̄):
    #       σ_h / √B;
    #   se_boot (relative):  √((κ_h - 1) / (4B));
    #   band endpoint Q_q(s*) + vdBc - m̄ = Q_q(s*) - 2m̄ + const:  at most
    #       sd(Q_q) + 2σ_h/√B, with sd(Q_q) = √(q(1-q)/B) / f(Q_q) and the
    #       sparsity 1/f estimated from the draws (GLOctave.quantile_sd).
    # Julia is compared with the average of the two Octave runs (a difference
    # of one run and the mean of two independent runs: factor √(1 + 1/2)), the
    # two Octave runs with each other (factor √2), and every horizon must lie
    # within NSIG = 4 such standard deviations. Over 21 horizons the chance of a
    # 4σ exceedance is below 21 · 6.3e-5 ≈ 0.13% per procedure and quantity
    # (horizons are positively correlated, which lowers it further), so the
    # bounds are not tuned to the seed: over 20 other seeds the largest
    # deviation was 3.7σ (block bootstrap vdBc), and the Octave pair passes the
    # same bounds (largest 3.1σ, simulation R² vdBc). Run time of the five
    # procedures: ~5 s.
    using LocalProjections
    using DataFrames, LinearAlgebra, Statistics, StableRNGs, StatsModels, Test
    O = GLOctave
    LPj = LocalProjections

    df = O.sec5()
    c = O.app_case(:tfp_y)
    B, H = 2000, 20
    f_r2 = fevd(c.fm, df; horizon = H, estimator = :r2)
    # BootstrapVARLPB: the LP set is the forecast-error set (fevd warns that the
    # prewhitened AVar is infeasible at long horizons, irrelevant here)
    f_var_lpb = O.quiet(() -> fevd(c.fm, df; horizon = H, estimator = :lpb))
    # SimulationAVarLPB: the parsimonious LP set
    f_sim_lpb = fevd(c.fm, df; horizon = H, estimator = :lpb, irf_formula = c.fmlp)
    var_kw = (; method = :var, nlags = 1:10, criterion = :hqic, skipfirst = true,
        burnin = 100, nboot = B)
    runs = [
        ("var_r2", f_r2, () -> fevdbootstrap(f_r2, df; var_kw..., rng = StableRNG(101))),
        ("var_lpb", f_var_lpb,
            () -> fevdbootstrap(f_var_lpb, df; var_kw..., rng = StableRNG(102))),
        ("block_r2", f_r2,
            () -> fevdbootstrap(f_r2; method = :block, blocklength = 4, nboot = B,
                rng = StableRNG(103))),
        ("sim_r2", f_r2,
            () -> fevdbootstrap(f_r2; method = :simulation, prewhiten = true, bound = true,
                nboot = B, rng = StableRNG(104))),
        ("sim_lpb", f_sim_lpb,
            () -> fevdbootstrap(f_sim_lpb; method = :simulation, prewhiten = true,
                nboot = B, rng = StableRNG(105)))]

    # delta-method s.e. with the gradient at a given share (the simulation
    # method's convention: at vdBc)
    function se_at(f, s, V)
        map(0:(f.horizon)) do h
            θ = f.theta[h + 1]
            k = h + 1
            g = if f.estimator === :r2
                vcat(θ[(k + 1):(2k)], θ[1:k], -s[h + 1]) ./ θ[end]
            else
                LPj._gl_lp_gradient(f.estimator, f.irf[1:k], f.shock_var, θ[end],
                    f.mse[h + 1], s[h + 1], (length(θ) - 2) ÷ k)
            end
            sqrt(dot(g, V[h + 1] * g))
        end
    end
    # every horizon within NSIG Monte Carlo standard deviations
    within(d, sd) = all(abs.(d) .<= O.NSIG .* sd)

    for (proc, f, run) in runs
        b = run()
        o1 = O.fixture("app_stoch_$(proc)_tfp_y_seed1.csv")
        o2 = O.fixture("app_stoch_$(proc)_tfp_y_seed2.csv")
        avg(col) = (o1[!, col] .+ o2[!, col]) ./ 2

        # deterministic columns
        @test b.nfail == 0
        @test size(b.draws) == (B, H + 1)
        @test b.share ≈ o1.vd rtol=1e-10
        @test o1.vd ≈ o2.vd rtol=1e-14

        # Monte Carlo columns
        σ = b.se_boot
        κ = [O.kurtosis(d) for d in eachcol(b.draws)]
        sd_mean = σ ./ sqrt(B)
        @test within(b.biascorrected .- avg(:vdbc), sqrt(1.5) .* sd_mean)
        @test within(b.mean_draws .- avg(:mean_draws), sqrt(1.5) .* sd_mean)
        @test within(o1.vdbc .- o2.vdbc, sqrt(2) .* sd_mean)
        sd_rel = sqrt.((κ .- 1) ./ (4B))
        @test within(b.se_boot ./ avg(:se_boot) .- 1, sqrt(1.5) .* sd_rel)
        @test within(o1.se_boot ./ o2.se_boot .- 1, sqrt(2) .* sd_rel)
        # band = :shift, bias-corrected: the reference's cbIndicator = 0
        s = summarize(b; level = 0.90)
        for (q, mine, col) in ((0.05, s.lower, :cb05), (0.95, s.upper, :cb95))
            sd_band = [O.quantile_sd(d, q) for d in eachcol(b.draws)] .+ 2 .* sd_mean
            @test within(mine .- avg(col), sqrt(1.5) .* sd_band)
            @test within(o1[!, col] .- o2[!, col], sqrt(2) .* sd_band)
        end

        if startswith(proc, "var")
            @test b.vars == [:tfp, :dy, :infl, :rr, :effr]
            @test b.nlags == Int(only(unique(o1.lagvar))) == 2
            @test b.pseudo_truth ≈ o1.svar rtol=1e-10
        end
        if proc in ("var_r2", "block_r2")
            # SubR2's s.e. at ŝ
            @test b.se_pw ≈ o1.se_avar_pw rtol=1e-10
            @test b.se ≈ o1.se_avar_nopw rtol=1e-10
        end
        if startswith(proc, "sim")
            # s.e. with the gradient at vdBc: exact given vdBc (a deterministic
            # check of the convention), then Julia's at its own vdBc
            for o in (o1, o2)
                @test se_at(f, o.vdbc, f.avar_pw) ≈ o.se_avar_pw_at_vdbc rtol=1e-10
                @test se_at(f, o.vdbc, f.avar) ≈ o.se_avar_nopw_at_vdbc rtol=1e-10
            end
            @test b.se_pw ≈ se_at(f, b.biascorrected, f.avar_pw) rtol=1e-12
            @test b.se ≈ se_at(f, b.biascorrected, f.avar) rtol=1e-12
        end
        if proc in ("var_lpb", "sim_lpb")
            @test f.irf ≈ o1.ir rtol=1e-10
            @test f.irf_se ≈ o1.irse_avar rtol=1e-10
            @test sqrt(f.shock_var) ≈ only(unique(o1.stdx)) rtol=1e-12
        end
        if proc == "var_lpb"
            irstr = f.irf .* sqrt(f.shock_var)
            @test irstr ≈ o1.irstr rtol=1e-10
            # irBc = 2ψ̂ - mean ψ_b and the bootstrap s.e. of ψ
            sdψ = b.irf_se_boot ./ sqrt(B)
            @test within(b.irf_bc .- avg(:irbc), sqrt(1.5) .* sdψ)
            @test within(o1.irbc .- o2.irbc, sqrt(2) .* sdψ)
            κψ = [O.kurtosis(d) for d in eachcol(b.irf_draws)]
            @test within(b.irf_se_boot ./ avg(:irse_boot) .- 1,
                sqrt(1.5) .* sqrt.((κψ .- 1) ./ (4B)))
            # irStrCb = ψ̂σ̂_x + Q_q(irStr_b - mean irStr_b): sd ≤ sd(Q_q) + sd(mean)
            D = b.irfstr_draws
            for (q, col) in ((0.05, :irstr_cb05), (0.95, :irstr_cb95))
                band = irstr .+ [O.quantile5(d .- mean(d), q) for d in eachcol(D)]
                sd_band = [O.quantile_sd(d, q) for d in eachcol(D)] .+
                          vec(std(D; dims = 1)) ./ sqrt(B)
                @test within(band .- avg(col), sqrt(1.5) .* sd_band)
                @test within(o1[!, col] .- o2[!, col], sqrt(2) .* sd_band)
            end
        end
    end
end
