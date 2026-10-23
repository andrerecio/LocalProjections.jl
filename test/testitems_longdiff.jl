# Horizon-tracking right-hand-side terms and the long-difference responses
# ldiff/hbr/firstdiff (ported from the andrerecio fork onto the 0.2 API).

@testitem "horizon-tracking cumul() on the lpiv right-hand side" tags = [
    :lpiv, :cumul, :verification,
] begin
    using LocalProjections
    using DataFrames, LinearAlgebra, StableRNGs, StatsBase, Test

    rng = StableRNG(20260901)
    T = 240
    z = zeros(T)
    g = zeros(T)
    y = zeros(T)
    for t in 3:T
        z[t] = 0.4 * z[t - 1] + randn(rng)
        g[t] = 0.6 * g[t - 1] + 0.8 * z[t] + 0.3 * randn(rng)
        y[t] = 0.5 * y[t - 1] + 0.7 * g[t] + randn(rng)
    end
    df = DataFrame(y = y, g = g, z = z)

    H = 6
    res = lpiv(
        TSLS(), @formula(cumul(y) ~ (cumul(g) ~ z) + lags(y, 2) + lags(g, 2)), df;
        horizon = H
    )

    # Coefficient names must stay horizon-invariant, otherwise coefpath /
    # summarize / the plots cannot follow the path.
    @test coefnames(res) == ["(Intercept)", "y_lag1", "y_lag2", "g_lag1", "g_lag2", "cumul(g)"]
    @test res.shock == Symbol("cumul(g)")
    @test res.horizon == H

    path = coefpath(res; term = res.shock)
    @test length(path) == H + 1
    @test all(isfinite, path)

    # Reference: build the cumulative columns by hand and run one just-identified
    # 2SLS per horizon, the way jordagk.do does.
    function manual(h)
        cum(v) = Union{Missing, Float64}[t + h <= T ? sum(v[t:(t + h)]) : missing for t in 1:T]
        L(v, k) = Union{Missing, Float64}[i - k >= 1 ? v[i - k] : missing for i in 1:T]
        cy, cg = cum(y), cum(g)
        ctrl = [L(y, 1), L(y, 2), L(g, 1), L(g, 2)]
        cols = vcat([cy, cg, z], ctrl)
        idx = findall(t -> all(c -> !ismissing(c[t]), cols), 1:T)
        W = hcat(ones(length(idx)), [Float64.(c[idx]) for c in ctrl]...)
        X = hcat(W, Float64.(cg[idx]))
        Z = hcat(W, Float64.(z[idx]))
        b = (Z'X) \ (Z'Float64.(cy[idx]))
        return (nobs = length(idx), coef = b[end])
    end

    for h in 0:H
        m = manual(h)
        @test path[h + 1] ≈ m.coef rtol = 1.0e-10
        @test nobs(res)[h + 1] == m.nobs
    end

    # The usable sample shrinks by one observation per horizon: the cumulative
    # response and regressor both need y_{t+h}, g_{t+h}.
    @test nobs(res) == [nobs(res)[1] - h for h in 0:H]

    # sample = :fixed keeps the rows complete at every horizon while the
    # cumulative regressor is still rebuilt per horizon
    fixed = lpiv(
        TSLS(), @formula(cumul(y) ~ (cumul(g) ~ z) + lags(y, 2) + lags(g, 2)), df;
        horizon = H, sample = :fixed
    )
    @test all(==(nobs(res)[end]), nobs(fixed))
    @test coefpath(fixed)[end] ≈ path[end] rtol = 1.0e-12
end

@testitem "horizon-tracking cumul()/leads() on the lp right-hand side" tags = [
    :cumul, :leads, :verification,
] begin
    using LocalProjections
    using DataFrames, StableRNGs, StatsBase, Test

    rng = StableRNG(4242)
    T = 200
    x = randn(rng, T)
    y = zeros(T)
    for t in 2:T
        y[t] = 0.4 * y[t - 1] + 0.9 * x[t] + randn(rng)
    end
    df = DataFrame(y = y, x = x)

    H = 4
    res = lp(@formula(cumul(y) ~ cumul(x) + lags(y, 1)), df; horizon = H)
    @test coefnames(res) == ["(Intercept)", "cumul(x)", "y_lag1"]

    path = coefpath(res; term = Symbol("cumul(x)"))
    for h in 0:H
        cum(v) = Union{Missing, Float64}[t + h <= T ? sum(v[t:(t + h)]) : missing for t in 1:T]
        cy, cx = cum(y), cum(x)
        ly = Union{Missing, Float64}[i > 1 ? y[i - 1] : missing for i in 1:T]
        idx = findall(t -> !ismissing(cy[t]) && !ismissing(cx[t]) && !ismissing(ly[t]), 1:T)
        X = hcat(ones(length(idx)), Float64.(cx[idx]), Float64.(ly[idx]))
        b = X \ Float64.(cy[idx])
        @test path[h + 1] ≈ b[2] rtol = 1.0e-10
    end

    # leads() on the RHS tracks the horizon in the same way.
    resl = lp(@formula(leads(y) ~ leads(x) + lags(y, 1)), df; horizon = 2)
    @test coefnames(resl) == ["(Intercept)", "leads(x)", "y_lag1"]
    pl = coefpath(resl; term = Symbol("leads(x)"))
    # y_{t+h} on x_{t+h} reproduces the static contemporaneous regression at every h.
    static = lp(@formula(leads(y) ~ x + lags(y, 1)), df; horizon = 0)
    @test pl[1] ≈ coefpath(static; term = :x)[1] rtol = 1.0e-10
    @test all(isfinite, pl)
end

@testitem "explicit horizons pin RHS terms and leave the static path intact" tags = [
    :cumul, :api,
] begin
    using LocalProjections
    using DataFrames, StableRNGs, StatsBase, Test

    rng = StableRNG(77)
    T = 160
    x = randn(rng, T)
    y = cumsum(randn(rng, T)) .+ 0.5 .* x
    df = DataFrame(y = y, x = x)

    # cumul(x, 2) carries its own horizon, so the column is the SAME at every h
    # and the coefficient name records the pinned horizon.
    pinned = lp(@formula(cumul(y) ~ cumul(x, 2)), df; horizon = 3)
    @test coefnames(pinned) == ["(Intercept)", "cumul(x, 2)"]

    cx2 = Union{Missing, Float64}[t + 2 <= T ? sum(x[t:(t + 2)]) : missing for t in 1:T]
    path = coefpath(pinned; term = Symbol("cumul(x, 2)"))
    for h in 0:3
        cy = Union{Missing, Float64}[t + h <= T ? sum(y[t:(t + h)]) : missing for t in 1:T]
        idx = findall(t -> !ismissing(cy[t]) && !ismissing(cx2[t]), 1:T)
        X = hcat(ones(length(idx)), Float64.(cx2[idx]))
        @test path[h + 1] ≈ (X \ Float64.(cy[idx]))[2] rtol = 1.0e-10
    end

    # A formula with no horizon-tracking RHS term is unaffected.
    plain = lp(@formula(leads(y) ~ x), df; horizon = 3)
    @test coefnames(plain) == ["(Intercept)", "x"]
    @test length(coefpath(plain; term = :x)) == 4
end

@testitem "horizon-tracking terms in panels are built within each unit" tags = [
    :cumul, :ldiff, :panel,
] begin
    using LocalProjections
    using DataFrames, StableRNGs, StatsBase, Test

    rng = StableRNG(5)
    T, N = 40, 3
    df = DataFrame(id = repeat(1:N, inner = T), t = repeat(1:T, N))
    df.x = randn(rng, T * N)
    df.y = cumsum(randn(rng, T * N)) .+ df.x

    H = 3
    r = lp(@formula(ldiff(y) ~ cumul(x)), df; horizon = H, unit = :id, time = :t)
    for h in 0:H
        Y, X = Float64[], Float64[]
        for u in 1:N
            y, x = df.y[df.id .== u], df.x[df.id .== u]
            for t in 2:(T - h)
                push!(Y, y[t + h] - y[t - 1])
                push!(X, sum(x[t:(t + h)]))
            end
        end
        @test nobs(r)[h + 1] == length(Y)
        @test coefpath(r; term = Symbol("cumul(x)"))[h + 1] ≈ (hcat(ones(length(X)), X) \ Y)[2] rtol = 1.0e-10
    end
end

@testitem "ldiff() and firstdiff() match hand-built long differences" tags = [
    :ldiff, :verification,
] begin
    using LocalProjections
    using DataFrames, StableRNGs, StatsBase, Test

    rng = StableRNG(2025)
    T = 220
    x = randn(rng, T)
    y = cumsum(0.5 .* x .+ randn(rng, T))
    w = cumsum(randn(rng, T))
    df = DataFrame(y = y, x = x, w = w)

    H = 5
    r = lp(@formula(ldiff(y) ~ x + lags(firstdiff(y), 2) + lags(firstdiff(w), 2)), df; horizon = H)
    @test coefnames(r) == [
        "(Intercept)", "x", "firstdiff(y)_lag1", "firstdiff(y)_lag2",
        "firstdiff(w)_lag1", "firstdiff(w)_lag2",
    ]
    @test r.response === :y
    @test !r.tautological_h0

    d(v) = [NaN; diff(v)]
    L(v, j) = [t > j ? v[t - j] : NaN for t in eachindex(v)]
    dy, dw = d(y), d(w)
    X = hcat(ones(T), x, L(dy, 1), L(dy, 2), L(dw, 1), L(dw, 2))
    for h in 0:H
        Y = [2 <= t && t + h <= T ? y[t + h] - y[t - 1] : NaN for t in 1:T]
        ok = .!isnan.(Y) .& vec(all(!isnan, X; dims = 2))
        @test coef(horizonmodel(r, h)) ≈ X[ok, :] \ Y[ok] rtol = 1.0e-10
        @test nobs(r)[h + 1] == count(ok)
    end

    # cumul(ldiff(y)) sums the long differences over horizons 0, ..., h
    c = lp(@formula(cumul(ldiff(y)) ~ x + lags(firstdiff(y), 2)), df; horizon = 3)
    Xc = X[:, 1:4]
    for h in 0:3
        Y = [2 <= t && t + h <= T ? sum(y[t + j] - y[t - 1] for j in 0:h) : NaN for t in 1:T]
        ok = .!isnan.(Y) .& vec(all(!isnan, Xc; dims = 2))
        @test coef(horizonmodel(c, h)) ≈ Xc[ok, :] \ Y[ok] rtol = 1.0e-10
    end

    # anchor(y, lag(y)) is the same long difference
    a = lp(
        @formula(anchor(y, lag(y)) ~ x + lags(firstdiff(y), 2) + lags(firstdiff(w), 2)), df;
        horizon = H
    )
    @test coefpath(a) ≈ coefpath(r) rtol = 1.0e-12

    # A missing value excludes only the horizons that need it
    ym = Union{Missing, Float64}[y;]
    ym[100] = missing
    rm = lp(@formula(ldiff(y) ~ x), DataFrame(y = ym, x = x); horizon = 2)
    for h in 0:2
        Y = [
            2 <= t && t + h <= T && t + h != 100 && t - 1 != 100 ? y[t + h] - y[t - 1] : NaN
                for t in 1:T
        ]
        ok = .!isnan.(Y)
        @test coef(horizonmodel(rm, h)) ≈ hcat(ones(count(ok)), x[ok]) \ Y[ok] rtol = 1.0e-10
    end
end

@testitem "hbr() and cumul(hbr()) match hand-built HBR responses in lp and lpiv" tags = [
    :hbr, :lpiv, :verification,
] begin
    using LocalProjections
    using DataFrames, StableRNGs, StatsBase, Test

    rng = StableRNG(1889)
    T = 240
    z = randn(rng, T)
    Y = 100 .+ cumsum(1 .+ 0.3 .* z .+ randn(rng, T))
    G = 20 .+ cumsum(0.2 .+ 0.5 .* z .+ 0.5 .* randn(rng, T))
    df = DataFrame(Y = Y, G = G, z = z)

    L(v, j) = [t > j ? v[t - j] : NaN for t in eachindex(v)]
    X = hcat(ones(T), L(z, 1), L(log.(Y), 1), L(log.(G), 1))
    hb(x, t, h) = 2 <= t && t + h <= T ? (x[t + h] - x[t - 1]) / Y[t - 1] : NaN
    chb(x, t, h) = 2 <= t && t + h <= T ? sum(x[t + j] - x[t - 1] for j in 0:h) / Y[t - 1] : NaN

    H = 4
    ry = lp(@formula(hbr(Y) ~ z + lags(z, 1) + lags(log(Y), 1) + lags(log(G), 1)), df; horizon = H)
    rg = lp(@formula(hbr(G, Y) ~ z + lags(z, 1) + lags(log(Y), 1) + lags(log(G), 1)), df; horizon = H)
    @test ry.response === :Y && rg.response === :G
    Xz = hcat(X[:, 1], z, X[:, 2:end])
    for h in 0:H, (r, x) in ((ry, Y), (rg, G))
        yh = [hb(x, t, h) for t in 1:T]
        ok = .!isnan.(yh) .& vec(all(!isnan, Xz; dims = 2))
        @test coef(horizonmodel(r, h)) ≈ Xz[ok, :] \ yh[ok] rtol = 1.0e-8
    end

    # One-step cumulative multiplier in HBR units, by 2SLS
    m = lpiv(
        TSLS(),
        @formula(
            cumul(hbr(Y)) ~ (cumul(hbr(G, Y)) ~ z) + lags(z, 1) + lags(log(Y), 1) +
                lags(log(G), 1)
        ),
        df; horizon = H
    )
    @test m.shock === Symbol("cumul(hbr(G, Y))")
    @test m.endogenous_names == ["cumul(hbr(G, Y))"]
    for h in 0:H
        cy = [chb(Y, t, h) for t in 1:T]
        cg = [chb(G, t, h) for t in 1:T]
        Xf, Zf = hcat(X, cg), hcat(X, z)
        ok = .!isnan.(cy) .& vec(all(!isnan, Xf; dims = 2))
        Xo, Zo, yo = Xf[ok, :], Zf[ok, :], cy[ok]
        P = Zo * ((Zo' * Zo) \ Zo')
        @test coef(horizonmodel(m, h)) ≈ (Xo' * P * Xo) \ (Xo' * P * yo) rtol = 1.0e-8
    end

    # at h = 0 with one instrument the multiplier is exactly the
    # indirect-least-squares ratio of the two h = 0 reduced forms
    @test coefpath(m)[1] ≈ coefpath(ry)[1] / coefpath(rg)[1] rtol = 1.0e-8
end

@testitem "long-difference terms: pinned horizons, parsing and guards" tags = [
    :ldiff, :hbr, :api,
] begin
    using LocalProjections
    using DataFrames, StableRNGs, StatsBase, StatsModels, Test

    rng = StableRNG(36)
    T = 150
    x = randn(rng, T)
    s = 10 .+ cumsum(abs.(randn(rng, T)))
    y = cumsum(randn(rng, T)) .+ x
    g = 0.3 .* s .+ randn(rng, T)
    df = DataFrame(y = y, x = x, s = s, g = g)

    # Explicit horizons pin RHS terms, which then do not track the projection
    p = lp(@formula(leads(y) ~ x + hbr(g, s, 2) + cumul(ldiff(g), 1) + ldiff(g, 0)), df; horizon = 3)
    @test coefnames(p) == ["(Intercept)", "x", "hbr(g, s, 2)", "cumul(ldiff(g), 1)", "ldiff(g, 0)"]
    @test !LocalProjections._rhs_tracks_horizon(p.base_formula)
    # hbr(x, 3) is hbr(x, x, 3); ldiff(x, 0) is the first difference
    q = lp(@formula(leads(y) ~ x + hbr(s, 3) + ldiff(g, 0) + firstdiff(g)), df; horizon = 0)
    @test coefnames(q)[3] == "hbr(s, 3)"
    X = modelmatrix(horizonmodel(q, 0))
    @test X[:, 4] == X[:, 5]

    # A bare ldiff/hbr on the RHS tracks the horizon, in lp and lpiv
    tr = lpiv(TSLS(), @formula(leads(y) ~ (cumul(hbr(g, s)) ~ x)), df; horizon = 2)
    @test LocalProjections._rhs_tracks_horizon(tr.base_formula)
    @test LocalProjections._rhs_tracks_horizon(lp(@formula(leads(y) ~ ldiff(g)), df; horizon = 1).base_formula)

    # Horizons written on the left-hand side are overridden by the projection
    @test coefpath(lp(@formula(ldiff(y, 7) ~ x), df; horizon = 2)) ≈
        coefpath(lp(@formula(ldiff(y) ~ x), df; horizon = 2))

    # A long difference on y_t is not an identity at h = 0, even with y as the shock
    @test !lp(@formula(ldiff(y) ~ y + x), df; horizon = 1).tautological_h0

    # Errors
    @test_throws "long differences use ldiff(y) or hbr(x, y)" lp(@formula(y ~ x), df; horizon = 1)
    @test_throws ArgumentError lp(@formula(cumul(ldiff(y, 3)) ~ x), df; horizon = 1)
    @test_throws ArgumentError lp(@formula(hbr(g, s, x) ~ x), df; horizon = 1)
    @test_throws ArgumentError lp(@formula(firstdiff(y, 2) ~ x), df; horizon = 1)
    df0 = copy(df)
    df0.s[10] = 0.0
    @test_throws "scale variable is zero" lp(@formula(hbr(g, s) ~ x), df0; horizon = 1)

    # Formulas built programmatically work too
    f = ldiff(term(:y)) ~ term(1) + term(:x) + lags(firstdiff(term(:y)), 2)
    @test coefpath(lp(f, df; horizon = 2)) ≈
        coefpath(lp(@formula(ldiff(y) ~ x + lags(firstdiff(y), 2)), df; horizon = 2))
end

@testitem "the RHS horizon-tracking detector" tags = [:cumul, :api] begin
    using LocalProjections
    using StatsModels, Test

    tracks = LocalProjections._rhs_tracks_horizon
    @test tracks(@formula(leads(y) ~ x + cumul(x)))
    @test tracks(@formula(leads(y) ~ leads(x) + lags(y, 2)))
    @test tracks(@formula(cumul(y) ~ (cumul(x) ~ z) + lags(y, 2)))   # IV block
    @test tracks(@formula(leads(y) ~ x & cumul(x)))                 # interaction
    @test tracks(@formula(leads(y) ~ x + hbr(g, s)))
    @test !tracks(@formula(cumul(y) ~ x + lags(y, 2)))               # LHS only
    @test !tracks(@formula(leads(y) ~ cumul(x, 3) + leads(x, 2)))    # pinned
    @test !tracks(@formula(leads(y) ~ cumul(ldiff(x), 3)))           # pinned as a whole
end

@testitem "HBR multipliers reproduce Stata (newey, ivreg2) on the Ramey-Zubairy data" tags = [
    :hbr, :verification,
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

    # Reference: stata_test/rz_hbr_compare.do (fork), variables rebuilt from
    # rzdatnew.csv; -newey, lag(5)- and -ivreg2, robust bw(6)-, h = 8
    irf = lp(
        @formula(
            hbr(ypc) ~ newsy_hbr + lags(newsy_hbr, 4) + lags(log(ypc), 4) + lags(log(gpc), 4)
        ),
        rz; horizon = 8
    )
    @test nobs(irf)[9] == 492
    @test coefpath(irf)[9] ≈ 3.12765742949768e-1 rtol = 1.0e-9
    # OLS Bartlett carries n/(n-k), like -newey-
    @test stderror(irf + vcov(Bartlett(6)); term = :newsy_hbr)[9] ≈ 1.05398551104355e-1 rtol = 1.0e-9

    f = @formula(
        cumul(hbr(ypc)) ~ (cumul(hbr(gpc, ypc)) ~ newsy_hbr) + lags(newsy_hbr, 4) +
            lags(log(ypc), 4) + lags(log(gpc), 4)
    )
    m = lpiv(TSLS(), f, rz; horizon = 8)
    g = Symbol("cumul(hbr(gpc, ypc))")
    @test coefpath(m)[9] ≈ 7.03828970486729e-1 rtol = 1.0e-8
    # -ivreg2, robust bw(6)- has no n/(n-k); Regress 0.3 applies it on IV models
    # too, so the stored covariance is wider by exactly sqrt(n/(n-k))
    j = findfirst(==(String(g)), coefnames(m))
    m8 = horizonmodel(m, 8)
    @test sqrt(vcov(Bartlett(6), m8; dofadjust = false)[j, j]) ≈ 6.99921592211031e-2 rtol = 1.0e-7
    n, k = nobs(m8), length(coefnames(m))
    @test stderror(m + vcov(Bartlett(6)); term = g)[9] ≈ 6.99921592211031e-2 * sqrt(n / (n - k)) rtol = 1.0e-7
    @test weakivtest(m + vcov(Bartlett(6)), 8).F_robust ≈ 1.44834530781619e+1 rtol = 1.0e-8
end

@testitem "long-difference LPs reproduce Piger-Stockwell (Stata newey)" tags = [
    :ldiff, :verification,
] begin
    using LocalProjections
    using DataFrames, StatsBase, Statistics, Test
    using CovarianceMatrices: Bartlett

    path = joinpath(@__DIR__, "..", "docs", "src", "data", "piger_stockwell.csv")
    lines = readlines(path)
    hdr = Symbol.(split(lines[1], ","))
    cols = [Union{Missing, Float64}[] for _ in hdr]
    for l in lines[2:end], (j, v) in enumerate(split(l, ","; keepempty = true))
        push!(cols[j], isempty(v) ? missing : parse(Float64, v))
    end
    ps = DataFrame(cols, hdr)[1:(length(lines) - 1 - 57), :]
    ps.mp = coalesce.(ps.mp ./ std(skipmissing(ps.mp)), NaN)

    # Reference: stata_test/ps_compare.do (fork), -newey, lag(37)- on the data of
    # their application code (Figure 20), which matches the MATLAB loop to 1e-14
    r = lp(
        @formula(
            ldiff(lip) ~ mp + lags(firstdiff(lip), 12) + lags(firstdiff(lcpi), 12) +
                lags(firstdiff(lsp500), 12) + lags(firstdiff(ebp), 12) + lags(firstdiff(gs1), 12)
        ),
        ps; horizon = 36
    )
    se = stderror(r + vcov(Bartlett(38)); term = :mp)
    @test nobs(r)[13] == 347 && nobs(r)[37] == 323
    @test coefpath(r)[13] ≈ -2.54697060725518e-1 rtol = 1.0e-10
    @test se[13] ≈ 2.25082209688571e-1 rtol = 1.0e-10
    @test coefpath(r)[37] ≈ -3.25545542946992e-1 rtol = 1.0e-10
    @test se[37] ≈ 2.98261379130978e-1 rtol = 1.0e-10
end
