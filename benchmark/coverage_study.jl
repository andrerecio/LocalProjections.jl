"""
Monte Carlo coverage study for `varbootstrap`.

This is a *validation* study, not a benchmark: it checks that the VAR residual
moving-block bootstrap delivers close-to-nominal coverage of the true impulse
response, and quantifies what the two analytical corrections buy. It is far too
slow and too stochastic to live in the test suite, so it is kept here to be run
by hand.

Usage:

```
julia --project=benchmark benchmark/coverage_study.jl              # defaults
julia --project=benchmark benchmark/coverage_study.jl 100 200      # nmc nboot
```

The DGP is a bivariate VAR(1) with the shock ordered first and `Σ = I`, so the
unit-impact vector is `e₁` and the true response of `y` to the shock at horizon
`h` is `(A₁^h)[2,1]` — known in closed form, which is what makes coverage
measurable.

Headline findings from `main(300, 500)` at `T = 240`, 90% nominal (2026-08-31;
reproducible, the study is seeded with `StableRNG`):

  - The bootstrap's advantage is *stability across horizons*. Under a
    near-unit-root DGP (ρ_y = 0.99) the delta-method HC1 band decays from 0.917
    at h = 0 to 0.830 at h = 8, while Hall percentile-t holds 0.86–0.90
    throughout — it widens where the normal approximation does not (h = 8 mean
    width 0.901 vs 0.820). The same pattern appears at ρ_y = 0.95.
  - The Herbst–Johannsen correction removes most of the point-estimate bias at
    long horizons (h = 8: −0.077 → −0.014).
  - It does *not* materially change coverage (h = 8: 0.853 corrected vs 0.843
    uncorrected): the corrections shift the center and the bands shift with it.
    They buy a better point estimate, not better coverage.
  - The Pope stability safeguard is not cosmetic — at ρ_y = 0.99 the full
    correction applied in only ~73% of samples (mean δ ≈ 0.89). Check
    `b.pope_delta` on persistent data.
  - Both bootstrap variants sit slightly below nominal (≈0.86) at T = 240,
    consistent with the undercoverage Montiel Olea, Plagborg-Møller, Qian &
    Wolf (2025) report for LP bootstraps.
  - The point estimate falls outside its own Hall-t interval in at most 0.3% of
    horizon-replications. Rare, but non-zero — which is why the test suite
    asserts band ordering, not containment of θ̂.
"""

using LocalProjections
using DataFrames
using StatsModels    # @formula
using StableRNGs
using CovarianceMatrices
using LinearAlgebra
using Printf
using Random
using Statistics

# ============================================================================
# DGP
# ============================================================================

"""
    simulate_var1(A1, T, rng; burn=300) -> Matrix{Float64}

Simulate `T` observations of `Yₜ = A₁Yₜ₋₁ + εₜ` with `ε ~ N(0, I)`, discarding
`burn` initial observations. Column 1 is the shock.
"""
function simulate_var1(A1::AbstractMatrix, T::Int, rng::AbstractRNG; burn::Int = 300)
    n = size(A1, 1)
    Y = zeros(T + burn, n)
    for t in 2:(T + burn)
        Y[t, :] = A1 * Y[t - 1, :] + randn(rng, n)
    end
    return Y[(burn + 1):end, :]
end

"""
    population_irf(A1, H) -> Vector{Float64}

True response of variable 2 to a unit shock in variable 1, horizons `0:H`.
With `Σ = I` and the shock ordered first the unit-impact vector is `e₁`, so
this is just `(A₁^h)[2,1]`.
"""
population_irf(A1::AbstractMatrix, H::Int) = [(A1 ^ h)[2, 1] for h in 0:H]

# One LP specification is used throughout; `@formula` cannot interpolate the lag
# count, so it is fixed here rather than passed as a parameter.
const LP_FORMULA = @formula(leads(y) ~ shock + lags(y, 4) + lags(shock, 4))
const LP_LAGS = 4

# ============================================================================
# Study 1: coverage of the three interval constructions
# ============================================================================

"""
    coverage_study(A1; T, H, nmc, nboot, level, seed0)

Estimate the coverage of the true IRF by each bootstrap interval construction
(`:hall_t`, `:hall`, `:efron`) and by the delta-method HC1 band, plus the mean
band width and how often the point estimate falls outside its own Hall-`t`
interval.

That last quantity is not a defect: a percentile-`t` interval corrects for bias
and can sit entirely to one side of θ̂. It is measured here to confirm it stays
rare (≈0.3% of horizon-replications in these DGPs).
"""
function coverage_study(A1::AbstractMatrix;
        T::Int = 240, H::Int = 8, nmc::Int = 300, nboot::Int = 500,
        level::Float64 = 0.90, seed0::Int = 1000)
    truth = population_irf(A1, H)
    methods = (:hall_t, :hall, :efron)
    hit = Dict(m => zeros(Int, H + 1) for m in methods)
    hit[:normal] = zeros(Int, H + 1)
    width = Dict(m => zeros(H + 1) for m in methods)
    width[:normal] = zeros(H + 1)
    outside = zeros(Int, H + 1)
    done = 0

    for r in 1:nmc
        Y = simulate_var1(A1, T, StableRNG(seed0 + r))
        df = DataFrame(shock = Y[:, 1], y = Y[:, 2])
        try
            m = lp(LP_FORMULA, df; horizon = H)
            b = varbootstrap(m, df; vars = [:shock, :y], nlags = LP_LAGS,
                nboot = nboot, rng = StableRNG(seed0 + 500_000 + r),
                threaded = true)
            for mth in methods
                s = summarize(b; level = level, method = mth)
                for h in 1:(H + 1)
                    (s.lower[h] <= truth[h] <= s.upper[h]) && (hit[mth][h] += 1)
                    width[mth][h] += s.upper[h] - s.lower[h]
                    if mth === :hall_t &&
                       (b.theta[h] < s.lower[h] || b.theta[h] > s.upper[h])
                        outside[h] += 1
                    end
                end
            end
            sn = summarize(m, HC1(); level = level)
            for h in 1:(H + 1)
                (sn.lower[h] <= truth[h] <= sn.upper[h]) && (hit[:normal][h] += 1)
                width[:normal][h] += sn.upper[h] - sn.lower[h]
            end
            done += 1
        catch err
            err isa InterruptException && rethrow()
            # A replication that cannot be estimated is dropped and reported.
        end
    end

    return (; truth, done, nmc, level,
        coverage = Dict(k => v ./ done for (k, v) in hit),
        meanwidth = Dict(k => v ./ done for (k, v) in width),
        outside = outside ./ done)
end

# ============================================================================
# Study 2: what the Pope and Herbst-Johannsen corrections buy
# ============================================================================

"""
    correction_study(A1; T, H, nmc, nboot, level, seed0)

Compare the recommended combination (Pope-corrected VAR DGP + Herbst–Johannsen
corrected LP) against the uncorrected "script 23" variant on the same simulated
samples, reporting coverage, mean point-estimate bias, and how often the Pope
stability safeguard attenuated or skipped the correction.
"""
function correction_study(A1::AbstractMatrix;
        T::Int = 240, H::Int = 8, nmc::Int = 300, nboot::Int = 500,
        level::Float64 = 0.90, seed0::Int = 3000)
    truth = population_irf(A1, H)
    cov_full = zeros(Int, H + 1)
    cov_plain = zeros(Int, H + 1)
    cov_norm = zeros(Int, H + 1)
    bias_plain = zeros(H + 1)
    bias_full = zeros(H + 1)
    deltas = Float64[]
    done = 0

    for r in 1:nmc
        Y = simulate_var1(A1, T, StableRNG(seed0 + r))
        df = DataFrame(shock = Y[:, 1], y = Y[:, 2])
        try
            m = lp(LP_FORMULA, df; horizon = H)
            boot_rng() = StableRNG(seed0 + 500_000 + r)
            bf = varbootstrap(m, df; vars = [:shock, :y], nlags = LP_LAGS,
                nboot = nboot, rng = boot_rng(), threaded = true)
            bp = varbootstrap(m, df; vars = [:shock, :y], nlags = LP_LAGS,
                nboot = nboot, rng = boot_rng(), threaded = true,
                biascorrect = false, popecorrect = false)
            sf = summarize(bf; level = level)
            sp = summarize(bp; level = level)
            sn = summarize(m, HC1(); level = level)
            for h in 1:(H + 1)
                (sf.lower[h] <= truth[h] <= sf.upper[h]) && (cov_full[h] += 1)
                (sp.lower[h] <= truth[h] <= sp.upper[h]) && (cov_plain[h] += 1)
                (sn.lower[h] <= truth[h] <= sn.upper[h]) && (cov_norm[h] += 1)
                bias_plain[h] += bp.theta[h] - truth[h]
                bias_full[h] += bf.theta[h] - truth[h]
            end
            push!(deltas, bf.pope_delta)
            done += 1
        catch err
            err isa InterruptException && rethrow()
        end
    end

    return (; truth, done, nmc, level,
        cov_full = cov_full ./ done,
        cov_plain = cov_plain ./ done,
        cov_norm = cov_norm ./ done,
        bias_plain = bias_plain ./ done,
        bias_full = bias_full ./ done,
        pope_full_fraction = isempty(deltas) ? NaN : mean(deltas .== 1.0),
        pope_mean_delta = isempty(deltas) ? NaN : mean(deltas))
end

# ============================================================================
# Reporting
# ============================================================================

function print_row(label, v)
    @printf("  %-26s", label)
    for x in v
        @printf("%8.3f", x)
    end
    println()
end

function report_coverage(name, res)
    H = length(res.truth) - 1
    println("\n", name)
    println("  replications: ", res.done, "/", res.nmc,
        "   nominal: ", res.level)
    @printf("  %-26s", "horizon")
    for h in 0:H
        @printf("%8d", h)
    end
    println()
    print_row("true IRF", res.truth)
    print_row("coverage: Hall percentile-t", res.coverage[:hall_t])
    print_row("coverage: Hall", res.coverage[:hall])
    print_row("coverage: Efron", res.coverage[:efron])
    print_row("coverage: normal (HC1)", res.coverage[:normal])
    print_row("width: Hall percentile-t", res.meanwidth[:hall_t])
    print_row("width: normal (HC1)", res.meanwidth[:normal])
    print_row("theta-hat outside band", res.outside)
end

function report_corrections(name, res)
    H = length(res.truth) - 1
    println("\n", name)
    println("  replications: ", res.done, "/", res.nmc,
        "   nominal: ", res.level)
    @printf("  %-26s", "horizon")
    for h in 0:H
        @printf("%8d", h)
    end
    println()
    print_row("true IRF", res.truth)
    print_row("cov: Pope + H-J", res.cov_full)
    print_row("cov: uncorrected", res.cov_plain)
    print_row("cov: normal (HC1)", res.cov_norm)
    print_row("bias: uncorrected", res.bias_plain)
    print_row("bias: H-J corrected", res.bias_full)
    @printf("  Pope applied in full: %.1f%% of samples (mean delta = %.3f)\n",
        100 * res.pope_full_fraction, res.pope_mean_delta)
end

# ============================================================================
# Entry point
# ============================================================================

const DGPS = [
    ("moderate persistence  (rho_y = 0.70)", [0.5 0.0; 0.6 0.70]),
    ("high persistence      (rho_y = 0.95)", [0.5 0.0; 0.6 0.95]),
    ("near unit root        (rho_y = 0.99)", [0.5 0.0; 0.6 0.99])
]

function main(nmc::Int = 300, nboot::Int = 500)
    println("VAR moving-block bootstrap: Monte Carlo coverage study")
    println("T = 240, horizons 0:8, ", nmc, " replications, ", nboot,
        " bootstrap draws, threads = ", Threads.nthreads())

    for (i, (name, A1)) in enumerate(DGPS)
        res = coverage_study(A1; nmc = nmc, nboot = nboot, seed0 = 1000 * i)
        report_coverage(name, res)
    end

    # The corrections matter most where the LP is most biased.
    name, A1 = DGPS[end]
    res = correction_study(A1; nmc = nmc, nboot = nboot, seed0 = 9000)
    report_corrections("corrections, " * name, res)

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    nmc = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 300
    nboot = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 500
    main(nmc, nboot)
end
