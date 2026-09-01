"""
Benchmarks for LocalProjections.jl

Benchmark suite measuring:
1. Local projection estimation across horizons
2. IV local projections
3. Post-estimation variance computation
4. Coefficient path extraction
"""

using BenchmarkTools
using DataFrames
using Random
using StableRNGs

# Optional: benchmark against Apple's Accelerate BLAS instead of OpenBLAS.
# macOS only, so it is loaded on request rather than unconditionally:
#     LP_ACCELERATE=1 julia --project=benchmark benchmark/run_asv.jl
# Must be loaded before the packages whose linear algebra we are timing.
if get(ENV, "LP_ACCELERATE", "0") == "1"
    using AppleAccelerate
end

using LocalProjections
using CovarianceMatrices
using StatsModels    # @formula

# ============================================================================
# Data Generation
# ============================================================================

const DEFAULT_SEED = 20240612

function generate_lp_data(rng::AbstractRNG, n::Int)
    # Generate AR(1) process for response
    y = zeros(n)
    y[1] = randn(rng)
    for t in 2:n
        y[t] = 0.7 * y[t - 1] + randn(rng)
    end

    # Generate shock variable
    x = randn(rng, n)

    # Generate controls
    w = randn(rng, n)
    z = randn(rng, n)

    df = DataFrame(y = y, x = x, w = w, z = z)
    return df
end

function generate_lpiv_data(rng::AbstractRNG, n::Int)
    # Generate instruments
    z1 = randn(rng, n)
    z2 = randn(rng, n)

    # Generate endogenous variable (correlated with error)
    u = randn(rng, n)
    x = 0.5 * z1 + 0.3 * z2 + 0.5 * u + randn(rng, n)

    # Generate AR response
    y = zeros(n)
    y[1] = randn(rng)
    for t in 2:n
        y[t] = 0.5 * y[t - 1] + 2.0 * x[t] + u[t]
    end

    # Controls
    w = randn(rng, n)

    df = DataFrame(y = y, x = x, z1 = z1, z2 = z2, w = w)
    return df
end

# ============================================================================
# Benchmark Suite
# ============================================================================

const SUITE = BenchmarkGroup()

# ----------------------------------------------------------------------------
# Standard Local Projection Benchmarks
# ----------------------------------------------------------------------------

SUITE["lp"] = BenchmarkGroup()

# Short horizon, medium sample
let rng = StableRNG(DEFAULT_SEED)
    df = generate_lp_data(rng, 500)

    SUITE["lp"]["h12_n500"] = @benchmarkable lp(@formula(leads(y) ~ x + lag(w)), $df; horizon = 12)
end

# Medium horizon, medium sample
let rng = StableRNG(DEFAULT_SEED + 1)
    df = generate_lp_data(rng, 500)

    SUITE["lp"]["h24_n500"] = @benchmarkable lp(@formula(leads(y) ~ x + lag(w)), $df; horizon = 24)
end

# Short horizon, large sample
let rng = StableRNG(DEFAULT_SEED + 2)
    df = generate_lp_data(rng, 2000)

    SUITE["lp"]["h12_n2000"] = @benchmarkable lp(@formula(leads(y) ~ x + lag(w)), $df; horizon = 12)
end

# Long horizon, large sample
let rng = StableRNG(DEFAULT_SEED + 3)
    df = generate_lp_data(rng, 2000)

    SUITE["lp"]["h36_n2000"] = @benchmarkable lp(@formula(leads(y) ~ x + lag(w)), $df; horizon = 36)
end

# ----------------------------------------------------------------------------
# Cumulative Response Benchmarks
# ----------------------------------------------------------------------------

SUITE["lp_cumul"] = BenchmarkGroup()

let rng = StableRNG(DEFAULT_SEED + 10)
    df = generate_lp_data(rng, 500)

    SUITE["lp_cumul"]["h12_n500"] = @benchmarkable lp(@formula(cumul(y) ~ x + lag(w)), $df; horizon = 12)
    SUITE["lp_cumul"]["h24_n500"] = @benchmarkable lp(@formula(cumul(y) ~ x + lag(w)), $df; horizon = 24)
end

# ----------------------------------------------------------------------------
# IV Local Projection Benchmarks
# ----------------------------------------------------------------------------

SUITE["lpiv"] = BenchmarkGroup()

let rng = StableRNG(DEFAULT_SEED + 20)
    df = generate_lpiv_data(rng, 500)

    SUITE["lpiv"]["h12_n500"] = @benchmarkable lpiv(@formula(leads(y) ~ (x ~ z1 + z2) + w), $df; horizon = 12)
end

let rng = StableRNG(DEFAULT_SEED + 21)
    df = generate_lpiv_data(rng, 1000)

    SUITE["lpiv"]["h12_n1000"] = @benchmarkable lpiv(@formula(leads(y) ~ (x ~ z1 + z2) + w), $df; horizon = 12)
    SUITE["lpiv"]["h24_n1000"] = @benchmarkable lpiv(@formula(leads(y) ~ (x ~ z1 + z2) + w), $df; horizon = 24)
end

# ----------------------------------------------------------------------------
# Post-Estimation Benchmarks
# ----------------------------------------------------------------------------

SUITE["vcov"] = BenchmarkGroup()

let rng = StableRNG(DEFAULT_SEED + 30)
    df = generate_lp_data(rng, 500)
    lp_result = lp(@formula(leads(y) ~ x + lag(w)), df; horizon = 12)

    SUITE["vcov"]["hc1_h12"] = @benchmarkable vcov(HC1(), $lp_result)
end

let rng = StableRNG(DEFAULT_SEED + 31)
    df = generate_lp_data(rng, 500)
    lp_result = lp(@formula(leads(y) ~ x + lag(w)), df; horizon = 24)

    SUITE["vcov"]["hc1_h24"] = @benchmarkable vcov(HC1(), $lp_result)
end

# ----------------------------------------------------------------------------
# Coefficient Path Extraction Benchmarks
# ----------------------------------------------------------------------------

SUITE["coefpath"] = BenchmarkGroup()

let rng = StableRNG(DEFAULT_SEED + 40)
    df = generate_lp_data(rng, 500)
    lp_result = lp(@formula(leads(y) ~ x + lag(w)), df; horizon = 24)

    SUITE["coefpath"]["h24"] = @benchmarkable coefpath($lp_result; term = :x)
end

# ----------------------------------------------------------------------------
# Lag Selection and VAR Bootstrap Benchmarks
# ----------------------------------------------------------------------------

function generate_var_data(rng::AbstractRNG, n::Int)
    shock = zeros(n)
    y = zeros(n)
    for t in 2:n
        shock[t] = 0.3 * shock[t - 1] + randn(rng)
        y[t] = 0.6 * y[t - 1] + 0.8 * shock[t] + randn(rng)
    end
    return DataFrame(shock = shock, y = y)
end

# AirspeedVelocity loads this file against every revision it compares, so
# benchmarks of features absent from the baseline must be skipped there.
if isdefined(LocalProjections, :lagselect) && isdefined(LocalProjections, :varbootstrap)
    SUITE["lagselect"] = BenchmarkGroup()

    let rng = StableRNG(DEFAULT_SEED + 50)
        df = generate_var_data(rng, 500)

        SUITE["lagselect"]["aic_pmax10"] = @benchmarkable lagselect(
            $df, [:shock, :y]; maxlags = 10)
    end

    SUITE["bootstrap"] = BenchmarkGroup()

    let rng = StableRNG(DEFAULT_SEED + 51)
        df = generate_var_data(rng, 240)
        m = lp(@formula(leads(y) ~ shock + lags(y, 4) + lags(shock, 4)), df; horizon = 20)

        # One draw is the unit of work that dominates the bootstrap; keep nboot small
        # so the suite stays fast under AirspeedVelocity.
        SUITE["bootstrap"]["nboot100_h20"] = @benchmarkable varbootstrap(
            $m, $df; vars = [:shock, :y], nlags = 4, nboot = 100,
            rng = StableRNG(1))

        SUITE["bootstrap"]["nboot100_h20_nocorrections"] = @benchmarkable varbootstrap(
            $m, $df; vars = [:shock, :y], nlags = 4, nboot = 100,
            rng = StableRNG(1), biascorrect = false, popecorrect = false)

        b = varbootstrap(m, df; vars = [:shock, :y], nlags = 4, nboot = 100,
            rng = StableRNG(1))
        SUITE["bootstrap"]["summarize_hall_t"] = @benchmarkable summarize(
            $b; level = 0.90, method = :hall_t)
    end
end # isdefined guard
