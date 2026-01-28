# ============================================================================
# Tests for MacroEconometricTools Integration
# ============================================================================

using TestItems

@testitem "as_irf_result basic construction" tags=[:met, :smoke] begin
    using LocalProjections
    using MacroEconometricTools
    using DataFrames
    using StatsModels
    using AxisArrays

    # Generate simple data
    n = 100
    df = DataFrame(
        y = randn(n) .+ cumsum(randn(n) * 0.1),
        x = randn(n)
    )

    # Fit local projection
    lp_result = lp(@formula(leads(y) ~ x), df; horizon = 5)

    # Convert to MET type
    irf_result = as_irf_result(lp_result)

    @test irf_result isa MacroEconometricTools.LocalProjectionIRFResult
    @test irf_result isa MacroEconometricTools.AbstractIRFResult{Float64}
    @test irf_result.data isa AxisArray
end

@testitem "as_irf_result data dimensions" tags=[:met, :smoke] begin
    using LocalProjections
    using MacroEconometricTools
    using DataFrames
    using StatsModels
    using AxisArrays

    n = 100
    df = DataFrame(
        y = randn(n),
        x = randn(n)
    )

    lp_result = lp(@formula(leads(y) ~ x), df; horizon = 10)

    irf_result = as_irf_result(lp_result)

    # Check dimensions: (response × shock × horizon)
    @test size(irf_result.data) == (1, 1, 11)  # 1 response, 1 shock, 11 horizons (0:10)

    # Check axis names
    @test axisnames(irf_result.data) == (:response, :shock, :horizon)

    # Check axis values
    response_axis = AxisArrays.axes(irf_result.data, Axis{:response})
    @test AxisArrays.axisvalues(response_axis)[1] == [lp_result.response]

    horizon_axis = AxisArrays.axes(irf_result.data, Axis{:horizon})
    @test AxisArrays.axisvalues(horizon_axis)[1] == collect(0:10)
end

@testitem "as_irf_result with vcov_estimator" tags=[:met] begin
    using LocalProjections
    using MacroEconometricTools
    using DataFrames
    using StatsModels
    using CovarianceMatrices
    using AxisArrays

    n = 100
    df = DataFrame(
        y = randn(n),
        x = randn(n)
    )

    lp_result = lp(@formula(leads(y) ~ x), df; horizon = 5)

    # Convert with vcov estimator for confidence bands
    irf_result = as_irf_result(lp_result; vcov_estimator = HC1())

    # Should have stderr
    @test irf_result.stderr isa AxisArray
    @test size(irf_result.stderr) == (1, 1, 6)
    @test all(irf_result.stderr .>= 0)  # Standard errors are non-negative

    # Should have confidence bands
    @test length(irf_result.coverage) == 3  # Default [0.68, 0.90, 0.95]
    @test length(irf_result.lower) == 3
    @test length(irf_result.upper) == 3

    # Lower should be <= data <= upper for all coverage levels
    data_arr = Array(irf_result.data)
    for i in 1:3
        lower_arr = Array(irf_result.lower[i])
        upper_arr = Array(irf_result.upper[i])
        @test all(lower_arr .<= data_arr)
        @test all(upper_arr .>= data_arr)
    end
end

@testitem "as_irf_result custom coverage" tags=[:met] begin
    using LocalProjections
    using MacroEconometricTools
    using DataFrames
    using StatsModels
    using CovarianceMatrices

    n = 100
    df = DataFrame(
        y = randn(n),
        x = randn(n)
    )

    lp_result = lp(@formula(leads(y) ~ x), df; horizon = 3)

    # Custom coverage levels
    irf_result = as_irf_result(lp_result;
        vcov_estimator = HC1(),
        coverage = [0.50, 0.99])

    @test irf_result.coverage == [0.50, 0.99]
    @test length(irf_result.lower) == 2
    @test length(irf_result.upper) == 2
end

@testitem "as_irf_result accessor functions" tags=[:met, :smoke] begin
    using LocalProjections
    using MacroEconometricTools
    using DataFrames
    using StatsModels
    using CovarianceMatrices
    using AxisArrays

    n = 100
    df = DataFrame(
        y = randn(n),
        x = randn(n)
    )

    lp_result = lp(@formula(leads(y) ~ x), df; horizon = 5)
    irf_result = as_irf_result(lp_result; vcov_estimator = HC1())

    # Test accessor functions
    @test MacroEconometricTools.has_draws(irf_result) == false
    @test MacroEconometricTools.n_draws(irf_result) == 0

    # Test point_estimate (returns data directly for LP)
    pe = MacroEconometricTools.point_estimate(irf_result)
    @test pe === irf_result.data

    # Test bounds accessors
    @test length(MacroEconometricTools.lowerbounds(irf_result)) == 3
    @test length(MacroEconometricTools.upperbounds(irf_result)) == 3
    @test MacroEconometricTools.coverages(irf_result) == [0.68, 0.90, 0.95]
end

@testitem "as_irf_result without vcov (zeros for stderr/bounds)" tags=[:met] begin
    using LocalProjections
    using MacroEconometricTools
    using DataFrames
    using StatsModels
    using AxisArrays

    n = 100
    df = DataFrame(
        y = randn(n),
        x = randn(n)
    )

    lp_result = lp(@formula(leads(y) ~ x), df; horizon = 3)

    # No vcov_estimator - should have zero stderr
    irf_result = as_irf_result(lp_result)

    @test all(Array(irf_result.stderr) .== 0.0)

    # Bounds should also be zeros
    for lb in irf_result.lower
        @test all(Array(lb) .== 0.0)
    end
    for ub in irf_result.upper
        @test all(Array(ub) .== 0.0)
    end
end

@testitem "as_irf_result custom term" tags=[:met] begin
    using LocalProjections
    using MacroEconometricTools
    using DataFrames
    using StatsModels
    using CovarianceMatrices
    using AxisArrays

    n = 100
    df = DataFrame(
        y = randn(n),
        x1 = randn(n),
        x2 = randn(n)
    )

    # LP with multiple regressors
    lp_result = lp(@formula(leads(y) ~ x1 + x2), df; horizon = 3, shock = :x1)

    # Get IRF for x2 instead of default (x1)
    irf_x2 = as_irf_result(lp_result; term = :x2, vcov_estimator = HC1())

    # Check that shock axis reflects the term
    shock_axis = AxisArrays.axes(irf_x2.data, Axis{:shock})
    @test AxisArrays.axisvalues(shock_axis)[1] == [:x2]

    # Verify data matches coefpath for x2
    coef_x2 = coefpath(lp_result; term = :x2)
    @test vec(Array(irf_x2.data)) ≈ coef_x2
end

@testitem "as_irf_result metadata preservation" tags=[:met] begin
    using LocalProjections
    using MacroEconometricTools
    using DataFrames
    using StatsModels

    n = 100
    df = DataFrame(
        y = randn(n),
        x = randn(n)
    )

    lp_result = lp(@formula(leads(y) ~ x), df; horizon = 8)
    irf_result = as_irf_result(lp_result; term = :x)

    # Metadata should be preserved
    @test haskey(irf_result.metadata, :horizon)
    @test irf_result.metadata.horizon == 8
    @test haskey(irf_result.metadata, :term)
    @test irf_result.metadata.term == :x
end

@testitem "as_irf_result with IV local projection" tags=[:met, :iv] begin
    using LocalProjections
    using MacroEconometricTools
    using DataFrames
    using StatsModels
    using CovarianceMatrices
    using AxisArrays

    # Generate data with endogeneity
    n = 200
    z = randn(n)
    u = randn(n)
    x = 0.5 * z + 0.5 * u  # Endogenous
    y = 1.0 .+ 2.0 * x .+ u

    df = DataFrame(y = y, x = x, z = z)

    # IV local projection
    result = lpiv(@formula(leads(y) ~ (x ~ z)), df; horizon = 5)

    # Convert to MET type
    irf_result = as_irf_result(result; vcov_estimator = HC1())

    @test irf_result isa MacroEconometricTools.LocalProjectionIRFResult
    @test irf_result.data isa AxisArray
    @test size(irf_result.data) == (1, 1, 6)  # (response, shock, horizons)

    # Accessor functions should work
    @test MacroEconometricTools.has_draws(irf_result) == false
    pe = MacroEconometricTools.point_estimate(irf_result)
    @test pe === irf_result.data
end

@testitem "as_irf_result IV with custom coverage" tags=[:met, :iv] begin
    using LocalProjections
    using MacroEconometricTools
    using DataFrames
    using StatsModels
    using CovarianceMatrices

    n = 200
    z = randn(n)
    u = randn(n)
    x = 0.5 * z + 0.5 * u
    y = 1.0 .+ 2.0 * x .+ u

    df = DataFrame(y = y, x = x, z = z)

    result = lpiv(@formula(leads(y) ~ (x ~ z)), df; horizon = 3)

    irf_result = as_irf_result(result;
        vcov_estimator = HC1(),
        coverage = [0.80])

    @test irf_result.coverage == [0.80]
    @test length(irf_result.lower) == 1
    @test length(irf_result.upper) == 1
end
