using TestItems

@testitem "cumul transformation" tags = [:cumul, :core] begin
    using LocalProjections
    using DataFrames, StatsModels, Regress, StatsBase, Test

    # Create simple synthetic data
    n = 100
    df = DataFrame(x = 1.0:n, y = sin.(1.0:n) .+ (1.0:n) ./ 10)

    # Test cumul transformation in lp() context
    horizon = 3
    lp_result = lp(@formula(cumul(y) ~ x), df; horizon = horizon)

    # Manually compute cumulative sums and compare
    # Note: Must replicate lp's filtering logic exactly
    df_filtered = dropmissing(df, [:y, :x], disallowmissing = true)

    for h in 0:horizon
        # Get coefficient from lp result
        lp_coef = coef(lp_result.models[h + 1])[2]  # x coefficient (index 2, after intercept)

        # Manually compute cumulative y at horizon h using StatsModels
        cumul_term_h = CumulTerm{typeof(Term(:y))}(Term(:y), h)
        y_h = StatsModels.modelcols(cumul_term_h, df_filtered)
        y_h_manual = map(x -> sum(x), map(t -> df_filtered.y[t:(t + h)], 1:(nrow(df_filtered) - h)))

        # Find complete observations (no NaN in y_h)
        complete_obs = .!isnan.(y_h)
        y_manual = y_h[complete_obs]
        # Manually run regression on complete data
        X_manual = hcat(ones(sum(complete_obs)), df_filtered.x[complete_obs])

        @test y_manual == y_h_manual  # Verify manual cumulative matches StatsModels output
        manual_coef = (X_manual \ y_manual)[2]  # x coefficient

        # Compare coefficients
        @test lp_coef ≈ manual_coef atol = 1e-10
    end
end

@testitem "leads transformation" tags = [:leads, :core] begin
    using LocalProjections
    using DataFrames, StatsModels, Regress, StatsBase, Test
    using ShiftedArrays: lead

    # Create simple synthetic data
    n = 100
    df = DataFrame(x = 1.0:n, y = cos.(1.0:n) .+ (1.0:n) ./ 20)

    # Test leads transformation in lp() context
    horizon = 3
    lp_result = lp(@formula(leads(y) ~ x), df; horizon = horizon)

    # Manually compute leads and compare
    df_filtered = dropmissing(df, [:y, :x], disallowmissing = true)

    for h in 0:horizon
        # Get coefficient from lp result
        lp_coef = coef(lp_result.models[h + 1])[2]  # x coefficient

        # Manually compute lead of y at horizon h using StatsModels
        leads_term_h = LeadTerm{typeof(Term(:y))}(Term(:y), h)
        y_h = StatsModels.modelcols(leads_term_h, df_filtered)

        # Find complete observations (no NaN)
        complete_obs = .!isnan.(y_h)

        # Manually run regression on complete data
        X_manual = hcat(ones(sum(complete_obs)), df_filtered.x[complete_obs])
        y_manual = y_h[complete_obs]
        manual_coef = (X_manual \ y_manual)[2]  # x coefficient

        # Compare coefficients
        @test lp_coef ≈ manual_coef atol = 1e-10
    end

    # Verify NaN handling (not missing)
    y_lead_test = lead(df.y, 5, default = NaN)
    @test eltype(y_lead_test) == Float64
    @test any(isnan, y_lead_test)  # Should have NaN at boundaries
end

@testitem "anchor function syntax" tags = [:anchor, :core] begin
    using LocalProjections
    using DataFrames, StatsModels, Regress, StatsBase, Test

    # Create simple synthetic data with both y and z
    n = 100
    df = DataFrame(x = 1.0:n, y = sin.(1.0:n) .+ (1.0:n) ./ 10, z = cos.(1.0:n))

    # Test anchor transformation with function syntax
    horizon = 3
    lp_result = lp(@formula(anchor(y, z) ~ x), df; horizon = horizon)

    # Manually compute anchored response and compare
    df_filtered = dropmissing(df, [:y, :z, :x], disallowmissing = true)

    for h in 0:horizon
        # Get coefficient from lp result
        lp_coef = coef(lp_result.models[h + 1])[2]  # x coefficient

        # Manually compute anchored response using StatsModels
        inner_leads = LeadTerm{typeof(Term(:y))}(Term(:y), h)
        anchor_h = AnchorTerm{typeof(inner_leads), typeof(Term(:z))}(inner_leads, Term(:z), 0)  # horizon=0 because lead is in inner term
        y_h = StatsModels.modelcols(anchor_h, df_filtered)

        # Find complete observations (no NaN)
        complete_obs = .!isnan.(y_h)

        # Manually run regression on complete data
        X_manual = hcat(ones(sum(complete_obs)), df_filtered.x[complete_obs])
        y_manual = y_h[complete_obs]
        manual_coef = (X_manual \ y_manual)[2]  # x coefficient

        # Compare coefficients
        @test lp_coef ≈ manual_coef atol = 1e-10
    end
end

@testitem "anchor pipe syntax" tags = [:anchor, :core] begin
    using LocalProjections
    using DataFrames, StatsModels, Regress, StatsBase, Test

    # Create simple synthetic data
    n = 100
    df = DataFrame(x = 1.0:n, y = sin.(1.0:n) .+ (1.0:n) ./ 10, z = cos.(1.0:n))

    # Test anchor with pipe syntax
    horizon = 3
    lp_pipe = lp(@formula(leads(y) | z ~ x), df; horizon = horizon)

    # Test anchor with function syntax (should be identical)
    lp_func = lp(@formula(anchor(y, z) ~ x), df; horizon = horizon)

    # Compare results from both syntaxes
    for h in 0:horizon
        pipe_coef = coef(lp_pipe.models[h + 1])[2]
        func_coef = coef(lp_func.models[h + 1])[2]

        @test pipe_coef ≈ func_coef atol = 1e-10
    end

    # Also verify against manual computation
    df_filtered = dropmissing(df, [:y, :z, :x], disallowmissing = true)

    for h in 0:horizon
        lp_coef = coef(lp_pipe.models[h + 1])[2]

        # Manual computation using StatsModels
        inner_leads = LeadTerm{typeof(Term(:y))}(Term(:y), h)
        anchor_h = AnchorTerm{typeof(inner_leads), typeof(Term(:z))}(inner_leads, Term(:z), 0)  # horizon=0 because lead is in inner term
        y_h = StatsModels.modelcols(anchor_h, df_filtered)

        y_h_manual = lead(df_filtered.y, h, default = NaN) .- df_filtered.z
        complete_obs = .!isnan.(y_h)
        X_manual = hcat(ones(sum(complete_obs)), df_filtered.x[complete_obs])
        y_manual = y_h[complete_obs]
        y_manual_manual = y_h_manual[complete_obs]
        @test y_manual == y_manual_manual  # Verify manual matches StatsModels output
        manual_coef = (X_manual \ y_manual)[2]

        @test lp_coef ≈ manual_coef atol = 1e-10
    end
end

@testitem "modelcols anchor matches manual lead computation" tags = [:anchor, :verification] begin
    using LocalProjections
    using DataFrames, StatsModels, Test

    # Create test data
    n = 100
    df = DataFrame(y = sin.(1.0:n) .+ (1.0:n) ./ 10, z = cos.(1.0:n))

    # Test that modelcols(AnchorTerm) matches manual lead() - z computation
    for h in 0:5
        # Method 1: Using StatsModels.modelcols with AnchorTerm
        inner_leads = LeadTerm{typeof(Term(:y))}(Term(:y), h)
        anchor_h = AnchorTerm{typeof(inner_leads), typeof(Term(:z))}(inner_leads, Term(:z), 0)
        y_modelcols = StatsModels.modelcols(anchor_h, df)

        # Method 2: Manual computation using lead() - z
        y_manual = lead(df.y, h, default = NaN) .- df.z

        # Compare (must handle NaN carefully)
        for i in 1:n
            if isnan(y_modelcols[i]) && isnan(y_manual[i])
                @test true  # Both NaN, OK
            elseif isnan(y_modelcols[i]) || isnan(y_manual[i])
                @test false  # One NaN, other not - FAIL
            else
                @test y_modelcols[i] ≈ y_manual[i] atol = 1e-10
            end
        end
    end
end

@testitem "cumulative anchor (nested)" tags = [:nested, :anchor, :cumul] begin
    using LocalProjections
    using DataFrames, StatsModels, Regress, StatsBase, Test

    # Create simple synthetic data
    n = 100
    df = DataFrame(x = 1.0:n, y = sin.(1.0:n) .+ (1.0:n) ./ 10, z = cos.(1.0:n))

    # Test cumulative anchor: cumul(y)|z
    horizon = 3
    lp_result = lp(@formula(cumul(y) | z ~ x), df; horizon = horizon)

    # Manually compute cumulative anchored response
    df_filtered = dropmissing(df, [:y, :z, :x], disallowmissing = true)

    for h in 0:horizon
        # Get coefficient from lp result
        lp_coef = coef(lp_result.models[h + 1])[2]

        # Manual computation using StatsModels: first cumul, then anchor
        inner_cumul = CumulTerm{typeof(Term(:y))}(Term(:y), h)
        anchor_h = AnchorTerm{typeof(inner_cumul), typeof(Term(:z))}(inner_cumul, Term(:z), 0)  # horizon=0 because cumul already has horizon
        y_h = StatsModels.modelcols(anchor_h, df_filtered)

        # Find complete observations (no NaN)
        complete_obs = .!isnan.(y_h)

        # Manually run regression
        X_manual = hcat(ones(sum(complete_obs)), df_filtered.x[complete_obs])
        y_manual = y_h[complete_obs]
        manual_coef = (X_manual \ y_manual)[2]

        # Compare coefficients
        @test lp_coef ≈ manual_coef atol = 1e-10
    end
end

@testitem "nested log transformations" tags = [:nested, :core] begin
    using LocalProjections
    using DataFrames, StatsModels, Regress, StatsBase, Test

    # Create simple synthetic data (positive values for log)
    n = 100
    df = DataFrame(
        x = 1.0:n,
        y = exp.(1.0:n) ./ 100,  # Positive values
        z = exp.(2.0:101) ./ 150
    )

    # Test cumul(log(y))
    horizon = 2
    lp_cumul_log = lp(@formula(cumul(log(y)) ~ x), df; horizon = horizon)
    df_filtered1 = dropmissing(df, [:y, :x], disallowmissing = true)

    for h in 0:horizon
        lp_coef = coef(lp_cumul_log.models[h + 1])[2]

        # Manual computation: cumulative sum of log(y) from t to t+h
        log_y = log.(df_filtered1.y)
        y_h = [t + h <= length(log_y) ? sum(log_y[t:(t + h)]) : NaN
               for t in 1:length(log_y)]
        complete_obs = .!isnan.(y_h)

        X_manual = hcat(ones(sum(complete_obs)), df_filtered1.x[complete_obs])
        y_manual = y_h[complete_obs]
        manual_coef = (X_manual \ y_manual)[2]

        @test lp_coef ≈ manual_coef atol = 1e-10
    end

    # Test leads(log(y))
    lp_lead_log = lp(@formula(leads(log(y)) ~ x), df; horizon = horizon)
    df_filtered2 = dropmissing(df, [:y, :x], disallowmissing = true)

    for h in 0:horizon
        lp_coef = coef(lp_lead_log.models[h + 1])[2]

        # Manual computation: lead of log(y) by h
        log_y = log.(df_filtered2.y)
        y_h = [t + h <= length(log_y) ? log_y[t + h] : NaN for t in 1:length(log_y)]
        complete_obs = .!isnan.(y_h)

        X_manual = hcat(ones(sum(complete_obs)), df_filtered2.x[complete_obs])
        y_manual = y_h[complete_obs]
        manual_coef = (X_manual \ y_manual)[2]

        @test lp_coef ≈ manual_coef atol = 1e-10
    end

    # Test anchor(log(y), z)
    lp_anchor_log = lp(@formula(anchor(log(y), z) ~ x), df; horizon = horizon)
    df_filtered3 = dropmissing(df, [:y, :z, :x], disallowmissing = true)

    for h in 0:horizon
        lp_coef = coef(lp_anchor_log.models[h + 1])[2]

        # Manual computation: lead of log(y) by h, minus z at t
        log_y = log.(df_filtered3.y)
        y_h = [t + h <= length(log_y) ? log_y[t + h] - df_filtered3.z[t] : NaN
               for
               t in 1:length(log_y)]
        complete_obs = .!isnan.(y_h)

        X_manual = hcat(ones(sum(complete_obs)), df_filtered3.x[complete_obs])
        y_manual = y_h[complete_obs]
        manual_coef = (X_manual \ y_manual)[2]

        # Use relative tolerance for large values
        @test lp_coef ≈ manual_coef rtol = 1e-10
    end
end

@testitem "summarize function" tags = [:summarize, :api] begin
    using LocalProjections
    using DataFrames, StatsModels, Test
    using CovarianceMatrices: HC1

    n = 100
    df = DataFrame(x = randn(n), y = randn(n))
    lp_result = lp(@formula(leads(y) ~ x), df; horizon = 5)
    cov = LocalProjections.vcov(HC1(), lp_result)

    # Test basic summarize returns IRFSummary
    summary_obj = summarize(lp_result, cov)
    @test summary_obj isa IRFSummary
    @test length(summary_obj.horizon) == 6  # horizons 0-5
    @test summary_obj.horizon == collect(0:5)
    @test summary_obj.term == :x
    @test summary_obj.level == 0.95

    # Test conversion to DataFrame
    summary_df = DataFrame(summary_obj)
    @test summary_df isa DataFrame
    @test nrow(summary_df) == 6
    @test names(summary_df) == ["horizon", "coef", "se", "lower", "upper"]

    # Test with scale
    summary_scaled = summarize(lp_result, cov; scale = 100)
    @test summary_scaled.coef ≈ summary_obj.coef .* 100
    @test summary_scaled.scale == 100.0

    # Test with estimator directly
    summary_direct = summarize(lp_result, HC1())
    @test summary_direct.coef ≈ summary_obj.coef atol = 1e-10

    # Test confidence bounds are sensible (lower < coef < upper when se > 0)
    for i in 1:length(summary_obj.horizon)
        if summary_obj.se[i] > 0
            @test summary_obj.lower[i] < summary_obj.coef[i] < summary_obj.upper[i]
        end
    end

    # Test different confidence level
    summary_90 = summarize(lp_result, cov; level = 0.90)
    @test summary_90.level == 0.90
    # 90% CI should be narrower than 95% CI
    @test all(
        summary_90.upper .- summary_90.lower .< summary_obj.upper .- summary_obj.lower,
    )
end

@testitem "plus vcov operator" tags = [:vcov, :api] begin
    using LocalProjections
    using LocalProjections: VcovSpec
    using DataFrames, StatsModels, Test
    using CovarianceMatrices: HC1, Bartlett, NeweyWest
    using Regress: vcov

    n = 100
    df = DataFrame(x = randn(n), y = randn(n))
    lp_result = lp(@formula(leads(y) ~ x), df; horizon = 5)

    # Test + vcov() syntax
    lp_robust = lp_result + vcov(Bartlett{NeweyWest}())

    # Result should be a LocalProjection
    @test lp_robust isa LocalProjection

    # Coefficients should be unchanged
    @test coefpath(lp_result) == coefpath(lp_robust)

    # Horizon and metadata should be preserved
    @test lp_robust.horizon == lp_result.horizon
    @test lp_robust.response == lp_result.response
    @test lp_robust.shock == lp_result.shock
    @test lp_robust.base_formula == lp_result.base_formula
    @test lp_robust.coef_names == lp_result.coef_names

    # Number of models should be the same
    @test length(lp_robust.models) == length(lp_result.models)

    # Test chaining: (lp + vcov(A)) + vcov(B)
    lp_chained = (lp_result + vcov(HC1())) + vcov(Bartlett{NeweyWest}())
    @test lp_chained isa LocalProjection
    @test coefpath(lp_chained) == coefpath(lp_result)
end

@testitem "lpiv basic functionality" tags = [:lpiv, :iv, :core] begin
    using LocalProjections
    using LocalProjections: FirstStageIV
    using DataFrames, StatsModels, Test
    using Random
    using CovarianceMatrices: HC1
    using LinearAlgebra

    Random.seed!(889977)

    # Generate data with endogeneity
    n = 200
    z = randn(n)
    u = randn(n)
    x = 0.5 * z + 0.5 * u      # Endogenous (correlated with u)
    y = 1.0 .+ 2.0 * x .+ u  # True effect = 2.0

    df = DataFrame(y = y, x = x, z = z)

    # Fit IV local projection
    horizon = 3
    result = lpiv(@formula(leads(y) ~ (x ~ z)), df; horizon = horizon)

    # Check struct type
    @test result isa LocalProjectionIV
    @test result.horizon == horizon
    @test result.response == :y
    @test result.shock == :x

    # Check that we have the right number of models
    @test length(result.models) == horizon + 1

    # Check endogenous and instrument names
    @test :x in Symbol.(result.endogenous_names) || "x" in result.endogenous_names
    @test :z in Symbol.(result.instrument_names) || "z" in result.instrument_names

    # Extract IRF - coefficient should be close to 2.0 for horizon 0
    irf = coefpath(result; term = :x)
    @test length(irf) == horizon + 1
    # IV should recover true effect ~2.0 (with some noise due to finite sample)
    @test irf[1] > 1.0 && irf[1] < 3.0  # Reasonable range

    # Test first_stage
    fs = first_stage(result, 0)
    @test fs isa FirstStageIV
    @test fs.F_nonrobust[1] > 0
    @test fs.F_robust[1] > 0

    # Test first_stage for all horizons
    all_fs = first_stage(result)
    @test length(all_fs) == horizon + 1

    # Test vcov
    cov = LocalProjections.vcov(HC1(), result)
    @test cov isa LocalProjectionCovariance
    se = stderror(cov; term = :x)
    @test length(se) == horizon + 1
    @test all(se .> 0)  # Standard errors should be positive
end

@testitem "lpiv with controls" tags = [:lpiv, :iv] begin
    using LocalProjections
    using DataFrames, StatsModels, Test
    using Random
    using LinearAlgebra

    Random.seed!(123)

    # Generate data with endogeneity and exogenous control
    n = 200
    z = randn(n)
    w = randn(n)           # Exogenous control
    u = randn(n)
    x = 0.5 * z + 0.3 * w + 0.4 * u  # Endogenous
    y = 1.0 .+ 2.0 * x .+ 0.5 * w .+ u

    df = DataFrame(y = y, x = x, z = z, w = w)

    # Fit IV LP with control
    horizon = 2
    result = lpiv(@formula(leads(y) ~ (x ~ z) + w), df; horizon = horizon)

    @test result isa LocalProjectionIV

    # Check coefficient names include both x and w
    names = coefnames(result)
    @test "x" in names
    @test "w" in names

    # Get coefficients for both terms
    irf_x = coefpath(result; term = :x)
    irf_w = coefpath(result; term = :w)

    @test length(irf_x) == horizon + 1
    @test length(irf_w) == horizon + 1
end

@testitem "lpiv plus vcov operator" tags = [:lpiv, :vcov, :api] begin
    using LocalProjections
    using LocalProjections: VcovSpec
    using DataFrames, StatsModels, Test
    using CovarianceMatrices: HC1, Bartlett, NeweyWest
    using Regress: vcov
    using Random
    using LinearAlgebra

    Random.seed!(456)

    n = 150
    z = randn(n)
    u = randn(n)
    x = 0.6 * z + 0.4 * u
    y = 0.5 .+ 1.5 * x .+ u

    df = DataFrame(y = y, x = x, z = z)

    result = lpiv(@formula(leads(y) ~ (x ~ z)), df; horizon = 3)

    # Test + vcov() syntax
    result_hac = result + vcov(Bartlett{NeweyWest}())

    # Result should be a LocalProjectionIV
    @test result_hac isa LocalProjectionIV

    # Coefficients should be unchanged
    @test coefpath(result) == coefpath(result_hac)

    # Metadata should be preserved
    @test result_hac.horizon == result.horizon
    @test result_hac.response == result.response
    @test result_hac.shock == result.shock
    @test result_hac.endogenous_names == result.endogenous_names
    @test result_hac.instrument_names == result.instrument_names

    # Test chaining
    result_chained = (result + vcov(HC1())) + vcov(Bartlett{NeweyWest}())
    @test result_chained isa LocalProjectionIV
    @test coefpath(result_chained) == coefpath(result)
end

@testitem "lpiv with cumul response" tags = [:lpiv, :cumul] begin
    using LocalProjections
    using DataFrames, StatsModels, Test
    using Random
    using LinearAlgebra

    Random.seed!(789)

    n = 150
    z = randn(n)
    u = randn(n)
    x = 0.5 * z + 0.5 * u
    y = 1.0 .+ 2.0 * x .+ u

    df = DataFrame(y = y, x = x, z = z)

    # Cumulative IV local projection
    horizon = 2
    result = lpiv(@formula(cumul(y) ~ (x ~ z)), df; horizon = horizon)

    @test result isa LocalProjectionIV
    @test result.response == :y

    irf = coefpath(result; term = :x)
    @test length(irf) == horizon + 1

    # Cumulative IRF should grow (roughly) with horizon
    # At h=0: ~2.0 (impact)
    # At h=1: ~4.0 (sum of two periods)
    # This is approximate due to estimation noise
    @test irf[1] > 0  # Impact positive
end

@testitem "lpiv summarize" tags = [:lpiv, :summarize, :api] begin
    using LocalProjections
    using DataFrames, StatsModels, Test
    using CovarianceMatrices: HC1
    using Random
    using LinearAlgebra

    Random.seed!(1011)

    n = 150
    z = randn(n)
    u = randn(n)
    x = 0.5 * z + 0.5 * u
    y = 1.0 .+ 2.0 * x .+ u

    df = DataFrame(y = y, x = x, z = z)

    result = lpiv(@formula(leads(y) ~ (x ~ z)), df; horizon = 3)
    cov = LocalProjections.vcov(HC1(), result)

    # Test summarize
    summary_obj = summarize(result, cov)
    @test summary_obj isa IRFSummary
    @test length(summary_obj.horizon) == 4  # 0-3
    @test summary_obj.term == :x

    # Convert to DataFrame
    summary_df = DataFrame(summary_obj)
    @test summary_df isa DataFrame
    @test nrow(summary_df) == 4

    # Test with scale
    summary_scaled = summarize(result, cov; scale = 100)
    @test summary_scaled.coef ≈ summary_obj.coef .* 100

    # Test with estimator directly
    summary_direct = summarize(result, HC1())
    @test summary_direct.coef ≈ summary_obj.coef atol = 1e-10
end

@testitem "lpiv comparison with manual 2SLS" tags = [:lpiv, :iv, :verification] begin
    using LocalProjections
    using DataFrames, StatsModels, Test
    using LinearAlgebra
    using Random

    Random.seed!(2022)

    # Generate data
    n = 200
    z = randn(n)
    u = randn(n)
    x = 0.5 * z + 0.5 * u
    y = 1.0 .+ 2.0 * x .+ u

    df = DataFrame(y = y, x = x, z = z)

    # lpiv result for horizon 0
    result = lpiv(@formula(leads(y) ~ (x ~ z)), df; horizon = 0)
    lpiv_coef = coefpath(result; term = :x)[1]

    # Manual 2SLS for comparison
    # Z = [ones, z], X = [ones, x]
    Z_manual = hcat(ones(n), z)
    X_manual = hcat(ones(n), x)
    y_manual = y

    # Stage 1: Xhat = Z * (Z'Z)^{-1} Z' X
    Pz = Z_manual * inv(Z_manual' * Z_manual) * Z_manual'
    X_hat = Pz * X_manual

    # Stage 2: beta = (Xhat'Xhat)^{-1} Xhat'y
    beta_2sls = (X_hat' * X_hat) \ (X_hat' * y_manual)

    # Compare coefficients (second element is x coefficient)
    @test lpiv_coef ≈ beta_2sls[2] atol = 1e-8
end

@testitem "weakivtest for lpiv" tags = [:lpiv, :weakiv, :iv] begin
    using LocalProjections
    using DataFrames, StableRNGs, StatsModels

    rng = StableRNG(4433222)
    n = 300
    z = randn(rng, n)
    z2 = randn(rng, n)
    z3 = randn(rng, n)
    x = 0.5 * z + 0.3 * z2 + 0.2 * z3 + randn(rng, n)
    y = cumsum(x) + 0.5 * randn(rng, n)
    df = DataFrame(y = y, x = x, z = z, z2 = z2, z3 = z3)

    horizon = 4
    result = lpiv(@formula(leads(y) ~ (x ~ z + z2 + z3)), df; horizon = horizon)

    # Test single-horizon call
    r0 = weakivtest(result, 0)
    @test r0 isa WeakIVTestResult
    @test r0.K == 3
    @test r0.F_eff > 0
    @test r0.F_robust > 0
    @test r0.F_nonrobust > 0
    @test all(cv -> cv > 0, r0.cv_TSLS)
    @test r0.cv_TSLS[1] > r0.cv_TSLS[2] > r0.cv_TSLS[3] > r0.cv_TSLS[4]

    # Test all-horizons call
    results = weakivtest(result)
    @test length(results) == horizon + 1
    @test all(r -> r isa WeakIVTestResult, results)
    @test all(r -> r.F_eff > 0, results)

    # Test bounds checking
    @test_throws BoundsError weakivtest(result, -1)
    @test_throws BoundsError weakivtest(result, horizon + 1)

    # Test keyword forwarding
    r_ols = weakivtest(result, 0; benchmark = :ols)
    @test r_ols.F_eff ≈ r0.F_eff  # F-stats are the same regardless of benchmark
end

@testitem "lpiv with lags() exogenous controls" tags = [:lpiv, :iv, :lags] begin
    using LocalProjections
    using DataFrames, StableRNGs, StatsModels

    # Regression test: lpiv with lags() terms referencing columns other than the
    # response variable. Previously failed because StatsModels.termvars was not
    # defined for FunctionTerm{typeof(lags)} and LagTerm, so ModelFrame did not
    # select the needed columns.
    rng = StableRNG(9988)
    n = 200
    z = randn(rng, n)
    x = 0.8 * z + randn(rng, n)
    w1 = randn(rng, n)
    w2 = randn(rng, n)
    y = cumsum(x + 0.3 * w1 + 0.2 * w2) + 0.5 * randn(rng, n)
    df = DataFrame(y = y, x = x, z = z, w1 = w1, w2 = w2)

    horizon = 4

    # This should not error — lags(w1, 2) and lags(w2, 2) reference columns
    # :w1 and :w2 which are not the response variable :y
    result = lpiv(
        @formula(leads(y) ~ (x ~ z) + lags(w1, 2) + lags(w2, 2)),
        df; horizon = horizon
    )
    @test result isa LocalProjectionIV
    @test length(result.models) == horizon + 1

    # Verify coefficients are finite
    irf = coefpath(result; term = :x)
    @test length(irf) == horizon + 1
    @test all(isfinite, irf)

    # Verify the expected number of coefficients:
    # intercept + 2 lags of w1 + 2 lags of w2 + x = 6
    @test length(result.coef_names) == 6
end

@testitem "tautological h=0 when response == shock" tags = [:lpiv, :iv, :tautological] begin
    using LocalProjections
    using DataFrames, StableRNGs, StatsModels
    using CovarianceMatrices: HC1

    rng = StableRNG(7766)
    n = 200
    z = randn(rng, n)
    x = 0.8 * z + randn(rng, n)
    w = randn(rng, n)
    y = cumsum(x + 0.3 * w) + 0.5 * randn(rng, n)
    df = DataFrame(y = y, x = x, z = z, w = w)

    horizon = 4

    # IV LP where response == shock (x is both response and endogenous)
    result = lpiv(@formula(leads(x) ~ (x ~ z) + w), df; horizon = horizon)

    @test result isa LocalProjectionIV
    @test result.tautological_h0

    # coefpath: h=0 should be exactly 1.0 for the shock term
    irf = coefpath(result; term = :x)
    @test irf[1] == 1.0
    @test all(isfinite, irf[2:end])

    # coefpath for a non-shock term: h=0 should be 0.0
    irf_w = coefpath(result; term = :w)
    @test irf_w[1] == 0.0

    # vcov: h=0 variance should be 0.0
    cov = vcov(HC1(), result)
    se = stderror(cov; term = :x)
    @test se[1] == 0.0
    @test all(isfinite, se[2:end])

    # summarize: should not error
    summary = summarize(result, HC1())
    @test summary.coef[1] == 1.0
    @test summary.se[1] == 0.0
    @test summary.lower[1] == 1.0
    @test summary.upper[1] == 1.0

    # weakivtest: should not error, h=0 returns sentinel with F_eff = Inf
    wiv = weakivtest(result)
    @test length(wiv) == horizon + 1
    @test wiv[1].F_eff == Inf
    @test all(r -> isfinite(r.F_eff), wiv[2:end])

    # Single-horizon weakivtest at h=0
    wiv0 = weakivtest(result, 0)
    @test wiv0.F_eff == Inf

    # Single-horizon weakivtest at h>0 should work normally
    wiv1 = weakivtest(result, 1)
    @test isfinite(wiv1.F_eff)

    # OLS LP where response == shock
    ols_result = lp(@formula(leads(x) ~ x + w), df; horizon = horizon)
    @test ols_result.tautological_h0
    irf_ols = coefpath(ols_result; term = :x)
    @test irf_ols[1] == 1.0

    # Non-tautological case: response != shock
    result_ok = lpiv(@formula(leads(y) ~ (x ~ z) + w), df; horizon = horizon)
    @test !result_ok.tautological_h0
end

@testitem "EWC bandwidth rule (LLSW 2018)" tags = [:ewc, :vcov, :api] begin
    using LocalProjections
    using DataFrames, StatsModels, StatsBase, Test

    # Integer rule: B = floor(0.41 * T0^(2/3))
    @test ewc_bandwidth(256) == 16   # baseline in the inference guide
    @test ewc_bandwidth(100) == 8
    @test ewc_bandwidth(1) == 1      # floor never below 1

    # LP method uses the horizon-zero effective sample size
    n = 150
    df = DataFrame(x = randn(n), y = randn(n))
    lp_result = lp(@formula(leads(y) ~ x + lags(y, 2)), df; horizon = 6)
    T0 = Int(nobs(lp_result.models[1]))
    @test T0 == n - 2  # two lags consumed
    @test ewc_bandwidth(lp_result) == ewc_bandwidth(T0)
end

@testitem "EWC inference uses Student-t_B critical values" tags = [
    :ewc, :vcov, :summarize, :api] begin
    using LocalProjections
    using DataFrames, StatsModels, Test
    using CovarianceMatrices: EWC, HC1
    using Distributions: Normal, TDist, quantile

    n = 200
    df = DataFrame(x = randn(n), y = randn(n))
    lp_result = lp(@formula(leads(y) ~ x + lags(y, 2)), df; horizon = 6)
    B = ewc_bandwidth(lp_result)

    beta = coefpath(lp_result; term = :x)
    cov_ewc = LocalProjections.vcov(EWC(B), lp_result)
    se_ewc = stderror(cov_ewc; term = :x)

    # summarize pairs EWC with t_B, not the normal
    s90 = summarize(lp_result, cov_ewc; term = :x, level = 0.90)
    tcrit = quantile(TDist(B), 0.95)
    zcrit = quantile(Normal(), 0.95)
    @test s90.lower ≈ beta .- tcrit .* se_ewc
    @test s90.upper ≈ beta .+ tcrit .* se_ewc
    @test !(s90.upper ≈ beta .+ zcrit .* se_ewc)  # t_B bands are wider

    # Non-EWC estimators keep normal critical values
    cov_hc1 = LocalProjections.vcov(HC1(), lp_result)
    se_hc1 = stderror(cov_hc1; term = :x)
    s90_hc1 = summarize(lp_result, cov_hc1; term = :x, level = 0.90)
    @test s90_hc1.upper ≈ beta .+ zcrit .* se_hc1

    # as_irf_result follows the same pairing
    irf = as_irf_result(lp_result; vcov_estimator = EWC(B), coverage = [0.90])
    @test vec(irf.upper[1].data) ≈ beta .+ tcrit .* se_ewc
    @test vec(irf.lower[1].data) ≈ beta .- tcrit .* se_ewc
end

@testitem "EWC covariance matches direct cosine-projection formula" tags = [
    :ewc, :vcov, :verification] begin
    using LocalProjections
    using DataFrames, StatsModels, LinearAlgebra, Test
    using CovarianceMatrices
    using CovarianceMatrices: EWC
    using StatsBase: modelmatrix, residuals

    # AR(1) errors so the long-run correction actually matters
    n = 250
    e = zeros(n)
    for t in 2:n
        e[t] = 0.5 * e[t - 1] + randn()
    end
    x = randn(n)
    y = 0.8 .* x .+ e
    df = DataFrame(x = x, y = y)
    lp_result = lp(@formula(leads(y) ~ x + lags(y, 1)), df; horizon = 3)
    B = ewc_bandwidth(lp_result)

    for m in lp_result.models
        X = modelmatrix(m)
        u = residuals(m)
        T, k = size(X)
        g = X .* u
        # Guide §2.2: Type-II DCT projections of the score
        Lam = [sqrt(2 / T) * sum(cos(π * j * (t - 0.5) / T) * g[t, :] for t in 1:T)
               for j in 1:B]
        Omega = sum(l * l' for l in Lam) / B
        # Sandwich with the T/(T-k) correlated-dof adjustment
        V_direct = (T / (T - k)) * T * ((X'X) \ Omega) / (X'X)
        V_pkg = CovarianceMatrices.vcov(EWC(B), m)
        @test V_pkg ≈ V_direct rtol = 1e-10
    end
end

@testitem "weakivtest warns on unsupported vcov estimators" tags = [
    :lpiv, :weakiv, :vcov, :ewc] begin
    using LocalProjections
    using DataFrames, StatsModels, Test, StableRNGs
    using CovarianceMatrices: EWC, Bartlett, NeweyWest, HR1

    # The warning gate itself, unit-tested deterministically
    @test_logs (:warn, r"HC0-style") LocalProjections._warn_unsupported_weakiv_estimator(EWC(10))
    @test_logs LocalProjections._warn_unsupported_weakiv_estimator(HR1())

    rng = StableRNG(20260831)
    n = 200
    z = randn(rng, n)
    u = randn(rng, n)
    x = 0.5 .* z .+ 0.5 .* u
    y = 2.0 .* x .+ u
    df = DataFrame(y = y, x = x, z = z)
    result = lpiv(@formula(leads(y) ~ (x ~ z)), df; horizon = 3)

    # Supported estimators: no logging (default HR1, kernel HAC)
    @test_logs weakivtest(result, 1)
    @test_logs weakivtest(result + vcov(Bartlett{NeweyWest}()), 1)

    # EWC is silently downgraded to HC0-style weighting upstream: warn.
    # The warning fires before the computation; the computation itself runs
    # Regress's numerically fragile Nagar routines, which throw on
    # near-singular intermediates in a LAPACK-version-dependent way (drafted
    # as an upstream issue) — so tolerate a downstream throw here.
    result_ewc = result + vcov(EWC(10))
    w_ewc = @test_logs (:warn, r"HC0-style") match_mode = :any begin
        try
            weakivtest(result_ewc, 1)
        catch
            nothing
        end
    end
    # Vector method warns once for all horizons
    @test_logs (:warn, r"HC0-style") match_mode = :any begin
        try
            weakivtest(result_ewc)
        catch
            nothing
        end
    end

    # Where the computation completes, the warning does not change the
    # returned statistics (documents the upstream fallback)
    if w_ewc !== nothing
        w_plain = weakivtest(result, 1)
        @test w_ewc.F_eff == w_plain.F_eff
    end
end

@testitem "Herbst–Johannsen correction matches direct lp_biascorr.m port" tags = [
    :biascorr, :verification] begin
    using LocalProjections
    using DataFrames, StatsModels, LinearAlgebra, Statistics, Test, StableRNGs
    using StatsBase: modelmatrix

    rng = StableRNG(20260830)
    n = 200
    x = randn(rng, n)
    y = zeros(n)
    for t in 2:n
        y[t] = 0.6 * y[t - 1] + 0.5 * x[t] + randn(rng)
    end
    df = DataFrame(y = y, x = x)
    lp_result = lp(@formula(leads(y) ~ x + lags(y, 2)), df; horizon = 8)
    bc = biascorrect(lp_result)

    # Direct port of lp_biascorr.m (guide §3) on the same inputs
    keep = [!(nm in ("(Intercept)", "x")) for nm in lp_result.coef_names]
    w = modelmatrix(lp_result.models[1])[:, keep]
    irs = coefpath(lp_result; term = :x)
    T = size(w, 1)
    H = lp_result.horizon
    wd = w .- mean(w; dims = 1)
    S0 = (wd' * wd) / (T - 1)
    acf = [1 + tr(S0 \ ((wd[1:(end - j), :]' * wd[(j + 1):end, :]) / (T - j - 1)))
           for j in 1:H]
    irs_c = copy(irs)
    for h in 1:H
        irs_c[h + 1] = irs[h + 1] +
                       (1 / (T - h)) * sum(acf[j] * irs_c[h - j + 1] for j in 1:h)
    end

    @test coefpath(bc) ≈ irs_c
    @test bc.c ≈ acf
    @test bc.T0 == T
    @test bc.controls == ["y_lag1", "y_lag2"]
    # Impact response is untouched; later horizons genuinely move
    @test coefpath(bc)[1] == irs[1]
    @test maximum(abs.(coefpath(bc)[2:end] .- irs[2:end])) > 0
end

@testitem "biascorrect conventions in summarize/as_irf_result" tags = [
    :biascorr, :summarize, :api] begin
    using LocalProjections
    using DataFrames, StatsModels, Test, StableRNGs
    using CovarianceMatrices: EWC, HC1
    using Distributions: TDist, quantile

    rng = StableRNG(20260830)
    n = 200
    x = randn(rng, n)
    y = zeros(n)
    for t in 2:n
        y[t] = 0.6 * y[t - 1] + 0.5 * x[t] + randn(rng)
    end
    df = DataFrame(y = y, x = x)
    lp_result = lp(@formula(leads(y) ~ x + lags(y, 2)), df; horizon = 8)
    bc = biascorrect(lp_result)

    # Wrapped-field forwarding
    @test bc.horizon == lp_result.horizon
    @test bc.shock === :x && bc.response === :y

    # Script-22 convention: bands centered on θ̂ᶜ with SEs of the
    # uncorrected OLS coefficients
    s_bc = summarize(bc, HC1(); level = 0.90)
    s_lp = summarize(lp_result, HC1(); level = 0.90)
    @test s_bc.coef ≈ coefpath(bc)
    @test s_bc.se == s_lp.se
    @test (s_bc.upper .- s_bc.coef) ≈ (s_lp.upper .- s_lp.coef)

    # Estimator/critical-value pairing flows through (t_B for EWC)
    B = ewc_bandwidth(bc)
    s_ewc = summarize(bc, EWC(B); level = 0.90)
    tcrit = quantile(TDist(B), 0.95)
    @test s_ewc.upper ≈ s_ewc.coef .+ tcrit .* s_ewc.se

    # bc + vcov keeps the correction, changes only the attached estimator
    bc2 = bc + LocalProjections.vcov(EWC(B))
    @test bc2.c == bc.c && coefpath(bc2) ≈ coefpath(bc)

    # as_irf_result centers on the corrected path and flags the metadata
    irf = as_irf_result(bc; vcov_estimator = HC1(), coverage = [0.90])
    @test vec(irf.data) ≈ coefpath(bc)
    @test irf.metadata.bias_corrected === true
end

@testitem "biascorrect guards and edge cases" tags = [:biascorr, :api] begin
    using LocalProjections
    using DataFrames, StatsModels, Test, StableRNGs

    rng = StableRNG(20260830)
    n = 200
    x = randn(rng, n)
    y = zeros(n)
    for t in 2:n
        y[t] = 0.6 * y[t - 1] + 0.5 * x[t] + randn(rng)
    end
    df = DataFrame(y = y, x = x)

    # Derived for OLS LP only
    iv_result = lpiv(@formula(leads(y) ~ (x ~ x) + lags(y, 1)), df; horizon = 2)
    @test_throws ArgumentError biascorrect(iv_result)

    # Correction is defined for the shock's impulse response only
    bc = biascorrect(lp(@formula(leads(y) ~ x + lags(y, 2)), df; horizon = 4))
    @test_throws ArgumentError coefpath(bc; term = :y_lag1)

    # Internal NaN gap → horizon-h sample is no longer the truncated
    # horizon-0 sample (guide §3.2: T − h must be checked, not copied)
    df_gap = copy(df)
    df_gap.y[100] = NaN
    lp_gap = lp(@formula(leads(y) ~ x + lags(y, 2)), df_gap; horizon = 4)
    @test_throws ArgumentError biascorrect(lp_gap)

    # Collinear controls → Σ̂₀ (near-)singular
    df_col = copy(df)
    df_col.z = 2 .* df.y
    lp_col = lp(@formula(leads(y) ~ x + lags(y, 1) + lags(z, 1)), df_col;
        horizon = 4)
    @test_throws ArgumentError biascorrect(lp_col)

    # Tautological h=0 (response == shock): θ̂₀ᶜ stays pinned at 1
    lp_taut = lp(@formula(leads(x) ~ x + lags(y, 2)), df; horizon = 4)
    @test lp_taut.tautological_h0
    @test coefpath(biascorrect(lp_taut))[1] == 1.0

    # No controls beyond intercept + shock: ĉ_j ≡ 1, recursion still runs
    lp_min = lp(@formula(leads(y) ~ x), df; horizon = 4)
    bc_min = biascorrect(lp_min)
    @test all(bc_min.c .== 1.0)
    @test coefpath(bc_min)[1] == coefpath(lp_min)[1]
end

@testitem "lagselect matches direct ic_var.m port" tags = [:lagselect, :verification] begin
    using LocalProjections
    using DataFrames, LinearAlgebra, StableRNGs, Test

    rng = StableRNG(20260831)
    T = 300
    sh = zeros(T)
    y = zeros(T)
    for t in 3:T
        sh[t] = 0.5 * sh[t - 1] + randn(rng)
        y[t] = 0.7 * y[t - 1] - 0.35 * y[t - 2] + 0.8 * sh[t] + 0.4 * sh[t - 1] +
               randn(rng)
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
    @test sel.aic ≈ aic_ref rtol=1e-12
    @test sel.bic ≈ bic_ref rtol=1e-12
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
end

@testitem "_var_ols matches var_estim.m conventions and MacroEconometricTools" tags = [
    :lagselect, :verification] begin
    using LocalProjections
    using DataFrames, LinearAlgebra, StableRNGs, Test
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
    @test v.Σu ≈ (v.U' * v.U) ./ ((T - p) - k) rtol=1e-12

    # Cross-check the hand-rolled VAR against the hub package.
    m = MET.VAR(Y, p)
    @test v.A ≈ m.coefficients.lags rtol=1e-10
    @test v.c ≈ m.coefficients.intercept rtol=1e-10
    @test v.Σu ≈ Matrix(m.Σ) rtol=1e-10

    # Residuals are orthogonal to the regressors.
    @test maximum(abs, sum(v.U; dims = 1)) < 1e-9
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

    @test_throws ArgumentError lagselect(df, [:shock, :y]; criterion = :hq)
    @test_throws ArgumentError lagselect(df, [:shock, :y]; maxlags = 0)
    @test_throws ArgumentError lagselect(df, [:shock, :y]; maxlags = T)
    @test_throws ArgumentError lagselect(df, [:shock, :nope])
    @test_throws ArgumentError lagselect(df, [:shock, :shock])
    @test_throws ArgumentError lagselect(df, Symbol[])

    dfm = copy(df)
    dfm.y = Vector{Union{Missing, Float64}}(dfm.y)
    dfm.y[10] = missing
    @test_throws ArgumentError lagselect(dfm, [:shock, :y])

    dfn = copy(df)
    dfn.y[10] = NaN
    @test_throws ArgumentError lagselect(dfn, [:shock, :y])
end

@testitem "Pope correction matches var_biascorr.m and its closed form" tags = [
    :bootstrap, :verification] begin
    using LocalProjections
    using LinearAlgebra, Test

    # Pope (1990) for a univariate AR(1): the scaled negative bias b equals
    # exactly 1 + 3*rho. This is an external check, not a re-port.
    for ρ in (0.0, 0.3, 0.5, 0.7, 0.9, 0.95)
        T = 1000.0
        Ac, δ = LocalProjections._pope_biascorrect(fill(ρ, 1, 1), fill(2.7, 1, 1), T)
        @test δ == 1.0
        @test (Ac[1, 1] - ρ) * T ≈ 1 + 3ρ rtol=1e-10
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
    aux = inv(I - At) + At * inv(I - At * At) +
          sum(λ * inv(I - λ * At) for λ in eigvals(Acomp))
    b = real.(G * (aux / Γ0))
    ref = (Acomp + b ./ Tn)[1:n, :]

    got, δ2 = LocalProjections._pope_biascorrect(A, Σ, Tn)
    @test δ2 == 1.0
    @test got ≈ ref rtol=1e-10

    # The 3-D method is the same correction, reshaped.
    A3 = Array{Float64}(undef, 2, 2, 2)
    A3[:, :, 1] = A[:, 1:2]
    A3[:, :, 2] = A[:, 3:4]
    got3, δ3 = LocalProjections._pope_biascorrect(A3, Σ, Tn)
    @test δ3 == δ2
    @test got3[:, :, 1] ≈ ref[:, 1:2] rtol=1e-12
    @test got3[:, :, 2] ≈ ref[:, 3:4] rtol=1e-12
end

@testitem "moving-block resampling matches var_boot.m" tags = [:bootstrap, :verification] begin
    using LocalProjections
    using LinearAlgebra, StableRNGs, Statistics, Test

    rng = StableRNG(202)
    Tu, n = 97, 3
    U = randn(rng, Tu, n)
    for t in 2:Tu
        U[t, :] .+= 0.4 .* U[t - 1, :]
    end
    ℓ = 12

    # var_boot.m computes the position means with
    #   filter(ones(1, Tu-l+1)/(Tu-l+1), 1, res), rows end-l+1:end
    # which is the SLIDING-WINDOW mean over the Tu-l+1 possible block starts.
    means = LocalProjections._position_means(U, ℓ)
    @test size(means) == (ℓ, n)
    for s in 1:ℓ, j in 1:n

        @test means[s, j] ≈ mean(U[s:(Tu - ℓ + s), j]) rtol=1e-12
    end
    # Each position averages Tu - l + 1 terms, not Tu/l of them.
    @test length(1:(Tu - ℓ + 1)) == Tu - ℓ + 1

    # It is NOT the stride-l mean used by MacroEconometricTools.bootstrap_irf_block.
    stride_means = [mean(U[s:ℓ:(Tu - ℓ + s), j]) for s in 1:ℓ, j in 1:n]
    @test !isapprox(means, stride_means; rtol = 1e-3)

    # Block layout with fixed starts is deterministic and matches the reference.
    starts = [0, 5, 40, 60, 17, 33, 80, 2, 51]   # cld(97, 12) == 9 blocks
    @test length(starts) == cld(Tu, ℓ)
    dest = Matrix{Float64}(undef, Tu, n)
    LocalProjections._mbb_residuals!(dest, U, means, starts)
    for (b, off) in enumerate(starts), s in 1:ℓ

        row = (b - 1) * ℓ + s
        row > Tu && continue
        for j in 1:n
            @test dest[row, j] ≈ U[off + s, j] - means[s, j] rtol=1e-12
        end
    end

    # Recentering makes the resampled residuals mean-zero in expectation: averaging
    # over every admissible start at a fixed position gives exactly zero.
    for s in 1:ℓ, j in 1:n

        @test abs(mean(U[(0:(Tu - ℓ)) .+ s, j] .- means[s, j])) < 1e-10
    end
end

@testitem "VAR simulation, impact vector and IRF recursion" tags = [
    :bootstrap, :verification] begin
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
        expected = c .+ A[:, :, 1] * Y[t - 1, :] .+ A[:, :, 2] * Y[t - 2, :] .+
                   Ustar[t - 2, :]
        @test Y[t, :] ≈ expected rtol=1e-12
    end

    # _var_impact with innov_index = 1 is Sigma[:,1]/Sigma[1,1] = L[:,1]/L11.
    rngu = StableRNG(6)
    U = randn(rngu, 500, 3) * [1.0 0.0 0.0; 0.4 1.0 0.0; -0.3 0.2 1.0]'
    ν = LocalProjections._var_impact(U, 1)
    S = (U' * U) ./ size(U, 1)
    @test ν ≈ S[:, 1] ./ S[1, 1] rtol=1e-10
    L = cholesky(Symmetric(S)).L
    @test ν ≈ L[:, 1] ./ L[1, 1] rtol=1e-10
    @test ν[1] ≈ 1.0 rtol=1e-12   # unit-impact normalization

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
        @test irf[:, h + 1] ≈ Θ[h + 1] * [1.0, 0.0] rtol=1e-12
    end
end

@testitem "Hall percentile-t intervals match boot_ci.m" tags = [:bootstrap, :verification] begin
    using LocalProjections
    using DataFrames, Random, StableRNGs, Statistics, StatsModels, Test

    rng = StableRNG(20260901)
    T = 220
    sh = zeros(T)
    y = zeros(T)
    for t in 2:T
        sh[t] = 0.3 * sh[t - 1] + randn(rng)
        y[t] = 0.6 * y[t - 1] + 0.8 * sh[t] + randn(rng)
    end
    df = DataFrame(shock = sh, y = y)
    m = lp(@formula(leads(y) ~ shock + lags(y, 2) + lags(shock, 2)), df; horizon = 6)
    b = varbootstrap(m, df; vars = [:shock, :y], nlags = 2, nboot = 300,
        rng = StableRNG(77))

    level = 0.90
    α = 1 - level

    # Direct port of _estim/boot_ci.m for all three constructions.
    for h in 1:(b.horizon + 1)
        θs = [x for x in b.theta_boot[:, h] if isfinite(x)]
        ts = [(b.theta_boot[i, h] - b.pseudo_truth[h]) / b.se_boot[i, h]
              for i in 1:b.nboot
              if isfinite(b.theta_boot[i, h]) && isfinite(b.se_boot[i, h]) &&
                     b.se_boot[i, h] > 0]

        ql, qu = quantile(θs, α / 2), quantile(θs, 1 - α / 2)
        tl, tu = quantile(ts, α / 2), quantile(ts, 1 - α / 2)

        efron = (ql, qu)
        hall = (b.theta[h] + b.pseudo_truth[h] - qu,
            b.theta[h] + b.pseudo_truth[h] - ql)
        # Note the quantile reversal on the studentized statistic.
        hall_t = (b.theta[h] - b.se[h] * tu, b.theta[h] - b.se[h] * tl)

        se_ = summarize(b; level = level, method = :efron)
        sh_ = summarize(b; level = level, method = :hall)
        st_ = summarize(b; level = level, method = :hall_t)

        @test (se_.lower[h], se_.upper[h]) == efron
        @test (sh_.lower[h], sh_.upper[h]) == hall
        @test (st_.lower[h], st_.upper[h]) == hall_t
    end

    # The three constructions genuinely differ.
    st = summarize(b; level = level, method = :hall_t)
    sh_ = summarize(b; level = level, method = :hall)
    se_ = summarize(b; level = level, method = :efron)
    @test st.lower != sh_.lower
    @test st.lower != se_.lower
    @test sh_.lower != se_.lower

    # Percentile-t bands are asymmetric around the point estimate.
    @test !isapprox(st.upper .- st.coef, st.coef .- st.lower; rtol = 1e-6)
end

@testitem "varbootstrap end-to-end conventions" tags = [:bootstrap, :api] begin
    using LocalProjections
    using DataFrames, LinearAlgebra, Random, StableRNGs, Statistics, StatsModels, Test
    using CovarianceMatrices

    rng = StableRNG(20260902)
    T = 240
    sh = zeros(T)
    y = zeros(T)
    for t in 2:T
        sh[t] = 0.3 * sh[t - 1] + randn(rng)
        y[t] = 0.6 * y[t - 1] + 0.8 * sh[t] + randn(rng)
    end
    df = DataFrame(shock = sh, y = y)
    m = lp(@formula(leads(y) ~ shock + lags(y, 4) + lags(shock, 4)), df; horizon = 10)

    b = varbootstrap(m, df; vars = [:shock, :y], nlags = 4, nboot = 200,
        rng = StableRNG(1))
    @test b isa LPBootstrap
    @test b.nboot == 200
    @test b.nfail == 0
    @test b.biascorrected && b.popecorrected
    @test b.blocklength == ceil(Int, 5.03 * T^(1 / 4))
    @test size(b.theta_boot) == (200, 11)
    @test size(b.se_boot) == (200, 11)

    # Reported path is the bias-corrected one; SEs are those of the UNCORRECTED
    # coefficients (the reference never recomputes them).
    @test b.theta ≈ coefpath(biascorrect(m); term = :shock) rtol=1e-12
    @test b.se ≈ stderror(vcov(HC1(), m); term = :shock) rtol=1e-12
    @test !isapprox(b.theta, coefpath(m; term = :shock); rtol = 1e-8)

    # Property forwarding to the inner LP.
    @test b.horizon == 10
    @test b.shock == :shock
    @test b.response == :y
    @test coefpath(b) == b.theta

    # Reproducibility: identical output threaded and unthreaded.
    b_serial = varbootstrap(m, df; vars = [:shock, :y], nlags = 4, nboot = 100,
        rng = StableRNG(42), threaded = false)
    b_thread = varbootstrap(m, df; vars = [:shock, :y], nlags = 4, nboot = 100,
        rng = StableRNG(42), threaded = true)
    @test b_serial.theta_boot == b_thread.theta_boot
    @test b_serial.se_boot == b_thread.se_boot

    # Script-23 variant: no Pope, no Herbst-Johannsen.
    b23 = varbootstrap(m, df; vars = [:shock, :y], nlags = 4, nboot = 100,
        rng = StableRNG(1), biascorrect = false, popecorrect = false)
    @test !b23.biascorrected && !b23.popecorrected
    @test isnan(b23.pope_delta)
    @test b23.theta ≈ coefpath(m; term = :shock) rtol=1e-12
    @test !isapprox(b23.theta, b.theta; rtol = 1e-8)

    # summarize returns a usable IRFSummary with pointwise bands.
    s = summarize(b; level = 0.90)
    @test s isa IRFSummary
    @test s.level == 0.90
    @test length(s.lower) == 11
    # Ordering is guaranteed (q_{1-a/2} >= q_{a/2}); containment of the point
    # estimate is NOT: a Hall percentile-t interval corrects for bias, so it can
    # sit entirely to one side of theta-hat when the bootstrap t-distribution is
    # shifted. Do not assert containment here.
    @test all(s.lower .<= s.upper)
    @test DataFrame(s) isa DataFrame

    # Wider level gives wider bands.
    s68 = summarize(b; level = 0.68)
    @test all(s68.upper .- s68.lower .<= s.upper .- s.lower .+ 1e-12)

    # Pseudo-truth is the VAR-implied response, not the LP path.
    @test !isapprox(b.pseudo_truth, b.theta; rtol = 1e-6)
    @test length(b.pseudo_truth) == 11

    io = IOBuffer()
    show(io, b)
    @test occursin("LPBootstrap", String(take!(io)))
    show(io, MIME"text/plain"(), b)
    out = String(take!(io))
    @test occursin("moving-block", out)
    @test occursin("Herbst", out)
    @test occursin("Pope", out)
end

@testitem "varbootstrap plotting and as_irf_result" tags = [:bootstrap, :api] begin
    using LocalProjections
    using DataFrames, Plots, StableRNGs, StatsModels, Test

    rng = StableRNG(20260903)
    T = 200
    sh = randn(rng, T)
    y = zeros(T)
    for t in 2:T
        y[t] = 0.5 * y[t - 1] + 0.9 * sh[t] + randn(rng)
    end
    df = DataFrame(shock = sh, y = y)
    m = lp(@formula(leads(y) ~ shock + lags(y, 2)), df; horizon = 8)
    b = varbootstrap(m, df; vars = [:shock, :y], nlags = 2, nboot = 150,
        rng = StableRNG(3))

    # Plot recipe: single and multiple levels, and each interval method.
    @test plot(b) isa Plots.Plot
    @test plot(b; levels = [0.68, 0.90]) isa Plots.Plot
    @test plot(b; levels = [0.90], method = :efron) isa Plots.Plot
    @test plot(b; levels = [0.90], method = :hall) isa Plots.Plot
    @test_throws ArgumentError plot(b; levels = [1.5])

    r = as_irf_result(b; coverage = [0.68, 0.90])
    @test r isa LocalProjectionIRFResult
    @test r.coverage == [0.68, 0.90]
    @test r.metadata.bootstrap == true
    @test r.metadata.bootstrap_method == :hall_t
    @test r.metadata.nboot == 150
    @test r.metadata.bias_corrected == true
    @test r.metadata.pope_corrected == true
    @test r.metadata.var_lags == 2
    @test vec(r.data[1, 1, :]) ≈ b.theta rtol=1e-12

    # Bands carried over are the bootstrap ones, and asymmetric.
    s90 = summarize(b; level = 0.90)
    @test vec(r.lower[2][1, 1, :]) ≈ s90.lower rtol=1e-12
    @test vec(r.upper[2][1, 1, :]) ≈ s90.upper rtol=1e-12

    @test_throws ArgumentError as_irf_result(b; term = :y)
end

@testitem "varbootstrap guards and edge cases" tags = [:bootstrap, :api] begin
    using LocalProjections
    using DataFrames, StableRNGs, StatsModels, Test

    rng = StableRNG(20260904)
    T = 180
    sh = randn(rng, T)
    y = zeros(T)
    z = randn(rng, T)
    for t in 2:T
        y[t] = 0.5 * y[t - 1] + 0.9 * sh[t] + randn(rng)
    end
    df = DataFrame(shock = sh, y = y, z = z)
    m = lp(@formula(leads(y) ~ shock + lags(y, 2)), df; horizon = 5)

    # The shock must be in the VAR data vector.
    @test_throws ArgumentError varbootstrap(m, df; vars = [:y], nlags = 2, nboot = 5)
    # Every variable the LP formula needs must be simulated.
    @test_throws ArgumentError varbootstrap(m, df; vars = [:shock], nlags = 2, nboot = 5)
    # Block length must fit the residual sample.
    @test_throws ArgumentError varbootstrap(m, df; vars = [:shock, :y], nlags = 2,
        nboot = 5, blocklength = 10_000)
    @test_throws ArgumentError varbootstrap(m, df; vars = [:shock, :y], nlags = 2,
        nboot = 0)
    # Interval method must be recognized.
    # The low-draw warning is part of the contract (guide 5 checklist).
    b = @test_logs (:warn, r"usable draws") match_mode=:any varbootstrap(
        m, df; vars = [:shock, :y], nlags = 2, nboot = 50, rng = StableRNG(8))
    @test_throws ArgumentError summarize(b; method = :bogus)
    @test_throws ArgumentError summarize(b; level = 1.5)
    @test_throws ArgumentError coefpath(b; term = :y)

    # missing / NaN in the VAR columns are rejected (guide LP-1).
    dfm = copy(df)
    dfm.y = Vector{Union{Missing, Float64}}(dfm.y)
    dfm.y[20] = missing
    @test_throws ArgumentError varbootstrap(m, dfm; vars = [:shock, :y], nlags = 2,
        nboot = 5)
    dfn = copy(df)
    dfn.y[20] = NaN
    @test_throws ArgumentError varbootstrap(m, dfn; vars = [:shock, :y], nlags = 2,
        nboot = 5)

    # Anchored responses are not supported.
    ma = lp(@formula(leads(y) | shock ~ shock + lags(y, 2)), df; horizon = 4)
    @test_throws ArgumentError varbootstrap(ma, df; vars = [:shock, :y], nlags = 2,
        nboot = 5)

    # IV local projections are rejected, as for biascorrect.
    x = 0.7 .* z .+ randn(StableRNG(9), T)
    dfiv = copy(df)
    dfiv.x = x
    miv = lpiv(@formula(leads(y) ~ (x ~ z) + lags(y, 2)), dfiv; horizon = 4)
    @test_throws ArgumentError varbootstrap(miv, dfiv; vars = [:x, :y], nlags = 2,
        nboot = 5)

    # cumul responses are supported and the pseudo-truth is cumulated.
    mc = lp(@formula(cumul(y) ~ shock + lags(y, 2)), df; horizon = 6)
    bc = varbootstrap(mc, df; vars = [:shock, :y], nlags = 2, nboot = 120,
        rng = StableRNG(10))
    bl = varbootstrap(lp(@formula(leads(y) ~ shock + lags(y, 2)), df; horizon = 6),
        df; vars = [:shock, :y], nlags = 2, nboot = 120, rng = StableRNG(10))
    @test bc.pseudo_truth ≈ cumsum(bl.pseudo_truth) rtol=1e-12
    @test all(isfinite, summarize(bc).lower)

    # Tautological h = 0 (response === shock) has a degenerate band, not a NaN.
    mt = lp(@formula(leads(shock) ~ shock + lags(y, 2)), df; horizon = 5)
    bt = varbootstrap(mt, df; vars = [:shock, :y], nlags = 2, nboot = 120,
        rng = StableRNG(12))
    st = summarize(bt; level = 0.90)
    @test st.coef[1] == 1.0
    @test st.se[1] == 0.0
    @test st.lower[1] == 1.0 && st.upper[1] == 1.0
    @test all(isfinite, st.lower) && all(isfinite, st.upper)
end
