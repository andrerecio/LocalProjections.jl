using TestItems

@testitem "cumul transformation" tags=[:cumul, :core] begin
    using LocalProjections
    using DataFrames, StatsModels, Regress, StatsBase, Test

    # Create simple synthetic data
    n = 100
    df = DataFrame(
        x = 1.0:n,
        y = sin.(1.0:n) .+ (1.0:n) ./ 10
    )

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
        y_h_manual = map(x->sum(x), map(t->df_filtered.y[t:(t + h)], 1:(nrow(df_filtered) - h)))

        # Find complete observations (no NaN in y_h)
        complete_obs = .!isnan.(y_h)
        y_manual = y_h[complete_obs]
        # Manually run regression on complete data
        X_manual = hcat(ones(sum(complete_obs)), df_filtered.x[complete_obs])

        @test y_manual == y_h_manual  # Verify manual cumulative matches StatsModels output
        manual_coef = (X_manual \ y_manual)[2]  # x coefficient

        # Compare coefficients
        @test lp_coef ≈ manual_coef atol=1e-10
    end
end

@testitem "leads transformation" tags=[:leads, :core] begin
    using LocalProjections
    using DataFrames, StatsModels, Regress, StatsBase, Test
    using ShiftedArrays: lead

    # Create simple synthetic data
    n = 100
    df = DataFrame(
        x = 1.0:n,
        y = cos.(1.0:n) .+ (1.0:n) ./ 20
    )

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
        @test lp_coef ≈ manual_coef atol=1e-10
    end

    # Verify NaN handling (not missing)
    y_lead_test = lead(df.y, 5, default = NaN)
    @test eltype(y_lead_test) == Float64
    @test any(isnan, y_lead_test)  # Should have NaN at boundaries
end

@testitem "anchor function syntax" tags=[:anchor, :core] begin
    using LocalProjections
    using DataFrames, StatsModels, Regress, StatsBase, Test

    # Create simple synthetic data with both y and z
    n = 100
    df = DataFrame(
        x = 1.0:n,
        y = sin.(1.0:n) .+ (1.0:n) ./ 10,
        z = cos.(1.0:n)
    )

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
        anchor_h = AnchorTerm{typeof(inner_leads), typeof(Term(:z))}(
            inner_leads, Term(:z), 0)  # horizon=0 because lead is in inner term
        y_h = StatsModels.modelcols(anchor_h, df_filtered)

        # Find complete observations (no NaN)
        complete_obs = .!isnan.(y_h)

        # Manually run regression on complete data
        X_manual = hcat(ones(sum(complete_obs)), df_filtered.x[complete_obs])
        y_manual = y_h[complete_obs]
        manual_coef = (X_manual \ y_manual)[2]  # x coefficient

        # Compare coefficients
        @test lp_coef ≈ manual_coef atol=1e-10
    end
end

@testitem "anchor pipe syntax" tags=[:anchor, :core] begin
    using LocalProjections
    using DataFrames, StatsModels, Regress, StatsBase, Test

    # Create simple synthetic data
    n = 100
    df = DataFrame(
        x = 1.0:n,
        y = sin.(1.0:n) .+ (1.0:n) ./ 10,
        z = cos.(1.0:n)
    )

    # Test anchor with pipe syntax
    horizon = 3
    lp_pipe = lp(@formula(leads(y)|z ~ x), df; horizon = horizon)

    # Test anchor with function syntax (should be identical)
    lp_func = lp(@formula(anchor(y, z) ~ x), df; horizon = horizon)

    # Compare results from both syntaxes
    for h in 0:horizon
        pipe_coef = coef(lp_pipe.models[h + 1])[2]
        func_coef = coef(lp_func.models[h + 1])[2]

        @test pipe_coef ≈ func_coef atol=1e-10
    end

    # Also verify against manual computation
    df_filtered = dropmissing(df, [:y, :z, :x], disallowmissing = true)

    for h in 0:horizon
        lp_coef = coef(lp_pipe.models[h + 1])[2]

        # Manual computation using StatsModels
        inner_leads = LeadTerm{typeof(Term(:y))}(Term(:y), h)
        anchor_h = AnchorTerm{typeof(inner_leads), typeof(Term(:z))}(
            inner_leads, Term(:z), 0)  # horizon=0 because lead is in inner term
        y_h = StatsModels.modelcols(anchor_h, df_filtered)

        y_h_manual = lead(df_filtered.y, h, default = NaN) .- df_filtered.z
        complete_obs = .!isnan.(y_h)
        X_manual = hcat(ones(sum(complete_obs)), df_filtered.x[complete_obs])
        y_manual = y_h[complete_obs]
        y_manual_manual = y_h_manual[complete_obs]
        @test y_manual == y_manual_manual  # Verify manual matches StatsModels output
        manual_coef = (X_manual \ y_manual)[2]

        @test lp_coef ≈ manual_coef atol=1e-10
    end
end

@testitem "modelcols anchor matches manual lead computation" tags=[:anchor, :verification] begin
    using LocalProjections
    using DataFrames, StatsModels, Test

    # Create test data
    n = 100
    df = DataFrame(
        y = sin.(1.0:n) .+ (1.0:n) ./ 10,
        z = cos.(1.0:n)
    )

    # Test that modelcols(AnchorTerm) matches manual lead() - z computation
    for h in 0:5
        # Method 1: Using StatsModels.modelcols with AnchorTerm
        inner_leads = LeadTerm{typeof(Term(:y))}(Term(:y), h)
        anchor_h = AnchorTerm{typeof(inner_leads), typeof(Term(:z))}(
            inner_leads, Term(:z), 0)
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
                @test y_modelcols[i] ≈ y_manual[i] atol=1e-10
            end
        end
    end
end

@testitem "cumulative anchor (nested)" tags=[:nested, :anchor, :cumul] begin
    using LocalProjections
    using DataFrames, StatsModels, Regress, StatsBase, Test

    # Create simple synthetic data
    n = 100
    df = DataFrame(
        x = 1.0:n,
        y = sin.(1.0:n) .+ (1.0:n) ./ 10,
        z = cos.(1.0:n)
    )

    # Test cumulative anchor: cumul(y)|z
    horizon = 3
    lp_result = lp(@formula(cumul(y)|z ~ x), df; horizon = horizon)

    # Manually compute cumulative anchored response
    df_filtered = dropmissing(df, [:y, :z, :x], disallowmissing = true)

    for h in 0:horizon
        # Get coefficient from lp result
        lp_coef = coef(lp_result.models[h + 1])[2]

        # Manual computation using StatsModels: first cumul, then anchor
        inner_cumul = CumulTerm{typeof(Term(:y))}(Term(:y), h)
        anchor_h = AnchorTerm{typeof(inner_cumul), typeof(Term(:z))}(
            inner_cumul, Term(:z), 0)  # horizon=0 because cumul already has horizon
        y_h = StatsModels.modelcols(anchor_h, df_filtered)

        # Find complete observations (no NaN)
        complete_obs = .!isnan.(y_h)

        # Manually run regression
        X_manual = hcat(ones(sum(complete_obs)), df_filtered.x[complete_obs])
        y_manual = y_h[complete_obs]
        manual_coef = (X_manual \ y_manual)[2]

        # Compare coefficients
        @test lp_coef ≈ manual_coef atol=1e-10
    end
end

@testitem "nested log transformations" tags=[:nested, :core] begin
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

        @test lp_coef ≈ manual_coef atol=1e-10
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

        @test lp_coef ≈ manual_coef atol=1e-10
    end

    # Test anchor(log(y), z)
    lp_anchor_log = lp(@formula(anchor(log(y), z) ~ x), df; horizon = horizon)
    df_filtered3 = dropmissing(df, [:y, :z, :x], disallowmissing = true)

    for h in 0:horizon
        lp_coef = coef(lp_anchor_log.models[h + 1])[2]

        # Manual computation: lead of log(y) by h, minus z at t
        log_y = log.(df_filtered3.y)
        y_h = [t + h <= length(log_y) ? log_y[t + h] - df_filtered3.z[t] : NaN
               for t in 1:length(log_y)]
        complete_obs = .!isnan.(y_h)

        X_manual = hcat(ones(sum(complete_obs)), df_filtered3.x[complete_obs])
        y_manual = y_h[complete_obs]
        manual_coef = (X_manual \ y_manual)[2]

        # Use relative tolerance for large values
        @test lp_coef ≈ manual_coef rtol=1e-10
    end
end

@testitem "summarize function" tags=[:summarize, :api] begin
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
    @test summary_direct.coef ≈ summary_obj.coef atol=1e-10

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
    @test all(summary_90.upper .- summary_90.lower .<
              summary_obj.upper .- summary_obj.lower)
end

@testitem "plus vcov operator" tags=[:vcov, :api] begin
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

@testitem "lpiv basic functionality" tags=[:lpiv, :iv, :core] begin
    using LocalProjections
    using LocalProjections: FirstStageResult
    using DataFrames, StatsModels, Test
    using Random
    using CovarianceMatrices: HC1
    using LinearAlgebra

    Random.seed!(42)

    # Generate data with endogeneity
    n = 200
    z = randn(n)
    u = randn(n)
    x = 0.5*z + 0.5*u      # Endogenous (correlated with u)
    y = 1.0 .+ 2.0*x .+ u  # True effect = 2.0

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
    @test fs isa FirstStageResult
    @test fs.F_joint > 0  # F-statistic should be positive

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

@testitem "lpiv with controls" tags=[:lpiv, :iv] begin
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
    x = 0.5*z + 0.3*w + 0.4*u  # Endogenous
    y = 1.0 .+ 2.0*x .+ 0.5*w .+ u

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

@testitem "lpiv plus vcov operator" tags=[:lpiv, :vcov, :api] begin
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
    x = 0.6*z + 0.4*u
    y = 0.5 .+ 1.5*x .+ u

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

@testitem "lpiv with cumul response" tags=[:lpiv, :cumul] begin
    using LocalProjections
    using DataFrames, StatsModels, Test
    using Random
    using LinearAlgebra

    Random.seed!(789)

    n = 150
    z = randn(n)
    u = randn(n)
    x = 0.5*z + 0.5*u
    y = 1.0 .+ 2.0*x .+ u

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

@testitem "lpiv summarize" tags=[:lpiv, :summarize, :api] begin
    using LocalProjections
    using DataFrames, StatsModels, Test
    using CovarianceMatrices: HC1
    using Random
    using LinearAlgebra

    Random.seed!(1011)

    n = 150
    z = randn(n)
    u = randn(n)
    x = 0.5*z + 0.5*u
    y = 1.0 .+ 2.0*x .+ u

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
    @test summary_direct.coef ≈ summary_obj.coef atol=1e-10
end

@testitem "lpiv comparison with manual 2SLS" tags=[:lpiv, :iv, :verification] begin
    using LocalProjections
    using DataFrames, StatsModels, Test
    using LinearAlgebra
    using Random

    Random.seed!(2022)

    # Generate data
    n = 200
    z = randn(n)
    u = randn(n)
    x = 0.5*z + 0.5*u
    y = 1.0 .+ 2.0*x .+ u

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
    @test lpiv_coef ≈ beta_2sls[2] atol=1e-8
end

@testitem "weakivtest for lpiv" tags=[:lpiv, :weakiv, :iv] begin
    using LocalProjections
    using DataFrames, StableRNGs, StatsModels

    rng = StableRNG(42)
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
