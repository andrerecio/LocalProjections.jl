using TestItems

@testitem "cumul transformation" tags = [:cumul, :core] begin
    using LocalProjections
    using LocalProjections: CumulTerm
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
        lp_coef = coef(horizonmodel(lp_result, h))[2]  # x coefficient (index 2, after intercept)

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
        @test lp_coef ≈ manual_coef atol = 1.0e-10
    end
end

@testitem "leads transformation" tags = [:leads, :core] begin
    using LocalProjections
    using LocalProjections: LeadTerm
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
        lp_coef = coef(horizonmodel(lp_result, h))[2]  # x coefficient

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
        @test lp_coef ≈ manual_coef atol = 1.0e-10
    end

    # Verify NaN handling (not missing)
    y_lead_test = lead(df.y, 5, default = NaN)
    @test eltype(y_lead_test) == Float64
    @test any(isnan, y_lead_test)  # Should have NaN at boundaries
end

@testitem "anchor function syntax" tags = [:anchor, :core] begin
    using LocalProjections
    using LocalProjections: LeadTerm, AnchorTerm
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
        lp_coef = coef(horizonmodel(lp_result, h))[2]  # x coefficient

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
        @test lp_coef ≈ manual_coef atol = 1.0e-10
    end
end

@testitem "anchor pipe syntax" tags = [:anchor, :core] begin
    using LocalProjections
    using LocalProjections: LeadTerm, AnchorTerm
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
        pipe_coef = coef(horizonmodel(lp_pipe, h))[2]
        func_coef = coef(horizonmodel(lp_func, h))[2]

        @test pipe_coef ≈ func_coef atol = 1.0e-10
    end

    # Also verify against manual computation
    df_filtered = dropmissing(df, [:y, :z, :x], disallowmissing = true)

    for h in 0:horizon
        lp_coef = coef(horizonmodel(lp_pipe, h))[2]

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

        @test lp_coef ≈ manual_coef atol = 1.0e-10
    end
end

@testitem "modelcols anchor matches manual lead computation" tags = [:anchor, :verification] begin
    using LocalProjections
    using LocalProjections: LeadTerm, AnchorTerm
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
                @test y_modelcols[i] ≈ y_manual[i] atol = 1.0e-10
            end
        end
    end
end

@testitem "cumulative anchor (nested)" tags = [:nested, :anchor, :cumul] begin
    using LocalProjections
    using LocalProjections: CumulTerm, AnchorTerm
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
        lp_coef = coef(horizonmodel(lp_result, h))[2]

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
        @test lp_coef ≈ manual_coef atol = 1.0e-10
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
        lp_coef = coef(horizonmodel(lp_cumul_log, h))[2]

        # Manual computation: cumulative sum of log(y) from t to t+h
        log_y = log.(df_filtered1.y)
        y_h = [
            t + h <= length(log_y) ? sum(log_y[t:(t + h)]) : NaN
                for t in 1:length(log_y)
        ]
        complete_obs = .!isnan.(y_h)

        X_manual = hcat(ones(sum(complete_obs)), df_filtered1.x[complete_obs])
        y_manual = y_h[complete_obs]
        manual_coef = (X_manual \ y_manual)[2]

        @test lp_coef ≈ manual_coef atol = 1.0e-10
    end

    # Test leads(log(y))
    lp_lead_log = lp(@formula(leads(log(y)) ~ x), df; horizon = horizon)
    df_filtered2 = dropmissing(df, [:y, :x], disallowmissing = true)

    for h in 0:horizon
        lp_coef = coef(horizonmodel(lp_lead_log, h))[2]

        # Manual computation: lead of log(y) by h
        log_y = log.(df_filtered2.y)
        y_h = [t + h <= length(log_y) ? log_y[t + h] : NaN for t in 1:length(log_y)]
        complete_obs = .!isnan.(y_h)

        X_manual = hcat(ones(sum(complete_obs)), df_filtered2.x[complete_obs])
        y_manual = y_h[complete_obs]
        manual_coef = (X_manual \ y_manual)[2]

        @test lp_coef ≈ manual_coef atol = 1.0e-10
    end

    # Test anchor(log(y), z)
    lp_anchor_log = lp(@formula(anchor(log(y), z) ~ x), df; horizon = horizon)
    df_filtered3 = dropmissing(df, [:y, :z, :x], disallowmissing = true)

    for h in 0:horizon
        lp_coef = coef(horizonmodel(lp_anchor_log, h))[2]

        # Manual computation: lead of log(y) by h, minus z at t
        log_y = log.(df_filtered3.y)
        y_h = [
            t + h <= length(log_y) ? log_y[t + h] - df_filtered3.z[t] : NaN
                for
                t in 1:length(log_y)
        ]
        complete_obs = .!isnan.(y_h)

        X_manual = hcat(ones(sum(complete_obs)), df_filtered3.x[complete_obs])
        y_manual = y_h[complete_obs]
        manual_coef = (X_manual \ y_manual)[2]

        # Use relative tolerance for large values
        @test lp_coef ≈ manual_coef rtol = 1.0e-10
    end
end

@testitem "summarize function" tags = [:summarize, :api] begin
    using LocalProjections
    using DataFrames, StatsModels, Test
    using CovarianceMatrices: HC1

    n = 100
    df = DataFrame(x = randn(n), y = randn(n))
    lp_result = lp(@formula(leads(y) ~ x), df; horizon = 5)

    # Test basic summarize returns IRFSummary
    summary_obj = summarize(lp_result)
    @test summary_obj isa IRFSummary
    @test length(summary_obj.horizon) == 6  # horizons 0-5
    @test summary_obj.horizon == collect(0:5)
    @test summary_obj.term == :x
    @test summary_obj.level == 0.95

    # Test conversion to DataFrame
    summary_df = DataFrame(summary_obj)
    @test summary_df isa DataFrame
    @test nrow(summary_df) == 6
    @test names(summary_df) ==
        ["response", "term", "horizon", "coef", "se", "lower", "upper", "level", "scale"]

    # Test with scale
    summary_scaled = summarize(lp_result; scale = 100)
    @test summary_scaled.coef ≈ summary_obj.coef .* 100
    @test summary_scaled.scale == 100.0

    # Default standard errors are the stored HC1 ones; an explicit HC1 matches
    @test summary_obj.se == stderror(lp_result)
    summary_direct = summarize(lp_result, HC1())
    @test summary_direct.coef ≈ summary_obj.coef atol = 1.0e-10
    @test summary_direct.se ≈ summary_obj.se

    # Test confidence bounds are sensible (lower < coef < upper when se > 0)
    for i in 1:length(summary_obj.horizon)
        if summary_obj.se[i] > 0
            @test summary_obj.lower[i] < summary_obj.coef[i] < summary_obj.upper[i]
        end
    end

    # Test different confidence level
    summary_90 = summarize(lp_result; level = 0.9)
    @test summary_90.level == 0.9
    # 90% CI should be narrower than 95% CI
    @test all(
        summary_90.upper .- summary_90.lower .< summary_obj.upper .- summary_obj.lower,
    )
end

@testitem "IRFSummary table, display and level validation" tags = [:summarize, :api] begin
    using LocalProjections
    using DataFrames, StatsModels, Tables, Test
    using MacroEconometricTools: point_estimate, lowerbounds, upperbounds

    x = sin.(0.3 .* (1:80))
    df = DataFrame(x = x, y = 0.5 .* x .+ cos.(1:80))
    r = lp(@formula(leads(y) ~ x), df; horizon = 4)
    s = summarize(r; level = 0.9)

    # Tables.jl interface carries every field, one row per horizon
    @test Tables.istable(s)
    ct = Tables.columntable(s)
    @test ct.horizon == 0:4
    @test all(==(:y), ct.response) && all(==(:x), ct.term)
    @test all(==(0.9), ct.level) && all(==(1.0), ct.scale)
    @test ct.coef == s.coef && ct.lower == s.lower

    # Same numbers as the MET IRF object at that coverage
    irf = as_irf_result(r; coverage = [0.9])
    @test s.coef == vec(Array(point_estimate(irf)))
    @test s.lower == vec(Array(only(lowerbounds(irf))))
    @test s.upper == vec(Array(only(upperbounds(irf))))

    # A negative scale flips the response and keeps lower <= upper
    sneg = summarize(r; level = 0.9, scale = -2)
    @test sneg.coef ≈ -2 .* s.coef
    @test sneg.se ≈ 2 .* s.se
    @test sneg.lower ≈ -2 .* s.upper
    @test sneg.upper ≈ -2 .* s.lower

    # Display: no ANSI codes unless the caller asks for color; compact form has no newline
    plain = sprint(show, MIME("text/plain"), s)
    @test !occursin('\e', plain)
    @test occursin("Impulse response of y to x", plain)
    @test occursin("Lower 90%", plain)
    @test !occursin("0.0 │", plain)  # horizons print as integers
    @test sprint(show, s) == "IRFSummary(response=y, term=x, level=0.9, scale=1.0)"

    # One validation of confidence levels for every entry point
    msg = "confidence levels must lie strictly between 0 and 1; got 95"
    @test_throws msg summarize(r; level = 95)
    @test_throws msg as_irf_result(r; coverage = [0.9, 95])
    @test_throws "at least one confidence level is required" as_irf_result(r; coverage = Float64[])
end

@testitem "plus vcov operator" tags = [:vcov, :api] begin
    using LocalProjections
    using Regress: VcovSpec
    using DataFrames, StatsModels, Test
    using CovarianceMatrices: HC1, Bartlett, NeweyWest

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
    @test length(nobs(lp_robust)) == length(nobs(lp_result))

    # Test chaining: (lp + vcov(A)) + vcov(B)
    lp_chained = (lp_result + vcov(HC1())) + vcov(Bartlett{NeweyWest}())
    @test lp_chained isa LocalProjection
    @test coefpath(lp_chained) == coefpath(lp_result)
end

@testitem "stored vcov estimator drives inference" tags = [:vcov, :api] begin
    using LocalProjections
    using DataFrames, StatsModels, StableRNGs, Test
    using CovarianceMatrices: HC1, Bartlett, NeweyWest
    import Regress

    rng = StableRNG(4242)
    n = 200
    df = DataFrame(x = randn(rng, n), w = randn(rng, n), y = randn(rng, n))
    r = lp(@formula(leads(y) ~ x + w), df; horizon = 4)
    hac = Bartlett{NeweyWest}()
    r_hac = r + vcov(hac)

    # The generics are shared with Regress, so loading both is unambiguous
    @test vcov === Regress.vcov
    @test stderror === Regress.stderror

    # stderror(lp; term) collects each horizon model's stored standard errors
    k = findfirst(==("w"), coefnames(r))
    @test stderror(r) == [stderror(horizonmodel(r, h))[2] for h in 0:4]
    @test stderror(r_hac; term = :w) == [stderror(horizonmodel(r_hac, h))[k] for h in 0:4]
    @test stderror(r_hac) != stderror(r)
    @test_throws "term z not present in model" stderror(r; term = :z)

    # Downstream functions default to the stored estimator
    @test summarize(r_hac).se == stderror(r_hac)
    @test summarize(r_hac).se ≈ summarize(r, hac).se
    @test summarize(r_hac, HC1()).se ≈ summarize(r).se

    # `show` reports the stored estimator; `repr` module-qualifies the names unless
    # CovarianceMatrices is loaded in `Main`
    @test occursin(r"Vcov: +(CovarianceMatrices\.)?HR1\(\)", sprint(show, MIME"text/plain"(), r))
    @test occursin(
        r"Bartlett\{(CovarianceMatrices\.)?NeweyWest\}\(\)", sprint(show, MIME"text/plain"(), r_hac)
    )
end

@testitem "lpiv basic functionality" tags = [:lpiv, :iv, :core] begin
    using LocalProjections
    using Regress: FirstStageIV
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
    result = lpiv(TSLS(), @formula(leads(y) ~ (x ~ z)), df; horizon = horizon)

    # Check struct type
    @test result isa LocalProjectionIV
    @test result.horizon == horizon
    @test result.response == :y
    @test result.shock == :x

    # Check that we have the right number of models
    @test length(nobs(result)) == horizon + 1

    # Check endogenous and instrument names
    @test :x in Symbol.(result.endogenous_names) || "x" in result.endogenous_names
    @test :z in Symbol.(result.instrument_names) || "z" in result.instrument_names

    # The stored estimator is the covariance estimator, not the IV estimator
    @test occursin(r"Vcov: +(CovarianceMatrices\.)?HR1\(\)", sprint(show, MIME"text/plain"(), result))

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

    # Standard errors across horizons
    se = stderror(result; term = :x)
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
    result = lpiv(TSLS(), @formula(leads(y) ~ (x ~ z) + w), df; horizon = horizon)

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
    using Regress: VcovSpec
    using DataFrames, StatsModels, Test
    using CovarianceMatrices: HC1, Bartlett, NeweyWest
    using Random
    using LinearAlgebra

    Random.seed!(456)

    n = 150
    z = randn(n)
    u = randn(n)
    x = 0.6 * z + 0.4 * u
    y = 0.5 .+ 1.5 * x .+ u

    df = DataFrame(y = y, x = x, z = z)

    result = lpiv(TSLS(), @formula(leads(y) ~ (x ~ z)), df; horizon = 3)

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
    result = lpiv(TSLS(), @formula(cumul(y) ~ (x ~ z)), df; horizon = horizon)

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

    result = lpiv(TSLS(), @formula(leads(y) ~ (x ~ z)), df; horizon = 3)

    # Test summarize
    summary_obj = summarize(result)
    @test summary_obj isa IRFSummary
    @test length(summary_obj.horizon) == 4  # 0-3
    @test summary_obj.term == :x

    # Convert to DataFrame
    summary_df = DataFrame(summary_obj)
    @test summary_df isa DataFrame
    @test nrow(summary_df) == 4

    # Test with scale
    summary_scaled = summarize(result; scale = 100)
    @test summary_scaled.coef ≈ summary_obj.coef .* 100

    # Test with estimator directly
    summary_direct = summarize(result, HC1())
    @test summary_direct.coef ≈ summary_obj.coef atol = 1.0e-10
    @test summary_direct.se ≈ summary_obj.se
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
    result = lpiv(TSLS(), @formula(leads(y) ~ (x ~ z)), df; horizon = 0)
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
    @test lpiv_coef ≈ beta_2sls[2] atol = 1.0e-8
end

@testitem "Intercept flag matches Regress formula fits" tags = [:core, :lpiv, :verification] begin
    using LocalProjections
    using DataFrames, StatsModels, Test
    using StatsAPI: adjr2, r2, rss
    using StableRNGs
    import Regress

    rng = StableRNG(7)
    n, h = 300, 2
    z = randn(rng, n)
    x = 0.6 .* z .+ randn(rng, n)
    w = randn(rng, n)
    y = 3.0 .+ 0.5 .* x .+ 0.2 .* w .+ randn(rng, n)
    df = DataFrame(; y, x, z, w)
    # Row t of `fit` holds y[t + h], as `leads(y)` does at horizon h.
    fit = DataFrame(; yh = [y[(1 + h):end]; fill(missing, h)], x, z, w)

    r = lp(@formula(leads(y) ~ x + w), df; horizon = h)
    ref = Regress.ols(fit, @formula(yh ~ x + w))
    @test r2(horizonmodel(r, h)) ≈ r2(ref)
    @test adjr2(horizonmodel(r, h)) ≈ adjr2(ref)

    # Without an intercept, R² is uncentered, as in the formula fit.
    r0 = lp(@formula(leads(y) ~ 0 + x + w), df; horizon = h)
    ref0 = Regress.ols(fit, @formula(yh ~ 0 + x + w))
    @test r2(horizonmodel(r0, h)) ≈ r2(ref0)
    @test r2(horizonmodel(r0, h)) ≈ 1 - rss(ref0) / sum(abs2, y[(1 + h):end])

    riv = lpiv(TSLS(), @formula(leads(y) ~ (x ~ z) + w), df; horizon = h)
    refiv = Regress.iv(TSLS(), fit, @formula(yh ~ (x ~ z) + w))
    @test r2(horizonmodel(riv, h)) ≈ r2(refiv)
    fs = first_stage(riv, h)
    @test r2(only(fs.models)) ≈ r2(Regress.ols(fit[1:(n - h), :], @formula(x ~ z + w)))
    @test fs.F_nonrobust ≈ Regress.first_stage(refiv).F_nonrobust
end

@testitem "k-class lpiv matches Regress formula fits" tags = [:lpiv, :iv, :verification] begin
    using LocalProjections
    using DataFrames, StatsModels, Test
    using StatsAPI: coef
    using StableRNGs
    import Regress

    rng = StableRNG(11)
    n, h = 300, 2
    z1, z2, z3, u, w = (randn(rng, n) for _ in 1:5)
    x = 0.5 .* z1 .+ 0.4 .* z2 .+ 0.3 .* z3 .+ 0.6 .* u
    y = 1.0 .+ 0.8 .* x .+ 0.2 .* w .+ u .+ 0.15 .* z3
    df = DataFrame(; y, x, z1, z2, z3, w)
    # Row t of `fit` holds y[t + h], as `leads(y)` does at horizon h.
    fit = DataFrame(; yh = [y[(1 + h):end]; fill(missing, h)], x, z1, z2, z3, w)
    f = @formula(leads(y) ~ (x ~ z1 + z2 + z3) + w)
    rt = lpiv(TSLS(), f, df; horizon = h)

    for est in (LIML(), Fuller(1.0), KClass(1.5))
        # Fuller(1)'s kappa is just below 1 at horizon 2 on this sample, which warns
        fits = () -> (
            lpiv(est, f, df; horizon = h),
            Regress.iv(est, fit, @formula(yh ~ (x ~ z1 + z2 + z3) + w)),
        )
        r, ref = est isa Fuller ?
            (@test_logs (:warn, r"kappa < 1") match_mode = :any fits()) : fits()
        m = horizonmodel(r, h)
        @test m.estimator == est
        @test coef(m) ≈ coef(ref)
        @test stderror(m) ≈ stderror(ref)
        @test coefpath(r; term = :x) != coefpath(rt; term = :x)
        # The first stage does not depend on the second-stage estimator
        @test first_stage(r, h).F_robust ≈ first_stage(rt, h).F_robust
        @test weakivtest(r, h).F_eff ≈ weakivtest(rt, h).F_eff
    end
end

@testitem "Per-horizon models carry coefficient names" tags = [:core, :lpiv, :api] begin
    using LocalProjections
    using DataFrames, StatsModels, Test
    using CovarianceMatrices: HC3
    using StatsAPI: coeftable
    using StableRNGs

    rng = StableRNG(11)
    n, H = 200, 3
    z = randn(rng, n)
    x = 0.6 .* z .+ randn(rng, n)
    w = randn(rng, n)
    y = 0.5 .* x .+ 0.2 .* w .+ randn(rng, n)
    df = DataFrame(; y, x, z, w)

    r = lp(@formula(leads(y) ~ x + w), df; horizon = H)
    @test coefnames(r) == ["(Intercept)", "x", "w"]
    for h in 0:H
        @test coefnames(horizonmodel(r, h)) == coefnames(r)
    end
    rh = r + vcov(HC3())
    @test coefnames(horizonmodel(rh, H)) == coefnames(r)
    @test coeftable(horizonmodel(r, 1)).rownms == ["x", "w", "(Intercept)"]
    @test occursin("w", sprint(show, coeftable(horizonmodel(r, 1))))

    riv = lpiv(TSLS(), @formula(leads(y) ~ (x ~ z) + w), df; horizon = H)
    @test coefnames(horizonmodel(riv, 2)) == coefnames(riv) == ["(Intercept)", "w", "x"]
    @test coeftable(horizonmodel(riv, 2)).rownms == ["w", "x", "(Intercept)"]
    fs = first_stage(riv, 2)
    @test fs.endogenous_names == ["x"]
    @test fs.instrument_names == ["z"]
    @test coefnames(only(fs.models)) == ["(Intercept)", "w", "z"]
end

@testitem "@formula is exported" tags = [:api] begin
    using LocalProjections
    using DataFrames, Test
    import StatsModels

    @test LocalProjections.var"@formula" === StatsModels.var"@formula"
    df = DataFrame(y = randn(50), x = randn(50))
    @test lp(@formula(leads(y) ~ x), df; horizon = 2) isa LocalProjection
end

@testitem "weakivtest for lpiv" tags = [:lpiv, :weakiv, :iv] begin
    using LocalProjections
    using Regress: WeakIVTestResult
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
    result = lpiv(TSLS(), @formula(leads(y) ~ (x ~ z + z2 + z3)), df; horizon = horizon)

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
    @test_throws "horizon -1 is outside the estimated horizons 0:$horizon" weakivtest(result, -1)
    @test_throws "horizon $(horizon + 1) is outside the estimated horizons 0:$horizon" first_stage(result, horizon + 1)

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
        TSLS(),
        @formula(leads(y) ~ (x ~ z) + lags(w1, 2) + lags(w2, 2)),
        df; horizon = horizon
    )
    @test result isa LocalProjectionIV
    @test length(nobs(result)) == horizon + 1

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
    using Regress: FirstStageIV
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
    result = lpiv(TSLS(), @formula(leads(x) ~ (x ~ z) + w), df; horizon = horizon)

    @test result isa LocalProjectionIV
    @test result.tautological_h0

    # coefpath: h=0 should be exactly 1.0 for the shock term
    irf = coefpath(result; term = :x)
    @test irf[1] == 1.0
    @test all(isfinite, irf[2:end])

    # coefpath for a non-shock term: h=0 should be 0.0
    irf_w = coefpath(result; term = :w)
    @test irf_w[1] == 0.0

    # stderror: h=0 standard error is exactly 0, also after changing the estimator
    se = stderror(result; term = :x)
    @test se[1] == 0.0
    @test all(isfinite, se[2:end])
    @test stderror(result + vcov(HC1()); term = :w)[1] == 0.0

    # summarize: should not error
    summary = summarize(result, HC1())
    @test summary.coef[1] == 1.0
    @test summary.se[1] == 0.0
    @test summary.lower[1] == 1.0
    @test summary.upper[1] == 1.0

    # weakivtest: h=0 has no IV estimate to test
    wiv = weakivtest(result)
    @test length(wiv) == horizon + 1
    @test wiv[1] === nothing
    @test all(r -> isfinite(r.F_eff), wiv[2:end])
    @test_throws ArgumentError weakivtest(result, 0)
    @test_throws "weakivtest is undefined at horizon 0 when the response equals the shock (x)" weakivtest(result, 0)
    @test wiv[2].F_eff == weakivtest(result, 1).F_eff
    @test isfinite(weakivtest(result, 1).F_eff)

    # first_stage at h=0 is still available
    @test first_stage(result, 0) isa FirstStageIV

    # OLS LP where response == shock
    ols_result = lp(@formula(leads(x) ~ x + w), df; horizon = horizon)
    @test ols_result.tautological_h0
    irf_ols = coefpath(ols_result; term = :x)
    @test irf_ols[1] == 1.0

    # Non-tautological case: response != shock
    result_ok = lpiv(TSLS(), @formula(leads(y) ~ (x ~ z) + w), df; horizon = horizon)
    @test !result_ok.tautological_h0
end

@testitem "shock selection and formula validation" tags = [:core, :lpiv, :api] begin
    using LocalProjections
    using DataFrames, StatsModels, Test
    using StableRNGs

    rng = StableRNG(20261019)
    n = 120
    df = DataFrame(
        y = randn(rng, n), x = randn(rng, n), w = randn(rng, n), z = randn(rng, n)
    )

    @test lp(@formula(leads(y) ~ x), df; horizon = 1) isa AbstractLocalProjection
    @test lpiv(TSLS(), @formula(leads(y) ~ (x ~ z)), df; horizon = 1) isa
        AbstractLocalProjection

    # Default shock: first regressor other than the intercept, chosen by name
    @test lp(@formula(leads(y) ~ w + x), df; horizon = 1).shock == :w
    r0 = lp(@formula(leads(y) ~ 0 + x + w), df; horizon = 1)
    @test r0.shock == :x
    @test r0.coef_names == ["x", "w"]
    @test lp(@formula(leads(y) ~ 0 + x), df; horizon = 1).shock == :x

    # IV default shock: first endogenous variable, wherever it appears
    @test lpiv(TSLS(), @formula(leads(y) ~ w + (x ~ z)), df; horizon = 1).shock == :x
    riv0 = lpiv(TSLS(), @formula(leads(y) ~ 0 + (x ~ z)), df; horizon = 1)
    @test riv0.shock == :x
    @test riv0.coef_names == ["x"]

    # Explicit shock
    @test lp(@formula(leads(y) ~ x + w), df; horizon = 1, shock = :w).shock == :w
    @test lpiv(TSLS(), @formula(leads(y) ~ (x ~ z) + w), df; horizon = 1, shock = :w).shock ==
        :w

    # Unknown shock: same message for OLS and IV
    msg = "shock `nope` is not a coefficient of the model; available coefficients: (Intercept), x"
    @test_throws msg lp(@formula(leads(y) ~ x), df; horizon = 1, shock = :nope)
    @test_throws msg lpiv(TSLS(), @formula(leads(y) ~ (x ~ z)), df; horizon = 1, shock = :nope)

    # The shock is validated before any horizon is fitted
    df_nan = DataFrame(y = fill(NaN, n), x = df.x, z = df.z)
    @test_throws "shock `nope`" lp(@formula(leads(y) ~ x), df_nan; horizon = 1, shock = :nope)
    @test_throws "No complete observations available for horizon 0" lp(
        @formula(leads(y) ~ x), df_nan; horizon = 1
    )

    # Response without a horizon transformation: same message for OLS and IV
    msg = "the response must use leads(), cumul(), anchor(), or the y|z syntax"
    @test_throws msg lp(@formula(y ~ x), df; horizon = 1)
    @test_throws msg lpiv(TSLS(), @formula(y ~ (x ~ z)), df; horizon = 1)

    @test_throws "horizon must be non-negative (got -1)" lp(@formula(leads(y) ~ x), df; horizon = -1)
    @test_throws "horizon must be non-negative (got -1)" lpiv(
        TSLS(), @formula(leads(y) ~ (x ~ z)), df; horizon = -1
    )

    # OLS and IV formulas are not interchangeable
    @test_throws "use `lpiv(TSLS(), formula, data; horizon)`" lp(
        @formula(leads(y) ~ (x ~ z)), df; horizon = 1
    )
    @test_throws "lpiv requires an `(endogenous ~ instruments)` term" lpiv(
        TSLS(), @formula(leads(y) ~ x), df; horizon = 1
    )
    @test_throws "Not enough instruments: 1 < 2" lpiv(
        TSLS(), @formula(leads(y) ~ (x + w ~ z)), df; horizon = 1
    )
end

@testitem "Tables.jl input, categorical controls and sample policy" tags = [:core, :lpiv, :api] begin
    using LocalProjections
    using DataFrames, StatsModels, StatsBase, Tables, Test
    using CovarianceMatrices: HC1
    using StableRNGs

    rng = StableRNG(20261020)
    n = 150
    x = randn(rng, n)
    z = x .+ randn(rng, n)
    w = randn(rng, n)
    y = cumsum(randn(rng, n)) .+ 0.5 .* x
    g = rand(rng, ["a", "b", "c"], n)
    df = DataFrame(; y, x, z, w, g)
    H = 4

    # Any Tables.jl table gives the same estimates as a DataFrame
    f = @formula(leads(y) ~ x + lags(w, 2))
    r_df = lp(f, df; horizon = H)
    r_nt = lp(f, (; y, x, z, w); horizon = H)
    @test coefpath(r_nt) == coefpath(r_df)
    @test nobs(r_nt) == nobs(r_df)
    fiv = @formula(cumul(y) ~ (x ~ z) + w)
    @test coefpath(lpiv(TSLS(), fiv, Tables.rowtable(df); horizon = H)) ==
        coefpath(lpiv(TSLS(), fiv, df; horizon = H))
    @test_throws "data must be a Tables.jl table" lp(f, [y x]; horizon = H)

    # String columns enter through StatsModels contrasts (dummy coding)
    rc = lp(@formula(leads(y) ~ x + g), df; horizon = 1)
    @test rc.coef_names == ["(Intercept)", "x", "g: b", "g: c"]
    Xg = hcat(ones(n), x, g .== "b", g .== "c")
    @test coef(horizonmodel(rc, 0)) ≈ Xg \ y
    riv = lpiv(TSLS(), @formula(leads(y) ~ (x ~ z) + g), df; horizon = 1)
    @test riv.coef_names == ["(Intercept)", "g: b", "g: c", "x"]
    g_miss = allowmissing(g)
    g_miss[3] = missing
    @test_throws "categorical variable `g` contains missing values" lp(
        @formula(leads(y) ~ x + g), DataFrame(; y, x, g = g_miss); horizon = 1
    )

    # Missing values in numeric columns drop rows per horizon without shifting time:
    # with y[50] missing, horizon h loses row 50 - h (and the last h rows).
    y_miss = allowmissing(y)
    y_miss[50] = missing
    rm = lp(@formula(leads(y) ~ x), DataFrame(; y = y_miss, x); horizon = 2)
    @test nobs(rm) == [n - 1, n - 2, n - 3]
    for h in 0:2
        keep = [t + h <= n && !ismissing(y_miss[t + h]) for t in 1:n]
        y_h = Float64[y_miss[t + h] for t in 1:n if keep[t]]
        @test coef(horizonmodel(rm, h)) ≈ hcat(ones(sum(keep)), x[keep]) \ y_h
    end

    # Fixed sample: every horizon uses the rows available at all horizons
    r_fix = lp(f, df; horizon = H, sample = :fixed)
    @test r_fix.sample === :fixed
    @test r_df.sample === :varying
    @test allequal(nobs(r_fix))
    @test nobs(horizonmodel(r_fix, 0)) == n - 2 - H  # two lags at the start, H leads at the end
    @test coef(horizonmodel(r_fix, H)) ≈ coef(horizonmodel(r_df, H))
    rows = 3:(n - H)
    X_fix = hcat(ones(length(rows)), x[rows], w[rows .- 1], w[rows .- 2])
    @test coef(horizonmodel(r_fix, 0)) ≈ X_fix \ y[rows]
    riv_fix = lpiv(TSLS(), fiv, df; horizon = H, sample = :fixed)
    @test riv_fix.sample === :fixed
    @test allequal(nobs(riv_fix))
    @test riv_fix.sample === (riv_fix + vcov(HC1())).sample

    msg = "sample must be :varying or :fixed (got :all)"
    @test_throws msg lp(f, df; horizon = H, sample = :all)
    @test_throws msg lpiv(TSLS(), fiv, df; horizon = H, sample = :all)

    # Per-horizon sample sizes are displayed
    shown = sprint(show, MIME"text/plain"(), r_df)
    @test occursin("Sample:     varying", shown)
    @test occursin("Obs (h=0:$H): $(join(nobs(r_df), ", "))", shown)
    @test occursin("Sample:       fixed", sprint(show, MIME"text/plain"(), riv_fix))
end

@testitem "Panel local projections" tags = [:panel, :core, :lpiv, :api] begin
    using DataFrames, StatsBase, StatsModels, Test
    using StableRNGs

    rng = StableRNG(20261022)
    N, T = 3, 15
    df = DataFrame(
        id = repeat(["a", "b", "c"], inner = T), t = repeat(1:T, N),
        y = randn(rng, N * T), x = randn(rng, N * T), z = randn(rng, N * T)
    )
    shuffled = df[sortperm(randn(rng, N * T)), :]
    H = 3

    # Regression of resp(g, i, h) on regressors(g, i), pooling rows i of every unit g
    # for which all values exist inside that unit
    function pooled(resp, regressors, first_row, h)
        X = Vector{Float64}[]
        Y = Float64[]
        for g in groupby(df, :id; sort = true), i in first_row:(T - h)
            push!(Y, resp(g, i, h))
            push!(X, regressors(g, i))
        end
        return reduce(hcat, X)' \ Y
    end

    # Leads and lags are computed within unit; unit dummies through a string column
    r = lp(@formula(leads(y) ~ x + lags(z, 1) + id), shuffled; horizon = H, unit = :id, time = :t)
    @test coefnames(r) == ["(Intercept)", "x", "z_lag1", "id: b", "id: c"]
    @test nobs(r) == [N * (T - 1 - h) for h in 0:H]
    for h in 0:H
        @test coef(horizonmodel(r, h)) ≈ pooled(
            (g, i, h) -> g.y[i + h],
            (g, i) -> [1.0, g.x[i], g.z[i - 1], g.id[i] == "b", g.id[i] == "c"], 2, h
        )
    end
    rc = lp(@formula(cumul(y) ~ x), shuffled; horizon = H, unit = :id, time = :t)
    ra = lp(@formula(y | z ~ x), shuffled; horizon = H, unit = :id, time = :t)
    for h in 0:H
        @test coef(horizonmodel(rc, h)) ≈
            pooled((g, i, h) -> sum(g.y[i:(i + h)]), (g, i) -> [1.0, g.x[i]], 1, h)
        @test coef(horizonmodel(ra, h)) ≈
            pooled((g, i, h) -> g.y[i + h] - g.z[i], (g, i) -> [1.0, g.x[i]], 1, h)
    end
    riv = lpiv(TSLS(), @formula(leads(y) ~ (x ~ z)), shuffled; horizon = H, unit = :id, time = :t)
    @test nobs(riv) == [N * (T - h) for h in 0:H]
    @test (riv.unit, riv.time) == (:id, :t)

    # Row order does not matter; a single unit matches the time-series fit
    @test coefpath(lp(@formula(cumul(y) ~ x), df; horizon = H, unit = :id, time = :t)) ==
        coefpath(rc)
    a = df[df.id .== "a", :]
    f = @formula(leads(y) ~ x + lags(z, 2))
    @test coefpath(lp(f, a; horizon = H, unit = :id, time = :t)) == coefpath(lp(f, a; horizon = H))
    @test coefpath(lp(f, a[end:-1:1, :]; horizon = H, time = :t)) == coefpath(lp(f, a; horizon = H))

    # Any sortable time values; units of different lengths, some shorter than the horizon
    quarterly = transform(shuffled, :t => ByRow(t -> "$(2000 + (t - 1) ÷ 4)Q$(mod1(t, 4))") => :q)
    rq = lp(
        @formula(leads(y) ~ x + lags(z, 1) + id), quarterly; horizon = H, unit = :id, time = :q
    )
    @test coefpath(rq) == coefpath(r)
    short = vcat(df[df.t .>= 4, :], DataFrame(id = "d", t = [1, 2], y = 1.0, x = 1.0, z = 1.0))
    rs = lp(@formula(leads(y) ~ x), short; horizon = H, unit = :id, time = :t)
    @test nobs(rs) == [N * (T - 3 - h) + max(2 - h, 0) for h in 0:H]

    # Display
    shown = sprint(show, MIME"text/plain"(), r)
    @test occursin("Unit:       id", shown)
    @test occursin("Time:       t", shown)
    @test !occursin("Unit:", sprint(show, MIME"text/plain"(), lp(f, a; horizon = H)))

    # Bad panel structure fails fast
    f0 = @formula(leads(y) ~ x)
    @test_throws "time 1 appears in more than one row in unit \"a\"" lp(
        f0, vcat(df, df[1:1, :]); horizon = 1, unit = :id, time = :t
    )
    @test_throws "time 2 appears in more than one row" lp(
        f0, a[[1, 2, 2, 3], :]; horizon = 1, time = :t
    )
    @test_throws "no row at time 5 in unit \"a\", between 4 and 6; add a row with missing values" lp(
        f0, df[Not(5), :]; horizon = 1, unit = :id, time = :t
    )
    @test_throws "`unit` requires `time`" lp(f0, df; horizon = 1, unit = :id)
    @test_throws "time column `date` is not in the data" lp(
        f0, df; horizon = 1, unit = :id, time = :date
    )
    @test_throws "unit column `country` is not in the data" lp(
        f0, df; horizon = 1, unit = :country, time = :t
    )
    dm = allowmissing(df, :id)
    dm.id[3] = missing
    @test_throws "unit column `id` contains missing or NaN values" lp(
        f0, dm; horizon = 1, unit = :id, time = :t
    )
    @test_throws "time column `t` contains missing or NaN values" lp(
        f0, transform(df, :t => (t -> [t[1:(end - 1)]; NaN]) => :t); horizon = 1, unit = :id, time = :t
    )
end

@testitem "Panel clustered inference" tags = [:panel, :vcov, :lpiv, :api] begin
    using DataFrames, StatsBase, StatsModels, Test
    using CovarianceMatrices: CovarianceMatrices, HC1, CR0, CR1, CR2, Bartlett, DriscollKraay, Clustering
    using StableRNGs

    rng = StableRNG(20261023)
    N, T = 4, 12
    df = DataFrame(
        id = repeat(["a", "b", "c", "d"], inner = T), t = repeat(1:T, N),
        y = randn(rng, N * T), x = randn(rng, N * T), z = randn(rng, N * T)
    )
    df.grp = repeat(1:2, inner = 2T)
    shuffled = df[sortperm(randn(rng, N * T)), :]
    H = 3
    f = @formula(leads(y) ~ x)

    # Rows used at horizon h, in (unit, time) order: periods 1:(T - h) of every unit
    used(h) = df[df.t .<= T - h, :]
    manual_se(r, est, h) = stderror(horizonmodel(r + vcov(HC1()), h) + vcov(est))[2]

    # Panels cluster by unit by default
    r = lp(f, shuffled; horizon = H, unit = :id, time = :t)
    @test occursin("Vcov:       CR1(:id)", sprint(show, MIME"text/plain"(), r))
    @test stderror(r) ≈ [manual_se(r, CR1(used(h).id), h) for h in 0:H]
    @test stderror(r) == stderror(r + vcov(CR1(:id)))
    @test DataFrame(summarize(r)).se == stderror(r)
    @test stderror(r + vcov(HC1())) != stderror(r)

    # Multi-way clusters, other CR estimators, columns outside the formula
    r2 = r + vcov(CR1((:id, :t)))
    @test occursin("Vcov:       CR1((:id, :t))", sprint(show, MIME"text/plain"(), r2))
    @test stderror(r2) ≈ [manual_se(r, CR1((used(h).id, used(h).t)), h) for h in 0:H]
    @test stderror(r + vcov(CR0(:grp))) ≈ [manual_se(r, CR0(used(h).grp), h) for h in 0:H]
    @test stderror(r + vcov(CR2(:id))) ≈ [manual_se(r, CR2(used(h).id), h) for h in 0:H]

    # A single series clusters by any column; its default stays HC1
    a = df[df.id .== "a", :]
    a.week = repeat(1:3, inner = 4)
    ra = lp(f, a; horizon = H)
    @test !occursin("CR1", sprint(show, MIME"text/plain"(), ra))
    @test stderror(ra + vcov(CR1(:week))) ≈
        [manual_se(ra, CR1(a.week[1:(T - h)]), h) for h in 0:H]

    # Driscoll-Kraay names the time and unit columns
    for dk in (DriscollKraay(Bartlett(2); tis = :t, iis = :id), DriscollKraay(Bartlett(2), :t, :id; correction = :sss))
        rdk = r + vcov(dk)
        @test occursin("Vcov:       $(repr(dk))", sprint(show, MIME"text/plain"(), rdk))
        @test stderror(rdk) ≈ [
            manual_se(r, DriscollKraay(Bartlett(2), used(h).t, used(h).id; correction = CovarianceMatrices.correction(dk)), h)
                for h in 0:H
        ]
    end

    # A period with no rows at some horizon keeps its distance from its neighbors: with
    # y missing at t = 6 for every unit, horizon h has no rows at period 6 - h
    gap = transform(df, [:y, :t] => ByRow((y, t) -> t == 6 ? missing : y) => :y)
    rg = lp(f, gap; horizon = H, unit = :id, time = :t)
    dk = DriscollKraay(Bartlett(3); tis = :t, iis = :id)
    for h in 0:H
        rows = (gap.t .<= T - h) .& (gap.t .!= 6 - h)
        th, uh = gap.t[rows], gap.id[rows]
        on_grid = DriscollKraay(Bartlett(3), Clustering(th, T - h), uh)
        compressed = DriscollKraay(Bartlett(3), th, uh)
        @test stderror(rg + vcov(dk))[h + 1] ≈ manual_se(rg, on_grid, h)
        @test !(stderror(rg + vcov(dk))[h + 1] ≈ manual_se(rg, compressed, h))
    end

    # IV local projections
    riv = lpiv(TSLS(), @formula(leads(y) ~ (x ~ z)), shuffled; horizon = H, unit = :id, time = :t)
    @test occursin("Vcov:         CR1(:id)", sprint(show, MIME"text/plain"(), riv))
    @test stderror(riv) ≈ [manual_se(riv, CR1(used(h).id), h) for h in 0:H]
    @test stderror(riv + vcov(dk)) ≈ [
        manual_se(riv, DriscollKraay(Bartlett(3), used(h).t, used(h).id), h) for h in 0:H
    ]

    # Estimators that cannot be resolved per horizon fail fast
    @test_throws "cluster column `country` is not in the data" r + vcov(CR1(:country))
    @test_throws "holds one cluster value per observation" r + vcov(CR1(df.id))
    @test_throws "holds one time and unit index per observation" r + vcov(
        DriscollKraay(Bartlett(2), df.t, df.id)
    )
    @test_throws "time index column `date` is not in the data" r + vcov(
        DriscollKraay(Bartlett(2), :date, :id)
    )
    dm = allowmissing(df, :grp)
    dm.grp[T] = missing   # unit "a", period T: used only at h = 0
    rm = lp(f, dm; horizon = H, unit = :id, time = :t)
    @test_throws "cluster column `grp` contains missing or NaN values in the rows used at horizon 0" rm +
        vcov(CR1(:grp))
end

@testitem "Absorbed fixed effects" tags = [:panel, :vcov, :lpiv, :api] begin
    using DataFrames, StatsBase, StatsModels, Test
    using CovarianceMatrices: HC1, CR1
    using Regress: Regress
    using StableRNGs

    rng = StableRNG(20261020)
    N, T = 12, 15
    df = DataFrame(id = repeat(1:N, inner = T), t = repeat(1:T, N))
    a = randn(rng, N)
    df.x = randn(rng, N * T) .+ a[df.id]
    df.z = df.x .+ randn(rng, N * T)
    df.w = randn(rng, N * T)
    df.y = 0.5 .* df.x .+ 0.3 .* df.w .+ 2 .* a[df.id] .+ randn(rng, N * T)
    df.sid = string.(df.id)
    shuffled = df[sortperm(randn(rng, N * T)), :]
    H = 3

    # Response at t + h within each unit; rows without it are dropped
    function at_horizon(h)
        d = copy(df)
        d.yh = [i + h <= nrow(d) && d.id[i + h] == d.id[i] ? d.y[i + h] : missing for i in 1:nrow(d)]
        return dropmissing(d, :yh)
    end

    r = lp(@formula(leads(y) ~ x + w + fe(id)), shuffled; horizon = H, unit = :id, time = :t)
    @test coefnames(r) == ["x", "w"]
    @test all(h -> coefnames(horizonmodel(r, h)) == ["x", "w"], 0:H)
    # Absorbing the unit effects gives the coefficients of unit dummies
    rd = lp(@formula(leads(y) ~ x + w + sid), shuffled; horizon = H, unit = :id, time = :t)
    @test coefpath(r) ≈ coefpath(rd)
    @test coefpath(r; term = :w) ≈ coefpath(rd; term = :w)

    for h in 0:H
        d = at_horizon(h)
        fm = Regress.ols(d, @formula(yh ~ x + w + fe(id)); drop_singletons = false)
        @test coef(horizonmodel(r, h)) ≈ coef(fm)
        # Unit FE nested in unit clusters count once in the small-sample factor
        @test stderror(horizonmodel(r, h)) ≈ stderror(fm + vcov(CR1(:id)))
        @test stderror(horizonmodel(r + vcov(HC1()), h)) ≈ stderror(fm + vcov(HC1()))
        @test nobs(horizonmodel(r, h)) == nobs(fm)
    end

    r2 = lp(@formula(leads(y) ~ x + w + fe(id) + fe(t)), df; horizon = H, unit = :id, time = :t)
    f2 = Regress.ols(at_horizon(2), @formula(yh ~ x + w + fe(id) + fe(t)); drop_singletons = false)
    @test coef(horizonmodel(r2, 2)) ≈ coef(f2)
    @test stderror(horizonmodel(r2, 2)) ≈ stderror(f2 + vcov(CR1(:id)))

    # An explicit intercept is absorbed too
    @test coefnames(lp(@formula(leads(y) ~ 1 + x + fe(id)), df; horizon = 1, unit = :id, time = :t)) == ["x"]

    riv = lpiv(TSLS(), @formula(leads(y) ~ (x ~ z) + w + fe(id)), df; horizon = H, unit = :id, time = :t)
    fiv = Regress.iv(TSLS(), at_horizon(2), @formula(yh ~ (x ~ z) + w + fe(id)); drop_singletons = false) +
        vcov(CR1(:id))
    @test coef(horizonmodel(riv, 2)) ≈ coef(fiv)
    @test stderror(horizonmodel(riv, 2)) ≈ stderror(fiv)
    @test first_stage(riv, 2).F_robust ≈ Regress.first_stage(fiv).F_robust

    pf(f, d = df) = lp(f, d; horizon = 1, unit = :id, time = :t)
    @test_throws "fixed effects must appear as separate `fe(column)` terms" pf(@formula(leads(y) ~ x + fe(id) & w))
    @test_throws "fixed effects must appear as separate `fe(column)` terms" lpiv(
        TSLS(), @formula(leads(y) ~ (x ~ z + fe(id))), df; horizon = 1
    )
    @test_throws "formula must contain at least one regressor besides fixed effects" pf(@formula(leads(y) ~ fe(id)))
    @test_throws "fixed effect column `nope` is not in the data" pf(@formula(leads(y) ~ x + fe(nope)))
    dm = allowmissing(df, :id)
    dm.g = copy(dm.id)
    dm.g[3] = missing
    @test_throws "fixed effect column `g` contains missing or NaN values" pf(@formula(leads(y) ~ x + fe(g)), dm)
end

@testitem "Horizon-indexed model access" tags = [:api, :core] begin
    using DataFrames, StatsBase, StatsModels
    using StableRNGs

    rng = StableRNG(20261021)
    n = 120
    x = randn(rng, n)
    z = x .+ randn(rng, n)
    y = cumsum(randn(rng, n)) .+ 0.5 .* x
    df = DataFrame(; y, x, z)
    H = 3

    r = lp(@formula(leads(y) ~ x), df; horizon = H)
    ix = findfirst(==("x"), coefnames(r))
    @test coefnames(r) == ["(Intercept)", "x"]
    @test [coef(horizonmodel(r, h))[ix] for h in 0:H] == coefpath(r)
    @test nobs(r) == [nobs(horizonmodel(r, h)) for h in 0:H] == [n - h for h in 0:H]
    @test horizonmodel(r, Int32(2)) === horizonmodel(r, 2)
    @test_throws "horizon -1 is outside the estimated horizons 0:3" horizonmodel(r, -1)
    @test_throws "horizon 4 is outside the estimated horizons 0:3" horizonmodel(r, H + 1)
    @test_throws MethodError coefnames(r, 0)

    riv = lpiv(TSLS(), @formula(leads(y) ~ (x ~ z)), df; horizon = H)
    @test [coef(horizonmodel(riv, h))[ix] for h in 0:H] == coefpath(riv)
    @test nobs(riv) == [n - h for h in 0:H]
    @test_throws "horizon 4 is outside the estimated horizons 0:3" horizonmodel(riv, H + 1)
end

@testitem "Plotting draws the IRF object" tags = [:plots, :api] begin
    ENV["GKSwstype"] = "nul"  # headless GR for Plots
    using LocalProjections
    using DataFrames, StatsModels, CovarianceMatrices, Test
    using MacroEconometricTools: point_estimate, lowerbounds, upperbounds
    import CairoMakie, Plots

    x = sin.(0.3 .* (1:80))
    df = DataFrame(x = x, y = 0.5 .* x .+ cos.(1:80))
    r = lp(@formula(leads(y) ~ x), df; horizon = 4)
    ys(pts) = [p[2] for p in pts]

    # Makie, single panel: line and band are the IRF object's estimate and bounds
    fig = CairoMakie.Figure()
    ax = CairoMakie.Axis(fig[1, 1])
    @test irfplot!(ax, r, HC3(); levels = [0.9]) === ax
    irf = as_irf_result(r + vcov(HC3()); coverage = [0.9])
    line = only(filter(p -> p isa CairoMakie.Lines, ax.scene.plots))
    band = only(filter(p -> p isa CairoMakie.Band, ax.scene.plots))
    @test ys(line[1][]) ≈ vec(Array(point_estimate(irf)))
    @test ys(band[1][]) ≈ vec(Array(only(lowerbounds(irf))))
    @test ys(band[2][]) ≈ vec(Array(only(upperbounds(irf))))

    # Makie, figure: one band per level
    fig = irfplot(r; levels = [0.68, 0.9], xtickstep = 2)
    @test fig isa CairoMakie.Figure
    ax = only(filter(c -> c isa CairoMakie.Axis, fig.content))
    @test count(p -> p isa CairoMakie.Band, ax.scene.plots) == 2

    # Plots: the recipe draws the same bounds and estimate
    plt = Plots.plot(r; levels = [0.9])
    irf = as_irf_result(r; coverage = [0.9])
    @test any(s -> s[:y] ≈ vec(Array(point_estimate(irf))), plt.series_list)
    @test any(
        s -> s[:fillrange] isa AbstractVector &&
            s[:fillrange] ≈ vec(Array(only(lowerbounds(irf)))),
        plt.series_list
    )
    @test Plots.plot(r, HC3()) isa Plots.Plot

    # Levels go through the same validation as summarize and as_irf_result
    msg = "confidence levels must lie strictly between 0 and 1; got 95"
    @test_throws msg irfplot(r; levels = [95])
    @test_throws msg Plots.plot(r; levels = [95])
end

@testitem "export surface" tags = [:api] begin
    using LocalProjections
    using Test
    for name in (
            :AbstractLocalProjection, :lp, :lpiv, :TSLS, :LIML, :Fuller, :KClass,
            :leads, :lags, :lag, :lead, :cumul, :anchor,
            :first_stage, :weakivtest, :as_irf_result, :irfplot, :irfplot!, :fe,
        )
        @test Base.isexported(LocalProjections, name)
    end
    for name in (:CumulTerm, :LeadTerm, :AnchorTerm)
        @test Base.ispublic(LocalProjections, name)
        @test !Base.isexported(LocalProjections, name)
    end
    for name in (:WeakIVTestResult, :FirstStageIV, :LocalProjectionIRFResult, :LPSchemaContext, :VcovSpec)
        @test !Base.ispublic(LocalProjections, name)
    end
end
