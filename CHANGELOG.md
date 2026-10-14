# Changelog

## Unreleased

### Changed

- **Requires Regress.jl 0.2, CovarianceMatrices.jl 0.32 and Julia 1.12.** The new upstream versions fix two defects documented in `upstream_issue_drafts.md`. HAC estimators no longer carry kernel-weight state, so `weakivtest` on a formula-fitted IV model with `Bartlett{NeweyWest}()` no longer crashes (Issue 2). The IV path now excludes the intercept from automatic bandwidth selection, so `lp` and `lpiv` select the same Newey–West/Andrews bandwidth on identical data (Issue 4, REG-2b). Automatic-bandwidth `lpiv` standard errors change as a result; fixed-bandwidth kernel HAC, HR0/HR1 and EWC results are unchanged. HC2/HC3 on over-identified `lpiv` now use the correct IV leverage (Regress fixed the matrix-TSLS formula), so those standard errors change too.
- **Source split into per-topic files** (internal): `src/LocalProjections.jl` now only includes `terms.jl`, `types.jl`, `lp.jl`, `lpiv.jl`, `inference.jl`, `var.jl`, `fevd*.jl`, etc. No behavior change; the order in which multi-method docstrings (e.g. `?summarize`) are listed follows the new include order.
- **`vcov(estimator, lp; kwargs...)` forwards keyword arguments** to each horizon's model. `vcov(Bartlett(6), lpiv_result; dofadjust = true)` adds the `n/(n−k)` factor that `lp` applies to kernel HAC, so `lp` and `lpiv` bands can be put on one convention. EWC stays unscaled, and on `lp` results `dofadjust = false` is still ignored upstream.

### Added

- **Common estimation sample across horizons**: `lp` and `lpiv` take `sample = :horizon | :common`. The default `:horizon` keeps the previous behavior: every horizon uses all the observations available to it, so the sample shrinks as the horizon grows. `:common` estimates every horizon on the observations complete at all horizons `0:horizon` (the sample of the maximum horizon, intersected with any horizon-specific gaps), so the impulse response is not driven by a changing sample. The choice is stored in the new `sample` field of `LocalProjection`/`LocalProjectionIV`, shown by `show`, kept by `+ vcov(...)` and reused by `varbootstrap` in every draw. `biascorrect`, `varbootstrap(...; biascorrect = true)` and `fevd(lp_result, data)` reject `sample = :common`: the Herbst–Johannsen correction and the Gorodnichenko–Lee estimators are derived for horizon-specific samples.

- **Forecast-error variance decompositions (Gorodnichenko & Lee 2020)**: `fevd(formula, data; horizon, estimator = :r2 | :lpa | :lpb)` estimates the share of the forecast-error variance of `y_{t+h} − y_{t−1}` due to an observed shock, with delta-method standard errors with and without VAR(1) prewhitening (`LPFEVD`). The left-hand side is `cumul(v)` or `ldiff(y)`; plain regressors enter at date `t`, `lags(...)` form the information set, and `irf_formula` sets the parsimonious impulse-response regression of LP-A/B. `fevd(lp_result, data)` reuses an `lp` fit. `fevdbootstrap(f, data; method = :var | :block | :simulation)` applies the paper's three bias corrections (VAR residual bootstrap with an HQIC-selected VAR, Kilian–Kim block bootstrap for R², asymptotic simulation) with per-draw seeds and an optional `threaded` loop (`FEVDBootstrap`). `summarize` gives the paper's shifted-quantile band (MATLAB type-5 quantiles) or normal bands as an `FEVDSummary`; Plots recipes and `as_irf_result` cover both result types. Point estimates match the authors' MATLAB code to four decimals on the application data, now shipped as `docs/src/data/gorodnichenko_lee.csv` and `gorodnichenko_lee_rz.csv`; the deliberate deviations are listed in the new FEVD tutorial. With `method = :var`, a level that enters only through `lags(firstdiff(y), p)` next to its first difference (e.g. `cumul(dy)` with the paper's `lags(firstdiff(y), 1)`) is not simulated: each draw rebuilds it by cumulating the simulated difference, as the reference does, and linearly dependent VAR variables raise an `ArgumentError`. After an exact-fit horizon (the paper's monetary-policy specification at h = 0) the QR-prewhitened `se_pw`/`avar_pw`, and with them the default `:simulation` results, differ materially from the reference, whose normal-equation prewhitening is dominated by rounding error there. Internally `_var_irf` is now built on a new `_var_ma` (unchanged results), and `varbootstrap` shares the draw loop of `fevdbootstrap` and no longer swallows an `InterruptException` inside a draw.

- **State-dependent local projections (Ramey & Zubairy 2018)**: `lp` and `lpiv` accept `state = :col` (with `statelag = 1` and `regimes = ("a", "b")`). Every regressor, the intercept included — and, in `lpiv`, every endogenous regressor and instrument — is split by the lagged state, so both regimes are estimated jointly; coefficient names get a regime suffix (`x_s0`, `x_s1`). A weight in `[0, 1]` (Auerbach–Gorodnichenko) is accepted as well as an indicator. New `statetest(lp, estimator)` tests equality of the two regimes horizon by horizon (it uses the covariance between the regime coefficients), and `weakivtest(lpiv, h; regime = ...)` runs the Montiel Olea–Pflueger test on one regime. `biascorrect` and `varbootstrap` reject state-dependent results. Verified against `ivreg2` on the Ramey–Zubairy data to ~1e-10 (`stata_test/rz_vs_lp.md`).

- **`xtickstep` kwarg for Makie extension**: The `IRFPlotMakie` recipe now supports `xtickstep` (x-axis tick spacing).

- **`irfplot_axis` applies `xtickstep`**: The convenience wrapper now sets x-axis ticks and limits based on the `xtickstep` keyword argument.

### Removed

- **`irf_scale` and `flipshock` from plotting**: The Makie extension and RecipesBase recipes no longer accept `irf_scale` or `flipshock`. Use `rescale` from MacroEconometricTools.jl to scale results before plotting.

### Changed

- **`irfplot!` now imported from MacroEconometricTools.jl**: The `irfplot!` function is no longer defined locally — it is imported from MacroEconometricTools.jl (same as `irfplot`). This avoids export collisions when both packages are loaded together. The LP Makie extension adds methods to the shared function.

### Refactoring

- **`lags()` / `LagTerm` moved to Regress.jl**: Removed the local definition (~60 lines) and now imports `lags` and `LagTerm` from Regress.jl. Both packages share the same implementation.

- **`first_stage` extends `Regress.first_stage`**: Methods for `LocalProjectionIV` now extend the Regress.jl function instead of defining a separate `LocalProjections.first_stage`.

- **`weakivtest` extends `Regress.weakivtest`**: Methods for `LocalProjectionIV` now extend the Regress.jl function instead of defining a separate `LocalProjections.weakivtest`.
