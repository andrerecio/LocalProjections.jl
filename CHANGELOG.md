# Changelog

## Unreleased

### Changed

- **Requires Regress.jl 0.2, CovarianceMatrices.jl 0.32 and Julia 1.12.** The new upstream versions fix two defects documented in `upstream_issue_drafts.md`. HAC estimators no longer carry kernel-weight state, so `weakivtest` on a formula-fitted IV model with `Bartlett{NeweyWest}()` no longer crashes (Issue 2). The IV path now excludes the intercept from automatic bandwidth selection, so `lp` and `lpiv` select the same Newey–West/Andrews bandwidth on identical data (Issue 4, REG-2b). Automatic-bandwidth `lpiv` standard errors change as a result; fixed-bandwidth kernel HAC, HR0/HR1 and EWC results are unchanged. HC2/HC3 on over-identified `lpiv` now use the correct IV leverage (Regress fixed the matrix-TSLS formula), so those standard errors change too.
- **`vcov(estimator, lp; kwargs...)` forwards keyword arguments** to each horizon's model. `vcov(Bartlett(6), lpiv_result; dofadjust = true)` adds the `n/(n−k)` factor that `lp` applies to kernel HAC, so `lp` and `lpiv` bands can be put on one convention. EWC stays unscaled, and on `lp` results `dofadjust = false` is still ignored upstream.

### Added

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
