# Changelog

## Unreleased

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
