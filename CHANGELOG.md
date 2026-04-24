# Changelog

## Unreleased

### Added

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
