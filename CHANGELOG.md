# Changelog

## Unreleased

### Refactoring

- **`lags()` / `LagTerm` moved to Regress.jl**: Removed the local definition (~60 lines) and now imports `lags` and `LagTerm` from Regress.jl. Both packages share the same implementation.

- **`first_stage` extends `Regress.first_stage`**: Methods for `LocalProjectionIV` now extend the Regress.jl function instead of defining a separate `LocalProjections.first_stage`.

- **`weakivtest` extends `Regress.weakivtest`**: Methods for `LocalProjectionIV` now extend the Regress.jl function instead of defining a separate `LocalProjections.weakivtest`.
