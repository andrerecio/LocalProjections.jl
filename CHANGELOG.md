# Changelog

## Unreleased (andrerecio fork, port onto 0.2)

### Added

- **Long differences**: `ldiff(y)` (``y_{t+h} - y_{t-1}``, Piger & Stockwell 2025), `hbr(x, y)` (``(x_{t+h} - x_{t-1})/y_{t-1}``, Hall 2009, Barro & Redlick 2011; `hbr(x)` scales by `x` itself), and `cumul` of either, which sums the changes over horizons `0..h` with the baseline counted `h + 1` times. `firstdiff(x)` gives the lagged-difference controls `lags(firstdiff(y), p)`. A long difference on the left is never treated as a tautological `h = 0`. `cumul(ldiff(y, 3))` is an error; write `cumul(ldiff(y), 3)`.
- **Horizon-tracking regressors**: a bare `cumul(x)`, `leads(x)`, `ldiff(x)` or `hbr(x, y)` on the right-hand side (exogenous, endogenous or instrument) is rebuilt at every horizon, under one coefficient name, so the one-step cumulative multiplier is a single call, `lpiv(TSLS(), @formula(cumul(y) ~ (cumul(g) ~ z) + ...), df; horizon)`. In panels, these terms are built within each unit. An explicit horizon (`cumul(x, 3)`) pins the column.

### Fixed

- `anchor(y, lag(y))` (any anchor that is `missing` in its first rows) no longer throws when converting the anchored response; it equals `ldiff(y)`.

## 0.2.0

This release breaks the 0.1 API. No deprecation shims are provided.

### Added

- **Panel local projections**: `lp` and `lpiv` accept `unit` and `time` column names. Rows are sorted by unit and time, and leads, cumulative sums, anchors and lags are built within each unit. Repeated (unit, time) pairs, a unit that skips a period present elsewhere in the data, missing or NaN identifiers, and `unit` without `time` are errors. Panel results default to `CR1(unit)` standard errors.
- **Clustered and Driscoll-Kraay inference by column name**: `lp + vcov(CR1(:id))` and `lp + vcov(DriscollKraay(K; tis = :t, iis = :id))` build the estimator for each horizon from the rows that horizon uses.
- **Absorbed fixed effects**: `fe(column)` terms in `lp` and `lpiv` formulas, e.g. `leads(y) ~ x + fe(id) + fe(t)`, are absorbed by Regress at each horizon on that horizon's rows. The model has no intercept, and the absorbed levels enter the degrees of freedom and the clustered small-sample factor (a fixed effect nested in the cluster counts once). `fe` is exported. Fixed effects interacted with other terms or placed inside `(endogenous ~ instruments)` are errors.
- **Tables.jl input**: `lp` and `lpiv` take any Tables.jl table. String and categorical columns enter through StatsModels contrasts. `missing` in a numeric column excludes the row only at the horizons that need it.
- **`sample = :fixed`**: estimates every horizon on the rows available at all horizons. The default, `:varying`, keeps per-horizon samples.
- **`horizonmodel(lp, h)`**: returns the Regress model at horizon `h` in `0:lp.horizon`. Out-of-range horizons throw an `ArgumentError`. The model carries the formula's coefficient names, so `coefnames(horizonmodel(lp, h)) == coefnames(lp)` and `coeftable` prints them; first-stage models from `first_stage(lpiv, h)` are named by the exogenous regressors and instruments.
- **`nobs(lp)`** and **`stderror(lp; term)`**: per-horizon observation counts and standard errors, as methods on the StatsAPI generics.
- **`AbstractLocalProjection`**: exported supertype of `LocalProjection` and `LocalProjectionIV`.
- **`IRFSummary` is a Tables.jl table**, with `response`, `term`, `level` and `scale` columns, so `DataFrame(summary)` works without DataFrames as a dependency.

### Changed

- **`lpiv` takes the IV estimator first**: `lpiv(TSLS(), formula, data; horizon)`, mirroring `Regress.iv`. `TSLS()`, `LIML()`, `Fuller(a)` and `KClass(kappa)` are accepted and re-exported; each horizon is fitted by Regress's matrix `iv` for that estimator and matches the Regress formula fit.
- **Inference uses the stored estimator**: `summarize`, `as_irf_result` and the plotting functions use the estimator set by `lp + vcov(estimator)` (HC1 by default). Passing an estimator is the same as applying it with `+ vcov(estimator)` first.
- **One computation of IRF bands**: `summarize`, `as_irf_result` and the plots all build their numbers from one `LocalProjectionIRFResult`.
- **Plots are drawn by MacroEconometricTools**: `irfplot`, `irfplot!` and the Plots recipe hand the IRF object to MacroEconometricTools' Makie methods and Plots recipe. `irfplot!` draws onto any `Axis`.
- **Default shock is chosen by name**: the first endogenous variable, otherwise the first coefficient other than the intercept. Formulas without an intercept (`0 + x`) work.
- **`weakivtest(lpiv, 0)` throws** when the response is the endogenous shock; `weakivtest(lpiv)` returns `nothing` at horizon 0.
- **Per-horizon R² is centered when the formula has an intercept**: each horizon model, and each first-stage regression behind `first_stage`, is fitted with Regress's `has_intercept` set from the formula, so `r2` and `adjr2` equal those of the Regress formula fit. Formulas without an intercept keep the uncentered R². Coefficients, standard errors and first-stage F statistics are unchanged.
- **Pipe syntax `y|z`** becomes an `AnchorTerm` only inside `lp` and `lpiv`; other packages' formulas keep the StatsModels meaning of `|`.
- **`first_stage` and `weakivtest` extend the Regress.jl functions**, and `irfplot!` is the MacroEconometricTools function.
- **`lags()` / `LagTerm` come from Regress.jl.**
- **Requires CovarianceMatrices 0.33.1, Regress 0.3, MacroEconometricTools 0.2, and Julia 1.13.** Julia 1.13 is the floor that Regress and MacroEconometricTools set.

### Removed

- `LocalProjectionCovariance`, `vcov(estimator, lp)` and `stderror(cov; term)`. `vcov` and `stderror` are the StatsAPI functions.
- `coefnames(lp, h)`; `coefnames(lp)` covers all horizons.
- The Makie extension, the `IRFPlot` recipe and `irfplot_axis`.
- `irf_scale` and `flipshock` plotting keywords. Use `rescale` from MacroEconometricTools.jl before plotting.
- Exports of `FirstStageIV`, `WeakIVTestResult` (public in Regress) and `LocalProjectionIRFResult` (public in MacroEconometricTools). `CumulTerm`, `LeadTerm` and `AnchorTerm` are `public` but no longer exported.
