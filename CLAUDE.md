# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this branch is

`port-0.2` in the fork [andrerecio/LocalProjections.jl](https://github.com/andrerecio/LocalProjections.jl) (`origin`). It starts from [gragusa/LocalProjections.jl](https://github.com/gragusa/LocalProjections.jl) 0.2.0 (`upstream`, merge `a08af5f`, 2026-10-21) and re-applies, one PR per piece, the fork's extensions developed on `dev` against upstream 0.1. Each piece is rewritten for the 0.2 API, where the covariance estimator is stored on the result.

**Never open a pull request against gragusa's repository.** `gh pr create` on a fork defaults to the parent repo, so always pass `--repo andrerecio/LocalProjections.jl` (and `--base port-0.2` for port PRs).

`port-0.2` is the fork's default branch since 2026-10-27, and CI (tests on Julia 1.13 and the latest release, Runic, benchmarks on pull requests) runs on pushes and pull requests to it. The old 0.1-based `main` is kept unchanged and tagged `v0.1-fork`.

Port order and status (the `dev` branch is the reference implementation of each piece):

1. `ldiff`/`hbr`/`firstdiff` and horizon-tracking right-hand-side terms (done)
2. EWC with Student-t_B critical values, `ewc_bandwidth` (guide §2, EWC-1) (done; `src/ewc.jl`, hooked into `_irf_result`)
3. `sample = :common` is **dropped**: upstream's `sample = :fixed` is the same estimator (names `:varying`/`:fixed`)
4. State dependence (`state`/`statelag`/`regimes`, `statetest`) (done; `src/state.jl`)
5. VAR core and `lagselect` (done; `src/var.jl`, Tables.jl input)
6. Herbst–Johannsen `biascorrect` (guide §3) (done; `src/biascorrect.jl`; swapped with the bootstrap, which calls it in every draw)
7. `varbootstrap` (guide §4) (done; `src/varbootstrap.jl`)
8. Gorodnichenko–Lee `fevd`/`fevdbootstrap` (done; `src/fevd_core.jl`, `fevd.jl`, `fevd_bootstrap.jl`, `fevd_summary.jl`, `fevd_plotting.jl`)
9. Plots for the new result types, docs and tutorials: (a) docs build fixed, API page complete, Plotting/FEVD tutorials and an EWC section in the manual (done); (b) Ramey–Zubairy tutorial with its four `docs/make_rz_*.jl` figure scripts and `docs/src/assets/` (done; the scripts read the CSV with a local `readclean` because the docs env has no CSV.jl, and the tutorial's `julia` blocks are not executed, so rerun them by hand after API changes); (c) README with the Piger–Stockwell figure (`docs/make_ps_figure.jl`) (done; README numbers are rechecked by running its blocks, the scratch checks are not in the repo); (d) the inference procedures guide (done: kept as the 0.1.0 audit, with a status note and dated updates; math in Documenter syntax, ` ```math ` and double-backtick inline, since `\[…\]` and `$…$` do not render); (e) `benchmark/coverage_study.jl` (done, with the `lagselect`/`varbootstrap` benchmarks; the default 300×500 run takes ~2.5 min on 4 threads and reproduces its docstring numbers). The `dev` Quarto site (`docs/quarto`) is not ported

New features of upstream 0.2 that a ported procedure does not support (panel `unit`/`time`, `fe()`, LIML/Fuller/k-class) must throw a clear `ArgumentError`, not run silently.

`docs/src/inference_procedures_guide.md` is the mathematical guide for pieces 2, 6 and 7; its §5 checklist and §6 audit are authoritative. `docs/src/code/` (MATLAB replication suites) is untracked, read-only reference material.

## Commands

Julia 1.13 is required (the floor of Regress 0.3 / MacroEconometricTools 0.2). The machine default may be 1.12, so call `julia +1.13` explicitly.

```bash
julia +1.13 --project -e 'using Pkg; Pkg.instantiate()'
julia +1.13 --project -e 'using Pkg; Pkg.test()'

# Subset of test items (TestEnv lives in the local dev environment)
julia +1.13 --project -e 'push!(LOAD_PATH, "dev"); using TestEnv; TestEnv.activate(); using TestItemRunner; @run_package_tests filter=ti->(:ldiff in ti.tags)'

# Format with Runic (CI checks it; there is also a pre-commit hook)
julia +1.13 --project=dev -m Runic --inplace src test

# Docs (local build only)
julia +1.13 --project=docs docs/make.jl
```

`docs/build/` is tracked upstream, so a local docs build rewrites it: don't commit it (`git stash push -- docs/build`), or build from a copy of `make.jl` with `build = <scratch dir>`. In this git worktree Documenter cannot find the remote, so that copy also needs `remotes = nothing` and `root = "<worktree>/docs"`. After a dependency change, `Pkg.resolve()` the `docs` environment (its Manifest is local).

- `dev/Project.toml` holds the dev tools (Runic, TestEnv, StableRNGs, Revise), deliberately not in the package `[deps]`: Aqua's `stale_deps` check fails on unused dependencies. `test_pkg/` is upstream's personal environment with local paths; do not use it.
- `Regress`, `MacroEconometricTools` are unregistered git packages pinned to `rev = "main"` in `[sources]`; instantiation needs network and can break when their `main` moves.
  The code needs Regress `main` at or after `c7957a01` (Regress.jl pull request 24, "Unify paths", 2026-10-24; the version is still 0.3.0, so compat cannot express it): `_regime_iv_model` reads `PostEstimationDataIVMatrix.endo_cols`. That PR also made matrix fits drop collinear columns by sweep in formula order and recategorize endogenous regressors the instruments explain as exogenous (so `weakivtest` throws where an instrument equals its regressor). After it moves, `Pkg.update("Regress")` in the package **and** the `docs` environment (separate manifests) and rerun the suite.
- Aqua runs with type piracy checks **on** and `test/ExplicitImports.jl` checks every import: import names explicitly, and only extend StatsModels/Regress functions for types or functions this package owns.

## Architecture

`src/LocalProjections.jl` is upstream's single file, kept as close to upstream as possible so that future upstream changes merge cleanly. Fork code goes into new `include`d files, with only small hooks in the main file:

- `src/longdiff.jl`: `ldiff`/`hbr`/`firstdiff` (`LongDiffTerm`, `DiffTerm`), their StatsModels methods, `_create_long_difference`, and `_check_cumul_inner` (called from upstream's `cumul` `apply_schema`).
- `src/horizons.jl`: the per-horizon response by dispatch on the left-hand-side term (`_lhs_at`, `_lhs_response`, `_lhs_tautological`), replacing upstream's `_unwrap_lhs`/`_build_lhs_for_horizon` flag tuple; and right-hand-side horizon tracking (`_has_dynamic_horizon`, `_specialize_horizon`, `_rhs_tracks_horizon`).

Both are included after the pipe-operator section and before `AbstractLocalProjection`.

- `src/ewc.jl` (included before the shared `coefpath`/`stderror` methods): `_critical_distribution`/`_critical_value`, `ewc_bandwidth`.
- `src/var.jl` (included after `ewc.jl`; no result types involved): `_var_data` (any Tables.jl table, rejects `missing`/`NaN`), `_var_ols` (`var_estim.m`), `lagselect`/`VARLagSelection` (`ic_var.m`), `_pope_biascorrect` (`var_biascorr.m`, validated against Pope's AR(1) closed form `b = 1 + 3ρ`), `_var_impact`/`_var_ma`/`_var_irf` (`var_ir_estim.m`/`var_ir.m`), `_var_simulate` (`var_sim.m`), `_var_fevd`/`_check_var_rank` (FEVD). MacroEconometricTools has no Pope correction, no Lyapunov solver and no lag-selection driver, and its `bootstrap_irf_block` recenters with a stride-ℓ mean rather than the sliding-window mean the reference requires: do not substitute it.
- `src/biascorrect.jl` (included after `var.jl`): `BiasCorrectedLP <: AbstractLocalProjection` forwards properties to the wrapped `LocalProjection`, so every shared method (`stderror`, `summarize`, `as_irf_result`, plots, `ewc_bandwidth`, `nobs`) works unchanged; only `coefpath` (the corrected path) and `+` (keeps the correction) are its own. `_irf_result` adds `bias_corrected = true` to the metadata. `_lhs_kind` (raw formula) lives in `horizons.jl`.
- `src/varbootstrap.jl` (included after `biascorrect.jl`): `_run_draws!` (shared with the FEVD bootstrap), `_position_means`/`_mbb_residuals!` (`var_boot.m`), `varbootstrap`/`LPBootstrap`, `_boot_interval` (`boot_ci.m`). Each draw rebuilds the data as a `NamedTuple` of the simulated `vars` and calls `lp` with the original formula and `sample`; standard errors are always HC1 (the reference), not the stored estimator. `_irf_result(b::LPBootstrap, levels, method)` is the single source of `as_irf_result`, `irfplot`/`irfplot!` and the Plots recipe.
- FEVD (Gorodnichenko & Lee 2020), included after `varbootstrap.jl` in the order `fevd_core.jl` (`_gl_*` QR matrix core) → `fevd.jl` (`_GLSpec`/`_GLDesign`, `LPFEVD`, `fevd`) → `fevd_bootstrap.jl` → `fevd_summary.jl` (`FEVDSummary`) → `fevd_plotting.jl` (`as_irf_result`, Makie `irfplot`, the Plots recipes). It ports `SubR2.m`/`SubLPA.m`/`SubLPB.m` and `SimulationAVar*.m` (MATLAB tree `docs/src/code/goro_lee_r2/`, gitignored, read-only), reusing the formula front end (`apply_schema`, `modelcols`, `_lhs_at`) on a gap-free column table but not `lp`/Regress for estimation, so the reference conventions hold (own window `t0..T−h` per horizon, no dof factor in the Newey–West, full-sample shock moments). Every public entry point converts `data` once with `_gl_table`; helpers take a `NamedTuple`. QR vs the reference's normal equations is rounding-level **except** after an exact-fit horizon (MP case, h = 0): the reference's prewhitened `se_pw` is then rounding-dominated at every later h and the package's differs materially (by design; tested in `testitems_fevd_octave.jl`). Bands use MATLAB type-5 quantiles (`_gl_quantile5`), never clamped. Deviations from the MATLAB code are tabulated in `docs/src/tutorials/fevd.md`. Data: `docs/src/data/gorodnichenko_lee.csv` (160×, Sec. 5) and `gorodnichenko_lee_rz.csv` (384×, App. H2), provenance in `gorodnichenko_lee_README.md`. The Octave fixture CSVs in `test/fixtures/goro_lee/` are gitignored (copy them from the main checkout); `test/runtests.jl` skips the `:octave` items without them, so a custom TestItemRunner filter must add `!(:octave in ti.tags)`.
- `src/state.jl` (included right after `abstract type AbstractLocalProjection`, because `LPState` is a field type of both result structs): `LPState`, `_state_weights` (lag within each panel unit), `_split_by_state`, `_resolve_state_shock`, `_shock_terms`, `statetest`/`StateTestResult` (a Tables.jl table), and the regime-specific `weakivtest` model (`_weakiv_model`, `_regime_iv_model`). Hooks in the main file: a `state` field at the end of `LocalProjection`/`LocalProjectionIV` (pass it on in every constructor call, including the `+` methods), the regime split inside `_lp_design`'s `design(h)`, `_shock_terms` in `coefpath`, and `regime` in `weakivtest`. `lp.shock` is the first-regime name, `lp.state.shock` the unsplit one; `has_intercept` stays true after the split (the regime intercepts span the constant). A new response transform is a term type plus `_lhs_at`/`_lhs_response`/`_lhs_tautological` methods (and `_specialize_horizon`/`_has_dynamic_horizon`/`_tracks_horizon` if it may appear on the right).

Estimation pipeline (shared by `lp` and `lpiv`): `_lp_design` parses the formula, orders panel rows (`_panel_order`), converts `missing` to `NaN` without dropping rows, and returns per-horizon design matrices `Xs`/`Zs` (one shared pair unless a regressor tracks the horizon) plus `tautological_h0`. `_lp_estimate_horizons` builds the response per horizon with `_lhs_at`, masks incomplete rows per horizon (`:varying`) or on their intersection (`:fixed`), and fits `Regress.ols`/`Regress.iv(estimator, ...)`. Panels: every model column is computed per unit block (`_model_columns`), so leads, lags, cumulations and long differences never cross units.

Inference: the result stores `vcov_estimator`; `result + vcov(est)` re-applies it per horizon (clusters and Driscoll–Kraay named by column are rebuilt on each horizon's rows). `summarize`, `as_irf_result` and the plots all build bands in `_irf_result`, which takes the critical value from `_critical_value(_vcov_estimator(lp), level)` (`src/ewc.jl`): Student-t_B for `EWC(B)`, normal otherwise. A new estimator with its own reference distribution is one `_critical_distribution` method.

## Known issues

- **Regress 0.3 conventions (verified 2026-10-23):** the finite-sample factor `n/(n−K)` is applied to kernel HAC, EWC, VARHAC and smoothers on OLS *and* IV models; `vcov(est, m; dofadjust = false)` removes it on both. REG-1 and REG-2 of the 0.1-era fork are resolved. Consequence: `lpiv` kernel HAC equals Stata `ivlpirf`/`ivreg2` (unadjusted) × √(n/(n−k)), and EWC equals `harreg` × √(n/(n−k)).
- **`m + vcov(est)` cannot carry `dofadjust`** (`VcovSpec` holds only the estimator), so the no-factor `harreg` convention cannot be stored on a result. Draft upstream issue 7 in the fork's local `upstream_issue_drafts.md`.
- **EWC-1 resolved (port step 2):** EWC bands use Student-t_B critical values; upstream 0.2 used the normal for every estimator. The EWC variance keeps Regress's `n/(n−K)` (user decision, 2026-10-23), so it is `harreg` × √(n/(n−k)) with the same critical values (tested at h = 8 on the Ramey–Zubairy data).
- **`weakivtest` with EWC:** Regress 0.3 throws an `ArgumentError` for EWC (and HC2–HC5, CR2/CR3), where 0.1 silently fell back to an HC0-style weight; the fork's warning is no longer needed.
- **LP-1 is fixed upstream:** 0.2 never drops rows before building leads and lags.
