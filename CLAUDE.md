# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this branch is

`port-0.2` in the fork [andrerecio/LocalProjections.jl](https://github.com/andrerecio/LocalProjections.jl) (`origin`). It starts from [gragusa/LocalProjections.jl](https://github.com/gragusa/LocalProjections.jl) 0.2.0 (`upstream`, merge `a08af5f`, 2026-10-21) and re-applies, one PR per piece, the fork's extensions developed on `dev` against upstream 0.1. Each piece is rewritten for the 0.2 API, where the covariance estimator is stored on the result.

**Never open a pull request against gragusa's repository.** `gh pr create` on a fork defaults to the parent repo, so always pass `--repo andrerecio/LocalProjections.jl` (and `--base port-0.2` for port PRs).

Port order and status (the `dev` branch is the reference implementation of each piece):

1. `ldiff`/`hbr`/`firstdiff` and horizon-tracking right-hand-side terms (done)
2. EWC with Student-t_B critical values, `ewc_bandwidth` (guide §2, EWC-1)
3. `sample = :common` is **dropped**: upstream's `sample = :fixed` is the same estimator (names `:varying`/`:fixed`)
4. State dependence (`state`/`statelag`/`regimes`, `statetest`)
5. VAR core and `lagselect`
6. `varbootstrap` (guide §4)
7. Herbst–Johannsen `biascorrect` (guide §3)
8. Gorodnichenko–Lee `fevd`/`fevdbootstrap`
9. Plots for the new result types, docs and tutorials (the upstream docs build already fails on missing docstrings and broken `@ref`s)

New features of upstream 0.2 that a ported procedure does not support (panel `unit`/`time`, `fe()`, LIML/Fuller/k-class) must throw a clear `ArgumentError`, not run silently.

`docs/src/inference_procedures_guide.md` (on `dev`) is the mathematical guide for pieces 2, 6 and 7; its §5 checklist and §6 audit are authoritative. `docs/src/code/` (MATLAB replication suites) is untracked, read-only reference material.

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

- `dev/Project.toml` holds the dev tools (Runic, TestEnv, StableRNGs, Revise), deliberately not in the package `[deps]`: Aqua's `stale_deps` check fails on unused dependencies. `test_pkg/` is upstream's personal environment with local paths; do not use it.
- `Regress`, `MacroEconometricTools` are unregistered git packages pinned to `rev = "main"` in `[sources]`; instantiation needs network and can break when their `main` moves.
- Aqua runs with type piracy checks **on** and `test/ExplicitImports.jl` checks every import: import names explicitly, and only extend StatsModels/Regress functions for types or functions this package owns.

## Architecture

`src/LocalProjections.jl` is upstream's single file, kept as close to upstream as possible so that future upstream changes merge cleanly. Fork code goes into new `include`d files, with only small hooks in the main file:

- `src/longdiff.jl`: `ldiff`/`hbr`/`firstdiff` (`LongDiffTerm`, `DiffTerm`), their StatsModels methods, `_create_long_difference`, and `_check_cumul_inner` (called from upstream's `cumul` `apply_schema`).
- `src/horizons.jl`: the per-horizon response by dispatch on the left-hand-side term (`_lhs_at`, `_lhs_response`, `_lhs_tautological`), replacing upstream's `_unwrap_lhs`/`_build_lhs_for_horizon` flag tuple; and right-hand-side horizon tracking (`_has_dynamic_horizon`, `_specialize_horizon`, `_rhs_tracks_horizon`).

Both are included after the pipe-operator section and before `AbstractLocalProjection`. A new response transform is a term type plus `_lhs_at`/`_lhs_response`/`_lhs_tautological` methods (and `_specialize_horizon`/`_has_dynamic_horizon`/`_tracks_horizon` if it may appear on the right).

Estimation pipeline (shared by `lp` and `lpiv`): `_lp_design` parses the formula, orders panel rows (`_panel_order`), converts `missing` to `NaN` without dropping rows, and returns per-horizon design matrices `Xs`/`Zs` (one shared pair unless a regressor tracks the horizon) plus `tautological_h0`. `_lp_estimate_horizons` builds the response per horizon with `_lhs_at`, masks incomplete rows per horizon (`:varying`) or on their intersection (`:fixed`), and fits `Regress.ols`/`Regress.iv(estimator, ...)`. Panels: every model column is computed per unit block (`_model_columns`), so leads, lags, cumulations and long differences never cross units.

Inference: the result stores `vcov_estimator`; `result + vcov(est)` re-applies it per horizon (clusters and Driscoll–Kraay named by column are rebuilt on each horizon's rows). `stderror(lp; term)`, `summarize`, `as_irf_result` and the plots all go through `_irf_result`, which currently uses **normal** critical values for every estimator (EWC-1).

## Known issues

- **Regress 0.3 conventions (verified 2026-10-23):** the finite-sample factor `n/(n−K)` is applied to kernel HAC, EWC, VARHAC and smoothers on OLS *and* IV models; `vcov(est, m; dofadjust = false)` removes it on both. REG-1 and REG-2 of the 0.1-era fork are resolved. Consequence: `lpiv` kernel HAC equals Stata `ivlpirf`/`ivreg2` (unadjusted) × √(n/(n−k)), and EWC equals `harreg` × √(n/(n−k)).
- **`m + vcov(est)` cannot carry `dofadjust`** (`VcovSpec` holds only the estimator), so the no-factor `harreg` convention cannot be stored on a result. Draft upstream issue 7 in the fork's local `upstream_issue_drafts.md`.
- **EWC-1 (methodological):** bands use normal quantiles even for EWC; canonical EWC inference pairs it with Student-t_B. Switching is an explicit methodological change (port step 2).
- **LP-1 is fixed upstream:** 0.2 never drops rows before building leads and lags.
