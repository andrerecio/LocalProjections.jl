# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A fork of [gragusa/LocalProjections.jl](https://github.com/gragusa/LocalProjections.jl) (remote `origin` is `andrerecio/LocalProjections.jl`, `upstream` is gragusa). The package estimates local projection impulse response functions via horizon-specific linear regressions with a formula DSL.

**Purpose of the fork:** extend the inference procedures beyond the current HAC/HC options, following `docs/src/inference_procedures_guide.md` — a mathematical guide covering (1) equal-weighted-cosine (EWC) HAR inference with fixed-smoothing Student-t_B critical values, (2) the Herbst–Johannsen local projection bias correction, and (3) the VAR residual moving-block bootstrap with Hall percentile-t bands. Read that guide before touching inference code; its §5 "safe adaptation checklist" and §6 audit are authoritative.

`docs/src/code/lp_var_nberma-main/` is the untracked third-party MATLAB replication suite for Montiel Olea, Plagborg-Møller, Qian & Wolf (2025) — the reference implementation the guide maps to. Treat it as read-only reference material: never format, refactor, or commit it.

## Commands

```bash
# Instantiate / resolve the environment (needs network — see note below)
julia --project -e 'using Pkg; Pkg.instantiate()'

# Full test suite (TestItemRunner + Aqua)
julia --project -e 'using Pkg; Pkg.test()'

# Run a subset of test items by tag (tags: :cumul :leads :anchor :nested :summarize
# :vcov :api :lpiv :iv :weakiv :lags :tautological :verification :core :met :smoke
# :ewc :biascorr).
# There is no test/Project.toml (test deps live in [extras]), so use TestEnv.jl
# (installed in the global environment) to activate the test target:
julia --project -e 'using TestEnv; TestEnv.activate(); using TestItemRunner; @run_package_tests filter=ti->(:vcov in ti.tags)'

# Format (required before pushing — CI auto-commits sciml formatting on every push)
# Dev tools live in the local `dev` environment, NOT the global one.
julia --project=dev dev/format.jl

# Docs (deploydocs is disabled; builds locally only)
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl

# Benchmarks (AirspeedVelocity; also runs on PRs via CI)
julia --project=benchmark benchmark/run_asv.jl

# Monte Carlo coverage study for varbootstrap (validation, run by hand; ~3 min)
julia --project=benchmark -t 4 benchmark/coverage_study.jl        # 300 reps, 500 draws
julia --project=benchmark -t 4 benchmark/coverage_study.jl 40 150 # quick smoke run
```

- Two dependencies are **unregistered git packages** pinned in `[sources]` of Project.toml: `Regress.jl` and `MacroEconometricTools.jl`, both `rev = "main"` from gragusa's GitHub. `Pkg.instantiate` needs network access to them, and builds can break when their `main` moves.
- Compat requires Julia ≥ 1.11. PrettyTables v3 and Makie 0.24.8 APIs are pinned and API-sensitive.
- `test/Aqua.jl` deliberately sets `piracies = false` (the package overloads StatsModels `apply_schema`/`termvars` for its custom terms).
- **Dev tooling lives in `dev/Project.toml`** (JuliaFormatter, StableRNGs, TestEnv, Revise), deliberately *not* in the package `[deps]`: `test/Aqua.jl` sets `stale_deps = true`, so any dependency unused in `src/` fails the suite. To get `StableRNGs` (or the others) in a REPL running on the main project, stack the environment rather than adding a dep: `push!(LOAD_PATH, "dev")`.
- A Kaimon.jl MCP server may be configured for this machine: when the `mcp__kaimon__*` tools are available, prefer evaluating Julia code through the live Kaimon session (persistent REPL, no TTFX) over spawning `julia` processes for exploratory work and single tests.

## Architecture

The entire package is **one file**, `src/LocalProjections.jl` (~3000 lines), with no `include`s; it is organized by banner comments in this order: formula-term types (`CumulTerm`, `LeadTerm`, `AnchorTerm`, `|` pipe) → `LocalProjection` types → data transforms → `lp()` → `stderror`/`IRFSummary` → IV block (`lpiv`, first-stage/weak-IV) → shared `coefpath`/`vcov`/`summarize` → Herbst–Johannsen bias correction → reduced-form VAR and lag selection → VAR moving-block bootstrap → critical values → plotting (RecipesBase) → `as_irf_result` bridge. Navigate by section, not by file.

- **Formula DSL:** the LHS must be `leads(y)` (y_{t+h}), `cumul(y)` (cumulative), or `anchor(y, z)` (y_{t+h} − z_t; `|` is pipe sugar, e.g. `leads(y)|z`); a plain LHS throws. RHS `lags(x, n)` and the OLS/IV engines come from Regress.jl. `lp`/`lpiv` build the RHS design matrix once, then loop `h = 0:horizon` rebuilding only the LHS and calling `Regress.ols` (or IV) per horizon. `missing` values are converted to `NaN` sentinels for type stability and rows are masked with `isnan` per horizon.
- **Key types:** `LocalProjection` / `LocalProjectionIV` (holding a `Vector` of Regress models plus metadata; union alias `LPResult` for shared dispatch), `LocalProjectionCovariance` (stores only the per-term variance **diagonals** per horizon, not full matrices), `IRFSummary`. When `response === shock`, `tautological_h0` forces the h=0 coefficient to 1 and its variance to 0.
- **Inference** is delegated to **CovarianceMatrices.jl** through Regress.jl, with two equivalent idioms: `vcov(estimator, lp_result)` or `lp_result + vcov(Bartlett{NeweyWest}())`. `summarize(lp, cov; term, level)` builds confidence bands (currently always normal critical values — see EWC-1 below).
- **Lag selection & bootstrap:** `lagselect` ports `ic_var.m` (AIC/BIC on the VAR system, penalty denominator fixed at `T - p_max` but `Σ̂ₚ` from the full sample); `varbootstrap` ports the §4 procedure — `_var_ols` (`var_estim.m`), `_pope_biascorrect` (`var_biascorr.m`, validated against Pope's AR(1) closed form `b = 1 + 3ρ`), `_var_impact`/`_var_irf` (`var_ir_estim.m`/`var_ir.m`), `_position_means`/`_mbb_residuals!` (`var_boot.m`), `_var_simulate` (`var_sim.m`), and `_boot_interval` (`boot_ci.m`). Note `MacroEconometricTools` has **no** Pope correction, no Lyapunov solver and no lag-selection driver, and its `bootstrap_irf_block` recenters with a *stride-ℓ* mean rather than the sliding-window (Brüggemann–Jentsch–Trenkler) mean the reference requires — do not substitute it.
- **Re-exports:** several exported names are not defined here — `lags`, `first_stage`, `weakivtest`, `WeakIVTestResult`, `FirstStageIV` come from Regress.jl; `irfplot`, `irfplot!`, `LocalProjectionIRFResult` from MacroEconometricTools.jl; `lag`/`lead` from ShiftedArrays.
- **Plotting:** Plots.jl via `@recipe` (`plot(lp, cov; term, levels)`), and Makie via the weakdep extension `ext/LocalProjectionsMakieExt.jl` (`irfplot`, `irfplot_axis`). `as_irf_result` converts to MacroEconometricTools' `LocalProjectionIRFResult` (AxisArrays-based).

## Known issues (audited 2026-08-27, guide §6)

These are confirmed against the pinned trees and must be kept in mind when extending inference:

- **LP-1 (bug):** `lp`/`lpiv` call `dropmissing` on the base variables **before** constructing lags/leads, so internal `missing` values collapse the time axis and produce time-incorrect lags. Workaround: encode internal gaps as `NaN`, not `missing`. A proper fix must build lags/leads on the original time axis and mask incomplete rows afterward.
- **REG-1 (upstream bug in Regress.jl):** `vcov(est, model; dofadjust=false)` is accepted but ignored for HAC/EWC estimators; also `LocalProjections.vcov` does not forward keyword arguments to the per-horizon models.
- **Bootstrap scope:** `varbootstrap` supports OLS local projections with `leads` or `cumul` responses only; anchored responses and `lpiv` throw. `missing`/`NaN` in the VAR columns are rejected (the safe side of LP-1).
- **EWC-1 (methodological):** `summarize`, the plot recipes, and `as_irf_result` hard-code normal critical values; canonical EWC inference (Lazarus et al. 2018) pairs the EWC variance with Student-t_B critical values. Switching changes results — treat it as an explicit methodological change, never a silent refactor.
