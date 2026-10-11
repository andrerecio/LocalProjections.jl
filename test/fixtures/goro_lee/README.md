# Gorodnichenko–Lee (2020) reference fixtures

These fixtures come from running the authors' MATLAB code, under GNU Octave 11.3, for the LP-FEVD port in LocalProjections.jl.

- **Reference code.** The code lives in `docs/src/code/goro_lee_r2/` and `lib/`. That tree is gitignored and read-only, so regenerating any fixture requires a local copy of it plus Octave 11.3.
- **Shims.** The shims in `octave_shims/` replace the MATLAB toolbox functions that Octave lacks.
- **Drivers.** The driver scripts are in `drivers/`. Four of them (`sim_avar_r2_det.m`, `sim_avar_lpA_det.m`, `sim_avar_lpB_det.m`, `ic_block.m`) contain lines copied verbatim from the third-party reference code, so they are **not committed** (`drivers/.gitignore`). Generate them with `sh test/fixtures/goro_lee/drivers/gen_verbatim_drivers.sh` before running the synthetic-fixture driver. The application drivers do not use them.
- **Local only.** The CSV fixtures (~6.9 MB) are **not committed** (`.gitignore` in this directory): only the drivers, shims and READMEs are. Generate them with the commands below. Without them `test/runtests.jl` skips the items of `test/testitems_fevd_octave.jl` (tag `:octave`; a custom test filter must exclude that tag), and CI checks the key application values pinned from these fixtures in `test/testitems_fevd_checks.jl` ("application values pinned from the authors' MATLAB code").
- **Coverage.** When present, every fixture file in this directory is read by `test/testitems_fevd_octave.jl` (or, for `app_data_*.csv`, checked against `docs/src/data`). The drivers can write a superset; only the files listed in this README and in `synthetic/README.md` are used.

## Synthetic fixtures (`synthetic/`)

These are deterministic, synthetic-data fixtures (T = 200, N = 3) for the literal-port tests. They cover:
- `SubR2`;
- `SubLPA`/`SubLPB`, with both the shared and the separate LP set;
- the deterministic prefixes of `SimulationAVar{R2,LPA,LPB}`, i.e. V_h and the delta-method s.e. at ŝ;
- `lrv_nw` and its pre-whitened variant;
- the bandwidth rule;
- `var_estimation`/`var_chol`, the FEVD of a 3-variable VAR;
- the lag-selection IC block of `BootstrapVARR2`;
- `var_bootstrap`, with the recorded `idx`/`init` draws.

To regenerate them, first generate the verbatim-copy drivers, then run the main script:

```
sh test/fixtures/goro_lee/drivers/gen_verbatim_drivers.sh
octave --no-gui test/fixtures/goro_lee/drivers/make_synthetic_fixtures.m
```

See `synthetic/README.md` for every file, call and shape. Re-running both (2026-10-10, Octave 11.3) reproduced every fixture bit for bit, as did `app_deterministic.m` and `app_stochastic.m 1 2000`.

## Application fixtures (`app_*.csv`)

These fixtures cover the Section 5 application (TFP and monetary-policy shocks, T = 160, 1969Q1–2008Q4) and the Appendix H2 application (Ramey–Zubairy military news, T = 384, 1920Q1–2015Q4). The inputs are built exactly as `AppSimulation.m` and `AppSimulationRZ.m` build them.

`AppData.m` and `AppDataRZ.m` hold the data as inline matrices, so no `xlsread` and no spreadsheet is involved.

**Regenerating.** Run these from the repo root:

```
octave --no-gui test/fixtures/goro_lee/drivers/app_deterministic.m     # < 1 s
octave --no-gui test/fixtures/goro_lee/drivers/app_stochastic.m 1 2000 # ~ 15 s
octave --no-gui test/fixtures/goro_lee/drivers/app_stochastic.m 2 2000
```

**Drivers:**
- `app_setup.m` adds the paths.
- `app_cases.m` defines the inputs and the per-case arguments.
- `app_varic.m` is a hand transcription of the IC / VAR-DGP block, §2–§3.1, of `BootstrapVARR2.m`, with the comments stripped. It is not a `sed` extract. The statements are the reference ones, apart from three changes: only the multi-candidate IC branch is kept, the unused `InvXVAROrder` block is dropped, and `var_chol` gets an MC count of 2000 instead of `NBootstrap`, which has no effect because `MCIndicator = 0`. The file header lists the line ranges it covers.
- `app_write_csv.m` writes every value as `%.17g` and adds a one-line header.

**Octave shims** (`octave_shims/`): `lagmatrix`, `cholcov`, `mvnrnd` and `norminv`.
- `lagmatrix` follows MATLAB semantics. Columns are grouped by lag in the order given, so `lagmatrix(Y, 1:p)` is lag-major, as `var_estimation.m` assumes.
- `mvnrnd` turns a column `mu` into a row, as MATLAB does.
- Octave's built-in `quantile` already defaults to Hyndman–Fan type 5, which is MATLAB's method. This was checked against a hand computation, so it is not shimmed.

### Case arguments

All cases use `lagy = 0`, `lagX = 4` for every column of X, `trend = 0`, `horizon = 20`, `dfIndicator = 0`, `lagVARSet = 1:10` and `ICIndicator = 1` (HQIC).

| case | y | X columns (col 1 = shock) | ylevel | XVAROrder | yPlace | XtIndicator | lagyLP | lagXLP |
|---|---|---|---|---|---|---|---|---|
| `tfp_y` | cumsum(Δy) | tfp, dy, pi, rr, effr | 0 | [1,3,4,5] | 2 | [1,0,0,0,0] | 1 | [1,0,0,0,0] |
| `tfp_pi` | π | tfp, dy, pi, rr, effr | 1 | [1,2,4,5] | 3 | [1,0,0,0,0] | 1 | [1,0,0,0,0] |
| `mp_y` | cumsum(Δy) | rr, tfp, dy, pi, effr | 0 | [2,4,1,5] | 2 | [1,1,1,1,0] | 0 | [0,0,0,0,0] |
| `mp_pi` | π | rr, tfp, dy, pi, effr | 1 | [2,3,1,5] | 3 | [1,1,1,1,0] | 0 | [0,0,0,0,0] |
| `rz_y` | cumsum(Δy) | news, dy, pi, tbill | 0 | [1,3,4] | 2 | [1,0,0,0] | 1 | [1,0,0,0] |
| `rz_pi` | π | news, dy, pi, tbill | 1 | [1,2,4] | 3 | [1,0,0,0] | 1 | [1,0,0,0] |

The XVAR is `[TFP, Δy, π, RR, EFFR]` in all four Section 5 cases and `[news, Δy, π, tbill]` in H2. With `ylevel = 0` it uses rows 2:T (Q34).

### Deterministic files

Each output depends only on the data. All of them agree with the clean-room values in SPEC §6.C to ≤ 1.5e-13 absolute.

| file | content | shape |
|---|---|---|
| `app_data_sec5.csv` | `data` of `AppData.m`: time, dtfp_util, rr_shock, dgdp, dcons, dinv, effr, infl | 160×8 |
| `app_data_rz.csv` | `macro_quarter` of `AppDataRZ.m`: time, news, dgdp, infl, unemp, tbill | 384×6 |
| `app_<case>_inputs.csv` | the exact `y` (col 1) and `X` (cols 2..) passed to the estimators | T×(1+N) |
| `app_sec5_var4_chol.csv` | `var_chol(timeseries, 4, 20, ·, s, 0, 2, 0, 0)` for s = 1 (TFP) and s = 4 (MP). Columns `psi_s{s}_v{v}` are one-s.d. IRFs (row 2, Δy, is cumulated); columns `vd_s{s}_v{v}` are FEVDs | 21×(1+20) |
| `app_rz_var4_chol.csv` | same as above, for H2 with s = 1 (news) | 21×(1+8) |
| `app_<case>_r2.csv` | `SubR2(y,X,0,lagX,20,ylevel,0,1)`: h, t0, tend, Th, vd, se_pw (`seAVarWPW`), se_nopw (`seAVarWoPW`) | 21×7 |
| `app_tfp_y_r2_theta.csv` | the `theta{h}` of SubR2, long format (h, i, theta), i = 1..2h+3 | 483×3 |
| `app_tfp_y_r2_gogt.csv` | `GOGTWPW{h}` and `GOGTWoPW{h}` of SubR2, long format (h, i, j, gogt_pw, gogt_nopw), column-major | Σ(2h+3)²×5 |
| `app_tfp_y_r2_fe.csv` | the `forecastError` of SubR2, column h = FE(t+h\|t−1), NaN outside t0..T−h | 160×21 |
| `app_<case>_lp{a,b}_varset.csv` | `SubLP{A,B}(y,X,0,lagX,XtInd,20,ylevel,0,0,1,0)`, the set used in `BootstrapVARLP*`: h, t0, tend, t0lp, ir (unit shock), irse (`irSeAVar`), vd, mse, [sigma2v (B)], meanx, varx | 21×10/11 |
| `app_<case>_lp{a,b}_lpset.csv` | the same call with `XLP = X, lagyLP, lagXLP, XtIndicatorLP = XtInd`, the parsimonious set used in `SimulationAVarLP*` | 21×10/11 |
| `app_<case>_svar.csv` | the bootstrap-DGP FEVD `vdVAR` (pseudo-truth s_VAR) at the HQIC lag: h, lagvar, svar | 21×3 |
| `app_ic.csv` | the HQIC of `BootstrapVARR2` for p = 1..10, per case (1..6 = tfp_y, tfp_pi, mp_y, mp_pi, rz_y, rz_pi); columns case, p, xvar_rows, T_IC, x_location, y_location, hqic, selected | 60×8 |

Selected HQIC lags: p = 2 for all four Section 5 cases (T_IC = 149 for y, 150 for π) and p = 3 for H2 (T_IC = 373 / 374). The first effective row is t0 = 5 everywhere.

### Stochastic files (`app_stoch_<proc>_tfp_y_seed{1,2}.csv`)

These cover TFP → y only. The paper settings are NBootstrap = 2000, TBurnIn = 100, quantiles [0.05, 0.95], seIndicator = 0, cbIndicator = 0, boundIndicator = 1 and jointIndicator = 0.

- **Seeding.** Before each procedure call the driver runs `rand('state',SEED); randn('state',SEED)`, using Octave's Mersenne twister. MATLAB streams differ, so Julia can match these files only up to Monte Carlo error. Use the seed1/seed2 spread to calibrate tolerances.
- **Run time.** One seed takes about 13 s in total (`var_lpb` 7.4 s, `var_r2` 4.3 s, `block_r2` 1.4 s, the two simulations < 0.2 s).
- **Common columns.** Every file has `h, vd, vdbc, se_boot (std of draws, N−1), cb05, cb95` (the band for cbIndicator = 0, i.e. the draw distribution translated so that its mean is vdBc) and `mean_draws`.
- **`mean_draws` is derived, not read.** The procedures do not return the draws. For the VAR bootstrap it is `vd − vdBc + svar`; for the block bootstrap and the simulations it is `2·vd − vdBc`.

| proc | call | extra columns |
|---|---|---|
| `var_r2` | `BootstrapVARR2(y,X,20,0,0,0,lagX,1:10,[1,3,4,5],2,1,2000,100,q,0,0)` | svar, lagvar (`details.lagVAR`), se_avar_pw/nopw (SubR2 at ŝ) |
| `var_lpb` | `BootstrapVARLPB(y,X,20,0,0,0,lagX,[1,0,0,0,0],1:10,[1,3,4,5],2,1,2000,100,q,0)` | svar, lagvar, ir, irbc, irse_boot, irse_avar, stdx, irstr, irstr_cb05/95 |
| `block_r2` | `BootstrapBlockR2(y,X,20,0,0,0,lagX,4,2000,100,q,0,0)` (not called by AppSimulation.m; package defaults) | se_avar_pw/nopw (at ŝ) |
| `sim_r2` | `SimulationAVarR2(y,X,20,0,0,0,lagX,2000,q,0,0,1)` | se_avar_pw/nopw_at_vdbc |
| `sim_lpb` | `SimulationAVarLPB(y,X,20,0,0,0,lagX,[1,0,0,0,0],2000,0,q,0,0,0,X,1,[1,0,0,0,0],[1,0,0,0,0])` | se_avar_pw/nopw_at_vdbc, ir, irse_avar, stdx |

**How `vd` and `ir` relate to the deterministic files:**
- In `var_*` and `block_r2`, the `vd`/`ir` columns are bit-identical to the deterministic files.
- In `sim_*` they differ by up to 6e-16. `SimulationAVar*` demeans the shock before SubR2/SubLPB demeans it again.
- The `sim_lpb` vd is the LP-set (`lpset`) estimate.

## Caveats for the Julia tests (audit, 2026-10-10)

- **These are Octave numbers, not MATLAB numbers.** The reference code ran unmodified, but on Octave's BLAS/LAPACK, `\` and libm `pow`. The results should agree with MATLAB to rounding (~1e-12 relative), not bit for bit. Compare with `rtol ≈ 1e-10`, as SPEC §6.A says.
- **Rebuilding `y = cumsum(Δy)` in Julia.** `Base.cumsum` on `Float64` uses pairwise summation, so it differs from MATLAB/Octave's sequential `cumsum` by up to 1.7e-13 on the T = 160 GDP series. Take `y` from `app_<case>_inputs.csv` (column 1), or rebuild it with `accumulate(+, dy)`, which is bit-identical.
- **MP cases at h = 0 are degenerate (SPEC Q33).** `XtIndicator = [1,1,1,1,0]` puts Δy_t and π_t on the right-hand side. At h = 0 the LHS is then an exact linear combination of the regressors, and the LP fit is perfect: Δy_t for `mp_y`, and π_t − π_{t−1} for `mp_pi` when π_{t−1} is a regressor.
  - Row h = 0 holds rounding noise (`ir` ~1e-15, `irse` ~1e-15 to 1e-14, `vd` ~1e-31 to 1e-29) in two groups of files: all four `app_mp_y_lp{a,b}_{varset,lpset}.csv`, and `app_mp_pi_lp{a,b}_varset.csv`.
  - `app_mp_pi_lp{a,b}_lpset.csv` is not degenerate. Its LP set has no π_{t−1}, so h = 0 gives ir = 0.018 and vd = 1.1e-4.
  - Compare those entries with an absolute tolerance, or skip them. The port is meant to return NaN for the AVar there.
- **Exact bounds.** FEVD shares can exceed 1 by one ulp. For example, `synthetic/var/vd_*.csv` has 1.0000000000000002 for the shock's own share at h = 0. Test `0 ≤ vd ≤ 1` with a small slack.
  - LP-A `vd` is not bounded by 1: `app_mp_pi_lpa_varset` reaches 1.3521 at h = 20, and `lpset` reaches 1.066.
  - Bias-corrected `vdbc` can be slightly negative at short horizons: down to −0.006 at h = 1 in `block_r2`, and at h = 1–2 in `sim_r2`.
- **Seed-to-seed spread of the stochastic files.** Measured over all 21 horizons, with z = Δvdbc / (se_boot·√(2/2000)):

  | proc | max \|z\| |
  |---|---|
  | `var_r2` | 1.23 |
  | `var_lpb` | 0.56 |
  | `block_r2` | 2.56 |
  | `sim_r2` | 2.97 |
  | `sim_lpb` | 1.41 |

  - The band endpoints `cb05`/`cb95` differ by at most 0.24·se_boot, and `se_boot` by at most 6.5%.
  - Hold a Julia run with a different RNG to |Δvdbc| ≲ 4·se_boot·√(2/2000) on every horizon and ≈0.3·se_boot on the bands. Do not use the 1.2σ seen at a few selected horizons.
- **Bandwidth near-ties.** `L(T) = round(0.75·T^(1/3) − 1)` depends on the last bit of `T^(1/3)`. With `^`, the computed values at T = 216 and T = 1000 sit just below the .5 tie, giving L = 3 and L = 6. `cbrt` lands exactly on the tie and gives 4 and 7. T = 8 is an exact tie (0.5 → 1, ties away from zero). Julia's `^` reproduces `synthetic/lrv/bandwidth.csv` for all T = 1..2000.
