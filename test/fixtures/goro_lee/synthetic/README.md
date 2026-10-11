# Synthetic Gorodnichenko–Lee fixtures

These are deterministic fixtures for the literal-port tests of the Gorodnichenko & Lee (2020) FEVD estimators. All of them come from running the authors' MATLAB reference code under **GNU Octave 11.3**.

## Reproducing

The reference code is read-only and is added to the path with `addpath`. It is never modified, and it lives in the gitignored tree `docs/src/code/goro_lee_r2/lib`. To regenerate the fixtures:

```sh
sh  test/fixtures/goro_lee/drivers/gen_verbatim_drivers.sh        # always first: its outputs are not committed
octave --no-gui test/fixtures/goro_lee/drivers/make_synthetic_fixtures.m
```

- **Shims.** Octave has no Econometrics/Statistics toolboxes, so the shims in `../octave_shims/` stand in for them. Only `lagmatrix` is used on these deterministic paths. It follows MATLAB semantics: a positive lag shifts down with NaN padding, a negative lag is a lead, and with several lags the output is lag-major.
- **Quantiles.** Octave's default `quantile` method is already Hyndman–Fan type 5, the same as MATLAB's, so no shim is needed. These fixtures do not use it anyway.
- **Determinism.** Re-running the script reproduces the files bit for bit.

### Driver files (`../drivers/`)

| file | role |
|---|---|
| `make_synthetic_fixtures.m` | main script. It generates the data, calls the reference functions, checks finiteness, and writes the CSVs. |
| `syn_writecsv.m` | headerless CSV writer using `%.17g` |
| `gen_verbatim_drivers.sh` | uses `sed` to extract the line ranges below verbatim and writes the next four files. They are generated, not committed (`../drivers/.gitignore`), because they hold third-party code; run the script before `make_synthetic_fixtures.m` |
| `sim_avar_r2_det.m` | `SimulationAVarR2.m` lines 107-121 and 183-190. The random lines 123-174 are replaced by `vdBc = vd`. |
| `sim_avar_lpA_det.m`, `sim_avar_lpB_det.m` | `SimulationAVarLP{A,B}.m` lines 156, 194-214, 216-261 and 313-328. The random lines 262-311 are skipped and `vdBc(h+1) = vd(h+1)` is set instead, so the gradient is taken at ŝ. `jointIndicator = 0`. |
| `ic_block.m` | `BootstrapVARR2.m` lines 104 and 133-173: XVAR construction and the information criterion |

Lines written by the driver are marked `% DRIVER`. The copied blocks sit between `BEGIN/END VERBATIM` markers.

## Format

- Every file is a headerless, comma-separated numeric matrix written with `%.17g`, which round-trips Float64 exactly.
- Read a file in Julia with `readdlm(path, ',', Float64)`.
- `NaN` is written literally.
- Vectors are columns (n×1) and scalars are 1×1.
- Cell arrays indexed by horizon are written one file per horizon, `<name>_h<h>.csv` for h = 0..8.
- Indices (`idxt`, `t0LP`, `window`, `idx`, `init`) are MATLAB **1-based** row numbers.

## Inputs

| file | content |
|---|---|
| `data_y.csv` | `y` (200×1), in levels: `y = 10 + cumsum(dy)` |
| `data_X.csv` | `X = [x z2 z3]` (200×3). `x` is iid N(0,1) and is the shock. `z2_t = .5 z2_{t-1} + .3 x_{t-1} + u`, `z3_t = .4 z3_{t-1} + .2 dy_{t-1} + .5u`, `dy_t = .1 + .4 dy_{t-1} + .6 x_t + .3 x_{t-1} + .2 z2_{t-1} + .8e` (Octave `randn('state', 20200101)`). |

Common settings: T = 200, `horizon` H = 8, `lagy = 2`, `lagX = [4 2 3]`.

## Outputs

### `r2_<case>/`: `SubR2(y, X, 2, [4 2 3], 8, ylevel, trend, 1)`

| case | ylevel | trend |
|---|---|---|
| `r2_yl0_tr0` | 0 | 0 |
| `r2_yl1_tr0` | 1 | 0 |
| `r2_yl0_tr1` | 0 | 1 |

Files:
- `vd` (9×1)
- `forecastError` (200×9; NaN outside `idxt`)
- `seAVarWPW`, `seAVarWoPW` (9×1)
- `idxt` (9×2, `[t0, T-h]`)
- `effectiveT` (9×1)
- `theta_h<h>` ((2h+3)×1, ordered `[θ1; θ2; θ3]`)
- `GOGTWPW_h<h>`, `GOGTWoPW_h<h>` ((2h+3)×(2h+3))

### `lpa_<case>/`, `lpb_<case>/`: `SubLPA` / `SubLPB`

All of these use `XtIndicator = [1 1 0]`, `trend = 0`, `extraOutputIndicator = 1` and `semIndicator = 0`. The case name is `yl{0,1}_df{0,1}_{shared,sep}`, where `df` is `dfIndicator`; `dfIndicator = 0` means the denominator is `T_h - k_W`.

- `shared`: called with 11 arguments, so the LP set equals the FE set.
- `sep`: `XLP = X, lagyLP = 1, lagXLP = [1 0 0], XtIndicatorLP = [1 0 0]`. This is the paper's parsimonious LP set.

Files, in the function's output order:
- `ir`, `vd`, `irSeAVar`, `mse` (9×1)
- `forecastError` (200×9)
- `meanx`, `varx` (1×1; `varx` uses T−1)
- `idxt` (9×2)
- `dhy` (200×9, `y_{t+h} − y_{t−1}`)
- `rhsAllLP` (200×Nq; Nq = 14 shared, 4 sep)
- `residualLP` (200×9)
- `BetaHat` (Nq×9)
- `t0LP` (1×1)

LP-B also has `vFE` (200×9) and `sigma2vHat` (9×1).

### `simavar_r2_yl0/`: deterministic prefix of `SimulationAVarR2` (ylevel 0, trend 0)

Files: `vd`, `theta_h<h>`, `GOGTWPW_h<h>`, `GOGTWoPW_h<h>`, and `seAVarWPW`/`seAVarWoPW` (the delta-method s.e. at ŝ).

These equal `r2_yl0_tr0` up to rounding (≈2e-16), because the driver demeans x once more before calling SubR2.

### `simavar_lp{a,b}_yl{0,1}/`: deterministic prefix of `SimulationAVarLP{A,B}`

Settings: `jointIndicator = 0`, `dfIndicator = 0`, `trend = 0`, and the parsimonious LP set above (Nq = 4).

Files:
- `vd`, `ir`, `irSeAVar`, `forecastErrorVar` (9×1; σ̂²_f for A, m_h for B)
- `varx`, `Sigmaq` (4×4), `Nq`, `t0LP`
- `window` (9×3, `[tmin_h, tmax_h, T_h]`)
- `theta_h<h>` ((4(h+1)+2)×1, `[vec(BetaHat(:,1:h+1)); varx; σ̂²_{f or v}]`)
- `GOGTWPW_h<h>`, `GOGTWoPW_h<h>` (V_h with and without pre-whitening)
- `Deltah_h<h>` (1×(4(h+1)+2), the gradient at ŝ)
- `seAVarWPW`, `seAVarWoPW` (9×1, `sqrt(Δ V Δ')`)

The moment matrices Z_h are not stored (about 6 MB). They can be rebuilt from `lp*_sep` (`rhsAllLP`, `residualLP`, `forecastError`/`vFE`).

### `lrv/`

| file | content |
|---|---|
| `Z.csv` | 60×3 VAR(1) series (`randn('state', 7)`) |
| `lrv_L{0,1,3}.csv` | `lrv_nw(Z, L)` |
| `pw_A.csv`, `pw_U.csv`, `pw_L.csv`, `pw_Omega.csv` | the pre-whitened LRV, using the operations of `SubR2.m` lines 214-220 on `ZFull = Z` with `L = round(0.75*60^(1/3) − 1)` |
| `bandwidth.csv` | `[T, round(0.75*T^(1/3) − 1)]` for T = 1..2000 (MATLAB `round` is half away from zero) |

### `var/`: `var_chol(input, 2, 8, 0, 1, 0, dIdx, 0, 0)`

`input.csv` is `[x(2:T), diff(y), z2(2:T)]` (199×3). The shock is in position 1 and the response of interest is in position 2.

- `vd_nodiff` / `vd_diff` (3×9; all variables): `dIdx = []` and `dIdx = 2` respectively.
- `psi_nodiff` / `psi_diff` (3×9): IRFs to a one-s.d. shock 1.
- `c` (3×1)
- `phi` (3×6, `reshape(phi, 3, 3*2) = [Φ1 Φ2]`)
- `omega` (3×3, divided by T_eff)
- `residual` (3×197)
- `k_T` (`[n²p, T_eff]`)
- `psiReduced` (3×27, `[Θ0 … Θ8]`)

### `ic/`: `ic_block(y, X, ylevel, 0, 1:6, [1 2 3], 2, ICIndicator)`

These come from `BootstrapVARR2.m` lines 133-173. `XVAR_yl{0,1}.csv` is the VAR system `[x, y or Δy, z2, z3]`: 199×4 when ylevel = 0 (rows 2:T), 200×4 when ylevel = 1.

- `ic_yl{0,1}_{bic,hqic,aic}.csv` (6×1): the IC values for p = 1..6 on the common sample.
- `lag_yl*_*.csv`: `[chosen p, T_IC]`.

Chosen lags: p = 1 everywhere except `yl1_aic`, which gives p = 2.

### `varboot/`: `var_bootstrap(c, phi, input, residual, 200, 100)`

| case | system | VAR order | rows |
|---|---|---|---|
| `case1` | the 3-variable system of `var/` | VAR(2) | 199 (Tout = Ty = 200 ≠ rows, Q34) |
| `case2` | the ylevel = 1 `XVAR` of `ic/` | VAR(1) | 200 |

Before each call, the driver seeds `rand('twister', 4242 + case)` and draws `idx = ceil(T_eff*rand(Tend,1))` and `init = ceil((rows − p + 1)*rand)`. It then reseeds, so `var_bootstrap` consumes exactly the same draws. `idx(1)` is drawn but unused.

The driver checks that a hand-written replay with the recorded `idx`/`init` reproduces the output exactly (`isequal`).

Files:
- `<case>_input`, `_c`, `_phi` (n×np), `_residual` (n×T_eff)
- `_idx` (300×1, 1-based residual columns)
- `_init` (1-based)
- `_sim` (200×n, the simulated series)
- `_meta` (`[p, T, TBurnIn, T_eff]`)

The trend clock in the simulation is `s = 1..Tend`, which is irrelevant here because trend = 0.
