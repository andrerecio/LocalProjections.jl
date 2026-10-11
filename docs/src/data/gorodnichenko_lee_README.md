# Gorodnichenko & Lee (2020) application data

Gorodnichenko, Y. and B. Lee (2020), "Forecast Error Variance Decompositions with
Local Projections", *Journal of Business & Economic Statistics* 38(4), 921–933.

These two files hold the data matrices that the authors' MATLAB replication code feeds
to its FEVD estimators:

| file | MATLAB source | used by | rows | sample |
|---|---|---|---|---|
| `gorodnichenko_lee.csv` | `AppData.m` (`data`, 160×8) | `AppSimulation.m`: Section 5 (Figures 3–4) and Appendix H1 | 160 | 1969Q1–2008Q4 |
| `gorodnichenko_lee_rz.csv` | `AppDataRZ.m` (`macro_quarter`, 384×6) | `AppSimulationRZ.m`: Appendix H2 | 384 | 1920Q1–2015Q4 |

The MATLAB files are in `docs/src/code/goro_lee_r2/`, which is gitignored. Both scripts
use the full matrix (`T1 = 1`, `T2 = 160` and `T2 = 384`), so the CSVs contain no rows
outside the estimation sample and need no trimming. There are no missing values.

## Precision and provenance of the numbers

- Each value comes from the numeric literal in the `.m` file. It is written as the
  shortest decimal that parses back to the **same IEEE double** MATLAB uses, so the
  numbers are bit-identical to the MATLAB matrices (checked in Julia with `==`).
- The `.m` literals are rounded. `AppData.m` uses 14 decimals. `AppDataRZ.m` uses about
  10 significant digits for `dy` and `infl` and 2 decimals for `unemp` and `tbill`. The
  CSVs reproduce the `.m` literals, not the more precise spreadsheet cells.
  Maximum absolute differences against the spreadsheets:
  - `AppData.xls`, sheet `Data`: ≤ 5.2e-14 in every column.
  - `AppDataRZ.xlsx`, sheet `Quarterly`, columns J–O:
    - `newsy`: exact;
    - `dy`, `infl`: 4.9e-9;
    - `unemp`, `tbill`: 5e-3 (2-decimal rounding).
- The `y` column is derived. It is `cumsum(dy)` computed as a strict left-to-right
  running sum, which is how MATLAB's `cumsum` adds. Julia's `Base.cumsum` uses a
  different summation order, so on the same column it differs by up to about 4e-13.

## `gorodnichenko_lee.csv` (Section 5 / Appendix H1)

| column | `AppData.m` col | description | units |
|---|---|---|---|
| `date` | — | first day of the quarter (ISO) | |
| `quarter` | 1 | decimal quarter (1969.0, 1969.25, …) | |
| `tfp` | 2 | utilization-adjusted TFP growth `dtfp_util`. Source: Fernald (2014), quarterly series, vintage 2017-06-05 | 400·Δlog (annualized %) |
| `rr` | 3 | Romer–Romer monetary policy shock, updated by Coibion, Gorodnichenko, Kueng & Silvia (2017) ("RR Updated Shocks", quarterly) | percentage points |
| `dy` | 4 | real GDP growth, 400·Δlog(GDPC1) | annualized % |
| `dc` | 5 | real consumption growth excluding durables, 400·Δlog(PCECC96 − durables) (**unused**) | annualized % |
| `di` | 6 | real investment growth including consumer durables, 400·Δlog(GPDIC1 + durables) (**unused**) | annualized % |
| `effr` | 7 | effective federal funds rate (FEDFUNDS), quarterly average | % per annum |
| `infl` | 8 | GDP deflator inflation, 400·Δlog(GDPDEF) | annualized % |
| `y` | derived | `cumsum(dy)` = 400·log(RGDP_t / RGDP_1968Q4) | 400×log |

Notes on the construction (from the `AppData.xls` sheets `FREDraw`, `Fernald`,
`Updated R&R`, `Manipulation` and `Data`):

- The macro series are FRED downloads (chained-2009 vintage).
- The 1969Q1 growth rates use the 1968Q4 levels.
- Real durables consumption before 1999 is extrapolated from its chain index. The
  level/index ratio is 10.23320901925391.
- Nothing is normalized at the source, but `rr` already has a mean of essentially 0.
  The estimators demean the shock with its full-sample mean, and they scale IRFs by its
  sample standard deviation (N−1).
- Sanity moments (mean, s.d. with N−1):

  | series | mean | s.d. |
  |---|---|---|
  | `tfp` | 0.991278 | 3.267359 |
  | `rr` | ≈ 0 | 0.588040 |
  | `dy` | 2.872734 | 3.405052 |
  | `infl` | 3.897109 | 2.419157 |
  | `effr` | 6.4935 | 3.393294 |

### Mapping to `AppSimulation.m`

```matlab
timeseries = data(1:160, [2,4,8,3,7]);   % = [tfp, dy, infl, rr, effr]
y   = cumsum(timeseries(:,2));           % = y      (ylevely  = 0: Δy formed inside)
pi  = timeseries(:,3);                   % = infl   (ylevelpi = 1: used in levels)
XTFP = timeseries;                       % = [tfp, dy, infl, rr, effr]
XMP  = timeseries(:, [4,1,2,3,5]);       % = [rr, tfp, dy, infl, effr]
```

Column 1 of `X` is always the shock. The common settings are:

- `lagy = 0`, `lagX = [4,4,4,4,4]`, `trend = 0` (intercept only), `horizon = 20`.
- The LP left-hand side is `y_{t+h} − y_{t−1}`:
  - for GDP it equals `cumul(dy)` and `ldiff(y)`;
  - for inflation it equals `ldiff(infl)`.
- With `lagy = 0` there are no lags of the response beyond those already in `X`. The
  regressors are lags 1–4 of every `X` column plus an intercept (`t0 = 5`).
- R2 uses lags only. LP-A and LP-B also include the contemporaneous `X` columns flagged
  by `XtIndicator`.

| case | MATLAB `y` (ylevel) | `X` (CSV columns, in order) | `XtIndicator` (contemporaneous) | `XVAROrder` | `yPlace` |
|---|---|---|---|---|---|
| TFP → GDP | `y` (0) | `[tfp, dy, infl, rr, effr]` | `[1,0,0,0,0]`: `tfp_t` | `[1,3,4,5]` | 2 |
| TFP → inflation | `infl` (1) | `[tfp, dy, infl, rr, effr]` | `[1,0,0,0,0]`: `tfp_t` | `[1,2,4,5]` | 3 |
| MP → GDP | `y` (0) | `[rr, tfp, dy, infl, effr]` | `[1,1,1,1,0]`: `rr_t, tfp_t, dy_t, infl_t` | `[2,4,1,5]` | 2 |
| MP → inflation | `infl` (1) | `[rr, tfp, dy, infl, effr]` | `[1,1,1,1,0]`: `rr_t, tfp_t, dy_t, infl_t` | `[2,3,1,5]` | 3 |

- **Bootstrap VAR (`method = :var`).** In all four cases the system is
  `[tfp, dy, infl, rr, effr]`, and `XVAROrder`/`yPlace` reassemble exactly this order.
  - The shock is variable 1 (TFP) or variable 4 (MP).
  - The lag is chosen by HQIC over 1:10, with `TBurnIn = 100` and `NBootstrap = 2000`.
  - For GDP (`ylevel = 0`), the code builds the VAR from rows 2:T, with `diff(y)` in
    place of `dy` (see spec Q34, `skipfirst = true`).
- **Point VAR(4) estimator.** `var_chol(timeseries, 4, 20, …, shock, 0, 2, …)` is a
  VAR(4) on `[tfp, dy, infl, rr, effr]` with Cholesky identification in that order.
  Row 2 (`dy`) is cumulated. The shock is 1 for TFP and 4 for MP.
- **Asymptotic-simulation variants (`SimulationAVar*`).** The forecast-error part keeps
  `lagy = 0` and `lagX = 4`. The LP set (`XLP = X`) differs by shock:
  - TFP: `lagyLP = 1` and `lagXLP = [1,0,0,0,0]`, so the regressors are
    `tfp_t, tfp_{t−1}`, the lagged response (`dy_{t−1}` or `infl_{t−1}`) and a constant.
  - MP: `lagyLP = 0`, `lagXLP = 0` and `XtIndicatorLP = [1,1,1,1,0]`, so the regressors
    are `rr_t, tfp_t, dy_t, infl_t` and a constant.

## `gorodnichenko_lee_rz.csv` (Appendix H2, Ramey–Zubairy military news)

| column | `AppDataRZ.m` col | description | units |
|---|---|---|---|
| `date` | — | first day of the quarter (ISO) | |
| `quarter` | 1 | decimal quarter (1920.0, …, 2015.75) | |
| `newsy` | 2 | military news shock normalized by lagged trend GDP: `news_t / (rgdp_pott6_{t−1} · pgdp_{t−1})` (RZ `newsShock`, the `newsy` of `ramey_zubairy.csv`) | share of potential GDP |
| `dy` | 3 | per-capita real GDP growth, 400·Δlog(rgdp/pop) | annualized % |
| `infl` | 4 | GDP deflator inflation, 400·Δlog(pgdp) | annualized % |
| `unemp` | 5 | civilian unemployment rate (**unused**) | % |
| `tbill` | 6 | 3-month Treasury bill rate, secondary market (RZ `tbill`, which equals `tbill_interp` from 1920) | % per annum |
| `y` | derived | `cumsum(dy)` = 400·log(ypc_t / ypc_1919Q4) | 400×log |

Source and checks:

- The source is the Ramey & Zubairy (2018, JPE) quarterly data (`rzdat` sheet of
  `AppDataRZ.xlsx`, revised November 21, 2016).
- The sample starts in 1920Q1 because the T-bill series starts there.
- Compared with the package's `ramey_zubairy.csv`:
  - `newsy` matches to 5e-10 and `infl` to 4.8e-9;
  - `ramey_zubairy.csv` has no T-bill column and no exact `dy`
    (`400·Δlog(ypc)` there is off by up to 3e-5 in the early 1920s).
- Sanity moments (mean, s.d. with N−1):

  | series | mean | s.d. | other |
  |---|---|---|---|
  | `newsy` | 0.008011 | 0.063484 | 94 of 384 quarters nonzero |
  | `dy` | 2.027075 | 8.035405 | |
  | `infl` | 2.413708 | 5.423818 | |
  | `tbill` | 3.497839 | 2.987718 | |

### Mapping to `AppSimulationRZ.m`

```matlab
timeseries = macro_quarter(1:384, [2,3,4,6]);   % = [newsy, dy, infl, tbill]
y  = cumsum(timeseries(:,2));                   % = y     (ylevel = 0)
pi = timeseries(:,3);                           % = infl  (ylevel = 1)
X  = timeseries;                                % = [newsy, dy, infl, tbill] for both responses
```

The common settings are `lagy = 0`, `lagX = [4,4,4,4]`,
`XtIndicator = [1,0,0,0]` (contemporaneous `newsy_t` only), `trend = 0` and
`horizon = 20` (`t0 = 5`).

| case | MATLAB `y` (ylevel) | `XVAROrder` | `yPlace` |
|---|---|---|---|
| news → GDP per capita | `y` (0) | `[1,3,4]` | 2 |
| news → inflation | `infl` (1) | `[1,2,4]` | 3 |

- The bootstrap VAR system is `[newsy, dy, infl, tbill]`, with the shock as variable 1.
- The point VAR(4) is `var_chol(timeseries, 4, 20, …, 1, 0, 2, …)` with `dy` cumulated.
- The `SimulationAVar*` LP set uses `lagyLP = 1`, `lagXLP = [1,0,0,0]` and
  `XtIndicatorLP = [1,0,0,0]`.
