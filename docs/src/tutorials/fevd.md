# Tutorial 5: Forecast-Error Variance Decompositions (Gorodnichenko–Lee)

This tutorial covers `fevd` and `fevdbootstrap`, a port of the local-projection
estimators of the forecast-error variance decomposition (FEVD) in

> Gorodnichenko, Y. and B. Lee (2020), "Forecast Error Variance Decompositions
> with Local Projections", *Journal of Business & Economic Statistics* 38(4),
> 921–933.

The reference implementation is the authors' MATLAB code (`SubR2.m`,
`SubLPA.m`, `SubLPB.m`, `SimulationAVar*.m`, `BootstrapVAR*.m`,
`BootstrapBlockR2.m`). Point estimates on the paper's data agree with it to
four decimals; the decisions where the port deliberately departs from the code
are listed [at the end](#Decisions-relative-to-the-MATLAB-reference).

## The estimand

Let ``x_t`` be an observed shock and ``y_t`` a variable in levels. The FEVD
targets the forecast error of the ``h``-step change
``y_{t+h} - y_{t-1}`` given the information set ``W_{t-1}``:

```math
f_{t+h|t-1} = (y_{t+h} - y_{t-1}) - \mathrm{E}[\,y_{t+h} - y_{t-1} \mid W_{t-1}],
\qquad
s_h = \frac{\sigma^2_x \sum_{i=0}^{h} \psi_i^2}{\mathrm{Var}(f_{t+h|t-1})},
```

with ``\psi_i`` the response of ``y_{t+i} - y_{t-1}`` to a unit shock. Every
estimator first regresses ``y_{t+h} - y_{t-1}`` on ``W_{t-1}`` on the rows
``t = t_0, \dots, T-h`` (``T_h = T - h - t_0 + 1`` rows, one sample per horizon)
and keeps the residual ``\hat f_{t+h|t-1}``.

| `estimator` | estimate | range |
|---|---|---|
| `:r2` (main text) | uncentred ``R^2`` of the forecast error on ``x_t, \dots, x_{t+h}`` | ``[0, 1]`` |
| `:lpa` | ``\hat\sigma^2_x \sum_{i\le h}\hat\psi_i^2 / \hat\sigma^2_{f,h}`` | unbounded |
| `:lpb` | ``\hat\sigma^2_x \sum_{i\le h}\hat\psi_i^2 / (\hat\sigma^2_x \sum_{i\le h}\hat\psi_i^2 + \hat\sigma^2_{v,h})`` | ``[0, 1]`` |

The shock is demeaned with its full-sample mean, and ``\hat\sigma^2_x`` is its
full-sample variance (``T-1`` denominator). In LP-A/B, ``\hat\psi_h`` is the
local-projection coefficient on ``x_t``, ``\hat\sigma^2_{f,h}`` the variance of
the forecast error (date-``t`` controls enter purged of ``W_{t-1}``), and
``\hat v`` the forecast error net of ``\sum_{i\le h}\hat\psi_i x_{t+h-i}``.

## Data

`docs/src/data/gorodnichenko_lee.csv` holds the 160 quarters (1969Q1–2008Q4) of
the paper's Section 5: utilization-adjusted TFP growth `tfp`, the Romer–Romer
shock `rr`, GDP growth `dy`, inflation `infl`, the federal funds rate `effr`,
and `y = cumsum(dy)`; `gorodnichenko_lee_rz.csv` holds the Ramey–Zubairy data
of Appendix H2. The provenance is in `docs/src/data/gorodnichenko_lee_README.md`.

```julia
using LocalProjections, DataFrames, CSV, Random
using StatsModels: @formula

df = CSV.read("docs/src/data/gorodnichenko_lee.csv", DataFrame)
```

(CSV.jl is not a dependency of LocalProjections; any reader that returns a
`DataFrame` will do.)

## The formula

One formula describes both the local projection and the information set:

```julia
fm = @formula(cumul(dy) ~ tfp + lags(tfp, 4) + lags(dy, 4) + lags(infl, 4) +
                          lags(rr, 4) + lags(effr, 4))
f = fevd(fm, df; horizon = 20)            # estimator = :r2
```

Displaying `f` prints:

```
LPFEVD (Gorodnichenko–Lee R²)
  Response:   cumul(dy)
  Shock:      tfp
  Formula:    :(cumul(dy)) ~ tfp + :(lags(tfp, 4)) + :(lags(dy, 4)) + :(lags(infl, 4)) + :(lags(rr, 4)) + :(lags(effr, 4))
  Sample:     T = 160, t0 = 5, T_h = T - h - t0 + 1
     Forecast-error variance share of tfp
╭─────────┬────────┬───────────┬────────┬─────╮
│ Horizon │  Share │ s.e. (PW) │   s.e. │ T_h │
├─────────┼────────┼───────────┼────────┼─────┤
│       0 │ 0.0437 │    0.0298 │ 0.0295 │ 156 │
│       1 │ 0.0089 │    0.0154 │ 0.0152 │ 155 │
│       2 │ 0.0243 │    0.0263 │ 0.0237 │ 154 │
│       3 │ 0.0453 │    0.0414 │ 0.0367 │ 153 │
│       4 │ 0.0517 │    0.0491 │ 0.0424 │ 152 │
│       ⋮ │      ⋮ │         ⋮ │      ⋮ │   ⋮ │
╰─────────┴────────┴───────────┴────────┴─────╯
```

- **Left-hand side**: `cumul(v)` with ``v`` a first difference
  (``\sum_{j\le h} v_{t+j} = y_{t+h} - y_{t-1}``), or `ldiff(y)` with ``y`` in
  levels. Inflation, used in levels by the paper, is `ldiff(infl)`.
- **Plain variables** enter at date ``t`` (the reference's `XtIndicator`). The
  shock is the first of them, or the `shock` keyword. For LP-A/B the others are
  date-``t`` controls; the R² estimator ignores them, as `SubR2.m` does.
- **`lags(v, p)` and `lags(firstdiff(v), p)`** form ``W_{t-1}``, with the
  required intercept and, with `trend = k`, the powers ``t, \dots, t^k`` of the
  absolute row index.
- Anything else (`leads`, `anchor`, `hbr`, functions, interactions) throws.
  Lags are built on the original rows, so the data must have no `missing` or
  `NaN` in the columns used.

The paper's monetary-policy specification has the date-``t`` controls
`tfp`, `dy` and `infl` next to the shock:

```julia
fmp = @formula(cumul(dy) ~ rr + tfp + dy + infl + lags(tfp, 4) + lags(dy, 4) +
                           lags(infl, 4) + lags(rr, 4) + lags(effr, 4))
fevd(fmp, df; horizon = 20, estimator = :lpb)
```

This specification warns twice. The regression fits exactly at ``h = 0``,
because `dy` is both the response and a date-``t`` control, so the share is 0
and the standard errors are `NaN` there. Prewhitening is also infeasible from
``h = 5`` on; the `irf_formula` of the next section addresses that.

`fevd(lp_result, df; estimator)` reuses the formula, horizon and shock of an
existing `lp` fit.

## Asymptotic standard errors

`f.se_pw` and `f.se` are delta-method standard errors of ``\hat s_h``, from the
moment conditions of the paper's Appendices A/B, with Bartlett long-run
variances of the moments with and without VAR(1) prewhitening. The lag is
``L_h = \mathrm{round}(0.75\,T_h^{1/3} - 1)`` (ties away from zero, e.g.
``L = 3`` for ``T_h \approx 150``), and there is no degrees-of-freedom factor.
The R² moments treat the forecast error as data, which is valid when the shock
is unpredictable from ``W_{t-1}``.

The LP-A/B moment vector stacks the impulse-response coefficients of horizons
``0, \dots, h``, so its length is ``N_q(h+1) + 2`` with ``N_q`` the number of
regressors of the impulse-response regression. With the full formula above
(``N_q = 22``) prewhitening becomes infeasible after a few horizons; `fevd`
then returns `NaN` with a warning. The paper estimates ``\hat\psi_h`` in a
parsimonious regression for this reason, which `irf_formula` reproduces (the
forecast errors still come from `formula`). The paper's lagged response is
formed from the level, hence `lags(firstdiff(y), 1)` (see *Sample start*
below):

```julia
fl = fevd(fm, df; horizon = 20, estimator = :lpb,
          irf_formula = @formula(cumul(dy) ~ tfp + lags(tfp, 1) + lags(firstdiff(y), 1)))
fl.irf                               # ψ̂_h, response to a unit shock
fl.irf .* sqrt(fl.shock_var)         # one-standard-deviation response
```

`dofadjust = true` (the default, the paper's `dfIndicator = 0`) divides the
LP-A/B variances ``\hat\sigma^2_{f,h}`` and ``\hat\sigma^2_{v,h}`` by
``T_h - k_W``, ``k_W`` the number of columns of ``W_{t-1}``; `false` divides by
``T_h``.

## Bias corrections

The estimators are biased in samples of the size used in macroeconomics. The
paper offers three corrections, all in `fevdbootstrap`. Each produces draws
``s^*_{b,h}`` with mean ``\bar m_h``:

| `method` | estimators | draws | bias-corrected share ``\tilde s_h`` |
|---|---|---|---|
| `:var` | R², LP-A, LP-B | the same estimator on data simulated from a VAR fitted to `data` | ``\hat s_h - (\bar m_h - s^{VAR}_h)`` |
| `:block` | R² | the R² on moving blocks of regression rows (Kilian–Kim) | ``2\hat s_h - \bar m_h`` |
| `:simulation` | R², LP-A, LP-B | ``\theta^* \sim N(\hat\theta_h, \hat V_h)`` mapped into the share | ``2\hat s_h - \bar m_h`` |

```julia
b = fevdbootstrap(f, df; method = :var, nboot = 2000, rng = Xoshiro(1))
fevdbootstrap(f; method = :block, blocklength = 4, rng = Xoshiro(1))
fevdbootstrap(fl; method = :simulation, prewhiten = true, rng = Xoshiro(1))
```

- **`:var`.** A VAR in `vars` is selected by `criterion = :hqic` among
  `nlags = 1:10` on a common sample, then iterated with iid resampled residuals
  for `burnin = 100` plus ``T`` periods from a random block of ``p`` actual
  observations; the full specification is re-estimated on each sample, without
  re-selecting lags. ``s^{VAR}_h`` is the FEVD of the fitted VAR for the
  Cholesky shock of the shock variable (cumulated for a `cumul(v)` response).
  The default `vars` orders the date-``t`` controls first, then the shock, then
  the other variables in order of appearance: `[tfp, dy, infl, rr, effr]` for
  both shocks of the paper. ``s^{VAR}_h``, and hence ``\tilde s_h``, depends on
  this order. For LP-A/B the result also carries the impulse-response draws and
  ``2\hat\psi_h - \mathrm{mean}_b\,\psi^*_{b,h}`` (`b.irf_bc`).
- **`:block`** resamples, horizon by horizon, ``\lceil T_h/\ell\rceil``
  non-circular blocks of the regression rows ``(y_{t+h} - y_{t-1}, W_{t-1},
  x_t, \dots, x_{t+h})``; the data are not re-simulated.
- **`:simulation`** draws from the normal approximation stored by `fevd`
  (prewhitened covariance with `prewhiten = true`), redraws draws with a
  nonpositive variance, and, with `bound = true`, clamps the R² draws to
  ``[0, 1]`` before averaging (LP-A/B draws are never clamped).

Draws are seeded per draw from `rng`, so `threaded = true` gives bit-identical
results. Failed draws are excluded and counted in `nfail`. The bias-corrected
share is never clamped to ``[0, 1]``.

## Bands

`summarize` returns an [`FEVDSummary`](@ref):

```julia
summarize(f; level = 0.90)                       # ŝ ± z·se_pw, deterministic
summarize(b; level = 0.90)                       # the paper's band (band = :shift)
summarize(b; band = :normal, prewhiten = true)   # s̃ ± z·se_pw
summarize(b; biascorrected = false)              # uncorrected band of the figures
```

With centre ``c_h`` (``\tilde s_h``, or ``\hat s_h`` with `biascorrected =
false`):

- `band = :shift` (the paper's `cbIndicator = 0`): ``Q_q(s^*_{\cdot,h}) + c_h -
  \bar m_h``, the distribution of the draws translated so that its mean is the
  centre, for ``q = \alpha/2, 1 - \alpha/2``. ``Q_q`` is MATLAB's `quantile`
  (Hyndman–Fan type 5, `quantile(v, q; alpha = 0.5, beta = 0.5)`); the
  `LPBootstrap` bands use type 7. With `biascorrected = false` this is the band
  ``Q_q(s^*) + \hat s_h - \bar m_h`` drawn around the uncorrected estimate in the
  paper's figures.
- `band = :normal` (`cbIndicator = 1`/`2`): ``c_h \pm z\,\mathrm{se}_h`` with
  `b.se_pw` or `b.se`.

The keyword is `band`, not `method` (the keyword of `summarize(::LPBootstrap)`),
because `method` already names the bias correction. On the paper's TFP → GDP
case with `nboot = 2000` and `Xoshiro(1)`, HQIC picks ``p = 2`` and:

| ``h`` | 0 | 4 | 8 | 12 | 16 | 20 |
|---|---|---|---|---|---|---|
| ``\hat s_h`` (R²) | .0437 | .0517 | .1025 | .1790 | .2803 | .3906 |
| ``s^{VAR}_h`` | .0428 | .0273 | .0460 | .0611 | .0733 | .0827 |
| ``\tilde s_h`` | .0463 | .0351 | .0692 | .1239 | .1994 | .2817 |
| 90% band | [.011, .100] | [−.001, .107] | [.003, .195] | [.029, .301] | [.076, .421] | [.133, .526] |

Bands are pointwise, not simultaneous, and not clamped to ``[0, 1]``.

## Plotting and conversion

```julia
using Plots
plot(f; levels = [0.68, 0.90])               # ŝ with normal bands
plot(b; levels = [0.90])                     # s̃ with its band, ŝ dashed
plot(b; band = :normal, biascorrected = false)
irf = as_irf_result(b; coverage = [0.68, 0.90])
```

The default vertical range covers ``[0, 1]`` and the bands. `as_irf_result`
returns a MacroEconometricTools `LocalProjectionIRFResult` whose `data` are
**shares**, with `metadata.fevd = true` and `metadata.quantity = :fevd_share`.
With a Makie backend loaded, `irfplot(f; levels, prewhiten)` and
`irfplot(b; levels, band, biascorrected, prewhiten)` draw the same bands from that
object.

## Conventions and reproducing the paper

- **Sample start.** ``t_0 \ge 2`` is the first row where every regressor
  exists. To match the reference's rows, map each lag to the matching MATLAB
  input. `lags(v, p)` of a column in the data is MATLAB's `lagX`: the
  application's `lags(dy, 4)` gives ``t_0 = 5``, as in the reference.
  `lags(firstdiff(y), p)` (with `y` the level) is MATLAB's `lagy`/`lagyLP`,
  which lag ``\Delta y`` formed from the level and so start one row later. Use
  it only where the reference sets `lagy`/`lagyLP > 0`, e.g. the paper's TFP
  impulse-response set
  `irf_formula = @formula(cumul(dy) ~ tfp + lags(tfp, 1) + lags(firstdiff(y), 1))`
  (``t_0^{LP} = 3``). The left-hand side can stay `cumul(dy)`; `ldiff(y)` gives
  the same rows. With `method = :var`, a level that enters only through
  `firstdiff` is not put in the VAR next to its difference (which would make
  the VAR singular): each draw rebuilds it by cumulating the simulated
  difference, as the reference does (`yb = cumsum(Δy_b)`).
- **`skipfirst`** (`:var`, `cumul(v)` responses). The reference fits the VAR on
  rows ``2:T`` (it forms ``\Delta y`` from the level) while the draws have ``T``
  rows. The default uses all rows; `skipfirst = true` reproduces the
  reference's lag choice and ``s^{VAR}_h`` exactly (TFP → GDP:
  ``s^{VAR}_0 = .0422``, ``s^{VAR}_4 = .0256``).
- **Simulation standard errors.** In `:simulation`, `b.se_pw` / `b.se` are the
  delta-method standard errors evaluated at the **bias-corrected** share (the
  authors' choice, `SimulationAVarR2.m:180-182`), so they depend on the draws;
  `fevd` reports them at ``\hat s_h``.
- **Not implemented:** the VAR FEVD estimator and its VAR–VAR bootstrap, joint
  inference across horizons (`jointIndicator = 1`), Driscoll–Kraay standard
  errors (`semIndicator = 1`), and `method = :var` with `trend ≥ 1` (it throws).

## Decisions relative to the MATLAB reference

R = replicated, F = fixed (documented deviation).

| Item | Reference behaviour | Here |
|---|---|---|
| Shock moments | full-sample mean and variance, uncentred R² | R |
| LP-A purging | ``X^\perp`` estimated once on ``t_0..T``; LP-B ``\hat v`` uses the raw shock | R |
| Variance denominators | `dfIndicator` documented inverted | R the code: `dofadjust = true` ≡ `dfIndicator = 0` |
| Bandwidth | ``\mathrm{round}(0.75\,T_h^{1/3} - 1)``, prewhitened `lrv` divides by ``T_h - 1`` | R (ties away from zero) |
| Prewhitening | VAR(1), no eigenvalue capping | R; `NaN` + warning when infeasible |
| Regressions (incl. the prewhitening VAR(1)) | normal equations | F: QR. Usually rounding-level differences, but see the exact-fit row |
| ``t_0`` in `SubR2` | can be 1, giving `NaN` | F: ``t_0 \ge 2`` |
| Exact fit at a horizon (MP case, ``h = 0``) | near-singular moments | F: `se_pw`/`se`/`avar_pw`/`avar` `NaN` for that ``h``, warning (`irf`, `irf_se` stay numerically zero). The moments of the exact-fit horizon are pure rounding noise and stay in the stacked moment vector at every later ``h``. The reference's normal-equation prewhitening is then ill-conditioned, so its prewhitened variance (`se_pw`, `avar_pw`, and the default `:simulation` draws and bands) is dominated by rounding error at every ``h \ge 1``: in the paper's MP specification the package's value differs by up to about 2.5x (LP-B, ``h = 16``: 0.678 vs 0.272). The package's QR value is the accurate one: it equals the result of dropping the degenerate block. `se` without prewhitening agrees to rounding |
| Simulation covariance | chosen by `seIndicator`; eigenvalue repair floors the whole matrix in an unbounded loop | F: `prewhiten` keyword; diagonal floor, one pass |
| Simulation SEs | gradient at ``\tilde s_h``; complex when the variance is negative | R the gradient; F: `NaN` + warning |
| R² simulation clamp | draws clamped to ``[0, 1]`` | R (`bound = true`) |
| Block bootstrap burn-in | 100 rows drawn and discarded | F: ``\lceil T_h/\ell\rceil`` blocks (equivalent when ``\ell`` divides 100) |
| VAR lag candidates | ``p = 0`` allowed, crashes | F: candidates ``\ge 1`` |
| VAR sample with ``\Delta y`` | rows ``2:T`` | F by default; `skipfirst = true` replicates |
| VAR variable layout | positional, some layouts crash | F: `vars` by name, validated |
| LP impulse-response correction | ``2\hat\psi - \mathrm{mean}\,\psi^*`` | R |
| Failed draws | `NaN` propagates | F: excluded, `nfail`, warning |
| Quantiles | MATLAB `quantile` | R (type 5) |
| Band clamping | none | R (none) |
