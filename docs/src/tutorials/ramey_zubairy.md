# Tutorial 4: Ramey–Zubairy Cumulative Fiscal Multipliers

This tutorial replicates the **baseline (linear) cumulative government-spending
multipliers** of Ramey and Zubairy (2018, *JPE*), "Government Spending
Multipliers in Good Times and in Bad: Evidence from US Historical Data".

The reference implementation is the Stata program `jordagk.do` in the authors'
replication package. Its "ESTIMATION OF CUMULATIVE" block is what we reproduce
here.

## The estimand

Ramey and Zubairy use the Gordon–Krenn normalisation: every variable is divided
by an estimate of *potential* GDP, so that regression coefficients are already
in dollar-for-dollar units and no ex-post conversion factor is needed. With

```math
y_t = \frac{\text{real GDP}_t}{\text{potential GDP}_t},
\qquad
g_t = \frac{\text{nominal gov.\ purchases}_t / P_t}{\text{potential GDP}_t},
```

the *one-step* cumulative multiplier at horizon ``h`` is the coefficient
``m_h`` in the 2SLS regression

```math
\sum_{j=0}^{h} y_{t+j}
  \;=\; m_h \sum_{j=0}^{h} g_{t+j}
  \;+\; \gamma' w_{t-1} \;+\; u_{t+h},
\qquad
\sum_{j=0}^{h} g_{t+j} \ \text{ instrumented by the shock } s_t ,
```

where ``w_{t-1}`` collects four lags of the controls and a constant. Two shocks
are used:

| Shock | Instrument ``s_t`` | Controls ``w_{t-1}`` |
|---|---|---|
| Military news | `newsy` ``= \text{news}_t / (\text{potential GDP}_{t-1} P_{t-1})`` | 4 lags of `newsy`, `y`, `g` |
| Blanchard–Perotti | `g` (current spending, taken as predetermined) | 4 lags of `y`, `g` |

This is *not* the "two-step" multiplier — the ratio of the cumulative sums of
separately estimated impulse responses — which Ramey and Zubairy also report and
which is computed further below.

## The dataset

`docs/src/data/ramey_zubairy.csv` is the cleaned analysis file, built from
`rzdatnew.csv` in the replication package with exactly the variable definitions
of `jordagk.do`. It is quarterly, 1889Q1–2015Q4 (508 observations).

| Column | Definition (`jordagk.do`) |
|---|---|
| `quarter` | date as a decimal year (`1889.0`, `1889.25`, …) |
| `y` | `rgdp / rgdp_pott6` |
| `g` | `(ngov / pgdp) / rgdp_pott6` |
| `newsy` | `news / (L.rgdp_pott6 * L.pgdp)` |
| `taxy` | `nfedcurrreceipts_nipa / ngdp` |
| `debty` | `pubfeddebt_treas / L.ngdp` |
| `infl` | `400 * D.log(pgdp)` |
| `unemp` | civilian unemployment rate |
| `slack` | `unemp >= 6.5` |
| `zlb` | `zlb_dummy` |
| `recession` | NBER recession indicator |
| `wwii` | `quarter >= 1941.5 & quarter < 1946` (rationing period) |
| `ag` | Auerbach–Gorodnichenko state, `exp(-1.5 z)/(1 + exp(-1.5 z))` |
| `y_cbo`, `g_cbo`, `newsy_cbo` | as above but normalised by `rgdp_potcbo` |

Two conventions are worth flagging:

* Pre-1889 rows are dropped **before** any lag or lead is formed, exactly as in
  `jordagk.do`, so `L.` at 1889Q1 is missing and `newsy` starts in 1890Q1.
* Stata evaluates `gen slack = unemp >= 6.5` to `1` when `unemp` is missing.
  We emit `missing` instead. The two differ only over 1889Q1–1889Q4, which no
  Ramey–Zubairy regression sample reaches.

## Estimation

The endogenous regressor ``\sum_{j=0}^{h} g_{t+j}`` moves with the horizon, and
so does the response. Both are expressed with `cumul`, which tracks the
projection horizon on either side of the formula (see the "Horizon-tracking
right-hand-side terms" note in the [`lpiv`](@ref) docstring), so the whole
multiplier path is one call:

```julia
using LocalProjections, DataFrames, CSV
using StatsModels: @formula
using CovarianceMatrices: Bartlett, NeweyWest

rz = CSV.read(joinpath(@__DIR__, "..", "data", "ramey_zubairy.csv"),
              DataFrame; missingstring = "")
rz.bp = rz.g          # Blanchard-Perotti shock: current spending

news = lpiv(@formula(cumul(y) ~ (cumul(g) ~ newsy) +
                     lags(newsy, 4) + lags(y, 4) + lags(g, 4)), rz; horizon = 20)

bp   = lpiv(@formula(cumul(y) ~ (cumul(g) ~ bp) +
                     lags(y, 4) + lags(g, 4)), rz; horizon = 20)

summarize(news, vcov(Bartlett(30.0), news))   # multiplier path, s.e. and bands
summarize(bp,   vcov(Bartlett(30.0), bp))
```

The multiplier is the coefficient on `cumul(g)`, which is what `shock` picks by
default, so `summarize` needs no `term`. The design `lpiv` builds is exactly
the one `ivreg2` builds for
`ivreg2 f{h}cumuly (f{h}cumulg = bp) L(1/4).y L(1/4).g` — a constant, four lags
of each control, and the cumulative-spending regressor last:

```julia
julia> bp.coef_names
10-element Vector{String}:
 "(Intercept)"
 "y_lag1"
 "y_lag2"
 "y_lag3"
 "y_lag4"
 "g_lag1"
 "g_lag2"
 "g_lag3"
 "g_lag4"
 "cumul(g)"

julia> [m.nobs for m in bp.models][[1, 9, 21]]   # h = 0, 8, 20
3-element Vector{Int64}:
 504
 496
 484
```

Those sample sizes are Stata's exactly: 1890Q1 through 2015Q4 minus the horizon.

## Results

Full sample, 1889Q1–2015Q4, four lags, no trends, no tax controls, WWII
rationing *not* omitted — the shipped defaults of `jordagk.do`. `RZ` is the
authors' published `multlin1`/`seylin` from
`Multiplier-Standard-Errors.xlsx`. Standard errors are shown both with the
automatic Newey–West bandwidth (`auto`) and with a fixed bandwidth of 30
(`fix`); the next section explains why.

| h | news | s.e. auto | s.e. fix | RZ | RZ s.e. | BP | s.e. auto | s.e. fix | RZ | RZ s.e. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.306 | 0.321 | 0.348 | 1.255 | 0.329 | 0.179 | 0.149 | 0.163 | 0.208 | 0.155 |
| 1 | 1.060 | 0.261 | 0.247 | 1.035 | 0.235 | 0.223 | 0.119 | 0.145 | 0.235 | 0.143 |
| 2 | 0.847 | 0.173 | 0.173 | 0.820 | 0.155 | 0.257 | 0.106 | 0.133 | 0.257 | 0.133 |
| 3 | 0.706 | 0.124 | 0.130 | 0.695 | 0.123 | 0.254 | 0.101 | 0.134 | 0.251 | 0.133 |
| 4 | 0.680 | 0.100 | 0.099 | 0.674 | 0.097 | 0.272 | 0.105 | 0.134 | 0.271 | 0.132 |
| 6 | 0.677 | 0.076 | 0.074 | 0.673 | 0.074 | 0.356 | 0.098 | 0.119 | 0.353 | 0.119 |
| **8** | **0.669** | 0.061 | 0.059 | 0.667 | 0.060 | **0.413** | 0.089 | 0.105 | 0.411 | 0.104 |
| 10 | 0.706 | 0.055 | 0.053 | 0.705 | 0.053 | 0.442 | 0.089 | 0.102 | 0.439 | 0.102 |
| 12 | 0.719 | 0.051 | 0.051 | 0.717 | 0.052 | 0.461 | 0.091 | 0.102 | 0.458 | 0.102 |
| 14 | 0.717 | 0.045 | 0.045 | 0.716 | 0.046 | 0.473 | 0.094 | 0.104 | 0.471 | 0.105 |
| **16** | **0.710** | 0.043 | 0.044 | 0.708 | 0.046 | **0.469** | 0.107 | 0.119 | 0.467 | 0.119 |
| 18 | 0.715 | 0.046 | 0.050 | 0.713 | 0.053 | 0.456 | 0.119 | 0.134 | 0.453 | 0.133 |
| 20 | 0.729 | 0.054 | 0.060 | 0.727 | 0.063 | 0.443 | 0.125 | 0.139 | 0.440 | 0.139 |

Horizons 8 and 16 are the two-year and four-year multipliers reported in the
paper. The headline finding reproduces: the multiplier is well below one over
the full sample — about 0.67 at two years for the news shock and about 0.41 for
the Blanchard–Perotti shock.

Every point estimate lands within **0.19 published standard errors** of the
authors' value, with a median deviation of 0.03 standard errors across the two
shocks and 21 horizons. The residual gap is most likely a data-vintage
difference: `Multiplier-Standard-Errors.xlsx` is dated 21 November 2016 while
the shipped `RZDAT.xlsx` was revised on 28 November 2016.

## Why not the automatic bandwidth

Ramey and Zubairy run `ivreg2 …, robust bw(auto)`. Both that and
`CovarianceMatrices.Bartlett{NeweyWest}` claim to implement the Newey–West
(1994) automatic bandwidth, and they agree on the kernel, on what the bandwidth
parameter means, and on the absence of a degrees-of-freedom correction — but
not on the number:

| h | ``S_T`` selected by `Bartlett{NeweyWest}` | bandwidth implied by RZ's published s.e. |
|---:|---:|---:|
| 0 | 15.6 | 19.5 |
| 2 | 3.1 | 30.0 |
| 4 | 8.3 | 28.5 |
| 8 | 14.3 | 29.0 |
| 12 | 15.9 | 30.5 |
| 16 | 16.5 | 30.0 |
| 20 | 16.7 | 29.5 |

The implied bandwidth is flat at ``\approx 30`` while the data-driven one
swings between 3 and 17, even though the horizon-``h`` residual is MA(``h``) by
construction. Over the full grid the automatic rule gives standard errors
0.76–1.12 times the published ones; a fixed `Bartlett(30.0)` narrows that to
0.99–1.05 (Blanchard–Perotti) and 0.95–1.11 (military news).

Section 8 of [the inference guide](../inference_procedures_guide.md) gives the
exact formula each side implements and the evidence behind this table. For
local projections a fixed, horizon-aware bandwidth is the defensible default:
the MA(``h``) truncation is known a priori, so there is little to gain from
estimating it.

## The two-step multiplier

The alternative Ramey–Zubairy estimator takes the ratio of the cumulative sums
of the separately estimated impulse responses of `y` and `g`.

```julia
function twostep_multiplier(rz, shock::Symbol; hmax = 20)
    fy, fg = shock === :newsy ?
        (@formula(leads(y) ~ newsy + lags(newsy, 4) + lags(y, 4) + lags(g, 4)),
         @formula(leads(g) ~ newsy + lags(newsy, 4) + lags(y, 4) + lags(g, 4))) :
        (@formula(leads(y) ~ bp + lags(y, 4) + lags(g, 4)),
         @formula(leads(g) ~ bp + lags(y, 4) + lags(g, 4)))
    ry, rg = lp(fy, rz; horizon = hmax), lp(fg, rz; horizon = hmax)
    by = coefpath(ry; term = shock)
    bg = coefpath(rg; term = shock)
    DataFrame(h = 0:hmax, irf_y = by, irf_g = bg,
              multiplier = cumsum(by) ./ cumsum(bg))
end
```

The two estimators agree closely here — the largest gap across the 21 horizons
is 0.0031 for the news shock and 0.0004 for the Blanchard–Perotti shock —
because the one-step regressions run on the same sample as the
impulse-response regressions. Ramey and Zubairy note that in other
specifications the two diverge, and the reason is precisely that the samples
differ.

## Caveats

* `varbootstrap` does not apply to this specification: it supports OLS local
  projections only, and rejects `lpiv`.
* The Kleibergen–Paap rk Wald *F* reported by `ivreg2` is not implemented here.
  `weakivtest` returns the Montiel Olea–Pflueger effective *F* and a robust
  first-stage *F*; for the just-identified case the latter is the closer
  analogue. For the Blanchard–Perotti shock at `h = 0` the instrument *equals*
  the endogenous regressor, so the first stage is degenerate by construction and
  2SLS collapses to OLS.
* State-dependent (slack / ZLB) multipliers are not covered here. They need the
  shock interacted with the lagged state on both sides of the IV block, and the
  `recf{h}cumulg` / `expf{h}cumulg` split of `jordagk.do` — expressible as
  interactions of `cumul(g)`, but not yet tested against the published output.
