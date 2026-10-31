# Bilal & Känzig temperature-shock data (Plagborg-Møller's course extract)

`bilal_kanzig.csv` holds the annual series of the running example of Mikkel
Plagborg-Møller's short course *Local projections for applied macroeconomics*
(2026, MIT licence, `stata/micc_data.dta`), which is based on the empirical
application of Bilal, A. and D. R. Känzig (2026), "The Macroeconomic Impact of Climate
Change: Global vs. Local Temperature" (replication package
`github.com/dkaenzig/micc_replication`).

The extract keeps the years the course's `load_data.do` keeps (1960–2019, 60 rows, no
missing values) and the columns its `settings.do` and `lecture1.do` use:

| column | Stata label | role in the course |
|---|---|---|
| `year` | | time index |
| `lnrgdppc_world_pwt` | log world real GDP per capita (national accounts), PWT (×100) | outcome `y` |
| `gtmp_bkly_aw_dtfe2s` | Hamilton filter (h = 2, p = 2) forecast error of global temperature, Berkeley | impulse `x` |
| `gtmp_bkly_aw` | global absolute temperature, Berkeley (°C) | normalization variable of the LP-IV |
| `recessiondates` | major global recession dates indicator | contemporaneous control |
| `lnpoil_wti` | log WTI crude oil price (×100) | lagged control |
| `treasury1y` | 1-year US Treasury yield | lagged control |

The values were exported from the `.dta` with full precision (`format %21.17g`,
`export delimited ..., datafmt`) and rewritten as the shortest decimals that parse to
the same doubles. `lnrgdppc_world_pwt`, `recessiondates` and `lnpoil_wti` are Stata
`float`s, so their doubles are the exact float values Stata computes with.

The course's specification, in the package's formula language:

```julia
lp(@formula(leads(lnrgdppc_world_pwt) ~ gtmp_bkly_aw_dtfe2s + recessiondates +
            lags(recessiondates, 2) + lags(lnrgdppc_world_pwt, 2) +
            lags(gtmp_bkly_aw_dtfe2s, 2) + lags(lnpoil_wti, 2) + lags(treasury1y, 2)),
   bk; horizon = 10)
```
