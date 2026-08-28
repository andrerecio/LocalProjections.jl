# Alternative inference for local projections: mathematical guide

This note documents the three inference procedures implemented in
`21_nasa_baseline_ewc.jl`, `22_nasa_baseline_biascorr.jl`, and
`23_nasa_baseline_bootstrap.jl`. It makes the mathematical objects in
those scripts explicit so that the code can be adapted to another
local-projection specification without changing the inferential procedure by
accident.

The guide describes **the code as it currently runs**. Where the current code
differs from the corresponding reference recommendation, the difference is
flagged explicitly.

## 1. Common local-projection setup

For outcome $y_t$, observed shock $s_t$ (`NASAshock`), and control vector
$w_t$, the horizon-$h$ local projection is

\[
y_{t+h} = \alpha_h + \theta_h s_t + \gamma_h'w_t + e_{t,h},
\qquad h=0,\ldots,H.
\]

The controls in the baseline application are four lags of the shock and four
lags of the variables selected in `baseline_lp_common.jl`. Define

\[
q_t = (1,s_t,w_t')', \qquad
Q_h = \sum_{t\in\mathcal I_h} q_tq_t',
\]

where $\mathcal I_h$ is the usable sample at horizon $h$, with size $T_h$.
Horizon-by-horizon OLS gives

\[
\widehat\delta_h
=
\begin{pmatrix}
\widehat\alpha_h\\
\widehat\theta_h\\
\widehat\gamma_h
\end{pmatrix}
= Q_h^{-1}\sum_{t\in\mathcal I_h}q_ty_{t+h}.
\]

The impulse response is the coefficient on the contemporaneous shock,
$\widehat\theta_h$. In the Julia code it is extracted by

```julia
beta = coefpath(lp_out; term=:NASAshock)
```

All three scripts apply the baseline peak normalization

\[
a = \frac{1}{\max_{0\leq h\leq H}
|\widehat\theta_h^{\,\mathrm{grdk}}|},
\qquad
\widetilde\theta_h=a\widehat\theta_h.
\]

The same fixed estimate $a$ multiplies point estimates, standard errors, and
confidence limits. The scripts therefore **condition on the estimated
normalization**: uncertainty in $a$ is not included in any band.

### What each procedure changes

| Script | Point estimate | Uncertainty estimate | Confidence interval |
|---|---|---|---|
| `21_nasa_baseline_ewc.jl` | OLS LP | EWC HAR covariance | Symmetric, normal critical value |
| `22_nasa_baseline_biascorr.jl` | Bias-corrected OLS LP | HC1 covariance of the uncorrected LP | Symmetric around corrected estimate |
| `23_nasa_baseline_bootstrap.jl` | OLS LP | HC1 studentization inside a VAR residual MBB | Pointwise percentile-$t$, generally asymmetric |

The dashed lines in all three figures are the baseline Bartlett/Newey-West
90% limits. They are comparisons only and do not enter the alternative
procedures.

## 2. Equal-weighted-cosine HAR inference

### 2.1 Regression score

At each horizon, form the OLS residual and score

\[
\widehat e_{t,h}=y_{t+h}-q_t'\widehat\delta_h,
\qquad
\widehat g_{t,h}=q_t\widehat e_{t,h}.
\]

Serial correlation in $\widehat g_{t,h}$ makes the usual
heteroskedasticity-only covariance inappropriate. EWC estimates its long-run
covariance using projections on Type-II discrete cosine basis functions.

### 2.2 Cosine projections and long-run covariance

For $j=1,\ldots,B$, define

\[
\widehat\Lambda_{h,j}
=
\sqrt{\frac{2}{T_h}}
\sum_{t=1}^{T_h}
\cos\!\left[\frac{\pi j(t-1/2)}{T_h}\right]
\widehat g_{t,h}.
\]

The equal-weighted-cosine estimator is

\[
\widehat\Omega_{h}^{\mathrm{EWC}}
=
\frac{1}{B}\sum_{j=1}^{B}
\widehat\Lambda_{h,j}\widehat\Lambda_{h,j}'.
\]

Because it is an average of outer products,
$\widehat\Omega_h^{\mathrm{EWC}}$ is positive semidefinite. Let

\[
\widehat A_h=\frac{1}{T_h}\sum_{t\in\mathcal I_h}q_tq_t'.
\]

Before the regression degrees-of-freedom adjustment, the OLS sandwich
covariance is

\[
\widehat V_{h,\mathrm{EWC}}^{\,0}
=
\frac{1}{T_h}
\widehat A_h^{-1}
\widehat\Omega_h^{\mathrm{EWC}}
\widehat A_h^{-1}.
\]

For $k_h$ estimated coefficients, the `Regress.jl` covariance interface
used by `LocalProjections.jl` applies the correlated-error
degrees-of-freedom factor:

\[
\widehat V_{h,\mathrm{EWC}}
=
\frac{T_h}{T_h-k_h}\widehat V_{h,\mathrm{EWC}}^{\,0}.
\]

The standard error $\widehat{se}_{h,\mathrm{EWC}}$ is the square root of the
diagonal element corresponding to $s_t$. `CovarianceMatrices.jl`
constructs the score projections, the `Regress.jl` covariance method
applies the sandwich and the finite-sample adjustment, and
`LocalProjections.vcov` loops over the horizon-specific models
collecting the diagonals:

```julia
cv_ewc = LocalProjections.vcov(EWC(B), lp_out)
se_ewc = stderror(cv_ewc; term=:NASAshock)
```

### 2.3 Choice of $B$ and intervals in script 21

The script uses the Lazarus--Lewis--Stock--Watson rule calibrated for a single
restriction,

\[
B=\left\lfloor 0.41T_0^{2/3}\right\rfloor,
\]

where $T_0$ is the effective sample size of the horizon-zero LP. The same
integer $B$ is held fixed at every horizon, even though $T_h=T_0-h$ in a
balanced sample.

The current 90% and 68% pointwise bands are

\[
\widehat\theta_h
\ \pm\ z_{0.95}\widehat{se}_{h,\mathrm{EWC}},
\qquad
\widehat\theta_h
\ \pm\ z_{0.84}\widehat{se}_{h,\mathrm{EWC}},
\]

with hard-coded values $z_{0.95}=1.6449$ and $z_{0.84}=0.9945$.

> **Reference-procedure distinction.** The canonical EWC test in Lazarus et
> al. (2018) uses fixed-smoothing critical values, which for one restriction
> are Student-$t_B$ critical values. Script 21 uses normal critical values.
> Replacing `Z90` and `Z68` by `quantile(TDist(B), ...)`
> would therefore be a methodological change, not a refactor.

### 2.4 What must change for another LP

Only three inputs are specification-specific:

1. the LP formula and shock coefficient name;
2. the effective horizon-zero sample $T_0$, from which $B$ is computed;
3. any deterministic response scaling.

Do not manually construct autocovariances in addition to calling `EWC(B)`;
that would apply two long-run variance corrections.

## 3. Herbst--Johannsen LP bias correction with HC1 inference

This procedure addresses small-sample bias in the **coefficient path**, not
serial correlation in the regression score.

### 3.1 Control autocovariances

Let $w_t$ contain all LP regressors except the intercept and the
contemporaneous shock. In script 22 it is taken from the horizon-zero model
matrix:

```julia
keep = [!(n in ("(Intercept)", "NASAshock")) for n in coefnames(lp_out)]
w = modelmatrix(lp_out.models[1])[:, keep]
```

For $T=T_0$, demean the controls,

\[
\widetilde w_t=w_t-\bar w,
\]

and estimate their variance and lag-$j$ autocovariance:

\[
\widehat\Sigma_0
=\frac{1}{T-1}\sum_{t=1}^{T}\widetilde w_t\widetilde w_t',
\]

\[
\widehat\Sigma_j
=\frac{1}{T-j-1}\sum_{t=1}^{T-j}
\widetilde w_t\widetilde w_{t+j}',
\qquad j=1,\ldots,H.
\]

Define the persistence adjustment

\[
\widehat c_j
=1+\operatorname{tr}
\left(\widehat\Sigma_0^{-1}\widehat\Sigma_j\right).
\]

The Julia expression `Σ0 \ Σj` computes
$\widehat\Sigma_0^{-1}\widehat\Sigma_j$ without explicitly forming an
inverse.

### 3.2 Recursive correction

Leave the impact response unchanged:

\[
\widehat\theta_0^{\,c}=\widehat\theta_0.
\]

For $h=1,\ldots,H$, recursively compute

\[
\boxed{
\widehat\theta_h^{\,c}
=
\widehat\theta_h
+\frac{1}{T-h}
\sum_{j=1}^{h}
\widehat c_j\widehat\theta_{h-j}^{\,c}
}.
\]

The recursion uses already-corrected lower-horizon responses on the right-hand
side. Replacing them with the original $\widehat\theta_{h-j}$ changes the
estimator.

This formula assumes the only horizon-related sample loss is the $h$ future
observations. If an adapted LP has horizon-specific missingness, the use of
$T-h$ must be checked rather than copied mechanically.

### 3.3 HC1 standard errors and bands

For the original horizon-$h$ OLS regression, the HC0 covariance is

\[
\widehat V_{h,\mathrm{HC0}}
=Q_h^{-1}
\left(\sum_{t\in\mathcal I_h}
\widehat e_{t,h}^{\,2}q_tq_t'\right)
Q_h^{-1}.
\]

With $k_h$ estimated coefficients, HC1 applies the degrees-of-freedom
correction

\[
\widehat V_{h,\mathrm{HC1}}
=\frac{T_h}{T_h-k_h}\widehat V_{h,\mathrm{HC0}}.
\]

Script 22 centers the bands on the corrected estimate but uses the HC1
standard error of the **uncorrected** OLS coefficient:

\[
CI_{h,1-\alpha}^{\mathrm{BCC+HC1}}
=
\left[
\widehat\theta_h^{\,c}
-z_{1-\alpha/2}\widehat{se}_{h,\mathrm{HC1}},
\ \widehat\theta_h^{\,c}
+z_{1-\alpha/2}\widehat{se}_{h,\mathrm{HC1}}
\right].
\]

Thus the code does not propagate estimation uncertainty in $\widehat c_j$,
nor the covariance across lower-horizon responses in the recursion. This
matches the convention in the `lp_var_nberma` implementation, where
bias correction changes `irs` after the ordinary LP standard errors
have been calculated.

In the figure, the shaded bands are centered on the corrected estimate
$\widehat\theta_h^{\,c}$; the dashed comparison lines are the baseline
Newey--West 90% limits around the **uncorrected** estimate
$\widehat\theta_h$, and the uncorrected path itself is drawn dotted.

### 3.4 What must change for another LP

The crucial adaptation is the definition of $w_t$: it must include every
control used by the LP and exclude only the intercept and contemporaneous
shock. Hard-coding column positions is unsafe; select columns by coefficient
name as in script 22. Also verify that $\widehat\Sigma_0$ is nonsingular.

## 4. VAR residual moving-block bootstrap with percentile-$t$ bands

This procedure uses a fitted VAR as a bootstrap data-generating process
(DGP), re-estimates the LP on each artificial sample, and uses the bootstrap
distribution of a studentized statistic.

### 4.1 Bootstrap VAR and pseudo-true response

For the panel-specific vector $Y_t\in\mathbb R^m$, fit

\[
Y_t=c+A_1Y_{t-1}+\cdots+A_pY_{t-p}+u_t,
\qquad p=4.
\]

Let $\widehat\Sigma_u$ be the fitted residual covariance, and let $L$ be its
lower-triangular Cholesky factor:

\[
\widehat\Sigma_u=LL'.
\]

`NASAshock` is ordered first. The impact vector is normalized to a unit
impact change in the shock:

\[
v=\frac{Le_1}{e_1'Le_1}=\frac{L_{:,1}}{L_{11}}.
\]

If $C$ is the VAR companion matrix and

\[
b=(v',0',\ldots,0')',
\]

then the VAR-implied response of outcome $r$ is

\[
\theta_{h}^{\mathrm{VAR}}=e_r'C^hb.
\]

This is the **pseudo-true value** used to center the bootstrap statistic. It
is not the real-data LP estimate.

### 4.2 Moving-block resampling of VAR residuals

Let $T_u$ be the number of fitted VAR residuals and choose

\[
\ell=\left\lceil 5.03T^{1/4}\right\rceil,
\]

where script 23 uses the number $T$ of observations in the VAR data matrix,
not $T_u$. Set $N=\lceil T_u/\ell\rceil$.

For block $b=1,\ldots,N$, independently draw a starting index

\[
I_b\sim\operatorname{Uniform}\{0,\ldots,T_u-\ell\}.
\]

At within-block position $s=1,\ldots,\ell$, the uncentered resampled residual
is

\[
\widehat u^{*,\mathrm{unc}}_{(b-1)\ell+s}
=\widehat u_{I_b+s}.
\]

The position-specific mean is

\[
\bar u_s
=\frac{1}{T_u-\ell+1}
\sum_{r=0}^{T_u-\ell}\widehat u_{s+r}.
\]

The centered bootstrap residual is

\[
\widehat u^*_{(b-1)\ell+s}
=\widehat u^{*,\mathrm{unc}}_{(b-1)\ell+s}-\bar u_s.
\]

After concatenation, the sequence is truncated to $T_u$ observations. This
position-specific recentering is part of the
Brüggemann--Jentsch--Trenkler procedure; subtracting only the overall
residual mean is not equivalent.

### 4.3 Bootstrap data and LP estimates

For each draw:

1. Draw one of the $T-p+1$ contiguous blocks of $p$ observations from the
   real data as initial conditions.
2. Iterate the fitted VAR with those initial conditions and the resampled
   residuals to produce $Y_1^*,\ldots,Y_T^*$.
3. Re-estimate exactly the same horizon-by-horizon LP on the artificial data.
4. Store its shock coefficient $\widehat\theta_{b,h}^*$ and HC1 standard
   error $\widehat{se}_{b,h}^*$.

The bootstrap statistic is

\[
t_{b,h}^*
=
\frac{
\widehat\theta_{b,h}^*-\theta_h^{\mathrm{VAR}}
}{
\widehat{se}_{b,h}^*
}.
\]

Centering at $\theta_h^{\mathrm{VAR}}$ is essential because the fitted VAR,
not the unrestricted LP, generates the bootstrap sample.

### 4.4 Hall percentile-$t$ confidence interval

Let $q_{h,a}^*$ denote the empirical $a$-quantile of the bootstrap statistics
$\{t_{b,h}^*\}_{b=1}^{R}$. For real-data LP estimate $\widehat\theta_h$ and
real-data HC1 standard error $\widehat{se}_h$, the pointwise $1-\alpha$
interval is

\[
\boxed{
CI_{h,1-\alpha}^{\mathrm{boot}}
=
\left[
\widehat\theta_h-\widehat{se}_h q_{h,1-\alpha/2}^*,
\ \widehat\theta_h-\widehat{se}_h q_{h,\alpha/2}^*
\right]
}.
\]

The reversal of the quantiles follows from solving the studentized inequality
for $\theta_h$. The resulting interval need not be symmetric around
$\widehat\theta_h$.

Script 23 uses $R=1000$, quantiles $0.05/0.95$ for 90% bands, and
$0.16/0.84$ for 68% bands. The bands are **pointwise across horizons**, not
simultaneous confidence bands.

### 4.5 Current implementation versus the replication recommendation

The core resampling and percentile-$t$ formulas follow
`lp_var_nberma`, but script 23 implements a simpler variant:

| Component | Script 23 | `lp_var_nberma` recommendation |
|---|---|---|
| Lag length | Fixed $p=4$ | Fixed by the researcher or selected by VAR AIC |
| Bootstrap VAR coefficients | OLS | Pope (1990) bias-corrected VAR coefficients |
| Real and bootstrap LP estimates | Uncorrected OLS LP | Normally Herbst--Johannsen bias-corrected LP |
| LP studentizing SE | HC1 | Same: the Eicker--Huber--White sandwich with the $T_h/(T_h-k_h)$ adjustment in `linreg.m` **is** HC1 |
| Bootstrap interval | Hall percentile-$t$ | Same: Hall percentile-$t$ recommended |

Therefore, adding VAR coefficient bias correction or calling
`lp_biascorr` inside script 23 would change the implemented procedure.
If either is added, it must be applied consistently when computing the
pseudo-truth, simulating every bootstrap sample, estimating the real-data LP,
and estimating each bootstrap LP.

### 4.6 Exact implementation in `lp_var_nberma`

The recommended replication call in
`docs/code/lp_var_nberma-main/README.md` is equivalent to

```matlab
opts_lp = {'resp_ind',  2, ...
           'innov_ind', 1, ...
           'estimator', 'lp', ...
           'shrinkage', false, ...
           'bias_corr', true, ...
           'alpha',     0.05, ...
           'se_homosk', false};

block_length = ceil(5.03*size(data_y,1)^(1/4));

[irs, ses, cis, cis_boot] = ir_estim( ...
    data_y, p, horzs, opts_lp{:}, ...
    'bootstrap', 'var', ...
    'boot_num', 1000, ...
    'boot_blocklength', block_length);
```

The single option `bias_corr=true` controls **two different analytical
corrections**:

1. Herbst--Johannsen correction of the real-data and bootstrap LP estimates;
2. Pope correction of the VAR slope coefficients used as the bootstrap DGP.

The implementation proceeds as follows.

#### Step 1: estimate and bias-correct the real-data LP

In the non-shrinkage LP branch of `ir_estim.m`, every horizon calls

```matlab
[the_irs_all, the_irs_all_varcov, ...] = ...
    lp_ir_estim(Y, p, the_horz, resp_ind, innov_ind, ...
                se_homosk, no_const);
```

`lp_ir_estim.m` runs the horizon-specific OLS regression and calls
`linreg.m` for its covariance. With
`se_homosk=false`, `linreg.m` constructs the Eicker--Huber--White
sandwich and multiplies it by the finite-sample factor $T_h/(T_h-k_h)$.
Therefore, at this point,

\[
\widehat\theta_h^{\,\mathrm{OLS}}
\quad\text{and}\quad
\widehat{se}_{h,\mathrm{HC1}}
\]

have been stored for all horizons.

After the horizon loop, `ir_estim.m` checks
`bias_corr`. When it is true, the code constructs

```matlab
w = [Y(:, 1:innov_ind-1), lagmatrix(Y, 1:p)];
w = w(p+1:end, :);
irs = lp_biascorr(irs, w);
```

Thus $w_t$ contains the contemporaneous variables ordered before the
innovation and $p$ lags of all VAR variables, but excludes the
contemporaneous innovation itself. For an observed shock ordered first,
`innov_ind=1`, the first block is empty and $w_t$ consists only of the
$p$ lags.

`lp_biascorr.m` then replaces the OLS path with

\[
\widehat\theta_0^{\,c}=\widehat\theta_0^{\,\mathrm{OLS}},
\]

\[
\widehat\theta_h^{\,c}
=
\widehat\theta_h^{\,\mathrm{OLS}}
+\frac{1}{T-h}
\sum_{j=1}^{h}
\left[
1+\operatorname{tr}
\left(\widehat\Sigma_0^{-1}\widehat\Sigma_j\right)
\right]
\widehat\theta_{h-j}^{\,c}.
\]

In these equations, $T=\operatorname{size}(w,1)$, which is the control
sample after removing the first $p$ observations—not the full row count of
`Y` used later by the Pope correction.

The standard errors are **not recomputed after this correction**. The
real-data objects passed to the bootstrap interval routine are therefore

\[
\texttt{irs}_h=\widehat\theta_h^{\,c},
\qquad
\texttt{ses}_h=\widehat{se}_{h,\mathrm{HC1}}.
\]

#### Step 2: estimate the VAR bootstrap DGP

Because `bootstrap='var'`, the bootstrap section of
`ir_estim.m` calls

```matlab
[irs_var, ~, Ahat_var, ~, res_var] = var_ir_estim( ...
    Y, innov_ind, p, horzs, ...
    bias_corr, se_homosk, no_const);
```

Inside `var_ir_estim.m`:

1. `var_estim.m` estimates the reduced-form VAR by OLS;
2. it stores the OLS residuals $\widehat u_t$;
3. it estimates
   \[
   \widehat\Sigma_u
   =\frac{\sum_t\widehat u_t\widehat u_t'}
          {T_u-k_{\mathrm{VAR}}};
   \]
4. because `bias_corr=true`, it sends only the VAR slope matrices, not
   the intercept, to `var_biascorr.m`.

The residuals and $\widehat\Sigma_u$ are computed from the original OLS VAR.
They are **not recomputed** after the Pope correction. The corrected slope
matrices are combined with the original intercept and OLS residuals when
bootstrap samples are generated.

#### Step 3: apply the Pope correction

Let

\[
\widehat A=(\widehat A_1,\ldots,\widehat A_p)
\]

be the OLS VAR slopes, and define their companion form

\[
\mathcal A
=
\begin{pmatrix}
\widehat A_1 & \cdots & \widehat A_p\\
I            & \cdots & 0
\end{pmatrix}.
\]

Let

\[
G=\operatorname{diag}
\left(\widehat\Sigma_u,0,\ldots,0\right),
\]

and compute the companion-state covariance $\Gamma_0$ from the discrete
Lyapunov equation

\[
\Gamma_0=\mathcal A\Gamma_0\mathcal A'+G.
\]

The code in `var_biascorr.m` constructs

\[
M
=
(I-\mathcal A')^{-1}
+\mathcal A'(I-\mathcal A'\mathcal A')^{-1}
+\sum_{\lambda\in\operatorname{eig}(\mathcal A)}
\lambda(I-\lambda\mathcal A')^{-1},
\]

followed by

\[
b=G\,M\,\Gamma_0^{-1}.
\]

The Pope-corrected companion matrix is

\[
\mathcal A^{\,c}=\mathcal A+\frac{b}{T}.
\]

Here $T$ is the full row count of `Y` from
`T = size(Y,1)` in `var_ir_estim.m`, not the shorter number
$T_u=T-p$ of fitted VAR residuals.

Only the first $n$ rows are returned as the corrected VAR slope matrices.
The correction has two stability safeguards:

1. if the original companion matrix has an eigenvalue outside the unit
   circle, `var_biascorr.m` returns the original OLS slopes;
2. if the fully corrected matrix is unstable, the code replaces
   $b/T$ by $\delta b/T$ and lowers $\delta$ from 1 toward 0 in increments
   of 0.01 until stability is restored.

Thus requesting `bias_corr=true` does not guarantee a full Pope
correction: it can be skipped or attenuated when required by these stability
checks.

#### Step 4: compute the corrected-VAR pseudo-truth

`var_ir_estim.m` computes the identified VAR response using the
corrected slopes $\widehat A^{\,c}$ and a shock vector estimated from the
original OLS VAR residuals. The selected response is stored as

```matlab
pseudo_truth = irs_var(resp_ind, :);
```

Therefore,

\[
\theta_h^{\mathrm{pseudo}}
=
\theta_h^{\mathrm{VAR}}
\left(\widehat A^{\,c},\widehat\Sigma_u\right),
\]

not the real-data corrected LP response
$\widehat\theta_h^{\,c}$.

#### Step 5: generate each bootstrap sample from the corrected VAR

For draw $b$, `ir_estim.m` calls

```matlab
Y_boot = var_boot(Ahat_var, res_var, Y, p, ...
                  boot_blocklength, no_const);
```

`var_boot.m`:

1. resamples blocks of the original OLS VAR residuals;
2. applies position-specific recentering;
3. draws a contiguous block of $p$ real observations as initial conditions;
4. simulates
   \[
   Y_t^*
   =\widehat c
   +\sum_{\ell=1}^{p}\widehat A_\ell^{\,c}Y_{t-\ell}^*
   +\widehat u_t^*.
   \]

Hence the bootstrap DGP combines:

- Pope-corrected VAR slopes;
- the uncorrected OLS intercept, when an intercept is included;
- centered blocks drawn from the original OLS VAR residuals.

#### Step 6: estimate a bias-corrected LP in every bootstrap sample

For every artificial sample, the code calls

```matlab
[estims_boot(b,:), ses_boot(b,:)] = ...
    ir_estim(Y_boot, p, horzs, varargin{:});
```

The original `varargin` is passed again, including
`estimator='lp'`, `bias_corr=true`, and
`se_homosk=false`. Consequently, each bootstrap replication:

1. estimates the horizon-specific OLS LPs;
2. computes their HC1 standard errors;
3. applies `lp_biascorr.m` to the entire bootstrap LP path;
4. returns
   \[
   \widehat\theta_{b,h}^{*,c}
   \quad\text{and}\quad
   \widehat{se}_{b,h,\mathrm{HC1}}^*.
   \]

Although `varargin` still contains `bootstrap='var'`, this call
does **not** start a bootstrap inside the bootstrap. It requests only two
outputs. The line

```matlab
if nargout <= 2
    return;
end
```

is reached immediately after point estimates and standard errors are
computed, before the bootstrap section of `ir_estim.m`.

#### Step 7: construct the percentile-$t$ interval

`boot_ci.m` forms

\[
t_{b,h}^*
=
\frac{
\widehat\theta_{b,h}^{*,c}
-\theta_h^{\mathrm{pseudo}}
}{
\widehat{se}_{b,h,\mathrm{HC1}}^*
}.
\]

If $q_{h,a}^*$ is the empirical $a$-quantile of these statistics, the
recommended Hall percentile-$t$ interval is

\[
\left[
\widehat\theta_h^{\,c}
-\widehat{se}_{h,\mathrm{HC1}}q_{h,1-\alpha/2}^*,
\quad
\widehat\theta_h^{\,c}
-\widehat{se}_{h,\mathrm{HC1}}q_{h,\alpha/2}^*
\right].
\]

The three centers in this calculation must not be confused:

| Object | Center or estimate used |
|---|---|
| Reported real-data response | Bias-corrected LP, $\widehat\theta_h^{\,c}$ |
| Bootstrap response | Bias-corrected bootstrap LP, $\widehat\theta_{b,h}^{*,c}$ |
| Center of bootstrap $t$-statistic | Pope-corrected VAR pseudo-truth, $\theta_h^{\mathrm{pseudo}}$ |

This is the complete recommended `lp_var_nberma` combination:
**Pope-corrected VAR DGP + bias-corrected LP in the real and bootstrap
samples + HC1 studentization + Hall percentile-$t$ inversion**.

## 5. Safe adaptation checklist

When adapting scripts 21--23 to another local projection, verify the following
in order.

### Common to all three procedures

- The outcome, contemporaneous shock, lags, controls, transformations, and
  sample restriction match the target LP exactly.
- The coefficient extracted from every fitted model is the contemporaneous
  shock coefficient.
- The normalization is applied once, after inference quantities are computed.
- If the normalization is estimated and its uncertainty should matter, the
  current fixed-scale convention is no longer sufficient.
- The reported bands are described as pointwise, unless a separate
  simultaneous-band construction is added.

### EWC

- Compute $B$ from the intended sample size and decide explicitly whether it
  is fixed across horizons.
- Decide explicitly between the repository's normal critical values and the
  reference EWC $t_B$ critical values.
- Confirm that `EWC(B)` is applied to the target LP models, not to a
  separate regression with a different sample.

### Bias correction

- Construct $w_t$ from the actual LP design and remove only the intercept and
  contemporaneous shock.
- Keep the controls in their regression units; do not standardize only for the
  bias correction.
- Check the rank and conditioning of $\widehat\Sigma_0$.
- Check whether the usable horizon-$h$ sample is actually $T-h$.
- Preserve the recursion through corrected lower-horizon responses.

### Moving-block bootstrap

- The VAR data vector must contain the shock and all variables needed to
  reconstruct the LP formula in every draw.
- Keep the identifying ordering explicit; the current code requires
  `NASAshock` first.
- Use the same variable transformations before fitting the VAR and LP.
- Preserve position-specific residual recentering and random contiguous
  initial conditions.
- Refit the complete LP and HC1 covariance in every draw.
- If reproducing the recommended `lp_var_nberma` procedure, apply the
  Pope correction to the bootstrap VAR slopes and the Herbst--Johannsen
  correction to both the real-data and every bootstrap LP path.
- Center bootstrap $t$-statistics at the VAR-implied response, not at the
  real-data LP response.
- Record failed draws and ensure the number of successful draws remains
  adequate at every horizon.
- Set and report the random seed, number of replications, VAR lag length, and
  block length.

## 6. Audit of `LocalProjections.jl` and `Regress.jl`

This audit was run on 2026-08-27 against the exact package trees pinned in
`Manifest.toml`:

- `LocalProjections.jl` 0.1.0, tree
  `801aa26dafe961f7593c1f5bb4f7667f61343658`;
- `Regress.jl` 0.1.0, tree
  `3160eb8b56d369373cfc4452855113f160da3bd5`;
- `CovarianceMatrices.jl` 0.30.5, tree
  `e431c291f614f63752a208ac03fd23d056fc454a`.

The review traced the call path

```text
LocalProjections.vcov
    -> CovarianceMatrices.vcov dispatched to the Regress model method
        -> CovarianceMatrices.aVar / EWC cosine projections
```

and compared the results with direct matrix calculations and the routines in
`docs/code/lp_var_nberma-main/_estim/`.

### 6.1 Findings summary

| ID | Classification | Component | Effect on scripts 21--23 |
|---|---|---|---|
| LP-1 | Confirmed package bug | Missing-value handling before lag/lead construction | No effect on the present complete baseline sample; dangerous for adapted LPs with internal missing values |
| REG-1 | Confirmed package API bug | `dofadjust=false` is ignored for HAC/EWC | No effect because the scripts use the default adjustment |
| EWC-1 | Confirmed method-specific interval mismatch | Normal rather than Student-$t_B$ critical values | Affects the EWC bands in script 21 relative to the Lazarus et al. procedure |

No error was found in the default OLS coefficient path, HC1 covariance, EWC
cosine covariance, BCC recursion, moving-block residual recentering, or
percentile-$t$ inversion.

### 6.2 LP-1: internal missing values are removed before lags are built

**Status: confirmed bug in the pinned `LocalProjections.jl`.**

The `lp` implementation first applies
`dropmissing(df_base, base_vars)` and only afterward constructs the RHS
lags and LHS leads. Consequently, an internal missing observation is deleted
from the time axis before lagging. Observations on opposite sides of the gap
then become adjacent.

For example, consider

\[
y=(1,2,\mathrm{missing},4,5,6)'.
\]

For the formula

```julia
@formula(leads(y) ~ shock + lags(y, 1))
```

the pinned package constructs the following horizon-zero rows:

| Current $y_t$ | Constructed lag |
|---:|---:|
| 2 | 1 |
| 4 | 2 |
| 5 | 4 |
| 6 | 5 |

The row for $y_t=4$ should not be usable because its true time-series lag is
missing. Instead, the package assigns it the value 2 from two periods earlier.
This changes both the regression sample and the regressors.

The correct order of operations for time-series semantics is:

1. preserve the original ordered time axis;
2. construct lags and leads on that axis;
3. remove rows whose transformed regression variables are incomplete.

**Impact on this repository.** The 1955--2019 baseline contains 260
quarterly-contiguous observations, with no missing values in any series used
by scripts 21--23. LP-1 therefore does not change their current estimates.

**Safe adaptation rule.** Before calling `lp`, either reject internal
missing values explicitly or preserve their positions as `NaN` values
so that the package's post-transformation `isnan` filter removes the
correct current and lagged/led rows. Do not rely on `lp` to construct
time-correct lags from columns containing Julia `missing` values.

### 6.3 REG-1: the `dofadjust` keyword is accepted but ignored

**Status: confirmed API bug in the pinned `Regress.jl`.**

The model-level covariance method has the signature

```julia
CovarianceMatrices.vcov(estimator, model; dofadjust=true, ...)
```

but its scale calculation does not branch on `dofadjust`. For HAC and
correlated estimators such as EWC,

```julia
vcov(estimator, model; dofadjust=false)
```

returns exactly the same matrix as the default call.

In an $n=80$, $k=3$ numerical check:

- package HC1 versus the direct HC1 formula differed by at most
  $6.9\times10^{-17}$;
- package EWC versus the direct EWC formula **with** the
  $n/(n-k)$ adjustment differed by at most $4.2\times10^{-17}$;
- `dofadjust=false` versus the default differed by exactly zero for
  both Bartlett HAC and EWC;
- the returned EWC matrix differed from the requested unadjusted matrix by
  up to 0.0032.

The same equality for HC1 is not separate evidence of a bug: the
$n/(n-k)$ factor defines HC1 itself, so requesting HC1 should not turn it
into HC0. REG-1 concerns the additional finite-sample correction applied by
the package to HAC/EWC even when `dofadjust=false`.

Thus the default covariance used in scripts 21--23 is correct and agrees with
the finite-sample factor in `lp_var_nberma/_estim/linreg.m`. The bug is
that callers cannot turn that adjustment off using the advertised keyword.

There is a second API constraint at the LP level:
`LocalProjections.vcov(estimator, lp_obj)` does not forward keyword
arguments to the horizon-specific `Regress.jl` models. Even after
REG-1 is fixed, disabling the adjustment through the LP wrapper would require
that wrapper to accept and forward the keyword.

### 6.4 EWC-1: covariance is correct, but the canonical critical value is not used

**Status: confirmed inference mismatch for canonical EWC testing.**

The EWC score projections and sandwich covariance produced by
`CovarianceMatrices.jl` and `Regress.jl` match a direct
implementation:

- EWC long-run covariance maximum difference:
  $1.8\times10^{-15}$;
- final EWC regression covariance maximum difference:
  $4.2\times10^{-17}$.

The problem is interval construction. The pinned `LocalProjections.jl`
uses

```julia
quantile(Normal(), 0.5 + level / 2)
```

unconditionally in `summarize`, the plot recipe, and
`as_irf_result`. It does so even when the covariance object contains an
`EWC(B)` estimator. Lazarus et al. (2018) instead pair the EWC
long-run variance estimator with fixed-smoothing Student-$t_B$ critical
values for one restriction.

Script 21 bypasses those package helpers, but independently hard-codes the
same normal critical values. In the current baseline,

\[
T_0=256,\qquad
B=\lfloor0.41T_0^{2/3}\rfloor=16.
\]

The relevant comparisons are:

| Band | Normal critical value | $t_{16}$ critical value | Canonical/current half-width |
|---|---:|---:|---:|
| 90% | 1.6449 | 1.7459 | 1.0614 |
| 68% | 0.9945 | 1.0263 | 1.0321 |

Hence the canonical 90% EWC half-width is 6.14% wider than script 21's, and
the canonical 68% half-width is 3.21% wider (equivalently, the current bands
are 5.79% and 3.11% narrower than the canonical ones). This
does not mean the EWC **standard errors** are wrong; it means they are paired
with the wrong reference distribution if the target is the complete Lazarus
et al. test.

Using `TDist(B)` would be an inference correction relative to that
reference, but it would change the current figures and must be treated as a
methodological change.

### 6.5 Results checked against `lp_var_nberma`

The following parts agree with the replication code:

- **OLS LP coefficients.** Synthetic horizon-by-horizon coefficients from
  `LocalProjections.jl` matched direct OLS calculations within
  $5.6\times10^{-17}$.
- **HC1/EHW covariance.** `Regress.jl` matches
  `linreg.m`: the Eicker--Huber--White sandwich is multiplied by
  $T/(T-k)$.
- **BCC.** Script 22 reproduces `lp_biascorr.m`, including the
  denominators in $\widehat\Sigma_j$, the trace adjustment, and recursion
  through already-corrected lower-horizon responses.
- **Residual MBB.** Script 23 reproduces `var_boot.m` block sampling,
  position-specific residual recentering, and random contiguous initial
  conditions.
- **Percentile-$t$.** Script 23 reproduces `boot_ci.m`, including
  centering the bootstrap statistic at the VAR-implied pseudo-truth and
  reversing its quantiles when the interval is inverted.

Two previously documented differences in script 23 remain **method choices,
not package bugs**:

1. the bootstrap DGP uses uncorrected OLS VAR coefficients rather than the
   Pope (1990) bias-corrected coefficients used by the recommended
   `lp_var_nberma` configuration;
2. the real-data and bootstrap LPs are uncorrected rather than applying the
   Herbst--Johannsen correction used by that configuration.

These choices mean script 23 is a simpler variant of the replication
procedure. They should be changed only if exact replication of the recommended
algorithm is the goal.

## 7. Source-to-code map

The local files most useful for checking an implementation are:

- `baseline_lp_common.jl`: sample, transformations, LP controls, panel
  order, peak normalization, and plotting convention;
- `21_nasa_baseline_ewc.jl`: EWC bandwidth and covariance calls;
- `22_nasa_baseline_biascorr.jl`: direct Julia port of the BCC recursion;
- `23_nasa_baseline_bootstrap.jl`: VAR pseudo-truth, residual MBB, and
  percentile-$t$ inversion;
- `docs/code/lp_var_nberma-main/_estim/lp_ir_estim.m`: MATLAB LP design
  and Eicker--Huber--White covariance;
- `docs/code/lp_var_nberma-main/_estim/lp_biascorr.m`: original BCC
  routine ported by script 22;
- `docs/code/lp_var_nberma-main/_estim/var_estim.m`: OLS VAR slopes,
  intercept, residuals, and residual covariance;
- `docs/code/lp_var_nberma-main/_estim/var_biascorr.m`: Pope analytical
  correction and its stability safeguards;
- `docs/code/lp_var_nberma-main/_estim/var_ir_estim.m`: combines the
  corrected slopes with the OLS residual covariance and constructs the VAR
  pseudo-truth;
- `docs/code/lp_var_nberma-main/_estim/var_boot.m`: original MBB routine
  ported by script 23;
- `docs/code/lp_var_nberma-main/_estim/boot_ci.m`: Efron, Hall, and Hall
  percentile-$t$ formulas;
- `docs/code/lp_var_nberma-main/_estim/ir_estim.m`: orchestration and use
  of the VAR-implied pseudo-truth.

Primary methodological references:

- Lazarus, Lewis, Stock, and Watson (2018), [HAR Inference: Recommendations
  for Practice](https://doi.org/10.1080/07350015.2018.1506926).
- Herbst and Johannsen (2024), [Bias in Local
  Projections](https://doi.org/10.1016/j.jeconom.2024.105655).
- Brüggemann, Jentsch, and Trenkler (2016), [Inference in VARs with
  Conditional Heteroskedasticity of Unknown
  Form](https://doi.org/10.1016/j.jeconom.2015.10.004).
- Jentsch and Lunsford (2019), [Asymptotically Valid Bootstrap Inference for
  Proxy SVARs](https://doi.org/10.26509/frbc-wp-201908).
- Montiel Olea, Plagborg-Møller, Qian, and Wolf (2025), [Local Projections or
  VARs? A Primer for
  Macroeconomists](https://economics.mit.edu/sites/default/files/2025-03/lp_var_primer.pdf)
  and its
  [supplement](https://economics.mit.edu/sites/default/files/2025-03/lp_var_primer_supplement.pdf).
