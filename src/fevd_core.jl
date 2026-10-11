# ============================================================================
# FEVD matrix core: R², LP-A/LP-B estimators, gradients, asymptotic variances
# ============================================================================

"""
    _gl_bandwidth(n) -> Int

Newey–West lag of Gorodnichenko and Lee (2020) for a sample of `n` rows,
``L = \\mathrm{round}(0.75\\, n^{1/3} - 1)`` (`SubR2.m`, `SubLPA.m`), floored
at zero as `lrv_nw.m` does. MATLAB rounds half away from zero, and the cube
root is taken with `^` rather than `cbrt`, which rounds differently at
`n = 1000`. `lrv_nw(⋅, L)` corresponds to CovarianceMatrices' `Bartlett(L + 1)`.
"""
_gl_bandwidth(n::Integer) = max(round(Int, 0.75 * n^(1 / 3) - 1, RoundNearestTiesAway), 0)

"""
    _gl_lrv_nw(Z, L) -> Matrix{Float64}

Bartlett long-run variance of the rows of `Z`, a port of `lrv_nw.m`:
``\\Gamma_0 + \\sum_{i=1}^{L} (1 - i/(L+1)) (\\Gamma_i + \\Gamma_i')`` with
``\\Gamma_i = \\sum_{t=i+1}^{n} z_t z_{t-i}' / n``. No demeaning and no
degrees-of-freedom factor; `L ≤ 0` returns ``\\Gamma_0``.
"""
function _gl_lrv_nw(Z::AbstractMatrix{Float64}, L::Integer)
    n = size(Z, 1)
    V = (Z' * Z) ./ n
    for i in 1:min(L, n - 1)
        Γ = (view(Z, (i + 1):n, :)' * view(Z, 1:(n - i), :)) ./ n
        w = 1 - i / (L + 1)
        V .+= w .* (Γ .+ Γ')
    end
    return V
end

"""
    _gl_prewhitened_lrv(Z, L) -> Matrix{Float64}

VAR(1)-prewhitened long-run variance (`SubR2.m:213-220`,
`SimulationAVarLPA.m:249-256`): ``z_t = A z_{t-1} + u_t`` by least squares
without intercept, then ``(I - A)^{-1}\\, \\mathrm{lrv}(U, L)\\, (I - A')^{-1}``.
`lrv(U, L)` divides by the ``n - 1`` residual rows while `L` comes from `n`, and
``A`` is not capped (no Andrews–Monahan eigenvalue adjustment), as in the
reference. The VAR is solved by QR rather than the normal equations.
"""
function _gl_prewhitened_lrv(Z::AbstractMatrix{Float64}, L::Integer)
    n = size(Z, 1)
    Z0 = Z[1:(n - 1), :]
    Z1 = Z[2:n, :]
    A = Matrix(transpose(Z0 \ Z1))
    U = Z1 - Z0 * transpose(A)
    IA = I - A
    return (IA \ _gl_lrv_nw(U, L)) / transpose(IA)
end

# Covariance of the moment estimator, G⁻¹ Ω G⁻ᵀ / n (the reference's `GOGT`),
# from a factorization of G that the two calls of a horizon (with and without
# prewhitening) share
function _gl_gogt(F::Factorization, Ω::AbstractMatrix, n::Integer)
    X = F \ Ω
    return Matrix(transpose(F \ Matrix(transpose(X)))) ./ n
end
_gl_gogt(G::AbstractMatrix, Ω::AbstractMatrix, n::Integer) = _gl_gogt(lu(G), Ω, n)

# Delta-method standard error; NaN when the quadratic form is negative or NaN
function _gl_delta_se(g::AbstractVector{Float64}, V::AbstractMatrix{Float64})
    v = dot(g, V * g)
    return v >= 0 ? sqrt(v) : NaN
end

"""
    _gl_r2_gradient(θ, s) -> Vector{Float64}

Gradient of the R² share ``\\theta_2'\\theta_1/\\theta_3`` with respect to
``\\theta = (\\theta_1, \\theta_2, \\theta_3)`` (`SubR2.m`,
`SimulationAVarR2.m:180-190`): ``[\\theta_2', \\theta_1', -s] / \\theta_3``.
Exact at ``s = \\hat s``; the simulation method plugs in the bias-corrected share.
"""
function _gl_r2_gradient(θ::AbstractVector{Float64}, s::Float64)
    k = (length(θ) - 1) ÷ 2
    return vcat(θ[(k + 1):(2k)], θ[1:k], -s) ./ θ[end]
end

"""
    _gl_r2(y, W, xf; avar = true) -> NamedTuple

The R² estimator at one horizon, a port of `SubR2.m:157-230`. Rows are the
horizon-``h`` window ``t_0, \\dots, T-h``: `y` holds ``y_{t+h} - y_{t-1}``,
`W` the information set ``W_{t-1}`` (intercept included) and `xf` the demeaned
shocks ``x_t, \\dots, x_{t+h}``. With ``\\hat f`` the residual of `y` on `W`
and ``n`` the number of rows,

```math
\\hat\\Sigma_x = \\frac{X_f'X_f}{n},\\quad \\hat\\theta_2 = \\frac{X_f'\\hat f}{n},\\quad
\\hat\\theta_1 = \\hat\\Sigma_x^{-1}\\hat\\theta_2,\\quad \\hat\\theta_3 = \\frac{\\hat f'\\hat f}{n},\\quad
\\hat s = \\frac{\\hat\\theta_2'\\hat\\theta_1}{\\hat\\theta_3},
```

the uncentred ``R^2`` of ``\\hat f`` on ``X_f``. With `avar = true` the result
also holds the covariance of ``\\hat\\theta = (\\hat\\theta_1, \\hat\\theta_2,
\\hat\\theta_3)`` with and without prewhitening (`avar_pw`, `avar`, i.e.
``G^{-1}\\hat\\Omega G^{-\\prime}/n``) and the delta-method standard errors of
``\\hat s`` (`se_pw`, `se`). ``\\hat f`` is treated as data: the estimation
error of the forecast-error regression is ignored, which is valid when the
shock is unpredictable from ``W_{t-1}``. The prewhitened quantities are `NaN`
when the ``2h+3`` moments cannot be prewhitened on ``n - 1`` rows
(`feasible = false`).
"""
function _gl_r2(y::AbstractVector{Float64}, W::AbstractMatrix{Float64},
        xf::AbstractMatrix{Float64}; avar::Bool = true)
    n = length(y)
    k = size(xf, 2)
    fe = y - W * (W \ y)
    Σx = (xf' * xf) ./ n
    θ2 = (xf' * fe) ./ n
    θ1 = Σx \ θ2
    θ3 = dot(fe, fe) / n
    s = dot(θ2, θ1) / θ3
    θ = vcat(θ1, θ2, θ3)
    avar || return (; s, theta = θ, fe)

    m = 2k + 1
    e = fe - xf * θ1
    exf = e .* xf
    Z = hcat(exf .- mean(exf; dims = 1), fe .* xf .- transpose(θ2), fe .^ 2 .- θ3)
    G = lu(-cat(Σx, Matrix{Float64}(I, k + 1, k + 1); dims = (1, 2)))
    ξ = _gl_r2_gradient(θ, s)
    L = _gl_bandwidth(n)
    V = _gl_gogt(G, _gl_lrv_nw(Z, L), n)
    feasible = m < n - 1
    V_pw = feasible ? _gl_gogt(G, _gl_prewhitened_lrv(Z, L), n) : fill(NaN, m, m)
    return (; s, theta = θ, fe, avar_pw = V_pw, avar = V,
        se_pw = _gl_delta_se(ξ, V_pw), se = _gl_delta_se(ξ, V), bandwidth = L, feasible)
end

"""
    _gl_lp(dhy, C, W, q, x, σ²x, t0, t0lp, H; estimator, dofadjust, irf_se = true)

LP-A / LP-B estimators for all horizons, a port of `SubLPA.m` / `SubLPB.m`
(`semIndicator = 0`). `dhy` is ``T \\times (H+1)`` with column ``h+1`` equal to
``y_{t+h} - y_{t-1}``; `C` the contemporaneous regressors (the demeaned shock
first), `W` the information set ``W_{t-1}``, `x` the demeaned shock and `σ²x` its
full-sample variance. `q` is the regressor matrix of a separate impulse-response
regression (shock first, sample from `t0lp`), or `nothing` to read ``\\hat\\psi_h``
off the forecast-error regression ``[C, W]``.

- ``X^\\perp = C - W\\hat\\Gamma`` from **one** regression on rows ``t_0, \\dots, T``;
- per horizon, on rows ``t_0, \\dots, T-h``: ``\\hat b_h`` from the regression of
  `dhy` on ``[C, W]`` and ``\\hat f_t = \\hat u_t + X^{\\perp\\prime}_t \\hat b_{h,C}``;
- ``D_h = T_h - k_W`` with `dofadjust` (the reference's `dfIndicator = 0`), else ``T_h``;
- LP-A: ``\\hat\\sigma^2_{f,h} = \\sum \\hat f^2 / D_h`` and
  ``\\hat s_h = \\hat\\sigma^2_x \\sum_{i \\le h} \\hat\\psi_i^2 / \\hat\\sigma^2_{f,h}``;
- LP-B: ``\\hat v_t = \\hat f_t - \\sum_{i \\le h}\\hat\\psi_i x_{t+h-i}``,
  ``\\hat\\sigma^2_{v,h} = \\sum \\hat v^2 / D_h``,
  ``m_h = \\hat\\sigma^2_x\\sum_{i\\le h}\\hat\\psi_i^2 + \\hat\\sigma^2_{v,h}`` and
  ``\\hat s_h = \\hat\\sigma^2_x\\sum_{i\\le h}\\hat\\psi_i^2 / m_h``.

`irf_se` is the Newey–West standard error of ``\\hat\\psi_h`` with lag
`_gl_bandwidth` of the impulse-response sample and no
degrees-of-freedom factor. Returns a named tuple with the per-horizon
estimates and the matrices the asymptotic variance needs (`resid_lp`,
`beta_lp`, `fe`, `v`).
"""
function _gl_lp(dhy::Matrix{Float64}, C::Matrix{Float64}, W::Matrix{Float64},
        q::Union{Nothing, Matrix{Float64}}, x::Vector{Float64}, σ²x::Float64,
        t0::Int, t0lp::Int, H::Int; estimator::Symbol, dofadjust::Bool,
        irf_se::Bool = true)
    T = size(dhy, 1)
    nc = size(C, 2)
    kW = size(W, 2)
    R = hcat(C, W)
    shared = q === nothing
    Q = shared ? R : q
    Nq = size(Q, 2)
    lpb = estimator === :lpb

    # X⊥: one regression on rows t0..T, not t0..T-h (SubLPA.m:311-315)
    r0 = t0:T
    Cperp = fill(NaN, T, nc)
    Cperp[r0, :] = C[r0, :] - W[r0, :] * (W[r0, :] \ C[r0, :])

    psi = zeros(H + 1)
    se_psi = irf_se ? zeros(H + 1) : Float64[]
    mse = zeros(H + 1)
    sigma2v = lpb ? zeros(H + 1) : Float64[]
    fe = fill(NaN, T, H + 1)
    v = lpb ? fill(NaN, T, H + 1) : Matrix{Float64}(undef, 0, 0)
    resid_lp = fill(NaN, T, H + 1)
    beta_lp = zeros(Nq, H + 1)
    perfect = falses(H + 1)

    for h in 0:H
        rows = t0:(T - h)
        Th = length(rows)
        y = dhy[rows, h + 1]
        Rh = R[rows, :]
        b = Rh \ y
        u = y - Rh * b
        if shared
            rlp, ylp, Qh, blp, ulp = rows, y, Rh, b, u
        else
            rlp = t0lp:(T - h)
            ylp = dhy[rlp, h + 1]
            Qh = Q[rlp, :]
            blp = Qh \ ylp
            ulp = ylp - Qh * blp
        end
        psi[h + 1] = blp[1]
        beta_lp[:, h + 1] = blp
        resid_lp[rlp, h + 1] = ulp
        perfect[h + 1] = sum(abs2, ulp) <= 1e-20 * sum(abs2, ylp)

        f = u + Cperp[rows, :] * b[1:nc]
        fe[rows, h + 1] = f
        D = Th - (dofadjust ? kW : 0)
        if lpb
            vh = copy(f)
            @inbounds for i in 0:h, (j, t) in enumerate(rows)

                vh[j] -= psi[i + 1] * x[t + h - i]
            end
            v[rows, h + 1] = vh
            sigma2v[h + 1] = sum(abs2, vh) / D
            mse[h + 1] = σ²x * sum(abs2, view(psi, 1:(h + 1))) + sigma2v[h + 1]
        else
            mse[h + 1] = sum(abs2, f) / D
        end

        if irf_se
            nlp = length(rlp)
            SX = (Qh' * Qh) ./ nlp
            Ω = _gl_lrv_nw(Qh .* ulp, _gl_bandwidth(nlp))
            Vψ = ((SX \ Ω) / SX) ./ nlp
            se_psi[h + 1] = sqrt(max(Vψ[1, 1], 0.0))
        end
    end
    share = σ²x .* cumsum(psi .^ 2) ./ mse
    return (; share, psi, irf_se = se_psi, mse, sigma2v, fe, v, resid_lp, beta_lp,
        Q, perfect)
end

"""
    _gl_lp_gradient(estimator, ψ, σ²x, σ²h, m, s, Nq) -> Vector{Float64}

Gradient of the LP-A / LP-B share with respect to
``\\theta_h = (\\mathrm{vec}(b_0, \\dots, b_h), \\sigma^2_x, \\sigma^2_h)``
(`SimulationAVarLP*.m:319-323`), with ``\\psi_j`` the first of the `Nq`
coefficients of ``b_j``:

- LP-A: ``\\frac{1}{\\sigma^2_{f}}\\,[2\\sigma^2_x\\psi' \\otimes e_1',\\ \\sum\\psi^2,\\ -s]``;
- LP-B: ``\\frac{1-s}{m}\\,[2\\sigma^2_x\\psi' \\otimes e_1',\\ \\sum\\psi^2,\\ -\\frac{s}{1-s}]``,

where `m` is the LP-B denominator (pass `m = σ²h` for LP-A). Both are exact at
``s = \\hat s``; the reference's simulation method plugs in the bias-corrected
share instead. The LP-B last entry is computed as ``-s/m``, which is the same
number without dividing by ``1 - s``.
"""
function _gl_lp_gradient(estimator::Symbol, ψ::AbstractVector{Float64}, σ²x::Float64,
        σ²h::Float64, m::Float64, s::Float64, Nq::Int)
    np = length(ψ)
    g = zeros(Nq * np + 2)
    c = estimator === :lpa ? 1 / σ²h : (1 - s) / m
    for j in 1:np
        g[1 + (j - 1) * Nq] = c * 2 * ψ[j] * σ²x
    end
    g[end - 1] = c * sum(abs2, ψ)
    g[end] = estimator === :lpa ? -s / σ²h : -s / m
    return g
end

"""
    _gl_lp_avar(Q, resid_lp, err, x, σ²x, σ²h, beta_lp, Σq, t0, t0lp, h) -> NamedTuple

Horizon-``h`` asymptotic covariance of
``\\hat\\theta_h = (\\mathrm{vec}(\\hat b_0, \\dots, \\hat b_h), \\hat\\sigma^2_x, \\hat\\sigma^2_h)``
for LP-A / LP-B (`SimulationAVarLP*.m:212-262`, `jointIndicator = 0`). On rows
``\\max(t_0, t_0^{LP}), \\dots, T-h`` the moments are
``[q_t\\hat u_{t,0}, \\dots, q_t\\hat u_{t,h},\\ x_t^2 - \\hat\\sigma^2_x,\\ e_t^2 - \\hat\\sigma^2_h]``
with `err` ``= \\hat f`` (LP-A) or ``\\hat v`` (LP-B), demeaned column by column;
``G = -\\mathrm{blkdiag}(I_{h+1} \\otimes \\hat\\Sigma_q, I_2)`` with `Σq` the
second-moment matrix of ``q_t`` over ``t_0^{LP}, \\dots, T``.
"""
function _gl_lp_avar(Q::Matrix{Float64}, resid_lp::Matrix{Float64},
        err::AbstractVector{Float64}, x::Vector{Float64}, σ²x::Float64, σ²h::Float64,
        beta_lp::Matrix{Float64}, Σq::Matrix{Float64}, t0::Int, t0lp::Int, h::Int)
    T, Nq = size(Q)
    rows = max(t0, t0lp):(T - h)
    n = length(rows)
    m = Nq * (h + 1) + 2
    Z = Matrix{Float64}(undef, n, m)
    for j in 0:h
        Z[:, (j * Nq + 1):((j + 1) * Nq)] = Q[rows, :] .* view(resid_lp, rows, j + 1)
    end
    Z[:, m - 1] = view(x, rows) .^ 2 .- σ²x
    Z[:, m] = view(err, rows) .^ 2 .- σ²h
    Z .-= mean(Z; dims = 1)
    G = lu(-cat(kron(Matrix{Float64}(I, h + 1, h + 1), Σq), Matrix{Float64}(I, 2, 2);
        dims = (1, 2)))
    L = _gl_bandwidth(n)
    V = _gl_gogt(G, _gl_lrv_nw(Z, L), n)
    feasible = m < n - 1
    V_pw = feasible ? _gl_gogt(G, _gl_prewhitened_lrv(Z, L), n) : fill(NaN, m, m)
    θ = vcat(vec(beta_lp[:, 1:(h + 1)]), σ²x, σ²h)
    return (; theta = θ, avar_pw = V_pw, avar = V, bandwidth = L, nobs = n, feasible)
end
