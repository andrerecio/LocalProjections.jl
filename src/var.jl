# ============================================================================
# Reduced-form VAR and lag-order selection
# ============================================================================

"""
    _var_data(data, vars) -> Matrix{Float64}

Materialize the VAR data matrix `Y` (`T × n`) from `vars`, in the given order.

`missing` and `NaN` are rejected: the moving-block bootstrap simulates a
complete series and the VAR recursion has no notion of a gap. This is also the
safe side of issue LP-1 (`lp` drops missing rows *before* lags are built, so an
internal gap would silently collapse the time axis).
"""
function _var_data(data::AbstractDataFrame, vars::AbstractVector{Symbol})
    isempty(vars) && throw(ArgumentError("`vars` must name at least one variable"))
    length(unique(vars)) == length(vars) ||
        throw(ArgumentError("`vars` contains duplicate names"))
    T = nrow(data)
    Y = Matrix{Float64}(undef, T, length(vars))
    for (j, v) in enumerate(vars)
        hasproperty(data, v) ||
            throw(ArgumentError("variable $v named in `vars` is not a column of the data"))
        col = data[!, v]
        any(ismissing, col) &&
            throw(ArgumentError("column $v contains `missing`; the VAR bootstrap " *
                                "requires a gap-free sample"))
        @inbounds for t in 1:T
            Y[t, j] = Float64(col[t])
        end
        any(isnan, view(Y, :, j)) &&
            throw(ArgumentError("column $v contains `NaN`; the VAR bootstrap " *
                                "requires a gap-free sample"))
    end
    return Y
end

"""
    _var_ols(Y::AbstractMatrix{Float64}, p::Int)

Least-squares reduced-form VAR(`p`) with intercept. Port of
`_estim/var_estim.m` in `lp_var_nberma`.

Returns a named tuple `(; c, A, U, Σu, T, Tu)` where `c` is the `n`-vector of
intercepts, `A` is `n × n × p` with `A[:, :, l]` the coefficient matrix on
``Y_{t-l}``, `U` is the `Tu × n` residual matrix, and

```math
\\widehat\\Sigma_u = \\frac{U'U}{T_u - (np + 1)},
\\qquad T_u = T - p,
```

the denominator used by `var_estim.m`: effective observations minus the number
of regressors per equation, intercept included.
"""
function _var_ols(Y::AbstractMatrix{Float64}, p::Int)
    p >= 1 || throw(ArgumentError("the VAR lag length must be at least 1 (got $p)"))
    T, n = size(Y)
    Tu = T - p
    k = n * p + 1
    Tu > k || throw(ArgumentError(
        "a VAR($p) on $n variables needs more than $k usable observations, " *
        "but T - p = $Tu are available (T = $T)"))
    X = Matrix{Float64}(undef, Tu, k)
    @inbounds for l in 1:p, j in 1:n, t in 1:Tu
        X[t, (l - 1) * n + j] = Y[p + t - l, j]
    end
    @inbounds X[:, end] .= 1.0
    Yd = Y[(p + 1):T, :]
    B = X \ Yd                       # k × n
    U = Yd - X * B
    Σu = (U' * U) ./ (Tu - k)
    A = Array{Float64}(undef, n, n, p)
    @inbounds for l in 1:p
        A[:, :, l] = transpose(B[((l - 1) * n + 1):(l * n), :])
    end
    return (; c = Vector{Float64}(B[end, :]), A = A, U = Matrix{Float64}(U),
        Σu = Matrix{Float64}(Σu), T = T, Tu = Tu)
end

"""
    VARLagSelection

Result of [`lagselect`](@ref): the AIC and BIC paths over `p = 1:maxlags` for a
reduced-form VAR, and the selected lag order.

Fields: `vars`, `maxlags`, `criterion`, `lags`, `aic`, `bic`, `selected`,
`nobs`. Use [`nlags`](@ref) to get the selected order as an `Int`.
"""
struct VARLagSelection
    vars::Vector{Symbol}
    maxlags::Int
    criterion::Symbol
    lags::Vector{Int}
    aic::Vector{Float64}
    bic::Vector{Float64}
    selected::Int
    nobs::Int
end

"""
    nlags(s::VARLagSelection) -> Int

The lag order selected by [`lagselect`](@ref) under its `criterion`.
"""
nlags(s::VARLagSelection) = s.selected

"""
    lagselect(data, vars; maxlags=10, criterion=:aic) -> VARLagSelection

Select the lag order of a reduced-form VAR in `vars` by an information
criterion. Port of `_estim/ic_var.m` in `lp_var_nberma`, which is what the
Montiel Olea, Plagborg-Møller, Qian & Wolf (2025) simulations use to set the
lag length of both the local-projection controls and the bootstrap VAR.

With ``n`` variables, ``T`` observations and ``\\widehat\\Sigma_p`` the
residual covariance of a VAR(``p``),

```math
\\mathrm{AIC}(p) = \\log\\det\\widehat\\Sigma_p
                 + \\frac{2\\,(n^2p + n)}{T - p_{\\max}},
\\qquad
\\mathrm{BIC}(p) = \\log\\det\\widehat\\Sigma_p
                 + \\frac{(n^2p + n)\\log(T - p_{\\max})}{T - p_{\\max}}.
```

Two conventions are inherited deliberately from the reference and are *not*
bugs: the penalty denominator is the fixed ``T - p_{\\max}`` for every ``p``, so
the criteria are comparable across the grid, but ``\\widehat\\Sigma_p`` is
estimated by `_var_ols` on the full sample (``T - p`` rows, with the
``(T-p) - (np+1)`` degrees-of-freedom denominator) rather than on a common
subsample. The grid starts at ``p = 1``; ``p = 0`` is not considered.

Both criteria are always computed; `criterion` (`:aic` or `:bic`) only decides
which one `selected` minimizes. A single order is chosen for the whole system —
there is no per-horizon or per-equation selection.

# Example
```julia
sel = lagselect(df, [:shock, :y]; maxlags = 10, criterion = :aic)
nlags(sel)     # selected p, to be used in the LP formula and in `varbootstrap`
```

See also [`nlags`](@ref), [`varbootstrap`](@ref).
"""
function lagselect(data::AbstractDataFrame, vars::AbstractVector{Symbol};
        maxlags::Integer = 10, criterion::Symbol = :aic)
    criterion in (:aic, :bic) ||
        throw(ArgumentError("`criterion` must be :aic or :bic (got :$criterion)"))
    maxlags >= 1 || throw(ArgumentError("`maxlags` must be at least 1 (got $maxlags)"))
    Y = _var_data(data, vars)
    T, n = size(Y)
    pmax = Int(maxlags)
    denom = T - pmax
    denom > 0 || throw(ArgumentError(
        "`maxlags` = $pmax is not smaller than the sample size T = $T"))
    aics = Vector{Float64}(undef, pmax)
    bics = Vector{Float64}(undef, pmax)
    for p in 1:pmax
        v = _var_ols(Y, p)
        ld = logdet(Symmetric(v.Σu))
        k = n^2 * p + n
        aics[p] = ld + 2 * k / denom
        bics[p] = ld + k * log(denom) / denom
    end
    sel = argmin(criterion === :aic ? aics : bics)
    return VARLagSelection(collect(vars), pmax, criterion, collect(1:pmax),
        aics, bics, sel, T)
end

function lagselect(data, vars::AbstractVector{Symbol}; kwargs...)
    lagselect(DataFrame(data), vars; kwargs...)
end

function DataFrames.DataFrame(s::VARLagSelection)
    DataFrame(lags = s.lags, aic = s.aic, bic = s.bic)
end

function Base.show(io::IO, s::VARLagSelection)
    print(io, "VARLagSelection(criterion=:", s.criterion, ", selected=", s.selected, ")")
end

function Base.show(io::IO, ::MIME"text/plain", s::VARLagSelection)
    data = (; Lags = s.lags, AIC = s.aic, BIC = s.bic)
    labels = ["Lags", "AIC", "BIC"]
    chosen = s.criterion === :aic ? 2 : 3
    hl = TextHighlighter(
        (d, i, j) -> s.lags[i] == s.selected && j == chosen,
        crayon"bold"
    )
    title = "VAR lag-order selection (ic_var): " *
            join(string.(s.vars), ", ") *
            "  |  T = $(s.nobs), p = 1:$(s.maxlags)"
    ioc = IOContext(io, :color => true)
    pretty_table(ioc, data;
        column_labels = labels,
        title = title,
        highlighters = [hl],
        formatters = [fmt__round(4, [2, 3])],
        alignment = [:r, :r, :r],
        table_format = TextTableFormat(
            borders = text_table_borders__unicode_rounded))
    println(io, "Selected by :", s.criterion, ": p = ", s.selected)
    return nothing
end

# ============================================================================
# VAR primitives shared by varbootstrap and fevdbootstrap (Pope, IRF, simulation, FEVD)
# ============================================================================

"""
    _pope_biascorrect(A, Σu, T) -> (A_corrected, δ)

Pope (1990, eq. 9) analytical bias correction of the VAR slope matrices. Port
of `_estim/var_biascorr.m` in `lp_var_nberma`.

With companion matrix ``\\mathcal A``, ``G = \\mathrm{blkdiag}(\\Sigma_u, 0)``
and ``\\Gamma_0`` solving the discrete Lyapunov equation
``\\Gamma_0 = \\mathcal A\\Gamma_0\\mathcal A' + G``,

```math
M = (I-\\mathcal A')^{-1}
  + \\mathcal A'(I-\\mathcal A'\\mathcal A')^{-1}
  + \\sum_{\\lambda\\in\\mathrm{eig}(\\mathcal A)}\\lambda(I-\\lambda\\mathcal A')^{-1},
\\qquad b = G\\,M\\,\\Gamma_0^{-1},
\\qquad \\mathcal A^{c} = \\mathcal A + b/T.
```

Two details are easy to get wrong and are deliberate here:

  - the middle term is ``\\mathcal A'(I - \\mathcal A'\\mathcal A')^{-1}``, i.e.
    the square of the *transpose*, not ``\\mathcal A'\\mathcal A``;
  - `T` is the **full** row count of the VAR data, not the number `Tu = T - p`
    of fitted residuals.

Two stability safeguards, both from the reference: if the OLS companion matrix
already has an eigenvalue outside the unit circle the correction is skipped
entirely (returning `δ = 0`); otherwise `δ` is lowered from 1 in steps of 0.01
until ``\\mathcal A + \\delta b/T`` is stable. Requesting the correction
therefore does not guarantee a full correction — inspect the returned `δ`.

Validated against Pope's closed form for the AR(1) case, where `b` must equal
``1 + 3\\rho`` exactly.
"""
function _pope_biascorrect(A::AbstractMatrix{Float64}, Σu::AbstractMatrix{Float64},
        T::Real)
    n, np = size(A)
    Acomp = np == n ? Matrix{Float64}(A) :
            [Matrix{Float64}(A); Matrix{Float64}(I, np - n, np - n) zeros(np - n, n)]
    maximum(abs, eigvals(Acomp)) > 1 && return (Matrix{Float64}(A), 0.0)
    G = zeros(Float64, np, np)
    G[1:n, 1:n] = Σu
    # Discrete Lyapunov solve: vec(𝒜 Γ 𝒜') = (𝒜 ⊗ 𝒜) vec(Γ).
    Γ0 = reshape((I - kron(Acomp, Acomp)) \ vec(G), np, np)
    At = Matrix{Float64}(transpose(Acomp))
    M = inv(I - At) + At * inv(I - At * At)
    for λ in eigvals(Acomp)
        M = M + λ * inv(I - λ * At)
    end
    b = real.(G * (M / Γ0))
    Acorr = Acomp + b ./ T
    δ = 1.0
    while maximum(abs, eigvals(Acorr)) > 1 && δ > 0
        δ -= 0.01
        Acorr = Acomp + δ .* b ./ T
    end
    return (Matrix{Float64}(Acorr[1:n, :]), δ)
end

"""
    _pope_biascorrect(A::Array{Float64,3}, Σu, T)

Method taking the `n × n × p` slope array produced by `_var_ols` and
returning a corrected array of the same shape.
"""
function _pope_biascorrect(A::Array{Float64, 3}, Σu::AbstractMatrix{Float64}, T::Real)
    n, _, p = size(A)
    flat = Matrix{Float64}(undef, n, n * p)
    @inbounds for l in 1:p
        flat[:, ((l - 1) * n + 1):(l * n)] = A[:, :, l]
    end
    corrected, δ = _pope_biascorrect(flat, Σu, T)
    out = Array{Float64}(undef, n, n, p)
    @inbounds for l in 1:p
        out[:, :, l] = corrected[:, ((l - 1) * n + 1):(l * n)]
    end
    return (out, δ)
end

"""
    _var_impact(U, innov_index) -> Vector{Float64}

Unit-impact structural shock vector. Port of the Cholesky step in
`_estim/var_ir_estim.m`, which obtains it as the numerically equivalent
regression of all residual columns on columns `1:innov_index` without a
constant, keeping the coefficient on column `innov_index`.

For `innov_index == 1` this reduces to ``\\Sigma_u[:,1]/\\Sigma_u[1,1]``, i.e.
``L_{:,1}/L_{11}`` for the lower-triangular Cholesky factor `L` — the impact
vector normalized to a unit change in the shock. It is always computed from the
**original OLS** residuals, never from Pope-corrected quantities.
"""
function _var_impact(U::AbstractMatrix{Float64}, innov_index::Int)
    Z = U[:, 1:innov_index]
    B = Z \ U                       # innov_index × n
    return Vector{Float64}(B[innov_index, :])
end

"""
    _var_ma(A, H) -> Array{Float64,3}

Reduced-form moving-average coefficients of a VAR, `n × n × (H+1)`: the
recursion ``\\Theta_h = \\sum_{l=1}^{\\min(h,p)} A_l \\Theta_{h-l}`` with
``\\Theta_0 = I``, so that `Θ[:, :, h+1]` is ``[F^h]_{1:n,1:n}`` for the
companion matrix ``F`` (`var_ir.m`, `var_estimation.m`).
"""
function _var_ma(A::Array{Float64, 3}, H::Int)
    n, _, p = size(A)
    Θ = Array{Float64}(undef, n, n, H + 1)
    Θ[:, :, 1] = Matrix{Float64}(I, n, n)
    for h in 1:H
        M = zeros(Float64, n, n)
        for l in 1:min(h, p)
            M += A[:, :, l] * Θ[:, :, h - l + 1]
        end
        Θ[:, :, h + 1] = M
    end
    return Θ
end

"""
    _var_irf(A, ν, H) -> Matrix{Float64}

VAR-implied structural impulse responses, `n × (H+1)`. Port of
`_estim/var_ir.m`: ``\\Theta_h \\nu`` at each horizon, with ``\\Theta_h`` the
moving-average coefficients of `_var_ma`. Equivalent to ``e_r' C^h b`` on the
companion form but cheaper.
"""
function _var_irf(A::Array{Float64, 3}, ν::AbstractVector{Float64}, H::Int)
    Θ = _var_ma(A, H)
    out = Matrix{Float64}(undef, size(A, 1), H + 1)
    out[:, 1] = ν
    for h in 1:H
        out[:, h + 1] = Θ[:, :, h + 1] * ν
    end
    return out
end

"""
    _var_simulate(c, A, Ustar, Yinit) -> Matrix{Float64}

Iterate the fitted VAR forward from the drawn initial conditions. Port of
`_estim/var_sim.m`. `Yinit` supplies the first `p` rows verbatim and the
remaining `size(Ustar, 1)` rows follow
``Y_t^* = c + \\sum_{l=1}^{p} A_l Y_{t-l}^* + u_t^*``, so the returned matrix
has `p + size(Ustar, 1)` rows.
"""
function _var_simulate(c::AbstractVector{Float64}, A::Array{Float64, 3},
        Ustar::AbstractMatrix{Float64}, Yinit::AbstractMatrix{Float64})
    n, _, p = size(A)
    Tu = size(Ustar, 1)
    T = size(Yinit, 1) + Tu
    Y = Matrix{Float64}(undef, T, n)
    @inbounds Y[1:size(Yinit, 1), :] = Yinit
    @inbounds for t in (p + 1):T
        for j in 1:n
            Y[t, j] = c[j] + Ustar[t - p, j]
        end
        for l in 1:p, i in 1:n, j in 1:n
            Y[t, j] += A[j, i, l] * Y[t - l, i]
        end
    end
    return Y
end

"""
    _var_fevd(A, Ω, H; shock, response, cumulate = false) -> Vector{Float64}

Share of the forecast-error variance of variable `response` due to the Cholesky
shock of variable `shock` in a VAR with slopes `A` (`n × n × p`) and residual
covariance `Ω`, horizons ``0, \\dots, H`` (`var_chol.m:54-81`). With
``\\Psi_h = \\Theta_h L`` the one-standard-deviation responses
(``\\Theta_h`` from `_var_ma`, ``L`` the lower Cholesky factor of ``\\Omega``),

```math
s_h = \\frac{\\sum_{l \\le h} \\Psi_l[r, j]^2}{\\sum_{l \\le h} \\sum_m \\Psi_l[r, m]^2}.
```

With `cumulate = true` the response row is first cumulated over horizons, for
all shocks (`var_chol.m:67`): the share then refers to ``y_{t+h} - y_{t-1}``
when the VAR holds the first difference of ``y``. The share is invariant to a
scalar rescaling of ``\\Omega`` but depends on the Cholesky order.
"""
function _var_fevd(A::Array{Float64, 3}, Ω::AbstractMatrix{Float64}, H::Int;
        shock::Int, response::Int, cumulate::Bool = false)
    C = cholesky(Symmetric(Matrix{Float64}(Ω)); check = false)
    issuccess(C) || throw(ArgumentError(
        "the VAR residual covariance is not positive definite: the VAR variables " *
        "are linearly dependent (e.g. a level together with its first difference)"))
    L = C.L
    Θ = _var_ma(A, H)
    n = size(A, 1)
    ψ = zeros(n)
    num = 0.0
    den = 0.0
    s = Vector{Float64}(undef, H + 1)
    for h in 0:H
        row = vec(transpose(view(Θ, response, :, h + 1)) * L)
        ψ = cumulate ? ψ .+ row : row
        num += ψ[shock]^2
        den += sum(abs2, ψ)
        s[h + 1] = num / den
    end
    return s
end

# A VAR residual covariance that is singular up to rounding (linearly dependent
# variables) gives meaningless shares even when its Cholesky factorization
# happens to succeed: test the smallest eigenvalue of the correlation matrix.
function _check_var_rank(Ω::AbstractMatrix{Float64}, varlist = nothing)
    d = [Ω[i, i] for i in axes(Ω, 1)]
    singular = !all(>(0), d) ||
               eigmin(Symmetric(Ω ./ sqrt.(d .* transpose(d)))) <= sqrt(eps()) * size(Ω, 1)
    singular && throw(ArgumentError(
        "the VAR residual covariance is singular: the VAR variables" *
        (varlist === nothing ? "" : " $(varlist)") * " are linearly dependent (e.g. " *
        "a level together with its first difference)"))
    return nothing
end
