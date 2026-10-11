# Literal-port parity tests for the Gorodnichenko–Lee (2020) FEVD estimators.
#
# `GLMatlabPort` transcribes the MATLAB reference (docs/src/code/goro_lee_r2/lib:
# SubR2.m, SubLPA.m, SubLPB.m, SimulationAVar*.m, BootstrapBlockR2.m,
# BootstrapVAR*.m, var_estimation.m, var_chol.m, var_bootstrap.m, lrv_nw.m) line
# by line, from the .m files rather than from the package: `lagmatrix` NaN
# padding, normal-equation OLS, MATLAB `var`/`std`/`quantile`/`round`. The
# package's private matrix core and its public `fevd`/`fevdbootstrap`/`summarize`
# are compared with it at rtol = 1e-10 (the package solves by QR, so agreement
# is at rounding level). Bootstrap and simulation draws are replayed from the
# package's per-draw seeds, so they are compared draw by draw.

# JuliaFormatter (sciml) is switched off for this file: with both `tags = ...` and
# `setup = ...` on a @testitem it formats the whole item body as macro keyword
# arguments (`a = 1` becomes `a=1`). The file is formatted by hand.
#! format: off

@testmodule GLMatlabPort begin
    # MATLAB transcriptions (semIndicator = 0, jointIndicator = 0 throughout);
    # function names follow the .m files.

    using LinearAlgebra, Statistics

    # lagmatrix(v, lags): column j is v(t - lags(j)), NaN where undefined
    function lagmatrix(v::AbstractVector, lags)
        T = length(v)
        out = fill(NaN, T, length(lags))
        for (j, k) in enumerate(lags), t in 1:T
            s = t - k
            1 <= s <= T && (out[t, j] = v[s])
        end
        return out
    end

    # MATLAB round: half away from zero (x - trunc(x) is exact in floating point)
    function mround(x::Float64)
        r = trunc(x)
        return abs(x - r) >= 0.5 ? r + sign(x) : r
    end

    # MATLAB quantile(x, p) of a vector: sorted data at plotting positions
    # (i - 0.5)/n, linear interpolation, the extremes outside [0.5/n, 1 - 0.5/n]
    function mquantile(x::AbstractVector, p::Real)
        xs = sort(collect(Float64, x))
        n = length(xs)
        q = [0.0; ((1:n) .- 0.5) ./ n; 1.0]
        xx = [xs[1]; xs; xs[n]]
        k = searchsortedlast(q, p)
        k >= length(q) && return xx[end]
        return xx[k] + (p - q[k]) / (q[k + 1] - q[k]) * (xx[k + 1] - xx[k])
    end

    # lrv_nw.m
    function lrv_nw(data::AbstractMatrix, lag)
        T, N = size(data)
        V = data' * data / T
        for i in 1:Int(lag)
            data_temp = [zeros(i, N); data[1:(end - i), :]]
            G_temp = data' * data_temp / T
            V = V + (1 - i / (lag + 1)) * (G_temp + G_temp')
        end
        return V
    end

    # blkdiag
    function blkdiag(A::AbstractMatrix, B::AbstractMatrix)
        M = zeros(size(A, 1) + size(B, 1), size(A, 2) + size(B, 2))
        M[1:size(A, 1), 1:size(A, 2)] = A
        M[(size(A, 1) + 1):end, (size(A, 2) + 1):end] = B
        return M
    end

    rhs_trend(T, trend) = hcat(ones(T), [(1:T) .^ k for k in 1:trend]...)

    # The prewhitened / plain long-run variance blocks shared by SubR2.m:213-229 and
    # SimulationAVarLP*.m:249-261
    function pw_omega(ZFull, Tlrv)
        Z_1 = ZFull[2:end, :]
        Z_0 = ZFull[1:(end - 1), :]
        A = (Z_0' * Z_0) \ (Z_0' * Z_1)
        A = Matrix(A')
        U = Z_1 - Z_0 * A'
        m = size(ZFull, 2)
        L = Int(mround(0.75 * Tlrv^(1 / 3) - 1))
        OmegaWPW = ((I(m) - A) \ lrv_nw(U, L)) / (I(m) - A')
        OmegaWoPW = lrv_nw(ZFull, L)
        return OmegaWPW, OmegaWoPW, L
    end

    # SubR2.m (extraOutputIndicator = 1)
    function SubR2(y, X, lagy, lagX, horizon, ylevel, trend)
        T, N = size(X)
        vd = zeros(horizon + 1)
        forecastError = fill(NaN, T, horizon + 1)
        seAVarWPW = zeros(horizon + 1)
        seAVarWoPW = zeros(horizon + 1)
        GOGTWPW = Vector{Matrix{Float64}}(undef, horizon + 1)
        GOGTWoPW = Vector{Matrix{Float64}}(undef, horizon + 1)
        effectiveT = zeros(Int, horizon + 1)
        bw = zeros(Int, horizon + 1)
        idxt = zeros(Int, horizon + 1, 2)
        theta = Vector{Vector{Float64}}(undef, horizon + 1)

        X = copy(X)
        X[:, 1] = X[:, 1] .- mean(X[:, 1])
        t0 = maximum([lagy + (1 - ylevel); lagX]) + 1
        yendo = ylevel == 0 ? [NaN; diff(y)] : y
        dhy = lagmatrix(y, 0:-1:(-horizon)) - repeat(lagmatrix(y, [1]), 1, horizon + 1)
        rhsy = lagmatrix(yendo, 1:lagy)
        rhsX = zeros(T, sum(lagX))
        columnFilled = 0
        for idxn in 1:N
            columnFilledNew = columnFilled + lagX[idxn]
            rhsX[:, (columnFilled + 1):columnFilledNew] = lagmatrix(X[:, idxn], 1:lagX[idxn])
            columnFilled = columnFilledNew
        end
        rhsAll = [rhsX rhsy rhs_trend(T, trend)]
        xFuture = lagmatrix(X[:, 1], 0:-1:(-horizon))

        for h in 0:horizon
            lhs = dhy[t0:(T - h), h + 1]
            rhs = rhsAll[t0:(T - h), :]
            fe = lhs - rhs * ((rhs' * rhs) \ (rhs' * lhs))
            forecastError[t0:(T - h), h + 1] = fe
            idxt[h + 1, :] = [t0, T - h]
            fexh = xFuture[t0:(T - h), 1:(h + 1)]
            TTemp = T - h - t0 + 1
            Sigmax = fexh' * fexh / TTemp
            theta2 = fexh' * fe / TTemp
            theta1 = Sigmax \ theta2
            theta3 = fe' * fe / TTemp
            theta[h + 1] = [theta1; theta2; theta3]
            vd[h + 1] = theta2' * theta1 / theta3

            effectiveT[h + 1] = TTemp
            residual = fe - fexh * theta1
            G = -blkdiag(Sigmax, Matrix(1.0I, h + 2, h + 2))
            xi = 1 / theta3 * [theta2; theta1; -vd[h + 1]]
            rx = repeat(residual, 1, h + 1) .* fexh
            B1 = rx - repeat(mean(rx; dims = 1), TTemp, 1)
            B2 = repeat(fe, 1, h + 1) .* fexh - repeat(theta2', TTemp, 1)
            B3 = fe .^ 2 .- theta3
            ZFull = [B1 B2 B3]
            OmegaWPW, OmegaWoPW, bw[h + 1] = pw_omega(ZFull, TTemp)
            seAVarWPW[h + 1] = sqrt(xi' * (G \ OmegaWPW / G') * xi / TTemp)
            GOGTWPW[h + 1] = (G \ OmegaWPW / G') / TTemp
            seAVarWoPW[h + 1] = sqrt(xi' * (G \ OmegaWoPW / G') * xi / TTemp)
            GOGTWoPW[h + 1] = (G \ OmegaWoPW / G') / TTemp
        end
        return (; vd, forecastError, seAVarWPW, seAVarWoPW, theta, GOGTWPW, GOGTWoPW,
            effectiveT, idxt, t0, bw, dhy, rhsAll, xFuture)
    end

    # SubR2.m with extraOutputIndicator = 0: vd only
    function SubR2_vd(y, X, lagy, lagX, horizon, ylevel, trend)
        T, N = size(X)
        vd = zeros(horizon + 1)
        X = copy(X)
        X[:, 1] = X[:, 1] .- mean(X[:, 1])
        t0 = maximum([lagy + (1 - ylevel); lagX]) + 1
        yendo = ylevel == 0 ? [NaN; diff(y)] : y
        dhy = lagmatrix(y, 0:-1:(-horizon)) - repeat(lagmatrix(y, [1]), 1, horizon + 1)
        rhsy = lagmatrix(yendo, 1:lagy)
        rhsX = zeros(T, sum(lagX))
        columnFilled = 0
        for idxn in 1:N
            columnFilledNew = columnFilled + lagX[idxn]
            rhsX[:, (columnFilled + 1):columnFilledNew] = lagmatrix(X[:, idxn], 1:lagX[idxn])
            columnFilled = columnFilledNew
        end
        rhsAll = [rhsX rhsy rhs_trend(T, trend)]
        xFuture = lagmatrix(X[:, 1], 0:-1:(-horizon))
        for h in 0:horizon
            lhs = dhy[t0:(T - h), h + 1]
            rhs = rhsAll[t0:(T - h), :]
            fe = lhs - rhs * ((rhs' * rhs) \ (rhs' * lhs))
            fexh = xFuture[t0:(T - h), 1:(h + 1)]
            TTemp = T - h - t0 + 1
            Sigmax = fexh' * fexh / TTemp
            theta2 = fexh' * fe / TTemp
            theta1 = Sigmax \ theta2
            theta3 = fe' * fe / TTemp
            vd[h + 1] = theta2' * theta1 / theta3
        end
        return vd
    end

    # SubLPA.m / SubLPB.m common set-up (#1-#2): the FE and LP right-hand sides and Xperp
    function sublp_setup(y, X, lagy, lagX, XtIndicator, horizon, ylevel, trend, XLP,
            lagyLP, lagXLP, XtIndicatorLP)
        T, N = size(X)
        TLP, NLP = size(XLP)
        @assert XtIndicator[1] == 1 && T == TLP
        meanx = mean(X[:, 1])
        varx = var(X[:, 1])
        t0 = maximum([lagy + (1 - ylevel); lagX; 1]) + 1
        t0LP = maximum([lagyLP + (1 - ylevel); lagXLP; 1]) + 1
        yendo = ylevel == 0 ? [NaN; diff(y)] : y
        X = copy(X)
        XLP = copy(XLP)
        X[:, 1] = X[:, 1] .- meanx
        XLP[:, 1] = XLP[:, 1] .- meanx
        dhy = lagmatrix(y, 0:-1:(-horizon)) - repeat(lagmatrix(y, [1]), 1, horizon + 1)
        rhsyFE = lagmatrix(yendo, 1:lagy)
        rhsyLP = lagmatrix(yendo, 1:lagyLP)
        function rhsX(Xm, lags, ind)
            out = zeros(T, sum(lags) + sum(ind))
            columnFilled = sum(ind)
            out[:, 1:columnFilled] = Xm[:, Bool.(ind)]
            for idxn in 1:size(Xm, 2)
                columnFilledNew = columnFilled + lags[idxn]
                out[:, (columnFilled + 1):columnFilledNew] = lagmatrix(Xm[:, idxn], 1:lags[idxn])
                columnFilled = columnFilledNew
            end
            return out
        end
        rhsTrend = rhs_trend(T, trend)
        rhsAllFE = [rhsX(X, lagX, XtIndicator) rhsyFE rhsTrend]
        rhsAllLP = [rhsX(XLP, lagXLP, XtIndicatorLP) rhsyLP rhsTrend]
        nc = sum(XtIndicator)
        Wt = rhsAllFE[t0:end, (nc + 1):end]
        Xperp = fill(NaN, size(X))
        Xperp[t0:end, :] = X[t0:end, :] - Wt * ((Wt' * Wt) \ (Wt' * X[t0:end, :]))
        shared = isequal(XLP, X) && isequal(lagyLP, lagy) && isequal(lagXLP, lagX) &&
                 isequal(XtIndicatorLP, XtIndicator)
        return (; T, meanx, varx, t0, t0LP, X, XLP, dhy, rhsAllFE, rhsAllLP, Xperp, shared)
    end

    function lp_irse(lhsLP, rhsLP, betah, TTempLP)
        feLP = lhsLP - rhsLP * betah
        SigmaX = rhsLP' * rhsLP / TTempLP
        Omega = lrv_nw(rhsLP .* repeat(feLP, 1, size(rhsLP, 2)), mround(0.75 * TTempLP^(1 / 3) - 1))
        V = (SigmaX \ Omega / SigmaX) / TTempLP
        return feLP, sqrt(V[1, 1])
    end

    # SubLPA.m (semIndicator = 0, extraOutputIndicator = 1)
    function SubLPA(y, X, lagy, lagX, XtIndicator, horizon, ylevel, trend, dfIndicator,
            XLP = X, lagyLP = lagy, lagXLP = lagX, XtIndicatorLP = XtIndicator)
        s = sublp_setup(y, X, lagy, lagX, XtIndicator, horizon, ylevel, trend, XLP, lagyLP,
            lagXLP, XtIndicatorLP)
        T, t0, t0LP = s.T, s.t0, s.t0LP
        ir = zeros(horizon + 1)
        irSeAVar = zeros(horizon + 1)
        mse = zeros(horizon + 1)
        idxt = zeros(Int, horizon + 1, 2)
        forecastError = fill(NaN, T, horizon + 1)
        BetaHat = zeros(size(s.rhsAllLP, 2), horizon + 1)
        residualLP = fill(NaN, T, horizon + 1)
        nc = sum(XtIndicator)
        for h in 0:horizon
            lhsFE = s.dhy[t0:(T - h), h + 1]
            rhsFE = s.rhsAllFE[t0:(T - h), :]
            betah = (rhsFE' * rhsFE) \ (rhsFE' * lhsFE)
            feFE = lhsFE - rhsFE * betah + s.Xperp[t0:(T - h), Bool.(XtIndicator)] * betah[1:nc]
            forecastError[t0:(T - h), h + 1] = feFE
            idxt[h + 1, :] = [t0, T - h]
            TTempFE = T - h - t0 + 1
            mse[h + 1] = sum(feFE .^ 2) /
                         (TTempFE - (1 - dfIndicator) * (lagy + sum(lagX) + (trend + 1)))
            if s.shared
                BetaHat[:, h + 1] = betah
                ir[h + 1] = betah[1]
                feLP, irSeAVar[h + 1] = lp_irse(lhsFE, rhsFE, betah, TTempFE)
                residualLP[t0LP:(T - h), h + 1] = feLP
            else
                TTempLP = T - h - t0LP + 1
                lhsLP = s.dhy[t0LP:(T - h), h + 1]
                rhsLP = s.rhsAllLP[t0LP:(T - h), :]
                betah = (rhsLP' * rhsLP) \ (rhsLP' * lhsLP)
                BetaHat[:, h + 1] = betah
                ir[h + 1] = betah[1]
                feLP, irSeAVar[h + 1] = lp_irse(lhsLP, rhsLP, betah, TTempLP)
                residualLP[t0LP:(T - h), h + 1] = feLP
            end
        end
        vd = s.varx * cumsum(ir .^ 2) ./ mse
        return (; ir, vd, forecastError, irSeAVar, meanx = s.meanx, varx = s.varx, mse, idxt,
            dhy = s.dhy, rhsAllFE = s.rhsAllFE, rhsAllLP = s.rhsAllLP, residualLP, BetaHat,
            t0, t0LP, Xperp = s.Xperp)
    end

    # SubLPB.m (semIndicator = 0, extraOutputIndicator = 1)
    function SubLPB(y, X, lagy, lagX, XtIndicator, horizon, ylevel, trend, dfIndicator,
            XLP = X, lagyLP = lagy, lagXLP = lagX, XtIndicatorLP = XtIndicator)
        s = sublp_setup(y, X, lagy, lagX, XtIndicator, horizon, ylevel, trend, XLP, lagyLP,
            lagXLP, XtIndicatorLP)
        T, t0, t0LP = s.T, s.t0, s.t0LP
        ir = zeros(horizon + 1)
        irSeAVar = zeros(horizon + 1)
        mse = zeros(horizon + 1)
        idxt = zeros(Int, horizon + 1, 2)
        forecastError = fill(NaN, T, horizon + 1)
        vFE = fill(NaN, T, horizon + 1)
        sigma2vHat = zeros(horizon + 1)
        BetaHat = zeros(size(s.rhsAllLP, 2), horizon + 1)
        residualLP = fill(NaN, T, horizon + 1)
        xFuture = lagmatrix(s.X[:, 1], 0:-1:(-horizon))
        nc = sum(XtIndicator)
        for h in 0:horizon
            TTempLP = T - h - t0LP + 1
            lhsLP = s.dhy[t0LP:(T - h), h + 1]
            rhsLP = s.rhsAllLP[t0LP:(T - h), :]
            betah = (rhsLP' * rhsLP) \ (rhsLP' * lhsLP)
            BetaHat[:, h + 1] = betah
            ir[h + 1] = betah[1]
            feLP, irSeAVar[h + 1] = lp_irse(lhsLP, rhsLP, betah, TTempLP)
            residualLP[t0LP:(T - h), h + 1] = feLP
            if s.shared
                lhsFE = lhsLP
                rhsFE = rhsLP
            else
                lhsFE = s.dhy[t0:(T - h), h + 1]
                rhsFE = s.rhsAllFE[t0:(T - h), :]
                betah = (rhsFE' * rhsFE) \ (rhsFE' * lhsFE)
            end
            feFE = lhsFE - rhsFE * betah + s.Xperp[t0:(T - h), Bool.(XtIndicator)] * betah[1:nc]
            forecastError[t0:(T - h), h + 1] = feFE
            idxt[h + 1, :] = [t0, T - h]
            TTempFE = T - h - t0 + 1
            xFutureh = xFuture[t0:(T - h), (h + 1):-1:1]
            vh = feFE - xFutureh * ir[1:(h + 1)]
            vFE[t0:(T - h), h + 1] = vh
            sigma2vHat[h + 1] = sum(vh .^ 2) /
                                (TTempFE - (1 - dfIndicator) * (lagy + sum(lagX) + (trend + 1)))
            mse[h + 1] = sum(ir[1:(h + 1)] .^ 2) * s.varx + sigma2vHat[h + 1]
        end
        vd = s.varx * cumsum(ir .^ 2) ./ mse
        return (; ir, vd, forecastError, irSeAVar, meanx = s.meanx, varx = s.varx, mse, idxt,
            dhy = s.dhy, rhsAllFE = s.rhsAllFE, rhsAllLP = s.rhsAllLP, residualLP, vFE,
            BetaHat, sigma2vHat, t0, t0LP, Xperp = s.Xperp)
    end

    # SimulationAVarLPA.m / SimulationAVarLPB.m, jointIndicator = 0, #1-#3 and #5:
    # per horizon thetah, GOGTWPW, GOGTWoPW and the delta-method gradient Deltah(s)
    function SimulationAVarLP(which::Symbol, y, X, horizon, ylevel, trend, lagy, lagX,
            XtIndicator, dfIndicator, XLP = X, lagyLP = lagy, lagXLP = lagX,
            XtIndicatorLP = XtIndicator)
        T, N = size(X)
        X = copy(X)
        XLP = copy(XLP)
        x = X[:, 1] .- mean(X[:, 1])
        X[:, 1] = x
        XLP[:, 1] = XLP[:, 1] .- mean(XLP[:, 1])
        sub = which === :A ?
              SubLPA(y, X, lagy, lagX, XtIndicator, horizon, ylevel, trend, dfIndicator, XLP,
            lagyLP, lagXLP, XtIndicatorLP) :
              SubLPB(y, X, lagy, lagX, XtIndicator, horizon, ylevel, trend, dfIndicator, XLP,
            lagyLP, lagXLP, XtIndicatorLP)
        q = sub.rhsAllLP
        t0LP = sub.t0LP
        varx = sub.varx
        forecastErrorVar = sub.mse
        Sigmaq = q[t0LP:end, :]' * q[t0LP:end, :] / (T - t0LP + 1)
        Nq = size(q, 2)
        thetas = Vector{Vector{Float64}}(undef, horizon + 1)
        GOGTWPW = Vector{Matrix{Float64}}(undef, horizon + 1)
        GOGTWoPW = Vector{Matrix{Float64}}(undef, horizon + 1)
        for h in 0:horizon
            thetah = sub.BetaHat[:, 1:(h + 1)]
            last = which === :A ? forecastErrorVar[h + 1] : sub.sigma2vHat[h + 1]
            thetas[h + 1] = [vec(thetah); varx; last]
            Gh = -blkdiag(kron(Matrix(1.0I, h + 1, h + 1), Sigmaq), Matrix(1.0I, 2, 2))
            tminh = maximum([sub.idxt[1:(h + 1), 1]; t0LP])
            tmaxh = minimum(sub.idxt[1:(h + 1), 2])
            Th = tmaxh - tminh + 1
            Zhtemp = zeros(Th, 0)
            for htemp in 0:h
                Zhtemp = [Zhtemp q[tminh:tmaxh, :] .*
                                 repeat(sub.residualLP[tminh:tmaxh, htemp + 1], 1, Nq)]
            end
            err = which === :A ? sub.forecastError : sub.vFE
            errvar = which === :A ? forecastErrorVar[h + 1] : sub.sigma2vHat[h + 1]
            Zh = [Zhtemp x[tminh:tmaxh] .^ 2 .- varx err[tminh:tmaxh, h + 1] .^ 2 .- errvar]
            ZFull = Zh - repeat(mean(Zh; dims = 1), Th, 1)
            OmegaWPW, OmegaWoPW, _ = pw_omega(ZFull, Th)
            GOGTWPW[h + 1] = (Gh \ OmegaWPW / Gh') / Th
            GOGTWoPW[h + 1] = (Gh \ OmegaWoPW / Gh') / Th
        end
        # Deltah at a share value s (vdBc in the reference)
        function Delta(h, s)
            iota = zeros(Nq)
            iota[1] = 1
            ir = sub.ir[1:(h + 1)]
            if which === :A
                return 1 / forecastErrorVar[h + 1] *
                       [vec(kron(2 * ir' * varx, iota')); sum(ir .^ 2); -s]
            else
                return (1 - s) / forecastErrorVar[h + 1] *
                       [vec(kron(2 * ir' * varx, iota')); sum(ir .^ 2); -s / (1 - s)]
            end
        end
        return (; sub, thetas, GOGTWPW, GOGTWoPW, Delta, Sigmaq, Nq, x)
    end

    # var_estimation.m (trend = 0 or more): normal equations on kron(I_n, X)
    function var_estimation(time_series, lag, impulse_response_order, trend)
        T, n = size(time_series)
        T = T - lag
        k = n^2 * lag
        lag_matrix = hcat([lagmatrix(time_series[:, j], [l]) for l in 1:lag for j in 1:n]...)
        Y = vec(time_series[(lag + 1):end, :])
        X = [ones(T) lag_matrix[(lag + 1):end, :] hcat(zeros(T, 0), [(1:T) .^ j for j in 1:trend]...)]
        XX = kron(Matrix(1.0I, n, n), X)
        phi_hat = (XX' * XX) \ (XX' * Y)
        residual = Y - XX * phi_hat
        phi_hat = Matrix(reshape(phi_hat, n * lag + 1 + trend, n)')
        residual = Matrix(reshape(residual, T, n)')
        c = [phi_hat[:, 1] phi_hat[:, (end + 1 - trend):end]]
        phi_hat = phi_hat[:, 2:(end - trend)]
        phi = zeros(n, n, lag)
        for idx in 1:lag
            phi[:, :, idx] = phi_hat[:, (n * (idx - 1) + 1):(n * idx)]
        end
        omega = zeros(n, n)
        for idx in 1:T
            omega = omega + residual[:, idx] * residual[:, idx]'
        end
        omega = omega / T
        companion = [phi_hat; Matrix(1.0I, n * (lag - 1), n * (lag - 1)) zeros(n * (lag - 1), n)]
        psi = zeros(n, n, impulse_response_order + 1)
        for idx in 1:(impulse_response_order + 1)
            temporary = companion^(idx - 1)
            psi[:, :, idx] = temporary[1:n, 1:n]
        end
        return (; c, phi, omega, psi, k, T, residual)
    end

    # var_chol.m:54-81 (MCIndicator = 0): the variance decomposition
    function var_chol_vd(raw_data, lag, impulse_response_order, var_place,
            first_difference_index, trend)
        est = var_estimation(raw_data, lag, impulse_response_order, trend)
        n = size(est.c, 1)
        R = Matrix(cholesky(Symmetric(est.omega, :L)).L)    # chol(omega, 'lower')
        psi_R = zeros(n, n, impulse_response_order + 1)
        for idx in 1:(impulse_response_order + 1)
            psi_R[:, :, idx] = est.psi[:, :, idx] * R
        end
        for i in first_difference_index
            psi_R[i, :, :] = cumsum(psi_R[i, :, :]; dims = 2)
        end
        psi = psi_R[:, var_place, :]
        vd = zeros(n, impulse_response_order + 1)
        mse = est.omega
        variance = R[:, var_place] * R[:, var_place]'
        for idx in 1:impulse_response_order
            vd[:, idx] = diag(variance) .* (diag(mse) .^ (-1))
            mse = mse + psi_R[:, :, idx + 1] * psi_R[:, :, idx + 1]'
            variance = variance + psi[:, idx + 1] * psi[:, idx + 1]'
        end
        vd[:, impulse_response_order + 1] = diag(variance) .* (diag(mse) .^ (-1))
        return (; vd, est)
    end

    # BootstrapVARR2.m #2: information criterion on the common sample
    function var_ic(XVAR, lagVARSet, ICIndicator, trend)
        IC = zeros(length(lagVARSet))
        lagVARMax = maximum(lagVARSet)
        T = size(XVAR, 1) - lagVARMax          # = Ty - lagVARMax - (1 - ylevel)
        for (idxIC, lagVARtemp) in enumerate(lagVARSet)
            e = var_estimation(XVAR[(lagVARMax - lagVARtemp + 1):end, :], lagVARtemp, 0, trend)
            omega, k = e.omega, e.k
            IC[idxIC] = ICIndicator == 0 ? log(det(omega)) + k * log(T) / T :
                        ICIndicator == 1 ? log(det(omega)) + 2 * k * log(log(T)) / T :
                        log(det(omega)) + 2 * k / T
        end
        return lagVARSet[argmin(IC)], IC
    end

    # var_bootstrap.m with the random draws supplied: `idx` (Tend residual columns,
    # idx[1] unused) and `initialb`
    function var_bootstrap(c, phi, timeseries, residual, T, TBurnIn, idx, initialb)
        N, trend = size(c)
        trend = trend - 1
        lagVAR = size(phi, 3)
        Tinit = TBurnIn + 1
        Tend = TBurnIn + T
        zb = zeros(Tend, N * lagVAR)
        F = [reshape(phi, N, N * lagVAR); Matrix(1.0I, N * (lagVAR - 1), N * (lagVAR - 1)) zeros(N * (lagVAR - 1), N)]
        @assert length(idx) == Tend
        residualb = residual[:, idx]
        z = vec(timeseries[(initialb + lagVAR - 1):-1:initialb, :]')
        zb[1, :] = z'
        for idxtb in 2:Tend
            z = [c * (idxtb .^ (0:trend)); zeros(N * (lagVAR - 1))] + F * z +
                [residualb[:, idxtb]; zeros(N * (lagVAR - 1))]
            zb[idxtb, :] = z'
        end
        return zb[Tinit:Tend, 1:N]
    end

    # ------------------------------------------------------------------
    # Test data and helpers (not part of the MATLAB transcription)
    # ------------------------------------------------------------------
    using DataFrames, Random, StableRNGs

    # T = 200 draws of N = 3 regressors and two responses: `dy` (with level
    # y = cumsum(dy), the reference's ylevel = 0) and a stationary level `yl`
    # (ylevel = 1). The shock x1 has a nonzero mean so that demeaning matters.
    function gl_data(seed::Integer = 20261010; T::Int = 200)
        rng = StableRNG(seed)
        e = randn(rng, T, 5)
        x1 = 0.3 .+ e[:, 1]
        x2 = zeros(T)
        x3 = zeros(T)
        dy = zeros(T)
        yl = zeros(T)
        dy[1] = e[1, 4]
        for t in 2:T
            x2[t] = 0.5 * x2[t - 1] + 0.3 * x1[t] + e[t, 2]
            x3[t] = 0.6 * x3[t - 1] + 0.2 * x1[t - 1] + e[t, 3]
            dy[t] = 0.3 * dy[t - 1] + 0.8 * x1[t] + 0.4 * x1[t - 1] + 0.3 * x2[t] +
                    0.2 * x3[t - 1] + e[t, 4]
            yl[t] = 0.7 * yl[t - 1] + 0.6 * x1[t] + 0.3 * x1[t - 1] + 0.2 * x2[t] + e[t, 5]
        end
        return DataFrame(x1 = x1, x2 = x2, x3 = x3, dy = dy, y = cumsum(dy), yl = yl)
    end

    # The package builds the lags of the shock from the raw column and orders
    # W as [1, lags...]; the reference lags the demeaned shock and orders
    # [lags..., y lags, 1]. Both span the same space: Q_pkg = Q_ref * M on the
    # regression rows, so b_pkg = M⁻¹ b_ref and the LP-A/B moment covariance
    # transforms with J = blkdiag(I_{h+1} ⊗ M⁻¹, I_2).
    function lp_reparam(Qref::AbstractMatrix, Qpkg::AbstractMatrix, rows, h::Int)
        M = Qref[rows, :] \ Qpkg[rows, :]
        J = blkdiag(kron(Matrix(1.0I, h + 1, h + 1), inv(M)), Matrix(1.0I, 2, 2))
        return J, M
    end

    # Fixed residual indices and initial block for var_bootstrap.m, as the
    # package's `fevdbootstrap(...; method = :var)` draws them for draw `b`
    # (`Xoshiro(seed_b)`: burnin + T - 1 residual rows from 1:Tu, then the
    # initial block from 1:(Tu + 1)). The reference's idx(1) is drawn but never
    # used; it is set to 1 here.
    function var_draw_indices(seed::UInt64, Tu::Int, T::Int, burnin::Int)
        drng = Random.Xoshiro(seed)
        idx = rand(drng, 1:Tu, burnin + T - 1)
        init = rand(drng, 1:(Tu + 1))
        return [1; idx], init
    end


    # BootstrapBlockR2.m:196-223, one draw at horizon h with the block starts
    # `blockShufflehb` supplied (the reference draws them as
    # ceil((Th - sizeBlock + 1) * rand) + (t0 - 1))
    function block_r2_draw(dhy, rhsAll, xFuture, h, Th, blockShufflehb, sizeBlock,
            TBurnIn)
        nBlockh = length(blockShufflehb)
        hbShuffle = kron(reshape(blockShufflehb, 1, :), ones(1, sizeBlock))
        hbShuffle = hbShuffle + kron(ones(1, nBlockh), reshape(0:(sizeBlock - 1), 1, :))
        hb = Int.(vec(hbShuffle))
        lhsb = dhy[hb, h + 1]
        rhsb = rhsAll[hb, :]
        fexhb = xFuture[hb, 1:(h + 1)]
        lhsb = lhsb[(TBurnIn + 1):(TBurnIn + Th)]
        rhsb = rhsb[(TBurnIn + 1):(TBurnIn + Th), :]
        fexhb = fexhb[(TBurnIn + 1):(TBurnIn + Th), :]
        feb = lhsb - rhsb * ((rhsb' * rhsb) \ (rhsb' * lhsb))
        Sigmaxb = fexhb' * fexhb / Th
        theta2b = fexhb' * feb / Th
        theta1b = Sigmaxb \ theta2b
        theta3b = feb' * feb / Th
        return theta2b' * theta1b / theta3b, hb[(TBurnIn + 1):(TBurnIn + Th)]
    end

    # NaN-aware comparison of the T × (H+1) arrays the reference initializes to NaN
    function nanapprox(a, b; rtol = 1e-10)
        ok = .!isnan.(b)
        return isequal(isnan.(a), isnan.(b)) && isapprox(a[ok], b[ok]; rtol = rtol)
    end
end

@testitem "fevd parity: lrv_nw.m, VAR(1) prewhitening and the bandwidth rule" tags = [
    :fevd, :verification] setup = [GLMatlabPort] begin
    using LocalProjections
    using LinearAlgebra, Statistics, StableRNGs, Test
    P = GLMatlabPort
    LPj = LocalProjections

    rng = StableRNG(7)
    n = 150
    Z = randn(rng, n, 5)
    for t in 2:n
        Z[t, 2] += 0.6 * Z[t - 1, 2]
        Z[t, 4] += 0.4 * Z[t - 1, 1]
    end
    for L in 0:6
        @test LPj._gl_lrv_nw(Z, L) ≈ P.lrv_nw(Z, L) rtol=1e-13
    end

    # Prewhitening (SubR2.m:213-220): VAR(1) by normal equations, lrv of the n - 1
    # residual rows with the lag of n, recoloured by (I - A)⁻¹; QR in the package
    Zd = Z .- mean(Z; dims = 1)
    for m in (n, 40, 12)
        ZFull = Zd[1:m, :]
        Ωpw, Ω, L = P.pw_omega(ZFull, m)
        @test LPj._gl_bandwidth(m) == L
        @test LPj._gl_prewhitened_lrv(ZFull, L) ≈ Ωpw rtol=1e-10
        @test LPj._gl_lrv_nw(ZFull, L) ≈ Ω rtol=1e-13
    end

    # round(0.75 * T_h^(1/3) - 1): MATLAB rounds half away from zero and the cube
    # root is taken with ^
    for m in 1:6000
        @test LPj._gl_bandwidth(m) == Int(P.mround(0.75 * m^(1 / 3) - 1))
    end
    # an exact tie: 0.75 * 8^(1/3) - 1 == 0.5 rounds to 1 (round-to-even gives 0)
    @test 0.75 * 8^(1 / 3) - 1 == 0.5
    @test round(Int, 0.75 * 8^(1 / 3) - 1) == 0
    @test LPj._gl_bandwidth(8) == 1
    # T_h = 1000: 1000^(1/3) is just below 10, so the lag is 6; cbrt gives 10 and 7
    @test 1000^(1 / 3) < 10 && cbrt(1000.0) == 10
    @test LPj._gl_bandwidth(1000) == 6
    @test Int(P.mround(0.75 * cbrt(1000.0) - 1)) == 7
    # the application's windows (Sec. 5: T_h = 156 - h; App. H2: 380 - h)
    @test all(==(3), [LPj._gl_bandwidth(156 - h) for h in 0:20])
    @test all(==(4), [LPj._gl_bandwidth(380 - h) for h in 0:20])
end

@testitem "fevd parity: SubR2.m (share, θ, GOGT with and without prewhitening, s.e.)" tags = [
    :fevd, :verification] setup = [GLMatlabPort] begin
    using LocalProjections
    using DataFrames, LinearAlgebra, Statistics, StatsModels, Test
    P = GLMatlabPort
    LPj = LocalProjections

    df = P.gl_data()
    T = nrow(df)
    X = Matrix(df[!, [:x1, :x2, :x3]])
    lagX = [4, 2, 3]
    lagy = 2
    H = 6
    cases = [
        # ylevel = 0: lags of Δy, as cumul(dy) or as lags(firstdiff(y)) with ldiff(y);
        # the contemporaneous control x2 is ignored by R² (SubR2 has no XtIndicator)
        (0, 0, df.y,
            @formula(cumul(dy) ~ x1 + lags(x1, 4) + lags(x2, 2) + lags(x3, 3) + lags(dy, 2))),
        (0, 0, df.y,
            @formula(cumul(dy) ~ x1 + x2 + lags(x1, 4) + lags(x2, 2) + lags(x3, 3) +
                                 lags(dy, 2))),
        (0, 0, df.y,
            @formula(ldiff(y) ~ x1 + lags(x1, 4) + lags(x2, 2) + lags(x3, 3) +
                                lags(firstdiff(y), 2))),
        # ylevel = 1: lags of the level
        (1, 0, df.yl,
            @formula(ldiff(yl) ~ x1 + lags(x1, 4) + lags(x2, 2) + lags(x3, 3) + lags(yl, 2))),
        # a linear trend on the absolute row index
        (1, 1, df.yl,
            @formula(ldiff(yl) ~ x1 + lags(x1, 4) + lags(x2, 2) + lags(x3, 3) + lags(yl, 2)))]

    for (ylevel, trend, y, fm) in cases
        ref = P.SubR2(y, X, lagy, lagX, H, ylevel, trend)
        @test ref.t0 == 5

        # private core on the reference's own matrices (demeaned shock lags, W
        # ordered [x lags, y lags, 1, t])
        for h in 0:H
            rows = (ref.t0):(T - h)
            r = LPj._gl_r2(ref.dhy[rows, h + 1], ref.rhsAll[rows, :],
                ref.xFuture[rows, 1:(h + 1)])
            @test r.s ≈ ref.vd[h + 1] rtol=1e-10
            @test r.theta ≈ ref.theta[h + 1] rtol=1e-10
            @test r.fe ≈ ref.forecastError[rows, h + 1] rtol=1e-10
            @test r.avar_pw ≈ ref.GOGTWPW[h + 1] rtol=1e-10
            @test r.avar ≈ ref.GOGTWoPW[h + 1] rtol=1e-10
            @test r.se_pw ≈ ref.seAVarWPW[h + 1] rtol=1e-10
            @test r.se ≈ ref.seAVarWoPW[h + 1] rtol=1e-10
            @test r.bandwidth == ref.bw[h + 1]
            @test r.feasible
        end

        # public estimator: same sample, estimates and inference
        f = fevd(fm, df; horizon = H, estimator = :r2, trend = trend)
        @test f.t0 == ref.t0
        @test f.nobs == ref.effectiveT
        @test f.window == ref.idxt
        @test f.bandwidth == ref.bw
        @test f.share ≈ ref.vd rtol=1e-10
        @test f.se_pw ≈ ref.seAVarWPW rtol=1e-10
        @test f.se ≈ ref.seAVarWoPW rtol=1e-10
        @test f.mse ≈ [θ[end] for θ in ref.theta] rtol=1e-10
        @test P.nanapprox(f.forecast_errors, ref.forecastError)
        for h in 0:H
            @test f.theta[h + 1] ≈ ref.theta[h + 1] rtol=1e-10
            @test f.avar_pw[h + 1] ≈ ref.GOGTWPW[h + 1] rtol=1e-10
            @test f.avar[h + 1] ≈ ref.GOGTWoPW[h + 1] rtol=1e-10
        end
        @test f.shock_mean ≈ mean(X[:, 1]) rtol=1e-14
        @test f.shock_var ≈ var(X[:, 1]) rtol=1e-14
    end
end

@testitem "fevd parity: SubLPA.m / SubLPB.m with the shared LP set" tags = [
    :fevd, :verification] setup = [GLMatlabPort] begin
    using LocalProjections
    using DataFrames, LinearAlgebra, Statistics, StatsModels, Test
    P = GLMatlabPort
    LPj = LocalProjections

    df = P.gl_data()
    T = nrow(df)
    X = Matrix(df[!, [:x1, :x2, :x3]])
    lagX = [4, 2, 3]
    lagy = 2
    XtInd = [1, 1, 0]              # two contemporaneous regressors, shock first
    H = 6
    x = X[:, 1] .- mean(X[:, 1])
    fm_yl = @formula(ldiff(yl) ~ x1 + x2 + lags(x1, 4) + lags(x2, 2) + lags(x3, 3) +
                                 lags(yl, 2))
    forms = [
        # (ylevel, y, trend, formula)
        (0, df.y, 0,
            @formula(cumul(dy) ~ x1 + x2 + lags(x1, 4) + lags(x2, 2) + lags(x3, 3) +
                                 lags(dy, 2))),
        (1, df.yl, 0, fm_yl),
        # a linear trend: k_W = lagy + sum(lagX) + trend + 1 in the d.o.f. adjustment
        (1, df.yl, 1, fm_yl)]

    for (est, which) in ((:lpa, :A), (:lpb, :B)), dof in (true, false),
        (ylevel, y, trend, fm) in forms

        dfInd = dof ? 0 : 1        # the reference's dfIndicator = 0 is the d.o.f. adjustment
        ref = which === :A ?
              P.SubLPA(y, X, lagy, lagX, XtInd, H, ylevel, trend, dfInd) :
              P.SubLPB(y, X, lagy, lagX, XtInd, H, ylevel, trend, dfInd)
        @test ref.t0 == 5 && ref.t0LP == 5

        # private core on the reference's matrices: C = contemporaneous columns,
        # W = the rest of rhsAllFE
        C = ref.rhsAllFE[:, 1:2]
        W = ref.rhsAllFE[:, 3:end]
        r = LPj._gl_lp(ref.dhy, C, W, nothing, x, ref.varx, ref.t0, ref.t0LP, H;
            estimator = est, dofadjust = dof)
        @test r.share ≈ ref.vd rtol=1e-10
        @test r.psi ≈ ref.ir rtol=1e-10
        @test r.irf_se ≈ ref.irSeAVar rtol=1e-10
        @test r.mse ≈ ref.mse rtol=1e-10
        @test r.beta_lp ≈ ref.BetaHat rtol=1e-10
        @test P.nanapprox(r.fe, ref.forecastError)
        @test P.nanapprox(r.resid_lp, ref.residualLP)
        if which === :B
            @test r.sigma2v ≈ ref.sigma2vHat rtol=1e-10
            @test P.nanapprox(r.v, ref.vFE)
        end

        # public estimator
        f = fevd(fm, df; horizon = H, estimator = est, dofadjust = dof, trend = trend)
        @test f.t0 == ref.t0 && f.t0lp == ref.t0LP
        @test f.window == ref.idxt
        @test f.share ≈ ref.vd rtol=1e-10
        @test f.irf ≈ ref.ir rtol=1e-10
        @test f.irf_se ≈ ref.irSeAVar rtol=1e-10
        @test f.mse ≈ ref.mse rtol=1e-10
        @test f.shock_var ≈ ref.varx rtol=1e-14
        @test f.shock_mean ≈ ref.meanx rtol=1e-14
        @test P.nanapprox(f.forecast_errors, ref.forecastError)
        if which === :B
            @test f.sigma2v ≈ ref.sigma2vHat rtol=1e-10
        else
            @test f.sigma2v === nothing
        end
    end

    # the shock keyword moves the shock to the first contemporaneous column
    # (the reference requires X(:,1) to be the shock), and fevd(lp_result)
    # reuses the projection's formula, horizon and shock
    ref = P.SubLPB(df.y, X, lagy, lagX, XtInd, H, 0, 0, 0)
    fsw = fevd(@formula(cumul(dy) ~ x2 + x1 + lags(x1, 4) + lags(x2, 2) + lags(x3, 3) +
                                    lags(dy, 2)), df; horizon = H, estimator = :lpb,
        shock = :x1)
    @test fsw.contemporaneous == [:x1, :x2]
    @test fsw.share ≈ ref.vd rtol=1e-10
    @test fsw.irf ≈ ref.ir rtol=1e-10
    m = lp(@formula(cumul(dy) ~ x1 + x2 + lags(x1, 4) + lags(x2, 2) + lags(x3, 3) +
                                lags(dy, 2)), df; horizon = H)
    flp = fevd(m, df; estimator = :lpb)
    @test flp.share ≈ ref.vd rtol=1e-10
    @test flp.irf ≈ ref.ir rtol=1e-10
    @test flp.irf_se ≈ ref.irSeAVar rtol=1e-10
end

@testitem "fevd parity: SubLPA.m / SubLPB.m with a separate LP set (XLP)" tags = [
    :fevd, :verification] setup = [GLMatlabPort] begin
    using LocalProjections
    using DataFrames, LinearAlgebra, Statistics, StatsModels, Test
    P = GLMatlabPort
    LPj = LocalProjections

    df = P.gl_data()
    T = nrow(df)
    X = Matrix(df[!, [:x1, :x2, :x3]])
    lagX = [4, 2, 3]
    lagy = 2
    XtInd = [1, 1, 0]
    H = 6
    x = X[:, 1] .- mean(X[:, 1])
    cases = [
        # ylevel = 0, lagyLP = 1, lagXLP = [1,0,0], XtIndicatorLP = [1,0,0]
        # (the TFP set of the application): t0LP = 3 needs lags(firstdiff(y))
        (0, df.y, 1, [1, 0, 0], [1, 0, 0],
            @formula(ldiff(y) ~ x1 + x2 + lags(x1, 4) + lags(x2, 2) + lags(x3, 3) +
                                lags(firstdiff(y), 2)),
            @formula(ldiff(y) ~ x1 + lags(x1, 1) + lags(firstdiff(y), 1))),
        # ylevel = 1, lagyLP = 1, lagXLP = [1,0,0], XtIndicatorLP = [1,1,0]
        (1, df.yl, 1, [1, 0, 0], [1, 1, 0],
            @formula(ldiff(yl) ~ x1 + x2 + lags(x1, 4) + lags(x2, 2) + lags(x3, 3) +
                                 lags(yl, 2)),
            @formula(ldiff(yl) ~ x1 + x2 + lags(x1, 1) + lags(yl, 1))),
        # ylevel = 0, lagyLP = 0, lagXLP = 0, XtIndicatorLP = [1,1,0] (the MP set
        # of the application, contemporaneous regressors only)
        (0, df.y, 0, [0, 0, 0], [1, 1, 0],
            @formula(ldiff(y) ~ x1 + x2 + lags(x1, 4) + lags(x2, 2) + lags(x3, 3) +
                                lags(firstdiff(y), 2)),
            @formula(ldiff(y) ~ x1 + x2))]

    for (est, which) in ((:lpa, :A), (:lpb, :B)), dof in (true, false),
        (ylevel, y, lagyLP, lagXLP, XtIndLP, fm, fmlp) in cases

        dfInd = dof ? 0 : 1
        args = (y, X, lagy, lagX, XtInd, H, ylevel, 0, dfInd, X, lagyLP, lagXLP, XtIndLP)
        ref = which === :A ? P.SubLPA(args...) : P.SubLPB(args...)

        C = ref.rhsAllFE[:, 1:2]
        W = ref.rhsAllFE[:, 3:end]
        r = LPj._gl_lp(ref.dhy, C, W, ref.rhsAllLP, x, ref.varx, ref.t0, ref.t0LP, H;
            estimator = est, dofadjust = dof)
        @test r.share ≈ ref.vd rtol=1e-10
        @test r.psi ≈ ref.ir rtol=1e-10
        @test r.irf_se ≈ ref.irSeAVar rtol=1e-10
        @test r.mse ≈ ref.mse rtol=1e-10
        @test r.beta_lp ≈ ref.BetaHat rtol=1e-10
        @test P.nanapprox(r.fe, ref.forecastError)
        @test P.nanapprox(r.resid_lp, ref.residualLP)
        if which === :B
            @test r.sigma2v ≈ ref.sigma2vHat rtol=1e-10
            @test P.nanapprox(r.v, ref.vFE)
        end

        f = fevd(fm, df; horizon = H, estimator = est, dofadjust = dof, irf_formula = fmlp)
        @test f.t0 == ref.t0
        @test f.t0lp == ref.t0LP
        @test f.share ≈ ref.vd rtol=1e-10
        @test f.irf ≈ ref.ir rtol=1e-10
        @test f.irf_se ≈ ref.irSeAVar rtol=1e-10
        @test f.mse ≈ ref.mse rtol=1e-10
        @test P.nanapprox(f.forecast_errors, ref.forecastError)
        which === :B && @test f.sigma2v ≈ ref.sigma2vHat rtol=1e-10
    end

    # Documented deviation (not a port): with a cumul(dy) response the package's
    # dy column exists at t = 1, so lags(dy, 1) starts the impulse-response sample
    # at t0lp = 2 where the reference (yendo = [NaN; diff(y)]) starts at 3.
    f = fevd(@formula(cumul(dy) ~ x1 + x2 + lags(x1, 4) + lags(x2, 2) + lags(x3, 3) +
                                  lags(dy, 2)), df; horizon = 2, estimator = :lpb,
        irf_formula = @formula(cumul(dy) ~ x1 + lags(x1, 1) + lags(dy, 1)))
    @test f.t0 == 5 && f.t0lp == 2
end

@testitem "fevd parity: SimulationAVarLPA.m / SimulationAVarLPB.m moment covariance and gradient" tags = [
    :fevd, :verification] setup = [GLMatlabPort] begin
    using LocalProjections
    using DataFrames, LinearAlgebra, Statistics, StatsModels, Test
    P = GLMatlabPort
    LPj = LocalProjections

    df = P.gl_data()
    T = nrow(df)
    X = Matrix(df[!, [:x1, :x2, :x3]])
    lagX = [4, 2, 3]
    lagy = 2
    XtInd = [1, 1, 0]
    fm0 = @formula(ldiff(y) ~ x1 + x2 + lags(x1, 4) + lags(x2, 2) + lags(x3, 3) +
                              lags(firstdiff(y), 2))
    cases = [
        # (H, ylevel, y, formula, irf_formula, lagyLP, lagXLP, XtIndicatorLP)
        (4, 0, df.y, fm0, nothing, lagy, lagX, XtInd),
        (6, 0, df.y, fm0, @formula(ldiff(y) ~ x1 + lags(x1, 1) + lags(firstdiff(y), 1)),
            1, [1, 0, 0], [1, 0, 0]),
        (6, 0, df.y, fm0, @formula(ldiff(y) ~ x1 + x2), 0, [0, 0, 0], [1, 1, 0]),
        (6, 1, df.yl,
            @formula(ldiff(yl) ~ x1 + x2 + lags(x1, 4) + lags(x2, 2) + lags(x3, 3) +
                                 lags(yl, 2)),
            @formula(ldiff(yl) ~ x1 + x2 + lags(x1, 1) + lags(yl, 1)), 1, [1, 0, 0],
            [1, 1, 0])]

    for (est, which) in ((:lpa, :A), (:lpb, :B)), dof in (true, false),
        (H, ylevel, y, fm, fmlp, lagyLP, lagXLP, XtIndLP) in cases

        sim = P.SimulationAVarLP(which, y, X, H, ylevel, 0, lagy, lagX, XtInd,
            dof ? 0 : 1, X, lagyLP, lagXLP, XtIndLP)
        sub = sim.sub
        for h in 0:H
            # private core on the reference's matrices
            err = which === :A ? sub.forecastError[:, h + 1] : sub.vFE[:, h + 1]
            σ²h = which === :A ? sub.mse[h + 1] : sub.sigma2vHat[h + 1]
            a = LPj._gl_lp_avar(sub.rhsAllLP, sub.residualLP, err, sim.x, sub.varx, σ²h,
                sub.BetaHat, sim.Sigmaq, sub.t0, sub.t0LP, h)
            @test a.theta ≈ sim.thetas[h + 1] rtol=1e-10
            @test a.avar_pw ≈ sim.GOGTWPW[h + 1] rtol=1e-10
            @test a.avar ≈ sim.GOGTWoPW[h + 1] rtol=1e-10
            @test a.nobs == T - h - max(sub.t0, sub.t0LP) + 1
            @test a.feasible
            # the gradient at ŝ and at another share value (the simulation method
            # evaluates it at vdBc)
            for s in (sub.vd[h + 1], sub.vd[h + 1] - 0.07)
                g = LPj._gl_lp_gradient(est, sub.ir[1:(h + 1)], sub.varx, σ²h,
                    sub.mse[h + 1], s, sim.Nq)
                @test g ≈ sim.Delta(h, s) rtol=1e-12
            end
        end

        # public estimator: θ̂_h and V_h in the package's parametrization of q
        f = fevd(fm, df; horizon = H, estimator = est, dofadjust = dof,
            irf_formula = fmlp)
        Qpkg = fmlp === nothing ? hcat(f.design.C, f.design.W) : f.design.q
        rows = (sub.t0LP):T
        for h in 0:H
            J, M = P.lp_reparam(sub.rhsAllLP, Qpkg, rows, h)
            @test sub.rhsAllLP[rows, :] * M ≈ Qpkg[rows, :] rtol=1e-12
            @test f.theta[h + 1] ≈ J * sim.thetas[h + 1] rtol=1e-10
            @test f.avar_pw[h + 1] ≈ J * sim.GOGTWPW[h + 1] * J' rtol=1e-10
            @test f.avar[h + 1] ≈ J * sim.GOGTWoPW[h + 1] * J' rtol=1e-10
            Δ = sim.Delta(h, f.share[h + 1])
            @test f.se_pw[h + 1] ≈ sqrt(Δ' * sim.GOGTWPW[h + 1] * Δ) rtol=1e-10
            @test f.se[h + 1] ≈ sqrt(Δ' * sim.GOGTWoPW[h + 1] * Δ) rtol=1e-10
        end
        # with q = [x_t, x2_t, 1] the two parametrizations coincide exactly
        if fmlp !== nothing && lagyLP == 0
            for h in 0:H
                @test f.theta[h + 1] ≈ sim.thetas[h + 1] rtol=1e-10
                @test f.avar_pw[h + 1] ≈ sim.GOGTWPW[h + 1] rtol=1e-10
            end
        end
    end
end

@testitem "fevd parity: G-L lag selection, var_chol.m FEVD and var_bootstrap.m recursion" tags = [
    :fevd, :verification] setup = [GLMatlabPort] begin
    using LocalProjections
    using DataFrames, LinearAlgebra, Statistics, StableRNGs, Test
    P = GLMatlabPort
    LPj = LocalProjections

    df = P.gl_data()
    T = nrow(df)
    X = Matrix(df[!, [:x1, :x2, :x3]])
    H = 12
    # ylevel = 0, XVAROrder = [2,1,3], yPlace = 4: XVAR = [X(2:end,[2,1,3]), diff(y)]
    # (Ty - 1 rows); the shock is variable 2, Δy (variable 4) is cumulated.
    # ylevel = 1, XVAROrder = [1,2,3], yPlace = 4: XVAR = [X, y].
    systems = [([X[2:end, [2, 1, 3]] diff(df.y)], 2, [4]), ([X df.yl], 1, Int[])]

    for (XVAR, xloc, diffidx) in systems
        # BootstrapVARR2.m #2: common sample of rows(XVAR) - max(set), MLE Ω, n²p
        for (ic, crit) in ((0, :bic), (1, :hqic), (2, :aic)),
            set in ([1, 2, 3, 4, 5, 6], [2, 5, 3])

            pref, IC = P.var_ic(XVAR, set, ic, 0)
            sel = LPj._gl_lagselect(XVAR, set; criterion = crit)
            @test sel.p == pref
            @test sel.values ≈ IC rtol=1e-10
        end

        for p in (1, 2, 3)
            ref = P.var_chol_vd(XVAR, p, H, xloc, diffidx, 0)
            # var_chol's recursion on the reference's own A and Ω, every shock and
            # response, with and without the differencedIndex cumulation
            for j in 1:4
                refj = P.var_chol_vd(XVAR, p, H, j, diffidx, 0)
                refj_lvl = P.var_chol_vd(XVAR, p, H, j, Int[], 0)
                for r in 1:4
                    @test LPj._var_fevd(ref.est.phi, ref.est.omega, H; shock = j,
                        response = r, cumulate = r in diffidx) ≈ refj.vd[r, :] rtol=1e-10
                    @test LPj._var_fevd(ref.est.phi, ref.est.omega, H; shock = j,
                        response = r) ≈ refj_lvl.vd[r, :] rtol=1e-10
                end
            end
            # the package's VAR fit (QR) and MLE Ω reproduce var_estimation.m
            v = LPj._var_ols(XVAR, p)
            Ω = (v.U' * v.U) ./ v.Tu
            @test v.A ≈ ref.est.phi rtol=1e-10
            @test v.c ≈ ref.est.c[:, 1] rtol=1e-10
            @test Ω ≈ ref.est.omega rtol=1e-10
            @test v.U ≈ Matrix(ref.est.residual') rtol=1e-10
            @test LPj._var_fevd(v.A, Ω, H; shock = xloc, response = 4,
                cumulate = !isempty(diffidx)) ≈ ref.vd[4, :] rtol=1e-10
        end

        # var_bootstrap.m with fixed residual indices and initial block vs
        # _var_simulate under the mapping of spec §3.4: state s of the companion
        # recursion is row p + s - 1 of the simulated matrix, residual idx(s) feeds
        # row p + s - 1 (idx(1) unused), and the output rows TBurnIn+1:TBurnIn+Ty
        # are rows p+TBurnIn:p+TBurnIn+Ty-1.
        burnin = 100
        rng = StableRNG(99)
        for p in (1, 3)
            est = P.var_estimation(XVAR, p, 0, 0)
            Tu = size(XVAR, 1) - p
            for init in (1, Tu + 1, rand(rng, 1:(Tu + 1)))
                idx = rand(rng, 1:Tu, burnin + T)
                ref = P.var_bootstrap(est.c, est.phi, XVAR, est.residual, T, burnin, idx,
                    init)
                Yb = LPj._var_simulate(est.c[:, 1], est.phi,
                    Matrix(est.residual')[idx[2:end], :], XVAR[init:(init + p - 1), :])
                @test size(Yb, 1) == p + burnin + T - 1
                @test Yb[(p + burnin):(p + burnin + T - 1), :] ≈ ref rtol=1e-12
            end
        end
    end
end

@testitem "fevd parity: BootstrapVARR2.m / BootstrapVARLPA.m / BootstrapVARLPB.m replayed draw by draw" tags = [
    :fevd, :verification] setup = [GLMatlabPort] begin
    using LocalProjections
    using DataFrames, Distributions, LinearAlgebra, Random, Statistics, StableRNGs,
          StatsModels, Test
    P = GLMatlabPort

    # The residual indices and initial blocks of every draw are regenerated from
    # the package's per-draw seeds (`seeds = rand(rng, UInt64, nboot)`, then
    # `Xoshiro(seed_b)`) and fed to the literal var_bootstrap.m / SubR2.m /
    # SubLPB.m. This ties the test to the package's order of random draws: if that
    # order changes, update `P.var_draw_indices`.
    df = P.gl_data()
    T = nrow(df)
    X = Matrix(df[!, [:x1, :x2, :x3]])
    lagX = [4, 2, 3]
    lagy = 2
    XtInd = [1, 1, 0]
    H = 6
    B = 30
    burnin = 100
    q = [0.05, 0.95]

    # --- R², ylevel = 0, XVAROrder = [2,1,3], yPlace = 4, rows 2:T (skipfirst) ---
    f = fevd(@formula(cumul(dy) ~ x1 + lags(x1, 4) + lags(x2, 2) + lags(x3, 3) +
                                  lags(dy, 2)), df; horizon = H, estimator = :r2)
    b = fevdbootstrap(f, df; method = :var, vars = [:x2, :x1, :x3, :dy], nlags = 1:4,
        skipfirst = true, nboot = B, burnin = burnin, rng = StableRNG(5))
    @test b.nfail == 0

    XVAR = [X[2:end, [2, 1, 3]] diff(df.y)]
    XVAROrder = [2, 1, 3]
    InvXVAROrder = [findfirst(==(n), XVAROrder) for n in 1:3]
    lagVAR, IC = P.var_ic(XVAR, 1:4, 1, 0)
    @test b.nlags == lagVAR
    @test b.ic.values ≈ IC rtol=1e-10
    vc = P.var_chol_vd(XVAR, lagVAR, H, 2, [4], 0)
    vdVAR = vc.vd[4, :]
    @test b.pseudo_truth ≈ vdVAR rtol=1e-10
    ref = P.SubR2(df.y, X, lagy, lagX, H, 0, 0)

    seeds = rand(StableRNG(5), UInt64, B)
    Tu = size(XVAR, 1) - lagVAR
    vdb = zeros(B, H + 1)
    for idxb in 1:B
        idx, init = P.var_draw_indices(seeds[idxb], Tu, T, burnin)
        XVARb = P.var_bootstrap(vc.est.c, vc.est.phi, XVAR, vc.est.residual, T, burnin,
            idx, init)
        yb = cumsum(XVARb[:, 4])
        Xb = XVARb[:, setdiff(1:4, 4)][:, InvXVAROrder]
        vdb[idxb, :] = P.SubR2_vd(yb, Xb, lagy, lagX, H, 0, 0)
    end
    @test b.draws ≈ vdb rtol=1e-10
    mvdb = vec(mean(vdb; dims = 1))
    vdBc = ref.vd - (mvdb - vdVAR)
    @test b.mean_draws ≈ mvdb rtol=1e-10
    @test b.biascorrected ≈ vdBc rtol=1e-10
    @test b.se_boot ≈ vec(std(vdb, dims = 1)) rtol=1e-10
    @test b.se_pw ≈ ref.seAVarWPW rtol=1e-10        # SubR2's values at ŝ
    @test b.se ≈ ref.seAVarWoPW rtol=1e-10
    # cbIndicator = 0: quantile(vdb) + (vd - mvdb - (mvdb - vdVAR))
    cb = [P.mquantile(vdb[:, h + 1], qq) + (ref.vd[h + 1] - mvdb[h + 1] -
                                            (mvdb[h + 1] - vdVAR[h + 1]))
          for qq in q, h in 0:H]
    s = summarize(b; level = 0.90)
    @test s.lower ≈ cb[1, :] rtol=1e-10
    @test s.upper ≈ cb[2, :] rtol=1e-10
    # the uncorrected band of the figures (AppSimulation.m:325-329): vd + (cb - vdBc)
    s0 = summarize(b; level = 0.90, biascorrected = false)
    @test s0.lower ≈ ref.vd + (cb[1, :] - vdBc) rtol=1e-10
    @test s0.upper ≈ ref.vd + (cb[2, :] - vdBc) rtol=1e-10
    # cbIndicator = 1 / 2: vdBc + norminv(q) * vdSeAVarWPW / WoPW
    sn = summarize(b; level = 0.90, band = :normal)
    @test sn.lower ≈ vdBc + quantile(Normal(), q[1]) * ref.seAVarWPW rtol=1e-10
    @test sn.upper ≈ vdBc + quantile(Normal(), q[2]) * ref.seAVarWPW rtol=1e-10
    sn2 = summarize(b; level = 0.90, band = :normal, prewhiten = false)
    @test sn2.lower ≈ vdBc + quantile(Normal(), q[1]) * ref.seAVarWoPW rtol=1e-10

    # --- LP-A / LP-B (BootstrapVARLPA.m / BootstrapVARLPB.m), ylevel = 1,
    # XVAROrder = [1,2,3], yPlace = 4, fixed lagVAR = 2 ---
    XVAR = [X df.yl]
    vc = P.var_chol_vd(XVAR, 2, H, 1, Int[], 0)
    vdVAR = vc.vd[4, :]
    Tu = size(XVAR, 1) - 2
    for (est, which, seed) in ((:lpa, :A, 6), (:lpb, :B, 7))
        local f, b, ref, seeds, vdb, mvdb, cb, s
        sub = which === :A ? P.SubLPA : P.SubLPB
        f = fevd(@formula(ldiff(yl) ~ x1 + x2 + lags(x1, 4) + lags(x2, 2) + lags(x3, 3) +
                                      lags(yl, 2)), df; horizon = H, estimator = est)
        b = fevdbootstrap(f, df; method = :var, vars = [:x1, :x2, :x3, :yl], nlags = 2,
            nboot = B, burnin = burnin, rng = StableRNG(seed))
        @test b.nfail == 0 && b.nlags == 2 && b.ic === nothing
        @test b.pseudo_truth ≈ vdVAR rtol=1e-10
        ref = sub(df.yl, X, lagy, lagX, XtInd, H, 1, 0, 0)
        seeds = rand(StableRNG(seed), UInt64, B)
        irb = zeros(B, H + 1)
        irStrb = zeros(B, H + 1)
        vdb = zeros(B, H + 1)
        for idxb in 1:B
            idx, init = P.var_draw_indices(seeds[idxb], Tu, T, burnin)
            XVARb = P.var_bootstrap(vc.est.c, vc.est.phi, XVAR, vc.est.residual, T,
                burnin, idx, init)
            yb = XVARb[:, 4]
            Xb = XVARb[:, setdiff(1:4, 4)][:, [1, 2, 3]]
            rb = sub(yb, Xb, lagy, lagX, XtInd, H, 1, 0, 0)
            irb[idxb, :] = rb.ir
            vdb[idxb, :] = rb.vd
            irStrb[idxb, :] = rb.ir * std(Xb[:, 1])
        end
        @test b.draws ≈ vdb rtol=1e-10
        @test b.irf_draws ≈ irb rtol=1e-10
        @test b.irfstr_draws ≈ irStrb rtol=1e-10
        @test b.irf_bc ≈ 2 * ref.ir - vec(mean(irb; dims = 1)) rtol=1e-10
        @test b.irf_se_boot ≈ vec(std(irb, dims = 1)) rtol=1e-10
        mvdb = vec(mean(vdb; dims = 1))
        @test b.biascorrected ≈ ref.vd - (mvdb - vdVAR) rtol=1e-10
        @test b.se_boot ≈ vec(std(vdb, dims = 1)) rtol=1e-10
        cb = [P.mquantile(vdb[:, h + 1], qq) + (ref.vd[h + 1] - mvdb[h + 1] -
                                                (mvdb[h + 1] - vdVAR[h + 1]))
              for qq in q, h in 0:H]
        s = summarize(b; level = 0.90)
        @test s.lower ≈ cb[1, :] rtol=1e-10
        @test s.upper ≈ cb[2, :] rtol=1e-10
    end
end

@testitem "fevd parity: BootstrapBlockR2.m (fixed block starts, replay, degeneracy)" tags = [
    :fevd, :verification] setup = [GLMatlabPort] begin
    using LocalProjections
    using DataFrames, Distributions, LinearAlgebra, Random, Statistics, StableRNGs,
          StatsModels, Test
    P = GLMatlabPort
    LPj = LocalProjections

    df = P.gl_data()
    T = nrow(df)
    X = Matrix(df[!, [:x1, :x2, :x3]])
    lagX = [4, 2, 3]
    lagy = 2
    H = 4
    fm = @formula(cumul(dy) ~ x1 + lags(x1, 4) + lags(x2, 2) + lags(x3, 3) + lags(dy, 2))
    f = fevd(fm, df; horizon = H, estimator = :r2)
    ref = P.SubR2(df.y, X, lagy, lagX, H, 0, 0)
    t0 = ref.t0

    # fixed block starts through the package's row-resampling hook
    ℓ = 4
    for h in 0:H
        Th = T - h - t0 + 1
        nb = cld(Th, ℓ)                      # TBurnIn = 0 (spec Q11)
        starts = [t0 + mod(37k + 11, Th - ℓ + 1) for k in 1:nb]
        vd_ref, rows = P.block_r2_draw(ref.dhy, ref.rhsAll, ref.xFuture, h, Th, starts, ℓ, 0)
        @test LPj._gl_r2_rows(f.design, rows, h) ≈ vd_ref rtol=1e-10
    end

    # replay of fevdbootstrap(...; method = :block): the per-draw seeds give the
    # block starts, horizon by horizon, as the package draws them
    B = 40
    b = fevdbootstrap(f; method = :block, blocklength = ℓ, nboot = B, rng = StableRNG(8))
    @test b.nfail == 0
    seeds = rand(StableRNG(8), UInt64, B)
    vdb = zeros(B, H + 1)
    for idxb in 1:B
        drng = Xoshiro(seeds[idxb])
        for h in 0:H
            Th = T - h - t0 + 1
            starts = [rand(drng, t0:(T - h - ℓ + 1)) for _ in 1:cld(Th, ℓ)]
            vdb[idxb, h + 1] = P.block_r2_draw(ref.dhy, ref.rhsAll, ref.xFuture, h, Th,
                starts, ℓ, 0)[1]
        end
    end
    @test b.draws ≈ vdb rtol=1e-10
    mvdb = vec(mean(vdb; dims = 1))
    vdBc = ref.vd - (mvdb - ref.vd)
    @test b.biascorrected ≈ vdBc rtol=1e-10
    @test b.se_boot ≈ vec(std(vdb, dims = 1)) rtol=1e-10
    @test b.se_pw ≈ ref.seAVarWPW rtol=1e-10
    q = [0.05, 0.95]
    cb = [P.mquantile(vdb[:, h + 1], qq) + (ref.vd[h + 1] - mvdb[h + 1] -
                                            (mvdb[h + 1] - ref.vd[h + 1]))
          for qq in q, h in 0:H]
    s = summarize(b; level = 0.90)
    @test s.lower ≈ cb[1, :] rtol=1e-10
    @test s.upper ≈ cb[2, :] rtol=1e-10
    sn = summarize(b; level = 0.90, band = :normal, prewhiten = false)
    @test sn.upper ≈ vdBc + quantile(Normal(), q[2]) * ref.seAVarWoPW rtol=1e-10
    # another level: quantiles (0.16, 0.84)
    cb68 = [P.mquantile(vdb[:, h + 1], qq) + vdBc[h + 1] - mvdb[h + 1]
            for qq in (0.16, 0.84), h in 0:H]
    s68 = summarize(b; level = 0.68)
    @test s68.lower ≈ cb68[1, :] rtol=1e-10
    @test s68.upper ≈ cb68[2, :] rtol=1e-10

    # degeneracy: with sizeBlock = T_h the only block start is t0. The reference,
    # with its 100 burn-in rows, then re-estimates on a rotation of the sample
    # (same R²); the package draws exactly the original rows.
    ℓH = f.nobs[end]
    bH = fevdbootstrap(f; method = :block, blocklength = ℓH, nboot = 10, rng = StableRNG(1))
    @test all(==(f.share[end]), bH.draws[:, end])
    Th = T - H - t0 + 1
    nBlock = cld(Th + 100, ℓH)
    vd_ref, _ = P.block_r2_draw(ref.dhy, ref.rhsAll, ref.xFuture, H, Th, fill(t0, nBlock),
        ℓH, 100)
    @test vd_ref ≈ ref.vd[end] rtol=1e-10
    f0 = fevd(fm, df; horizon = 0, estimator = :r2)
    b0 = fevdbootstrap(f0; method = :block, blocklength = f0.nobs[1], nboot = 10,
        rng = StableRNG(2))
    @test all(==(f0.share[1]), b0.draws)
    @test b0.biascorrected == f0.share
    s0 = summarize(b0; level = 0.90)
    @test s0.lower ≈ f0.share rtol=1e-14
    @test s0.upper ≈ f0.share rtol=1e-14
end

@testitem "fevd parity: SimulationAVarR2.m / SimulationAVarLP*.m replayed draw by draw" tags = [
    :fevd, :verification] setup = [GLMatlabPort] begin
    using LocalProjections
    using DataFrames, Distributions, LinearAlgebra, Random, Statistics, StableRNGs,
          StatsModels, Test
    P = GLMatlabPort

    # mvnrnd(θ̂, V) draws θ̂ + chol(V)' z when V is positive definite. The standard
    # normal vectors z are regenerated from the package's per-draw seeds (one
    # `Xoshiro(seed_b)` stream per draw, horizons in order, a fresh z per
    # rejection) and fed to the literal simulation, so both see the same z.
    df = P.gl_data()
    T = nrow(df)
    X = Matrix(df[!, [:x1, :x2, :x3]])
    lagX = [4, 2, 3]
    lagy = 2
    XtInd = [1, 1, 0]
    H = 4
    B = 200
    q = [0.05, 0.95]

    # --- R² (SimulationAVarR2.m), boundIndicator = 1 / 0, seIndicator ≠ 2 / = 2 ---
    f = fevd(@formula(cumul(dy) ~ x1 + lags(x1, 4) + lags(x2, 2) + lags(x3, 3) +
                                  lags(dy, 2)), df; horizon = H, estimator = :r2)
    ref = P.SubR2(df.y, X, lagy, lagX, H, 0, 0)
    for (prewhiten, bound, seed) in ((true, true, 3), (false, true, 4), (true, false, 5))
        b = fevdbootstrap(f; method = :simulation, nboot = B, prewhiten = prewhiten,
            bound = bound, rng = StableRNG(seed))
        @test b.nfail == 0
        V = prewhiten ? ref.GOGTWPW : ref.GOGTWoPW
        Ls = Matrix{Float64}[]
        for h in 0:H
            @test isposdef(Symmetric(V[h + 1]))
            push!(Ls, Matrix(cholesky(Symmetric(V[h + 1], :U)).L))   # chol(V)'
        end
        seeds = rand(StableRNG(seed), UInt64, B)
        vdb = zeros(B, H + 1)
        nrej = 0
        for idxb in 1:B
            drng = Xoshiro(seeds[idxb])
            for h in 0:H
                θh = ref.theta[h + 1]
                z = Vector{Float64}(undef, 2h + 3)
                local θb
                while true
                    θb = θh + Ls[h + 1] * randn!(drng, z)
                    θb[end] > 0 && break
                    nrej += 1
                end
                vdb[idxb, h + 1] = sum(θb[1:(h + 1)] .* θb[(h + 2):(2h + 2)]) / θb[end]
            end
        end
        @test nrej == 0
        bound && (vdb = min.(max.(vdb, 0), 1))
        @test b.draws ≈ vdb rtol=1e-10
        mvdb = vec(mean(vdb; dims = 1))
        vdBc = ref.vd - (mvdb - ref.vd)
        @test b.biascorrected ≈ vdBc rtol=1e-10
        @test b.se_boot ≈ vec(std(vdb, dims = 1)) rtol=1e-10
        # standard errors with ξ evaluated at vdBc (SimulationAVarR2.m:183-190)
        sepw = zeros(H + 1)
        se = zeros(H + 1)
        for h in 0:H
            θh = ref.theta[h + 1]
            xih = 1 / θh[2h + 3] * [θh[(h + 2):(2 * (h + 1))]; θh[1:(h + 1)]; -vdBc[h + 1]]
            sepw[h + 1] = sqrt(xih' * ref.GOGTWPW[h + 1] * xih)
            se[h + 1] = sqrt(xih' * ref.GOGTWoPW[h + 1] * xih)
        end
        @test b.se_pw ≈ sepw rtol=1e-10
        @test b.se ≈ se rtol=1e-10
        # cbIndicator = 0 and 1
        cb = [P.mquantile(vdb[:, h + 1], qq) + (ref.vd[h + 1] - mvdb[h + 1] -
                                                (mvdb[h + 1] - ref.vd[h + 1]))
              for qq in q, h in 0:H]
        s = summarize(b; level = 0.90)
        @test s.lower ≈ cb[1, :] rtol=1e-10
        @test s.upper ≈ cb[2, :] rtol=1e-10
        sn = summarize(b; level = 0.90, band = :normal)
        @test sn.lower ≈ vdBc + quantile(Normal(), q[1]) * sepw rtol=1e-10
        @test sn.upper ≈ vdBc + quantile(Normal(), q[2]) * sepw rtol=1e-10
    end

    # --- LP-A / LP-B (SimulationAVarLP*.m, jointIndicator = 0) with the LP set
    # q = [x_t, x2_t, 1] (lagyLP = 0, lagXLP = 0, XtIndicatorLP = [1,1,0]), whose
    # coefficient vector is ordered as in the reference ---
    fm = @formula(ldiff(y) ~ x1 + x2 + lags(x1, 4) + lags(x2, 2) + lags(x3, 3) +
                             lags(firstdiff(y), 2))
    for (est, which, seed) in ((:lpa, :A, 11), (:lpb, :B, 12)),
        prewhiten in (true, false)

        flp = fevd(fm, df; horizon = H, estimator = est,
            irf_formula = @formula(ldiff(y) ~ x1 + x2))
        sim = P.SimulationAVarLP(which, df.y, X, H, 0, 0, lagy, lagX, XtInd, 0, X, 0,
            [0, 0, 0], [1, 1, 0])
        sub = sim.sub
        Nq = sim.Nq
        b = fevdbootstrap(flp; method = :simulation, nboot = B, prewhiten = prewhiten,
            rng = StableRNG(seed))
        @test b.nfail == 0
        @test b.bound == false                 # LP draws are never clamped
        V = prewhiten ? sim.GOGTWPW : sim.GOGTWoPW
        seeds = rand(StableRNG(seed), UInt64, B)
        vdb = zeros(B, H + 1)
        nrej = 0
        Ls = [Matrix(cholesky(Symmetric(V[h + 1], :U)).L) for h in 0:H]
        for idxb in 1:B
            drng = Xoshiro(seeds[idxb])
            for h in 0:H
                θh = sim.thetas[h + 1]
                z = Vector{Float64}(undef, length(θh))
                local θb
                while true
                    θb = θh + Ls[h + 1] * randn!(drng, z)
                    (θb[end - 1] > 0 && θb[end] > 0) && break
                    nrej += 1
                end
                psixhb = θb[1 .+ (0:h) * Nq]
                varxhb = θb[end - 1]
                spsixhb2 = sum(psixhb .^ 2)
                vdb[idxb, h + 1] = which === :A ? spsixhb2 * varxhb / θb[end] :
                                   spsixhb2 * varxhb / (spsixhb2 * varxhb + θb[end])
            end
        end
        @test nrej == 0
        @test b.draws ≈ vdb rtol=1e-10
        vdBc = 2 * sub.vd - vec(mean(vdb; dims = 1))
        @test b.biascorrected ≈ vdBc rtol=1e-10
        @test b.se_boot ≈ vec(std(vdb, dims = 1)) rtol=1e-10
        # Deltah at vdBc (SimulationAVarLPA.m:319-327, SimulationAVarLPB.m:319-323)
        @test b.se_pw ≈ [sqrt(sim.Delta(h, vdBc[h + 1])' * sim.GOGTWPW[h + 1] *
                              sim.Delta(h, vdBc[h + 1])) for h in 0:H] rtol=1e-10
        @test b.se ≈ [sqrt(sim.Delta(h, vdBc[h + 1])' * sim.GOGTWoPW[h + 1] *
                           sim.Delta(h, vdBc[h + 1])) for h in 0:H] rtol=1e-10
        mvdb = vec(mean(vdb; dims = 1))
        cb = [P.mquantile(vdb[:, h + 1], qq) + (sub.vd[h + 1] - mvdb[h + 1] -
                                                (mvdb[h + 1] - sub.vd[h + 1]))
              for qq in q, h in 0:H]
        s = summarize(b; level = 0.90)
        @test s.lower ≈ cb[1, :] rtol=1e-10
        @test s.upper ≈ cb[2, :] rtol=1e-10
    end
end

@testitem "fevd parity: MATLAB quantile (type 5)" tags = [:fevd, :verification] setup = [
    GLMatlabPort] begin
    using LocalProjections
    using StableRNGs, Statistics, Test
    P = GLMatlabPort
    LPj = LocalProjections

    rng = StableRNG(17)
    for n in (1, 2, 3, 4, 7, 20, 101, 2000)
        v = randn(rng, n)
        ps = unique([0.0, 0.01, 0.05, 0.25 / n, 0.5 / n, 0.75 / n, 0.1, 0.5, 0.9, 0.95,
            1 - 0.5 / n, 1 - 0.25 / n, 0.99, 1.0])
        for p in filter(p -> 0 <= p <= 1, ps)
            @test LPj._gl_quantile5(v, p) ≈ P.mquantile(v, p) rtol=1e-14 atol=1e-15
        end
    end
    # hand values: n = 4, positions (i - 0.5)/4 = 0.125, 0.375, 0.625, 0.875
    v = [4.0, 1.0, 3.0, 2.0]
    @test LPj._gl_quantile5(v, 0.1) == 1.0
    @test LPj._gl_quantile5(v, 0.125) == 1.0
    @test LPj._gl_quantile5(v, 0.25) ≈ 1.5
    @test LPj._gl_quantile5(v, 0.5) ≈ 2.5
    @test LPj._gl_quantile5(v, 0.95) == 4.0
    @test P.mquantile(v, 0.25) ≈ 1.5 && P.mquantile(v, 0.95) == 4.0
end
