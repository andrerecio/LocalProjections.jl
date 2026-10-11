module LocalProjections

export LocalProjection, LocalProjectionIV, LocalProjectionCovariance, IRFSummary
export lp, lpiv, coefpath, stderror, vcov, summarize, first_stage, weakivtest
export LPState, statetest, StateTestResult
export ewc_bandwidth
export biascorrect, BiasCorrectedLP
export lagselect, VARLagSelection, nlags
export varbootstrap, LPBootstrap
export fevd, LPFEVD, fevdbootstrap, FEVDBootstrap, FEVDSummary
export WeakIVTestResult, FirstStageIV
export lag, lead, cumul, CumulTerm, lags, leads, LeadTerm, anchor, AnchorTerm
export ldiff, hbr, LongDiffTerm, firstdiff, DiffTerm
export as_irf_result, LocalProjectionIRFResult
export irfplot, irfplot!, irfplot_axis

using DataFrames
using PrettyTables: pretty_table, TextHighlighter, TextTableFormat,
                    text_table_borders__unicode_rounded, fmt__round, @crayon_str
using Tables
using StatsModels
using StatsModels: AbstractTerm, Term, FunctionTerm, ConstantTerm, FormulaTerm,
                   ContinuousTerm, coefnames
using Regress
using Regress: OLSMatrixEstimator, IVMatrixEstimator, ols, iv, TSLS, VcovSpec,
               WeakIVTestResult, FirstStageIV, lags, LagTerm,
               first_stage, weakivtest
using CovarianceMatrices
using Random: AbstractRNG, Xoshiro
import Random
using Statistics
using LinearAlgebra: I, Symmetric, cholesky, cond, dot, eigen, eigmin, eigvals,
                     factorize, Factorization, issuccess, kron, logdet, lu, tr
using Distributions
using RecipesBase
using ShiftedArrays: lag, lead
using StatsBase
# Extend (and re-export) the shared StatsAPI generics rather than defining our
# own functions: this keeps `vcov`/`stderror` the same binding as the one
# exported by StatsModels, CovarianceMatrices, and Regress, so unqualified
# calls work when those packages are loaded together with LocalProjections.
import StatsBase: vcov, stderror
using AxisArrays: AxisArrays, AxisArray, Axis
using MacroEconometricTools: LocalProjectionIRFResult, irfplot, irfplot!

include("utils.jl")
include("terms.jl")
include("types.jl")
include("state.jl")
include("lp.jl")
include("lpiv.jl")
include("inference.jl")
include("biascorrect.jl")
include("var.jl")
include("varbootstrap.jl")
include("fevd_estimators.jl")
include("fevd.jl")
include("fevd_bootstrap.jl")
include("fevd_summary.jl")
include("plotting.jl")
include("irfresult.jl")

end # module LocalProjections
