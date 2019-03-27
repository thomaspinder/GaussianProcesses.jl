module GaussianProcesses

using Optim, PDMats, ElasticPDMats, ElasticArrays, Distances, FastGaussQuadrature, RecipesBase, Distributions
using StaticArrays
using StatsFuns, SpecialFunctions

using LinearAlgebra, Printf, Random, Statistics
import Statistics: mean, cov

# Functions that should be available to package
# users should be explicitly exported here
export GPBase, GP, GPE, GPV, GPMC, ElasticGPE, predict_f, predict_y, Kernel, Likelihood, CompositeKernel, SumKernel, ProdKernel, Masked, FixedKernel, fix, Noise, Const, SE, SEIso, SEArd, Periodic, Poly, RQ, RQIso, RQArd, Lin, LinIso, LinArd, Matern, Mat12Iso, Mat12Ard, Mat32Iso, Mat32Ard, Mat52Iso, Mat52Ard, #kernel functions
    MeanZero, MeanConst, MeanLin, MeanPoly, SumMean, ProdMean, MeanPeriodic, #mean functions
    GaussLik, BernLik, ExpLik, StuTLik, PoisLik, BinLik,       #likelihood functions
    mcmc, optimize!,                                           #inference functions
    set_priors!,set_params!, update_target!, autodiff
using ForwardDiff: GradientConfig, Dual, partials, copyto!, Chunk
import ForwardDiff: seed!


const φ = normpdf
const Φ = normcdf
const invΦ = norminvcdf

# all package code should be included here
include("means/means.jl")
include("kernels/kernels.jl")
include("likelihoods/likelihoods.jl")
include("common.jl")
include("utils.jl")
include("chol_utils.jl")
include("GP.jl")
include("GPE.jl")
include("GPEelastic.jl")
include("GPMC.jl")
include("GPV.jl")
include("mcmc.jl")
include("optimize.jl")
include("crossvalidation.jl")
include("plot.jl")

# ScikitLearnBase, which is a skeleton package.
include("ScikitLearn.jl")

end # module
