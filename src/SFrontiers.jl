module SFrontiers
#=  export cross models  =#

#=  export panel models  =#
export Cross, SNCre

#=  export functionality  of model construction =#
export useData, sfspec, sfopt, sfinit, sfmodel_fit, sfmarginal

#=  export functionality  of some tools =#
export sfpredict, jlmsbc, marginal

#=  export general distribution assumption and economic type  =#
export Half, half, h, Trun, trun, t, Expo, expo, e
export Prod, prod, p, Cost, cost, c

#= export model specific data type  =#
export AR, MA, ARMA  # SNCre

#=  export package-Optim's algorithms  =#
export NelderMead, SimulatedAnnealing, SAMIN, ParticleSwarm,
       ConjugateGradient, GradientDescent, BFGS, LBFGS,
       Newton, NewtonTrustRegion, IPNewton


#=  used packages reference  =#
using DataFrames, Distributions, Random, Statistics, Optim, LinearAlgebra, PrettyTables, StatsFuns, ForwardDiff
import Optim: TwiceDifferentiable, optimize, Options, minimizer, iterations, iteration_limit_reached,
              minimum, time_run
import DataStructures: OrderedDict
import NLSolversBase: hessian!
import StatsFuns: normpdf, normcdf
import Polynomials: fromroots, coeffs
import RowEchelon: rref_with_pivots
import HypothesisTests: pvalue
import ForwardDiff: gradient



#=  more source code  =#
include("toolbox/optional/utils.jl")
include("toolbox/optional/main.jl")
include("toolbox/distributions.jl")
include("toolbox/type.jl")
include("toolbox/main/utils.jl")
include("toolbox/main/main.jl")
include("models/SNCre/main.jl")
include("models/Cross/main.jl")
include("functions/MLE.jl")
include("functions/main.jl")


end  # end of module --SFrontiers
