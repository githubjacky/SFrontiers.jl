# dependencies
using CSV, DataFrames, BenchmarkTools, Plots, Revise
include("../src/SFrontiers.jl")
using .SFrontiers


#=  read data and create the constant column  =#
df = DataFrame(CSV.File("test/data/CrossData.csv"))
df._cons = ones(nrow(df))


#=  model estimation  =#
res = sfmodel_fit(
    model=sfspec(
        Cross, useData(df), type=Prod(),
        dist=trun(μ=(:age, :school, :yr, :_cons), σᵤ²=(:age, :school, :yr, :_cons)),
        σᵥ²=:_cons,
        depvar=:yvar,
        frontiers=(:Lland, :PIland, :Llabor, :Lbull, :Lcost, :yr, :_cons),
    ),
    options=sfopt(
        warmstart_solver=NelderMead(),
        warmstart_maxIT=400,
        main_solver=Newton(),
        main_maxIT=2000,
        tolerance=1e-8
    ),
    startpt=sfinit(log_σᵤ²=(-0.1, -0.1, -0.1, -0.1), log_σᵥ²=-0.1)
)


# efficiency and inefficiency index
h1 = histogram(res.jlms, xlabel="JLMS", bins=100, label=false)
h2 = histogram(res.bc, xlabel="BC", bins=50, label=false)
h1h2= plot(h1, h2, layout = (1,2), legend=false)


# marginal effect
marginal, marginal_mean = sfmarginal(res)
plot(
    df[:,:age],
    marginal[:, :marg_age],
    seriestype=:scatter,
    xlabel="age", 
    ylabel="marginal effect of age in E(u)",
    label=false
)
hline!([0.00], label = false)


# bootstrap marginal effect
err_ci, bsdata = sfmarginal(
    res,
    bootstrap=true,
    R=100,
    seed=1232,
    iter=100,
    getBootData=true
)