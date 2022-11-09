"""
- marginal effect
"""
sfmarginal(result; bootstrap=false) = marginalEffect(result.ξ, result.struc, result.data, bootstrap)

function sfmarginal(result,
                    mymisc,
                    R=500,
                    level=0.05,
                    iter=-1,
                    getBootData=false,
                    seed=-1,
                    every=10)
    """
         check some requirements of arguments
    """
    ((level > 0.0) && (level < 1.0)) || throw("The significance level (`level`) should be between 0 and 1.")

    level > 0.5 && (level = 1-level)  # 0.95 -> 0.05

    # In the following lines, the integer part had been taken care of in Type.
    (seed == -1) || ( seed > 0) || throw("`seed` needs to be a positive integer.")
    (iter == -1) || ( iter > 0) || throw("`iter` needs to be a positive integer.")
    (R > 0) || throw("`R` needs to be a positive integer.")
    iter > 0 && (result.opt[:main_maxit] = iter)

    marginal, marginal_mean = marginalEffect(
        result.ξ, result.struc, result.data, true
    )
    sim_res = Array{Real}(undef, R, length(marginal_mean))
    seed > 0 && (rng = MersenneTwister(seed))

    """
        bootstrap sampling
    """
    for i in 1:R
        @label start1
        if isa(data, Data)
            if seed == -1
                select_row = sample(1:data.nofobs, data.nofobs; replace=true)  # require StatsBase.jl
            else
                select_row = sample(rng, 1:data.nofobs, data.nofobs; replace=true)  # require StatsBase.jl
            end
            data = Data(
                result.data.type,
                typeof(result.data.dist)(
                    [i[selected_row, :] for i in unpack(result.data)]...
                ),
                result.data.σᵥ²[selected_row, :],
                result.data.frontiers[selected_row, :],
                result.data.depvar[selected_row, :],
                result.data.nofobs
            )
            if fieldcount(typeof(result.struc.data)) != 0
                struc_field = unpack(result.struc)
                struc = typeof(struc)(
                    struc_field[:end-1]...,
                    typeof(struc_field[end])(
                        struc_field[end][selected_row, :]
                    )
                )
            else
                struc = result.struc
            end
        else  # panel data
            if seed == -1
                select_row = [
                    sample(1:i, i; replace=true)
                    for i in size.(result.data.depvar, 1)
                ]
            else
                select_row = [
                    sample(rng, 1:i, i; replace=true)
                    for i in size.(result.data.depvar, 1)
                ]
            end
        end

        ξ, resample = bootstrapMLE(struc, data, result.opt, result.startpt)
        resample && @goto start1

        margianl, marginal_mean = marginalEffect(ξ, struc, data)
        if sum(isnan.(marginal_mean)) != 0  # in this run some of the element is NaN
            @goto start1
        else
            sim_res[i, :] = marginal_mean
        end 

        if opt[:verbose]
            i == 1 && printstyled(" * bootstrap in progress...\n\n", color=:yellow)
            i % every == 0 && print("$(i)..")
            i == R && printstyled(" * Done!\n\n", color=:yellow)
        end
    end  # for i=1:R


end


function sfmodel_CI(; bootdata::Any=nothing, observed::Union{Vector, Real, Tuple, NamedTuple}=nothing, level::Real=0.05, verbose::Bool=true)
end