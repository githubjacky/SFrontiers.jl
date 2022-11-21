"""
- marginal effect
"""
function sfmarginal(result::NamedTuple; bootstrap=false, kwargs...)
    if !bootstrap
        marginalEffect(result.ξ, result.struc, result.data, bootstrap=bootstrap)
    else
        bootStrapMarginal(result; kwargs...)
    end
end

function bootStrapMarginal(
    result::NamedTuple;
    mymisc=nothing,
    R::Int=500, 
    level=0.05,
    iter::Int=-1,
    getBootData=false,
    seed::Int=-1,
    every=10,
    verbose=true
)
    # check some requirements of data
    ((level > 0.0) && (level < 1.0)) || throw("The significance level (`level`) should be between 0 and 1.")

    level > 0.5 && (level = 1-level)  # 0.95 -> 0.05

    # In the following lines, the integer part had been taken care of in Type.
    (seed == -1) || ( seed > 0) || throw("`seed` needs to be a positive integer.")
    (iter == -1) || ( iter > 0) || throw("`iter` needs to be a positive integer.")
    (R > 0) || throw("`R` needs to be a positive integer.")
    iter > 0 && (result.options[:main_maxit] = iter)

    _, obs_marg_mean = marginalEffect(result.ξ, result.struc, result.data)
    sim_res = Matrix{Real}(undef, R, length(obs_marg_mean))
    rng = seed != -1 ? MersenneTwister(seed) : -1
    result.options[:warmstart_solver] = ()

    printstyled(" * bootstrap in progress...\n\n", color=:yellow)
    # bootstrap sampling
    for i in 1:R
        @label start1
        if fieldcount(typeof(result.struc.data)) == 0
            bootstrap_data = resampling(result.data; rng=rng)
            bootstrap_struc = result.struc
        else
            bootstrap_data, bootstrap_struc = resampling(result.data, result.struc; rng=rng)
        end
        
        Hessian, ξ, warmup_opt, main_opt = MLE(bootstrap_struc, bootstrap_data, result.options, result.ξ)
        if Optim.iteration_limit_reached(main_opt) || 
           isnan(Optim.g_residual(main_opt)) ||  
           Optim.g_residual(main_opt) > 1e-1
               @goto start1
        end
        numerical_hessian = hessian!(Hessian, ξ)
        var_cov_matrix = try
            inv(numerical_hessian)
        catch
            @goto start1
        end
        !all(diag(var_cov_matrix) .> 0) && (@goto start1)

        _, marginal_mean = marginalEffect(ξ, bootstrap_struc, bootstrap_data, bootstrap=true)
        # in this run some of the element is NaN
        sum(isnan.(marginal_mean)) == 0 ? (sim_res[i, :] = marginal_mean) : (@goto start1)
        
        verbose && i % every == 0 && print("$(i)..")
    end  # for i=1:R
    printstyled("\n * Done!\n\n", color=:yellow)

    theMean = collect(values(obs_marg_mean))
    theSTD = sqrt.(sum((sim_res .- theMean').^2, dims=1) ./(R-1))
    theSTD = reshape(theSTD, size(theSTD, 2))
    ci_mat = sfCI(bootdata=sim_res, observed=theMean, level=level, verbose=verbose)

    if verbose
        table_content = hcat(collect(keys(obs_marg_mean)), theMean, theSTD, ci_mat)
        table = [" " "mean of the marginal" "std.err. of the"  "bias-corrected"; 
                 " " "effect on E(u)"       "mean effect"      "$(100*(1-level))%  conf. int.";
                 table_content]
        pretty_table(
            table,
            noheader=true,
            body_hlines = [2],
            formatters = ft_printf("%0.5f", 2:4),
            compact_printing = true,
            backend = Val(result.options[:table_format])
        )
        println()
    end
    getBootData ? (return hcat(theSTD, ci_mat), sim_res) : (return hcat(theSTD, ci_mat))
end


function resampling(data::Data, struc=nothing; rng)
    if rng != -1
        selected_row = sample(rng, 1:data.nofobs, data.nofobs; replace=true)
    else
        selected_row = sample(1:data.nofobs, data.nofobs; replace=true)
    end
    bootstrap_data = Data(
        data.type,
        typeof(data.dist)([i[selected_row, :] for i in unpack(data.dist)]...),
        data.σᵥ²[selected_row, :],
        data.depvar[selected_row, :],
        data.frontiers[selected_row, :],
        data.nofobs
    )

    if struc !== nothing
        struc_field = unpack(struc)
        bootstrap_struc = typeof(struc)(
            struc_field[:end-1]...,
            typeof(struc_field[end])(struc_field[end][selected_row, :])
        )
        return bootstrap_data, bootstrap_struc
    else
        return bootstrap_data
    end
end


function resampling(seed, data::PanelData, struc=nothing)
end


function sfCI(
    ;bootdata::Any=nothing,
    observed::Union{Vector, Real, Tuple, NamedTuple}=nothing,
    level=0.05,
    verbose=false
    )
    # bias-corrected (but not accelerated) confidence interval 
    # For the "accelerated" factor, need to estimate the SF model 
    #    for every jack-knifed sample, which is expensive.

    isa(observed ,NamedTuple) && (observed = values(observed))
    ((level > 0.0) && (level < 1.0)) || throw("The significance level (`level`) should be between 0 and 1.")
    level > 0.50 && (level = 1-level)  # 0.95 -> 0.05

    nofobs, nofK = size(bootdata)  # number of statistics
    (nofK == length(observed)) || throw("The number of statistics (`observed`) does not fit the number of columns of bootstrapped data.")
    
    ci = Array{Any}(undef, nofK, 1)
    z1 = quantile(Normal(), level/2)
    z2 = quantile(Normal(), 1 - level/2)  #! why z1 != z2?

    for i in 1:nofK
        @views data = bootdata[:,i]
        count = sum(data .< observed[i])
        z0 = quantile(Normal(), count/nofobs) # bias corrected factor
        alpha1 = cdf(Normal(), z0 + ((z0 + z1) ))
        alpha2 = cdf(Normal(), z0 + ((z0 + z2) ))
        order_data = sort(data)
        beg, en = order_data[Int(ceil(nofobs*alpha1))], order_data[Int(ceil(nofobs*alpha2))]
        ci[i,1] = (round(beg, digits=5), round(en, digits=5))
    end
    verbose && println("\nBias-Corrected $(100*(1-level))% Confidence Interval:\n")

    return ci
end