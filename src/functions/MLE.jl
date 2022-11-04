function MLE(struc::SFmodel, data, opt, startpt)
    warmstart = isa(opt[:warmstart_solver], Tuple{}) ? false : true
    _Hessian = TwiceDifferentiable(
        ξ -> LLT(ξ, struc, data),
        ones(length(startpt));
        autodiff=:forward
    )
    modelInfo(struc)

    warm_opt = ()
    if warmstart
        printstyled(" * optimization - warm start\n\n", color=:yellow)
        warm_opt = optimize(
            _Hessian,
            startpt,
            opt[:warmstart_solver],
            Options(
                g_tol=opt[:tolerance],
                iterations=opt[:warmstart_maxIT],
                store_trace=opt[:verbose],
                show_trace=opt[:verbose]
            )
        )
        startpt = minimizer(warm_opt)
        println(" Number of total iterations: $(iterations(warm_opt))\n")
        pretty_table(hcat(struc.varmat, startpt); header=["", "Var.", "Coef."])
    end

    printstyled("\n * optimization - main process\n\n", color=:yellow)
    main_opt = optimize(
        _Hessian,
        startpt,
        opt[:main_solver],
        Options(
            g_tol=opt[:tolerance],
            iterations=opt[:main_maxIT],
            store_trace=opt[:verbose],
            show_trace=opt[:verbose]
        )
    )
    _coevec = minimizer(main_opt)

    diagonal = post_estimation(_Hessian, _coevec, data, struc)
    modelEstimation(data.nofobs, warm_opt, main_opt)  # show the simulation results
    outputTable(struc.varmat, _coevec, diagonal, data.nofobs)  # output table
    
    return _coevec
end


"""
post estimation process
1. check the more rigorous converge criterion
2. calculate the p-value
"""
function post_estimation(_Hessinan, _coevec, data, struc)
    numerical_hessian = hessian!(_Hessinan, _coevec)
    var_cov_matrix = try  # check if the matrix is invertible
        inv(numerical_hessian)
    catch
        _ = isMultiCollinearity.([:frontiers, fieldnames(typeof(data.dist))..., :σᵥ²], 
                                 [data.frontiers, unpack(data.dist)..., data.σᵥ²])
        fieldcount(typeof(struc.data)) != 0 && (_ = isMultiCollinearity.(fieldnames(typeof(struc.data)), unpack(struc.data)))
        throw("The Hessian matrix is not invertible, indicating the model does not converge properly. The estimation is abort.")
    end
    diagonal = diag(var_cov_matrix)
    if !all( diagonal .> 0 )
        _ = isMultiCollinearity.([:frontiers, fieldnames(typeof(data.dist))..., :σᵥ²],
                                [data.frontiers, unpack(data.dist)..., data.σᵥ²])
         fieldcount(typeof(struc.data)) != 0 && (_ = isMultiCollinearity.(fieldnames(typeof(struc.data)), unpack(struc.data)))
                                                throw("Some of the diagonal elements of the var-cov matrix are non-positive, indicating problems in the convergence. The estimatIon is abort.")
    end
    return diagonal
end


function modelEstimation(nofobs, warm_opt, main_opt)
    converge = Optim.converged(main_opt) ? "converge successfully" : "not converge yet"

    # get the total iteratons
    iter = isa(warm_opt, Tuple{}) ? iterations(main_opt) : iterations(warm_opt) + iterations(main_opt)

    printstyled("*********************************\n "; color=:cyan)
    printstyled("      Estimation Results:\n"; color=:cyan); 
    printstyled("*********************************\n\n"; color=:cyan)
    println(" Number of total iterations: $(iter)")
    println(" Time Consuming:             $(time_run(main_opt))")
    println("")
    println(" Converge status:            $converge")
    println(" Reach Max iterations:       $(iteration_limit_reached(main_opt))")
    println(" Numberf of observations:    $(nofobs)")
    println(" Log-likelihood value:       $(round(-1*minimum(main_opt); digits=5))")
    println("")
end


function outputTable(varmat, _coevec, diagonal, nofobs)
    stddev = sqrt.(diagonal)
    t_stats = _coevec ./ stddev
    p_value = [pvalue(TDist(nofobs - length(_coevec)), i; tail=:both) for i in t_stats]
    tt = cquantile(Normal(0,1), 0.025)
    ci_low = [_coevec[i] - tt*stddev[i] for i = eachindex(_coevec)]
    ci_upp = [_coevec[i] + tt*stddev[i] for i = eachindex(_coevec)]
    pretty_table(
        hcat(varmat, _coevec, stddev, t_stats, p_value, ci_low, ci_upp);
        header=["", "Var.", "Coef.", "Std. Err.", "z", "P>|z|", "95%CI_l", "95%CI_u"]
    )
end