function sfspec(::Type{Cross}, df...; type, dist, σᵥ², depvar, frontiers)
    #=   get the data and ensure its type is matrix  =#
    crossdata, varkeys, varvals = baseGetVar(df, type, dist, σᵥ², depvar, frontiers)
    #=  ==========================================  =#
    
    #= construct remain varkeys  =#
    varkeys = completeInfo(varkeys, ())
    #=  =======================  =#

    #=  construct remain varvals  =#
    varvals = completeInfo(varvals, ())
    #=  ========================= =#

    #=  costruct the varMatrix  =#
    varmat = varMatrix(varkeys, varvals)
    #=  ======================  =#

    #=  generate the remain rule for slicing parameter(for details: utils/toolBox.jl->SliceParam())  =#
    ψ = completeInfo(Ψ(crossdata), ())  # generate the length of serially correlated error terms, σₑ² and correlated random effects 
    push!(ψ, sum(ψ))
    #=  ===========================================================================================  =#

    return Cross(ψ, varmat, CrossData()), crossdata
end


function modelInfo(Cross)
    modelinfo1 = "Base stochastic frontier model"
    
    modelinfo2 =
"""
    Yᵢₜ = Xᵢₜ*β + + ϵᵢₜ
        where ϵᵢₜ = vᵢₜ - uᵢₜ

         further,     
               vᵢₜ ∼ N(0, σᵥ²),
                 σᵥ²  = exp(log_σᵥ)
               uᵢₜ ∼ N⁺(0, σᵤ²),
                 σᵤ² = exp(log_σᵤ)

    In the case of type(cost), "- uᵢₜ" above should be changed to "+ uᵢₜ"
"""
    main_modelInfo(modelinfo1, modelinfo2)
end