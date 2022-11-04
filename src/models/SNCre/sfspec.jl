function sfspec(::Type{SNCre}, df...; type, dist, σᵥ², ivar, depvar, frontiers, SCE::AbstractSerialCorr, R, σₑ²)
    #=   get the data and ensure its type is matrix  =#
    paneldata, varkeys, varvals = baseGetVar(df, ivar, type, dist, σᵥ², depvar, frontiers)
    σₑ² = isa(df, Tuple{}) ? σₑ²[1] : Base.getindex(df[1], :, σₑ²)[1]  # since σₑ² will always be constant
    xmean = mean.(paneldata.frontiers, dims=1)
    _xmean = hcat.(xmean, [ones(1) for _ = eachindex(xmean)])
    ideal_xmean, pivots = isMultiCollinearity(:_xmean, _xmean)  # still return the panelized data
    #=  ==========================================  =#
    
    #= construct remain varkeys  =#
    corrkey = isa(SCE, AR) ? (:ρ,) : (isa(SCE, MA) ? (:θ,) : (:ρ, :θ))
    varkeys = completeInfo(varkeys, [:αᵢ, :log_σₑ², corrkey...])
    #=  =======================  =#

    #=  construct remain varvals  =#
    if !isa(SCE, ARMA)
        var = corrkey[1]
        corrval = [[Symbol(var, i) for i = 1:SCE()]]
    else
        var1, var2 = corrkey[1], corrkey[2]
        corrval = [[Symbol(var1, i) for i = 1:SCE.p], 
                   [Symbol(var2, i) for i = 1:SCE.q]]
    end
    frontiers_mean = [Symbol(:mean_, i) for i in varvals[1]]
    push!(frontiers_mean, :_cons)
    frontiers_mean = frontiers_mean[pivots]

    varvals = completeInfo(varvals, [frontiers_mean, :_cons, corrval...])
    #=  ========================= =#

    #=  costruct the varMatrix  =#
    varmat = varMatrix(varkeys, varvals)
    #=  ======================  =#

    #=  generate the remain rule for slicing parameter(for details: utils/toolBox.jl->SliceParam())  =#
    ψ = completeInfo(Ψ(paneldata), [varNum(ideal_xmean), 1, SCE()])  # generate the length of serially correlated error terms, σₑ² and correlated random effects 
    push!(ψ, sum(ψ))
    #=  ===========================================================================================  =#

    return SNCre(SCE, R, σₑ², ideal_xmean, ψ, varmat, SNCreData()), paneldata
end


(t::AR)(::Symbol) = Symbol(typeof(t), t.p)
(t::MA)(::Symbol) = Symbol(typeof(t), t.q)
function (t::ARMA)(::Symbol)
    str = string(typeof(t))
    return Symbol(str[1:2], t.p, str[3:4], t.q)
end

function (t::AR)(::String)
    ϵᵢₜ = ""
    for i = 1:t.p
        ϵᵢₜ *= "ρ$i * ϵᵢₜ₋$i  + "
    end
    return ϵᵢₜ *= "ηᵢₜ"
end
function (t::MA)(::String)
    ϵᵢₜ = ""
    for i = 1:t.q
        ϵᵢₜ *= "θ$i * ηᵢₜ₋$i  + "
    end
    return ϵᵢₜ *= "ηᵢₜ"
end
function (t::ARMA)(::String)
    ϵᵢₜ = ""
    for i = 1:t.p
        ϵᵢₜ *= "ρ$i * ϵᵢₜ₋$i  + "
    end
    for i = 1:t.q
        ϵᵢₜ *= "θ$i * ηᵢₜ₋$i  + "
    end
    return ϵᵢₜ *= "ηᵢₜ"
end

function modelInfo(struc::SNCre)
    modelinfo1 = "flexible panel stochastic frontier model with serially correlated errors"
    
    SCE = struc.SCE
    modelinfo2 =
"""
    Yᵢₜ = αᵢ + Xᵢₜ*β + T*π + ϵᵢₜ
        where αᵢ = δ₀ + X̄ᵢ'* δ₁ + eᵢ,
        and since the serial correlated assumptions is $(SCE(:type)),
        ϵᵢₜ = $(SCE("info"))
               ηᵢₜ = vᵢₜ - uᵢₜ

         further,     
               vᵢₜ ∼ N(0, σᵥ²),
                 σᵥ²  = exp(log_σᵥ)
               uᵢₜ ∼ N⁺(0, σᵤ²),
                 σᵤ² = exp(log_σᵤ)
               eᵢ ∼ N(0, σₑ²)
                 σᵤ² = exp(log_σₑ)

    In the case of type(cost), "- uᵢₜ" above should be changed to "+ uᵢₜ"
"""
    main_modelInfo(modelinfo1, modelinfo2)
end