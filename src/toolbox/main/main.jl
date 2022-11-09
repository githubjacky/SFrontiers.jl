"""
- the template for getvar() function customized for both matrix and frame data
- to Construct the either "Data" or "Panel Data" type
- also the varkeys and varvals for the output optimization table
"""
function baseGetVar(::Tuple{}, type, dist, σᵥ², depvar, frontiers)
    σᵥ², depvar, frontiers = MatrixData.((σᵥ², depvar, frontiers))  # ensure matirx data

    varkeys = baseVarKeys(fieldnames(typeof(dist)))  # generate the var names for making table
    varvals = baseVarVals(frontiers, unpack(dist), σᵥ²)  # generate the var names for making table

    return Data(type, dist, σᵥ², depvar, frontiers, size(depvar, 1)), varkeys, varvals
end

function baseGetVar(df::Tuple{DataFrame}, type, dist, σᵥ², depvar, frontiers)
    df = df[1]
    _σᵥ², _depvar, _frontiers = FrametoMatrix.((σᵥ², depvar, frontiers), df=df)
    dist_type, dist_props = dist[1], dist[2:end]
    dist = dist_type(FrametoMatrix.(dist_props, df=df)...)

    varkeys = baseVarKeys(fieldnames(dist_type))  # generate the var names for making table
    varvals = baseVarVals(frontiers, dist_props, σᵥ²)  # generate the var names for making table
    
    return Data(type, dist, _σᵥ², _depvar, _frontiers, size(_depvar, 1)), varkeys, varvals
end

function baseGetVar(::Tuple{}, ivar, type, dist, σᵥ², depvar, frontiers)
    id = [findall(x->x==i, ivar) for i in unique(ivar)]

    σᵥ², depvar, frontiers = tidyPanel.(MatrixData.((σᵥ², depvar, frontiers)), id=id)  # ensure matirx and panel data
    dist = typeof(dist)(tidyPanel.(unpack(dist), id=id)...)

    varkeys = baseVarKeys(fieldnames(typeof(dist)))  # generate the var names for making table
    varvals = baseVarVals(frontiers, unpack(dist), σᵥ²)  # generate var names for making table

    return PanelData(type, dist, σᵥ², depvar, frontiers, sum(size.(_depvar, 1))), varkeys, varvals
end

function baseGetVar(df::Tuple{DataFrame}, ivar, type, dist, σᵥ², depvar, frontiers)
    df = df[1]
    ivar = Base.getindex(df, :, ivar)
    id = [findall(x->x==i, ivar) for i in unique(ivar)]

    _σᵥ², _depvar, _frontiers = tidyPanel.(FrametoMatrix.((σᵥ², depvar, frontiers), df=df), id=id)
    dist_type, dist_props = dist[1], dist[2:end]
    dist = dist_type(tidyPanel.(FrametoMatrix.(dist_props, df=df), id=id)...)
    
    varkeys = baseVarKeys(fieldnames(dist_type))  # generate the var names for making table
    varvals = baseVarVals(frontiers, dist_props, σᵥ²)  # generate the var names for making table

    return PanelData(type, dist, _σᵥ², _depvar, _frontiers, sum(size.(_depvar, 1))), varkeys, varvals
end


"""
- can be used for finishing structure after basic module is generated
- ex: varkeys, varvals, ψ
"""
function completeInfo(base, add)
    beg = findfirst(i->!isassigned(base, i), 1:length(base))
    length(add) == 0 && (return base[1:beg-1])
    en = beg + length(add) - 1
    base[beg:en] = add
    return base[1:en]
end


"""
- merge the varkeys and varvals
"""
function varMatrix(varkeys, varvals)
    varmat = Matrix{Any}(undef, sum(length.(varvals)), 2)
    beg = 1
    for i = eachindex(varkeys)
        tar = varvals[i]
        len = length(tar)
        en = beg + len - 1
        keys = repeat([Symbol()], len)
        keys[1] = varkeys[i]
        varmat[beg:en, 1] .= keys
        varmat[beg:en, 2] .= tar
        beg = en + 1
    end
    return varmat
end


"""
- template for ψ
- used to recored the length of parameters of MLE
"""
function Ψ(data::AbstractData)
    dist, σᵥ², frontiers = unpack(data, (:dist, :σᵥ², :frontiers))
    ψ = Vector(undef, 30)  # it's just for the impossible case
    ψ[1:3] .= varNum(frontiers), sum(varNum.(unpack(dist))), varNum(σᵥ²)
    return ψ
end


"""
- to print some customized model specific information(before optimization process)
"""
function main_modelInfo(modelinfo1, modelinfo2)
    printstyled("\n * Model specification\n\n", color=:yellow)
    println("    $(modelinfo1)\n")  # name of model
    println("$(modelinfo2)")  # some customized information
end


"""
- general function to slice the array given the length of each segament
- ψ is the length of each part of ξ
- if it's for MLE then the ψ[end] is sum(ψ[end-1])
"""
function slice(ξ::Vector, ψ::Vector; mle=false)  
    p = mle ? Vector(undef, length(ψ)-1) : Vector(undef, size(ψ, 1))
    beg = 1
    for i = eachindex(p)
        en = beg + ψ[i] - 1
        p[i] = ξ[beg:en]
        beg = en + 1
    end
    return p
end


"""
- the close form of the likelihood function
- include half normal, truncated normal, exponential
"""
function Like(::Type{Half}, σᵥ²::Vector, σᵤ²::Vector, ϵ::Vector)
    σ²  = σᵤ² + σᵥ²
    σₛ² = (σᵥ² .* σᵤ²) ./ σ²
    μₛ  = (-σᵤ² .* ϵ) ./ σ²

    return map(halfpdf, sqrt(σ²), σ², ϵ, μₛ, σₛ²)
end

function Like(::Type{Trun}, σᵥ²::Vector, μ::Vector, σᵤ²::Vector, ϵ::Vector)
    σ²  = σᵤ² + σᵥ²
    σₛ² = (σᵥ² .* σᵤ²) ./ σ²
    σᵤ = sqrt.(σᵤ²)
    μₛ  = (σᵥ² .* μ - σᵤ² .* ϵ) ./ σ²

    return map(trunpdf, σ², μ, ϵ, σₛ², σᵤ, μₛ)
end

function Like(::Type{Expo}, σᵥ²::Vector, λ²::Vector, ϵ::Vector)
    λ = sqrt.(λ²)
    σᵥ = sqrt.(σᵥ²)

    return map(expopdf, λ, ϵ, σᵥ, σᵥ², λ²)
end


"""
- the close form of the log-likelihood function
- include half normal, truncated normal, exponential
"""
function LogLike(::Type{Half}, σᵥ²::Vector, σᵤ²::Vector, ϵ::Vector)
    σ²  = σᵤ² + σᵥ²
    σₛ² = (σᵥ² .* σᵤ²) ./ σ²
    μₛ  = (-σᵤ² .* ϵ) ./ σ²

    return map(halflogpdf, σ², ϵ, μₛ, σₛ²)
end

function LogLike(::Type{Trun}, σᵥ²::Vector, μ::Vector, σᵤ²::Vector, ϵ::Vector)
    σ²  = σᵤ² + σᵥ²
    σₛ² = (σᵥ² .* σᵤ²) ./ σ²
    σᵤ = sqrt.(σᵤ²)
    μₛ  = (σᵥ² .* μ - σᵤ² .* ϵ) ./ σ²

    return map(trunlogpdf, σ², μ, ϵ, σₛ², σᵤ, μₛ)
end

function LogLike(::Type{Expo}, σᵥ²::Vector, λ²::Vector, ϵ::Vector)
    λ = sqrt.(λ²)
    σᵥ = sqrt.(σᵥ²)

    return map(expologpdf, λ, ϵ, σᵥ, σᵥ², λ²)
end


"""
- the close form of the jlms, bc index
- distribution assumption include half normal, normal, exponential
"""
function base_jlmsbc(::Type{Half}, σᵥ², σᵤ², ϵ)
    μ = 0.
    σ²  = σᵤ² + σᵥ² 
    μₛ  = @. (σᵥ² * μ - σᵤ² * ϵ) / σ²
    σₛ  = @. sqrt((σᵥ² * σᵤ²) / σ²)

    jlms = @. (σₛ * normpdf(μₛ / σₛ )) / normcdf(μₛ / σₛ) + μₛ
    bc = @. exp( -μₛ + 0.5 * (σₛ)^2  ) * ( normcdf( (μₛ / σₛ) - σₛ ) / normcdf(μₛ / σₛ)  )
    return jlms, bc
end

function base_jlmsbc(::Type{Trun}, σᵥ², μ, σᵤ², ϵ)
    σ²  = σᵤ² + σᵥ²
    μₛ  = @. (σᵥ² * μ - σᵤ² * ϵ) / σ²
    σₛ  = @. sqrt((σᵥ² * σᵤ²) / σ²)

    jlms = @. (σₛ * normpdf(μₛ / σₛ )) / normcdf(μₛ / σₛ) + μₛ
    bc = @. exp( -μₛ + 0.5 * (σₛ)^2  ) * ( normcdf( (μₛ / σₛ) - σₛ ) / normcdf(μₛ / σₛ)  )
    return jlms, bc
end

function base_jlmsbc(::Type{Expo}, σᵥ², λ, ϵ)
    σᵥ = sqrt(σᵥ²)
    μₛ  = (-ϵ) - (σᵥ² ./ λ) # don't know why this line cannot use @.

    jlms = @. (σᵥ * normpdf(μₛ / σᵥ )) / normcdf(μₛ / σᵥ) + μₛ
    bc = @. exp( -μₛ + 0.5 * σᵥ²  ) * ( normcdf( (μₛ / σᵥ) - σᵥ ) / normcdf(μₛ / σᵥ)  )
    return jlms, bc
end


"""
- base module to calculate the marginal effect of unconditional mean E(u)
- the only diversity to calculate the marginal effect is the unconditional mean
- there are two modes: one is standard the other is bootstrap(default to standard)
"""
function baseMarginal(ξ, struc, data, bootstrap)
    # if the data is panel data flat it, otherwise ignore it
    dist_data = map(flatPanel, unpack(data.dist))
    var_nums = [varNum(i) for i in dist_data]
    var_num = sum(var_nums)
    dist_data = hcat(dist_data...)
    dist_coef = slice(ξ, struc.ψ, mle=true)[2]

    mm = Matrix{Float64}(undef, obsNum(dist_data), var_num)
    for i = axes(mm, 1)
        mm[i, :] = ForwardDiff.gradient(
            marg -> uncondU(
                typeof(data.dist),
                [reshape(j, 1, length(j)) for j in slice(marg, var_nums)]...,
                dist_coef
            ),
            dist_data[i, begin:end]
        )
    end
    beg_label = varNum(data.frontiers) + 1
    en_label = beg_label + sum(var_nums) - 1
    label = struc.varmat[beg_label:en_label, 2]  # use the varmat to get the column name of datafrae
    mm, label = cleanMarginalEffect(mm, label)  # drop the duplicated and constant columns
    mean_marginal = mean(mm, dims=1)
    if bootstrap
        return mm, mean_marginal
    else
        label = [Symbol(:marg_, i) for i in label]
        return DataFrame(mm, label), NamedTuple{Tuple(label)}(mean_marginal)
    end
end  