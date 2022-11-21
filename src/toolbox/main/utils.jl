function baseVarKeys(dist_name)
    varkeys = Vector(undef, 20)
    distfn = [dist_name...]
    distfn[end] == :σᵤ² ?  (distfn[end] = :log_σᵤ²) : (distfn[end] = :log_η²)
    en = length(distfn) + 2
    varkeys[1:en] = [:fontiers, distfn..., :log_σᵥ²]
    return varkeys
end


function baseVarVals(frontiers::Tuple{Vararg{Symbol}}, dist_props, σᵥ²)
    varvals = Vector(undef, 20)
    en = length(dist_props) + 2
    varvals[1:en] = [frontiers, dist_props..., σᵥ²]
    return varvals
end


function baseVarVals(frontiers::Matrix{T}, dist_props, σᵥ²) where{T<:Number}
    varvals = Vector(undef, 20)
    en = length(dist_props) + 2
    varvals[1] = [Symbol(:frontiers, i) for i = axes(frontiers, 2)]
    for i = eachindex(dist_props)
        if size(dist_props[i], 2) == 1
            varvals[1+i] = (:_cons,)
        else
            varvals[1+i] = [Symbol(:exogenous, j) for j = axes(dist_props[i], 2)]
            varvals[1+i][end] = :_cons
        end
    end
    if size(σᵥ², 2) == 1
        varvals[en] = :_cons
    else
        varvals[en] = [Symbol(:exogenous, i) for i = axes(σᵥ², 2)]
        varvals[en][end] = :_cons
    end

    return varvals
end


function halfpdf(σ, σ², ϵ, μₛ, σₛ²)
    return (2 / σ) *
           normpdf(ϵ / sqrt(σ²)) *
           normcdf(μₛ / sqrt(σₛ²))
end

function trunpdf(σ, σ², μ, ϵ, σₛ², σᵤ, μₛ)
    return (1/σ) *
           normpdf((μ+ϵ) / sqrt(σ²)) *
           normcdf(μₛ / sqrt(σₛ²)) / 
           normcdf(μ / σᵤ)
end

function expopdf(λ, ϵ, σᵥ, σᵥ², λ²)
    return (1/λ) *
           normcdf(-(ϵ/σᵥ) - (σᵥ/λ)) *
           (ϵ/λ) *
           (σᵥ² / (2*λ²))
end

function halflogpdf(σ², ϵ, μₛ, σₛ²) 
    return (-0.5 * log(σ²)) + 
           normlogpdf(ϵ / sqrt(σ²)) + 
           normlogcdf(μₛ / sqrt(σₛ²)) -
           normlogcdf(0)
end

function trunlogpdf(σ², μ, ϵ, σₛ², σᵤ, μₛ)
    return (-0.5 * log(σ²)) +
           normlogpdf((μ+ϵ) / sqrt(σ²)) +
           normlogcdf(μₛ / sqrt(σₛ²)) - 
           normlogcdf(μ / σᵤ)
end

function expologpdf(λ, ϵ, σᵥ, σᵥ², λ²)
    return -log(λ) +
           normlogcdf(-(ϵ/σᵥ) - (σᵥ/λ)) +
           (ϵ/λ) +
           (σᵥ² / (2*λ²))
end


function cleanMarginalEffect(m::Matrix, labels::Vector{Any})
    unique_label = unique(labels)
    pos = Dict([(i, findall(x->x==i, labels)) for i in unique_label])
    id = Dict([(i, pos[i][1]) for i in unique_label])
    count = Dict([(i, length(pos[i])) for i in unique_label])
    drop = []
    for (i, label) in enumerate(labels)
        # task1: drop the constant columns
        if length(unique(m[:, i])) == 1
            append!(drop, i)
            print(count[label])
            count[label] -= 1
            if i == id[label] && count != 0
                id[label] = pos[label][1+(length(pos[label])-count[label])]
            end
            continue
        end
        # task2: drop the columns with duplicated column names
        if i != id[label]
            tar = id[label]
            m[:, tar] = m[:, tar] + m[:, i]
            append!(drop, i)
            count[label] -= 1
        end
    end
    length(labels) == length(drop) && error("there is no marginal effect")

    return m[:, Not(drop)], unique_label
end