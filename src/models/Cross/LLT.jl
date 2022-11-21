function LLT(ξ, struc::Cross, data::Data)
    ϵ, σᵥ², dist_param = compositeError(ξ, struc, data)

    return -sum(LogLike(typeof(data.dist), σᵥ², dist_param..., ϵ))
end


function compositeError(ξ, struc::Cross, data::Data)
    β, dist_coeff, Wᵥ = slice(ξ, struc.ψ, mle=true)
    return (data.type() * (data.depvar - data.frontiers*β))[:, 1], exp.(data.σᵥ² * Wᵥ), data.dist(dist_coeff)
end
