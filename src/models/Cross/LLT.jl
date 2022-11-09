function LLT(ξ, struc::Cross, data::Data)
    ϵ, σᵥ², dist_param = compositeError(ξ, struc, data)

    return -sum(LogLike(typeof(data.dist), σᵥ², dist_param..., ϵ))
end


function compositeError(ξ, struc::Cross, data::Data)
    type, dist, σᵥ², depvar, frontiers = unpack(data)
    β, dist_coeff, Wᵥ = slice(ξ, struc.ψ, mle=true)
    return (type() * (depvar - frontiers*β))[:, 1], exp.(σᵥ² * Wᵥ), dist(dist_coeff)
end
