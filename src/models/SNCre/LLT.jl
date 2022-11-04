function compositeError(ξ,  struc::SNCre, data::PanelData)
    type, dist, σᵥ², depvar, frontiers = unpack(data)
    SCE, R, σₑ², xmean, ψ = unpack(struc, (:SCE, :R, :σₑ², :xmean, :ψ))
    β, dist_coeff, Wᵥ, δ, Wₑ, ρ  = slice(ξ, ψ, mle=true)  # the order matters 
    
    σₑ² = exp((σₑ² * Wₑ)[1])  # since it must be constant
    Random.seed!(1234)
    T = size.(depvar, 1)
    e = [repeat(rand(Normal(0, sqrt(σₑ²)), 1, R), inner=(i, 1)) for i in T]
    ϵ =  [(((depvar[i] - frontiers[i]*β) .- xmean[i]*δ) .- e[i]) for i = eachindex(T)]

    η_param = (ρ, typeof(dist), unpack(dist)..., dist_coeff)  # dist_arma is for the MA and the ARMA process to calculate the mean of eta

    return type() * Η(SCE, ϵ, η_param), exp.(σᵥ² * Wᵥ), dist(dist_coeff)
end


function Η(SCE::AR, ϵ, η_param)
    l = SCE()
    ρ = η_param[1]
    l == 1 || (ρ = coeffs(fromroots(ρ./(abs.(ρ).+1)))[1:end-1])
    beg = 1+l
    return [i[beg:end, :] - sum([i[beg-j:end-j, :] * ρ[j] for j = eachindex(ρ)]) for i in ϵ]
end


function Η(SCE::MA, ϵ, η_param)
    l = SCE()
    θ = η_param[1]
    Εη = mean(uncondU(η_param[2:end]...))
    η = Vector(undef, length(ϵ))
    for i = eachindex(ϵ)
        tar = ϵ[i]
        T, R = size(tar)
        ηᵢ = Matrix(undef, T+l, R)
        ηᵢ[1:l, :] .= Εη[i]
        for j = l+1:T+l
            ηᵢ[j, :] = tar[j-l, :] - sum([ηᵢ[j-k, :]*θ[k]  for k = eachindex(θ)])
        end
        η[i] = ηᵢ[2*l+1:end, :]
    end

    return η
end


function Η(SCE::ARMA, ϵ, η_param)
    p, q = unpack(SCE)
    ρ, θ = η_param[1][1:p], η_param[1][p+1:end]
    p == 1 || (ρ = coeffs(fromroots(ρ./(abs.(ρ).+1)))[1:end-1])
    Εη = mean(uncondU(η_param[2:3]...))

    η = Vector(undef, length(ϵ))
    beg = 1 + p
    for i = eachindex(ϵ)
        tar = ϵ[i]
        sumη =  tar[beg:end, :] - sum([tar[beg-j:end-j, :] * ρ[j] for j = eachindex(ρ)])

        T, R = size(tar)
        ηᵢ = Matrix(undef, T-p+q, R)
        ηᵢ[1:q, :] .= Εη[i]
        for j = q+1:T-p+q
            ηᵢ[j, :] = sumη[j-q, :] - sum([ηᵢ[j-k, :]*θ[k] for k = eachindex(θ)])
        end
        η[i] = ηᵢ[2*q+1:end, :]
    end

    return η
end


function LLT(ξ, struc::SNCre, data::PanelData)
    η, σᵥ², dist_param = compositeError(ξ, struc, data)

    lnf_vec = Vector{Any}(undef, length(η))
    lag = length(σᵥ²[1]) - size(η[1], 1)
    dist_type = typeof(data.dist)
    for i = eachindex(η)
        tar = η[i]
        dist_paramᵢ, σᵥ²ᵢ = [k[i][begin+lag:end] for k in dist_param], σᵥ²[i][begin+lag:end]
        simulatd_likelihood = [Base.prod(Like(dist_type, σᵥ²ᵢ, dist_paramᵢ..., tar[:, j])) for j = axes(tar, 2)]
        lnfᵢ = log(mean(simulatd_likelihood))
        lnf_vec[i] = isinf(lnfᵢ) ? -1.0e+10 : lnfᵢ
    end

    return -sum(lnf_vec)
end