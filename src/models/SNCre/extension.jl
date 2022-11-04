function jlmsbc(ξ, struc::SNCre, data::PanelData)
    ϵ, σᵥ², dist_param = compositeError(ξ, struc, data)
    l = length(σᵥ²[1]) - size(ϵ[1], 1)
    _ϵ = Vector{Vector{Float64}}(undef, length(ϵ))
    for i = eachindex(ϵ)
        tar = ϵ[i]
        id_good = sum(isinf.(tar), dims=1)[1, :] .== 0
        _ϵ[i] = mean(tar[:, id_good], dims=2)[:, 1]
    end
    _ϵ = flatPanel(_ϵ)
   return base_jlmsbc(typeof(data.dist), flatTidyLag(σᵥ², dist_param...; l=l)..., _ϵ)
end

marginalEffect(ξ, struc::SNCre{T}, data) where{T<:AbstractSerialCorr} = baseMarginal(ξ, struc, data)