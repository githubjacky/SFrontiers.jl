function jlmsbc(ξ, struc::Cross, data::Data)
    ϵ, σᵥ², dist_param = compositeError(ξ, struc, data)
   return base_jlmsbc(typeof(data.dist), σᵥ², dist_param..., ϵ)
end

marginalEffect(ξ, struc::Cross, data; bootstrap=false) = baseMarginal(ξ, struc, data, bootstrap)