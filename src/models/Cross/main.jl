struct CrossData <: AbstractModelData end

struct Cross <: SFmodel
    ψ::Vector{Any}
    varmat::Matrix{Any}
    data::CrossData
end


#=  required functions  =#
include("./LLT.jl")
include("./sfspec.jl")
include("./extension.jl")