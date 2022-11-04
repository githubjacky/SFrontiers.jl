#=  model specific data type  =#
abstract type AbstractSerialCorr end
struct AR <: AbstractSerialCorr
    p::Int64
end

struct MA <: AbstractSerialCorr
    q::Int64
end

struct ARMA <: AbstractSerialCorr
    p::Int64
    q::Int64
end

(s::AR)() = s.p
(s::MA)() = s.q
(s::ARMA)() = s.p + s.q

struct SNCreData <: AbstractModelData end

struct SNCre{T<:AbstractSerialCorr} <: PanelModel
    SCE::T  # SCE stands for serially correlated error term
    R::Int64
    σₑ²::U where{U<:Real}
    xmean::Vector{Matrix{S}} where{S<:Real}
    ψ::Vector{Any}
    varmat::Matrix{Any}
    data::SNCreData
end
#=  end of model specific data type  =#


#=  required functions  =#
include("./LLT.jl")
include("./sfspec.jl")
include("extension.jl")
