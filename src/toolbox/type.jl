"""
Stochastic Frontiers Model
"""
abstract type SFmodel end
abstract type PanelModel <: SFmodel end

"""
- production of cost type
"""
abstract type AbstractEconomicType end
struct Prod <: AbstractEconomicType end
(t::Prod)() = 1
# allow different notaion
prod() = Prod()
p() = Prod()

struct Cost <: AbstractEconomicType end
(t::Cost)() = -1
# allow different notaion
cost() = Cost()
c() = Cost()

"""
- to store the commonly used data
"""
abstract type AbstractData end
struct Data{T<:AbstractEconomicType, S<:AbstractDist, U<:Real, V<:Real, W<:Real} <: AbstractData
    type::T
    dist::S
    σᵥ²::Matrix{U}
    depvar::Matrix{V}
    frontiers::Matrix{W}
    nofobs::Int64
end

"""
- to store the commonly used data
- the type of data is panel data
"""
struct PanelData{T<:AbstractEconomicType, S<:AbstractDist, U<:Real, V<:Real, W<:Real} <: AbstractData
    type::T
    dist::S
    σᵥ²::Vector{Matrix{U}}
    depvar::Vector{Matrix{V}}
    frontiers::Vector{Matrix{W}}
    nofobs::Int64
end

"""
- to store the model specific data type which should be check the isMultiCollinearity
"""
abstract type AbstractModelData end