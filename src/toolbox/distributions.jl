abstract type AbstractDist end


struct Half <: AbstractDist
    σᵤ²::T where{T<:AbstractArray}
end
# variaous construction of half normal(if it's frame data just return)
# ensure matrix data
HF(σᵤ²::Union{Symbol, Tuple{Vararg{Symbol}}}) = (Half, σᵤ²)  # frame data
HF(σᵤ²::Vector{T}) where {T<:Real} = Half(MatrixData(σᵤ²))  # matrix data
HF(σᵤ²::Vector{Vector{T}}) where {T<:Real} = Half(MatrixData(σᵤ²))  # matrix data
HF(σᵤ²::Vector{Matrix{T}}) where {T<:Real} = Half(σᵤ²)
HF(σᵤ²::Matrix) = Half(σᵤ²)
# allow different notation
Half(;σᵤ²) = HF(σᵤ²)
half(;σᵤ²) = HF(σᵤ²)
h(;σᵤ²) = HF(σᵤ²)
# functors: fit the marginal effect equation
function (s::Half)(x::Vector)
    return  (exp(s.σᵤ² * x),)  # note: Base.:*(::Matrix, ::Vector) will return vector
end

struct Trun <: AbstractDist
    μ::T where{T<:AbstractArray}
    σᵤ²::S where{S<:AbstractArray}
end
# various construction of truncated normal(if it's frame data just return)
# ensure matrix data
TN(μ::Union{Symbol, Tuple{Vararg{Symbol}}}, σᵤ²::Union{Symbol, Tuple{Vararg{Symbol}}}) = (Trun, μ, σᵤ²)  # frame data
TN(μ::Vector{S}, σᵤ²::Vector{T}) where{S<:Real, T<:Real} = Trun(DataMatrix(μ), DataMatrix(σᵤ²))  # matrix data
TN(μ::Vector{Vector{S}}, σᵤ²::Vector{Vector{T}}) where{S<:Real, T<:Real} = Trun(DataMatrix(μ), DataMatrix(σᵤ²))  # matrix data
TN(μ, σᵤ²::Vector{T}) where{T<:Real} = Trun(μ, DataMatrix(σᵤ²))  # matrix data
TN(μ, σᵤ²::Vector{Vector{T}}) where{T<:Real} = Trun(μ, DataMatrix(σᵤ²))  # matrix data
TN(μ::Vector{T}, σᵤ²) where{T<:Real} = Trun(DataMatrix(μ), σᵤ²)  # matrix data
TN(μ::Vector{Vector{T}}, σᵤ²) where{T<:Real} = Trun(DataMatrix(μ), σᵤ²)  # matrix data
TN(μ::Matrix, σᵤ²::Matrix) = Trun(μ, σᵤ²)
TN(μ::Vector{Matrix{T}}, σᵤ²::Vector{Matrix{T}}) where{T<:Real} = Trun(μ, σᵤ²)
# allow different notation
Trun(;μ, σᵤ²) = TN(μ, σᵤ²)
trun(;μ, σᵤ²) = TN(μ, σᵤ²)
t(;μ, σᵤ²) = TN(μ, σᵤ²)
# functors: fit the marginal effect equation
function (s::Trun)(x::Vector)
    μ, σᵤ² = unpack(s)
    n = varNum(μ)
    Wμ, Wᵤ = x[1:n], x[n+1:end] 
    return (μ * Wμ, exp(σᵤ²*Wᵤ))
end


struct Expo <: AbstractDist
    λ::T where{T<:AbstractArray}
end
# various construction of exponential
Expo(λ::Union{Symbol, Tuple{Vararg{Symbol}}}) = (Expo, λ)
Expo(λ::Vector{T}) where{T<:Real}  = Expo(MatrixData(λ))
Expo(λ::Vector{Vector{T}}) where{T<:Real}  = Expo(MatrixData(λ))
# allow different notaion
Expo(;λ) = Expo(λ)
expo(;λ) = Expo(λ)
e(;λ) = Expo(λ)
# functors: fit the marginal effect notation
(s::Expo)() = (:λ,)
function (s::Expo)(x::Vector)
    return  (exp(s.λ * x),)
end


"""
- calculate the unconditional mean of the composite error term
- if it's calculated to generate the marginal effect return Real required by the forwardDiff
"""
function uncondU(::Type{Half}, σᵤ², dist_coeff)
    res = sqrt.((2/π) * exp(σᵤ² * dist_coeff))
    if length(res) == 1
        return (res)[1]
    else
        return res
    end
end

function uncondU(::Type{Trun}, μ, σᵤ², dist_coeff)
    n = varNum(μ)
    Wμ, Wᵤ = dist_coeff[begin:n], dist_coeff[n+1:end] 
    μ = μ * Wμ
    σᵤ = exp(0.5 * σᵤ² * Wᵤ)

    Λ = μ ./ σᵤ 
    res = σᵤ .* (Λ + normpdf.(Λ) ./ normcdf.(Λ))
    if length(res) == 1
        return res[1]
    else
        return res    
    end
end

function uncondU(::Type{Expo}, λ, dist_coeff) 
    res = exp(λ * dist_coeff)
    if length(res) == 1
        return res[1]
    else
        return res
    end
end