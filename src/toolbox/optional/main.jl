"""
- ensure all form of data is matrix
"""
MatrixData(v::Vector{T}) where{T} = reshape(v, length(v), 1)
Matrixdata(v::Vector{Vector{T}}) where{T} = [reshape(i, length(i), 1) for i in v]
MatrixData(m::Matrix{T}) where{T} = m
MatrixData(m::Vector{Matrix{T}}) where{T} = m


"""
- used to check the number of explanatory variables
- two methods
1. for ondinary matrix data
2. for panelize matrix data
"""
varNum(m::Matrix) = size(m, 2)
varNum(v::Vector{Matrix{T}}) where{T<:Real} = size(v[1], 2)


"""
to calculate the number of observation
"""
obsNum(m::Matrix) = size(m, 1)
obsNum(v::Vector{T}) where{T<:Real} = length(v) 
obsNum(m::Vector{Matrix{T}}) where{T<:Real} = sum(size(i, 1) for i in m)
obsNum(v::Vector{Vector{T}}) where{T<:Real} = sum(length(i) for i in v)


"""
- operation for panelize data
"""
Base.exp(x::Vector) = exp.(x)
Base.exp(x::Vector{Matrix{T}}) where{T<:Real} = [exp.(i) for i in x]
Base.exp(x::Vector{Vector{T}}) where{T<:Real} = [exp(i) for i in x]
Base.sqrt(x::Vector) = sqrt.(x)
Base.:*(x::Vector{Matrix{T}}, y::Vector{S}) where{T<:Real, S<:Real} = [i * y for i in x]
Base.:*(x::Vector{Matrix{T}}, y::Vector{Matrix{S}}) where{T<:Real, S<:Real} = [x[i] .* y[i] for i = eachindex(x)]
Base.:*(x::Vector, y::Vector)  = x .* y
Base.:/(x::Vector{Vector{S}}, y::Vector{Vector{T}}) where{T<:Real, S<:Real} = [x[i] ./ y[i] for i = eachindex(x)]
StatsFuns.normpdf(x::Vector) = normpdf.(x)
StatsFuns.normcdf(x::Vector) = nomrcdf.(x)
Statistics.mean(x::Vector{Vector{T}}) where{T<:Real} = mean.(x)


"""
- ordered dictionary utilities to get multiple indexes more efficient
"""
function setindex!(A::OrderedDict, X, ind)
    defaultKey = keys(A)
    for i in eachindex(X)
        j = ind[i]
        j in defaultKey || throw("misspecification of the keywords argmentus $j in OrderedDict") 
        Base.setindex!(A, X[i], j)
    end
end

getindex(A::OrderedDict, ind::Vector) = [Base.getindex(A, i) for i in ind]
getindex(A::OrderedDict, ind::Tuple{Vararg{T}}) where{T} = [Base.getindex(A, i) for i in ind]


"""
- data frame utilities to create matrix data
"""
FrametoMatrix(ind::Symbol; df::DataFrame) = MatrixData(Base.getindex(df, :, ind))
FrametoMatrix(ind::Vector{Symbol}; df::DataFrame) = hcat([Base.getindex(df, :, i) for i in ind]...)
FrametoMatrix(ind::Tuple{Vararg{Symbol}}; df::DataFrame) = hcat([Base.getindex(df, :, i) for i in ind]...)


"""
- self defined data type utilities to get the multiple propertyies more efficient
- two methods
1. unpack all properties
2. only unpack the given fieldnames
"""
function unpack(A)
    return [Base.getproperty(A, i) for i in fieldnames(typeof(A))]
end


function unpack(A, ind::Tuple{Vararg{Symbol}})
    return [Base.getproperty(A, i) for i in ind]
end


"""
- two methods for tidy the panel:
1. given the index of each panel 
2. assuming each panel is continuous so only need to give the length of periods of each panel
- note:
aiming to make the broadcasting possible, we let the id to be the keyword arguments
and since we provide two method so there are two non-keyword arguments function
"""
function tidyPanel(data::Matrix; id)  # to multiple dispatch on keyword arguments
    tidyPanel(data, id)
end

function tidyPanel(data, id::Vector{Vector{Int64}})
    return [data[i, :] for i in id]
end

function tidyPanel(data, id::Vector{Int64})
    panelize = Vector{typeof(data)}(undef, length(id))
    beg = 1
    for i = eachindex(id)
        en = beg + id[i] - 1
        panelize[i] = data[beg:en, :]
        beg = en + 1
    end
    return panelize
end


flatPanel(x::Vector{Vector{T}}) where{T<:Real} = vcat(x...)
flatPanel(x::Vector{Matrix{T}}) where{T<:Real} = vcat(x...)
flatPanel(x::Vector{T}) where{T<:Real} = x
flatPanel(x::Matrix{T}) where{T<:Real} = x
# flatPanel(x...) = [flatPanel(i) for i in x]


function flatTidyLag(args...; l)
    res = Vector(undef, length(args))
    for i = eachindex(args)
        tar = args[i]
        resᵢ = Vector(undef, sum([length(j)-l for j in tar]))
        beg = 1
        for j in tar
            en =  beg + (length(j)-l) - 1
            resᵢ[beg:en] = j[1+l:end]
            beg = en + 1
        end
        res[i] = resᵢ
    end
    return res
end


"""
- two method to check the Multicollinearity: 
1. for ordinary matirx data 
2. for the panelize data
- two return
1. non-multicollinearity matrix
2. indexes of pivot columns
"""
function isMultiCollinearity(name, themat::Matrix)  # for cross data
    colnum = size(themat, 2)
    colnum == 1 && return
    pivots = rref_with_pivots(themat)[2]
    if length(pivots) != colnum
        printstyled("\n * Find Multicollinearity\n\n", color=:red)
        for j in filter(x->!(x in pivots), 1:colnum)
            println("    number $j column in $(name) is dropped")
        end
        return themat[:, pivots], pivots
    end
    return themat, pivots
end

function isMultiCollinearity(name, themat::Vector{Matrix{T}}) where{T<:Real}  # for panel data
    _themat = vcat(themat...)
    colnum = size(_themat, 2)
    colnum == 1 && return
    pivots = rref_with_pivots(_themat)[2]
    if length(pivots) != colnum
        printstyled("\n * Find Multicollinearity\n\n", color=:red)
        for j in filter(x->!(x in pivots), 1:colnum)
            println("    number $j column in $(name) is dropped")
        end
        _themat = _themat[:, pivots]
    end
    return tidyPanel(_themat; id=size.(themat, 1)), pivots
end


"""
- to print some long vector with better view(threee items each row)
"""
function prettyParam(v::Vector)
    v = string.(v)
    len = length(v)
    remain = len % 3
    en = len - remain
    for i = 1:3:en
        println("         $(v[i]), $(v[i+1]), $(v[i+2]),")
     end
    if remain == 1
        println("         $(v[end])")
    else
        println("         $(v[end-1]), $(v[end])")
    end
end