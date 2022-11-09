useData(df::String) = DataFrame(File(df))
useData(df::DataFrame) = df
useData() = nothing


function sfopt(;kwargs...)
    # opt = (:warmstart_solver, :warmstart_maxIT, :main_solver, :main_maxIT, :tolerance,
    #        :verbose, :banner, :ineff_index, :marginal, :table_format)
    default_opt = OrderedDict(:warmstart_solver=>NelderMead(), :warmstart_maxIT=>100,
                          :main_solver=>Newton(), :main_maxIT=>2000, :tolerance=>1e-8,
                          :verbose=>false)
    length(kwargs) != 0 && setindex!(default_opt, values(values(kwargs)), keys(kwargs))
    # return Tuple(values(default))
    return default_opt

end


"""
1. given the all initial values of  parameters
2. given only some of the initial values and it's necessary to assign the keyword
"""
sfinit(startpt::Vector) = startpt
sfinit(;kwargs...) = values(kwargs)


function sfmodel_fit(model)
    struc, data = model
    startpt = ones(struc.ψ[end]) .* 0.1
    if isa(data.frontiers, Vector)
        startpt[1:struc.ψ[1]] = flatPanel(data.frontiers) \ flatPanel(data.depvar)
    else
        startpt[1:struc.ψ[1]] = data.frontiers \ data.depvar
    end

    ξ = MLE(struc, data, sfopt(), startpt)
    jlms, bc = jlmsbc(ξ, struc, data)
    
    return (ξ=ξ, struc=struc, data=data, opt=opt, startpt=startpt, jlms=jlms, bc=bc)
end


function sfmodel_fit(model, opt::OrderedDict, startpt::Union{Vector, NamedTuple})
    struc, data = model
    if isa(startpt, NamedTuple)  
        ψ = struc.ψ
        _startpt = ones(ψ[end]) .* 0.1
        if isa(data.frontiers, Vector)
            _startpt[1:struc.ψ[1]] = flatPanel(data.frontiers) \ flatPanel(data.depvar)
        else
            _startpt[1:struc.ψ[1]] = data.frontiers \ data.depvar
        end
        template = struc.varmat[:, 1]
        push!(template, :end)  # to prevent can't getnext
        for i in keys(startpt)
            beg = findfirst(x->x==i, template)
            en = findnext(x->x!=Symbol(""), template, beg+1) - 1
            _startpt[beg:en] .= Base.getproperty(startpt, i)
        end
        startpt = _startpt
    end

    ξ = MLE(struc, data, opt, startpt)
    jlms, bc = jlmsbc(ξ, struc, data)

    return (ξ=ξ, struc=struc, data=data, opt=opt, startpt=startpt, jlms=jlms, bc=bc)
end


function sfmodel_fit(model, opt::OrderedDict)
    struc, data = model
    startpt = ones(struc.ψ[end]) .* 0.1
    if isa(data.frontiers, Vector)
        startpt[1:struc.ψ[1]] = flatPanel(data.frontiers) \ flatPanel(data.depvar)
    else
        startpt[1:struc.ψ[1]] = data.frontiers \ data.depvar
    end

    ξ = MLE(struc, data, opt, startpt)
    jlms, bc = jlmsbc(ξ, struc, data)

    return (ξ=ξ, struc=struc, data=data, opt=opt, startpt=startpt, jlms=jlms, bc=bc)
end


function sfmodel_fit(model, startpt::Union{Vector, NamedTuple})
    struc, data = model
    if isa(startpt, NamedTuple)  
        ψ = struc.ψ
        _startpt = ones(ψ[end]) .* 0.1
        _startpt[1:struc.ψ[1]]  = data.frontiers \ data.depvar
        template = struc.varmat[:, 1]
        push!(template, :end)  # to prevent can't getnext
        for i in keys(startpt)
            beg = findfirst(x->x==i, template)
            en = findnext(x->x!=Symbol(""), template, beg+1) - 1
            _startpt[beg:en] .= Base.getproperty(startpt, i)
        end
        startpt = _startpt
    else
        startpt = [startpt...]
    end

    ξ = MLE(struc, data, sfopt(), startpt)
    jlms, bc = jlmsbc(ξ, struc, data)

    return (ξ=ξ, struc=struc, data=data, opt=opt, startpt=startpt, jlms=jlms, bc=bc)
end