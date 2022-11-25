useData(df::String) = DataFrame(File(df))
useData(df::DataFrame) = df
useData() = nothing


function sfopt(;kwargs...)
    default_opt = OrderedDict{Symbol, Any}(
        :warmstart_solver=>NelderMead(), 
        :warmstart_maxIT=>100,
        :main_solver=>Newton(),
        :main_maxIT=>2000,
        :tolerance=>1e-8,
        :verbose=>false,
        :table_format=>:text
    )
    length(kwargs) != 0 && setindex!(default_opt, values(values(kwargs)), keys(kwargs))
    return default_opt

end


"""
1. given the all initial values of  parameters
2. given only some of the initial values and it's necessary to assign the keyword
"""
sfinit(startpt::Vector{<:Real})  = startpt
sfinit(;kwargs...) = values(kwargs)

function sfstartpt(frontiers, depvar, ψ; panel)
    startpt = ones(ψ[end]) .* 0.1
    if panel
        startpt[1:ψ[1]] = flatPanel(frontiers) \ flatPanel(depvar)
    else
        startpt[1:ψ[1]] = frontiers \ depvar
    end
    return startpt
end


function sfmodel_fit(;model::Tuple{SFmodel, AbstractData},
                      options::OrderedDict=nothing,
                      startpt::Union{Vector, NamedTuple}=nothing
                    )
    struc, data = model
    options === nothing && (options = sfopt())

    if startpt === nothing
        startpt = sfstartpt(data.frontiers, data.depvar, struc.ψ; panel=isa(data, PanelData))
    else
        if isa(startpt, NamedTuple)  
            _startpt = sfstartpt(data.frontiers, data.depvar, struc.ψ; panel=isa(data, PanelData))
            template = struc.varmat[:, 1]
            push!(template, :end)  # to prevent can't getnext
            for i in keys(startpt)
                beg = findfirst(x->x==i, template)
                en = findnext(x->x!=Symbol(""), template, beg+1) - 1
                _startpt[beg:en] .= Base.getproperty(startpt, i)
            end
            startpt = _startpt
        end
    end
    
    modelInfo(struc)
    printstyled("\n * optimization \n\n", color=:yellow)
    Hessian, ξ, warmup_opt, main_opt = MLE(struc, data, options, startpt)
    diagonal = post_estimation(Hessian, ξ, data, struc)
    modelEstimation(data.nofobs, warmup_opt, main_opt)  # show the simulation results
    outputTable(struc.varmat, ξ, diagonal, data.nofobs, options[:table_format])  # output table
    jlms, bc = jlmsbc(ξ, struc, data)
    
    return (ξ=ξ, struc=struc, data=data, options=options, jlms=jlms, bc=bc)
end