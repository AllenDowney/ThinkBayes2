module EmpiricalDist
include("utils.jl")
import .Utils: checkpkgs
checkpkgs("PrettyTables", "Statistics", "StatsBase", "Distributions", "RecipesBase", "DataFrames", "Interpolations", "KernelDensity")

import PrettyTables, Statistics, StatsBase, RecipesBase, DataFrames
import Random, Interpolations, KernelDensity, Plots, Distributions

abstract type AbstractSeries{T <: Real, S} <: AbstractVector{T} end
Base.eltype(::AbstractSeries{T}) where T = T
Base.keytype(::AbstractSeries{T,S}) where {T,S} = S
Base.size(x::AbstractSeries) = size(values(x))
Base.findfirst(p::Function, x::AbstractSeries) = findfirst(p, keys(x))
Base.argmax(x::AbstractSeries) = keyat(x, argmax(values(x)))
valarraytype(x::AbstractSeries) = typeof(values(x))
keyat(x::AbstractSeries, i) = keys(x)[i]
valueat(x::AbstractSeries, i) = values(x)[i]
valueat!(x::AbstractSeries, i, v) = values(x)[i] = v
Base.iterate(x::AbstractSeries, i=1) = (i - 1) < length(x) ? (valueat(x, i), i + 1) : nothing
Base.pairs(x::AbstractSeries) = map(x -> Pair(x[1], x[2]),zip(keys(x), values(x)))

_check_matching_lengths(v, k) = length(v) == length(k) || error("keys and values must have same length")

function makeseries(AS::Type{<:AbstractSeries}, p, q; normalize = false, sort = false, name = nothing)
    _check_matching_lengths(p, q)
    k, v = q, p
    if normalize
        v = p ./ sum(p)
    end
    if sort
        perm = sortperm(k)
        k, v = k[perm], v[perm]        
    end
    AS(v, k, name)
end

makeseries(AS::Type{<:AbstractSeries}, p::Real, q; kwargs...) = makeseries(AS, fill(p, length(q)), q; kwargs...)
makeseries(AS::Type{<:AbstractSeries{T,S}}; name = nothing) where {T,S} = AS(Vector{T}(), Vector{S}(), name)

function makeseries(AS::Type{<:AbstractSeries}, p::AbstractVector, q::AbstractRange; kwargs...)
    :sort in keys(kwargs) && kwargs[:sort] && error("sort not supported for type $(typeof(q))")
    makeseries(AS, p, collect(q); kwargs...)
end

function makeseries(AS::Type{<:AbstractSeries}, x::AbstractDict; kwargs...)
    p = collect(x)
    k = map(x->x.first, p)
    v = map(x->x.second, p)
    makeseries(AS, v, k; kwargs...)
end

function makeseries(AS::Type{<:AbstractSeries}, a::AbstractVector; kwargs...)
    # if index appears in kwargs, `a` is *values*, otherwise, it's keys
    if :index in keys(kwargs)
        k = kwargs[:index]
        v = a
        kwargs = filter(x -> x.first != :index, kwargs) # remove index
    else
        c = collect(StatsBase.countmap(a))
        k = map(x->x.first, c)
        v = map(x->x.second, c)
        v = float(v) # force float
    end
    makeseries(AS, v, k; kwargs...)
end

function Base.first(x::T, n::Integer=1) where{T<:AbstractSeries}
    upperbound = n < length(x) ? n : :end
    makeseries(T, valueat(x, 1:upperbound), keyat(x, 1:upperbound); sort=false)
end

function Base.push!(x::AbstractSeries{T,S}, p::Pair{S, T}) where {T,S}
    push!(keys(x), p.first)
    push!(values(x), p.second)
    return x
end

Base.sort(x::AS) where {AS<:AbstractSeries} = makeseries(AS, valus(x), keys(x), sort=true)

function _check_matching_keys(series)::Bool
    if length(series) > 1
        k = keys(series[1])
        return all(s -> (keys(s) == k), series[2:end])
    end
    true
end

"""
creates an uninitialzied series of the same type as input series
if `empty` is not set, keys are copied to the new series
Otherwise, an empty series is returned

`name` is passed to the constructor as the new series name
"""
function Base.similar(x::T; empty::Bool = false, name=nothing) where{T<: AbstractSeries}
    if !empty
        v = valarraytype(x)(undef, size(x))
        k = copy(keys(x))
        T(v, k, name)
    else
        makeseries(T; name=name)
    end
end

findloc(x::AbstractSeries, i) = findfirst(==(i), keys(x)) # XXX pretty bad

function Base.push!(x::AbstractSeries, p::Pair{S, T}) where {T,S}
    push!(keys(x), p.first)
    push!(values(x), p.second)
    return x
end

function Base.getindex(x::AbstractSeries, i)
    if (loc = findloc(x, i)) !== nothing
        return valueat(x,loc)
    else
        throw(KeyError(i))
    end
end

function Base.getindex(x::AbstractSeries, i::BitVector)
    length(x) == length(i) || error("mask must have same length as series")
    makeseries(typeof(x), valueat(x, i), keyat(x, i))
end


(x::AbstractSeries{T})(i) where T = (loc = findloc(x, i); isnothing(loc) ? T(0) : valueat(x,loc))
(x::AbstractSeries{T})(a::AbstractVector) where T = [x(i) for i in a]

function Base.setindex!(x::AbstractSeries, v, i)
    if (loc = findloc(x, i)) !== nothing
        valueat!(x, loc, v)
    else
        push!(x, (i => v))
        return v
    end
end

colheader(x::AbstractSeries) = isnothing(name(x)) ? "values" : name(x)
function _show(io::IO, x::AbstractSeries; backend = Val(:text))
    PrettyTables.pretty_table(io, values(x); row_names=keys(x), header=[colheader(x)], 
                    crop = :both, compact_printing = true, backend = backend)
end
Base.show(io::IO, m::MIME{Symbol("text/plain")}, x::AbstractSeries) = _show(io, x, backend = Val(:text))
Base.show(io::IO, m::MIME{Symbol("text/html")}, x::AbstractSeries) = _show(io, x, backend = Val(:html))
Base.show(io::IO, x::AbstractSeries) = _show(io, x)

struct SeriesStyle <: Base.Broadcast.BroadcastStyle end
Base.Broadcast.BroadcastStyle(::Type{<:AbstractSeries}) = SeriesStyle()
Base.Broadcast.BroadcastStyle(::SeriesStyle, ::SeriesStyle) = SeriesStyle()
Base.Broadcast.BroadcastStyle(::SeriesStyle, ::Base.Broadcast.BroadcastStyle) = SeriesStyle()
Base.Broadcast.BroadcastStyle(::Base.Broadcast.BroadcastStyle, ::SeriesStyle) = SeriesStyle()

function mergeseries(::typeof(+), x::T, y::T) where {T<:AbstractSeries}
    # XXX merged series is sorted. Original order is lost
    # XXX only defined for addition
    xdict = Dict(zip(keys(x), values(x)))
    ydict = Dict(zip(keys(y), values(y)))
    
    merged = merge(+, xdict, ydict)
    return makeseries(T, merged; sort=true)
end

for op in (*, +, -, /, ^)
    @eval function Base.broadcasted(::typeof($op), x::T, y) where {T<:AbstractSeries}
        return T($op.(values(x), y), keys(x), name(x))
    end
    # XXX doesn't support adding AbstractSeries{Int} + AbstractSeries{Float}
    @eval function Base.broadcasted(::typeof($op), x::T, y::T) where {T<:AbstractSeries}
        if _check_matching_keys((x, y))
            T($op.(values(x), values(y)), keys(x), name(x))
        else
            mergeseries($op, x, y)
        end
    end
    @eval function Base.broadcasted(::typeof($op), x, y::T) where {T<:AbstractSeries}
        return T($op.(x, values(y)), keys(y), name(y))
    end
end

function Base.broadcasted(::typeof(Base.literal_pow), ^, x::T, ::Val{y}) where {T<:AbstractSeries, y}
    return T(values(x) .^ y, keys(x), name(x))
end

for op in (>, <, ==, >=, <=)
    # boolean operators: return a bit vector
    @eval Base.broadcasted(::typeof($op), x::AbstractSeries, y) = Base.broadcasted($op, values(x), y)
    @eval Base.broadcasted(::typeof($op), x, y::AbstractSeries) = Base.broadcasted($op, x, values(y))
    @eval function Base.broadcasted(::typeof($op), x::AbstractSeries, y::AbstractSeries)
        if !_check_matching_keys((x, y))
            if ==($op,==) # yep, this is valid Julia code
                return false
            else
                error("$op is undefined for series with different keys")
            end
        end
        Base.broadcasted($op, values(x), values(y))
    end
end

Base.isapprox(p1::AbstractSeries, p2::AbstractSeries) = _check_matching_keys((p1, p2)) && all(Base.isapprox.(values(p1), values(p2)))

RecipesBase.@recipe function f(p::AbstractSeries; fill_below = nothing, fill_above=nothing)
    n = name(p)
    if !isnothing(n) && length(n) > 0
        label --> n
    end

    if get(plotattributes, :label, nothing) !== nothing
        legend --> true
    end
    
    if !isnothing(fill_below)
        RecipesBase.@series begin
            fill --> (0,0.5,:grey)
            primary := false
            p[keys(p) .<= fill_below]
        end
    end
    if !isnothing(fill_above)
        RecipesBase.@series begin
            fill --> (0,0.5,:grey)
            primary := false
            p[keys(p) .>= fill_above]
        end
    end

    keys(p), values(p)
end

Base.copy(src::T) where {T<:AbstractSeries} = makeseries(T, copy(values(src)), copy(keys(src)), name=name(src))

Base.copyto!(dest::AbstractVector, src::AbstractSeries) = copyto!(dest, values(src))
function Base.copyto!(d::T, bc::Base.Broadcast.Broadcasted) where {T<:AbstractSeries}
    bcf = Base.Broadcast.flatten(bc)
    data = Base.copyto!(values(d), bcf)
    T(data, keys(d), name(d))
end

function normalize!(x::AbstractSeries)
    s = sum(x)
    if !iszero(s) # XXX what's the correct behavior?
        copyto!(values(x), values(x) ./ s)
    end
    return s
end

DataFrames.transform(f::Function, x::AS) where{AS<:AbstractSeries} = makeseries(AS, copy(values(x)), f.(keys(x)))


"""
Add a series to a DataFrame as a new column.
The series name is used as the new column name if `colname` is not given
if `overwrite` is `true`, the new column will replace an existing column with same name, otherwise
(the default), it will raise an exception

The series keys must match the `index` column (if `index` doesn't exist, it will be added)
"""
function Base.insert!(df::DataFrames.AbstractDataFrame, x::AbstractSeries; colname = nothing, overwrite = false)
    n = isnothing(colname) ? name(x) : Symbol(colname)
    if !isnothing(n)
        if (string(n) in names(df)) && !overwrite
            # XXX more meaningful exception?
            error("column $n already exists in dataframe")
        end
    end

    k = keys(x)
    v = values(x)
    
    # check index column
    if "index" in names(df)
        if k != df[!, :index]
            error("series keys do not match index column")
        end
    else
        df[!, "index"] = k
    end
    
    if !isnothing(n)
        df[!, n] = v
    else
        # use default column name 
        df[!, end+1] = v
    end
    nothing
end

function make_dataframe(x::T) where {T<:AbstractVector{<:AbstractSeries}}
    df = DataFrames.DataFrame()
    for s in x
        append!(df, Dict(zip(Symbol.(keys(s)), values(s))), cols = :union)
    end
    
    return df
end

make_dataframe(x::Tuple{Vararg{AbstractSeries}}) = make_dataframe(collect(x))

struct Series{T,S} <: AbstractSeries{T,S}
    values::Vector{T}
    index::Vector{S}
    name::Union{AbstractString, Nothing}
end
Base.values(x::Series) = x.values
Base.keys(x::Series) = x.index
name(x::Series) = x.name


struct Pmf{T <: AbstractFloat, S} <: AbstractSeries{T,S}
    ps::Vector{T}
    qs::Vector{S}
    name::Union{AbstractString, Nothing}
end

Pmf(a::AbstractSeries) = Pmf(values(a), keys(a), name(a))

# extract/deduce T, S
const _deffloattype = typeof(float(0))
Pmf(; kwargs...) = makeseries(Pmf{_deffloattype, Any}; kwargs...) # default floating point
Pmf{T,S}(; kwargs...) where {T<:Real, S} = makeseries(Pmf{T, S}; kwargs...) 

Pmf(x::AbstractDict{K,V}; sort=true, kwargs...) where {K,V}= makeseries(Pmf{V,K}, x; sort=sort, kwargs...)
Pmf(a::AbstractVector; kwargs...) where S = makeseries(Pmf, a; kwargs...)
Pmf(p, q; kwargs...) where T = makeseries(Pmf, float.(p), q; kwargs...) # force floating point
Pmf(n::Real, s::AbstractSet; kwargs...) = makeseries(Pmf, n, collect(s); kwargs...)

function makeuniform(q; kwargs...)
    pmf = Pmf(1.0, q; kwargs...)
    normalize!(pmf)
    return pmf
end

colheader(x::Pmf) = isnothing(name(x)) ? "probs" : name(x)

Base.values(x::Pmf) = x.ps
Base.keys(x::Pmf) = x.qs
name(x::Pmf) = x.name

function Base.sort!(x::Pmf)
    p = sortperm(x.qs)
    x.qs .= x.qs[p]
    x.ps .= x.ps[p]
    x
end

"""Probability of quantities greater than threshold."""
prob_ge(p::Pmf{<: Real}, threshold) = sum(p[keys(p) .>= threshold])
prob_gt(p::Pmf{<: Real}, threshold) = sum(p[keys(p) .> threshold])

"""Probability of quantities less than threshold."""
prob_le(p::Pmf{<: Real}, threshold) = sum(p[keys(p) .<= threshold])
prob_lt(p::Pmf{<: Real}, threshold) = sum(p[keys(p) .< threshold])

prob_eq(p::Pmf{<: Real}, threshold) = sum(p[keys(p) .== threshold])

function pmf_outer(op::Function, p1::Pmf{<:Real, <:Real}, p2::Pmf{<:Real, <:Real})
    qm = op.(keys(p1), keys(p2)');
    pm = values(p1) .* values(p2)';
    return sum(qm .* pm)
end

prob_ge(p1::Pmf{<: Real, <: Real}, p2::Pmf{<: Real, <: Real}) = pmf_outer(>=, p1, p2)
prob_gt(p1::Pmf{<: Real, <: Real}, p2::Pmf{<: Real, <: Real}) = pmf_outer(>, p1, p2)
prob_le(p1::Pmf{<: Real, <: Real}, p2::Pmf{<: Real, <: Real}) = pmf_outer(<=, p1, p2)
prob_lt(p1::Pmf{<: Real, <: Real}, p2::Pmf{<: Real, <: Real}) = pmf_outer(<, p1, p2)
prob_eq(p1::Pmf{<: Real, <: Real}, p2::Pmf{<: Real, <: Real}) = pmf_outer(==, p1, p2)

ge_dist = prob_ge
gt_dist = prob_gt
le_dist = prob_le
lt_dist = prob_lt
eq_dist = prob_eq

maxprob(x::Pmf) = argmax(x)
pmffromseq(seq::AbstractVector; sort=true, normalize=true, name=nothing) = Pmf(seq; sort, normalize, name)
pmffromseq(seq::AbstractArray; sort=true, normalize=true, name=nothing)  = pmffromseq(vec(seq); sort, normalize, name)


"""Make a discrete approximation.

dist: distribution object
qs: quantities

returns: Pmf
"""
function pmffromdist(dist, qs)
    ps = Distributions.pdf.(dist, qs)
    pmf = Pmf(ps, qs)
    normalize!(pmf)
    return pmf
end

Statistics.mean(x::Pmf) = sum(values(x) .* keys(x))
Statistics.var(x::Pmf) = sum((keys(x) .- Statistics.mean(x)) .^ 2 .* values(x))
Statistics.std(x::Pmf) = sqrt(Statistics.var(x))

function Statistics.quantile(x::Pmf, a::AbstractVector)
    # XXX we could just rely on Cdf.
    # E.g.:
    #    quantile(makecdf(x), a)
    k = keys(x)
    v = cumsum(values(x))
    [k[searchsortedfirst(v, i)] for i in a]
end

Statistics.quantile(x::Pmf, i::T) where{T<:AbstractFloat} = Statistics.quantile(x,[i])[1]
Statistics.median(x::Pmf)  = Statistics.quantile(x,[0.5])[1]

StatsBase.sample(p::Pmf) = StatsBase.sample(p.qs, StatsBase.pweights(p.ps))
StatsBase.sample(p::Pmf, n::Integer) = StatsBase.sample(p.qs, StatsBase.pweights(p.ps), n)
Random.rand(p::Pmf) = StatsBase.sample(p)
Random.rand(p::Pmf, n::Integer) = StatsBase.sample(p, n)

credibleinterval(x::Pmf, p::Real) = Statistics.quantile(x, [(1-p)/2, 1-(1-p)/2])

function convolvedist(ufunc::Function, xs::Vararg{Pmf{T,S}}) where {T<:Real, S<:Real}
    length(xs) > 1 || error("at least 2 series are needed for convolvedist")
    qs = keys(xs[1])
    ps = values(xs[1])
    for i in 2:length(xs)
        q, p = keys(xs[i]), values(xs[i])
        qs = vec(ufunc.(qs,q'))
        ps = vec(ps .* p')
    end
    d = StatsBase.addcounts!(Dict{S, T}(), qs, ps)
    pmf = Pmf(d, sort=true)
end

adddist(xs::Vararg{Pmf}) = convolvedist(+, xs...)
adddist(xs::AbstractVector{T}) where {T<:Pmf} = convolvedist(+, xs...)
adddist(x::Pmf, s::T) where{T<:Real} = Pmf(values(x), keys(x) .+ s)

subdist(xs::Vararg{Pmf}) = convolvedist(-, xs...)
subdist(xs::AbstractVector{T}) where {T<:Pmf} = convolvedist(-, xs...)
subdist(x::Pmf, s::T) where{T<:Real} = Pmf(values(x), keys(x) .- s)

muldist(xs::Vararg{Pmf}) = convolvedist(*, xs...)
muldist(xs::AbstractVector{T}) where {T<:Pmf} = convolvedist(*, xs...)

divdist(xs::Vararg{Pmf}) = convolvedist(/, xs...)
divdist(xs::AbstractVector{T}) where {T<:Pmf} = convolvedist(/, xs...)


function expand_seq(sp::AbstractVector{<:Pmf{T}}) where {T<:Real}
    # quite inefficient; assumes keys are sorted
    ks = sort(union([keys(p) for p in sp]...))
    k2l = Dict([x[2] =>x[1] for x in enumerate(ks)])
    m = zeros(T, (length(sp), length(ks)))
    for (i,s) in enumerate(sp)
        for (q,p) in pairs(s)
            m[i, k2l[q]] = p
        end
    end
    return (m, ks)
end

function makemixture(p::Pmf, s::AbstractVector{<:Pmf}; name = nothing)
    # XXX assumes Pmfs have same "base" (or can the extended to one)
    m, k = expand_seq(s)
    v = vec(sum(values(p) .* m, dims=1))
    Pmf(v, k, name)
end

"""Make a kernel density estimate from a sample."""
function kde_from_sample(sample::AbstractVector{<: Real}, qs::AbstractVector{<:Real})
    ps = Distributions.pdf(KernelDensity.InterpKDE(KernelDensity.kde(sample)), qs)
    pmf = Pmf(ps, qs)
    normalize!(pmf)
    return pmf
end

KernelDensity.kde(p::Pmf) = KernelDensity.kde(p.qs; weights = StatsBase.pweights(p.ps))

function marginal(p::Pmf{<:Real, <:Tuple}, d::Integer)
    _extract_tuple_ntype(::Type{<:Tuple{T}}, ::Val{1}) where T = T
    _extract_tuple_ntype(::Type{<:Tuple{T, <:Any}}, ::Val{1}) where T = T
    _extract_tuple_ntype(::Type{<:Tuple{<:Any, T}}, ::Val{2}) where T = T
    _extract_tuple_ntype(::Type{<:Tuple{T, <:Any, <:Any}}, ::Val{1}) where T = T
    _extract_tuple_ntype(::Type{<:Tuple{<:Any, T, <:Any}}, ::Val{2}) where T = T
    _extract_tuple_ntype(::Type{<:Tuple{<:Any, <:Any, T}}, ::Val{3}) where T = T

    T = eltype(keys(p))
    V = eltype(values(p))
    K = _extract_tuple_ntype(T, Val{d}())
    dict = Dict{K, V}()
    for i in 1:length(p)
        k = keyat(p, i)[d]
        (k in keys(dict)) || (dict[k] = 0)
        dict[k] += valueat(p, i)
    end
    Pmf(dict)
end



struct Cdf{T <: AbstractFloat, S} <: AbstractSeries{T,S}
    cps::Vector{T}
    qs::Vector{S}
    name::Union{AbstractString, Nothing}
end

Cdf(; kwargs...) = makeseries(Cdf{_deffloattype, Any}; kwargs...)
Cdf{T,S}(; kwargs...) where {T<:Real, S} = makeseries(Cdf{T, S}; kwargs...) 
Cdf(p, q; kwargs...) where T = makeseries(Cdf, float.(p), q; kwargs...) # force floating point



Base.values(x::Cdf) = x.cps
Base.keys(x::Cdf) = x.qs
name(x::Cdf) = x.name

Base.cumsum(x::AbstractSeries) = Cdf(cumsum(values(x)), collect(keys(x)), nothing)

function makecdf(x::Pmf; normalize = true)
    cdf = cumsum(x)
    if normalize
        normalize!(cdf)
    end
    return cdf
end

function cdffromseq(x::AbstractVector; normalize = true, name = nothing)
    pmf = makeseries(Pmf, x; normalize=false, sort = true, name = name)
    makecdf(pmf; normalize)
end

# TODO: memoize/cache the result (or store in Cdf struct)
interpolation_func(s, d) = Interpolations.LinearInterpolation(s, d)
forward_inerpolate(x::Cdf, s) = interpolation_func(keys(x), values(x))(s)
(x::Cdf)(i) = forward_inerpolate(x, i)

# see https://discourse.julialang.org/t/problem-with-interpolations-did-it-change/60685
_prepare_for_intepolation(x, y) = (idx = unique(i -> x[i], 1:length(x)); (x[idx], y[idx]))
backward_interpolate(x::Cdf, s) = interpolation_func(_prepare_for_intepolation(values(x), keys(x))...)(s)
Statistics.quantile(x::Cdf, s) = backward_interpolate(x, s)
credibleinterval(x::Cdf, p::Real) = Statistics.quantile(x, [(1-p)/2, 1-(1-p)/2])

colheader(x::Cdf) = isnothing(name(x)) ? "cumulative" : name(x)
normalize!(x::Cdf) = copyto!(values(x), values(x) ./ values(x)[end])

function makepmf(x::Cdf{T, S}) where {T<: AbstractFloat, S}
    v = diff(vcat(zeros(T, 1), values(x))) # need to "restore" first element
    return Pmf(v, keys(x), name(x))
end

Statistics.mean(x::Cdf) = Statistics.mean(makepmf(x))
Statistics.var(x::Cdf) = Statistics.var(makepmf(x))
Statistics.std(x::Cdf) = Statistics.std(makepmf(x))

prob_ge(p::Cdf, threshold) = prob_ge(makepmf(p), threshold)
prob_gt(p::Cdf, threshold) = prob_gt(makepmf(p), threshold)
prob_le(p::Cdf, threshold) = prob_le(makepmf(p), threshold)
prob_lt(p::Cdf, threshold) = prob_lt(makepmf(p), threshold)
prob_eq(p::Cdf, threshold) = prob_lt(makepmf(p), threshold)

"""Make a binomial Pmf."""
makebinomial(n, p; kwargs...) = makeseries(Pmf, Distributions.pdf.(Distributions.Binomial(n,p), 0:n), 0:n; kwargs...)

maxdist(x::Cdf, n::Real) = Cdf(values(x) .^ n, keys(x), nothing)
mindist(x::Cdf, n::Real) = Cdf(1 .- (1 .- values(x)) .^ n, keys(x), nothing)

abstract type AbstractJointDistribution end

Base.size(j::AbstractJointDistribution) = size(distribution(j))
Base.maximum(j::AbstractJointDistribution) = maximum(distribution(j))
Base.sum(j::AbstractJointDistribution) = sum(distribution(j))
Base.length(j::AbstractJointDistribution) = length(rows(j))
Base.first(j::JD, n::Integer=1) where{JD<:AbstractJointDistribution} = JD(distribution(j)[1:n, :], rows(j)[1:n], columns(j))
Base.transpose(j::JD)  where{JD<:AbstractJointDistribution} = JD(distribution(j)', columns(j), rows(j))
Base.adjoint(j::AbstractJointDistribution) = transpose(j)

function DataFrames.stack(j::AbstractJointDistribution)
    m, x, y = distribution(j), columns(j), rows(j)
    q = [(y[u]..., x[v]...) for u in 1:length(y) for v in 1:length(x)]
    p = [m[u, v] for u in 1:length(y) for v in 1:length(x)]
    return Pmf(p, q)
end

function marginal(j::AbstractJointDistribution, dim)
    if (dim == :x) || (dim == 1)
        return Pmf(vec(sum(distribution(j), dims=1)), columns(j))
    elseif (dim == :y) || (dim == 2)
        return Pmf(vec(sum(distribution(j), dims=2)), rows(j))
    else
        error("unknown dimension")
    end
end

Plots.contour(j::AbstractJointDistribution, args...; levels = 4, kwargs...) = 
                Plots.contour(columns(j), rows(j), distribution(j), args...; levels=levels, kwargs... )
Plots.contour!(j::AbstractJointDistribution, args...; levels = 4, kwargs...) = 
                Plots.contour!(columns(j), rows(j), distribution(j), args...; levels=levels, kwargs... )

function _show(io::IO, j::AbstractJointDistribution; backend)
    PrettyTables.pretty_table(io, distribution(j); row_names=rows(j), header=columns(j), 
                    crop = :both, compact_printing = true, backend = backend)
end

Base.show(io::IO, m::MIME{Symbol("text/plain")}, x::AbstractJointDistribution) = _show(io, x, backend = Val(:text))
Base.show(io::IO, m::MIME{Symbol("text/html")}, x::AbstractJointDistribution) = _show(io, x, backend = Val(:html))

function Base.argmax(j::AbstractJointDistribution)
    m, x, y = distribution(j), columns(j), rows(j)
    p = argmax(m)
    return y[p[1]], x[p[2]]
end

struct JDStyle <: Base.Broadcast.BroadcastStyle end
Base.Broadcast.BroadcastStyle(::Type{AbstractJointDistribution}) = JDStyle()
Base.Broadcast.BroadcastStyle(::JDStyle, ::JDStyle) = JDStyle()
Base.Broadcast.BroadcastStyle(::JDStyle, ::Base.Broadcast.BroadcastStyle) = JDStyle()
Base.Broadcast.BroadcastStyle(::Base.Broadcast.BroadcastStyle, ::JDStyle) = JDStyle()

function Base.broadcasted(::typeof(*), j::J, k::J) where{J<:AbstractJointDistribution}
    m1, x1, y1 = distribution(j), columns(j), rows(j)
    m2, x2, y2 = distribution(k), columns(k), rows(k)
    
    (x1 == x2 && y1 == y2) || error("dimensions must be the same")
    return J(m1 .* m2, y1, x1)
end

function Base.broadcasted(::typeof(*), j::J, k) where{J<:AbstractJointDistribution}
    m, x, y = distribution(j), columns(j), rows(j)
    return J(m .* k, y, x)
end

Base.copy(j::J) where{J<:AbstractJointDistribution} = J(copy(distribution(j)), copy(rows(j)), copy(columns(j)))

function Base.similar(j::J) where{J<: AbstractJointDistribution}
    m = similar(distribution(j))
    J(m, rows(j), columns(j))
end



# TODO: support naming axes
struct JointDistribution{M<:AbstractMatrix{<:Real}, V<:AbstractVector, U<:AbstractVector} <: AbstractJointDistribution
    m::M
    y::V # ↓
    x::U # →
end

distribution(j::JointDistribution) = j.m
columns(j::JointDistribution) = j.x
rows(j::JointDistribution) = j.y
index(j::JointDistribution) = rows(j)



function DataFrames.unstack(p::Pmf{<:Real, <:Tuple{<:Any, <:Any}})
    x = unique([k[2] for k in keys(p)])
    y = unique([k[1] for k in keys(p)])
    lenx = length(x)
    leny = length(y)
    
    m = [valueat(p, (i-1)*lenx+j) for i in 1:leny, j in 1:lenx]
    return JointDistribution(m, y, x)
end

DataFrames.unstack(p::Pmf{<:Real, <:Tuple{<:Any}}) = Pmf(values(p), [k[1] for k in keys(p)], name(p))

function DataFrames.unstack(p::Pmf{<:Real, <:Tuple})
    x = unique([k[end] for k in keys(p)])
    y = unique([k[1:end-1] for k in keys(p)])
    lenx = length(x)
    leny = length(y)
    
    m = [valueat(p, (i-1)*lenx+j) for i in 1:leny, j in 1:lenx]
    return JointDistribution(m, y, x)
end

normalize!(j::JointDistribution) = (s = sum(j.m); j.m ./= sum(j.m); s)
makejoint(a::Pmf, b::Pmf) = JointDistribution(values(a)' .* values(b), keys(b), keys(a))


Statistics.cov(df::DataFrames.AbstractDataFrame) = Statistics.cov(Matrix(df))

StatsBase.mean_and_cov(df::DataFrames.AbstractDataFrame) = StatsBase.mean_and_cov(Matrix(df))

"""
calculates a covariance matrix from a dataframe and returns a dataframe with an index column
"""
covdf(df::DataFrames.AbstractDataFrame) = 
        DataFrames.DataFrame("index" => names(df), 
                        map(i -> (names(df)[i] => Statistics.cov(df)[:, i]), 1:DataFrames.ncol(df))...)


"""Print the mean and CI of a distribution.

posterior: Pmf
digits: number of digits to round to
prob: probability in the CI
"""
function summarize(posterior::Pmf, digits::Integer=3, prob=0.9)
    μ = round(Statistics.mean(posterior), digits=3)
    ci = credibleinterval(posterior, prob)
    println("$μ, $ci")
end


export Pmf, Cdf, JointDistribution
export makepmf, normalize!, maxprob, pmffromseq, prob_ge, prob_le, prob_gt, prob_lt, maxprob, credibleinterval
export adddist, subdist, muldist, divdist, makebinomial, makecdf, cdffromseq, make_dataframe, maxdist, mindist
export prob_gt, prob_eq, lt_dist, gt_dist, le_dist, ge_dist, makemixture, makejoint, makeuniform, distribution
export summarize, covdf, marginal, rows, columns, kde_from_sample, pmffromdist, transform

end; # module