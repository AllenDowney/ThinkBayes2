
module Utils
# check that a package is installed
function checkpkgs(pkglist::Vararg{AbstractString, N}) where N
    missing = []
    for p in pkglist
        pkg = Symbol(p)
        try
            @eval using $pkg
        catch e
            push!(missing, p)
        end
    end
    if length(missing) > 0
        m = join(missing, ", ")
        error("""These packages are missing: $m
                 You can install with e.g. :
                    import Pkg; Pkg.add("$(missing[1])")
            """
        )
    end
end

checkpkgs("Downloads", "Plots", "Distributions", "DataFrames", "Loess", "Gumbo", "HTTP", "Cascadia")

import Downloads, DataFrames, Loess, Plots, Distributions
import Gumbo, Cascadia, HTTP, CSV

function getfile(url::String)
    filename = basename(url)
    if !isfile(filename)
        _local = Downloads.download(url, filename)
        println("Downloaded ", _local)
    end
end

default(linewidth=2, legend=false, size = (480, 320)); 

function Base.transpose(df::DataFrame; col_names=nothing)
    tdf = DataFrames.DataFrame(collect.(eachrow(df)), :auto)
    isnothing(col_names) || rename!(tdf, col_names)
    return tdf
end

function plotloess!(xs, ys; kwargs...)
    model = Loess.loess(xs, ys)
    us = range(extrema(xs)...; length = 101)
    vs = Loess.predict(model, us)
    Plots.plot!(us, vs; kwargs...)
end

# this is a hack
# unfortunately, there's no Julia equivalent for Pandas' read_html
function retrievetables(url::String; kwargs...)
    res = HTTP.request("GET",url)
    g = Gumbo.parsehtml(String(res.body))

    delim="∘"
    tbls = eachmatch(Cascadia.Selector("tbody"), g.root)
    v = []
    for tbl in tbls
        rows = children(tbl)
        io = IOBuffer()
        # this is an ugly hack: convert to CSV and let CSV deal with the mess
        # (hopefully, parsing types correctly)
        for row in rows
            entries = [strip(nodeText(c)) for c in children(row)]
            println(io, join(entries, delim))
        end
        seekstart(io)
        df = CSV.File(io, delim=delim; kwargs...) |> DataFrame
        push!(v, df)
    end
    return v
end

export checkpkgs, retrievetables, transpose, plotloess!, getfile
end;