module JointMoments

    using StatsBase

    import Base: std

    export
        normalize,
        coskew,
        cokurt,
        coskewness,
        cokurtosis,
        center,
        collapse,
        std,
        flip,
        _cov,
        outer,
        replicate,
        recombine

    include("center.jl")
    include("tensors.jl")
    include("replicate.jl")
    include("collapse.jl")
    include("statistics.jl")

end # module
