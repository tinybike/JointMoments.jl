module JointMoments

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
        _cov,
        outer

    include("center.jl")
    include("tensors.jl")
    include("collapse.jl")
    include("statistics.jl")

end # module
