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

    include("tensors.jl")

    include("statistics.jl")

end # module
