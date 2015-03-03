module JointMoments

    import Base: std

    export
        normalize,
        coskew,
        cokurt,
        coskewness,
        cokurtosis,
        std,
        _cov,
        outer

    include("tensors.jl")

    include("statistics.jl")

end # module
