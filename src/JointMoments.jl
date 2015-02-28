module JointMoments

export normalize, coskew, cokurt, coskewness, cokurtosis, _cov, _std, _transpose, _corners, _pairs

include("tensors.jl")

include("statistics.jl")

end # module
