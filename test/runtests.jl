using JointMoments
using Base.Test

include("data.jl")
include("precomputed.jl")
include("tolerance.jl")

ε = 1e-4

tests = ["center",
         "replicate",
         "tensors",
         "collapse", 
         "statistics"]

println("Running tests:")

for t in tests
    tfile = string(t, ".jl")
    println(" * $(tfile) ...")
    include(tfile)
end
