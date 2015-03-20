using JointMoments
using Base.Test

include("tolerance.jl")

ε = 1e-4

tests = ["data",
         "center",
         "tensors",
         "collapse", 
         "statistics"]

println("Running tests:")

for t in tests
    tfile = string(t, ".jl")
    println(" * $(tfile) ...")
    include(tfile)
end
