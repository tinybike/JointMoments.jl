using JointMoments
using Base.Test

tests = ["data",
         "tensors", 
         "statistics"]

println("Running tests:")

for t in tests
    tfile = string(t, ".jl")
    println(" * $(tfile) ...")
    include(tfile)
end
