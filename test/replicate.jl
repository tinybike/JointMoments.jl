replicated = replicate(data, w1)
coll = collapse(replicated'; order=4, standardize=true)
recombined = recombine(coll, w1)

@test all(repl .== replicated)
@test length(recombined) == length(w1)
@test all(recombined .== recombine(collapse(repl'; order=4, standardize=true), w1))

@test_throws DimensionMismatch replicate(data', w1)
