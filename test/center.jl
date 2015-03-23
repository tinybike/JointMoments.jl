for standardize in (true, false)
    for bias in (0, 1)
        cntr = center(data; standardize=standardize, bias=bias)[1]
        wcntr = center(data, uniform_weights; standardize=standardize, bias=bias)[1]
        @test tolerance(cntr, wcntr)
    end
end
