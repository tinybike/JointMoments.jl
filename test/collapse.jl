for standardize in (true, false)
    for bias in (0, 1)
        coll = collapse(data; order=4, standardize=standardize, bias=bias)
        wcoll = collapse(data, uniform_weights;
                         order=4, standardize=standardize, bias=bias)
        @test tolerance(coll, wcoll)

        # Verify equivalence to brute-force tensor collapse
        for d in (data, bigdata)

            # Second order
            covmat = _cov(d; bias=bias)
            summed = sum(covmat, 2)[:]
            collapsed = collapse(d; order=2, bias=bias)
            @test tolerance(summed, collapsed)

            # Third order
            tensor = coskew(d; standardize=standardize, bias=bias)
            summed = sum(sum(tensor, 3), 2)[:]
            collapsed = collapse(d; order=3, standardize=standardize, bias=bias)
            @test tolerance(summed, collapsed)

            # Fourth order
            tensor = cokurt(d; standardize=standardize, bias=bias)
            summed = sum(sum(sum(tensor, 4), 3), 2)[:]
            collapsed = collapse(d; order=4, standardize=standardize, bias=bias)
            @test tolerance(summed, collapsed)
        end
    end
end

@test_throws ErrorException collapse(nan_test_data,
                                     nan_test_wt;
                                     order=4,
                                     standardize=true,
                                     axis=2)

@test all(~isnan(collapse(nan_test_data,
                          nan_test_wt;
                          order=4,
                          standardize=false,
                          axis=2)))
