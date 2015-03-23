@test sum(normalize(wt)) == 1

num_samples, num_signals = size(data)
@test std(data, mean(data, 2), num_samples, num_signals) == std(data, vec(mean(data, 2)), num_samples, num_signals)
@test_approx_eq_eps std(data, mean(data, 2), num_samples, num_signals) [1.67369, 0.959536, 1.2842] ε

for i = 1:size(data, 2)

    # Compare standardized central moments
    @test_approx_eq_eps coskew(data, standardize=true)[i,i,i] expected_skewness[i] ε
    @test_approx_eq_eps cokurt(data, standardize=true)[i,i,i,i] expected_kurtosis[i] ε

    # Compare raw central moments
    @test_approx_eq_eps coskew(data, standardize=false)[i,i,i] expected_third_moment[i] ε
    @test_approx_eq_eps cokurt(data, standardize=false)[i,i,i,i] expected_fourth_moment[i] ε
end

@test_approx_eq_eps coskewness(data, wt) 0.100543398 ε
@test_approx_eq_eps cokurtosis(data, wt) 0.582933712 ε

@test_approx_eq_eps coskewness(data, wt, standardize=true) 0.148687894 ε
@test_approx_eq_eps cokurtosis(data, wt, standardize=true) 0.664085775 ε

@test_approx_eq_eps coskewness(data) 0.081011176 ε
@test_approx_eq_eps cokurtosis(data) 0.571876949 ε

@test_approx_eq_eps coskewness(data, bias=1) 0.084533401 ε
@test_approx_eq_eps cokurtosis(data, bias=1) 0.596741164 ε
