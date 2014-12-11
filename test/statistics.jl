@test sum(normalize(wt)) == 1

# Expected skewness, kurtosis, and central moments are calculated
# using StatsBase.
# (Note: + 3 has been added to the kurtosis values, to change the
# cumulant into a standardized central moment.)
expected_skewness = [0.2036496291131231, 1.055142530854644, 0.5466699866166419]
expected_kurtosis = [3.2674593510418695, 4.065496961897709, 2.6149221488093146]

expected_third_moment = [0.14757650128987837, 0.8615880265444518, 1.0178235320342266]
expected_fourth_moment = [2.1267754676390918, 3.1028802049533724, 5.989470624146371]

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
