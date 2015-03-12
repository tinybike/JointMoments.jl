expected_coskew = [ 0.147575   0.154872   0.126323   0.154872   0.414994  -0.033756   0.126323  -0.033756   0.013083
                    0.154872   0.414994  -0.033756   0.414994   0.861588  -0.301254  -0.033756  -0.301254  -0.287069
                    0.126323  -0.033756   0.013083  -0.033756  -0.301254  -0.287069   0.013083  -0.287069   1.017824]

expected_cokurt = [ 2.126762   1.118844   0.474779   1.118844   1.122935   0.187329   0.474779   0.187329   1.155241   1.118844   1.122935   0.187329   1.122935   1.404624  -0.026636   0.187329  -0.026636   0.276558   0.474779   0.187329  1.155241   0.187329  -0.026636   0.276558   1.155241   0.276558   0.178076
                    1.118844   1.122935   0.187329   1.122935   1.404624  -0.026636   0.187329  -0.026636   0.276558   1.122935   1.404624  -0.026636   1.404624   3.102880  -0.517198  -0.026636  -0.517198   0.779222   0.187329  -0.026636  0.276558  -0.026636  -0.517198   0.779222   0.276558   0.779222   0.218735
                    0.474779   0.187329   1.155241   0.187329  -0.026636   0.276558   1.155241   0.276558   0.178076   0.187329  -0.026636   0.276558  -0.026636  -0.517198   0.779222   0.276558   0.779222   0.218735   1.155241   0.276558  0.178076   0.276558   0.779222   0.218735   0.178076   0.218735   5.989492]

expected_coskew_tensor = reshape(expected_coskew, 3, 3, 3)
expected_cokurt_tensor = reshape(expected_cokurt, 3, 3, 3, 3)

@test_approx_eq_eps _cov(data, bias=1) cov(data, corrected=true) ε
@test_approx_eq_eps _cov(data, bias=0) cov(data, corrected=false) ε

@test_approx_eq_eps coskew(data, flatten=true) expected_coskew ε
@test_approx_eq_eps coskew(data) expected_coskew_tensor ε
@test_approx_eq coskew(data, flatten=true)[1,4] coskew(data, flatten=true, dense=false)[1,4]
@test_approx_eq coskew(data, flatten=true, dense=false)[1,4] coskew(data, dense=false)[2,1,1]
@test_approx_eq coskew(data, dense=false)[2,1,1] coskew(data)[2,1,1]

@test_approx_eq_eps cokurt(data, flatten=true) expected_cokurt ε
@test_approx_eq_eps cokurt(data) expected_cokurt_tensor ε
@test_approx_eq cokurt(data, flatten=true)[1,10] cokurt(data, flatten=true, dense=false)[1,10]
@test_approx_eq cokurt(data, flatten=true, dense=false)[1,10] cokurt(data, dense=false)[2,1,1,1]
@test_approx_eq cokurt(data, dense=false)[2,1,1,1] cokurt(data)[2,1,1,1]

num_samples, num_signals = size(data)
@inbounds begin
    avgs = vec(mean(data, 1))
    cntr = data .- avgs'
end

covmat = _cov(data; bias=0)
contracted = sum(covmat, 2)[:]
@test all(contracted - contraction(data, 2; bias=0) .< ε)

tensor = coskew(data; standardize=true, bias=0)
contracted = sum(sum(tensor, 3), 2)[:]
@test all(contracted - contraction(data, 3; standardize=true, bias=0) .< ε)

tensor = cokurt(data; standardize=true, bias=0)
contracted = sum(sum(sum(tensor, 4), 3), 2)[:]
@test all(contracted - contraction(data, 4; standardize=true, bias=0) .< ε)
