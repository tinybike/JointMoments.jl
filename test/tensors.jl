expected_coskew = [ 0.147575   0.154872   0.126323   0.154872   0.414994  -0.033756   0.126323  -0.033756   0.013083
                    0.154872   0.414994  -0.033756   0.414994   0.861588  -0.301254  -0.033756  -0.301254  -0.287069
                    0.126323  -0.033756   0.013083  -0.033756  -0.301254  -0.287069   0.013083  -0.287069   1.017824]

expected_cokurt = [ 2.126762   1.118844   0.474779   1.118844   1.122935   0.187329   0.474779   0.187329   1.155241   1.118844   1.122935   0.187329   1.122935   1.404624  -0.026636   0.187329  -0.026636   0.276558   0.474779   0.187329  1.155241   0.187329  -0.026636   0.276558   1.155241   0.276558   0.178076
                    1.118844   1.122935   0.187329   1.122935   1.404624  -0.026636   0.187329  -0.026636   0.276558   1.122935   1.404624  -0.026636   1.404624   3.102880  -0.517198  -0.026636  -0.517198   0.779222   0.187329  -0.026636  0.276558  -0.026636  -0.517198   0.779222   0.276558   0.779222   0.218735
                    0.474779   0.187329   1.155241   0.187329  -0.026636   0.276558   1.155241   0.276558   0.178076   0.187329  -0.026636   0.276558  -0.026636  -0.517198   0.779222   0.276558   0.779222   0.218735   1.155241   0.276558  0.178076   0.276558   0.779222   0.218735   0.178076   0.218735   5.989492]

expected_coskew_tensor = reshape(expected_coskew, 3, 3, 3)
expected_cokurt_tensor = reshape(expected_cokurt, 3, 3, 3, 3)

@test tolerance(_cov(data, bias=1), cov(data, corrected=true))
@test tolerance(_cov(data), cov(data, corrected=false))

ltm = _cov(data, dense=false)
@test tolerance(ltm + ltm' - diagm(diag(ltm)), _cov(data, dense=true))

@test tolerance(coskew(data, flatten=true), expected_coskew)
@test tolerance(coskew(data), expected_coskew_tensor)
@test tolerance(coskew(data, flatten=true)[1,4], coskew(data, flatten=true, dense=false)[1,4])
@test tolerance(coskew(data, flatten=true, dense=false)[1,4], coskew(data, dense=false)[2,1,1])
@test tolerance(coskew(data, dense=false)[2,1,1], coskew(data)[2,1,1])

@test tolerance(cokurt(data, flatten=true), expected_cokurt)
@test tolerance(cokurt(data), expected_cokurt_tensor)
@test tolerance(cokurt(data, flatten=true)[1,10], cokurt(data, flatten=true, dense=false)[1,10])
@test tolerance(cokurt(data, flatten=true, dense=false)[1,10], cokurt(data, dense=false)[2,1,1,1])

for d in (data, bigdata)
    covmat = _cov(d; bias=0)
    summed = sum(covmat, 2)[:]
    @test tolerance(summed, collapse(d, 2; bias=0))

    tensor = coskew(d; standardize=true)
    summed = sum(sum(tensor, 3), 2)[:]
    @test tolerance(summed, collapse(d, 3; standardize=true))

    tensor = cokurt(d; standardize=true)
    summed = sum(sum(sum(tensor, 4), 3), 2)[:]
    @test tolerance(summed, collapse(d, 4; standardize=true))
end
