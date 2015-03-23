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
